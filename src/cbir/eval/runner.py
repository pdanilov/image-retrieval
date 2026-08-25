"""End-to-end evaluation: config in, `RunRecord` out, row appended to `results/`.

This is the only place the full pipeline is composed — extract, train, encode, search,
score, record. Everything it calls already existed and was already tested in isolation;
what did not exist was anything that ran them in order, which meant the one recorded
baseline came from a throwaway script rather than from code under version control.

A sweep is a loop over `RunConfig`s (`run_all`), not a framework. Because
`results.append` is append-only, a sweep that dies halfway keeps every run that
finished — the survivors are already on disk, and re-running the sweep re-records
them rather than corrupting anything.
"""

from __future__ import annotations

from collections.abc import Iterable
from time import perf_counter
from typing import Any

from datasets import Dataset

from cbir.configs.run import RunConfig, descriptor_params
from cbir.data.holdout import HeldOutSplit
from cbir.data.revisitop import download
from cbir.eval.metrics import Protocol, QueryGroundTruth, evaluate
from cbir.eval.results import RunRecord, append
from cbir.eval.tracking import track
from cbir.search.exact import exact_search

PROTOCOLS: tuple[Protocol, ...] = ("easy", "medium", "hard")


def ground_truth(query_dataset: Dataset) -> list[QueryGroundTruth]:
    """The easy/hard/junk index lists per query, in query order.

    Each list holds positions into the *database* split, which is why
    `revisitop.download` refuses to return splits whose sizes don't match the verified
    counts: a single dropped image would shift every index here onto the wrong picture.
    """
    columns = (query_dataset["easy"], query_dataset["hard"], query_dataset["junk"])
    return [QueryGroundTruth(easy=easy, hard=hard, junk=junk) for easy, hard, junk in zip(*columns, strict=True)]


def run(config: RunConfig, *, record: bool = True, inputs: Any = None) -> RunRecord:
    """Evaluate one configuration on all three protocols and record the result.

    `record=False` skips the `results/runs.jsonl` append (and the trackio log) — for
    tests and for exploratory runs that should not enter the committed history.

    `inputs` reuses an already-prepared input set, whose type depends on the
    descriptor: RootSIFT descriptors for the classic tier, images for the CNN tier.
    Passing them matters more than it looks — the classic blobs are ~20 GB for one eval
    direction, so a sweep that re-prepares per configuration re-reads all of it for
    every point. They must match `config.dataset` *and* the descriptor's `inputs_kind`;
    this is not checked, because the only caller that supplies them is `run_all`, which
    keys them by both.
    """
    started = perf_counter()

    if inputs is None:
        inputs = config.descriptor.prepare(config.dataset)
    database_vectors, query_vectors = config.descriptor.train_and_encode(inputs)

    # Queries live in their own split (4993 database + 70 query images = the full 5063
    # of oxford5k), so no query appears in its own ranking and nothing needs excluding
    # here -- the only removals are the protocol's ignored images, inside `evaluate`.
    rankings = exact_search(query_vectors, database_vectors)

    # A second `download` call, but it resolves from the HF cache that
    # `prepare_classic_inputs` just populated. Threading the query split out of
    # `ClassicDescriptorInputs` instead would put ground-truth labels inside a
    # descriptor container, which is a worse trade than one cached re-read.
    query_dataset, _ = download(config.dataset)
    truth = ground_truth(query_dataset)
    results = [evaluate(rankings, truth, protocol, k=config.mp_at_k) for protocol in PROTOCOLS]

    run_record = RunRecord.from_eval_results(
        dataset=config.dataset,
        # From HeldOutSplit, not from `inputs`: the pairing is a pure function of the
        # eval dataset and holdout.py is its authority, so reading it off whatever the
        # descriptor happened to prepare would both couple the runner to the classic
        # tier's container and record a second, unvalidated copy of the same fact.
        held_out_dataset=HeldOutSplit.for_eval(config.dataset).held_out_dataset,
        technique=config.descriptor.technique,
        params=descriptor_params(config.descriptor),
        results=results,
        seconds=perf_counter() - started,
    )
    if record:
        append(run_record)
        track(run_record)
    return run_record


def run_all(configs: Iterable[RunConfig], *, record: bool = True, verbose: bool = False) -> list[RunRecord]:
    """Run every config in order, recording each as it finishes.

    Preparation depends only on the dataset and on what kind of input the descriptor
    wants — RootSIFT extraction is identical for BoW, VLAD and Fisher — so it happens
    once per `(dataset, inputs_kind)` and is shared by every configuration matching it.
    Re-preparing per configuration would re-read ~20 GB of cached descriptors for each
    point of a classic sweep. Keying on `inputs_kind` as well as the dataset is what
    stops a CNN configuration being handed the classic tier's descriptor arrays.

    Deliberately not parallel: a single run already saturates the machine (k-means over
    ~21M descriptors, then a full database encode), so running two at once would only
    contend for the same cores and the same descriptor cache.
    """
    configs = list(configs)
    prepared: dict[tuple[str, str], Any] = {}
    records = []
    for index, config in enumerate(configs, start=1):
        if verbose:
            # Built from the recorded params rather than `descriptor.k`: not every
            # technique has a `k` (the CNN tier has none), and this line must never be
            # the thing that breaks a sweep.
            knobs = " ".join(f"{key}={value}" for key, value in sorted(descriptor_params(config.descriptor).items()))
            print(f"[{index}/{len(configs)}] {config.descriptor.technique} {knobs} on {config.dataset}")
        key = (config.dataset, config.descriptor.inputs_kind)
        if key not in prepared:
            prepared[key] = config.descriptor.prepare(config.dataset)
        records.append(run(config, record=record, inputs=prepared[key]))
    return records
