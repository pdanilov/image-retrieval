"""cbir CLI entry point."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Annotated, Literal

import tyro

from cbir.configs.classic import BoWConfig, FisherConfig, VLADConfig
from cbir.configs.cnn import NeuralCodesConfig, PooledConfig, RMACConfig
from cbir.configs.presets import PRESETS
from cbir.configs.run import DescriptorConfig, RunConfig
from cbir.data.holdout import EvalDataset
from cbir.data.revisitop import SUPPORTED, download
from cbir.eval.metrics import Protocol
from cbir.eval.results import RunRecord, format_param, latest, load

PROTOCOLS: tuple[Protocol, ...] = ("easy", "medium", "hard")

Descriptor = Annotated[
    Annotated[BoWConfig, tyro.conf.subcommand("bow")]
    | Annotated[VLADConfig, tyro.conf.subcommand("vlad")]
    | Annotated[FisherConfig, tyro.conf.subcommand("fisher")]
    | Annotated[NeuralCodesConfig, tyro.conf.subcommand("neural-codes")]
    | Annotated[PooledConfig, tyro.conf.subcommand("gem")]
    | Annotated[RMACConfig, tyro.conf.subcommand("rmac")]
    # Presets join the *same* union rather than getting their own command: they produce
    # these very types, so `cbir evaluate gem-ft-r101` and `cbir evaluate gem --backbone
    # resnet101 ...` differ only in how much you had to type. Built from the instances,
    # so every field stays an overridable flag and the override re-validates --
    # `gem-ft-r101 --p 3.0` is rejected exactly as spelling it out by hand would be.
    #
    # A `--preset` flag cannot do this: a technique's knobs are defined *inside* its
    # subcommand, so there would be nowhere for `--scales` to live, and the descriptor
    # subcommand would still be required alongside the flag.
    | tyro.extras.subcommand_type_from_defaults(PRESETS),
    # Named "" so the technique reads as a bare subcommand (`cbir evaluate bow --k 5000`)
    # rather than `descriptor:bo-w-config`, and each technique's --help lists only its
    # own knobs.
    tyro.conf.arg(name=""),
]


def download_cmd(
    datasets: tuple[Literal["roxford5k", "rparis6k", "all"], ...],
) -> None:
    """Download a RevisitOP benchmark split.

    Args:
        datasets: Dataset(s) to download.
    """
    names = list(SUPPORTED) if "all" in datasets else datasets
    for name in names:
        download(name)


def _format_params(params: dict[str, object]) -> str:
    """`{"k": 5000, "seed": 0}` -> `k=5000 seed=0`, sorted so columns line up."""
    return " ".join(f"{key}={format_param(value)}" for key, value in sorted(params.items()))


def _render(rows: list[list[str]], headers: list[str]) -> str:
    """Left-aligned fixed-width table. No dependency -- the whole point of this command."""
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i]) for i in range(len(headers))
    ]
    lines = ["  ".join(header.ljust(width) for header, width in zip(headers, widths, strict=True)).rstrip()]
    lines.append("  ".join("-" * width for width in widths))
    for row in rows:
        lines.append("  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip())
    return "\n".join(lines)


def _sort_key(record: RunRecord) -> tuple[str, str, int, str, str]:
    """Order rows for reading a sweep: `k` numerically, everything else by name.

    Sorting on `_format_params` alone compares `k` as text, which puts k=20000 between
    k=1000 and k=5000 — precisely the wrong order for the column a sweep varies. `k` is
    pulled out as an int; the formatted params still break ties so that runs differing
    in some other knob stay grouped.
    """
    k = record.params.get("k")
    return (
        record.dataset,
        record.technique,
        k if isinstance(k, int) else 0,
        _format_params(record.params),
        record.recorded_at,
    )


def _row(record: RunRecord, metric: str) -> list[str]:
    cells = [record.dataset, record.held_out_dataset, record.technique, _format_params(record.params)]
    for protocol in PROTOCOLS:
        value = record.metrics.get(protocol, {}).get(metric)
        cells.append(f"{value:.4f}" if isinstance(value, int | float) else "-")
    cells.append(record.commit or "-")
    cells.append(record.recorded_at[:10])
    return cells


def results_cmd(
    dataset: str | None = None,
    technique: str | None = None,
    metric: Literal["map", "mp_at_k"] = "map",
    current_only: bool = False,
) -> None:
    """Show recorded evaluation runs from `results/runs.jsonl`.

    Args:
        dataset: Only show runs on this eval dataset.
        technique: Only show runs for this aggregator (bow/vlad/fisher).
        metric: Which metric to tabulate per protocol.
        current_only: Collapse re-runs of a configuration to the most recent row.
    """
    all_records = list(latest().values()) if current_only else load()
    records = [r for r in all_records if dataset is None or r.dataset == dataset]
    records = [r for r in records if technique is None or r.technique == technique]

    if not records:
        # An empty store and an over-narrow filter are different problems; saying
        # "nothing recorded" when rows exist would send you looking in the wrong place.
        if all_records:
            print(f"no runs match that filter ({len(all_records)} recorded)")
        else:
            print("no runs recorded yet (results/runs.jsonl)")
        return

    records.sort(key=_sort_key)
    headers = ["dataset", "held-out", "technique", "params", *PROTOCOLS, "commit", "recorded"]
    print(_render([_row(record, metric) for record in records], headers))


SWEEP_AXIS: dict[str, str] = {
    "bow": "k",
    "vlad": "k",
    "fisher": "k",
    "neural_codes": "dim",
    "gem": "dim",
    "rmac": "dim",
}
"""The one field `--sweep` varies per technique — whatever costs something to change.

For the classic tier that is `k`, which re-trains the codebook. Neural Codes has no
`k` at all; its equivalent is the PCA output width, so sweeping `k` there would raise
rather than mean nothing. A technique absent from this map cannot be swept.
"""


def sweep_configs(
    descriptor: DescriptorConfig, dataset: EvalDataset, mp_at_k: int, values: tuple[int, ...]
) -> list[RunConfig]:
    """One `RunConfig` per swept value, or a single run when nothing is swept.

    A sweep is just a list of configs (AGENTS.md: no sweeper plugin at this scale). Only
    one axis is swept, the one that costs something to change — see `SWEEP_AXIS`. The
    cheap post-processing knobs (`power`, `intra_norm`) are compared by re-running the
    encode instead.
    """
    if not values:
        return [RunConfig(descriptor=descriptor, dataset=dataset, mp_at_k=mp_at_k)]

    axis = SWEEP_AXIS.get(descriptor.technique)
    if axis is None:
        raise ValueError(f"{descriptor.technique} has no sweepable axis; run it one configuration at a time")
    return [
        RunConfig(descriptor=replace(descriptor, **{axis: value}), dataset=dataset, mp_at_k=mp_at_k) for value in values
    ]


def evaluate_cmd(
    descriptor: Descriptor,
    dataset: EvalDataset = "roxford5k",
    mp_at_k: int = 10,
    sweep: tuple[int, ...] = (),
    record: bool = True,
) -> None:
    """Train, encode, search and score one classic-tier configuration.

    The vocabulary is always trained on the *other* dataset — that pairing is derived,
    not configurable. Results are appended to `results/runs.jsonl`; read them back with
    `cbir results`.

    Args:
        descriptor: Technique to evaluate, with its own hyper-parameters — or a preset
            from `configs/presets.py`, whose knobs remain overridable.
        dataset: Benchmark to evaluate on.
        mp_at_k: Cutoff for mean precision@k. mAP is always over the full ranking.
        sweep: Run once per value, overriding the technique's swept axis — `k` for the
            classic tier, PCA `dim` for Neural Codes. Empty runs the config once.
        record: Append to results/runs.jsonl (and mirror to trackio). Off for scratch runs.
    """
    # Imported here, not at module scope: `runner` pulls in torch (via search/exact),
    # a multi-second import that `cbir download` and `cbir results` never need. The
    # config dataclasses do have to be imported eagerly -- tyro reads them off this
    # function's annotations to build the subcommands.
    from cbir.eval.runner import run_all

    configs = sweep_configs(descriptor, dataset, mp_at_k, sweep)
    # `run_all` rather than a loop over `run`: it extracts once per dataset and shares
    # the result. Looping here would re-read ~20 GB of cached descriptors per point.
    records = run_all(configs, record=record, verbose=len(configs) > 1)

    print()
    headers = ["dataset", "held-out", "technique", "params", *PROTOCOLS, "commit", "recorded"]
    print(_render([_row(run_record, "map") for run_record in records], headers))
    if not record:
        print("\n(not recorded: --no-record)")


def train_cmd(
    backbone: Literal["vgg16", "resnet101", "resnet50"] = "vgg16",
    weights: Literal["caffe", "torchvision"] = "caffe",
    epochs: int = 30,
    lr: float = 5e-7,
    margin: float | None = None,
    query_size: int = 2000,
    pool_size: int = 22000,
    image_size: int = 362,
    seed: int = 0,
    whiten: bool = True,
    whiten_pairs: int = 20000,
    patience: int | None = 5,
    min_delta: float = 0.001,
    resume: bool = True,
    out: str = "data/runs",
) -> None:
    """Fine-tune a backbone for retrieval on retrieval-SfM-120k.

    Trains GeM with a learnable exponent under a contrastive loss over SfM-mined
    landmark pairs, re-mining hard negatives against the current model each epoch.
    Checkpoints land in `out/<run>/`, in the same layout as the published ones, so
    `cbir evaluate` can read them back.

    This is hours of GPU, not minutes: roughly 15 minutes per epoch for vgg16 at the
    default sizes. The corpus is a manual download — see `descriptors/cnn/sfm.py`.

    Args:
        backbone: Architecture to fine-tune.
        weights: Which ImageNet weights to start from. The reference initializes from
            its Caffe-converted ones and the published numbers were reached that way, so
            that is the default; `torchvision` is the ablation. A manual download.
        epochs: Passes over the sampled tuples.
        lr: Adam learning rate. The published value is 5e-7 and is calibrated to a
            summed loss; raising it without also changing the reduction diverges.
        margin: Contrastive hinge width. Defaults to the published value for the
            backbone — 0.85 for resnet101, 0.7 for vgg16.
        query_size: Tuples sampled per epoch.
        pool_size: Images descriptors are extracted for, to mine negatives from.
        image_size: Longest side during training. Evaluation still runs at 1024.
        seed: Seeds sampling and initialization.
        whiten: Fit the supervised projection onto `best.pth` when training ends, which
            is what makes the checkpoint complete — see `cbir whiten`. Turning it off
            leaves a network that has to be evaluated against held-out PCA instead.
        whiten_pairs: Matching pairs the projection is fitted on.
        patience: Stop after this many epochs with no validation-mAP gain larger than
            `min_delta`. `None` runs the full schedule, as the reference does.
        min_delta: How much an epoch must beat the running best by to count as progress.
            Smaller gains still update `best.pth`; they just do not reset the patience
            counter.
        resume: Continue from `last.pth` if the run directory already holds one. Mining
            is seeded per epoch, so a resumed run draws the tuples a fresh one would have.
        out: Directory for checkpoints.
    """
    # Deferred like `evaluate`'s runner import: `cbir results` should not pay for torch.
    from cbir.training.loop import TrainConfig, train

    config = TrainConfig(
        backbone=backbone,
        weights=weights,
        epochs=epochs,
        lr=lr,
        margin=margin,
        query_size=query_size,
        pool_size=pool_size,
        image_size=image_size,
        seed=seed,
        whiten=whiten,
        whiten_pairs=whiten_pairs,
        patience=patience,
        min_delta=min_delta,
        resume=resume,
        out=Path(out),
    )
    best = train(config)
    print(f"\nbest checkpoint: {best}")


def whiten_cmd(
    checkpoint: str,
    backbone: Literal["vgg16", "resnet101", "resnet50"] = "vgg16",
    pairs: int = 20000,
    max_side: int = 1024,
    seed: int = 0,
) -> None:
    """Fit supervised whitening for a trained checkpoint and write it into the file.

    `cbir train` fine-tunes the network but not the projection applied to its output —
    the reference fits that separately, on SfM matching pairs, after training. Without it
    a trained checkpoint has to fall back on PCA over the held-out set, which measured
    0.5747 Medium against 0.6073 for the same published network with its own projection.

    Writes `meta['Lw']` in place, under the key the loader already reads, so afterwards
    the checkpoint scores with `--whiten --whiten-source learned` exactly as a published
    one does. Both the single- and multi-scale variants are fitted, since they are not
    interchangeable.

    This is one extraction pass per variant at evaluation resolution — tens of minutes,
    not minutes.

    Args:
        checkpoint: The `best.pth` to fit for and write into.
        backbone: Architecture the checkpoint holds.
        pairs: Matching pairs to fit on. Must exceed the descriptor width, and wants to
            exceed it comfortably; the cost is roughly two images extracted per pair.
        max_side: Longest image side during extraction. Match what evaluation uses.
        seed: Seeds the pair sample.
    """
    from cbir.training.whitening import attach, fit

    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint {path} not found")

    fitted = fit(checkpoint, backbone, pairs=pairs, max_side=max_side, seed=seed)
    attach(path, fitted)
    print(f"\nwrote Lw ({', '.join(sorted(fitted))}) into {path}")


def track_cmd(current_only: bool = False) -> None:
    """Mirror `results/runs.jsonl` into trackio, for `trackio show --project cbir`.

    Every evaluation already does this as it finishes, so this is for rebuilding: the
    trackio store is derived from the record, never the other way around. Delete
    `~/.cache/huggingface/trackio/cbir.db` first for a clean rebuild — replaying over
    an existing store adds runs rather than replacing them.

    Args:
        current_only: Replay only the most recent row per configuration.
    """
    from cbir.eval.tracking import replay

    records = list(latest().values()) if current_only else load()
    count = replay(records)
    if count:
        print(f"replayed {count} run(s) — view with: trackio show --project cbir")
    else:
        print("trackio is not installed (uv sync --extra tracking); results/runs.jsonl is unaffected")


def main() -> None:
    subcommands = {
        "download": download_cmd,
        "evaluate": evaluate_cmd,
        "results": results_cmd,
        "track": track_cmd,
        "train": train_cmd,
        "whiten": whiten_cmd,
    }
    tyro.extras.subcommand_cli_from_dict(subcommands)


if __name__ == "__main__":
    main()
