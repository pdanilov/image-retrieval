"""cbir CLI entry point."""

from __future__ import annotations

from dataclasses import replace
from typing import Annotated, Literal

import tyro

from cbir.configs.classic import BoWConfig, FisherConfig, VLADConfig
from cbir.configs.cnn import NeuralCodesConfig, PooledConfig
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
    | Annotated[PooledConfig, tyro.conf.subcommand("gem")],
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
        descriptor: Technique to evaluate, with its own hyper-parameters.
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
    }
    tyro.extras.subcommand_cli_from_dict(subcommands)


if __name__ == "__main__":
    main()
