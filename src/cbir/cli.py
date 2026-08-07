"""cbir CLI entry point."""

from __future__ import annotations

from typing import Literal

import tyro

from cbir.data.revisitop import SUPPORTED, download
from cbir.eval.metrics import Protocol
from cbir.eval.results import RunRecord, latest, load

PROTOCOLS: tuple[Protocol, ...] = ("easy", "medium", "hard")


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
    return " ".join(f"{key}={value}" for key, value in sorted(params.items()))


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

    records.sort(key=lambda r: (r.dataset, r.technique, _format_params(r.params), r.recorded_at))
    headers = ["dataset", "held-out", "technique", "params", *PROTOCOLS, "commit", "recorded"]
    print(_render([_row(record, metric) for record in records], headers))


def main() -> None:
    subcommands = {
        "download": download_cmd,
        "results": results_cmd,
    }
    tyro.extras.subcommand_cli_from_dict(subcommands)


if __name__ == "__main__":
    main()
