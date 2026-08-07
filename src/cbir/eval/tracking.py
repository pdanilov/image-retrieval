"""Optional trackio mirror of each recorded run.

`results/runs.jsonl` is the record; this is convenience on top of it — a queryable
local SQLite view (`trackio query cbir --sql ...`) and a dashboard (`trackio show`).
So the import is defensive: a machine without trackio installed still evaluates and
still writes its results, it just doesn't get the dashboard. Nothing here may raise
into the caller, because a tracking failure must never lose a completed run — the
evaluation that produced it can cost an hour.
"""

from __future__ import annotations

from typing import Any

from cbir.eval.results import RunRecord

PROJECT = "cbir"


def flatten_metrics(record: RunRecord) -> dict[str, float]:
    """`{"easy": {"map": 0.17, ...}}` -> `{"easy/map": 0.17, ...}`.

    trackio logs flat scalars, and the `protocol/metric` naming is what groups the
    three protocols into one chart per metric in the dashboard.
    """
    flat: dict[str, float] = {}
    for protocol, metrics in record.metrics.items():
        for name, value in metrics.items():
            if isinstance(value, int | float):
                flat[f"{protocol}/{name}"] = float(value)
    if record.seconds is not None:
        flat["seconds"] = record.seconds
    return flat


def run_name(record: RunRecord) -> str:
    """`bow-roxford5k-k5000-seed0` — readable in `trackio list runs` without a lookup."""
    params = "-".join(f"{key}{value}" for key, value in sorted(record.params.items()))
    return f"{record.technique}-{record.dataset}-{params}"


def config_of(record: RunRecord) -> dict[str, Any]:
    """Everything that determines the numbers, flat enough for trackio's config table."""
    return {
        "dataset": record.dataset,
        "held_out_dataset": record.held_out_dataset,
        "technique": record.technique,
        "commit": record.commit,
        **record.params,
    }


def track(record: RunRecord) -> None:
    """Mirror one finished run into trackio, or do nothing if it isn't installed."""
    try:
        import trackio
    except ImportError:
        return

    trackio.init(project=PROJECT, name=run_name(record), config=config_of(record))
    try:
        trackio.log(flatten_metrics(record))
    finally:
        trackio.finish()
