"""Flatten `results/runs.jsonl` into tidy rows for analysis.

`RunRecord.metrics` is nested (protocol -> metric -> value), which is right for the
record but wrong for plotting: every chart wants one row per (run x protocol) with the
protocol as a column. That reshape is needed by every figure, so it lives here rather
than being re-typed in each notebook — AGENTS.md keeps `notebooks/` to analysis and
figures only.

Returns plain dicts, not a DataFrame: pandas is not a dependency of this package (it
only arrives transitively via `datasets`), and `pd.DataFrame(tidy_rows())` is a single
call at the notebook end. Nothing here needs pandas to do its job.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cbir.eval.results import RunRecord, load

LOCAL_DIM = 128
"""RootSIFT descriptor dimensionality, the `d` in every encoded size below.

Hard-coded rather than imported from `local/rootsift.py`, which imports cv2 at module
scope — a plotting helper should not pull OpenCV into a notebook kernel to learn the
number 128. It is fixed by SIFT itself, not a tunable.
"""

CONV_CHANNELS = {
    "alexnet": 256,
    "vgg16": 512,
    "vgg19": 512,
    "resnet18": 512,
    "resnet34": 512,
    "resnet50": 2048,
    "resnet101": 2048,
}
"""Each backbone's last conv width, which is its descriptor width when `dim` is unset.

Duplicated from `descriptors/cnn/pooling.py` for the same reason `LOCAL_DIM` is
duplicated from `rootsift.py`: that module imports torch at scope, and a plotting
helper should not pull torch into a kernel to learn the number 2048. A test asserts
the two maps agree, so the copy cannot drift.
"""

FC_DIM = 4096
"""Neural Codes' fc6 width, the same for both backbones it supports."""


def descriptor_dim(technique: str, params: dict[str, Any]) -> int | None:
    """Encoded vector length, or None for a technique whose formula isn't known here.

    This is the axis on which the three techniques are actually comparable. At equal
    `k` they are not: BoW at k=64 is a 64-dim vector, VLAD is 8192-dim and Fisher
    16384-dim, so a chart of mAP against `k` alone flatters BoW enormously.
    """
    match technique:
        case "bow" | "vlad" | "fisher":
            k = params.get("k")
            if not isinstance(k, int):
                return None
            if technique == "bow":
                return k  # one weight per visual word
            if technique == "vlad":
                return k * LOCAL_DIM  # one residual vector per word
            return 2 * k * LOCAL_DIM  # first- and second-order, per component
        case "neural_codes" | "gem" | "rmac":
            # The CNN tier states its width directly, and `dim=None` means "no PCA" --
            # so the width is whatever the descriptor is natively.
            dim = params.get("dim")
            if isinstance(dim, int):
                return dim
            if technique == "neural_codes":
                return FC_DIM
            return CONV_CHANNELS.get(params.get("backbone", ""))
        case _:
            return None


def tidy_rows(path: Path | None = None, records: list[RunRecord] | None = None) -> list[dict[str, Any]]:
    """One row per (run x protocol), ready for `pd.DataFrame(...)`.

    Pass `records` (e.g. from `latest().values()`) to tabulate a subset; by default
    every recorded run is read from `path`.
    """
    records = load(path) if records is None else records
    rows: list[dict[str, Any]] = []
    for record in records:
        dim = descriptor_dim(record.technique, record.params)
        for protocol, metrics in sorted(record.metrics.items()):
            rows.append(
                {
                    "dataset": record.dataset,
                    "held_out_dataset": record.held_out_dataset,
                    "technique": record.technique,
                    "protocol": protocol,
                    "k": record.params.get("k"),
                    "dim": dim,
                    "map": metrics.get("map"),
                    "mp_at_k": metrics.get("mp_at_k"),
                    "num_queries": metrics.get("num_queries"),
                    "num_excluded": metrics.get("num_excluded"),
                    "seconds": record.seconds,
                    "commit": record.commit,
                    "recorded_at": record.recorded_at,
                    # Everything not promoted to a column, so a run is still
                    # identifiable when two rows share (technique, k) -- e.g. Fisher at
                    # two `sample` sizes, or VLAD with intra_norm on and off.
                    "params": dict(record.params),
                }
            )
    return rows
