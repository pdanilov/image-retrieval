"""Append-only record of every evaluated run, in `results/runs.jsonl`.

One JSON object per line, one line per run. Appending never rewrites earlier lines, so
a git diff of this file is always "new rows" and two runs can never conflict. Re-running
a configuration appends a *second* row rather than replacing the first: the history is
the point. When a number moves, the pair of rows and their `commit` fields say when it
moved and what changed, which is exactly the question a bare "current results" store
cannot answer. `latest` resolves a configuration to its most recent row when you want
only the current value.

Committed to git, unlike the descriptor caches under `data/` — these are small, they
are the actual findings, and they should travel with the code that produced them (see
AGENTS.md's layout).

`RunRecord` deliberately stores the *inputs* that determine a number alongside the
number: the eval dataset, the held-out dataset that trained the codebook, the technique,
and every hyper-parameter. A row that does not say which held-out set trained its
vocabulary cannot be reproduced or compared, and reproducibility is this project's
first priority.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cbir
from cbir.eval.metrics import EvalResult

# cbir/__init__.py -> cbir/ -> src/ -> repo root.
RESULTS_PATH = Path(cbir.__file__).resolve().parents[2] / "results" / "runs.jsonl"


def git_commit() -> str | None:
    """Short hash of HEAD, or None outside a git checkout.

    Recorded per row so a number can always be traced back to the code that produced
    it. Never raises: failing to describe the checkout must not lose a result that took
    an hour to compute.
    """
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=RESULTS_PATH.parent.parent,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None
    return completed.stdout.strip() or None


@dataclass(frozen=True)
class RunRecord:
    """One evaluated configuration and what it scored.

    `metrics` is keyed by protocol (`easy`/`medium`/`hard`), each holding that
    protocol's `map`, `mp_at_k`, `k`, `num_queries` and `num_excluded` — the excluded
    count matters, since a protocol that silently drops queries would otherwise look
    like a better score over a smaller set.
    """

    dataset: str  # eval dataset
    held_out_dataset: str  # trained the codebook -- never the eval dataset
    technique: str  # "bow" | "vlad" | "fisher"
    params: dict[str, Any]  # k, seed, and anything technique-specific
    metrics: dict[str, dict[str, Any]]  # protocol -> metric name -> value
    seconds: float | None = None
    commit: str | None = field(default_factory=git_commit)
    recorded_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    @classmethod
    def from_eval_results(
        cls,
        dataset: str,
        held_out_dataset: str,
        technique: str,
        params: dict[str, Any],
        results: list[EvalResult],
        seconds: float | None = None,
    ) -> RunRecord:
        """Build a record from `evaluate()` output, one `EvalResult` per protocol."""
        return cls(
            dataset=dataset,
            held_out_dataset=held_out_dataset,
            technique=technique,
            params=dict(params),
            metrics={
                r.protocol: {
                    "map": r.map,
                    "mp_at_k": r.mp_at_k,
                    "k": r.k,
                    "num_queries": r.num_queries,
                    "num_excluded": r.num_excluded,
                }
                for r in results
            },
            seconds=seconds,
        )

    @property
    def key(self) -> tuple[str, str, str]:
        """What makes two rows the same configuration, ignoring when they were run."""
        return (self.dataset, self.technique, json.dumps(self.params, sort_keys=True))


def format_param(value: Any) -> str:
    """One param value for display: `(1.0, 0.7071, 0.5)` -> `1|0.707107|0.5`.

    Sequences are the reason this exists — Python's default repr of a tuple carries
    spaces, and both display sites (the results table and the trackio run name) use
    spaces or dashes as their own separator. `%g` rather than a rounded format, so two
    scale sets that differ only in a late digit still render differently; `runs.jsonl`
    keeps full precision either way.

    A JSON round-trip turns a tuple into a list, so both must render the same or a
    reloaded row would not match a freshly built one.
    """
    if isinstance(value, list | tuple):
        return "|".join(format_param(v) for v in value)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def append(record: RunRecord, path: Path | None = None) -> None:
    """Append one row. Creates `results/` and the file on first use."""
    path = path or RESULTS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record), sort_keys=True) + "\n")


def load(path: Path | None = None) -> list[RunRecord]:
    """Every row, oldest first. Empty if the file does not exist yet."""
    path = path or RESULTS_PATH
    if not path.exists():
        return []
    return [RunRecord(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def latest(path: Path | None = None) -> dict[tuple[str, str, str], RunRecord]:
    """The most recent row per configuration, for "what does this score now?".

    Later rows win, which is why `append` must stay append-only and in order.
    """
    current: dict[tuple[str, str, str], RunRecord] = {}
    for record in load(path):
        current[record.key] = record
    return current


def iter_rows(path: Path | None = None) -> Iterator[dict[str, Any]]:
    """Raw dicts rather than `RunRecord`s, for reading rows written by older schemas.

    `load` would raise on a row with fields this version does not know about; this lets
    analysis code stay readable against the whole history.
    """
    path = path or RESULTS_PATH
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)
