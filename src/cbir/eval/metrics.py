"""mAP / mP@k evaluation for the Revisiting Oxford/Paris protocol.

Reimplements the evaluation algorithm from Radenović et al., "Revisiting Oxford and
Paris" (CVPR 2018) — https://github.com/filipradenovic/revisitop `evaluate.py` — so
the harness has no runtime dependency on that repo. Three protocols per query, Easy/
Medium/Hard, differing only in which ground-truth images count as positive and which
are ignored: Easy scores against `easy` positives (ignoring `hard`+`junk`), Medium
against `easy`+`hard` (ignoring `junk`), Hard against `hard` (ignoring `easy`+`junk`).
Ignored images are dropped from the ranking before scoring — they are not negatives.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

Protocol = Literal["easy", "medium", "hard"]


@dataclass(frozen=True)
class QueryGroundTruth:
    easy: Sequence[int]
    hard: Sequence[int]
    junk: Sequence[int]


@dataclass(frozen=True)
class EvalResult:
    protocol: Protocol
    k: int
    map: float
    mp_at_k: float
    num_queries: int
    num_excluded: int


def protocol_sets(
    easy: Sequence[int],
    hard: Sequence[int],
    junk: Sequence[int],
    protocol: Protocol,
) -> tuple[set[int], set[int]]:
    """Return (positive, ignored) index sets for one query under one protocol.

    Easy: positive=easy, ignored=hard+junk. Medium: positive=easy+hard, ignored=junk.
    Hard: positive=hard, ignored=easy+junk. Ignored images are dropped from the
    ranking before scoring — they are not treated as negatives.
    """
    easy_s, hard_s, junk_s = set(easy), set(hard), set(junk)
    match protocol:
        case "easy":
            return easy_s, hard_s | junk_s
        case "medium":
            return easy_s | hard_s, junk_s
        case "hard":
            return hard_s, easy_s | junk_s
        case _:
            raise ValueError(f"unknown protocol {protocol!r}")


def compute_ap(ranks: Sequence[int], num_relevant: int) -> float:
    """Average precision from ascending 0-based ranks of relevant items.

    `ranks` positions must already have ignored images removed. This is trapezoidal
    integration of the precision/recall curve — the exact form the Oxford5k/Paris6k/
    revisitop benchmarks use, not the simpler step-function AP. Reproducing published
    numbers requires this form.
    """
    if num_relevant <= 0:
        raise ValueError("num_relevant must be > 0")
    ap = 0.0
    recall_step = 1.0 / num_relevant
    for j, rank in enumerate(ranks):
        precision_0 = 1.0 if rank == 0 else j / rank
        precision_1 = (j + 1) / (rank + 1)
        ap += (precision_0 + precision_1) * recall_step / 2.0
    return ap


def _positive_positions(ranking: Sequence[int], positive: set[int], ignored: set[int]) -> list[int]:
    """0-based positions of `positive` items in `ranking` after dropping `ignored` items."""
    ranking_list = [rank for rank in ranking if rank not in ignored]
    positives = [i for i, rank in enumerate(ranking_list) if rank in positive]
    return positives


def query_ap(ranking: Sequence[int], positive: set[int], ignored: set[int]) -> float:
    """AP for one query. Returns NaN if `positive` is empty (query excluded from mAP)."""
    if not positive:
        return float("nan")
    positions = _positive_positions(ranking, positive, ignored)
    return compute_ap(positions, len(positive))


def precision_at_k(ranking: Sequence[int], positive: set[int], ignored: set[int], k: int) -> float:
    """P@k, capped so a query with fewer than `k` ground-truth positives isn't deflated.

    Matches the reference `kq = min(max(pos), k)` denominator: if only 2 positives
    exist for a query, P@10 is computed over the first 2 positive ranks, not diluted
    by dividing by 10.
    """
    if not positive:
        return float("nan")
    positions = np.asarray(_positive_positions(ranking, positive, ignored)) + 1  # 1-based
    kq = min(int(positions.max()), k)
    return float((positions <= kq).sum()) / kq


def evaluate(
    rankings: Sequence[Sequence[int]],
    ground_truth: Sequence[QueryGroundTruth],
    protocol: Protocol,
    k: int = 10,
) -> EvalResult:
    """mAP and mP@k over all queries for one protocol.

    `rankings[i]` is database indices sorted by descending similarity for query `i`,
    excluding the query image itself. Queries with no positives under this protocol
    are excluded from both averages.
    """
    aps = []
    precisions = []
    num_excluded = 0
    for ranking, gt in zip(rankings, ground_truth, strict=True):
        positive, ignored = protocol_sets(gt.easy, gt.hard, gt.junk, protocol)
        if not positive:
            num_excluded += 1
            continue
        aps.append(query_ap(ranking, positive, ignored))
        precisions.append(precision_at_k(ranking, positive, ignored, k))
    return EvalResult(
        protocol=protocol,
        k=k,
        map=float(np.mean(aps)),
        mp_at_k=float(np.mean(precisions)),
        num_queries=len(aps),
        num_excluded=num_excluded,
    )
