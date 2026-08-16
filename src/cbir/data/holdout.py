"""Held-out data for training the classic tier's vocabulary and PCA-whitening.

roxford5k and rparis6k are the entire evaluation surface — neither may be used to
train anything that will later be evaluated on it (fitting a vocabulary or a
whitening transform on the images it will search is a form of fitting on the test
set). Cross-dataset training is the standard classic-CBIR convention (Philbin,
Jégou, and Arandjelović & Zisserman all cross-train Oxford<->Paris this way) and
costs nothing extra here, since both datasets are already downloaded via
`cbir.data.revisitop.download()`: whichever dataset a run evaluates, the *other*
one's database supplies the held-out images.

`HeldOutSplit` exists so that pairing can't drift apart or get mismatched in caller
code — the held-out dataset is always derived from the eval dataset, never supplied
independently, and construction is validated regardless of how it happens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from datasets import Dataset

from cbir.data.revisitop import download

EvalDataset = Literal["roxford5k", "rparis6k"]

_HELD_OUT_FOR: dict[EvalDataset, EvalDataset] = {
    "roxford5k": "rparis6k",
    "rparis6k": "roxford5k",
}


@dataclass(frozen=True)
class HeldOutSplit:
    """An eval dataset paired with the dataset that trains its vocabulary/PCA.

    Prefer constructing via `for_eval` — but even a direct `HeldOutSplit(...)` call
    is validated in `__post_init__`, so there is no way to end up with a split where
    `held_out_dataset` matches (or otherwise disagrees with) `eval_dataset`.
    """

    eval_dataset: EvalDataset
    held_out_dataset: EvalDataset

    def __post_init__(self) -> None:
        expected = _HELD_OUT_FOR.get(self.eval_dataset)
        if expected is None:
            raise ValueError(f"unknown eval_dataset {self.eval_dataset!r}, expected one of {sorted(_HELD_OUT_FOR)}")
        if self.held_out_dataset != expected:
            raise ValueError(
                f"held_out_dataset must be {expected!r} when eval_dataset={self.eval_dataset!r}, "
                f"got {self.held_out_dataset!r} — training on the evaluation dataset itself "
                "invalidates the resulting mAP."
            )

    @classmethod
    def for_eval(cls, eval_dataset: EvalDataset) -> HeldOutSplit:
        """The (only) correct split for evaluating on `eval_dataset`."""
        if eval_dataset not in _HELD_OUT_FOR:
            raise ValueError(f"unknown dataset {eval_dataset!r}, expected one of {sorted(_HELD_OUT_FOR)}")
        return cls(eval_dataset=eval_dataset, held_out_dataset=_HELD_OUT_FOR[eval_dataset])


def held_out_database(split: HeldOutSplit) -> Dataset:
    """Database images of `split.held_out_dataset` — the training pool for vocabulary/PCA.

    Uses the database split rather than queries: the larger, more diverse pool, and
    what the classic-CBIR convention above actually trains on.
    """
    _, db = download(split.held_out_dataset)
    return db
