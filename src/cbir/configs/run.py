"""One evaluation: a descriptor configuration against one benchmark.

`RunConfig` lives here rather than beside any one tier's configs, because it composes
them: `DescriptorConfig` is the union across eras, and a tier that owned it would have
to import its sibling.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from cbir.configs.classic import ClassicConfig
from cbir.configs.cnn import CNNConfig
from cbir.data.holdout import EvalDataset

DescriptorConfig = ClassicConfig | CNNConfig
"""Every descriptor the harness can evaluate, across eras.

Members share no base class — only a shape: `technique`, `inputs_kind`, `prepare` and
`train_and_encode`. That is all `eval/runner.py` calls, and inheritance would add a
hierarchy without adding behaviour, since no two tiers share an implementation.
"""


@dataclass(frozen=True)
class RunConfig:
    """A descriptor, a benchmark, and how the score is cut.

    The held-out dataset is deliberately *not* a field — it is derived from `dataset`
    by `HeldOutSplit.for_eval`, so a run that trains its vocabulary (or fits its PCA)
    on the images it will be scored against is not expressible.

    Args:
        descriptor: Which descriptor to run, with its own knobs.
        dataset: Benchmark to evaluate on.
        mp_at_k: Cutoff for mean precision@k. mAP is always over the full ranking.
    """

    descriptor: DescriptorConfig
    dataset: EvalDataset = "roxford5k"
    mp_at_k: int = 10


def descriptor_params(descriptor: DescriptorConfig) -> dict[str, Any]:
    """The descriptor's knobs as a plain dict, for `RunRecord.params`.

    `technique` and `inputs_kind` are `ClassVar`s and so are not dataclass fields —
    `technique` is recorded separately as `RunRecord.technique`, and keeping both out
    of `params` is what makes `RunRecord.key` compare like-for-like across techniques.
    """
    return asdict(descriptor)
