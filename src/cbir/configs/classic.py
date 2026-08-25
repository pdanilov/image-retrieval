"""Typed configs for one classic-tier run — every knob that moves a number.

The three aggregators do not share a parameter set: BoW's only knobs are the
vocabulary size and seed, VLAD adds intra-normalization and a power law, Fisher has a
power law but no intra-normalization (its blocks are per-component already). A single
config with every field would let `cbir evaluate bow --intra-norm` parse cleanly and
then silently ignore the flag, so each technique gets its own dataclass and
`RunConfig.descriptor` is their union — tyro turns that into one subcommand per
technique, and `--help` on each shows only the knobs it actually honours.

They are united by structure, not inheritance: each carries `technique` and
implements `train_and_encode`, which is all `eval/runner.py` calls. There is no base
class because there is no shared behaviour to put in one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from cbir.configs.types import Encoded
from cbir.data.holdout import EvalDataset
from cbir.descriptors.classic.aggregate.bow import BagOfWords
from cbir.descriptors.classic.aggregate.fisher import FisherVector
from cbir.descriptors.classic.aggregate.vlad import VLAD
from cbir.descriptors.classic.prepare import ClassicDescriptorInputs, prepare_classic_inputs


@dataclass(frozen=True)
class BoWConfig:
    """Bag of visual words with tf-idf weighting.

    Args:
        k: Vocabulary size. This *is* the descriptor dimensionality for BoW, so it
            wants to be large — the literature uses 65k-1M words.
        seed: k-means seed. Part of the recorded config because the clustering is not
            deterministic across seeds.
    """

    k: int = 5000
    seed: int = 0

    technique: ClassVar[str] = "bow"
    inputs_kind: ClassVar[str] = "classic"

    def prepare(self, dataset: EvalDataset) -> ClassicDescriptorInputs:
        return prepare_classic_inputs(dataset)

    def train_and_encode(self, inputs: ClassicDescriptorInputs) -> Encoded:
        # `train_and_encode_database` rather than `train` + `encode`: idf is fitted on
        # the database histograms, so the two-call form walks the database twice (~19
        # min per walk at k=5000) to build the same array.
        model, database_vectors = BagOfWords.train_and_encode_database(inputs, k=self.k, seed=self.seed)
        return database_vectors, model.encode(inputs.query_descriptors)


@dataclass(frozen=True)
class VLADConfig:
    """VLAD: residuals to the nearest visual word.

    Args:
        k: Vocabulary size. VLAD's dimensionality is `k * 128`, so this stays small —
            k=64 is already an 8192-dim descriptor.
        seed: k-means seed.
        intra_norm: L2-normalize each per-word block before the global normalization.
        power: Signed power law applied before normalization; `None` skips it.
    """

    k: int = 64
    seed: int = 0
    intra_norm: bool = True
    power: float | None = None

    technique: ClassVar[str] = "vlad"
    inputs_kind: ClassVar[str] = "classic"

    def prepare(self, dataset: EvalDataset) -> ClassicDescriptorInputs:
        return prepare_classic_inputs(dataset)

    def train_and_encode(self, inputs: ClassicDescriptorInputs) -> Encoded:
        # Two encode calls rather than one: VLAD has no corpus-level fit, so database
        # and queries are independent passes through the same trained model.
        model = VLAD.train(inputs, k=self.k, seed=self.seed, intra_norm=self.intra_norm, power=self.power)
        return model.encode(inputs.database_descriptors), model.encode(inputs.query_descriptors)


@dataclass(frozen=True)
class FisherConfig:
    """Fisher vectors over a diagonal-covariance GMM.

    Args:
        k: Number of mixture components. Fisher's dimensionality is `2 * k * 128`,
            twice VLAD's at the same k, so this stays smaller still.
        seed: GMM initialization seed; also seeds the training subsample.
        power: Signed power law; 0.5 is the improved-Fisher signed square root.
        sample: Descriptors EM is fitted on, drawn from the held-out pool (0 = all).
            Defaults to 1M rather than 0 because sklearn's GaussianMixture is
            full-batch: on the ~21M-descriptor pools here, 0 means multi-GB E-step
            allocations and a full Lloyd k-means init. It is recorded in `params`
            and in the codebook cache key, so runs at different values stay distinct.
    """

    k: int = 64
    seed: int = 0
    power: float | None = 0.5
    sample: int = 1_000_000

    technique: ClassVar[str] = "fisher"
    inputs_kind: ClassVar[str] = "classic"

    def prepare(self, dataset: EvalDataset) -> ClassicDescriptorInputs:
        return prepare_classic_inputs(dataset)

    def train_and_encode(self, inputs: ClassicDescriptorInputs) -> Encoded:
        model = FisherVector.train(inputs, k=self.k, seed=self.seed, power=self.power, sample=self.sample)
        return model.encode(inputs.database_descriptors), model.encode(inputs.query_descriptors)


ClassicConfig = BoWConfig | VLADConfig | FisherConfig
