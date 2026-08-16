import numpy as np
import pytest

from cbir.configs.classic import BoWConfig, FisherConfig, RunConfig, VLADConfig, descriptor_params
from cbir.descriptors.classic.prepare import ClassicDescriptorInputs

# Three well-separated clusters in 2-D, so k-means/GMM at k=3 converge to something
# stable and the encoded dimensionality is small enough to assert by hand.
CLUSTERS = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]], dtype=np.float32)


def _inputs(num_database: int = 4, num_queries: int = 2) -> ClassicDescriptorInputs:
    rng = np.random.default_rng(0)
    pooled = np.repeat(CLUSTERS, 40, axis=0) + rng.normal(scale=0.1, size=(120, 2)).astype(np.float32)

    def per_image(count: int) -> list[np.ndarray]:
        return [
            CLUSTERS[rng.integers(0, 3, size=5)] + rng.normal(scale=0.1, size=(5, 2)).astype(np.float32)
            for _ in range(count)
        ]

    return ClassicDescriptorInputs(
        held_out_dataset="rparis6k",
        held_out_descriptors=pooled,
        database_descriptors=per_image(num_database),
        query_descriptors=per_image(num_queries),
    )


@pytest.mark.parametrize(
    ("config", "dim"),
    [
        (BoWConfig(k=3, seed=0), 3),  # BoW's dimensionality is k
        (VLADConfig(k=3, seed=0), 3 * 2),  # k * d
        (FisherConfig(k=3, seed=0), 2 * 3 * 2),  # 2 * k * d
    ],
)
def test_train_and_encode_returns_search_ready_vectors(config, dim, tmp_cache):
    database_vectors, query_vectors = config.train_and_encode(_inputs())

    assert database_vectors.shape == (4, dim)
    assert query_vectors.shape == (2, dim)
    # exact_search takes a plain dot product as cosine similarity, so anything that
    # reaches it must already be float32 and unit-norm -- otherwise the ranking is
    # silently scored on un-normalized magnitudes.
    for vectors in (database_vectors, query_vectors):
        assert vectors.dtype == np.float32
        assert np.linalg.norm(vectors, axis=1) == pytest.approx(1.0, abs=1e-5)


def test_database_and_queries_are_encoded_by_the_same_model(tmp_cache):
    # Queries encoded against a *different* vocabulary would still have the right
    # shape and norm, and would still produce a plausible-looking mAP -- just a wrong
    # one. Feeding one image through both paths pins that they share a codebook.
    inputs = _inputs()
    shared = inputs.database_descriptors[0]
    inputs = ClassicDescriptorInputs(
        held_out_dataset=inputs.held_out_dataset,
        held_out_descriptors=inputs.held_out_descriptors,
        database_descriptors=inputs.database_descriptors,
        query_descriptors=[shared],
    )

    database_vectors, query_vectors = VLADConfig(k=3, seed=0).train_and_encode(inputs)
    assert query_vectors[0] == pytest.approx(database_vectors[0], abs=1e-6)


def test_each_technique_exposes_only_its_own_knobs():
    # A single config carrying every field would let `--intra-norm` parse for BoW and
    # then be silently ignored; the union is what makes that unrepresentable.
    assert set(descriptor_params(BoWConfig())) == {"k", "seed"}
    assert set(descriptor_params(VLADConfig())) == {"k", "seed", "intra_norm", "power"}
    assert set(descriptor_params(FisherConfig())) == {"k", "seed", "power", "sample"}


def test_fisher_records_its_training_sample_size():
    # `sample` changes the fitted GMM, so a row that omitted it would not be
    # reproducible and two runs at different values would collide on RunRecord.key.
    assert descriptor_params(FisherConfig())["sample"] == 1_000_000
    assert descriptor_params(FisherConfig(sample=0))["sample"] == 0


def test_params_excludes_technique_so_keys_compare_like_for_like():
    # `technique` is a ClassVar, recorded as RunRecord.technique. If it leaked into
    # params it would be duplicated in every key and in the `cbir results` params column.
    assert "technique" not in descriptor_params(BoWConfig())
    assert BoWConfig.technique == "bow"
    assert VLADConfig.technique == "vlad"
    assert FisherConfig.technique == "fisher"


def test_run_config_has_no_held_out_field():
    # The held-out dataset is derived from `dataset`, never configured: a run that
    # trained its vocabulary on its own eval set would report an invalid mAP, and the
    # config type is the place that makes it impossible to ask for.
    fields = set(RunConfig(descriptor=BoWConfig()).__dataclass_fields__)
    assert fields == {"descriptor", "dataset", "mp_at_k"}
