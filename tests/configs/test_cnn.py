import numpy as np
import pytest

import cbir.configs.cnn as cnn_config
from cbir.configs.cnn import NeuralCodesConfig, descriptor_dim
from cbir.configs.run import descriptor_params
from cbir.descriptors.cnn.neural_codes import DIM
from cbir.descriptors.cnn.prepare import EvalImages


class _FakeModel:
    """Stands in for the network: one deterministic unit vector per image."""

    def __init__(self, *args, **kwargs):
        self.calls = []

    def extract(self, images):
        images = list(images)
        self.calls.append(len(images))
        rng = np.random.default_rng(len(images))
        out = rng.normal(size=(len(images), DIM)).astype(np.float32)
        return out / np.linalg.norm(out, axis=1, keepdims=True)


@pytest.fixture
def wired(monkeypatch, tmp_path):
    """Replace the network and image decoding; keep the config's own logic real."""
    model = _FakeModel()
    monkeypatch.setattr(cnn_config, "NeuralCodes", lambda *a, **k: model)
    # iter_images would try to open files that do not exist; the fake model only
    # counts what it is given.
    monkeypatch.setattr(cnn_config, "iter_images", lambda paths: list(paths))
    monkeypatch.setattr(cnn_config, "crop_query", lambda image, box: image)

    inputs = EvalImages(
        held_out_paths=[f"h{i}" for i in range(20)],
        database_paths=[f"d{i}" for i in range(6)],
        query_paths=["q0", "q1"],
        query_boxes=[[0, 0, 10, 10], [1, 1, 9, 9]],
    )
    return model, inputs


def test_uncompressed_returns_the_raw_code_width(wired):
    model, inputs = wired

    database, queries = NeuralCodesConfig(dim=None).train_and_encode(inputs)

    assert database.shape == (6, DIM)
    assert queries.shape == (2, DIM)
    # Only database and queries go through the network -- no PCA, so the held-out
    # images are never extracted.
    assert model.calls == [6, 2]


def test_pca_compresses_both_sides_to_the_configured_width(wired):
    _, inputs = wired

    database, queries = NeuralCodesConfig(dim=8).train_and_encode(inputs)

    assert database.shape == (6, 8)
    assert queries.shape == (2, 8)
    assert np.linalg.norm(database, axis=1) == pytest.approx(1.0, abs=1e-5)


def test_pca_is_fitted_on_the_held_out_images(wired):
    # The rule the whole benchmark rests on: nothing may be fitted on the images it
    # will be used to search. The third extraction call is the held-out pool, and it
    # only happens when a PCA is actually being fitted.
    model, inputs = wired

    NeuralCodesConfig(dim=8).train_and_encode(inputs)

    assert model.calls == [6, 2, 20]  # database, queries, then the 20 held-out images


def test_queries_are_cropped_to_their_ground_truth_box(monkeypatch, wired):
    _, inputs = wired
    cropped = []
    monkeypatch.setattr(cnn_config, "crop_query", lambda image, box: cropped.append((image, box)) or image)

    NeuralCodesConfig(dim=None).train_and_encode(inputs)

    # Every query, and only queries -- cropping a database image would be a bug.
    assert cropped == [("q0", [0, 0, 10, 10]), ("q1", [1, 1, 9, 9])]


def test_params_record_the_knobs_that_move_the_number():
    assert set(descriptor_params(NeuralCodesConfig())) == {"backbone", "dim", "seed"}
    assert "technique" not in descriptor_params(NeuralCodesConfig())
    assert NeuralCodesConfig.technique == "neural_codes"


def test_inputs_kind_separates_this_tier_from_the_classic_one():
    # run_all shares prepared inputs keyed on (dataset, inputs_kind); a CNN config
    # declaring "classic" would be handed RootSIFT arrays it cannot use.
    assert NeuralCodesConfig.inputs_kind == "images"


def test_descriptor_dim_reports_the_encoded_width():
    assert descriptor_dim(NeuralCodesConfig(dim=None)) == DIM
    assert descriptor_dim(NeuralCodesConfig(dim=128)) == 128
