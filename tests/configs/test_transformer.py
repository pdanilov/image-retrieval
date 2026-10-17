import numpy as np
import pytest

import cbir.configs.transformer as transformer_config
from cbir.configs.run import descriptor_params
from cbir.configs.transformer import ViTConfig
from cbir.descriptors.cnn.prepare import EvalImages
from cbir.descriptors.transformer.vit import SPECS

WIDTH = 16


class _FakeViT:
    """Stands in for the network: one deterministic unit vector per image."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.calls: list[list] = []

    def extract(self, images):
        images = list(images)
        self.calls.append(images)
        out = np.random.default_rng(len(images)).normal(size=(len(images), WIDTH)).astype(np.float32)
        return out / np.linalg.norm(out, axis=1, keepdims=True)


@pytest.fixture
def wired(monkeypatch):
    """Replace the network, decoding and cropping; keep the config's own logic real."""
    built: list[_FakeViT] = []

    def build(*args, **kwargs):
        built.append(_FakeViT(*args, **kwargs))
        return built[-1]

    monkeypatch.setattr(transformer_config, "FrozenViT", build)
    monkeypatch.setattr(transformer_config, "iter_images", lambda paths: list(paths))
    monkeypatch.setattr(transformer_config, "crop_query", lambda image, box: (image, tuple(box)))

    inputs = EvalImages(
        held_out_paths=[f"h{i}" for i in range(20)],
        database_paths=[f"d{i}" for i in range(6)],
        query_paths=["q0", "q1"],
        query_boxes=[[0, 0, 10, 10], [1, 1, 9, 9]],
    )
    return built, inputs


def test_whitening_is_fitted_on_the_held_out_images_only(wired, monkeypatch):
    built, inputs = wired
    fitted_on: list = []

    class _FakePCA:
        @staticmethod
        def fit(descriptors, *, dim, whiten, shrinkage, seed):
            fitted_on.append((len(descriptors), dim, whiten, shrinkage, seed))
            return _FakePCA()

        def transform(self, vectors):
            return vectors[:, :4]

    monkeypatch.setattr(transformer_config, "PCACompression", _FakePCA)

    database, queries = ViTConfig(whiten=True, shrinkage=0.1).train_and_encode(inputs)

    # database, queries, then the 20 held-out images -- never the database as fitting set
    assert [len(call) for call in built[0].calls] == [6, 2, 20]
    assert built[0].calls[2] == inputs.held_out_paths
    # Full width when whitening without an explicit dim: a rotation, not a compression.
    assert fitted_on == [(20, SPECS["dinov2-vitl14"].width, True, 0.1, 0)]
    assert database.shape == (6, 4) and queries.shape == (2, 4)


def test_learned_whitening_is_rejected_for_a_frozen_vit():
    with pytest.raises(ValueError, match="learned"):
        ViTConfig(whiten=True, whiten_source="learned")


def test_encodes_database_and_queries_and_fits_nothing(wired):
    built, inputs = wired

    database, queries = ViTConfig().train_and_encode(inputs)

    assert database.shape == (6, WIDTH)
    assert queries.shape == (2, WIDTH)
    # Nothing is fitted, so the held-out images are never extracted.
    (model,) = built
    assert [len(call) for call in model.calls] == [6, 2]


def test_queries_are_cropped_to_their_own_box_in_order(wired):
    built, inputs = wired

    ViTConfig().train_and_encode(inputs)

    assert built[0].calls[1] == [("q0", (0, 0, 10, 10)), ("q1", (1, 1, 9, 9))]


def test_the_network_receives_the_configured_model_and_size(wired):
    built, inputs = wired

    ViTConfig(model="dinov2-vitb14", image_size=448, resize="aspect", interpolation="bilinear").train_and_encode(inputs)

    assert built[0].args == ("dinov2-vitb14",)
    assert built[0].kwargs == {"image_size": 448, "resize": "aspect", "interpolation": "bilinear"}


def test_params_record_every_knob_and_nothing_else():
    assert descriptor_params(ViTConfig()) == {
        "model": "dinov2-vitl14",
        "image_size": 224,
        "resize": "pad",
        "interpolation": "bicubic",
        "dim": None,
        "whiten": False,
        "whiten_source": "held_out",
        "shrinkage": 0.0,
        "seed": 0,
    }
    assert ViTConfig.technique == "vit"
