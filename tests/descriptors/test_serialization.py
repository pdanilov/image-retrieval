"""The publishing format: `.safetensors` must give back exactly what `.pth` held."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from cbir.descriptors import mirror
from cbir.descriptors.serialization import FORMAT, read, write


def _checkpoint():
    state = {"features.0.weight": torch.randn(4, 3), "features.0.bias": torch.randn(4), "pool.p": torch.tensor(2.92)}
    meta = {
        "architecture": "vgg16",
        "pooling": "gem",
        "outputdim": 4,
        "mean": [0.485, 0.456, 0.406],
        # The published files store P in float64 and m in float32; both must survive.
        "Lw": {
            "retrieval-SfM-120k": {
                v: {"P": np.random.rand(4, 4), "m": np.random.rand(4, 1).astype(np.float32)} for v in ("ss", "ms")
            }
        },
    }
    return state, meta


def test_a_fine_tuned_checkpoint_round_trips_exactly(tmp_path):
    state, meta = _checkpoint()
    write(tmp_path / "c.safetensors", state, meta)
    got_state, got_meta = read(tmp_path / "c.safetensors")

    assert got_state.keys() == state.keys()
    assert all(torch.equal(got_state[k], v) and got_state[k].dtype == v.dtype for k, v in state.items())
    for variant in ("ss", "ms"):
        for name in ("P", "m"):
            want = meta["Lw"]["retrieval-SfM-120k"][variant][name]
            got = got_meta["Lw"]["retrieval-SfM-120k"][variant][name]
            assert got.dtype == want.dtype and np.array_equal(got, want)
    assert {k: v for k, v in got_meta.items() if k != "Lw"} == {k: v for k, v in meta.items() if k != "Lw"}


def test_a_bare_state_dict_round_trips_without_meta(tmp_path):
    state = {"0.weight": torch.randn(2, 2)}
    write(tmp_path / "s.safetensors", state)
    got, meta = read(tmp_path / "s.safetensors")
    assert meta is None and torch.equal(got["0.weight"], state["0.weight"])


def test_the_same_checkpoint_always_writes_the_same_bytes(tmp_path):
    # Deterministic bytes are what let anyone re-convert the original `.pth` and check
    # the mirror's SHA-256 instead of trusting it.
    state, meta = _checkpoint()
    write(tmp_path / "a.safetensors", state, meta)
    write(tmp_path / "b.safetensors", state, dict(reversed(list(meta.items()))))
    assert mirror.sha256(tmp_path / "a.safetensors") == mirror.sha256(tmp_path / "b.safetensors")


def test_pth_in_either_shape_reads_the_same_way(tmp_path):
    state, meta = _checkpoint()
    torch.save({"state_dict": state, "meta": meta}, tmp_path / "f.pth")
    torch.save(state, tmp_path / "bare.pth")
    assert read(tmp_path / "f.pth")[1]["architecture"] == "vgg16"
    assert read(tmp_path / "bare.pth")[1] is None


def test_a_safetensors_file_in_another_layout_is_refused(tmp_path):
    from safetensors.torch import save_file

    save_file({"w": torch.zeros(1)}, str(tmp_path / "x.safetensors"), metadata={"cbir": json.dumps({"meta": {}})})
    with pytest.raises(ValueError, match=FORMAT):
        read(tmp_path / "x.safetensors")


def test_a_changed_file_is_refused(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"weights")
    mirror.verify(path, mirror.sha256(path))
    with pytest.raises(ValueError, match="refusing a changed file"):
        mirror.verify(path, "0" * 64)


def test_only_mirror_prefixed_checkpoints_are_fetched(tmp_path):
    assert mirror.resolve(str(tmp_path / "local.pth")) == tmp_path / "local.pth"
    with pytest.raises(ValueError, match="not a mirrored file"):
        mirror.resolve("mirror:irt-deit-small-distilled-224/nothing.safetensors")


def test_a_missing_local_file_falls_back_to_the_mirror(monkeypatch, tmp_path):
    from cbir.descriptors.cnn import weights

    monkeypatch.setenv("CBIR_WEIGHTS_ROOT", str(tmp_path))
    name = mirror.reference(weights.CAFFE_FILES["vgg16"])
    monkeypatch.setitem(mirror.SHA256, name, "unused")
    monkeypatch.setattr(mirror, "fetch", lambda requested: tmp_path / requested.replace("/", "_"))
    assert weights.locate(weights.CAFFE_FILES["vgg16"], "url") == tmp_path / name.replace("/", "_")

    # A local original still wins over the mirror.
    original = tmp_path / weights.CAFFE_FILES["vgg16"]
    original.write_bytes(b"x")
    assert weights.locate(weights.CAFFE_FILES["vgg16"], "url") == original


def test_each_file_names_the_repository_of_its_own_network():
    from cbir.descriptors.cnn import finetuned, weights

    assert mirror.reference(weights.CAFFE_FILES["resnet101"]) == (
        "imagenet-caffe-resnet101/imagenet-caffe-resnet101-features-10a101d.safetensors"
    )
    assert mirror.reference(finetuned.FILES["vgg16"]) == (
        "retrieval-sfm120k-vgg16-gem/retrievalSfM120k-vgg16-gem-b4dcdc6.safetensors"
    )
    run = "deit-vitb16-dist-224-cls-margin0.85-lr5e-07-seed0"
    assert mirror.trained(run) == f"irt-deit-base-distilled-224/{run}.safetensors"
    with pytest.raises(ValueError, match="no repository"):
        mirror.trained("dinov3-vitl16-cls-margin0.85-lr1e-05-lora8-seed0")


def test_every_pinned_file_is_one_the_naming_rules_produce():
    from cbir.descriptors.cnn import finetuned, weights

    expected = {mirror.reference(f) for f in [*weights.CAFFE_FILES.values(), *finetuned.FILES.values()]}
    assert expected <= set(mirror.SHA256)
    assert all(
        name.split("/")[0] in {n.split("/")[0] for n in expected} | set(mirror.TRAINED.values())
        for name in mirror.SHA256
    )
