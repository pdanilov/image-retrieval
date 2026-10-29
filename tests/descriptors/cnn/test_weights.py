import pytest
import torch

from cbir.descriptors.cnn import weights
from cbir.descriptors.cnn.pooling import PooledCNN


def test_default_is_torchvision():
    # Every recorded row so far used torchvision weights; a changed default would make
    # them silently irreproducible.
    assert PooledCNN("alexnet", device="cpu").weights == "torchvision"


def test_unpublished_backbone_is_rejected():
    with pytest.raises(ValueError, match="no caffe weights published"):
        weights.load_caffe(torch.nn.Linear(1, 1), "alexnet")


def test_missing_file_names_the_download(monkeypatch, tmp_path):
    from cbir.descriptors import mirror

    monkeypatch.setenv("CBIR_WEIGHTS_ROOT", str(tmp_path))
    monkeypatch.setattr(mirror, "SHA256", {})
    with pytest.raises(FileNotFoundError, match="not mirrored — download it from"):
        weights.load_caffe(torch.nn.Linear(1, 1), "resnet101")


def test_an_unreachable_mirror_names_both_ways_to_get_the_file(monkeypatch, tmp_path):
    # Private repositories: no token, or no access, must read as "cannot get it", not as
    # a crash deep inside the Hub client.
    from huggingface_hub.errors import LocalEntryNotFoundError

    from cbir.descriptors import mirror

    monkeypatch.setenv("CBIR_WEIGHTS_ROOT", str(tmp_path))

    def offline(name):
        raise LocalEntryNotFoundError("offline")

    monkeypatch.setattr(mirror, "fetch", offline)
    with pytest.raises(FileNotFoundError, match=r"imagenet-caffe-resnet101 could not be read.*download it from"):
        weights.load_caffe(torch.nn.Linear(1, 1), "resnet101")


def test_a_changed_mirror_file_is_not_passed_off_as_unreachable(monkeypatch, tmp_path):
    from cbir.descriptors import mirror

    monkeypatch.setenv("CBIR_WEIGHTS_ROOT", str(tmp_path))

    def tampered(name):
        raise ValueError("refusing a changed file")

    monkeypatch.setattr(mirror, "fetch", tampered)
    with pytest.raises(ValueError, match="refusing a changed file"):
        weights.load_caffe(torch.nn.Linear(1, 1), "resnet101")


def test_a_mismatched_file_is_rejected_rather_than_half_loaded(monkeypatch, tmp_path):
    # load_state_dict(strict=False) silently accepts a file describing another network,
    # leaving a partly random model that produces plausible-looking descriptors.
    monkeypatch.setenv("CBIR_WEIGHTS_ROOT", str(tmp_path))
    path = tmp_path / weights.CAFFE_FILES["resnet50"]
    torch.save({"nonsense.weight": torch.zeros(2, 2)}, path)

    with pytest.raises(ValueError, match="does not match resnet50"):
        weights.load_caffe(PooledCNN._build("resnet50")[0], "resnet50")


def test_only_num_batches_tracked_may_be_absent(monkeypatch, tmp_path):
    # The published files omit exactly those buffers, which matter only while training.
    monkeypatch.setenv("CBIR_WEIGHTS_ROOT", str(tmp_path))
    model, _ = PooledCNN._build("resnet50")
    state = {k: v for k, v in model.state_dict().items() if not k.endswith("num_batches_tracked")}
    torch.save(state, tmp_path / weights.CAFFE_FILES["resnet50"])

    assert weights.load_caffe(model, "resnet50") is model
