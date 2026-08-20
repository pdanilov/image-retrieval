import pickle

import pytest

from cbir.descriptors.cnn import sfm


def test_path_nests_by_the_last_hash_bytes():
    # The archive nests by the *last* three byte-pairs reversed: a hash ending 8f4646
    # lives at 46/46/8f. Guessing the leading bytes silently finds nothing at all.
    path = sfm.image_path("0305c98b306b18a94d202a1d2e8f4646")
    assert path.parts[-4:] == ("46", "46", "8f", "0305c98b306b18a94d202a1d2e8f4646")


def test_root_is_overridable(monkeypatch, tmp_path):
    monkeypatch.setenv("CBIR_SFM_ROOT", str(tmp_path))
    assert sfm.root() == tmp_path


def test_held_out_is_not_a_corpus_here():
    # held_out paths come from the eval split; routing them through this module would
    # silently fit on whatever happened to be downloaded.
    with pytest.raises(ValueError, match="held_out draws its paths"):
        sfm.whitening_paths("held_out")


def test_missing_download_says_so_and_names_the_override(monkeypatch, tmp_path):
    monkeypatch.setenv("CBIR_SFM_ROOT", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="manual download"):
        sfm.whitening_paths("sfm30k")


def test_unextracted_images_are_caught_before_a_long_run(monkeypatch, tmp_path):
    # The manifest is 1 MB and the images are 37 GB; having one without the other is
    # the likely half-finished state, and it must not surface an hour into extraction.
    monkeypatch.setenv("CBIR_SFM_ROOT", str(tmp_path))
    (tmp_path / "retrieval-SfM-30k-whiten.pkl").write_bytes(pickle.dumps({"cids": ["0" * 32]}))
    with pytest.raises(FileNotFoundError, match="not extracted"):
        sfm.whitening_paths("sfm30k")


def test_paths_are_returned_in_manifest_order(monkeypatch, tmp_path):
    monkeypatch.setenv("CBIR_SFM_ROOT", str(tmp_path))
    cids = ["a" * 32, "b" * 32]
    (tmp_path / "retrieval-SfM-30k-whiten.pkl").write_bytes(pickle.dumps({"cids": cids}))
    for cid in cids:
        target = sfm.image_path(cid)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"")

    assert [p.split("/")[-1] for p in sfm.whitening_paths("sfm30k")] == cids
