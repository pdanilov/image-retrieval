import numpy as np

import cbir.descriptors.classic.prepare as prepare_module
from cbir.descriptors.classic.prepare import ClassicDescriptorInputs, prepare_classic_inputs


def test_prepare_classic_inputs_wires_held_out_database_and_queries(monkeypatch):
    calls = {}

    # Datasets are opaque here -- prepare.py only passes them to image_paths/download,
    # both faked below, so a name tag is all they need to be distinguishable.
    held_out_ds = {"name": "held_out"}
    database_ds = {"name": "database"}
    query_ds = {"name": "query", "bbx": [(0, 0, 1, 1), (2, 2, 3, 3)]}

    paths = {
        "held_out": ["ho0.jpg", "ho1.jpg"],
        "database": ["db0.jpg", "db1.jpg", "db2.jpg"],
        "query": ["q0.jpg", "q1.jpg"],
    }

    def fake_held_out_database(split):
        calls["held_out_split"] = split
        return held_out_ds

    def fake_download(name):
        calls["download_name"] = name
        return query_ds, database_ds

    def fake_image_paths(dataset):
        return paths[dataset["name"]]

    def fake_iter_images(image_paths_):
        return (f"opened({path})" for path in image_paths_)

    def fake_crop_query(image, bbx):
        calls.setdefault("cropped", []).append((image, bbx))
        return f"cropped({image})"

    def fake_cached_pooled_descriptors(dataset, role, images):
        # `images` is a lazy iterable (see rootsift_cache.py) -- consume it to confirm
        # it's iterable and to see the images, mirroring what a real cache miss does.
        materialized = list(images)
        calls.setdefault("pooled_calls", []).append((dataset, role, materialized))
        return np.zeros((len(materialized), 128), dtype=np.float32)

    def fake_cached_extract_many(dataset, role, images):
        materialized = list(images)
        calls.setdefault("extract_many_calls", []).append((dataset, role, materialized))
        return [np.zeros((1, 128), dtype=np.float32) for _ in materialized]

    monkeypatch.setattr(prepare_module, "held_out_database", fake_held_out_database)
    monkeypatch.setattr(prepare_module, "download", fake_download)
    monkeypatch.setattr(prepare_module, "image_paths", fake_image_paths)
    monkeypatch.setattr(prepare_module, "iter_images", fake_iter_images)
    monkeypatch.setattr(prepare_module, "crop_query", fake_crop_query)
    monkeypatch.setattr(prepare_module, "cached_pooled_descriptors", fake_cached_pooled_descriptors)
    monkeypatch.setattr(prepare_module, "cached_extract_many", fake_cached_extract_many)

    result = prepare_classic_inputs("roxford5k")

    assert isinstance(result, ClassicDescriptorInputs)
    # The held-out training pool comes from the *other* dataset, never roxford5k itself.
    assert calls["held_out_split"].eval_dataset == "roxford5k"
    assert calls["held_out_split"].held_out_dataset == "rparis6k"
    # Carried through so vocabulary_cache/gaussian_mixture_cache key on the training
    # data's actual identity, not the eval dataset.
    assert result.held_out_dataset == "rparis6k"
    # Eval database/queries come from the eval dataset itself.
    assert calls["download_name"] == "roxford5k"
    # Queries are cropped before extraction, paired with their own bbx; database and
    # held-out images are used as-is.
    assert calls["cropped"] == [("opened(q0.jpg)", (0, 0, 1, 1)), ("opened(q1.jpg)", (2, 2, 3, 3))]

    # Held-out extraction is cached under the *held-out* dataset's own "database" role
    # (it's literally rparis6k's database cache, not a separate held-out entry), read
    # as one pooled array rather than a per-image list.
    assert calls["pooled_calls"] == [("rparis6k", "database", ["opened(ho0.jpg)", "opened(ho1.jpg)"])]
    assert calls["extract_many_calls"][0] == (
        "roxford5k",
        "database",
        ["opened(db0.jpg)", "opened(db1.jpg)", "opened(db2.jpg)"],
    )
    assert calls["extract_many_calls"][1] == (
        "roxford5k",
        "query",
        ["cropped(opened(q0.jpg))", "cropped(opened(q1.jpg))"],
    )

    assert result.held_out_descriptors.shape == (2, 128)
    assert len(result.database_descriptors) == 3
    assert len(result.query_descriptors) == 2
