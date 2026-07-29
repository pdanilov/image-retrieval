import pytest

import cbir.descriptors.classic.cache_db as cache_db


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    """Redirect the whole classic-pipeline cache — index and blobs — into `tmp_path`.

    One assignment covers everything because `cache_db` derives both the database path
    and every blob path from `DATA_DIR`. Any test that touches a cached RootSIFT
    extraction, vocabulary, or GMM must request this fixture: without it the test
    would read and write the real `data/` cache, which holds tens of GB of genuinely
    expensive descriptors and must never be polluted by test fixtures.

    Yields the temporary root, so tests can assert on blob layout (`tmp_cache /
    "rootsift" / "roxford5k_database.npz"`).
    """
    monkeypatch.setattr(cache_db, "DATA_DIR", tmp_path)
    return tmp_path
