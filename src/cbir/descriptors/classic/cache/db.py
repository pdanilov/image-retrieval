"""Shared SQLite index for the classic pipeline's disk caches.

`rootsift.py`, `vocabulary.py`, and `gaussian_mixture.py` in this package each still
store their actual array data in `.npz` files under `data/<name>/` (gitignored — see
`.gitignore`'s `/data/` rule) -- this database (`data/cache.sqlite3`, also under that
same rule) is purely an index from a cache key to the `.npz` file that holds it. A
cache hit used to mean "a file exists at this conventionally-named path"; it now means
"a row exists for this key", with the file path read back out of that row -- so the
three cache modules are free to name their files however's convenient, rather than
that naming being the cache's actual source of truth.

One short-lived connection per operation, never held open: these are offline batch
jobs run one at a time (extraction, k-means, EM), not a service under concurrent load,
so there is no connection pooling to get right.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import cbir

# cbir/__init__.py -> cbir/ -> src/ -> repo root. Every cached artefact in the classic
# pipeline — the index and all the `.npz` blobs it points at — hangs off this one
# directory, so redirecting the whole cache (as the tests do) is a single assignment.
DATA_DIR = Path(cbir.__file__).resolve().parents[2] / "data"


def db_path() -> Path:
    """Where the index lives.

    A function rather than a module constant so that it tracks `DATA_DIR`: a constant
    would freeze the path at import time, and anything repointing `DATA_DIR` would
    have to know to repoint the database separately too.
    """
    return DATA_DIR / "cache.sqlite3"


def reserve_blob_path(table: str, stem: str) -> Path:
    """Where a fresh `.npz` for `table` should be written, parent directory created.

    One subdirectory per table, because stems are *not* unique across tables: `kmeans`
    and `gmm` both name their files `{dataset}_k{k}_seed{seed}`, so a flat directory
    would silently have a GMM overwrite a vocabulary trained with the same parameters.
    """
    path = DATA_DIR / table / f"{stem}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


SCHEMA = """
CREATE TABLE IF NOT EXISTS rootsift (
    dataset  TEXT NOT NULL,
    split    TEXT NOT NULL,
    filepath TEXT NOT NULL,
    PRIMARY KEY (dataset, split)
);
CREATE TABLE IF NOT EXISTS kmeans (
    dataset  TEXT NOT NULL,
    k        INTEGER NOT NULL,
    seed     INTEGER NOT NULL,
    filepath TEXT NOT NULL,
    PRIMARY KEY (dataset, k, seed)
);
CREATE TABLE IF NOT EXISTS gmm (
    dataset  TEXT NOT NULL,
    k        INTEGER NOT NULL,
    seed     INTEGER NOT NULL,
    -- How many descriptors EM actually saw (0 = the whole pool). Part of the key, not
    -- a detail: a GMM fitted on 1M descriptors is a different model from one fitted on
    -- 21M, and without this column the first would silently be served for the second.
    sample   INTEGER NOT NULL DEFAULT 0,
    filepath TEXT NOT NULL,
    PRIMARY KEY (dataset, k, seed, sample)
);
"""


@contextmanager
def connect() -> Generator[sqlite3.Connection]:
    """One connection, schema ensured to exist, committed on a clean exit."""
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()
