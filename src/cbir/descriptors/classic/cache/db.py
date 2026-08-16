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


def _rename_outdated_gmm(conn: sqlite3.Connection) -> bool:
    """Rename a pre-`sample` `gmm` table aside so `SCHEMA` can recreate it. True if renamed.

    `CREATE TABLE IF NOT EXISTS` does nothing at all when the table already exists — it
    does not reconcile columns. So adding `sample` to `SCHEMA` reached new databases and
    silently missed every existing one, and Fisher died on `no such column: sample`
    against a cache created before the column was added. Tests never saw it: each builds
    its database fresh in `tmp_path`, where the table really is absent.

    Renaming *before* `SCHEMA` runs is what lets the new table be created from the one
    canonical DDL above rather than a second copy of it kept in sync here.
    """
    columns = {row[1] for row in conn.execute("PRAGMA table_info(gmm)")}
    if not columns or "sample" in columns:
        return False
    conn.execute("ALTER TABLE gmm RENAME TO gmm_pre_sample")
    return True


@contextmanager
def connect() -> Generator[sqlite3.Connection]:
    """One connection, schema ensured to exist and up to date, committed on a clean exit."""
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        migrating = _rename_outdated_gmm(conn)
        conn.executescript(SCHEMA)
        if migrating:
            # Rows predating the column were fitted on the whole pool, which is what
            # sample = 0 means -- so they stay valid cache entries rather than being
            # discarded and refitted.
            conn.execute(
                "INSERT INTO gmm (dataset, k, seed, sample, filepath) "
                "SELECT dataset, k, seed, 0, filepath FROM gmm_pre_sample"
            )
            conn.execute("DROP TABLE gmm_pre_sample")
        yield conn
        conn.commit()
    finally:
        conn.close()
