import sqlite3

import cbir.descriptors.classic.cache.db as db


def _old_gmm_schema(path):
    """A `gmm` table as it existed before `sample` was added to SCHEMA."""
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE gmm (dataset TEXT NOT NULL, k INTEGER NOT NULL, seed INTEGER NOT NULL,"
        " filepath TEXT NOT NULL, PRIMARY KEY (dataset, k, seed));"
    )
    conn.execute("INSERT INTO gmm VALUES ('rparis6k', 64, 0, '/blob.npz')")
    conn.commit()
    conn.close()


def test_connect_migrates_a_gmm_table_created_before_sample_existed(tmp_cache):
    # `CREATE TABLE IF NOT EXISTS` does not reconcile columns, so a column added to
    # SCHEMA reaches only fresh databases. This is the exact failure that killed a
    # Fisher sweep: `sqlite3.OperationalError: no such column: sample`.
    _old_gmm_schema(db.db_path())

    with db.connect() as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(gmm)")}
        assert "sample" in columns


def test_migration_keeps_existing_fits_rather_than_discarding_them(tmp_cache):
    # A pre-`sample` row was fitted on the whole pool, which is what sample = 0 means.
    # Dropping it instead would silently throw away hours of EM.
    _old_gmm_schema(db.db_path())

    with db.connect() as conn:
        rows = conn.execute("SELECT dataset, k, seed, sample, filepath FROM gmm").fetchall()
    assert rows == [("rparis6k", 64, 0, 0, "/blob.npz")]


def test_migration_runs_once_and_leaves_no_scratch_table(tmp_cache):
    _old_gmm_schema(db.db_path())

    with db.connect():
        pass
    with db.connect() as conn:  # second open must be a no-op, not a re-migration
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        rows = conn.execute("SELECT COUNT(*) FROM gmm").fetchone()[0]

    assert "gmm_pre_sample" not in tables
    assert rows == 1  # not duplicated by a second copy pass


def test_sample_is_part_of_the_primary_key_after_migration(tmp_cache):
    # The whole point of the column: two fits differing only by `sample` must coexist.
    # ALTER TABLE ADD COLUMN could not have done this -- it cannot change a primary key.
    _old_gmm_schema(db.db_path())

    with db.connect() as conn:
        conn.execute("INSERT INTO gmm VALUES ('rparis6k', 64, 0, 1000000, '/other.npz')")
        assert conn.execute("SELECT COUNT(*) FROM gmm").fetchone()[0] == 2
