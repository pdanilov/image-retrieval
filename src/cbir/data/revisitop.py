"""Download RevisitOP benchmark splits.

Uses the `galilai-group/revisitop` HF loading script (`trust_remote_code=True`,
requires `datasets<4.0`). Only `roxford5k` and `rparis6k` are exposed here — the
`revisitop1m` and `oxfordparis` configs in that script are broken (see AGENTS.md)
and are rejected with an explanation rather than silently producing bad data.

Downloads land in the default HF cache (`~/.cache/huggingface`), shared across
projects, rather than under this repo's `data/` — the raw archives are large,
identical for anyone using this benchmark, and not specific to this project.
"""

from __future__ import annotations

from dataclasses import dataclass

REPO_ID = "galilai-group/revisitop"

UNSUPPORTED_CONFIGS = {
    "revisitop1m": (
        "the revisitop1m split generator passes a mismatched keyword "
        "(ground_truth_file vs ground_truth_files) and raises TypeError before "
        "yielding any examples"
    ),
    "oxfordparis": (
        "the oxfordparis config concatenates Oxford and Paris ground truth without "
        "offsetting Paris indices past Oxford's image count, so Paris queries' "
        "easy/hard/junk indices point at the wrong (Oxford) images"
    ),
}


@dataclass(frozen=True)
class DatasetSpec:
    config: str
    expected_db: int
    expected_queries: int


SUPPORTED: dict[str, DatasetSpec] = {
    "roxford5k": DatasetSpec("roxford5k", expected_db=4993, expected_queries=70),
    "rparis6k": DatasetSpec("rparis6k", expected_db=6322, expected_queries=70),
}


def download(name: str):
    """Download and sanity-check one dataset's `qimlist`/`imlist` splits.

    Returns the (query, database) HF Dataset pair. Raises if the loaded split
    sizes don't match the verified counts — a mismatch means the loader dropped
    or renamed files and every ground-truth index downstream would be wrong.
    """
    if name in UNSUPPORTED_CONFIGS:
        raise ValueError(f"'{name}' is not supported: {UNSUPPORTED_CONFIGS[name]}. See AGENTS.md for details.")
    if name not in SUPPORTED:
        raise ValueError(f"unknown dataset '{name}', expected one of {sorted(SUPPORTED)}")

    # Imported lazily so the rest of the CLI doesn't pay for it on every invocation.
    from aiohttp import ClientTimeout
    from datasets import load_dataset

    spec = SUPPORTED[name]

    # aiohttp's default total timeout is 300s, which a multi-GB file on a slow link
    # (observed ~1-3 MB/s to robots.ox.ac.uk / cmp.felk.cvut.cz) will not finish
    # inside. Without this, fsspec raises FSTimeoutError partway through a transfer.
    storage_options = {"client_kwargs": {"timeout": ClientTimeout(total=3600)}}

    query = load_dataset(
        REPO_ID,
        spec.config,
        split="qimlist",
        trust_remote_code=True,
        storage_options=storage_options,
    )
    db = load_dataset(
        REPO_ID,
        spec.config,
        split="imlist",
        trust_remote_code=True,
        storage_options=storage_options,
    )

    if len(db) != spec.expected_db or len(query) != spec.expected_queries:
        raise RuntimeError(
            f"{name}: expected {spec.expected_db} database / {spec.expected_queries} "
            f"query images, got {len(db)} / {len(query)}. Do not proceed with this "
            "data — see AGENTS.md's note on silent index corruption."
        )

    print(f"{name}: {len(db)} database images, {len(query)} queries — OK")
    return query, db
