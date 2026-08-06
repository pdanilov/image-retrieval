"""Disk caches for the expensive steps of the classic pipeline.

`RootSIFTCache`, `VocabularyCache`, and `GaussianMixtureCache` each cache one
computation — RootSIFT extraction, k-means vocabulary training, GMM fitting — as an
`.npz` blob indexed by `db.py`'s SQLite database. The index, not the filesystem layout,
decides whether something is cached; see `db.py`.

The split from `local/` and `codebook/` is deliberate. Cache keys are pipeline
provenance — which eval dataset, database or query images, which held-out set trained
this codebook — and none of that is a property of a SIFT extractor or a k-means model.
Keeping it here is what lets `RootSIFT`, `Vocabulary`, and `GaussianMixture` stay pure:
no filesystem, no SQLite, and unit tests that need no fixtures.

`db.py` stays plain functions rather than a class: it is the storage primitive the three
caches share, and `connect()` is a context manager, which is idiomatic as a function.
"""
