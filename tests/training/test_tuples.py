"""Hard-negative mining: the two exclusion rules, and that "hard" really means hard.

Mining is the part of this recipe with no loud failure mode. Take negatives from the
query's own cluster and the model is taught that matching images should be far apart;
take five from one wrong cluster and four of the five negatives are redundant. Either
bug trains fine and just produces a worse network, so the rules are pinned here.
"""

from __future__ import annotations

import pytest
import torch

from cbir.training.tuples import Corpus, mine_negatives, root

needs_corpus = pytest.mark.skipif(
    not (root() / "retrieval-SfM-120k.pkl").exists(),
    reason="retrieval-SfM-120k index not downloaded",
)
"""The three tests below read the real corpus index, which is a manual download.

The mining rules above are pinned on hand-built fixtures and always run; these assert
the corpus's own invariants, which cannot be checked without the corpus. Skipping keeps
CI honest about the difference rather than making the whole file conditional."""


def test_negatives_never_come_from_the_query_cluster():
    # Those are positives that happen not to be listed as pairs. Training against them
    # teaches the opposite of the intended thing.
    cluster = [0, 0, 0, 1, 2, 3]
    pool = [0, 1, 2, 3, 4, 5]
    # Rank the query's own cluster highest -- exactly the case that must be skipped.
    scores = torch.tensor([[9.0], [8.0], [7.0], [3.0], [2.0], [1.0]])

    mined = mine_negatives(scores, pool, query_clusters=[0], cluster=cluster, count=3)

    assert mined == [[3, 4, 5]]
    assert all(cluster[i] != 0 for i in mined[0])


def test_each_negative_comes_from_a_different_cluster():
    # Five photographs of one wrong building is one mistake counted five times.
    cluster = [0, 1, 1, 1, 2, 3]
    pool = [0, 1, 2, 3, 4, 5]
    scores = torch.tensor([[0.0], [9.0], [8.0], [7.0], [6.0], [5.0]])

    mined = mine_negatives(scores, pool, query_clusters=[0], cluster=cluster, count=3)

    assert mined == [[1, 4, 5]]
    assert len({cluster[i] for i in mined[0]}) == 3


def test_the_hardest_eligible_candidate_is_taken_first():
    cluster = [0, 1, 2, 3]
    pool = [0, 1, 2, 3]
    scores = torch.tensor([[0.0], [1.0], [5.0], [3.0]])

    mined = mine_negatives(scores, pool, query_clusters=[0], cluster=cluster, count=3)

    assert mined == [[2, 3, 1]]  # descending similarity


def test_mining_runs_per_query_independently():
    cluster = [0, 1, 2, 3]
    pool = [0, 1, 2, 3]
    scores = torch.tensor([[0.0, 0.0], [9.0, 1.0], [1.0, 9.0], [0.5, 0.5]])

    mined = mine_negatives(scores, pool, query_clusters=[0, 0], cluster=cluster, count=1)

    assert mined == [[1], [2]]


def test_a_short_pool_yields_fewer_negatives_rather_than_repeating():
    # Repeating a negative would double-count one mistake in the loss; returning fewer
    # is the honest outcome. A real 22000-image pool never reaches this.
    cluster = [0, 1]
    mined = mine_negatives(torch.tensor([[0.0], [1.0]]), [0, 1], [0], cluster, count=5)

    assert mined == [[1]]


@needs_corpus
def test_the_corpus_splits_are_disjoint_and_the_expected_size():
    # The val split exists to measure generalization; overlap would make it meaningless.
    train, val = Corpus.load("train"), Corpus.load("val")

    assert len(train.cids) == 91642
    assert len(val.cids) == 6403
    assert not set(train.cids) & set(val.cids)


@needs_corpus
def test_every_pair_indexes_a_real_image():
    val = Corpus.load("val")

    assert max(val.qidxs) < len(val.cids)
    assert max(val.pidxs) < len(val.cids)
    # A pair must be two different photographs.
    assert all(q != p for q, p in zip(val.qidxs, val.pidxs, strict=True))


@needs_corpus
def test_a_pair_is_always_within_one_cluster():
    # The corpus's own invariant, and the basis for excluding the query's cluster when
    # mining: if positives could span clusters, that exclusion would be wrong.
    val = Corpus.load("val")

    assert all(val.cluster[q] == val.cluster[p] for q, p in zip(val.qidxs, val.pidxs, strict=True))
