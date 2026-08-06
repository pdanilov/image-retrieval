"""Codebooks: the models fit to a pool of local descriptors on a held-out dataset.

`Vocabulary` (hard k-means centers, used by BoW and VLAD) and `GaussianMixture` (soft
diagonal-covariance GMM, used by Fisher vectors). Both MUST be fit on a held-out
dataset rather than the evaluation database — see AGENTS.md.
"""
