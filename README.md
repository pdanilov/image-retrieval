# image-retrieval
Modern image retrieval inspired by 10-year old bachelor's diploma nostalgia

## Setup

```
uv sync
```

## Downloading the benchmark data

```
uv run cbir download --datasets roxford5k rparis6k
# or: uv run cbir download --datasets all
```

Downloads the [ROxford5k and RParis6k](https://huggingface.co/datasets/galilai-group/revisitop)
image retrieval benchmarks into the default Hugging Face cache (`~/.cache/huggingface`),
shared across projects rather than duplicated locally. Each dataset is sanity-checked
against its known image/query counts before use.

Only these two configs are supported — the `revisitop1m` and `oxfordparis` configs in
the upstream loading script are broken and are rejected outright. See
[AGENTS.md](AGENTS.md) for details.

## Evaluating

```
uv run cbir evaluate --dataset roxford5k bow    --k 20000
uv run cbir evaluate --dataset roxford5k vlad   --k 256
uv run cbir evaluate --dataset roxford5k fisher --k 64

# a sweep is one run per k, sharing a single descriptor extraction
uv run cbir evaluate --dataset roxford5k --sweep-k 16 64 256 vlad --seed 0
```

Each run extracts RootSIFT, trains the codebook **on the other dataset**, encodes the
database and the bbx-cropped queries, searches exactly, scores all three protocols, and
appends a row to `results/runs.jsonl`. Read rows back with `cbir results`
(`--current-only` collapses re-runs). RootSIFT extraction and codebooks are cached under
`data/`, so re-running a `k` costs only the encode.

`k` means something different per technique — it is the descriptor dimensionality for
BoW, `k*128` for VLAD and `2*k*128` for Fisher — so compare on dimensionality, not `k`.

## Classic-tier results

roxford5k, codebook held out on rparis6k, seed 0, mAP ×100. Medium is the headline
number in the literature; Easy is near-saturated and reported in `results/runs.jsonl`
only.

| method | k | dim | Medium | Hard |
|---|---:|---:|---:|---:|
| BoW | 1000 | 1000 | 10.7 | 2.8 |
| BoW | 5000 | 5000 | 14.8 | 5.7 |
| BoW | 20000 | 20000 | 19.4 | 7.6 |
| BoW | 50000 | 50000 | 22.6 | 8.7 |
| Fisher | 16 | 4096 | 17.6 | 6.3 |
| Fisher | 64 | 16384 | 21.9 | 10.6 |
| Fisher | 128 | 32768 | 24.7 | 11.8 |
| Fisher | 256 | 65536 | 27.4 | 14.4 |
| VLAD | 16 | 2048 | 15.4 | 4.2 |
| VLAD | 32 | 4096 | 17.3 | 5.6 |
| VLAD | 64 | 8192 | 19.7 | 8.0 |
| VLAD | 128 | 16384 | 22.4 | 10.0 |
| VLAD | 256 | 32768 | 26.9 | 13.4 |
| VLAD | 512 | 65536 | 29.5 | 13.7 |
| **VLAD** | **1024** | **131072** | **33.1** | **16.2** |

**Validation.** Radenović et al. report `HesAff–rSIFT–VLAD` at Medium 33.9 / Hard 13.2 on
this benchmark ([1803.11285](https://arxiv.org/abs/1803.11285), Table 5). VLAD k=1024
here reaches 33.1 / 16.2 — with OpenCV DoG SIFT rather than Hessian-Affine, no
PCA-whitening, and a vocabulary trained on rparis6k rather than a separate landmark set.
Reproducing the reference number is what licenses trusting the rest of the table.

At matched dimensionality the ordering is **VLAD ≥ Fisher > BoW** at every point. Fisher
looks stronger at equal `k` only because its vector is twice as long there. No curve has
plateaued: every technique was still climbing at the largest `k` tried.

Caveats: single seed, so small gaps are unresolved; roxford5k only, so the ranking is not
cross-checked on rparis6k; Fisher's GMM is fitted on a seeded 1M-descriptor subsample.

`notebooks/classic_comparison.ipynb` plots all of this from `results/runs.jsonl`. Run it
top to bottom — it ships with outputs stripped.

### Reproducing the table

```
uv run cbir evaluate --dataset roxford5k --sweep-k 1000 5000 20000 50000 bow    --seed 0
uv run cbir evaluate --dataset roxford5k --sweep-k 16 32 64 128 256 512 1024 vlad --seed 0
uv run cbir evaluate --dataset roxford5k --sweep-k 16 64 128 256 fisher --seed 0
```

Roughly 9 hours cold on 16 cores, most of it BoW's k-means at k=20000 and k=50000. VLAD
is ~13–18 min per point at any `k` — its cost is loading descriptors, not clustering.
