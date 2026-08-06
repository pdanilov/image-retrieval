# AGENTS.md

Guidance for AI agents working in this repository.

## What this project is

A re-implementation of the author's bachelor's diploma topic, **Content-Based Image
Retrieval**, using modern computer vision. The point is a *controlled comparison*:
the diploma-era pipeline (hand-crafted local features → visual vocabulary → inverted
index) and the modern one (foundation-model global descriptors → dense vector search)
evaluated on the same benchmarks, with the same protocol, in the same harness.

Both halves matter. The classic baseline is not a throwaway — it is the thing the
project exists to measure against. Do not degrade it to make the modern side look
better, and report its numbers with the same rigor.

**Success = trustworthy, reproducible mAP numbers.** A pipeline that reports a high
mAP because of a ground-truth indexing bug is worse than no pipeline. Correctness of
the evaluation harness outranks descriptor quality, speed, and code elegance.

## Environment

- Python 3.13, managed with **uv** (`uv sync`, `uv run …`). Never call `pip install`
  directly or create venvs by hand.
- GPU: **RTX 5070 Ti, 16 GB, Blackwell (sm_120)**. This constrains two things:
  - PyTorch must come from the **cu128** wheel index. Wheels built for CUDA ≤ 12.4 do
    not contain sm_120 kernels and fail at runtime with "no kernel image is available".
    Configure this as an index in `pyproject.toml`, not as an ad-hoc install command.
  - 16 GB is the memory ceiling — batch sizes must respect it; prefer fp16/bf16
    inference. At the benchmark's actual scale (5–6k images) everything fits
    comfortably, so do not build streaming/sharding machinery for a problem we do
    not have.
- After any change to the torch install, sanity-check the GPU actually works — a
  successful install is not evidence of a working kernel:
  ```
  uv run python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability()); print((torch.randn(8,8,device='cuda')@torch.randn(8,8,device='cuda')).sum().item())"
  ```
  The matmul is the real test; `torch.cuda.is_available()` returns `True` on a broken
  sm_120 install.

## Layout

```
src/cbir/
  configs/       typed dataclass configs for datasets/descriptors/search/eval
  data/          dataset access, ground-truth parsing, image loading/cropping
  descriptors/
    classic/     SIFT/RootSIFT → k-means vocabulary → BoW, VLAD, (Fisher)
    cnn/         Neural Codes; GeM (⊃ SPoC, MAC), R-MAC, whitening
    foundation/  DINOv2/DINOv3, CLIP, SigLIP global descriptors
  search/        exact torch matmul search (no ANN — see Search backend)
  eval/          mAP / mP@k, the three protocols, results serialization
  cli.py         entry point: extract / index / search / evaluate
notebooks/       analysis and figures ONLY — no pipeline logic
results/         committed JSONL run outputs (append-only, one run per line)
tests/
data/            gitignored: cached descriptors (NOT raw dataset archives — see below)
```

Raw benchmark archives (images, ground-truth pickles) are **not** cached under this
repo's `data/`. `cbir.data.revisitop.download()` lets `datasets` use its own default
cache (`~/.cache/huggingface`) rather than overriding `HF_DATASETS_CACHE` — this data
is identical for anyone using this benchmark and multi-GB, so it belongs in the
user-wide HF cache, not duplicated per-project. Setting `HF_DATASETS_CACHE` after
`datasets` has already been imported is also a no-op (the library reads it at import
time) — if a project-local override is ever wanted, it must be set before the import.

Configs are plain Python dataclasses, not YAML — there is no separate config-loading
or path-resolution step. `tyro.cli()` builds the CLI directly from a top-level
`RunConfig` dataclass (composed of the per-group dataclasses: dataset, descriptor,
search, eval), so a run is invoked as e.g. `cbir extract --descriptor.p 3.0
--dataset.name roxford5k`, and the same dataclasses are constructed directly in code
for notebooks or tests. A parameter sweep (e.g. the GeM backbone/dataset/protocol
validation grid) is a Python loop building a list of `RunConfig` instances and calling
the run function for each — no sweeper plugin needed at this scale.

`notebooks/` imports from `src/cbir/`, never the other way around. Anything a notebook
needs twice belongs in the package.

## The benchmark

`revisitop` (Radenović et al., 2018, *Revisiting Oxford and Paris*). These are the
verified facts — they are load-bearing, so treat a mismatch as a bug in the loader,
not as an acceptable variation:

| config       | database images | queries | valid gt index range | used? |
|--------------|-----------------|---------|----------------------|-------|
| `roxford5k`  | 4993            | 70      | 0–4992               | **yes** |
| `rparis6k`   | 6322            | 70      | 0–6321               | **yes** |
| `revisitop1m`| ~1,001,001 distractors | —  | n/a                  | **no — do not download** |
| `oxfordparis`| —               | —       | —                    | **no — broken, never use** |

**The benchmark is `roxford5k` + `rparis6k`, and nothing else.** Those two are the
entire evaluation surface. Do not add a dataset to the comparison without being asked.

- **`revisitop1m` is deferred, not planned.** It is 444 GB across 100 `tar.gz`
  archives, plus a full extraction pass per descriptor config, and it forces ANN
  search — which then has to be characterized against exact search before any number
  from it means anything. It answers a scalability question, not the classic-vs-modern
  question this project exists to answer. **Do not download it**, do not write ingestion
  code for it, and do not treat its absence as an incomplete pipeline. Revisit only
  when the small-scale harness reproduces published GeM numbers *and* someone asks for
  it. A cheap distractor-subset scaling curve (50–100k) is the acceptable substitute if
  the trend is ever wanted — labelled as a subset, never reported as `revisitop1m`.
- **`oxfordparis` is broken and is not a dataset we have.** See the loading section
  below for why. There is no correct way to use it; the fix is to evaluate the two
  datasets separately, which is the standard protocol anyway.

Each query's ground truth has `bbx`, `easy`, `hard`, `junk`. The `easy`/`hard`/`junk`
values are **positional indices into the database list**, so the database ordering is
part of the ground truth. Never shuffle, sort, dedupe, or filter the database split.
If you build a subset, carry an explicit index remapping.

### Loading it — read this before writing any data code

The HF dataset [`galilai-group/revisitop`](https://huggingface.co/datasets/galilai-group/revisitop)
contains **only a loading script** (`revisitop.py`) — no parquet, no data. That has
consequences, and the script has confirmed bugs:

1. **Script datasets require `datasets<4.0` plus `trust_remote_code=True`.** Script
   loading was removed in `datasets` 4.x. Pin accordingly, or bypass the hub entirely.
2. **The `revisitop1m` config is broken.** Its split passes `ground_truth_file`
   (singular) while `_generate_examples` expects `ground_truth_files` — it raises
   `TypeError` before yielding anything. Moot, since we do not use this config, but
   recorded so nobody spends an afternoon debugging it.
3. **The `oxfordparis` config must never be used for evaluation.** It concatenates the
   two ground-truth files without offsetting Paris's indices past Oxford's 4993 images,
   so Paris queries' `easy`/`hard`/`junk` point at Oxford database images. It will
   produce numbers, and they will be meaningless. This is the reason it is marked
   unused above — it is not a preference, the config is simply wrong.
4. **Missing files silently shift every ground-truth index.** The generator skips
   images not found on disk (`if name in image_path_mapping`). One absent or corrupt
   file shortens the database list, shifts all later positions, and every subsequent
   index in the ground truth becomes wrong — with no error raised.

Because of (4), the loader **must assert the exact database and query counts from the
table above** before anything downstream runs. This assertion is not optional and
should not be relaxed to make a run proceed.

Given all of the above, prefer fetching the upstream sources directly — the VGG image
archives and the `gnd_*.pkl` files from `cmp.felk.cvut.cz/revisitop` — and parsing the
pickles ourselves. It is less code than working around the script, and it is the same
data the reference implementation uses. Treat the HF dataset as documentation of the
format rather than as the ingestion path.

**`revisitop1m` and `oxfordparis` are restricted at the download entry points, not just
discouraged in prose.** `cbir.data.revisitop.download()` raises `ValueError` for both
before touching the network, and `cbir` CLI's `download` subcommand doesn't accept them
as arguments at all — `datasets` is typed `Literal["roxford5k", "rparis6k", "all"]`, so
tyro rejects the value at argument-parsing time. If either config needs to become
usable in the future (e.g. `revisitop1m` fixed and taken on deliberately), that means
changing `SUPPORTED`/`UNSUPPORTED_CONFIGS` in `src/cbir/data/revisitop.py` and the
`Literal` in `src/cbir/cli.py` together — not routing around them with a raw
`load_dataset()` call elsewhere.

## Evaluation protocol

Three protocols per dataset, differing only in which images count as positive and
which are ignored (ignored images are removed from the ranking before scoring, they
are *not* negatives):

| protocol | positives    | ignored          |
|----------|--------------|------------------|
| Easy     | `easy`       | `hard` + `junk`  |
| Medium   | `easy`+`hard`| `junk`           |
| Hard     | `hard`       | `easy` + `junk`  |

Metrics: **mAP** and **mP@10**, averaged over queries. Queries with no positives under
a given protocol are excluded from that protocol's average — 2 of the 70 ROxford
queries have no `easy` positives, so Easy-protocol ROxford averages over 68.

Other protocol details that change the numbers:
- **Queries are cropped to `bbx`** before descriptor extraction — always, for every
  method, classic and neural alike. The benchmark does not treat this as a tunable
  setting: *"Only the cropped regions are to be used as queries; never the full image,
  since the ground-truth labeling strictly considers only the visual content inside the
  query region."* (revisitop §2.3). The annotations were made against the crop, so a
  full-image query is not a comparable-but-different number — it is scored against
  labels that do not describe it. A query image containing two landmarks, cropped to
  one, has ground truth for that one only; the full image can correctly retrieve the
  other and be counted wrong. Do not "fix" this by relaxing the crop.
  If a production-realistic uncropped setting is ever wanted, it is a **separate,
  explicitly-labelled ablation** that never enters the headline comparison and is never
  presented next to published mAP.
  `bbx` is `(x1, y1, x2, y2)` and **zero-based**, which is exactly PIL's
  `Image.crop((left, upper, right, lower))` — pass it straight through, no `+1`. The
  reference MATLAB code adds 1 because MATLAB indexes from 1; that adjustment is a
  MATLAB artifact and must not be ported to Python.
- Query images are not part of the database and must not be retrieved.
- Report Medium and Hard at minimum; Medium is the headline number in the literature.

When results look surprising, suspect the harness first. Validate against published
numbers for a known descriptor before trusting a new one.

## Methods in scope

Three eras, evaluated under the same protocol:

- **Classic**: RootSIFT + k-means vocabulary → BoW with tf-idf and inverted index;
  VLAD. Vocabulary trained on a held-out set, *never* on the evaluation database.
- **CNN era**: activations of a frozen CNN as global descriptors — Neural Codes, SPoC,
  MAC, R-MAC, GeM. See below; this tier is the bridge between the other two.
- **Modern**: frozen foundation-model global descriptors (DINOv2/DINOv3, CLIP, SigLIP),
  L2-normalized, cosine similarity. Start frozen; fine-tuning is a later step and must
  be justified by a baseline it beats.

Out of scope for now (do not add unprompted): local-feature re-ranking, geometric
verification, query expansion / diffusion. These are natural next steps but each one
changes the comparison, so they land as deliberate, separately-evaluated additions.

### CNN-era descriptors

| method       | paper | descriptor |
|--------------|-------|------------|
| Neural Codes | Babenko et al., ECCV 2014 ([1404.1777](https://arxiv.org/abs/1404.1777)) | activations of the first fully-connected layer |
| SPoC         | Babenko & Lempitsky, ICCV 2015 ([1510.07493](https://arxiv.org/abs/1510.07493)) | sum-pooled conv map, optional centering prior |
| MAC, R-MAC   | Tolias et al., ICLR 2016 ([1511.05879](https://arxiv.org/abs/1511.05879)) | max-pooled conv map; R-MAC instead max-pools per region over a multi-scale grid, whitens each region, and sums |
| siaMAC       | Radenović et al., ECCV 2016 ([1604.02426](https://arxiv.org/abs/1604.02426)) | MAC, but the backbone is fine-tuned and the whitening is learned |
| GeM          | Radenović et al., TPAMI 2018 ([1711.02512](https://arxiv.org/abs/1711.02512)) | generalized mean over the conv map, exponent `p` |

**Neural Codes** is the diploma-era method and the historical starting point: take the
4096-D output of the first fully-connected layer (`fc6` in AlexNet/VGG naming) of a
frozen ImageNet-trained CNN, L2-normalize, PCA-compress, L2-normalize again, compare
with Euclidean distance. Fit the PCA on a **held-out set, never on the evaluation
database**. It has no pooling stage at all — the FC layer's fixed input size is what
forces images to a single resolution, and removing that constraint is precisely what
the later methods are for.

Everything else in the table takes the last conv feature map, treats each spatial
position as a local descriptor, pools them into one vector, whitens, and L2-normalizes.
The pooling is the only difference.

Generalized-mean pooling is `f_k = (mean_{x in X_k} x^p)^(1/p)`. At `p = 1` this *is*
SPoC's sum pooling; as `p → ∞` it is MAC. So implement **one** parameterized pooling
module and expose `p` as a config knob — SPoC and MAC are settings of it, not separate
descriptor classes. Only R-MAC needs its own code, for the region grid.

SPoC's contribution is a negative result worth reproducing: Fisher/VLAD-style
aggregation, which wins for SIFT, *underperforms* plain summation on deep features
because their pairwise-similarity distributions differ. That result is what connects
the classic VLAD baseline to this tier — reproduce it rather than asserting it.

**siaMAC and GeM are one line of work, not two.** The ECCV 2016 paper is the
conference version; the TPAMI 2018 paper is its journal extension (arXiv flags the
substantial text overlap). Treat them as v1 and v2 of the same system, and do not
report them as independent methods:

| | ECCV 2016 (siaMAC) | TPAMI 2018 (GeM) |
|---|---|---|
| pooling | MAC | GeM, with learnable `p` |
| training data | ~30k images from SfM 3D models | retrieval-SfM-120k, also google-landmarks-2018 |
| added later | — | α-weighted query expansion, multi-scale at test time |

What ECCV 2016 contributes on its own, and what to take from it:

- **The training signal comes from a BoW retrieval system.** That is the title's
  claim, and it matters here specifically: the supervision is produced by BoW with
  Hessian-affine + RootSIFT, a 16M-word vocabulary, and spatial verification — the
  same classic pipeline this project implements in `descriptors/classic/`. SfM 3D
  reconstruction over those matches then yields hard positives (via camera geometry
  and co-observed 3D points) and hard negatives (from different 3D models), with no
  manual annotation anywhere. The classic tier is not only the baseline being measured
  against; historically it is what *taught* the CNN tier.
- **Learned whitening.** This is where discriminative whitening enters the line of
  work, not the TPAMI paper. Instead of PCA-whitening on an independent set, it uses
  linear discriminant projections: whiten by the inverse square root of the
  *intra*-class (matching-pair) covariance, then rotate by the PCA of the *inter*-class
  (non-matching-pair) covariance in that whitened space, keep the top `D` eigenvectors,
  L2-normalize. It needs matching labels, which the 3D models supply for free.
- Architecture: siamese, two shared-weight branches, contrastive loss on matching and
  non-matching pairs. FC layers are discarded entirely — the network is fully
  convolutional, which is what makes variable input resolution possible.

Implementation details that change the numbers:

- **Whitening is not optional.** SPoC/MAC/R-MAC numbers assume PCA-whitening; without
  it they lose several mAP points. Learn it on a **held-out set, never on the
  evaluation database** — same rule as the k-means vocabulary. The siaMAC/GeM released
  checkpoints have *learned* (discriminative) whitening baked in; do not PCA them again
  on top.
- **Multi-scale matters.** Published GeM numbers are multi-scale (typically scales
  `1, 1/√2, 1/2`, GeM-pooled across scales). Single-scale lands a few points lower.
  Make it a config flag so a deviation is labelled, not silent.
- Backbones are ImageNet-supervised VGG16 (512-D) and ResNet50/101/152 (2048-D).
  Document `D` per model as usual.

**GeM is the harness-validation descriptor.** Radenović et al. wrote both GeM and
`revisitop`, so published numbers exist for exactly the protocol implemented here.
SPoC's original numbers are on the *old* Oxford5k/Paris6k and cannot validate this
harness. Pretrained weights come from `cmp.felk.cvut.cz` — same host as the `gnd_*.pkl`
files, so fetch them the same way rather than through a hub loader. Reference numbers
(multi-scale, whitened, from `filipradenovic/cnnimageretrieval-pytorch`):

| model                        | ROxf (M) | RPar (M) | ROxf (H) | RPar (H) |
|------------------------------|----------|----------|----------|----------|
| rSfM120k-tl-resnet50-gem-w   | 64.7     | 76.3     | 39.0     | 54.9     |
| rSfM120k-tl-resnet101-gem-w  | 67.8     | 77.6     | 41.7     | 56.3     |
| rSfM120k-tl-resnet152-gem-w  | 68.8     | 78.0     | 41.3     | 57.2     |
| gl18-tl-resnet50-gem-w       | 63.6     | 78.0     | 40.9     | 57.5     |
| gl18-tl-resnet101-gem-w      | 67.3     | 80.6     | 44.3     | 61.5     |
| gl18-tl-resnet152-gem-w      | 68.7     | 79.7     | 44.2     | 60.3     |

Landing within ~0.5 mAP of the corresponding row is evidence the harness is correct.
Missing by more than that is a harness bug until proven otherwise — do not tune the
descriptor to close the gap.

These are all **frozen off-the-shelf checkpoints**, so they stay inside the current
scope — including siaMAC and GeM, whose weights happen to have been produced by
fine-tuning that someone else already ran. Running that fine-tuning here is a different
matter and needs its own justification.

## Search backend

- `roxford5k` / `rparis6k` are small. Use **exact** similarity via a single torch
  matmul on GPU. It is fast and introduces no approximation error into the benchmark.
- Since `revisitop1m` is out of scope, **exact search covers the whole benchmark** and
  there is currently no ANN in this project. Do not add FAISS, HNSW, IVF-PQ, or any
  approximate index — at 5–6k database vectors it would be slower than the matmul and
  would inject approximation error into the only numbers that matter.
- If `revisitop1m` is ever taken on, FAISS becomes necessary and the rule is: report
  the recall loss against exact search on the small sets first. An ANN-induced mAP drop
  must never be reported as a descriptor result.

## Configuration, tracking, results

- **tyro** for configuration — the same library the CLI already uses for `cbir download`.
  Every knob that affects a number lives in a typed dataclass in `src/cbir/configs/`,
  not in a bare Python default or a notebook cell, so it shows up in `--help` and in
  the run's recorded config.
- **Weights & Biases** for run tracking. Requires `WANDB_API_KEY` in the environment;
  never commit it. Support `WANDB_MODE=offline` and make sure the pipeline runs
  end-to-end without a W&B account — tracking is an observer, not a dependency.
- Every evaluation also appends a row to **`results/runs.jsonl`** containing the metrics,
  the resolved config, and the git commit. One JSON object per line, append-only: a run
  never rewrites an earlier row, so diffs are always new lines, two runs cannot conflict,
  and re-running a configuration records a *second* row rather than overwriting the
  first. That history is the point — when a number moves, the pair of rows and their
  commit hashes say when it moved and what changed, which a file holding only current
  values cannot answer. These are committed. W&B is convenience; `results/` is the record.
- Descriptor extraction is the expensive step. Cache descriptors to `data/` keyed by
  (dataset, descriptor config), and make the cache key include everything that changes
  the output. A stale cache silently invalidates results.

## Conventions

- Type hints on public functions. Format and lint with **ruff**.
- Descriptors are `float32`, L2-normalized, shape `(N, D)`, row order matching the
  database list order. Document `D` per model.
- Determinism: seed everything seedable and record the seed in the run output.
- Tests: the evaluation code gets real unit tests with hand-computed expected values —
  a known ranking with a known mAP. This is the part that must not be silently wrong.
- Never commit images, descriptors, or indexes. `data/` stays gitignored.
- Don't reference `AGENTS.md` by name in docstrings — a docstring should explain the
  code on its own terms, not point at a governance file that can move or change
  wording. Put "why" context directly in the docstring instead.

## Working agreements

- Report numbers as measured. If a result is worse than the baseline, or a run failed,
  say so plainly — the comparison is the deliverable, and a flattering number that
  cannot be reproduced destroys the project's purpose.
- Do not tune on the evaluation set or select protocols after seeing results.
- Long extractions belong in background runs, not in blocking foreground calls.
