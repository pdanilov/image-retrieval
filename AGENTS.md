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
    transformer/ frozen ViTs, one image embedding each (DINOv2, DINOv3, SigLIP 2)
  training/      fine-tuning for retrieval on retrieval-SfM-120k: the trainable half of
                 each neural tier (CNN today; the transformer tier's lands here too)
  search/        exact torch matmul search (no ANN — see Search backend)
  eval/          mAP / mP@k, the three protocols, the run pipeline, results serialization
  cli.py         entry point: download / evaluate / results
notebooks/       analysis and figures ONLY — no pipeline logic
scripts/         one-off diagnostics outside the harness — never write to results/
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
- **Transformer era**: vision transformers as global descriptors, split like the CNN
  tier into a **frozen** half (DINOv2, DINOv3, SigLIP 2, each read out by its own image
  embedding) and a
  **trainable** half (IRT, then DINOv2 fine-tuned in full and with LoRA), L2-normalized, cosine similarity.
  The frozen half comes first; the trainable half must beat it to earn its place. See
  below.

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
files — and that host stays the public source for them (see Weight files below for the
private mirror the loaders fall back to). Reference numbers
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

### Transformer-era descriptors

Split the same way as the CNN tier, and for the same reason: a network trained for
something else, read out as-is, is a different experiment from one trained here for
retrieval. Frozen extractors live in `descriptors/transformer/`; training reuses
`training/`.

**What changed from the CNN era is the pretraining, not the readout.** DINOv2 is
self-distilled on 142M curated images — an objective that already pulls views of the
same content together, which is roughly what the CNN tier had to *fine-tune* ImageNet
classifiers to do. Frozen DINOv2 ViT-L/14 is published at 75.1 / 54.0 roxford5k M / H,
against 67.8 / 41.7 for the fine-tuned GeM ResNet101. Expect the frozen half to beat
the whole CNN tier, trained arms included; the open question for the trainable half is
whether retrieval training still adds anything on top.

#### Frozen half

| model | role | weights |
|-------|------|---------|
| DINOv2 ViT-S/B/L/g-14 (Oquab et al., TMLR 2024, [2304.07193](https://arxiv.org/abs/2304.07193)) | the anchor of the frozen half | Apache-2.0, ungated |
| DINOv3 ViT-L/16 (Siméoni et al., 2025, [2508.10104](https://arxiv.org/abs/2508.10104)) | the same objective scaled up and distilled; the tier's best, and its one reproduced published row | **gated** on HF — accept the license first |
| SigLIP 2 (image-text) | the vision-language contrast: semantic, not instance-level. No CLS token — read out by its attention-pooling head; one checkpoint per resolution, each taking only its native size | Apache-2.0 |
| DeiT-S†/B† (distilled, ImageNet) | the trainable half's starting point, scored before training. Frozen it is the project's weakest row (0.301 / 0.085 roxford, B† 384) — the baseline IRT has to lift, not a contender. Read CLS from `last_hidden_state`: its `pooler_output` is an untrained head | Apache-2.0 |

Rules that change the numbers:

- **The readout is the CLS token after the final LayerNorm** — DINOv2's
  `x_norm_clstoken`, Hugging Face's `pooler_output`. Every published instance-retrieval
  row is measured on it. Mean or GeM pooling over patch tokens is an *ablation* to add
  on top of the CLS rows, not the default. GeM on patch tokens has a trap: the tokens are
  signed (LayerNorm output, no ReLU), and GeM's clamp to `eps` discards every negative
  component, so `p = 1` (mean) is the honest first comparison.
- **Protocol (DINOv3 App. D.8, the only one either paper documents):** resize so the
  *longer* side is 224, keeping the aspect ratio; centre on a black 224 x 224 square
  ("full center crop" of an image already 224 long can only pad); ImageNet mean/std;
  cosine on the raw CLS token; no PCA, no whitening. The paper names no interpolation —
  bicubic is used here because DINOv2's own evaluation transforms use it. Queries are cropped to
  `bbx` first, as for every method. Any other resolution is a labelled ablation.
- **Plain DINOv2, not the registers variant**, for the rows compared to published
  numbers. Registers are an ablation afterwards.
- `image_size` must be a multiple of the patch size (14 for DINOv2, 16 for DINOv3): the
  patch embedding silently drops the remainder rather than raising.
- **Do not whiten DINO by default.** Measured, not assumed: PCA-whitening costs DINOv3-L
  0.020 / 0.026 roxford M / H fitted on the held-out set and 0.032 / 0.044 on sfm30k (more
  data, bigger loss — not estimation noise). It *helps* SigLIP 2 by up to 0.107. So the
  cross-tier table compares each tier at its own best post-processing, and says which.
- **CLS beats pooled patch tokens by ~0.33 roxford Medium on DINO**, even unpadded.
  Pooled readouts (`mean`, `gem`) must be measured under `resize="aspect"`: under `pad`
  the black bars are patches and get averaged in, which alone costs them 0.10–0.15. CLS
  is insensitive to the bars. DeiT and SigLIP 2 cannot run unpadded (fixed position
  embeddings), so their pooled rows are confounded and not conclusions.

Published reference rows (mAP, Medium / Hard), all frozen CLS — read the note below
before comparing to them:

| model | ROxf M | ROxf H | RPar M | RPar H | source |
|-------|--------|--------|--------|--------|--------|
| DINOv2 ViT-S/14 | 68.8 | 43.2 | 84.6 | 68.5 | DINOv2 Table 9 |
| DINOv2 ViT-B/14 | 72.9 | 49.5 | 90.3 | 78.6 | DINOv2 Table 9 |
| DINOv2 ViT-L/14 | 75.1 | 54.0 | 92.7 | 83.5 | DINOv2 Table 9 |
| DINOv2 ViT-g/14 | 73.6 | 52.3 | 92.1 | 82.6 | DINOv2 Table 9 |
| DINOv2 ViT-g/14 | 78.2 | 58.2 | 92.7 | 84.6 | DINOv3 Table 23 |
| DINOv2 ViT-L/14 | — | 55.7 | — | — | DINOv3 distilled-model table |
| DINOv3 ViT-L/16 | — | 63.1 | — | — | DINOv3 distilled-model table |
| SigLIP 2 g/16 | 49.3 | 25.1 | 79.3 | 60.9 | DINOv3 Table 23 |

**DINOv2's Table 9 was measured on uncropped queries, so it is not a target for this
harness.** Investigated on this branch, not assumed: under the protocol above, ViT-S and
ViT-B missed the table by 8-11 mAP on roxford (ViT-L and ViT-g by 1-4). Padding,
interpolation and DINOv2's own centre-crop transform each moved ViT-S by at most 4 points
and never toward the table. What closed the gap was facebookresearch/dino's public
`eval_image_retrieval.py`, which the same group wrote: it loads the query images whole,
with no `bbx` crop at all, and `thumbnail`s everything to 224. Replicating that script
outside the harness reproduces Table 9 within 0.1-1.6 mAP for S, B and L on both
datasets. So:

- Rows here are cropped, as the benchmark requires, and are compared to Table 9 only
  with that difference stated. The gap is a protocol difference, not a harness bug.
- Do not "fix" it by uncropping queries. The uncropped numbers are the ones the
  benchmark rules out (see Evaluation protocol). The diagnostic that measured them is
  `scripts/dinov2_uncropped_queries.py` — outside the package, unreachable from the CLI,
  and it writes nothing to `results/runs.jsonl`.
- The crop costs small models far more than large ones. With DINO's preprocessing held
  fixed, cropping ViT-S's queries costs 0.137 roxford Medium (0.687 → 0.550); between
  DINO's script and our protocol ViT-L moves only 0.009. Context outside the box is what
  a weak descriptor leans on — a finding worth keeping, not noise.
- **The DINOv3 paper's own ViT-L/16 row does reproduce, cropped:** 63.1 roxford Hard
  published, 0.637 here at the 224 protocol. That is the transformer tier's validation
  row. Its *other* rows are looser: its DINOv2-L (55.7 Hard) sits 2.9 above ours, its
  DINOv2-g (78.2 / 58.2) far above our cropped ViT-g (69.7 / 49.4), and its SigLIP 2-L
  (21.4 Hard) about 7 above ours. The paper does not say how it fed models other than
  its own, so those gaps are recorded, not attributed.

GeM stays the harness validator; DINOv3-L/16 at 224 is the transformer tier's. Beyond
that the checks are internal: the ranking across model sizes (DINOv2 L above g on
roxford, as in its paper) and the cropped/uncropped split above.

#### Trainable half

**IRT** (El-Nouby et al., 2021, [2102.05644](https://arxiv.org/abs/2102.05644)) is the
GeM recipe with the backbone swapped: DeiT instead of ResNet, CLS instead of GeM, the
same retrieval-SfM-120k tuples and the same contrastive loss — the authors ran
`cnnimageretrieval-pytorch` itself. Recipe (IRT §4.2, checked against the paper's text):
tuples of 1 anchor, 1 positive, 5 negatives, 5 tuples per batch; 2000 pairs and a
22000-image negative pool per epoch; Adam, lr 5e-7, weight decay 1e-6; margin 0.85; 100
epochs. IRT_R adds a differential-entropy (KoLeo) term, `L = L_contr + λ L_KoLeo` with
`L_KoLeo = -(1/N) Σ_i log min_{j≠i} ‖z_i − z_j‖`, which spreads descriptors over the
sphere.

What the paper says, and does not say, that changes the numbers:

- **Inputs are stretched, not padded.** Training *and* evaluation images, queries and
  database alike, are "resized into a square image with the same resolution as used
  during the finetuning stage" — aspect ratio discarded. That is neither of the frozen
  tier's `pad`/`aspect`/`crop`, so IRT rows need their own `stretch` mode, and the frozen
  DeiT rows they are compared against must be re-scored under it. Queries are still
  cropped to `bbx` first; the reference code does that too.
- **PCA, not stated as whitening.** "The dimensionality reduction is achieved using a PCA
  trained on the SFM120k dataset." Whether the full-width rows were whitened is not said.
- **λ for the retrieval rows is not stated.** The 0.7 in §4.2 belongs to the
  category-level benchmarks; Table 6 sweeps λ for DeiT-S without saying which value the
  Table 8 row used. Choose λ on the SfM validation split, never on Oxford/Paris.

| IRT row (all trained the same way) | dims | ROxf M | ROxf H | RPar M | RPar H |
|------------------------------------|------|--------|--------|--------|--------|
| DeiT-S† 224², CLS, contrastive (Table 7) | 384 | 34.5 | 15.8 | 65.8 | 42.0 |
| DeiT-B† 224², CLS, contrastive (Table 7) | 768 | 39.5 | 17.4 | 67.5 | 43.6 |
| DeiT-B† 384², CLS, contrastive | 768 | 50.5 | 22.7 | 70.6 | 47.4 |
| DeiT-B† 384², CLS, + entropy reg (IRT_R) | 768 | 55.1 | 28.3 | 72.7 | 49.6 |
| ResNet101 GeM 384², same recipe | 2048 | 38.1 | 12.5 | 69.4 | 45.8 |

The DeiT-S† 224 row is the cheap first target: it exercises the whole pipeline at a
fraction of the B† 384 cost.

**No IRT checkpoints were published**, so unlike the CNN tier the target is the final
mAP alone — there are no weights or descriptors to compare against. Note also that IRT
beats ResNet101 GeM at the *same* 384 input but not the GeM evaluated at 1024; report
that rather than the flattering comparison.

What `training/` needs to host it, generalized in place rather than duplicated:

- `net.py`: a ViT + CLS head beside CNN + GeM, built from the frozen tier's `SPECS`, with
  CLS read from `last_hidden_state` (DeiT's `pooler` is untrained).
- `tuples.py`: square `stretch` resizing (the CNN recipe keeps aspect ratio at 362).
  Square inputs stack, so a tuple's seven images — and mining and validation — can run
  as batches rather than one forward per image.
- `loss.py`: the ViT margin and the optional KoLeo term, pinned by a hand-computed test.
- `loop.py`: no BatchNorm freezing for ViTs (they use LayerNorm); `p` logged only for
  GeM. Run the full 100 epochs: early stopping already cut ResNet101 short at 19.
- Evaluation: a `checkpoint` field on `ViTConfig`, so a trained network is scored by the
  same path as a frozen one; PCA fitted on `sfm120k`, which the frozen tier already has.

Order of work, cheapest first:

1. **Done.** `stretch` in the harness; frozen DeiT-S†/B† re-scored under it at 224 and
   384 — the "before" row in the protocol the "after" row will use.
2. **Done: IRT reproduces.** IRT_L trained here, 100 epochs, raw CLS: DeiT-S† 224 0.352 /
   0.166 roxford, 0.666 / 0.424 rparis (Table 7: 34.5 / 15.8, 65.8 / 42.0); DeiT-B† 224
   0.422 / 0.207, 0.674 / 0.430 (Table 7: 39.5 / 17.4, 67.5 / 43.6 — roxford ~3 above,
   one seed, recorded as a gap). PCA-whitening on SfM lowers the trained rows. **B† 384
   and IRT_R are deliberately not run**: both are DeiT, the best IRT row sits ~25 Medium
   below frozen DINOv3-L, and the two rows above already settle the reproduction.
3. The same recipe on DINOv2-S/B (full fine-tuning). It is justified only if it beats
   the frozen DINOv2 row at the same protocol (`stretch`, re-measured) — if it does not,
   that is the finding. `scripts/sfm_validation.py` gives each frozen model's SfM
   validation mAP, i.e. how much of the training objective it already solves: DINOv2-S
   0.858, B 0.887, L 0.905, DINOv3-L 0.928 — all above trained DeiT-B† (0.824). Frozen
   `stretch` rows: DINOv2-S 0.596 / 0.387 roxford, 0.827 / 0.662 rparis; B 0.648 /
   0.424, 0.888 / 0.759. **Done — it does not beat them.** Trained 100 epochs, best SfM
   epoch kept: S 0.575 / 0.375, 0.793 / 0.599 (−0.021 … −0.064); B 0.640 / 0.405, 0.874 /
   0.742 (−0.008 … −0.019). SfM validation peaks by epoch ~40 and then slides.
4. **LoRA** on the same DINOv2-S/B, then on DINOv3-L, the tier's best frozen row. The
   question it answers is whether retrieval training adds anything *without* overwriting
   what 142M-image pretraining learned, and it is what makes ViT-L trainable at all on
   16 GB. Rules: never for the IRT reproduction (IRT fine-tunes every weight, and a
   different method cannot reproduce its number); the recipe's lr 5e-7 is a full
   fine-tuning value, so lr, rank and target modules are new knobs, recorded in the run
   config and chosen on SfM validation; adapters are merged into the weights after
   training (`peft`'s `merge_and_unload`), so the checkpoint is a plain model and the
   evaluation path does not change. Implemented: `cbir train --lora-rank R --lr X`
   (adapters on q/k/v/o; `lr` is required, there is no published value). **Done — it
   does not beat frozen either.** DINOv2-S, rank 8, lr 1e-5 / 3e-5 / 1e-4: the higher the
   SfM validation, the lower rparis6k (Hard 0.643 / 0.605 / 0.585 vs frozen 0.662), so
   SfM validation selects the worst setting. DINOv3-L, rank 8, lr 1e-5 (not SfM-selected):
   SfM peaks at epoch 2 and falls after; benchmark +0.002–0.005, noise floor.

**Status: the trainable half is closed with a negative result** — retrieval training on
SfM-120k adds nothing to DINO. Reopening it needs a different training signal (data or
objective), not another schedule on this one.

Whether full fine-tuning of ViT-L fits 16 GB is unmeasured — about 5 GB of weights,
gradients and Adam state in fp32 plus activations, so possibly with bf16. Measure before
ruling it in or out.

## Weight files

Checkpoints that are not on the Hugging Face Hub under their authors' names — the
Caffe-converted ImageNet weights, Radenović et al.'s fine-tuned GeM checkpoints, and every
checkpoint `cbir train` produced — are kept as `.safetensors` in **private** model
repositories under `pdanilov`, one per network:

| repositories | contents |
|---|---|
| `imagenet-caffe-{vgg16,resnet50,resnet101,resnet152}` | mirrors of the reference's Caffe ImageNet weights |
| `retrieval-sfm120k-{vgg16,resnet101}-gem` | mirrors of the published GeM checkpoints |
| `gem-{vgg16,resnet101}-sfm120k-reproduced` | the GeM recipe trained here |
| `irt-deit-{small,base}-distilled-224` | IRT trained here |
| `dinov2-sfm120k-finetune` | the negative DINOv2 runs, full and LoRA |

DINOv3 fine-tunes are not mirrored (gated license, noise-floor result).

- **Resolution order** (`cnn/weights.py::locate`): a local `.pth` from the original host
  under `CBIR_WEIGHTS_ROOT` first, else the mirror. Trained checkpoints are addressed as
  `--checkpoint mirror:<repository>/<file>.safetensors`; that string is what the run row
  records, and with the pinned hash it fixes the content in a way a local path does not
  (`data/runs/.../best.pth` was overwritten between rows more than once).
- **Every download is checked** against `descriptors/mirror_sha256.json`. A mismatch is an
  error, never a fallback. An unreachable repository (no token, no access, offline) is
  reported as such, with the original URL — so the code still works for anyone without
  access, by the manual download.
- **Access** needs a token that can read the owner's private repositories: `hf auth login`
  locally, `HF_TOKEN` in CI. A token scoped only to gated models will not see them.
- **The format** (`descriptors/serialization.py`) round-trips the `.pth` exactly, whitening
  dtypes included, and is deterministic: re-running `scripts/export_safetensors.py` on the
  originals reproduces the pinned hashes. That script is the only way files enter the
  mirror — it verifies each conversion bit for bit and rewrites the manifest. Re-scoring six
  recorded rows from the converted files reproduced them to the last digit.
- **Do not link the private repositories from the README.** Readers cannot open them; the
  README's public path is the original host.

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
- **trackio** for run tracking, as an optional dependency. Import it defensively
  (`try: import trackio / except ImportError: return`) so a machine without it still
  evaluates — tracking is an observer, not a dependency. No account, key, or network is
  needed: it writes one SQLite file per project and is queryable from the terminal
  (`trackio list runs`, `trackio query project --sql ...`, `trackio show` for a
  dashboard, `trackio sync` to publish to a Hugging Face Space if ever wanted).
- Leave trackio's storage at its default `~/.cache/huggingface/trackio/`, for the same
  reason the benchmark archives live in the user-wide HF cache: it is local scratch, and
  a SQLite binary would not diff usefully in git. `TRACKIO_DIR` can relocate it, but
  do not point it into this repo.
- Not W&B, which an earlier version of this file specified. It needs an account and a
  secret, and its comparison story is the web UI, while this project is evaluation-only
  — every run is a dozen terminal-state scalars, never a training curve. Not DVCLive
  either: its experiments are git refs scoped to the commit they ran from, which plain
  `git push` silently drops and which vanish from `dvc exp show` after the next commit.
  Revisit if fine-tuning ever enters scope, since that is when curves start to matter.
- Every evaluation also appends a row to **`results/runs.jsonl`** containing the metrics,
  the resolved config, and the git commit. One JSON object per line, append-only: a run
  never rewrites an earlier row, so diffs are always new lines, two runs cannot conflict,
  and re-running a configuration records a *second* row rather than overwriting the
  first. That history is the point — when a number moves, the pair of rows and their
  commit hashes say when it moved and what changed, which a file holding only current
  values cannot answer. These are committed, and `cbir results` reads them back.
  trackio is convenience; `results/` is the record.
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
