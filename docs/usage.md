# Usage

Part of [image-retrieval](../README.md). Installing, fetching the benchmark, and running and reading back evaluations.

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
[AGENTS.md](../AGENTS.md) for details.

## Evaluating

```
uv run cbir evaluate --dataset roxford5k bow --k 20000
uv run cbir evaluate --dataset roxford5k vlad --k 256
uv run cbir evaluate --dataset roxford5k fisher --k 64

# a sweep is one run per k, sharing a single descriptor extraction
uv run cbir evaluate --dataset roxford5k --sweep 16 64 256 vlad --seed 0
```

### Presets

The interesting configurations carry up to ten flags, and the fine-tuned ones only work
in one combination. Every row the docs quote is available by name:

```
uv run cbir evaluate gem-ft-r101
uv run cbir evaluate gem-ft-r101 --scales 1.0   # a preset is a starting point
uv run cbir evaluate --help                     # lists techniques and presets together
```

A preset is a subcommand like any technique, not a `--preset` flag: a technique's knobs
are defined *inside* its subcommand, so a flag would have nowhere to put `--scales` and
would leave you with a configuration you cannot adjust.

They live in `configs/presets.py` as config *instances*, not YAML — so each one is
type-checked at import, validates on construction, and an override re-validates:
`--p 3.0` on a fine-tuned preset is rejected rather than quietly evaluating the network
at a pooling it was never trained for. `tests/configs/test_presets.py` holds every preset
to a row already in `results/runs.jsonl`, so a preset cannot drift from the configuration
that was actually measured.

## Weights

The ViT models and torchvision's ImageNet weights download themselves from the
Hugging Face Hub and PyTorch. Two sets of files do not: they come from the reference
implementation's host and go into `~/.cache/cbir/weights/` (or wherever `CBIR_WEIGHTS_ROOT`
points), under the names below.

Caffe-converted ImageNet weights, which the reference's numbers were reached from — needed
for every `--weights caffe` row and for `cbir train`:

| backbone | file |
|---|---|
| vgg16 | [`imagenet-caffe-vgg16-features-d369c8e.pth`](https://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-vgg16-features-d369c8e.pth) |
| resnet50 | [`imagenet-caffe-resnet50-features-ac468af.pth`](https://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-resnet50-features-ac468af.pth) |
| resnet101 | [`imagenet-caffe-resnet101-features-10a101d.pth`](https://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-resnet101-features-10a101d.pth) |
| resnet152 | [`imagenet-caffe-resnet152-features-1011020.pth`](https://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-resnet152-features-1011020.pth) |

Radenović et al.'s fine-tuned GeM checkpoints — needed for `--weights sfm120k` rows and the
`gem-ft-*` presets:

| backbone | file |
|---|---|
| vgg16 | [`retrievalSfM120k-vgg16-gem-b4dcdc6.pth`](https://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrievalSfM120k-vgg16-gem-b4dcdc6.pth) |
| resnet101 | [`retrievalSfM120k-resnet101-gem-b80fb85.pth`](https://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrievalSfM120k-resnet101-gem-b80fb85.pth) |

Networks trained in this project are not published.

## The SfM-120k corpus

Training (`cbir train`), supervised whitening (`cbir whiten`) and `--whiten-source
sfm30k|sfm120k` need retrieval-SfM-120k, the landmark corpus the reference trains on. It is
not fetched automatically: the image archive alone is 40.5 GB. It goes into
`~/.cache/cbir/sfm120k/` (or `CBIR_SFM_ROOT`):

```
mkdir -p ~/.cache/cbir/sfm120k && cd ~/.cache/cbir/sfm120k
BASE=https://cmp.felk.cvut.cz/cnnimageretrieval/data/train
curl -LO $BASE/dbs/retrieval-SfM-120k.pkl
curl -LO $BASE/dbs/retrieval-SfM-120k-whiten.pkl
curl -LO $BASE/dbs/retrieval-SfM-30k-whiten.pkl
curl -L -C - -O $BASE/ims/ims.tar.gz      # 40.5 GB; re-run to resume
mkdir ims && tar xzf ims.tar.gz -C ims    # the archive has no top-level folder
```

The server has been seen to end a transfer short while `curl` still reports success, so
check the archive's size (40528074099 bytes) before extracting, and resume until it
matches.
