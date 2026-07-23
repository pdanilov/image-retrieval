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
