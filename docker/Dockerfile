# syntax=docker/dockerfile:1

# A slim Python base, not nvidia/cuda: the cu128 torch wheels bundle their own CUDA
# runtime (cuBLAS, cuDNN and the rest arrive as nvidia-* pip packages), so the only thing
# needed from the host is the driver, which `--gpus` provides. A CUDA base image would
# add several GB of libraries that torch then shadows.
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# libglib2.0-0 for opencv-python-headless (headless still links it); git because
# `results/runs.jsonl` records the commit a run was produced at, and a row recorded
# without one is one nobody can trace back.
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
        git libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Build-time only. UV_CACHE_DIR is pinned rather than left to derive from $HOME: the
# cache mount below targets this exact path, and a later ENV that changes HOME would
# silently move uv's lookup and re-download 3 GB of torch. That cost 47 minutes once.
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_CACHE_DIR=/root/.cache/uv

# Dependencies first, in their own layer: they are ~3 GB and change far less often than
# the source, so editing a module does not re-resolve torch.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --extra tracking

COPY docker/healthcheck.sh ./docker/healthcheck.sh
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --extra tracking

# Runtime environment, set *after* the dependency layers so changing any of it rebuilds
# only the tail of the image.
#   PYTORCH_CUDA_ALLOC_CONF: variable image sizes fragment the caching allocator;
#     expandable segments let it reuse blocks across shapes instead of one per shape.
#   CBIR_*_ROOT: the corpus and weights are bind-mounted, never baked in -- ~77 GB.
#   HOME: the container runs as the host uid, which has no home directory here, so
#     anything that caches to $HOME (torch hub, matplotlib) needs somewhere writable.
#   TRACKIO_DIR: trackio defaults under $HOME, which is /tmp here and so would vanish
#     with the container. Pointed at a mount instead -- and deliberately *not* into the
#     repo, which AGENTS.md rules out: a SQLite file does not diff usefully in git.
#   USER: that uid has no /etc/passwd entry either, and torch's inductor calls
#     getpass.getuser() at *import* time to name its cache directory. Without this,
#     `import torchvision` raises before any of our code runs. getuser() reads USER
#     before it consults the password database, so naming it here is enough.
ENV PATH="/app/.venv/bin:${PATH}" \
    USER=cbir \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CBIR_SFM_ROOT=/cache/cbir/sfm120k \
    CBIR_WEIGHTS_ROOT=/cache/cbir/weights \
    HOME=/tmp \
    TRACKIO_DIR=/cache/trackio

ENTRYPOINT ["cbir"]
CMD ["--help"]
