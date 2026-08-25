"""The training loop: mine, train, validate, checkpoint.

Written here rather than taken from `pytorch_metric_learning.trainers`, which does not
fit this recipe on two structural counts. Its trainers assemble batches through a
DataLoader and call the trunk on one stacked tensor, but these images keep their aspect
ratio — `(3, 362, 271)` beside `(3, 204, 362)` — and cannot be stacked without either
padding the pooling with zeros or distorting every training image. And its mining
functions work inside the batch, while the negatives here come from a separate pass over
a 22000-image pool; mining a batch of five tuples would draw from 35 images instead, and
hard negatives are the part of this recipe that makes it work.

PML earns its place in the two spots where it fits the data rather than the loop: the
loss (see `loss.py`) and validation, where `AccuracyCalculator` takes precomputed
embeddings and so never touches a DataLoader.

An epoch is: re-mine negatives against the current model, walk the tuples accumulating
gradients, step every `batch_size` tuples, then measure retrieval on the held-out
landmark split. Mining first is deliberate — negatives mined by a stale model are the
one bug in this recipe that still trains, just worse.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from cbir.descriptors.cnn.pooling import Backbone
from cbir.descriptors.cnn.weights import WeightSource
from cbir.training.loss import MARGIN, contrastive_loss
from cbir.training.net import RetrievalNet
from cbir.training.tuples import (
    IMAGE_SIZE,
    NEG_NUM,
    POOL_SIZE,
    QUERY_SIZE,
    Corpus,
    load_tensor,
    sample_epoch,
)

PROJECT = "cbir-train"
"""Kept apart from the `cbir` project that holds evaluation runs: these are curves over
epochs, those are one terminal row per configuration, and mixing them makes both harder
to read. AGENTS.md anticipated exactly this split when fine-tuning entered scope."""

VAL_METRICS = ("precision_at_1", "mean_average_precision")
"""What validation reports. `mean_average_precision` is the one to watch — it is the same
quantity the benchmark reports, measured on held-out landmarks rather than on Oxford."""


@dataclass(frozen=True)
class TrainConfig:
    """One fine-tuning run.

    Defaults are the reference's published command, not choices made here: image size
    362, five negatives, 2000 queries against a 22000-image pool, Adam at 5e-7 with
    weight decay 1e-6, and an exponential decay of exp(-0.01) per epoch. `margin`
    defaults to the published value for the chosen backbone.

    `weights` defaults to **caffe** for the same reason. The reference fills the
    architecture from its own Caffe-converted ImageNet weights whenever it has them —
    which it does for all three backbones trainable here — so the published 0.619 / 0.647
    were reached from a Caffe initialization. Starting from torchvision's would diverge
    from the recipe before the first step, and the frozen tier already measured how much
    that matters: resnet101 GeM scores 0.347 on torchvision weights against 0.461 on
    Caffe. They are a manual download; see `descriptors/cnn/weights.py`.

    `resume` picks up `last.pth` from the run directory when one is there, which is what
    makes a restart policy worth having: without it a crash at epoch 25 restarts from
    zero, and a supervisor that restarts a non-resumable job is worse than no supervisor.
    Turn it off only to force a fresh run into a directory that already holds one.
    """

    backbone: Backbone = "vgg16"
    weights: WeightSource = "caffe"
    epochs: int = 30
    lr: float = 5e-7
    weight_decay: float = 1e-6
    lr_gamma: float = 0.99005  # exp(-0.01)
    batch_size: int = 5
    margin: float | None = None
    image_size: int = IMAGE_SIZE
    query_size: int = QUERY_SIZE
    pool_size: int = POOL_SIZE
    neg_num: int = NEG_NUM
    p: float = 3.0
    resume: bool = True
    seed: int = 0
    out: Path = field(default=Path("data/runs"))

    def resolved_margin(self) -> float:
        if self.margin is not None:
            return self.margin
        if self.backbone not in MARGIN:
            raise ValueError(f"no published margin for {self.backbone!r}; pass --margin explicitly")
        return MARGIN[self.backbone]


def validate(model: torch.nn.Module, corpus: Corpus, device: str, image_size: int, log=None) -> dict[str, float]:
    """Retrieval accuracy on the held-out split, by SfM cluster.

    A retrieval metric rather than the validation loss: loss falls as the mined negatives
    get harder as well as when the model improves, so it is not comparable across epochs.
    Cluster identity is the label — two images match when they reconstructed into the
    same 3D model, which is the same relation the loss is trained on.
    """
    from pytorch_metric_learning.distances import CosineSimilarity
    from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
    from pytorch_metric_learning.utils.inference import CustomKNN

    model.eval()
    with torch.inference_mode():
        vectors = []
        for i in range(len(corpus.cids)):
            vectors.append(model(load_tensor(corpus.path(i), image_size).unsqueeze(0).to(device)).squeeze(0))
            if log is not None and (i + 1) % 2000 == 0:
                log(f"  validating: {i + 1}/{len(corpus.cids)}")
        embeddings = torch.stack(vectors)

    labels = torch.tensor(corpus.cluster, device=embeddings.device)
    # Exact KNN by matmul, not the FaissKNN default: AGENTS.md keeps search exact and in
    # torch, and pulling in faiss to rank 6403 vectors would be a heavy way to matmul.
    calculator = AccuracyCalculator(include=VAL_METRICS, knn_func=CustomKNN(CosineSimilarity()))
    scores = calculator.get_accuracy(embeddings, labels, embeddings, labels, ref_includes_query=True)
    return {key: float(value) for key, value in scores.items()}


def train_epoch(model, tuples, optimizer, margin: float, batch_size: int, device: str, log=None) -> float:
    """One pass over the epoch's tuples. Returns the mean per-tuple loss."""
    model.train()
    optimizer.zero_grad()
    total = 0.0

    for index in range(len(tuples)):
        images = tuples[index]
        # One forward per image: they differ in size, so they cannot be stacked.
        descriptors = torch.cat([model(image.unsqueeze(0).to(device)) for image in images])
        query = descriptors[:1]
        positive = descriptors[1:2]
        negatives = descriptors[2:].unsqueeze(0)

        loss = contrastive_loss(query, positive, negatives, margin)
        loss.backward()
        total += float(loss.item())

        # Gradients accumulate across `batch_size` tuples before a step, which is how the
        # reference gets a batch out of a loop that can only forward one image at a time.
        if (index + 1) % batch_size == 0:
            optimizer.step()
            optimizer.zero_grad()
        if log is not None and (index + 1) % 200 == 0:
            log(f"  train: {index + 1}/{len(tuples)} loss {total / (index + 1):.4f}")

    if len(tuples) % batch_size:
        optimizer.step()
        optimizer.zero_grad()
    return total / max(len(tuples), 1)


def _track(step: int, values: dict[str, float], config: TrainConfig | None = None, name: str | None = None) -> None:
    """Mirror a curve point into trackio, never raising into the caller."""
    try:
        import trackio
    except ImportError:
        return
    try:
        if name is not None and config is not None:
            trackio.init(project=PROJECT, name=name, config={k: str(v) for k, v in asdict(config).items()})
        trackio.log(values, step=step)
    except Exception:
        # A tracking failure must never cost a training run that took hours.
        return


def _log(message: str) -> None:
    """Print and flush.

    Not plain `print`: stdout to a pipe is block-buffered, so a run redirected to a file
    shows nothing until 4KB accumulates. Over a multi-hour schedule that means no
    progress, and no chance to notice a run diverging until it ends.
    """
    print(message, flush=True)


def train(config: TrainConfig, device: str | None = None, log=_log) -> Path:
    """Run the whole schedule, checkpointing each epoch. Returns the best checkpoint."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.seed)

    model = RetrievalNet(config.backbone, p=config.p, weights=config.weights).to(device)
    margin = config.resolved_margin()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.lr_gamma)

    train_corpus, val_corpus = Corpus.load("train"), Corpus.load("val")
    run = f"{config.backbone}-gem-margin{margin}-lr{config.lr:g}-seed{config.seed}"
    directory = Path(config.out) / run
    directory.mkdir(parents=True, exist_ok=True)
    log(f"{run}: {len(train_corpus.cids)} train images, {len(val_corpus.cids)} val, device {device}")

    best_path = directory / "best.pth"
    last_path = directory / "last.pth"
    start = 1

    resumed = _resume(last_path, model, optimizer, scheduler, config, device, log) if config.resume else None
    if resumed is not None:
        start, best = resumed
        _track(start - 1, {}, config, run)
    else:
        # Measured before any training so every later epoch has something to be better
        # than — and because a baseline that is already wrong catches a broken init.
        baseline = validate(model, val_corpus, device, config.image_size, log)
        log(f"epoch 0 (untrained): {baseline}")
        _track(0, {f"val/{k}": v for k, v in baseline.items()} | {"p": model.pool.p.item()}, config, run)
        best = baseline["mean_average_precision"]
        _save(best_path, model, config, epoch=0, metrics=baseline)

    for epoch in range(start, config.epochs + 1):
        started = time.time()
        # Re-mined every epoch against the *current* model, which is what keeps the
        # negatives hard as the network improves.
        tuples = sample_epoch(
            train_corpus,
            model,
            query_size=config.query_size,
            pool_size=config.pool_size,
            neg_num=config.neg_num,
            device=device,
            seed=config.seed + epoch,
            log=log,
        )
        loss = train_epoch(model, tuples, optimizer, margin, config.batch_size, device, log)
        scheduler.step()
        metrics = validate(model, val_corpus, device, config.image_size, log)

        # Reserved, not allocated: images vary in size, so the caching allocator holds
        # blocks it cannot reuse for the next shape. Logged because that number growing
        # across epochs is the failure mode a long run dies of.
        reserved = torch.cuda.max_memory_reserved() / 1024**3 if device.startswith("cuda") else 0.0
        log(
            f"epoch {epoch}/{config.epochs}: loss {loss:.4f} "
            f"mAP {metrics['mean_average_precision']:.4f} p {model.pool.p.item():.4f} "
            f"({time.time() - started:.0f}s, {reserved:.1f} GB reserved)"
        )
        _track(
            epoch,
            {"train/loss": loss, "p": model.pool.p.item(), "lr": scheduler.get_last_lr()[0]}
            | {f"val/{k}": v for k, v in metrics.items()}
            | {"gpu/reserved_gb": reserved},
        )

        _save(last_path, model, config, epoch, metrics, optimizer, scheduler, best)
        if metrics["mean_average_precision"] > best:
            best = metrics["mean_average_precision"]
            _save(best_path, model, config, epoch, metrics)
            log(f"  new best: {best:.4f}")
            # Rewritten so an interrupt between the two saves cannot leave `last.pth`
            # claiming a `best` that no file on disk matches.
            _save(last_path, model, config, epoch, metrics, optimizer, scheduler, best)

    return best_path


def _resume(path: Path, model, optimizer, scheduler, config: TrainConfig, device: str, log) -> tuple[int, float] | None:
    """Restore a run from `last.pth`, returning the epoch to start at and the best so far.

    Returns `None` when there is nothing to resume, so a first run is the same code path.

    Mining is seeded `config.seed + epoch`, so a run resumed at epoch N draws exactly the
    tuples a fresh run would have drawn there — the schedule is a function of the epoch
    number, not of how many times the process started.
    """
    if not path.exists():
        return None

    payload = torch.load(path, map_location=device, weights_only=False)
    architecture = payload["meta"]["architecture"]
    if architecture != config.backbone:
        raise ValueError(f"{path} holds a {architecture} run; refusing to resume it as {config.backbone}")

    model.load_state_dict(payload["state_dict"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    done = int(payload["meta"]["epoch"])
    best = float(payload["best"])
    log(f"resuming from epoch {done} (best mAP {best:.4f}, p {model.pool.p.item():.4f})")
    return done + 1, best


def _save(
    path: Path,
    model,
    config: TrainConfig,
    epoch: int,
    metrics: dict,
    optimizer=None,
    scheduler=None,
    best: float | None = None,
) -> None:
    """Checkpoint in the reference's layout, so the eval path can read it unchanged.

    `RetrievalNet` names its submodules `features` and `pool`, which is exactly what
    `descriptors/cnn/finetuned.py` strips and reads — a checkpoint written here is
    therefore loadable by the same code that loads the published ones.

    `optimizer`/`scheduler`/`best` are written only for `last.pth`, the resume point.
    `best.pth` is the artifact that gets evaluated and stays free of training state —
    which also keeps it a third of the size.
    """
    payload = {
        "state_dict": model.state_dict(),
        "meta": {
            "architecture": config.backbone,
            "pooling": "gem",
            "outputdim": model.dim,
            "whitening": False,
            "epoch": epoch,
            "metrics": metrics,
            "config": {k: str(v) for k, v in asdict(config).items()},
        },
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if best is not None:
        payload["best"] = best
    torch.save(payload, path)
