"""Shared chart styling for the comparison notebooks.

Lives beside the notebooks rather than in `src/cbir/` because matplotlib is a dev
dependency: the package itself must import without it. AGENTS.md keeps `notebooks/` to
analysis and figures, and this is the figure half of that.

Colour is assigned by the job it does, never by rank:

* **Technique** (`TECHNIQUE`) — categorical identity, palette slots in fixed order, so
  filtering one technique out never repaints the others.
* **Tier** (`TIER`) — an *ordered* category (classic, off-the-shelf, fine-tuned), so it
  takes an ordinal blue ramp rather than three unrelated hues: darker reads as later.
* **Weight provenance** (`WEIGHTS`) — two categorical slots.

Every palette here was checked with the data-viz validator rather than by eye. The
ordinal ramp passes monotone lightness, adjacent ΔL, and the light-end contrast floor;
the categorical pairs pass CVD separation and the normal-vision floor. Three light-mode
slots sit under 3:1 on white, so **identity never rests on the swatch alone** — every
chart here also direct-labels or is backed by the table view above it.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
SURFACE = "#ffffff"

TECHNIQUE = {
    "bow": "#2a78d6",
    "vlad": "#eb6834",
    "fisher": "#1baf7a",
    "neural_codes": "#eda100",
    "gem": "#e87ba4",
    "rmac": "#008300",
}
"""Categorical slots 1-6, in the palette's fixed order."""

LABEL = {
    "bow": "BoW",
    "vlad": "VLAD",
    "fisher": "Fisher",
    "neural_codes": "Neural Codes",
    "gem": "GeM",
    "rmac": "R-MAC",
}

TIER = {
    "classic": "#86b6ef",
    "off-the-shelf CNN": "#2a78d6",
    "fine-tuned CNN": "#104281",
}
"""Ordinal ramp, light to dark, because the tiers are an ordered sequence.

Steps 250 / 450 / 650 of the blue ramp. The lightest clears the ordinal floor against
white (2.06:1) but not the 3:1 categorical one, which is why tiers are always direct-
labelled as well."""

TIER_ORDER = list(TIER)

WEIGHTS = {"torchvision": "#2a78d6", "caffe": "#eb6834"}

PROVENANCE = {"published": "#2a78d6", "trained here": "#eb6834"}
"""Categorical slots 1-2 for where a checkpoint came from.

The same two hues `WEIGHTS` uses, for a different question -- they never share a chart,
and slots 1-2 are the pair the validator clears most comfortably in both modes. Not an
ordinal ramp: neither provenance is "more" than the other, and the whole point of these
charts is to ask whether they land in the same place."""

PROTOCOLS = ["easy", "medium", "hard"]

RC = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": MUTED,
    "axes.titlecolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "font.size": 10,
    "figure.dpi": 120,
}


def use() -> None:
    """Apply the shared rcParams to this kernel."""
    plt.rcParams.update(RC)


def style(ax, title=None, xlabel=None, ylabel=None, axis="y"):
    """Recessive grid behind the marks, no top/right spines."""
    ax.set_axisbelow(True)
    ax.grid(True, axis=axis, alpha=0.9)
    if title:
        ax.set_title(title, loc="left", fontsize=11, pad=8)
    ax.set_xlabel(xlabel or "")
    ax.set_ylabel(ylabel or "")
    return ax


def tier_of(technique: str, params: dict) -> str:
    """Which era a run belongs to — the axis the project's story runs along."""
    if technique in ("bow", "vlad", "fisher"):
        return "classic"
    if params.get("weights") == "sfm120k":
        return "fine-tuned CNN"
    return "off-the-shelf CNN"


def label_point(ax, x, y, text, dx=7, dy=0, **kwargs):
    """Direct label in ink, never in the series colour.

    `va`/`fontsize`/`color` are defaults, not fixtures -- a caller rotating a label or
    lifting it off a rule needs to override them.
    """
    ax.annotate(
        text,
        (x, y),
        textcoords="offset points",
        xytext=(dx, dy),
        **{"fontsize": 8, "color": MUTED, "va": "center", **kwargs},
    )
