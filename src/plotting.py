#!/usr/bin/env python3
"""Shared plotting style for every figure notebook (single source of truth).

Nothing here computes science: colors, fonts, log-sampling of curves and file
output only. All exponents come from `fits.py`, all data from `io_reduced.py`.

Language colors, as printed in Figure 1:
English purple, French gold, Italian green, Spanish salmon, German magenta.

**Figures are drawn at their final printed size.** Every figure is authored
`TEXT_WIDTH_IN` inches wide and included with `\\includegraphics[width=\\linewidth]`,
so nothing is rescaled on the page and the point sizes set in `FONT` are the
point sizes that reach the reader. Do not pass explicit `figsize=(w, h)` tuples
in a notebook: call `figsize()` instead, or the type in that figure will end up a
different size from the type in every other one.
"""

from __future__ import annotations

import os
import shutil

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(REPO_ROOT, "outputs", "figures")
TABLE_DIR = os.path.join(REPO_ROOT, "outputs", "tables")


# --------------------------------------------------------------------------- #
# Languages: codes, names, colors, plot order
# --------------------------------------------------------------------------- #
LANG_NAMES = {
    "en": "English",
    "fr": "French",
    "it": "Italian",
    "es": "Spanish",
    "de": "German",
}

LANG_COLORS = {
    "en": "#7A5197",   # purple
    "fr": "#F5C63C",   # gold
    "it": "green",     # green
    "es": "#F47F6B",   # salmon
    "de": "#BB5098",   # magenta
}

#: Colors for the network / n-gram quantities of Figures 2 and 8, as printed.
QUANTITY_COLORS = {
    "degree": "#FB5454",    # red
    "strength": "#7A5197",  # purple
    "2gram": "#FB5454",
    "1gram": "#7A5197",
}

#: Colors for the model-versus-data comparison. Deliberately outside the
#: language and network palettes above: that comparison is between two *sources*,
#: not two languages and not two network quantities, so it must not borrow a
#: colour that already means something else in the paper. Blue against orange
#: also survives the common forms of colour blindness, which red against grey
#: does not.
MODEL_COLORS = {
    "data": "#33566E",       # slate blue
    "model": "#D1743B",      # terracotta
}

#: Colors for the learner/native comparison, as printed.
#:
#: `gutenberg` is the SPGC English corpus, which is also the `data` curve of the
#: model comparison, so it takes the same slate blue: the same corpus must not
#: appear in two colours, whether inside one figure (main-text Figure 4 draws it
#: in panels A, B and C) or across the paper and its appendices. It was grey
#: until 2026-08-10, when the learner panel was promoted into the main text
#: beside the model comparison and the clash became visible.
GROUP_COLORS = {
    "learners": "#F5A623",         # orange
    "natives": "#7A5197",          # purple — the COREFL native control
    "gutenberg": MODEL_COLORS["data"],
}

#: Order used for the legend, as printed in the paper.
LEGEND_ORDER = ["en", "fr", "it", "es", "de"]

#: Order used when drawing curves; last drawn is on top. German last so its
#: (mildest, longest) tail stays visible where the curves separate.
PLOT_ORDER = ["es", "fr", "it", "en", "de"]


# --------------------------------------------------------------------------- #
# Page geometry and typography
# --------------------------------------------------------------------------- #
#: Text width of the manuscript, in TeX points, measured from `arxiv.sty`
#: (`\the\linewidth` = 469.755 pt). A figure drawn this wide is included with
#: `width=\linewidth` at scale 1.
TEXT_WIDTH_PT = 469.755
TEXT_WIDTH_IN = TEXT_WIDTH_PT / 72.27          # 6.50 in

#: Type sizes in points, as they appear on the printed page. Raised twice over
#: the first draft of the figures, by one point each time: the rule is that no
#: text inside a figure should be smaller than the 10 pt body text it sits next
#: to, and after the first raise the tick labels were still a shade under it.
#: Changed here and nowhere else, so every figure in the paper and the SI keeps
#: a single typographic register -- which means every figure has to be
#: regenerated whenever this block moves.
FONT = {
    "tick": 10.5,
    "label": 11.5,
    "legend": 10.5,
    "annot": 10.0,     # in-panel annotations: guide slopes, marked scales
    "panel": 13.0,     # the A / B / C / D panel letters
}

#: Line widths in points, as printed.
LW = {
    "curve": 1.5,      # the data curves
    "aux": 1.2,        # secondary curves drawn for contrast
    "guide": 0.9,      # dashed reference power laws
    "mark": 0.9,       # marked scales (R*, D0) and other verticals
    "axes": 0.7,       # spines
}


def figsize(rel_width: float = 1.0, aspect: float = 0.62,
            n_cols: int = 1, n_rows: int = 1) -> tuple[float, float]:
    """Figure size in inches, for a figure included at `rel_width` x \\linewidth.

    `aspect` is the height/width ratio of a *single panel*, so a 1x3 row of
    panels and a single panel of the same aspect have panels of the same shape.
    Height is therefore ``width * aspect * n_rows / n_cols``.
    """
    width = TEXT_WIDTH_IN * float(rel_width)
    return (width, width * float(aspect) * n_rows / n_cols)


# --------------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------------- #
def usetex_available() -> bool:
    """True if a usable LaTeX toolchain is present (and not disabled by env).

    Set ``FIGSTYLE_USETEX=0`` to force the mathtext fallback (cloud sessions,
    phones, CI). ``FIGSTYLE_USETEX=1`` forces LaTeX on.
    """
    env = os.environ.get("FIGSTYLE_USETEX")
    if env is not None:
        return env.strip() not in ("0", "false", "False", "")
    return all(shutil.which(p) for p in ("latex", "dvipng"))


def setup_style(usetex: bool | None = None, fontsize: float | None = None) -> bool:
    """Apply the shared figure style. Returns whether LaTeX rendering is on.

    With LaTeX available the typography matches the published figures (Latin
    Modern); otherwise matplotlib's Computer Modern mathtext is used, which is
    visually equivalent for these labels.

    `fontsize` overrides the tick size of `FONT` and rescales the rest with it.
    Leave it unset: the defaults are the printed point sizes, and overriding it
    in one notebook is how the figures drifted out of register in the first
    place.
    """
    if usetex is None:
        usetex = usetex_available()
    scale = 1.0 if fontsize is None else float(fontsize) / FONT["tick"]

    # start from the defaults, but never touch the backend: inside Jupyter that
    # would switch away from the inline backend and silence every figure
    defaults = {k: v for k, v in mpl.rcParamsDefault.items()
                if k not in ("backend", "interactive")}
    mpl.rcParams.update(defaults)
    mpl.rcParams.update({
        "text.usetex": bool(usetex),
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": FONT["tick"] * scale,
        "axes.labelsize": FONT["label"] * scale,
        "xtick.labelsize": FONT["tick"] * scale,
        "ytick.labelsize": FONT["tick"] * scale,
        "legend.fontsize": FONT["legend"] * scale,
        "axes.linewidth": LW["axes"],
        "lines.linewidth": LW["curve"],
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 1.8,
        "ytick.minor.size": 1.8,
        "xtick.major.width": LW["axes"],
        "ytick.major.width": LW["axes"],
        "xtick.minor.width": LW["axes"] * 0.8,
        "ytick.minor.width": LW["axes"] * 0.8,
        "figure.figsize": figsize(),
        # NOT savefig.bbox="tight": a tight box is cropped to the ink, so the
        # saved width depends on the data and every figure would land on the
        # page at a different scale -- which is exactly how the type sizes
        # drifted apart before. Fixed size + constrained layout means each PDF
        # is 468 bp = 6.5 in wide and is included at scale 1.
        "savefig.bbox": None,
        "savefig.dpi": 400,
        "figure.dpi": 110,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.02,
        "figure.constrained_layout.w_pad": 0.02,
        "figure.constrained_layout.hspace": 0.04,
        "figure.constrained_layout.wspace": 0.04,
    })
    if usetex:
        mpl.rcParams["text.latex.preamble"] = r"\usepackage{lmodern}"
    return bool(usetex)


# --------------------------------------------------------------------------- #
# Curve helpers
# --------------------------------------------------------------------------- #
def log_indices(n: int, num: int = 200) -> np.ndarray:
    """`num` unique 0-based indices, log-spaced over [0, n-1] (endpoints kept).

    Rank-frequency curves span 6 decades with ~10**6 points; plotting them all
    makes a multi-MB vector figure whose low ranks are invisible anyway. Log
    sampling keeps the shape exactly and the file small. Fits always run on the
    full vector, never on the sampled one.
    """
    if n <= num:
        return np.arange(n)
    idx = np.unique(np.round(np.logspace(0, np.log10(n), num=num)).astype(np.int64) - 1)
    return np.clip(idx, 0, n - 1)


def rank_prob(freq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Descending count vector -> (rank, normalized probability p(R))."""
    freq = np.asarray(freq)
    freq = np.sort(freq[freq > 0])[::-1]
    return np.arange(1, freq.size + 1), freq / freq.sum()


def plateau_points(rank: np.ndarray, value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One representative point per constant run of a descending step curve.

    A rank-frequency tail is a staircase: thousands of consecutive ranks share
    a count of 4, then of 3, then of 2. Sampling that log-uniformly draws the
    treads, and the curve reads as a flight of steps rather than as a trend.
    Taking the geometric midpoint of each tread puts one point in the middle of
    each level instead, so the polyline through them follows the curve without
    tracing its corners. Nothing is fitted on these points: they are for
    drawing only, and every estimator still runs on the full vector.
    """
    rank = np.asarray(rank, dtype=float)
    value = np.asarray(value)
    edge = np.flatnonzero(np.diff(value)) + 1        # first index of each run
    starts = np.concatenate([[0], edge])
    stops = np.concatenate([edge, [value.size]]) - 1  # last index of each run
    return np.sqrt(rank[starts] * rank[stops]), value[starts]


def guide_line(ax, alpha: float, x: tuple[float, float], y_at: tuple[float, float],
               **kw) -> None:
    """Draw a dashed reference power law p ∝ R^-alpha through (x0, y0).

    `x` is (x_start, x_stop); `y_at` is the anchor point (x_anchor, y_anchor).
    """
    xs = np.logspace(np.log10(x[0]), np.log10(x[1]), 50)
    x0, y0 = y_at
    style = dict(ls="--", color="black", lw=LW["guide"], zorder=5)
    style.update(kw)
    ax.plot(xs, y0 * (xs / x0) ** (-alpha), **style)


def mark_zoom(ax, axins, corners=("upper", "lower"), **kw) -> None:
    """Outline the region shown by `axins` on `ax` and connect the two boxes.

    matplotlib's `indicate_inset_zoom` links matching corners, which for an
    inset placed to the lower-left of its source region draws a connector
    straight across the whole panel. Here the rectangle's left corners are
    linked to the inset's right corners instead.
    """
    from matplotlib.patches import ConnectionPatch, Rectangle

    x0, x1 = axins.get_xlim()
    y0, y1 = axins.get_ylim()
    style = dict(ec="black", fc="none", lw=LW["mark"], zorder=6)
    style.update(kw)
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, **style))
    for corner in corners:
        y_rect, y_ins = (y1, 1.0) if corner == "upper" else (y0, 0.0)
        ax.add_artist(ConnectionPatch(
            xyA=(x0, y_rect), coordsA=ax.transData,
            xyB=(1.0, y_ins), coordsB=axins.transAxes,
            color=style["ec"], lw=style["lw"], zorder=6))


def language_handles(order=None, lw: float | None = None) -> list[Line2D]:
    """Legend handles, one per language, in the paper's legend order."""
    order = LEGEND_ORDER if order is None else order
    lw = LW["curve"] * 1.4 if lw is None else lw
    return [Line2D([0], [0], color=LANG_COLORS[c], lw=lw, label=LANG_NAMES[c])
            for c in order]


# --------------------------------------------------------------------------- #
# Annotations
# --------------------------------------------------------------------------- #
def panel_label(ax, letter: str, **kw) -> None:
    """Put the panel letter (A, B, ...) above the top-left corner of `ax`.

    Set as a left-aligned axes *title* rather than free text, because a title is
    measured by the constrained layout engine: with the figure saved at a fixed
    size (no tight bounding box), free text placed above the axes would simply
    be cropped.

    Under `text.usetex` matplotlib's `fontweight` is ignored -- LaTeX sets the
    weight, not the renderer -- so the letter is wrapped in `\\textbf` instead.
    """
    style = dict(fontsize=FONT["panel"], loc="left", pad=3.0)
    if mpl.rcParams["text.usetex"]:
        letter = r"\textbf{%s}" % letter
    else:
        style["fontweight"] = "bold"
    style.update(kw)
    ax.set_title(letter, **style)


def mark_scales(ax, marks, y: float = 0.96, dy: float = 0.085,
                min_sep: float = 0.10, pad: float = 0.012,
                corner: str | None = None, fontsize: float | None = None,
                **kw) -> list:
    """Draw labelled vertical markers, stacking labels that would collide.

    `marks` is a sequence of ``(x, label, color)`` or of dicts with those keys
    plus any per-mark line style. The line spans the panel; the label sits at the
    top, in axes coordinates.

    Two placement rules, both of which the hand-placed annotations they replace
    got wrong on at least one figure:

    * **Collision.** When two markers are closer than `min_sep` in axes-x their
      labels would overprint, so each successive one is pushed down by `dy`.
      This is the case of the two $R^*$ verticals of Figure 4A, which differ by
      6% and so sit almost on top of each other -- which *is* the message, so
      they must stay where they are and only the text may move.
    * **Edge.** Every label is measured once it exists and flipped to the left
      of its line if it would cross the right spine. Measuring beats guessing
      from `x`: the labels here are long ("$R^*=9{,}388$") and a marker at 70%
      of a three-inch panel already overflows.

    Both rules read the axis limits, so call this *after* `set_xlim`.

    `corner` overrides both: pass ``"lower left"`` (or ``"lower right"``,
    ``"upper left"``, ``"upper right"``) and the labels are stacked as a small
    colour-coded block in that corner of the panel, while the verticals stay
    on their values. On a rank-frequency panel the curve runs from the upper
    left to the lower right, so a label attached to a vertical near the middle
    of the axis lands on the data whatever its height, and the two free
    corners are the only places it can go without overprinting something.

    Returns the list of `Text` artists, in the order given.
    """
    marks = [m if isinstance(m, dict) else dict(zip(("x", "label", "color"), m))
             for m in marks]
    fontsize = FONT["annot"] if fontsize is None else fontsize
    texts = []

    if corner is not None:
        vpos, hpos = corner.split()
        xa = 0.030 if hpos == "left" else 0.970
        ya = 0.035 if vpos == "lower" else 0.965
        for k, m in enumerate(marks):
            style = dict(ls="--", lw=LW["mark"], color=m.get("color", "black"),
                         alpha=0.9, zorder=4)
            style.update({s: m[s] for s in ("ls", "lw", "alpha", "zorder") if s in m})
            style.update(kw)
            ax.axvline(m["x"], **style)
            # stack away from the edge: upwards from a lower corner, down from
            # an upper one, in the order the marks were given
            step = k * dy * (1 if vpos == "lower" else -1)
            texts.append(ax.annotate(
                m["label"], xy=(xa, ya + step), xycoords="axes fraction",
                color=style["color"], fontsize=fontsize,
                ha=hpos, va="bottom" if vpos == "lower" else "top", zorder=6,
                bbox=dict(boxstyle=f"square,pad={pad}", fc="white", ec="none",
                          alpha=0.75)))
        return texts

    def frac(x):
        lo, hi = ax.get_xlim()
        if ax.get_xscale() == "log":
            return (np.log10(x) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
        return (x - lo) / (hi - lo)

    # stack the labels of markers that sit within `min_sep` of each other
    row, seen_f, seen_k = {}, None, None
    for k in sorted(range(len(marks)), key=lambda i: frac(marks[i]["x"])):
        f = frac(marks[k]["x"])
        row[k] = 0 if seen_f is None or f - seen_f > min_sep else row[seen_k] + 1
        seen_f, seen_k = f, k

    for k, m in enumerate(marks):
        style = dict(ls="--", lw=LW["mark"], color=m.get("color", "black"),
                     alpha=0.9, zorder=4)
        style.update({s: m[s] for s in ("ls", "lw", "alpha", "zorder") if s in m})
        style.update(kw)
        ax.axvline(m["x"], **style)
        texts.append(ax.annotate(
            m["label"], xy=(m["x"], y - row[k] * dy),
            xycoords=("data", "axes fraction"), xytext=(3, 0),
            textcoords="offset points", color=style["color"],
            fontsize=fontsize, ha="left", va="top", zorder=6,
            bbox=dict(boxstyle=f"square,pad={pad}", fc="white", ec="none", alpha=0.75)))

    # a label near the right spine would run out of the panel; measure it and
    # flip it to the other side of its line rather than guessing from `x` alone
    try:
        renderer = ax.figure.canvas.get_renderer()
    except AttributeError:                       # backend without a renderer yet
        ax.figure.canvas.draw()
        renderer = ax.figure.canvas.get_renderer()
    right_edge = ax.get_window_extent(renderer).x1
    for text in texts:
        if text.get_window_extent(renderer).x1 > right_edge:
            text.set_ha("right")
            text.xyann = (-3, 0)
    return texts


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def save_figure(fig, name: str, formats=("pdf", "png")) -> list[str]:
    """Save `fig` as outputs/figures/<name>.<fmt> for each format. Returns paths."""
    os.makedirs(FIG_DIR, exist_ok=True)
    paths = []
    for fmt in formats:
        path = os.path.join(FIG_DIR, f"{name}.{fmt}")
        fig.savefig(path, format=fmt)
        paths.append(path)
    return paths


def table_path(name: str) -> str:
    """Absolute path inside outputs/tables/ (directory created on demand)."""
    os.makedirs(TABLE_DIR, exist_ok=True)
    return os.path.join(TABLE_DIR, name)
