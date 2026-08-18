#!/usr/bin/env python3
"""Figure 1 — the two panels, and the composed two-panel figure.

Panel A is the 1-gram rank-frequency distribution of five languages; panel B is
the phrase distribution obtained from the Williams random partition
(`phrases_williams.py`), with the English single-word curve of panel A drawn on
the same axes for contrast.

The drawing lives here, not in the notebook, so that the panels and the composed
figure cannot drift apart: `word_curves` and `phrase_curves` build the two
curves, `compose` puts them side by side, and each panel can be drawn on its own
from the same functions. Notebook 04 calls `compose`.

Nothing here computes science: the curves come from `io_reduced` and
`phrases_williams`, the style from `plotting`.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

import plotting as P
from io_reduced import load_1gram, load_ngram
from phrases_williams import expand_for_fit, phrase_spectrum

Q = 0.5                       # canonical break probability of the partition
ORDERS = (1, 2, 3, 4, 5)
WORDS_STYLE = dict(color="0.35", ls=":", lw=P.LW["aux"])
WORDS_LABEL = "English, single words"


# --------------------------------------------------------------------------- #
# Curves
# --------------------------------------------------------------------------- #
def word_curves(langs=None) -> dict:
    """`{lang: (rank, prob, n2)}` for the 1-gram distributions.

    `n2` is the last rank with a count of at least 2, i.e. where the hapax
    plateau begins; it is excluded from every fit and drawn pale.
    """
    langs = P.LEGEND_ORDER if langs is None else langs
    out = {}
    for c in langs:
        g = load_1gram(c)
        rank, prob = P.rank_prob(g.freq)
        out[c] = (rank, prob, int((g.freq >= 2).sum()))
    return out


def phrase_curves(langs=None, q: float = Q, max_points: int = 400) -> dict:
    """`{lang: (rank, prob)}` for the Williams phrase distributions."""
    langs = P.LEGEND_ORDER if langs is None else langs
    return {c: expand_for_fit(phrase_spectrum(load_ngram(c), q=q, orders=ORDERS),
                              max_points=max_points)
            for c in langs}


# --------------------------------------------------------------------------- #
# Panels
# --------------------------------------------------------------------------- #
def _draw_words(ax, curves, c, lw=None, num=220, hapax=True) -> None:
    """One log-sampled rank-frequency curve; the f == 1 plateau drawn pale."""
    lw = P.LW["curve"] if lw is None else lw
    rank, prob, n2 = curves[c]
    # one point per constant run, not one per log-spaced rank: the tail is a
    # staircase of counts 5, 4, 3, 2 and sampling it directly draws the treads
    x, y = P.plateau_points(rank[:n2], prob[:n2])
    i = P.log_indices(x.size, num)
    ax.plot(x[i], y[i], "-", color=P.LANG_COLORS[c], lw=lw, zorder=3,
            solid_capstyle="round")
    if hapax:
        # the f == 1 plateau is genuinely flat, so three points draw it exactly:
        # the last resolved count, the step down, and the run to the last rank
        j = [n2 - 1, n2, rank.size - 1]
        ax.plot(rank[j], prob[j], "-", color=P.LANG_COLORS[c], lw=lw * 0.9,
                alpha=0.30, zorder=2)


def panel_words(ax, curves, inset: bool = True):
    """Panel A: 1-gram rank-frequency, five languages, with a tail zoom.

    Returns the inset axes, or None.
    """
    def p_at(c, r):
        prob = curves[c][1]
        return prob[min(int(r), prob.size) - 1]

    for c in P.PLOT_ORDER:
        _draw_words(ax, curves, c, hapax=False)

    # one reference slope only: the R^-2 guide belongs to the tail, and the tail
    # is what the inset is for, so drawing it here as well says it twice
    P.guide_line(ax, 1.0, (1, 2e3), (10, p_at("en", 10) * 2.6))
    ax.text(7e2, 1.3e-3, r"$R^{-1}$", ha="left", va="bottom",
            fontsize=P.FONT["annot"])

    ax.set(xscale="log", yscale="log", xlabel=r"rank $R$", ylabel=r"$p(R)$")
    ax.set_xlim(0.6, 1e6)
    ax.set_ylim(2e-9, 3e-1)
    ax.set_xticks([1e0, 1e2, 1e4, 1e6])
    ax.set_yticks([1e-8, 1e-6, 1e-4, 1e-2])
    if not inset:
        return None

    # --- inset: zoom on the tail, where the languages separate ---------------
    zx = (8.0e4, 6.0e5)
    axins = ax.inset_axes([0.030, 0.045, 0.50, 0.45])
    # no hapax plateau inside the zoom: the inset exists to show where the five
    # languages separate, which is the resolved part of the tail, and the pale
    # step down to p = 1/T would put five right angles across the same space
    for c in P.PLOT_ORDER:
        _draw_words(axins, curves, c, lw=P.LW["curve"] * 0.9, hapax=False)

    # guides bracket the fan, each parallel to the language it describes. The
    # labels sit clear of their own line: R^-2 below its left end, R^-1.5 above
    # its right end, which are the two places the fan does not reach
    g2_y0 = p_at("en", zx[0]) * 0.50       # just below the steepest curve (English)
    g15_y0 = p_at("de", zx[0]) * 1.90      # just above the mildest curve (German)
    P.guide_line(axins, 2.0, zx, (zx[0], g2_y0), lw=P.LW["guide"] * 0.9)
    P.guide_line(axins, 1.5, zx, (zx[0], g15_y0), lw=P.LW["guide"] * 0.9)
    # the R^-2 label goes under the left end of its guide, where the guide is
    # still high above the floor; under the right end it would need a floor
    # below the main panel's own y-limit, and the zoom rectangle would then be
    # drawn partly outside the panel and drag its connectors off the page
    axins.text(1.2e5, g2_y0 * (1.2e5 / zx[0]) ** -2.0 * 0.50, r"$R^{-2}$",
               fontsize=P.FONT["annot"], va="top", ha="center")
    axins.text(3.0e5, g15_y0 * (3.0e5 / zx[0]) ** -1.5 * 1.55, r"$R^{-1.5}$",
               fontsize=P.FONT["annot"], va="bottom", ha="center")

    # the zoom window must stay inside the main panel, else the outline and the
    # connectors run off the axes. The floor is set by the fan, not by the
    # guides: the lowest resolved point of any language, with room under it
    fan_floor = min(curves[c][1][curves[c][2] - 1] for c in curves)
    axins.set(xscale="log", yscale="log", xlim=zx,
              ylim=(fan_floor * 0.42, g15_y0 * 1.7))
    axins.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
    for axis in (axins.xaxis, axins.yaxis):
        axis.set_major_formatter(NullFormatter())
        axis.set_minor_formatter(NullFormatter())
    for spine in axins.spines.values():
        spine.set_linewidth(P.LW["axes"])
    P.mark_zoom(ax, axins)
    return axins


def panel_phrases(ax, phrases, word_curves_en=None):
    """Panel B: Williams phrases, five languages.

    `word_curves_en` is accepted and ignored: panel A already draws the English
    single-word curve, and repeating it here as a dotted line only asked the
    reader to tell two curves of the same corpus apart across two panels.
    """
    for c in P.PLOT_ORDER:
        r, p = phrases[c]
        ax.plot(r, p, "-", color=P.LANG_COLORS[c], lw=P.LW["curve"], zorder=3,
                solid_capstyle="round")

    r_en, p_en = phrases["en"]
    P.guide_line(ax, 1.0, (1, 3e8), (1e3, p_en[np.searchsorted(r_en, 1e3)] * 4.0))
    ax.text(3e4, 1.1e-4, r"$R^{-1}$", ha="left", va="bottom",
            fontsize=P.FONT["annot"])

    ax.set(xscale="log", yscale="log", xlabel=r"rank $R$", ylabel=r"$p(R)$")
    # the longest curve reaches 3.7*10^8, so the axis has to clear it: at 3*10^8
    # the fan ran off the right spine and its end could not be seen
    ax.set_xlim(0.5, 9e8)
    ax.set_ylim(1e-11, 3e-1)
    ax.set_xticks([1e0, 1e2, 1e4, 1e6, 1e8])
    ax.set_yticks([1e-10, 1e-8, 1e-6, 1e-4, 1e-2])
    return ax


# --------------------------------------------------------------------------- #
# The composed figure
# --------------------------------------------------------------------------- #
def legend_handles(with_words: bool = False) -> list[Line2D]:
    handles = P.language_handles()
    if with_words:
        handles.append(Line2D([0], [0], label=WORDS_LABEL, **WORDS_STYLE))
    return handles


def compose(words=None, phrases=None):
    """Build the two-panel Figure 1 of the manuscript. Returns `(fig, axes)`."""
    words = word_curves() if words is None else words
    phrases = phrase_curves() if phrases is None else phrases

    fig, (axA, axB) = plt.subplots(1, 2, figsize=P.figsize(aspect=0.92, n_cols=2))
    panel_words(axA, words)
    panel_phrases(axB, phrases, words["en"])
    for ax, letter in ((axA, "A"), (axB, "B")):
        P.panel_label(ax, letter)

    fig.legend(handles=legend_handles(), loc="outside upper center", ncol=5,
               frameon=False, handlelength=1.5, columnspacing=1.1,
               handletextpad=0.5)
    return fig, (axA, axB)


def main() -> int:
    P.setup_style()
    fig, _ = compose()
    for path in P.save_figure(fig, "fig1"):
        print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
