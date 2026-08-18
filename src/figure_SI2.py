#!/usr/bin/env python3
"""Appendix figure S2 — the local slope of the phrase and single-word curves.

Figure 1B shows that phrases stay close to Zipf over a much broader range than
single words do. A single fitted exponent cannot show *where* that holds, so this
figure plots the local slope itself: alpha(R) measured in a sliding window half a
decade wide, against rank.

The five phrase curves sit inside the shaded band 1 +- 0.15 over most of their
range, while the English single-word curve leaves it early and climbs towards 2.
That is the same statement as the "Zipf range" column of the phrase-exponent
table, drawn rather than summarised.

Style is shared with the composed Figure 1: the language colours, the dashed
single-word style and the legend all come from `figure_1.py`, so the two figures
cannot drift apart.

Usage:
    python src/figure_SI2.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt                                    # noqa: E402

import figure_1 as F1                                              # noqa: E402
import plotting as P                                               # noqa: E402
from fits import local_slope                                       # noqa: E402
from io_reduced import load_1gram                                  # noqa: E402
from phrase_exponents import LANGS, TOL, spectra                   # noqa: E402
from phrases_williams import expand_for_fit                        # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def draw(spec):
    """Draw the local-slope panel and return the figure."""
    fig, ax = plt.subplots(figsize=P.figsize(aspect=0.42))

    for c in P.PLOT_ORDER:
        r, p = expand_for_fit(spec[c])
        rc, a = local_slope(r, p, n_points=80, half_width=0.5)
        ax.plot(rc, a, "-", color=P.LANG_COLORS[c], lw=P.LW["curve"])

    g_en = load_1gram("en")
    rank, prob = P.rank_prob(g_en.freq)
    n2 = int((g_en.freq >= 2).sum())
    rc, a = local_slope(rank[:n2].astype(float), prob[:n2],
                        n_points=80, half_width=0.5)
    ax.plot(rc, a, **F1.WORDS_STYLE)

    ax.axhline(1.0, color="black", lw=P.LW["guide"], ls="--")
    ax.axhspan(1 - TOL, 1 + TOL, color="black", alpha=0.07, lw=0)
    ax.set(xscale="log", xlabel=r"rank $R$", ylabel=r"local $\alpha$")
    ax.set_xlim(1e1, 2e9)
    ax.set_ylim(0.4, 2.3)

    fig.legend(handles=F1.legend_handles(with_words=True),
               loc="outside upper center", ncol=6, frameon=False,
               handlelength=1.5, columnspacing=1.1, handletextpad=0.5)
    return fig


def main() -> int:
    P.setup_style()
    _, spec, _ = spectra(verbose=False)
    fig = draw(spec)
    for path in P.save_figure(fig, "SI2"):
        print("wrote", os.path.relpath(path, REPO))
    return 0


if __name__ == "__main__":
    sys.exit(main())
