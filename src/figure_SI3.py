#!/usr/bin/env python3
"""SI figure: how far apart the same concept sits across the five languages.

One row per concept, ordered by median rank. The segment spans the languages
that place the concept highest and lowest on the phrase curve; the dots are the
five languages in the paper's colours. The shaded band is the null: five
*different* concepts, one drawn per language, which is where the segments would
fall if a concept's position carried no cross-language information.

The message is the comparison of the two, not any individual concept, and the
panel is the whole rank axis so that the spread is read against the range it
lives in.

Inputs: the same committed data as `src/concept_ranks.py`. No raw corpus.
Output: outputs/figures/SI3.{pdf,png}

Usage:
    python src/figure_SI3.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt                                   # noqa: E402
import concept_ranks as CR                                        # noqa: E402
import plotting as P                                              # noqa: E402

N_NULL = 20_000

#: Line weight for both panels, in points. Deliberately below `P.LW["curve"]`
#: (1.5), the weight every other figure uses: this is the only figure that
#: prints alone on a page, with no body text beside it to set the scale, and at
#: the shared weight the 22 segments of panel A and the two step-like ECDFs of
#: panel B both read as heavy blocks rather than as lines.
LW_CURVE = 0.85



def load(seed: int = 0):
    """Per-concept log10 ranks in the five languages, plus the null draws."""
    df, span = CR.build()
    # the reported concept set is defined once, in concept_ranks.reported_ranks,
    # so this figure and every table describe the same 22 concepts
    rank, _, kk, order, _ = CR.reported_ranks(df)
    lg = np.log10(rank)

    rng = np.random.default_rng(seed)
    cols = {c: lg[c].to_numpy() for c in CR.LANGS}
    null = np.array([[rng.choice(cols[c]) for c in CR.LANGS] for _ in range(N_NULL)])
    span_dec = float(np.mean([span[c]["weighted"] for c in CR.LANGS]))
    return lg, kk, null, span_dec


def main() -> int:
    P.setup_style()
    lg, kk, null, span_dec = load()

    med = lg.median(axis=1).sort_values()
    order = list(med.index)
    lg, kk = lg.loc[order], kk.loc[order]

    sd = lg.std(axis=1, ddof=1)
    null_sd = null.std(axis=1, ddof=1)

    # Full text width, at the aspect that holds the height the figure had when
    # it was drawn narrower: the vertical size is what had to come down, and the
    # horizontal room was free. The wider panel B is what lets its legend sit in
    # the headroom without reaching the curves.
    fig, (ax, axr) = plt.subplots(
        1, 2, figsize=P.figsize(1.0, aspect=1.21, n_cols=2),
        gridspec_kw={"width_ratios": [2.2, 1.0]}, layout="constrained")

    # ---- left: one segment per concept ---------------------------------- #
    y = np.arange(len(order))
    lo, hi = lg.min(axis=1).to_numpy(), lg.max(axis=1).to_numpy()

    # Both panels measure spread by the same statistic, the standard deviation
    # of the five log-ranks: the bar spans one s.d. either side of the concept's
    # median, and the grey band is the null's median s.d. on the same scale.
    # Drawing the bar as the min-to-max range instead would compare a range
    # against a band built from a standard deviation.
    half = float(np.median(null_sd))
    ax.barh(y, 2 * half, left=med.to_numpy() - half, height=0.72,
            color="0.88", edgecolor="none", zorder=0,
            label="five different concepts")

    s = sd.to_numpy()
    ax.hlines(y, med.to_numpy() - s, med.to_numpy() + s,
              color="0.55", lw=LW_CURVE, zorder=1)
    for c in P.PLOT_ORDER if hasattr(P, "PLOT_ORDER") else CR.LANGS:
        if c not in lg.columns:
            continue
        ax.scatter(lg[c].to_numpy(), y, s=16, color=P.LANG_COLORS[c],
                   zorder=3, linewidths=0, label=P.LANG_NAMES[c])

    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{c}" + ("*" if kk.loc[c].max() != kk.loc[c].min() else "")
         for c in order], fontsize=P.FONT["tick"])
    ax.set_ylim(-0.8, len(order) - 0.2)
    ax.set_xlabel(r"$\log_{10}$ rank on the phrase curve")
    ax.set_xlim(0, span_dec)
    ax.grid(axis="x", color="0.9", lw=0.5, zorder=-1)
    ax.set_axisbelow(True)
    P.panel_label(ax, "A")

    # the legend goes outside: at printed size an in-panel box covers the top
    # rows, which are exactly the concepts with the widest spread
    handles, labels = ax.get_legend_handles_labels()

    # ---- right: the dispersion against its null -------------------------- #
    # an ECDF, not a histogram: with 22 concepts against 20,000 draws a binned
    # density is mostly bin noise, and the separation to be read is between two
    # distributions, which is what an ECDF shows without a bin width to choose
    def ecdf(v):
        v = np.sort(np.asarray(v, dtype=float))
        return v, np.arange(1, v.size + 1) / v.size

    # the panel is cut at the 99th percentile of the null, which the 22 observed
    # spreads never reach; past its largest observation an ECDF is exactly 1, so
    # each curve is carried flat to the right edge rather than stopping in mid
    # air, which would read as missing data instead of as a saturated curve
    xmax = float(np.percentile(null_sd, 99))

    def to_edge(x, y):
        return (np.append(x, xmax), np.append(y, 1.0)) if x[-1] < xmax else (x, y)

    xn, yn = to_edge(*ecdf(null_sd))
    xs, ys = to_edge(*ecdf(sd.to_numpy()))
    axr.plot(xn, yn, color="0.55", lw=LW_CURVE,
             label="different concepts")
    axr.plot(xs, ys, color=P.LANG_COLORS["en"], lw=LW_CURVE,
             label="same concept")
    axr.plot(sd.to_numpy(), np.full(sd.size, 0.022), "|",
             color=P.LANG_COLORS["en"], ms=5, mew=0.9)
    for v, col in ((float(sd.median()), P.LANG_COLORS["en"]),
                   (float(np.median(null_sd)), "0.55")):
        axr.vlines(v, 0, 0.5, color=col, lw=P.LW["mark"], ls="--")
    # Both ECDFs climb through the upper-left quadrant and both medians are
    # marked below y=0.5, so every corner of the unit square carries ink and a
    # legend placed in any of them overprints a curve. The axis is therefore
    # extended above 1 and the legend put in that strip: an ECDF cannot enter
    # it, so the placement is safe whatever the data do. The headroom is only as
    # deep as the legend needs -- two one-line entries at `FONT["annot"]` clear
    # y=1 inside 1.20, and anything more reads as an empty panel.
    axr.set_ylim(0, 1.20)
    axr.set_yticks(np.arange(0, 1.01, 0.2))
    axr.set_xlim(0, xmax)
    axr.set_xlabel("spread across the five\nlanguages (decades)")
    axr.set_ylabel("cumulative fraction")
    axr.legend(loc="upper left", frameon=False,
               fontsize=P.FONT["annot"], handlelength=1.1,
               labelspacing=0.25, borderaxespad=0.12)
    P.panel_label(axr, "B")

    # six entries: at the drawn width and the current type size neither one row
    # of six nor one of five fits, so they go two rows of three
    fig.legend(handles, labels, loc="outside upper center", ncol=3,
               frameon=False, fontsize=P.FONT["legend"], handletextpad=0.3,
               columnspacing=1.2)

    paths = P.save_figure(fig, "SI3")
    print(f"n = {len(order)} concepts, panel = {span_dec:.2f} decades")
    print(f"  same concept   : median s.d. {sd.median():.2f} dec "
          f"({100 * sd.median() / span_dec:.1f}% of the panel)")
    print(f"  five different : median s.d. {np.median(null_sd):.2f} dec "
          f"({100 * np.median(null_sd) / span_dec:.1f}%)")
    print(f"  ratio {np.median(null_sd) / sd.median():.1f}x; only "
          f"{100 * np.mean(null_sd <= sd.median()):.1f}% of random quintuples "
          f"are as tight as the median concept")
    print(f"  different word counts across languages: "
          f"{int((kk.max(axis=1) != kk.min(axis=1)).sum())}/{len(order)}")
    for p in paths:
        print("wrote", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
