#!/usr/bin/env python3
"""Figure 2 of the main text — the word co-occurrence network of English.

Three views of the same English corpus (SPGC):

  (A) log-binned probability density of node degree k (distinct neighbours) and
      strength s (total incident bigram weight);
  (B) the rank-size curves k(R) and s(R);
  (C) the rank-frequency curves f(R) of 1-grams and 2-grams.

The network is the undirected graph of adjacent tokens, built per book by
`build_reduced.py`: n-grams and edges never cross a book boundary, and
self-loops (a word adjacent to itself) are not counted — 0.45% of the bigram
positions in English.

A node's strength counts every bigram position it takes part in, so s = 2 f: a
word is once the left member of a bigram and once the right. The identity is
exact for every token except the two ends of a book and the self-loops excluded
above, so it fails for a minority of *types* — 5.2% of the English ranks, where
a hapax at a book boundary carries s = 1 rather than 2. `check_identity`
therefore asserts the median of s/2f and prints its 5th-95th percentiles, and it
is the rank curves that coincide: the strength curve of panel B *is* the Zipf
curve of Figure 1A on the same corpus, rescaled.

No reference slopes are drawn. Three panels once carried seven dashed guides
between them, none of which is a result; the fitted exponents and their bands
are the table this module writes.

Nothing here computes science beyond the fits: curves come from `io_reduced`,
estimators from `fits`, style from `plotting`.

Usage:
    python src/figure_2.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt                                    # noqa: E402

import plotting as P                                               # noqa: E402
from fits import (logbin_pdf, logbin_split_band, ols_rank_prob_band,  # noqa: E402
                  two_regime_ols_band)
from io_reduced import load_1gram, load_ngram_rank_freq, load_wcn  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LANG = "en"
HEAD = (1e0, 1e3)
SPLIT = 1e3          # split of the log-binned densities into two regimes
BINS = 24


def load():
    """Degree, strength and the 1-gram frequency vector, all descending."""
    wcn = load_wcn(LANG)
    gram = load_1gram(LANG)
    degree = wcn["degree"].astype(np.int64)
    strength = wcn["strength"].astype(np.int64)
    return wcn, gram, degree, strength, gram.freq


def check_identity(wcn, gram, strength, freq, verbose: bool = True) -> None:
    """Assert s(R) = 2 f(R), the identity that ties panel B to Figure 1A.

    Asserted on the *median* of s/2f, not pointwise: book boundaries and the
    excluded self-loops put a minority of types below 2f (the printed 5th
    percentile shows how far), so the identity holds for the curve rather than
    for every node.
    """
    ratio = strength / (2.0 * freq)              # both sorted descending
    if verbose:
        print(f"nodes      {int(wcn['n_nodes']):,}   "
              f"(1-gram vocabulary {gram.vocab:,})")
        print(f"edges      {int(wcn['n_edges']):,}")
        print(f"sum s      {strength.sum():,}   vs 2T = {2 * gram.total_tokens:,}")
        print(f"s / 2f  :  median {np.median(ratio):.4f}, 5th-95th pct "
              f"[{np.percentile(ratio, 5):.3f}, {np.percentile(ratio, 95):.3f}]")
    assert int(wcn["n_nodes"]) == gram.vocab, "network nodes must be the vocabulary"
    assert abs(np.median(ratio) - 1.0) < 1e-9, "strength must equal twice the frequency"


def last_rank_above_one(values, vmin=2) -> int:
    """Rank of the last entry >= vmin, i.e. the start of the trailing plateau."""
    return int(np.searchsorted(-np.asarray(values), -vmin, side="right"))


def exponents(degree, strength, freq) -> tuple[pd.DataFrame, dict]:
    """Fit every curve of the figure in two regimes; return the table and the pdfs.

    Rank curves use the dense two-regime OLS of Figure 1A, so they stay directly
    comparable with it. The 2-gram curve exists only as a frequency-of-frequency
    staircase (16.5 M distinct 2-grams compressed into 6.6 k blocks) and is
    fitted on a log grid instead. The log-binned densities span only a few
    decades, so sliding the window ends would run off the data: their band is the
    spread over where the two regimes are split.
    """
    rank_2, freq_2 = load_ngram_rank_freq(LANG, 2)
    rows = []

    # min value kept in the tail window: 2 excludes the trailing plateau, and
    # since s = 2f the plateau of the strength curve sits at s = 2, not s = 1
    for name, values, vmin in (("degree k(R)", degree, 2),
                               ("strength s(R)", strength, 4),
                               ("1-gram f(R)", freq, 2)):
        o = two_regime_ols_band(values, tail_min_count=vmin)
        rows.append({"curve": name, "estimator": "dense OLS",
                     "alpha_head": round(o["alpha1"], 2),
                     "head_lo": round(o["alpha1_band"]["lo"], 2),
                     "head_hi": round(o["alpha1_band"]["hi"], 2),
                     "R2_head": round(o["alpha1_r2"], 3),
                     "alpha_tail": round(o["alpha2"], 2),
                     "tail_lo": round(o["alpha2_band"]["lo"], 2),
                     "tail_hi": round(o["alpha2_band"]["hi"], 2),
                     "R2_tail": round(o["alpha2_r2"], 3),
                     "tail_window_end": last_rank_above_one(values, vmin)})

    rmax2 = int(rank_2[freq_2 >= 2][-1])
    head = ols_rank_prob_band(rank_2.astype(float), freq_2.astype(float), *HEAD)
    tail = ols_rank_prob_band(rank_2.astype(float), freq_2.astype(float), 1e4, rmax2)
    rows.append({"curve": "2-gram f(R)", "estimator": "log-grid OLS",
                 "alpha_head": round(head["alpha"], 2),
                 "head_lo": round(head["alpha_band"]["lo"], 2),
                 "head_hi": round(head["alpha_band"]["hi"], 2),
                 "R2_head": round(head["r2"], 3),
                 "alpha_tail": round(tail["alpha"], 2),
                 "tail_lo": round(tail["alpha_band"]["lo"], 2),
                 "tail_hi": round(tail["alpha_band"]["hi"], 2),
                 "R2_tail": round(tail["r2"], 3),
                 "tail_window_end": rmax2})

    pdfs = {"p(k)": logbin_pdf(degree, BINS), "p(s)": logbin_pdf(strength, BINS)}
    for name, b in pdfs.items():
        t = logbin_split_band(b["centre"], b["density"], split=SPLIT)
        rows.append({"curve": name, "estimator": "log-binned OLS",
                     "alpha_head": round(t["alpha_low"], 2),
                     "head_lo": round(t["alpha_low_band"]["lo"], 2),
                     "head_hi": round(t["alpha_low_band"]["hi"], 2),
                     "R2_head": round(t["alpha_low_r2"], 3),
                     "alpha_tail": round(t["alpha_high"], 2),
                     "tail_lo": round(t["alpha_high_band"]["lo"], 2),
                     "tail_hi": round(t["alpha_high_band"]["hi"], 2),
                     "R2_tail": round(t["alpha_high_r2"], 3),
                     "tail_window_end": int(b["centre"][-1])})

    table = pd.DataFrame(rows).set_index("curve")

    # the 1-gram row must be Figure 1A's English fit, on the same corpus
    assert (table.loc["1-gram f(R)", "alpha_head"],
            table.loc["1-gram f(R)", "alpha_tail"]) == (1.07, 1.99), "must match Fig 1A"
    assert abs(table.loc["strength s(R)", "alpha_tail"]
               - table.loc["1-gram f(R)", "alpha_tail"]) <= 0.02, \
        "s = 2f must hold in the tail too"
    return table, pdfs


def write_table(table: pd.DataFrame) -> str:
    """Write the exponent table as CSV and as the markdown of SI Table S6."""
    table.to_csv(P.table_path("fig2_exponents.csv"))

    head_md = ("| curve | head exponent | $R^2$ | tail exponent | $R^2$ |\n"
               "| --- | ---: | ---: | ---: | ---: |\n")
    body = "".join(f"| {i} | {r.alpha_head:.2f} [{r.head_lo:.2f}–{r.head_hi:.2f}] | "
                   f"{r.R2_head:.3f} | "
                   f"{r.alpha_tail:.2f} [{r.tail_lo:.2f}–{r.tail_hi:.2f}] | "
                   f"{r.R2_tail:.3f} |\n"
                   for i, r in table.iterrows())
    note = ("\nEnglish SPGC word co-occurrence network. Every exponent is reported as "
            "*central value* [*fit-window band*]: the band is the spread over a set "
            "of a-priori reasonable fit windows, which is the honest error bar for "
            "smooth log-log curves (the nominal OLS standard error assumes "
            "independent residuals and is ~100x smaller).\n\n"
            "Rank curves: head $R\\in[1,10^3]$, tail $R\\ge10^4$ up to the start of the "
            "value-1 plateau; bands over head windows $\\{[1,500],[1,10^3],[1,2\\cdot10^3]\\}$ "
            "and tail starts $\\{5\\cdot10^3,10^4,2\\cdot10^4\\}$ (dense OLS) or over "
            "$\\pm0.5$ decades of the window ends (log-grid OLS, 2-grams). Densities "
            "$p(k)$, $p(s)$: log-binned, split at $q=10^3$, band over splits "
            "$\\{3\\cdot10^2,10^3,3\\cdot10^3\\}$.\n")
    md = head_md + body + note
    with open(P.table_path("fig2_exponents.md"), "w") as fh:
        fh.write(md)
    return md


def draw(degree, strength, freq, pdfs):
    """Draw the three panels and return the figure."""
    LW = P.LW["curve"]
    CK, CS = P.QUANTITY_COLORS["degree"], P.QUANTITY_COLORS["strength"]

    rank_1, freq_1 = np.arange(1, freq.size + 1), freq
    rank_2, freq_2 = load_ngram_rank_freq(LANG, 2)

    # square panels, the shape of Figure 4: the two 1x3 figures of the paper sit
    # three pages apart and are read against each other
    fig, axes = plt.subplots(1, 3, figsize=P.figsize(aspect=1.0, n_cols=3),
                             layout="constrained")

    # --- A: log-binned densities ------------------------------------------- #
    axA = axes[0]
    axA.plot(pdfs["p(k)"]["centre"], pdfs["p(k)"]["density"], "-", color=CK,
             lw=LW, label=r"degree $k$")
    axA.plot(pdfs["p(s)"]["centre"], pdfs["p(s)"]["density"], "-", color=CS,
             lw=LW, label=r"strength $s$")
    axA.set(xscale="log", yscale="log", xlabel=r"quantity $q$",
            ylabel=r"probability $p(q)$")
    # the two densities run into the lower-left corner where the legend sits, so
    # the floor is dropped by a decade to open the strip the legend needs, and
    # the legend itself is pulled down into it -- at the default pad it grazed
    # the curve
    axA.set_ylim(3e-14, 3e0)
    axA.legend(fontsize=P.FONT["legend"], frameon=False, loc="lower left",
               borderaxespad=0.15, labelspacing=0.3)

    # --- B: rank-size of degree and strength -------------------------------- #
    # one point per constant run, as in Figure 1A: these are integer counts, so
    # the tail is a staircase and sampling it log-uniformly draws the treads
    axB = axes[1]
    for values, colour, label in ((degree, CK, r"degree $k$"),
                                  (strength, CS, r"strength $s$")):
        x, y = P.plateau_points(np.arange(1, values.size + 1), values)
        i = P.log_indices(x.size, 300)
        axB.plot(x[i], y[i], "-", color=colour, lw=LW, label=label)
    axB.set(xscale="log", yscale="log", xlabel=r"rank $R$",
            ylabel=r"quantity $q(R)$")
    axB.set_ylim(0.5, 2e8)
    axB.legend(fontsize=P.FONT["legend"], frameon=False, loc="lower left",
               borderaxespad=0.15, labelspacing=0.3)

    # --- C: rank-frequency of 1-grams and 2-grams --------------------------- #
    axC = axes[2]
    for rank, f, key, label in ((rank_2, freq_2, "2gram", "2-gram"),
                                (rank_1, freq_1, "1gram", "1-gram")):
        x, y = P.plateau_points(rank, f)
        i = P.log_indices(x.size, 300)
        axC.plot(x[i], y[i], "-", color=P.QUANTITY_COLORS[key], lw=LW, label=label)
    axC.set(xscale="log", yscale="log", xlabel=r"rank $R$",
            ylabel=r"frequency $f(R)$")
    axC.set_ylim(0.5, 4e7)
    axC.legend(fontsize=P.FONT["legend"], frameon=False, loc="lower left",
               borderaxespad=0.15, labelspacing=0.3)

    for ax, letter in zip(axes, "ABC"):
        P.panel_label(ax, letter)

    return fig


def main() -> int:
    P.setup_style()
    wcn, gram, degree, strength, freq = load()
    check_identity(wcn, gram, strength, freq)
    table, pdfs = exponents(degree, strength, freq)
    print()
    print(table.to_string())
    write_table(table)
    fig = draw(degree, strength, freq, pdfs)
    for path in P.save_figure(fig, "fig2"):
        print("wrote", os.path.relpath(path, REPO))
    return 0


if __name__ == "__main__":
    sys.exit(main())
