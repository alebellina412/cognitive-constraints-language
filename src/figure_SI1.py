#!/usr/bin/env python3
"""Appendix figure S1 — the network views of Figure 2, across all five languages.

Three panels, each holding all five corpora:

  (A) log-binned probability density of the node degree k;
  (B) the same for the node strength s;
  (C) the 2-gram rank-frequency curve, in relative frequency.

Panel C is normalised because the five corpora differ in size by a factor of
five: raw counts would separate the curves by corpus size rather than by shape,
which is the only thing this figure is about.

The five curves lie almost on top of one another, and that is the result. At the
shared curve width they merge into a single band, so this figure draws them
thinner than every other one.

No guide lines are drawn. Every exponent here is tabulated with its band by
`write_table` below, and dashed references were being read as fits of individual
curves.

Because a node's strength is twice its 1-gram frequency — exactly, except at
book boundaries and for the excluded self-loops — the s(R) columns of the table
must reproduce the Figure 1A exponents language by language; `exponents`
asserts it, and they do to the two decimals reported.

Usage:
    python src/figure_SI1.py
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt                                    # noqa: E402

import plotting as P                                               # noqa: E402
from fits import (logbin_pdf, logbin_split_band, ols_rank_prob_band,  # noqa: E402
                  two_regime_ols_band)
from io_reduced import load_ngram_rank_freq, load_wcn              # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LANGS = P.LEGEND_ORDER
BINS = 24
SPLIT = 1e3            # split of the log-binned densities into two regimes

QUANTITIES = ["p(k) low", "p(k) high", "p(s) low", "p(s) high",
              "k(R) head", "k(R) tail", "s(R) head", "s(R) tail",
              "2-gram head", "2-gram tail"]

#: Figure 1A exponents, language by language: s(R) must reproduce them
REF_1A = {"en": (1.07, 1.99), "fr": (1.07, 1.98), "it": (1.02, 1.71),
          "es": (1.03, 1.78), "de": (1.07, 1.48)}


def load():
    """Networks, log-binned densities and 2-gram curves for the five languages."""
    wcn = {c: load_wcn(c) for c in LANGS}
    pdf_k = {c: logbin_pdf(wcn[c]["degree"], BINS) for c in LANGS}
    pdf_s = {c: logbin_pdf(wcn[c]["strength"], BINS) for c in LANGS}
    ngram2 = {c: load_ngram_rank_freq(c, 2) for c in LANGS}
    return wcn, pdf_k, pdf_s, ngram2


def summary(wcn, ngram2) -> pd.DataFrame:
    """Size of each network, as a sanity view before the fits."""
    return pd.DataFrame(
        [{"language": P.LANG_NAMES[c],
          "nodes": int(wcn[c]["n_nodes"]),
          "edges": int(wcn[c]["n_edges"]),
          "max degree": int(wcn[c]["degree"][0]),
          "max strength": int(wcn[c]["strength"][0]),
          "distinct 2-grams": int(ngram2[c][0][-1])}
         for c in LANGS], index=LANGS)


def exponents(wcn, pdf_k, pdf_s, ngram2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit all ten quantities in all five languages; return centrals and bands."""
    rows, bands = [], []
    for c in LANGS:
        ok = two_regime_ols_band(wcn[c]["degree"], tail_min_count=2)
        # s = 2f, so the strength plateau sits at s = 2
        os_ = two_regime_ols_band(wcn[c]["strength"], tail_min_count=4)
        tk = logbin_split_band(pdf_k[c]["centre"], pdf_k[c]["density"], split=SPLIT)
        ts = logbin_split_band(pdf_s[c]["centre"], pdf_s[c]["density"], split=SPLIT)
        r2, f2 = ngram2[c]
        rmax2 = int(r2[f2 >= 2][-1])
        g2_lo = ols_rank_prob_band(r2.astype(float), f2.astype(float), 1e0, 1e3)
        g2_hi = ols_rank_prob_band(r2.astype(float), f2.astype(float), 1e4, rmax2)

        central = {"p(k) low": tk["alpha_low"], "p(k) high": tk["alpha_high"],
                   "p(s) low": ts["alpha_low"], "p(s) high": ts["alpha_high"],
                   "k(R) head": ok["alpha1"], "k(R) tail": ok["alpha2"],
                   "s(R) head": os_["alpha1"], "s(R) tail": os_["alpha2"],
                   "2-gram head": g2_lo["alpha"], "2-gram tail": g2_hi["alpha"]}
        band = {"p(k) low": tk["alpha_low_band"], "p(k) high": tk["alpha_high_band"],
                "p(s) low": ts["alpha_low_band"], "p(s) high": ts["alpha_high_band"],
                "k(R) head": ok["alpha1_band"], "k(R) tail": ok["alpha2_band"],
                "s(R) head": os_["alpha1_band"], "s(R) tail": os_["alpha2_band"],
                "2-gram head": g2_lo["alpha_band"], "2-gram tail": g2_hi["alpha_band"]}

        rows.append({"language": P.LANG_NAMES[c],
                     **{q: round(central[q], 2) for q in QUANTITIES}})
        bands.append({f"{q} {end}": round(band[q][end], 2)
                      for q in QUANTITIES for end in ("lo", "hi")})

    table = pd.DataFrame(rows, index=LANGS)
    band_table = pd.DataFrame(bands, index=LANGS)

    for c, (a1, a2) in REF_1A.items():
        got = (table.loc[c, "s(R) head"], table.loc[c, "s(R) tail"])
        assert got == (a1, a2), f"{c}: strength gives {got}, Figure 1A gives {(a1, a2)}"
    return table, band_table


def formatted(table, band_table) -> pd.DataFrame:
    """Central value with its band, one row per quantity, one column per language."""
    return pd.DataFrame(
        {c: [f"{table.loc[c, q]:.2f} "
             f"[{band_table.loc[c, q + ' lo']:.2f}-{band_table.loc[c, q + ' hi']:.2f}]"
             for q in QUANTITIES] for c in LANGS}, index=QUANTITIES)


def write_table(table, band_table) -> str:
    """Write the exponent table as CSV and as the markdown of SI Table S6."""
    table.join(band_table).to_csv(P.table_path("SI1_exponents.csv"),
                                  index_label="lang")

    # transposed for the SI: one row per quantity, one column per language, so
    # that each cell can carry its band and the table stays readable
    head_md = ("| quantity | " + " | ".join(P.LANG_NAMES[c] for c in LANGS) + " |\n"
               + "| --- | " + " | ".join("---:" for _ in LANGS) + " |\n")
    body = "".join(
        "| " + q + " | " + " | ".join(
            f"{table.loc[c, q]:.2f} [{band_table.loc[c, q + ' lo']:.2f}–"
            f"{band_table.loc[c, q + ' hi']:.2f}]" for c in LANGS) + " |\n"
        for q in QUANTITIES)
    note = ("\nEvery exponent is *central value* [*fit-window band*]. Densities "
            "$p(k)$, $p(s)$: log-binned, split at $q=10^3$, band over "
            "splits $\\{3\\cdot10^2,10^3,3\\cdot10^3\\}$. Rank curves $k(R)$, $s(R)$: dense "
            "two-regime OLS, head $R\\in[1,10^3]$, tail $R\\ge10^4$ above the trailing "
            "plateau; bands over head windows $\\{[1,500],[1,10^3],[1,2\\cdot10^3]\\}$ and "
            "tail starts $\\{5\\cdot10^3,10^4,2\\cdot10^4\\}$. 2-gram $f(R)$: log-grid OLS "
            "over the same windows, band over $\\pm0.5$ decades of the window ends.\n\n"
            "$s(R)$ is twice the word frequency, exactly except at book boundaries "
            "and for the excluded self-loops, so its two columns reproduce the "
            "Figure 1A exponents language by language (asserted above).\n")
    md = head_md + body + note
    with open(P.table_path("SI1_exponents.md"), "w") as fh:
        fh.write(md)
    return md


def draw(pdf_k, pdf_s, ngram2):
    """Draw the three panels and return the figure."""
    LW = P.LW["curve"] * 0.62
    fig, axes = plt.subplots(1, 3, figsize=P.figsize(aspect=1.00, n_cols=3))
    axA, axB, axC = axes

    tot2 = {c: float(ngram2[c][1].sum()) for c in LANGS}

    for c in P.PLOT_ORDER:
        col = P.LANG_COLORS[c]
        axA.plot(pdf_k[c]["centre"], pdf_k[c]["density"], "-", color=col, lw=LW)
        axB.plot(pdf_s[c]["centre"], pdf_s[c]["density"], "-", color=col, lw=LW)
        r2, f2 = ngram2[c]
        i2 = P.log_indices(r2.size, 300)
        axC.plot(r2[i2], f2[i2] / tot2[c], "-", color=col, lw=LW)

    axA.set(xscale="log", yscale="log", xlabel=r"degree $k$",
            ylabel=r"probability $p(k)$")
    axB.set(xscale="log", yscale="log", xlabel=r"strength $s$",
            ylabel=r"probability $p(s)$")
    axC.set(xscale="log", yscale="log", xlabel=r"rank $R$",
            ylabel=r"probability $p(R)$")

    for ax, letter in zip(axes, "ABC"):
        P.panel_label(ax, letter)

    fig.legend(handles=P.language_handles(), loc="outside upper center", ncol=5,
               frameon=False, handlelength=1.4, columnspacing=1.4)
    return fig


def main() -> int:
    P.setup_style()
    wcn, pdf_k, pdf_s, ngram2 = load()
    print(summary(wcn, ngram2).to_string())
    table, band_table = exponents(wcn, pdf_k, pdf_s, ngram2)
    print()
    print(formatted(table, band_table).to_string())
    write_table(table, band_table)
    fig = draw(pdf_k, pdf_s, ngram2)
    for path in P.save_figure(fig, "SI1"):
        print("wrote", os.path.relpath(path, REPO))
    return 0


if __name__ == "__main__":
    sys.exit(main())
