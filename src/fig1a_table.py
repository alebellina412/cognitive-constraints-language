#!/usr/bin/env python3
"""The Figure 1 exponent table, from the reduced 1-gram data — SI Table S2.

Prints the table and writes it to outputs/tables/fig1A_exponents.{csv,md},
which is Table S2 of the supplementary information.

For each language:
  * two-regime rank-frequency OLS  ->  alpha1 (head, R in [1,1e3])
                                       alpha2 (tail, R >= 1e4, count >= 2)
  * frequency-spectrum MLE with x_min = 2 (only the hapax dropped) ->
    alpha = 1/(beta-1), reported with its statistical error and its x_min
    sensitivity band (x_min in 2..5).
  * KS-optimal Clauset MLE, shown for contrast: it fits the most frequent words,
    i.e. the HEAD of the Zipf plot, and therefore lands near alpha ~ 1.

Usage:
    python src/fig1a_table.py [--xmin 2]
"""

from __future__ import annotations

import argparse

import pandas as pd

import plotting as P
from fits import (clauset_discrete_mle, two_regime_ols, two_regime_ols_band,
                  zipf_mle_spectrum_band)
from io_reduced import load_1gram

# Regression check: the (alpha1, alpha2) central values published in Table S2 of
# the supplementary information. Recomputing them from the reduced data must
# return these numbers, so a change in the estimators or in the reduction shows
# up here rather than silently in a figure.
REF = {
    "en": (1.07, 1.99),
    "fr": (1.07, 1.98),
    "es": (1.03, 1.78),
    "it": (1.02, 1.71),
    "de": (1.07, 1.48),
}
#: row order of Table S2, and of every other five-language table
ORDER = P.LEGEND_ORDER

XMIN = 2                       # reported spectrum-MLE cutoff (only hapax dropped)
XMIN_BAND = (2, 3, 4, 5)       # sensitivity range reported alongside it


def build(xmin: int = XMIN) -> pd.DataFrame:
    """Every exponent of the 1-gram curves, with its fit-window band."""
    rows = []
    for c in ORDER:
        g = load_1gram(c)
        ols = two_regime_ols_band(g.freq)                    # head + tail + bands
        mle = zipf_mle_spectrum_band(g.freq, xmin=xmin, xmins=XMIN_BAND)
        ks = clauset_discrete_mle(g.freq)                    # KS-optimal (head)
        rows.append({
            "language": P.LANG_NAMES[c],
            "vocab": ols["vocab"],
            "alpha1": round(ols["alpha1"], 2),
            "alpha1_lo": round(ols["alpha1_band"]["lo"], 2),
            "alpha1_hi": round(ols["alpha1_band"]["hi"], 2),
            "alpha1_se": round(ols["alpha1_se"], 4),
            "R2_head": round(ols["alpha1_r2"], 3),
            "alpha2": round(ols["alpha2"], 2),
            "alpha2_lo": round(ols["alpha2_band"]["lo"], 2),
            "alpha2_hi": round(ols["alpha2_band"]["hi"], 2),
            "alpha2_se": round(ols["alpha2_se"], 4),
            "R2_tail": round(ols["alpha2_r2"], 3),
            "alpha_mle": round(mle["alpha"], 2),
            "sigma_stat": round(mle["sigma_stat"], 3),
            "alpha_mle_lo": round(mle["alpha_min"], 2),
            "alpha_mle_hi": round(mle["alpha_max"], 2),
            "alpha_head_KS": round(ks["alpha"], 2),
            "xmin_KS": ks["xmin"],
        })
    table = pd.DataFrame(rows, index=ORDER)

    for c, (a1, a2) in REF.items():
        got = (table.loc[c, "alpha1"], table.loc[c, "alpha2"])
        assert got == (a1, a2), f"{c}: got {got}, Table S2 reports {(a1, a2)}"
    return table


def write_table(table: pd.DataFrame, xmin: int = XMIN) -> str:
    """Write the exponent table as CSV and as the markdown of SI Table S2."""
    table.to_csv(P.table_path("fig1A_exponents.csv"), index_label="lang")

    head = ("| language | vocab | $\\alpha_1$ (head) | $\\alpha_2$ (tail, OLS) | "
            "$\\alpha$ (spectrum MLE) |\n"
            "| --- | ---: | ---: | ---: | ---: |\n")
    body = "".join(
        f"| {r.language} | {r.vocab:,} | "
        f"{r.alpha1:.2f} [{r.alpha1_lo:.2f}–{r.alpha1_hi:.2f}] | "
        f"{r.alpha2:.2f} [{r.alpha2_lo:.2f}–{r.alpha2_hi:.2f}] | "
        f"{r.alpha_mle:.2f} [{r.alpha_mle_lo:.2f}–{r.alpha_mle_hi:.2f}] |\n"
        for r in table.itertuples())
    note = (
        f"\nEvery exponent is reported as *central value* [*fit-window band*]. "
        f"$\\alpha_1$: OLS over the head window $R\\in[1,10^3]$, band over "
        f"$R\\in\\{{[1,500],[1,10^3],[1,2\\cdot10^3]\\}}$. $\\alpha_2$: OLS over the "
        f"tail $R\\ge10^4$ with count $\\ge2$, band over tail starts "
        f"$\\{{5\\cdot10^3,10^4,2\\cdot10^4\\}}$. Spectrum MLE at $x_{{\\min}}={xmin}$ "
        f"(hapax dropped), $\\alpha=1/(\\beta-1)$, band over "
        f"$x_{{\\min}}\\in\\{{2,3,4,5\\}}$.\n\n"
        f"The band, not the nominal OLS standard error, is the honest error bar: "
        f"rank-frequency residuals are strongly autocorrelated, so the textbook "
        f"$se$ (English $\\alpha_2$: {table.loc['en', 'alpha2_se']:.4f}) "
        f"understates the window sensitivity by two orders of magnitude. The "
        f"nominal $se$ of each exponent is in the CSV.\n")
    md = head + body + note
    with open(P.table_path("fig1A_exponents.md"), "w") as fh:
        fh.write(md)
    return md


def main(xmin_spectrum: int = 2) -> None:
    print("            OLS two-regime |     spectrum-MLE (tail)      | KS-Clauset | ref")
    print("lang   vocab     a1     a2 |  aMLE +- stat   [xmin 2..5]  |  (head)    | a1 / a2")
    print("-" * 88)
    for c in ORDER:
        g = load_1gram(c)
        o = two_regime_ols(g.freq)
        s = zipf_mle_spectrum_band(g.freq, xmin=xmin_spectrum)
        m = clauset_discrete_mle(g.freq)
        r = REF[c]
        print(f"{c:4} {o['vocab']:>9,}  {o['alpha1']:5.2f}  {o['alpha2']:5.2f} | "
              f"{s['alpha']:5.2f} +- {s['sigma_stat']:.3f}  "
              f"[{s['alpha_min']:.2f}, {s['alpha_max']:.2f}] |   {m['alpha']:5.2f}    | "
              f"{r[0]:.2f} / {r[1]:.2f}")
    print(f"\nspectrum-MLE reported at x_min={xmin_spectrum} (hapax dropped), "
          f"alpha = 1/(beta-1).")
    print("The x_min band, not the sampling error, dominates the uncertainty;")
    print("the OLS tail (a2) and the spectrum-MLE agree within it.")

    write_table(build(xmin_spectrum), xmin_spectrum)
    print("\nwrote outputs/tables/fig1A_exponents.{csv,md}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xmin", type=int, default=2,
                    help="reported lower cutoff of the spectrum MLE (default 2)")
    main(ap.parse_args().xmin)
