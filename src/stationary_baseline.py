#!/usr/bin/env python3
"""Can a *stationary* urn model produce the two-regime curve? No — this shows it.

The paper argues that the crossover between the two Zipf regimes requires the
attenuation of novelty above a finite vocabulary D0. Stationary processes
(Yule-Simon, the standard UMT, Hoppe) are known not to produce a persistent
crossover; this script measures that directly rather than resting on the
literature alone.

The baseline is free: the same binary with D0 set far above any vocabulary the
run can reach switches attenuation off, so the comparison isolates exactly one
ingredient. Nothing else changes — same code, same seed, same T, same reductions,
same estimators.

Three UMT settings are compared, because one is not a fair test. With rho = nu
the process gives the head exponent and no tail; with rho = 2 it gives the tail
exponent everywhere, head included; and with a large initial urn it produces an
apparent two-regime curve out of a transient. The third is the strongest case the
baseline has, and `src/umt_transient_n0.py` tunes its n0 by the same crossover
criterion used on the model itself, so it is met at its best rather than at its
most convenient.

Prerequisites (all produced by `src/build_reduced_sim.py`):

    python src/build_reduced_sim.py --run --d0 9366       --T 100000000 --name sim_calibrated
    python src/build_reduced_sim.py --run --d0 1000000000 --T 100000000 --name sim_stationary
    python src/build_reduced_sim.py --run --d0 1000000000 --T 100000000 --rho 2 --name sim_umt_rho2
    python src/umt_transient_n0.py                        # selects n0, then:
    python src/build_reduced_sim.py --run --d0 1000000000 --T 100000000 --rho 2 --N0 <n0> --name sim_umt_transient

Usage:
    python src/stationary_baseline.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import estimate_d0 as E                                   # noqa: E402
import plotting as P                                      # noqa: E402
from fits import (heaps_exponent_band, ols_rank_prob_band,   # noqa: E402
                  two_regime_ols_band)  # noqa: E402
from io_reduced import load_heaps, load_wcn, sim_freq_from_wcn  # noqa: E402

#: (reduced name, label, regime).
#:
#: `regime` picks the estimator, and the choice is forced by the curve rather
#: than by preference. A stationary UMT has one scaling regime, so a two-regime
#: fit would report the difference between two windows cut out of one straight
#: line; those rows get a single power law over the whole usable range. Only the
#: curves that do change slope -- the data, the attenuated model, and the UMT run
#: with a large initial urn -- get alpha1 and alpha2.
CURVES = [
    ("en", "data (SPGC English, $10^8$)", "two"),
    ("sim_calibrated", r"model, attenuated ($D_0=9{,}366$)", "two"),
    ("sim_stationary", r"UMT, $\rho=\nu=1$, $n_0=1$", "one"),
    ("sim_umt_rho2", r"UMT, $\rho=2$, $\nu=1$, $n_0=1$", "one"),
    ("sim_umt_transient", r"UMT, $\rho=2$, $\nu=1$, $n_0$ tuned", "two"),
]

#: First rank of the single-power-law window. The last is the final rank whose
#: type occurs more than once, which is where the hapax plateau starts: the same
#: convention the phrase and MWE tables use.
SINGLE_FIT_RMIN = 10.0


def _freq_and_heaps(name: str):
    """Frequency vector and growth curve, from the corpus or from a simulation."""
    heaps = load_heaps(name)
    if name in ("en", "fr", "it", "es", "de"):
        return np.sort(heaps["freq_matched"])[::-1].astype(np.int64), heaps
    return sim_freq_from_wcn(load_wcn(name)), heaps


def _single_power_law(freq: np.ndarray) -> float:
    """One exponent over the whole usable range, for a curve with one regime."""
    rmax = float(max(int(np.count_nonzero(freq >= 2)), 100))
    rank = np.arange(1, freq.size + 1, dtype=float)
    return ols_rank_prob_band(rank, freq / freq.sum(), SINGLE_FIT_RMIN, rmax)["alpha"]


def table() -> pd.DataFrame:
    rows = []
    for name, label, regime in CURVES:
        freq, heaps = _freq_and_heaps(name)
        b = heaps_exponent_band(heaps["heaps_t"], heaps["heaps_mean"], 1e4, 1e8)
        r = E.estimate(freq)
        if regime == "two":
            z = two_regime_ols_band(freq)
            a1, a2, alpha = round(z["alpha1"], 2), round(z["alpha2"], 2), np.nan
        else:
            a1 = a2 = np.nan
            alpha = round(_single_power_law(freq), 2)
        rows.append({
            "curve": label,
            "regime": regime,
            "types D(T)": int(freq.size),
            "alpha": alpha,
            "alpha1": a1,
            "alpha2": a2,
            "heaps b": round(b["b"], 3),
            "R_star": round(r["R_star"]),
            # the size of the kink the free-breakpoint fit finds: on a straight
            # line the estimator still returns a breakpoint, but no slope change
            "bpl slope change": round(r["alpha2"] - r["alpha1"], 2),
        })
    return pd.DataFrame(rows).set_index("curve")


def _cells(row) -> str:
    """The exponent and crossover columns.

    A single-regime row carries its one exponent under alpha1 and leaves alpha2
    empty, and carries no crossover at all: the free-breakpoint estimator returns
    a value for any curve, straight lines included, so quoting one here would
    invite it to be read as a scale the process does not have. Both omitted
    values are kept in the CSV.
    """
    if row["regime"] == "two":
        return (f"{row.alpha1:.2f} | {row.alpha2:.2f} | "
                f"{row['heaps b']:.3f} | {int(row.R_star):,}")
    return (f"{row.alpha:.2f} | -- | {row['heaps b']:.3f} | --")


def main() -> int:
    t = table()
    t.to_csv(P.table_path("stationary_baseline.csv"))

    data, att = t.iloc[0], t.iloc[1]
    umt, rho2, transient = t.iloc[2], t.iloc[3], t.iloc[4]
    with open(P.table_path("stationary_baseline.md"), "w") as fh:
        fh.write("| curve | types $D(T)$ | $\\alpha_1$ | $\\alpha_2$ | "
                 "Heaps $b$ | $R^*$ |\n"
                 "| --- | ---: | ---: | ---: | ---: | ---: |\n")
        for label, r in t.iterrows():
            fh.write(f"| {label} | {int(r['types D(T)']):,} | {_cells(r)} |\n")
        fh.write(
            "\nAll rows at $T=10^8$ tokens, reduced and fitted identically. Every "
            "UMT row is the same simulator with $D_0$ set above any reachable "
            "vocabulary, so attenuation never switches on and only $n_0$, $\\rho$ "
            "and $\\nu$ differ. A curve with one scaling regime is fitted with one "
            "power law; only the curves that change slope carry $\\alpha_1$ and "
            "$\\alpha_2$.\n\n"
            "**No setting of the stationary model reproduces the pair of "
            f"exponents.** At $\\rho=\\nu=1$ the curve is a single power law of "
            f"{umt.alpha:.2f}: it has the head and not the tail, and the "
            f"vocabulary runs away to {int(umt['types D(T)']):,} types against "
            f"{int(data['types D(T)']):,} observed. At $\\rho=2$ the asymptotic "
            f"exponent is right but it holds everywhere, a single power law of "
            f"{rho2.alpha:.2f}: the tail and not the head. Neither row is given a "
            "crossover: refitted with the two-regime estimator as a check they "
            "return $\\alpha_2-\\alpha_1=0.04$ and an empty tail window "
            "respectively, which is what one regime looks like. (That 0.04 is "
            "the two-regime OLS of Table S2, not the `bpl slope change` column "
            "of the CSV, which is the kink of the free-breakpoint fit and is a "
            "different quantity.)\n\n"
            "**The transient is the best the baseline can do, and it still fails "
            "on the head.** With a large initial urn the first phase draws a "
            "novelty almost every step, so $D(t)\\sim t$ until the reinforced mass "
            "overtakes $n_0$; tuning $n_0$ by the crossover criterion "
            "(`src/umt_transient_n0.py`) puts the apparent knee at "
            f"$R^*={int(transient.R_star):,}$, against {int(data.R_star):,} "
            f"measured, and gives Heaps $b={transient['heaps b']:.3f}$ against "
            f"{data['heaps b']:.3f}. The tail follows at "
            f"$\\alpha_2={transient.alpha2:.2f}$. The head does not: "
            f"$\\alpha_1={transient.alpha1:.2f}$ against {data.alpha1:.2f} in the "
            f"data and {att.alpha1:.2f} in the attenuated model. A transient is a "
            "passage between regimes, not a regime, and the head exponent is where "
            "that shows.\n")

    print(t.to_string())
    print("\nwrote outputs/tables/stationary_baseline.{csv,md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
