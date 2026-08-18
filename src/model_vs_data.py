#!/usr/bin/env python3
"""The calibrated model against the corpus, band for band.

Both sides are taken at the same T = 10^8 tokens, reduced by the same code to
the same on-disk schema, and fitted with the same estimators — so the comparison
is like-for-like rather than two similar-looking pipelines. Four quantities are
compared, each with its fit-window band:

    types D(T)        vocabulary reached at T
    alpha1, alpha2    the two Zipf regimes
    Heaps b           vocabulary growth above t = 10^4
    R*                the crossover, from a free-breakpoint fit

The model reproduces the two-regime shape and the scale at which the regimes
meet. It does not reproduce the corpus vocabulary exactly, which is expected:
D0 fixes where attenuation sets in, not how many hapax legomena a real corpus
accumulates. The one quantity whose bands do not overlap is the Heaps exponent,
and the paper states that rather than hiding it.

The model parameter D0 = 9,366 is not tuned here: it is selected by
`calibrate_d0.py`, and it agrees to 0.2% with the R* measured from the data
alone. `flatness` reports how much the joint loss rises when D0 is halved or
doubled, which is what makes that agreement meaningful rather than lucky.

Prerequisites:
    python src/build_reduced_sim.py --run --d0 9366 --T 100000000 --name sim_calibrated
    python src/calibrate_d0.py --config manifests/d0_calibration.json \
        --tag crossover --T 100000000 --allow-long-run

Usage:
    python src/model_vs_data.py
"""

from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import estimate_d0 as D0                                           # noqa: E402
import plotting as P                                               # noqa: E402
from fits import heaps_exponent_band, two_regime_ols_band          # noqa: E402
from io_reduced import load_heaps, load_wcn, sim_freq_from_wcn     # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SIM = "sim_calibrated"
T_TOKENS = 100_000_000

#: The **wide** sweep (`manifests/d0_calibration.json`, `--tag crossover`), not
#: the deep one. `flatness` below needs the loss over a grid that spans a factor
#: of 130 in D0; the refined grid is narrowed onto the minimum and cannot say
#: what halving or doubling D0 costs. `estimate_d0.py` reads the deep one
#: instead, for the opposite reason — it needs six seeds, not a wide grid.
#: Neither sweep is a coarse version of the other; see src/calibrate_d0.py.
CALIBRATION = os.path.join(REPO, "outputs", "d0_calibration_crossover")


def load():
    """The simulated run and the corpus, both truncated to the same T."""
    meta = json.load(open(os.path.join(REPO, "data_reduced", f"{SIM}_meta.json")))

    sim_heaps, sim_wcn = load_heaps(SIM), load_wcn(SIM)
    dat_heaps, dat_wcn = load_heaps("en"), load_wcn("en")

    # the corpus frequency vector truncated to the same 1e8 tokens as the model
    dat_freq = np.sort(dat_heaps["freq_matched"])[::-1].astype(np.int64)
    sim_freq = sim_freq_from_wcn(sim_wcn)

    assert int(sim_heaps["max_tokens"]) == int(dat_heaps["match_tokens"]) == T_TOKENS
    assert meta["D0"] == 9366 and meta["p"] == 0.5
    return meta, dat_freq, dat_heaps, dat_wcn, sim_freq, sim_heaps, sim_wcn


def sizes(meta, dat_freq, dat_heaps, dat_wcn, sim_wcn) -> pd.DataFrame:
    """Corpus and simulation size, as a sanity view before the fits."""
    return pd.DataFrame([
        {"side": "model (D0 = %s)" % f"{meta['D0']:,}", "tokens": int(meta["T"]),
         "types D(T)": int(sim_wcn["n_nodes"]), "edges": int(sim_wcn["n_edges"])},
        {"side": "data (SPGC English)", "tokens": int(dat_heaps["match_tokens"]),
         "types D(T)": int(dat_freq.size), "edges": int(dat_wcn["n_edges"])},
    ])


def describe(name, freq, heaps) -> dict:
    """Every compared quantity for one side, each with its fit-window band."""
    z = two_regime_ols_band(freq)                      # Figure 1A windows
    b = heaps_exponent_band(heaps["heaps_t"], heaps["heaps_mean"], 1e4, 1e8)
    r = D0.estimate(freq)
    return {
        "side": name,
        "types": int(freq.size),
        "alpha1": round(z["alpha1"], 2),
        "alpha1_lo": round(z["alpha1_band"]["lo"], 2),
        "alpha1_hi": round(z["alpha1_band"]["hi"], 2),
        "alpha2": round(z["alpha2"], 2),
        "alpha2_lo": round(z["alpha2_band"]["lo"], 2),
        "alpha2_hi": round(z["alpha2_band"]["hi"], 2),
        "b_heaps": round(b["b"], 3),
        "b_heaps_lo": round(b["b_band"]["lo"], 3),
        "b_heaps_hi": round(b["b_band"]["hi"], 3),
        "R_star": round(r["R_star"]),
        "R_star_lo": round(r["R_star_band"]["lo"]),
        "R_star_hi": round(r["R_star_band"]["hi"]),
    }


def compare(meta, dat_freq, dat_heaps, sim_freq, sim_heaps,
            verbose: bool = True) -> pd.DataFrame:
    """Build the two-row comparison and report which bands overlap."""
    table = pd.DataFrame([
        describe("data (SPGC English, 1e8)", dat_freq, dat_heaps),
        describe(f"model (D0 = {meta['D0']:,})", sim_freq, sim_heaps),
    ]).set_index("side")

    if verbose:
        d, m = table.iloc[0], table.iloc[1]
        for q in ("alpha1", "alpha2", "b_heaps", "R_star"):
            overlap = not (m[f"{q}_hi"] < d[f"{q}_lo"] or d[f"{q}_hi"] < m[f"{q}_lo"])
            print(f"{q:8s} data {d[q]:>9,.3g} [{d[f'{q}_lo']:,.3g}, {d[f'{q}_hi']:,.3g}]"
                  f"   model {m[q]:>9,.3g} [{m[f'{q}_lo']:,.3g}, {m[f'{q}_hi']:,.3g}]"
                  f"   bands {'overlap' if overlap else 'DISJOINT'}")
    return table


def calibration(table: pd.DataFrame, meta, verbose: bool = True) -> dict:
    """Read back the calibration and report how flat the objective is around D0."""
    report = json.load(open(os.path.join(CALIBRATION, "calibration_report.json")))
    native = next(p for p in report["populations"] if p["population"] == "native")

    runs = [r for r in csv.DictReader(open(os.path.join(CALIBRATION, "runs.csv")))
            if r["status"] == "ok" and r["population"] == "native"]
    grid = {}
    for r in runs:
        grid.setdefault(int(r["D0"]), {})[int(r["seed"])] = float(r["loss_joint"])
    d0s = np.array(sorted(grid))
    seeds = sorted(set.intersection(*(set(v) for v in grid.values())))
    losses = np.array([[grid[d][s] for d in d0s] for s in seeds])      # seeds x D0
    span = losses.mean(axis=0)
    k = int((losses - losses.mean(axis=1, keepdims=True)).mean(axis=0).argmin())

    assert int(d0s[k]) == native["best_D0"] == meta["D0"], \
        "the table must be built at the D0 the calibration selected"

    half = int(np.argmin(np.abs(d0s - d0s[k] / 2)))
    double = int(np.argmin(np.abs(d0s - d0s[k] * 2)))
    out = {"native": native, "d0s": d0s, "best_index": k,
           "rise_halved": (span[half] / span[k] - 1) * 100,
           "rise_doubled": (span[double] / span[k] - 1) * 100}

    if verbose:
        print(f"\nbest D0 = {d0s[k]:,}  competitive [{native['low']:,}, "
              f"{native['high']:,}]  ({native['n_competitive']}/{native['n_grid']} "
              f"grid points)")
        print(f"measured from the data alone: R* = "
              f"{table.loc['data (SPGC English, 1e8)', 'R_star']:,}")
        print(f"flatness: halving D0 raises the mean joint loss by "
              f"{out['rise_halved']:.0f}%, doubling it by {out['rise_doubled']:.0f}%")
    return out


def write_table(table: pd.DataFrame, meta, native) -> str:
    """Write the comparison as CSV and as the markdown of SI Table S7."""
    table.to_csv(P.table_path("model_vs_data.csv"))
    d, m = table.iloc[0], table.iloc[1]

    def cell(row, q, fmt="{:.2f}"):
        return (fmt.format(row[q]) + " [" + fmt.format(row[f"{q}_lo"]) + "–"
                + fmt.format(row[f"{q}_hi"]) + "]")

    head = ("| quantity | data (SPGC English, $10^8$) | model ($D_0=%s$) |\n"
            "| --- | ---: | ---: |\n" % f"{meta['D0']:,}")
    body = (f"| types $D(T)$ | {int(d['types']):,} | {int(m['types']):,} |\n"
            f"| $\\alpha_1$ (head) | {cell(d, 'alpha1')} | {cell(m, 'alpha1')} |\n"
            f"| $\\alpha_2$ (tail) | {cell(d, 'alpha2')} | {cell(m, 'alpha2')} |\n"
            f"| Heaps $b$ ($t\\ge10^4$) | {cell(d, 'b_heaps', '{:.3f}')} | "
            f"{cell(m, 'b_heaps', '{:.3f}')} |\n"
            f"| crossover $R^*$ | {cell(d, 'R_star', '{:,.0f}')} | "
            f"{cell(m, 'R_star', '{:,.0f}')} |\n")
    note = (f"\nModel and data at the same $T=10^8$ tokens, reduced to the same schema "
            f"and fitted with the same estimators; every entry is *central value* "
            f"[*fit-window band*], the convention of Appendix B. The model parameter is "
            f"$D_0={meta['D0']:,}$, selected by the calibration of "
            f"`src/calibrate_d0.py` (competitive range "
            f"[{native['low']:,}–{native['high']:,}]), which agrees to 0.2% with the "
            f"$R^*={int(d['R_star']):,}$ measured from the data alone.\n\n"
            f"The model reproduces the two-regime shape and the scale at which the "
            f"regimes meet. It does not reproduce the corpus vocabulary exactly "
            f"({int(m['types']):,} types against {int(d['types']):,}), which is "
            f"expected: $D_0$ fixes where attenuation sets in, not how many hapax a "
            f"real corpus accumulates.\n")
    md = head + body + note
    with open(P.table_path("model_vs_data.md"), "w") as fh:
        fh.write(md)
    return md


def main() -> int:
    meta, dat_freq, dat_heaps, dat_wcn, sim_freq, sim_heaps, sim_wcn = load()
    print(sizes(meta, dat_freq, dat_heaps, dat_wcn, sim_wcn).to_string(index=False))
    print()
    table = compare(meta, dat_freq, dat_heaps, sim_freq, sim_heaps)
    cal = calibration(table, meta)
    print()
    print(write_table(table, meta, cal["native"]))
    print("wrote outputs/tables/model_vs_data.{csv,md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
