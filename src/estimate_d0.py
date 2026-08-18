#!/usr/bin/env python3
"""Measure the crossover scale D0 directly from a rank-frequency curve.

This is the **primary** estimate of D0 in the paper: it answers how D0 is
obtained and how sensitive the results are to it, and it is model-independent —
nothing here runs a simulation.

The idea. In the model D0 is a vocabulary size — the number of types beyond
which novelty is attenuated — so on a rank-frequency curve it is a *position in
the type ranking*. Fitting a continuous two-slope power law with a **free
breakpoint** (`fits.broken_power_law`) therefore returns a breakpoint R* that is
directly comparable with D0: same units, no conversion, no free normalisation.

Two uncertainties are reported, because they answer different questions:

* `R_star_interval` — the profile-likelihood interval of the breakpoint at fixed
  fit range. How sharply the data locate the kink.
* `R_star_band` — the spread when the lower end of the fit range is swept over
  r_min in {3, 10, 30, 100}. How much the answer depends on how much of the
  very-high-frequency head one calls "scaling regime". This is the honest error
  bar, the same fit-window convention used for every exponent in the paper.

Populations (all from `data_reduced/`, no raw corpus):

  native      SPGC English at 1e8 tokens          `spgc_en_heaps.npz`
  learner     COREFL L2 pooled es+de, 586k tok    `corefl_learner_all.npz`
  corefl_l1   COREFL native controls, 52k tok     `corefl_native_en.npz`

The last one is a genre-matched but very small control; it is reported to show
that the learner value is not an artefact of the corpus being conversational.

Estimating the crossover from the rank-frequency **only** (decision, user
2026-08-02): the Heaps curve is not an independent measurement of the same
quantity, so combining them would double-count.

Usage:
    python src/estimate_d0.py            # prints the table, writes the CSV
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import plotting as P                                    # noqa: E402
from fits import broken_power_law_band                  # noqa: E402
from io_reduced import DATA_REDUCED, load_corefl, load_heaps   # noqa: E402

#: lower ends of the fit range swept for the band
RMINS = (3.0, 10.0, 30.0, 100.0)
#: reported fit range starts here (a decade of head excluded)
RMIN = 10.0


def _rank_prob(freq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rank and normalised frequency of a descending count vector, hapax kept.

    The hapax plateau is *not* dropped here: the breakpoint we are after sits
    far above it (R* ~ 10^2-10^4 against a plateau starting at 10^4-10^5), and
    the fit range is bounded from above by `rmax` anyway.
    """
    freq = np.sort(np.asarray(freq)[np.asarray(freq) > 0])[::-1]
    return np.arange(1, freq.size + 1, dtype=float), freq / freq.sum()


def populations() -> dict:
    """The three rank-frequency curves, each over the range it actually has."""
    natives = load_heaps("en")
    learners = load_corefl("learner_all")
    corefl_l1 = load_corefl("native_en")
    return {
        "native (SPGC, 1e8 tok)": (natives["freq_matched"],
                                   int(natives["match_tokens"])),
        "learner (COREFL es+de)": (learners["freq"], int(learners["n_tokens"])),
        "L1 control (COREFL)": (corefl_l1["freq"], int(corefl_l1["n_tokens"])),
    }


def estimate(freq: np.ndarray, rmin: float = RMIN, rmax: float | None = None) -> dict:
    """R* and the two slopes of the broken power law, with both uncertainties."""
    rank, prob = _rank_prob(freq)
    return broken_power_law_band(rank, prob, rmin=rmin, rmax=rmax, rmins=RMINS)


def table() -> pd.DataFrame:
    """One row per population: R*, its interval, its band and the two slopes."""
    rows = []
    for name, (freq, n_tokens) in populations().items():
        b = estimate(freq)
        rows.append({
            "population": name,
            "tokens": n_tokens,
            "types": int(np.asarray(freq).size),
            "R_star": round(b["R_star"]),
            "profile_lo": round(b["R_star_interval"][0]),
            "profile_hi": round(b["R_star_interval"][1]),
            "band_lo": round(b["R_star_band"]["lo"]),
            "band_hi": round(b["R_star_band"]["hi"]),
            "alpha1": round(b["alpha1"], 2),
            "alpha2": round(b["alpha2"], 2),
            "r2": round(b["r2"], 4),
        })
    return pd.DataFrame(rows).set_index("population")


#: The calibration this table is compared against: the **deep** sweep
#: (`manifests/d0_calibration_refined.json`, `--tag refined`), not the wide one.
#: What this table quotes is `best_D0` with its competitive range, and that range
#: is only honest with the deep sweep's six seeds — with the wide sweep's two the
#: learners' range collapses onto a single grid point. `model_vs_data.py` and
#: `figure_SI5.py` read the wide sweep instead, for the opposite reason: they
#: need a grid that spans a factor of 130 and the `rstar_sim` column, neither of
#: which the deep sweep has. Neither is a coarse version of the other; see
#: src/calibrate_d0.py.
CALIBRATION = os.path.join(os.path.dirname(DATA_REDUCED), "outputs",
                           "d0_calibration_refined", "calibration_report.json")

#: which calibrated population corresponds to which measured one
CALIBRATION_OF = {"native (SPGC, 1e8 tok)": "native",
                  "learner (COREFL es+de)": "learner"}


def summary_with_calibration(t: pd.DataFrame) -> pd.DataFrame | None:
    """Join the measured R* with the model calibration, if it has been run.

    Three numbers per population, all in the same units (a rank):

    * `R_star (data)`   — measured from the empirical curve alone.
    * `D0 (model)`      — the calibrated parameter, i.e. which simulation best
      reproduces the empirical Heaps *and* Zipf curves.
    * `R_star (sim)`    — the crossover of that best-fit simulation, obtained by
      pushing its output through the *same* estimator used on the data.

    The third column is the one that makes the comparison honest. D0 is an input
    to the model; what the model actually produces is a curve, whose crossover
    need not equal D0. Fitting the simulation the way the data are fitted is the
    like-for-like test, and it is what exposes the learner discrepancy.
    """
    if not os.path.exists(CALIBRATION):
        return None
    report = json.load(open(CALIBRATION))
    calibrated = {p["population"]: p for p in report["populations"]}
    outdir = os.path.dirname(CALIBRATION)

    rows = []
    for name, r in t.iterrows():
        pop = CALIBRATION_OF.get(name)
        c = calibrated.get(pop) if pop else None
        row = {"population": name,
               "R_star (data)": int(r.R_star),
               "band (data)": f"[{int(r.band_lo):,}–{int(r.band_hi):,}]",
               "alpha2 (data)": r.alpha2}
        if c is None:
            rows.append({**row, "D0 (model)": "--", "range (model)": "--",
                         "R_star (sim)": "--", "alpha2 (sim)": "--",
                         "D0/data": "--", "sim/data": "--",
                         "_D0_over_data": float("nan"),
                         "_sim_over_data": float("nan")})
            continue

        # the best-fit run is kept by calibrate_d0.py precisely so that it can be
        # re-fitted here with the estimator used on the data
        best = os.path.join(outdir, f"best_{pop}_D0={c['best_D0']}_"
                                    f"seed={c['representative_seed']}.npz")
        sim = estimate(np.load(best)["frequencies"]) if os.path.exists(best) else None
        rows.append({
            **row,
            "D0 (model)": f"{c['best_D0']:,}",
            "range (model)": f"[{c['low']:,}–{c['high']:,}]",
            "R_star (sim)": f"{sim['R_star']:,.0f}" if sim else "--",
            "alpha2 (sim)": round(sim["alpha2"], 2) if sim else "--",
            "D0/data": round(c["best_D0"] / r.R_star, 2),
            "sim/data": round(sim["R_star"] / r.R_star, 2) if sim else "--",
            # unrounded, so the prose below can quote a percentage that is not
            # the rounding of a two-decimal ratio
            "_D0_over_data": c["best_D0"] / r.R_star,
            "_sim_over_data": sim["R_star"] / r.R_star if sim else float("nan"),
        })
    return pd.DataFrame(rows).set_index("population")


def main() -> int:
    t = table()
    t.to_csv(P.table_path("d0_from_data.csv"))

    with open(P.table_path("d0_from_data.md"), "w") as fh:
        fh.write("| population | tokens | $R^*$ | profile interval | $r_{\\min}$ band "
                 "| $\\alpha_1$ | $\\alpha_2$ |\n"
                 "| --- | ---: | ---: | --- | --- | ---: | ---: |\n")
        for name, r in t.iterrows():
            fh.write(f"| {name} | {r.tokens:,} | {r.R_star:,} | "
                     f"[{r.profile_lo:,}–{r.profile_hi:,}] | "
                     f"[{r.band_lo:,}–{r.band_hi:,}] | "
                     f"{r.alpha1:.2f} | {r.alpha2:.2f} |\n")
        fh.write("\nCrossover scale $R^*$ of a continuous two-slope power law with a "
                 "free breakpoint, fitted to the rank-frequency curve over "
                 f"$R\\ge{RMIN:.0f}$ on a grid uniform in $\\log R$. The profile "
                 "interval is the likelihood-ratio interval of the breakpoint at "
                 "fixed fit range; the band sweeps the lower end of that range over "
                 f"$r_{{\\min}}\\in\\{{{', '.join(f'{r:.0f}' for r in RMINS)}\\}}$ and "
                 "is the reported uncertainty. Because $R^*$ is a rank, it is "
                 "directly comparable with the model's $D_0$ — no conversion.\n")

    print(t.to_string())
    ratio = t.loc["native (SPGC, 1e8 tok)", "R_star"] / t.loc["learner (COREFL es+de)", "R_star"]
    print(f"\nnative / learner = {ratio:.0f}x, but the two corpora differ in "
          f"length by a factor 170 and R* is not length-independent: see "
          f"src/rstar_vs_length.py, which puts the ratio at matched length at "
          f"about 13x. That is the number to quote.")
    print("wrote outputs/tables/d0_from_data.{csv,md}")

    joint = summary_with_calibration(t)
    if joint is None:
        print("\n(no calibration report found - run src/calibrate_d0.py for the "
              "model-side estimate)")
        return 0

    # columns starting with "_" carry unrounded ratios for the prose below only
    joint.drop(columns=[c for c in joint.columns if c.startswith("_")]) \
        .to_csv(P.table_path("d0_summary.csv"))
    NAT, LEA = "native (SPGC, 1e8 tok)", "learner (COREFL es+de)"
    nat, lea = joint.loc[NAT], joint.loc[LEA]
    nat_gap = abs(nat["_D0_over_data"] - 1) * 100

    with open(P.table_path("d0_summary.md"), "w") as fh:
        fh.write("| population | $R^*$ measured | band | $D_0$ calibrated | "
                 "competitive range | $D_0$/data | $R^*$ of that simulation | "
                 "sim/data |\n"
                 "| --- | ---: | --- | ---: | --- | ---: | ---: | ---: |\n")
        for name, r in joint.iterrows():
            fh.write(f"| {name} | {r['R_star (data)']:,} | {r['band (data)']} | "
                     f"{r['D0 (model)']} | {r['range (model)']} | {r['D0/data']} | "
                     f"{r['R_star (sim)']} | {r['sim/data']} |\n")
        fh.write(
            "\nThree estimates of the same quantity, all of them ranks and therefore "
            "directly comparable with no conversion. $R^*$ *measured* comes from the "
            "shape of the empirical rank-frequency curve alone "
            "(`src/estimate_d0.py`). $D_0$ *calibrated* is the value whose simulation "
            "best reproduces the empirical Heaps *and* Zipf curves at $T=10^8$ "
            "(`src/calibrate_d0.py`, 24 values of $D_0$ x 6 seeds); the competitive "
            "range is the set of $D_0$ within one paired standard error of the loss "
            "minimum. The last two columns fit that best-fit **simulation** with the "
            "same estimator used on the data — the like-for-like test, since $D_0$ is "
            "an input to the model whereas what the model produces is a curve.\n\n"
            f"**Natives: the two independent estimates of $D_0$ agree to "
            f"{nat_gap:.1f}%** — {nat['R_star (data)']:,} measured from the data "
            f"against {nat['D0 (model)']} from the calibration. $D_0$ is therefore "
            "not a free parameter tuned on the model but a scale that can be read "
            "off the data.\n\n"
            f"That simulation then *realises* $R^*={nat['R_star (sim)']}$ on the "
            "single seed the calibration kept, against the measured "
            f"{nat['R_star (data)']:,}; its tail exponent, {nat['alpha2 (sim)']} "
            f"against {nat['alpha2 (data)']:.2f} measured, matches. A model's own "
            "crossover need not equal its $D_0$ input, and it carries seed-to-seed "
            "scatter of order 10% — `src/model_adequacy.py` averages it over seeds "
            "and is the number to quote for the realised crossover.\n\n"
            f"**Learners: they do not.** The calibration lands on "
            f"$D_0={lea['D0 (model)']}$, and that simulation realises "
            f"$R^*={lea['R_star (sim)']}$ against {lea['R_star (data)']:,} in the "
            f"data — a factor {lea['sim/data']} — with a tail exponent of "
            f"{lea['alpha2 (sim)']} against {lea['alpha2 (data)']:.2f}.\n\n"
            "`src/model_adequacy.py` establishes that this is a limitation of the "
            "model on that corpus and not an artefact of the calibration. Scanning "
            "$D_0$ and measuring what the simulation *realises*: the model does "
            "reach $R^*=282$ at $D_0=300$, i.e. the empirical crossover is within "
            "its range — the calibration simply does not go there. But at that "
            "$D_0$ **both** loss terms are worse, the Zipf one by a factor 3 and "
            "the Heaps one by 1.8, because the simulation then has 6,015 types "
            "against 16,096 in the data: matching the knee misplaces the whole "
            "curve. And across the entire scan the simulated tail exponent stays "
            "in 1.69–1.96, never approaching the empirical 1.48. No $D_0$ "
            "reproduces the learner curve.\n\n"
            "The likely reason is that the learner corpus is an aggregate of 3,326 "
            "texts averaging 176 tokens each, written by many different people — a "
            "mixture of micro-samples, whose vocabulary grows faster and whose tail "
            "is milder than any single-stream process can be. This "
            "individual-versus-aggregate limitation does not arise for the "
            "natives, whose texts are long and whose corpus is 170x larger.\n")

    print()
    print(joint.to_string())
    print("wrote outputs/tables/d0_summary.{csv,md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
