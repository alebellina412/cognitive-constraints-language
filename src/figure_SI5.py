#!/usr/bin/env python3
"""Appendix figure — the two criteria that estimate D0 from the model.

D0 can be read straight off a corpus as the crossover R* of the free-breakpoint
fit. From the model it can be obtained in two ways, and this figure is one panel
per way, with both populations in each.

  (A) the joint Heaps-and-Zipf loss: which D0 brings the whole simulated curve
      closest to the whole empirical one. This is the criterion the main text
      uses for the natives.
  (B) the crossover: which D0 makes the simulation bend where the corpus bends.
      Only the location of the knee enters, not the rest of the curve.

Splitting by criterion rather than by population is what makes the figure
readable: each panel then carries one quantity in one unit, and the comparison
that matters — the two populations behaving differently — happens inside a panel
instead of across the pair. The loss is plotted as a rise above its own minimum
so that two populations with different loss levels share an axis.

The natives agree between the two criteria, to within a tenth of a decade. The
learners do not: the loss lands a factor of five above the crossover, and the
appendix on D0 explains why the crossover is the one to trust there.

This runs no simulation of its own: it reads back the **wide** calibration sweep
written by notebook 8, `outputs/d0_calibration_crossover/runs.csv`, which is the
only one that records `rstar_sim` — the crossover the simulation itself produces
at each D0 — and the only one whose grid reaches down to D0 = 200, where the
learner criterion lands. That directory is *not* committed (see `.gitignore`);
notebook 8 has to have run.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt                                   # noqa: E402

import plotting as P                                              # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAL = os.path.join(REPO, "outputs", "d0_calibration_crossover")

#: R* measured on each corpus by the free-breakpoint fit of `app:fits`
R_STAR = {"native": 9_388, "learner": 259}

LABEL = {"native": "L1 natives", "learner": "L2 learners"}

POPULATIONS = ("native", "learner")


def colour(population: str):
    return (P.MODEL_COLORS["data"] if population == "native"
            else P.GROUP_COLORS["learners"])


def profiles(directory: str) -> dict:
    """Mean loss and mean simulated crossover over seeds, per D0."""
    out = {}
    rows = [r for r in csv.DictReader(open(os.path.join(directory, "runs.csv")))
            if r["status"] == "ok"]
    for population in POPULATIONS:
        loss, rstar = defaultdict(list), defaultdict(list)
        for r in rows:
            if r["population"] != population:
                continue
            loss[int(r["D0"])].append(float(r["loss_joint"]))
            rstar[int(r["D0"])].append(float(r["rstar_sim"]))
        d0 = np.array(sorted(loss))
        out[population] = {
            "D0": d0,
            "loss": np.array([np.mean(loss[d]) for d in d0]),
            "loss_se": np.array([np.std(loss[d], ddof=1) / np.sqrt(len(loss[d]))
                                 if len(loss[d]) > 1 else 0.0 for d in d0]),
            "rstar": np.array([np.mean(rstar[d]) for d in d0]),
        }
    return out


def crossing(d0: np.ndarray, rstar: np.ndarray, target: float) -> float:
    """D0 at which the simulated crossover equals the measured one.

    Interpolated in log-log, where the relation is close to a straight line of
    unit slope: the simulation bends slightly below D0 at every D0 on the grid,
    so the criterion selects a value a little above the crossover it has to hit.
    """
    return float(10 ** np.interp(np.log10(target), np.log10(rstar), np.log10(d0)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calibration", default=CAL,
                    help="directory holding runs.csv")
    args = ap.parse_args()

    P.setup_style()
    data = profiles(args.calibration)
    selected = {}

    fig, (ax_loss, ax_cross) = plt.subplots(
        1, 2, figsize=P.figsize(aspect=0.72, n_cols=2), layout="constrained")

    for population in POPULATIONS:
        d = data[population]
        c = colour(population)
        target = R_STAR[population]

        # ---- (A) the loss criterion ------------------------------------- #
        # plotted as a rise above its own minimum: the absolute level differs
        # between populations and carries nothing about where the minimum is
        rise = d["loss"] - d["loss"].min()
        by_loss = int(d["D0"][int(np.argmin(d["loss"]))])
        ax_loss.plot(d["D0"], rise, "-", color=c, lw=P.LW["curve"],
                     label=LABEL[population], zorder=3)
        ax_loss.fill_between(d["D0"], rise - d["loss_se"], rise + d["loss_se"],
                             color=c, alpha=0.22, lw=0, zorder=2)
        ax_loss.axvline(by_loss, color=c, ls="--", lw=P.LW["mark"], zorder=1)

        # ---- (B) the crossover criterion --------------------------------- #
        by_cross = crossing(d["D0"], d["rstar"], target)
        ax_cross.plot(d["D0"], d["rstar"], "-", color=c, lw=P.LW["curve"],
                      label=LABEL[population], zorder=3)
        ax_cross.axhline(target, color=c, ls=":", lw=P.LW["mark"], zorder=1)
        ax_cross.axvline(by_cross, color=c, ls="--", lw=P.LW["mark"], zorder=1)
        # the D0 the other criterion picks, so that the disagreement is visible
        # in this panel alone: on the natives it lands on the crossing, on the
        # learners it is a factor of five away
        ax_cross.axvline(by_loss, color=c, ls="-", lw=P.LW["mark"], alpha=0.35,
                         zorder=1)
        selected[population] = (by_loss, by_cross)

    ax_loss.set(xscale="log", xlabel=r"$D_0$",
                ylabel="joint loss above its minimum")
    ax_cross.set(xscale="log", yscale="log", xlabel=r"$D_0$",
                 ylabel=r"crossover $R^*$ of the simulation")
    # One legend for the figure, in panel A: the two panels use the same colour
    # for the same population, so a second copy carries nothing, and at this
    # aspect panel B has no free corner -- its curves rise across the diagonal
    # and the two horizontals at the measured R* cross whatever is left.
    ax_loss.legend(frameon=False, fontsize=P.FONT["legend"], loc="upper right",
                   handlelength=1.4, borderaxespad=0.15)
    for ax, letter in zip((ax_loss, ax_cross), "AB"):
        P.panel_label(ax, letter)

    for population in POPULATIONS:
        by_loss, by_cross = selected[population]
        print(f"{population:8s} loss -> D0 = {by_loss:>6,}   "
              f"crossover -> D0 = {by_cross:>8,.0f}   "
              f"(measured R* = {R_STAR[population]:,}); "
              f"ratio {max(by_loss, by_cross) / min(by_loss, by_cross):.1f}")
    for path in P.save_figure(fig, "SI5"):
        print("wrote", os.path.relpath(path, REPO))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
