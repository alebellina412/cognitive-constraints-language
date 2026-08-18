#!/usr/bin/env python3
"""Is the learner D0 a calibration artefact, or a limit of the model?

The calibration selects D0 = 1,324 for the L2 learners while the crossover
measured from the learner corpus itself is R* = 259 — a factor 5. For the
natives the same two numbers agree to 0.2%. That asymmetry deserves an answer
rather than a caveat, and this script is the answer.

The question is *not* "which D0 does the loss prefer" (that is
`src/calibrate_d0.py`) but "what does the model actually produce at each D0, and
can any D0 reproduce the empirical curve at all". So for a grid of D0 it runs the
simulator, truncates it to the empirical token count exactly as the calibration
does, and then measures the simulation with the same estimators used on the
data: the free-breakpoint crossover R*, the tail exponent alpha2, the vocabulary
size, and both loss terms.

What it shows for the learners, at the p = 1/2 the paper fixes:

* The model *does* realise R* = 259 — at D0 ~ 300, close to the measured value.
  So the crossover is reachable; the calibration simply does not go there.
* At that D0 **both** loss terms are worse than at the selected D0, not just the
  Heaps one (1.8x and 3.0x, joint 2.2x): the simulation then has 6,015 types
  against 16,096 in the data and a tail of 1.94 against 1.48, so the whole curve
  is misplaced, not only its knee.
* A tail as mild as 1.48 does appear on the grid, but only at D0 ~ 13,000, which
  realises R* = 3,895 and 30,764 types. Mild tail and early crossover cannot be
  had together.

**The cause is the saturation value, not the corpus**, which `--p` establishes
by re-running the scan. The attenuation saturates at p, so the asymptotic tail is
1/(1-p): fixing p = 1/2 pins it at 2.00. The natives measure alpha2 = 1.99, so
p = 1/2 is the right value for them and their two estimates of D0 agree to 0.2%.
The learners measure 1.48, which needs p = 0.323 — and p = 1/2 forces the model
to a tail it must then pay for elsewhere, which it does by inflating D0.

Re-running the same scan at p = 0.324 removes most of the discrepancy:

    p       crossover-matching D0    it realises            joint-loss D0
    0.5     300                      a2 = 1.94, 6,015 types    1,324
    0.324   400                      a2 = 1.50, 14,796 types     600

against measured R* = 259, alpha2 = 1.48, 16,096 types. At p = 0.324 the
crossover-matching D0 costs 1.01x the minimum joint loss instead of 2.2x, so the
knee and the loss optimum stop disagreeing; the residual factor ~1.7 between 400
and the measured 259 is what is left for the aggregation effect below.

Note that p is not thereby a free parameter: p = 1 - 1/alpha2 is *read off* the
measured tail rather than fitted, so D0 remains the only calibrated quantity, and
the model reproduces the alpha2 it was given (1.44-1.52 across the grid).

A second effect is also present: the learner corpus is an aggregate of 3,326
texts averaging 176 tokens each, by many different writers — a mixture of
micro-samples whose vocabulary grows faster than a single-stream process. This
individual-versus-aggregate limitation does not affect the natives. It is a real
effect, but on these numbers it is the smaller one.

The trajectory prefix does not depend on T (verified: identical results at
T = 1e6 and T = 1e8), so this runs at T = 1e6 in seconds.

Usage:
    python src/model_adequacy.py                       # learners, default grid
    python src/model_adequacy.py --population native --T 100000000
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import calibrate_d0 as C                                  # noqa: E402
import estimate_d0 as E                                   # noqa: E402
import plotting as P                                      # noqa: E402

#: D0 values scanned by default — spans the measured R* and the calibrated D0
DEFAULT_GRID = (200, 300, 400, 600, 800, 1324, 2000, 3200)
DEFAULT_SEEDS = (101, 202, 303)


def scan(population: str, d0_values, seeds, T: int) -> pd.DataFrame:
    """Realised statistics and losses of the simulation at each D0."""
    cfg = {**C.DEFAULTS["calibration"], "T": T}
    model = C.DEFAULTS["model"]
    emp = C.prepare((population,), cfg)[population]
    tokens = min(T, int(emp["frequency"].sum()))

    C.WORK.mkdir(exist_ok=True)
    binary = C.WORK / "umt_adequacy"
    C.compile_simulator(binary, C.WORK)
    rows = []
    try:
        for d0 in d0_values:
            per_seed = []
            for seed in seeds:
                with tempfile.TemporaryDirectory(prefix="adeq_", dir=C.WORK) as tmp:
                    traj = Path(tmp) / "t.cltraj.zst"
                    env = {**os.environ, "UMT_SEED": str(seed),
                           "CLTRAJ_OUTPUT": str(traj)}
                    subprocess.run(
                        [str(binary), str(model["N0"]), str(T), str(model["rho"]),
                         str(model["nu"]), str(model["p"]), str(d0)],
                        cwd=tmp, env=env, check=True,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    ds, freq = C.stream_summary(traj, emp["t"], tokens)
                    lh, lz, _ = C.metrics(emp, ds, freq, cfg)
                    b = E.estimate(freq)
                    per_seed.append((b["R_star"], b["alpha2"], freq.size, lh, lz))
            a = np.array(per_seed, dtype=float)
            rows.append({
                "D0": d0,
                "R_star (sim)": round(a[:, 0].mean()),
                "alpha2 (sim)": round(a[:, 1].mean(), 2),
                "types (sim)": round(a[:, 2].mean()),
                "loss_heaps": round(a[:, 3].mean(), 4),
                "loss_zipf": round(a[:, 4].mean(), 4),
                "loss_joint": round(0.5 * a[:, 3].mean() + 0.5 * a[:, 4].mean(), 4),
            })
    finally:
        binary.unlink(missing_ok=True)
    return pd.DataFrame(rows).set_index("D0")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--population", default="learner", choices=tuple(C.POPULATIONS))
    ap.add_argument("--p", type=float, default=None,
                    help="saturation value of the attenuation; default 0.5. "
                         "Pass 1-1/alpha2 to read it off the population's own "
                         "measured tail instead of fixing it at the bound.")
    ap.add_argument("--T", type=int, default=1_000_000)
    ap.add_argument("--d0-values", default=",".join(str(v) for v in DEFAULT_GRID))
    ap.add_argument("--seeds", default=",".join(str(v) for v in DEFAULT_SEEDS))
    args = ap.parse_args()

    d0_values = [int(v) for v in args.d0_values.split(",") if v]
    seeds = [int(v) for v in args.seeds.split(",") if v]

    cfg = {**C.DEFAULTS["calibration"], "T": args.T}
    emp = C.prepare((args.population,), cfg)[args.population]
    target = E.estimate(emp["frequency"])
    print(f"{args.population}: empirical R* = {target['R_star']:,.0f}, "
          f"alpha2 = {target['alpha2']:.2f}, "
          f"{len(emp['frequency']):,} types in "
          f"{int(emp['frequency'].sum()):,} tokens\n")

    if args.p is not None:
        C.DEFAULTS["model"]["p"] = args.p
    p = C.DEFAULTS["model"]["p"]
    print(f"attenuation saturates at p = {p:g}, so the model's asymptotic tail "
          f"exponent is 1/(1-p) = {1 / (1 - p):.2f}; the population's own tail is "
          f"{target['alpha2']:.2f}, which would need p = {1 - 1 / target['alpha2']:.3f}\n")

    t = scan(args.population, d0_values, seeds, args.T)
    print(t.to_string())

    # the D0 whose simulation reproduces the measured crossover, and what it costs
    k = int(np.abs(t["R_star (sim)"] - target["R_star"]).argmin())
    matching, best = t.index[k], t["loss_joint"].idxmin()
    print(f"\nD0 reproducing the measured crossover: {matching:,} "
          f"(realises R* = {t.loc[matching, 'R_star (sim)']:,.0f})")
    print(f"D0 selected by the joint loss        : {best:,} "
          f"(realises R* = {t.loc[best, 'R_star (sim)']:,.0f})")
    if matching != best:
        rh = t.loc[matching, "loss_heaps"] / t.loc[best, "loss_heaps"]
        rz = t.loc[matching, "loss_zipf"] / t.loc[best, "loss_zipf"]
        rj = t.loc[matching, "loss_joint"] / t.loc[best, "loss_joint"]
        # whether matching the knee is expensive is exactly what distinguishes a
        # model that cannot reach the data from one that merely prefers not to
        cost = ("so matching the knee misfits the rest of the curve"
                if rj > 1.5 else
                "but the joint loss barely separates the two, so the knee and the "
                "loss optimum are not in real conflict")
        print(f"  at the crossover-matching D0 the Heaps loss is {rh:.1f}x, the "
              f"Zipf loss {rz:.1f}x and the joint loss {rj:.2f}x the loss at the "
              f"selected D0 - {cost}")
    lo, hi = t["alpha2 (sim)"].min(), t["alpha2 (sim)"].max()
    # the verdict needs a grid to be a verdict; a one-point scan says nothing
    # about what the model can or cannot reach
    verdict = ""
    if len(t) >= 3:
        verdict = (" - no D0 reproduces the empirical tail"
                   if not lo <= target["alpha2"] <= hi
                   else " - reachable somewhere on the grid")
    print(f"\nsimulated alpha2 over the scanned grid: {lo:.2f}-{hi:.2f}, "
          f"against {target['alpha2']:.2f} measured{verdict}")
    # reachable *somewhere* is not the question: a tail exponent recovered only at
    # a D0 that misplaces the crossover is not a fit. What matters is the exponent
    # the model realises at the D0 that does reproduce the knee.
    print(f"at the crossover-matching D0 = {matching:,} the model realises "
          f"alpha2 = {t.loc[matching, 'alpha2 (sim)']:.2f} and "
          f"{int(t.loc[matching, 'types (sim)']):,} types, against "
          f"{target['alpha2']:.2f} and {len(emp['frequency']):,} measured")

    tag = "" if args.p is None else f"_p{args.p:g}"
    out = P.table_path(f"model_adequacy_{args.population}{tag}.csv")
    t.to_csv(out)
    print(f"\nwrote outputs/tables/{os.path.basename(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
