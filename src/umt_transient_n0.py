#!/usr/bin/env python3
"""Give the stationary baseline its best shot: the initial urn that mimics a crossover.

The Urn Model with Triggering has no scale at which its behaviour changes, so it
cannot produce a genuine crossover. It can however *look* as though it does. With
a large initial urn of n0 items the first phase draws a novelty almost every
step, so D(t) ~ t, and only once the reinforced mass overtakes n0 does the
asymptotic regime set in. On the vocabulary-growth curve alone that transient is
hard to tell from the real thing.

To make that comparison fair the baseline has to be tuned rather than guessed,
and tuned by the same rule the model itself is calibrated with: choose n0 so that
the crossover the simulation produces sits where the measured one does. This
script does that by bisection on log n0, using the free-breakpoint estimator of
Appendix B — the same function that measures R* on the corpus.

The point of the exercise is what the tuning cannot buy. Matching the crossover
and, with rho = 2, the tail exponent still leaves the head exponent far from the
alpha1 ~ 1 of every real corpus, because the first phase is a transient and not a
regime.

Output: outputs/tables/umt_transient_n0.csv, one row per evaluation.

Usage:
    python src/umt_transient_n0.py                 # target R* = 9,388 (SPGC English)
    python src/umt_transient_n0.py --target 9388 --T 100000000
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import calibrate_d0 as C                                     # noqa: E402
from fits import broken_power_law, two_regime_ols, heaps_exponent  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TABLES = REPO / "outputs" / "tables"

#: D0 far above any vocabulary a run can reach: attenuation off, so the process
#: is the plain UMT and nothing but n0, rho and nu is being varied.
NO_ATTENUATION = 10 ** 9


def simulate(binary: Path, n0: int, rho: float, nu: float, T: int,
             seed: int) -> dict:
    """One run, reduced on the fly to the quantities the baseline table reports."""
    with tempfile.TemporaryDirectory(prefix="umt_n0_", dir=C.WORK) as tmp:
        trajectory = Path(tmp) / "trajectory.cltraj.zst"
        env = {**os.environ, "UMT_SEED": str(seed),
               "CLTRAJ_OUTPUT": str(trajectory)}
        subprocess.run([str(binary), str(n0), str(T), str(rho), str(nu), "0.5",
                        str(NO_ATTENUATION)],
                       cwd=tmp, env=env, check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        times = np.unique(np.geomspace(10, T, 200).astype(np.int64))
        growth, frequency = C.stream_summary(trajectory, times, T)

    relative = frequency / frequency.sum()
    two = two_regime_ols(frequency)
    break_fit = broken_power_law(
        np.arange(1, len(frequency) + 1, dtype=float), relative)
    heaps = heaps_exponent(times, growth, 1e4, float(T))
    return {"n0": n0, "rho": rho, "nu": nu, "T": T, "seed": seed,
            "types": int(growth[-1]), "alpha1": two["alpha1"],
            "alpha2": two["alpha2"], "heaps_b": heaps["b"],
            "rstar": float(break_fit["R_star"])}


def select_n0(binary: Path, target: float, T: int, seed: int,
              low: int, high: int, iterations: int) -> list[dict]:
    """Bisect log n0 until the simulated crossover matches `target`.

    Monotone in n0 over any range worth searching: a larger initial urn holds the
    linear phase open longer and pushes the apparent knee to a higher rank. The
    bracket is checked rather than assumed.
    """
    rows = []

    def evaluate(n0: int) -> float:
        row = simulate(binary, n0, 2.0, 1.0, T, seed)
        rows.append(row)
        print(f"  n0={n0:>7,}  R*={row['rstar']:>9,.0f}  "
              f"alpha1={row['alpha1']:.2f}  alpha2={row['alpha2']:.2f}  "
              f"types={row['types']:>9,}", flush=True)
        return row["rstar"]

    r_low, r_high = evaluate(low), evaluate(high)
    if not (r_low <= target <= r_high):
        raise SystemExit(
            f"target {target:,.0f} is outside the bracket "
            f"[{r_low:,.0f}, {r_high:,.0f}]: widen --low/--high")

    lo, hi = float(low), float(high)
    for _ in range(iterations):
        mid = int(round(10 ** ((np.log10(lo) + np.log10(hi)) / 2)))
        if evaluate(mid) < target:
            lo = mid
        else:
            hi = mid
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=float, default=9388.0,
                    help="measured R* the transient must reproduce")
    ap.add_argument("--T", type=int, default=100_000_000)
    ap.add_argument("--seed", type=int, default=1,
                    help="1 matches the rows already in the baseline table")
    ap.add_argument("--low", type=int, default=5_000)
    ap.add_argument("--high", type=int, default=30_000)
    ap.add_argument("--iterations", type=int, default=5)
    args = ap.parse_args()

    C.WORK.mkdir(exist_ok=True)
    binary = C.WORK / "umt_seeded"
    # compile_simulator writes, patches and then removes its own copy of the
    # source, so nothing else has to be staged here
    C.compile_simulator(binary, C.WORK)

    print(f"target R* = {args.target:,.0f} | T = {args.T:,} | seed {args.seed}")
    start = time.monotonic()
    rows = select_n0(binary, args.target, args.T, args.seed,
                     args.low, args.high, args.iterations)

    frame = pd.DataFrame(rows)
    frame["distance"] = (np.log10(frame["rstar"])
                         - np.log10(args.target)).abs()
    frame = frame.sort_values("distance")
    TABLES.mkdir(parents=True, exist_ok=True)
    out = TABLES / "umt_transient_n0.csv"
    frame.to_csv(out, index=False)

    best = frame.iloc[0]
    print(f"\nselected n0 = {int(best['n0']):,}: R* = {best['rstar']:,.0f} "
          f"against a target of {args.target:,.0f} "
          f"({100 * (best['rstar'] / args.target - 1):+.1f}%)")
    print(f"  alpha1 = {best['alpha1']:.2f}, alpha2 = {best['alpha2']:.2f}, "
          f"Heaps b = {best['heaps_b']:.3f}, types = {int(best['types']):,}")
    print(f"[{time.monotonic() - start:.0f}s] wrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
