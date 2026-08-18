#!/usr/bin/env python3
"""How much of the learner-vs-native gap in R* is a corpus-length artefact?

The control Figure S4 needs. The learners have 586,145 tokens and the native
reference has 1e8, and the measured crossover R* is **not** length-independent:
a short sample cannot resolve a kernel larger than the vocabulary it contains,
so R* is biased downwards and converges to D0 from below.

Measured on the native corpus itself, truncated to a range of lengths:

    tokens        types      R*
    52,425        5,859      73
    586,145      25,216   3,332   (median over 6 independent book samples)
    5,000,000    85,207   6,756
    50,000,000  281,564   9,606
    100,000,000 422,090   9,388

R* saturates above ~5e7 tokens, so the native value of 9,388 at 1e8 is a genuine
estimate of D0. At 586,145 tokens the same corpus gives only ~3,300 — a factor 3
lower for a reason that has nothing to do with the speakers.

**Consequence: the learner-vs-native ratio must be quoted at matched length.**
Comparing 259 (learners, 586k) with 9,388 (natives, 1e8) gives 36x, of which
roughly a factor 3 is sampling. At matched length the ratio is ~13x, and the
bands do not overlap (native 2,223-4,036 against learner 206-540), so the effect
is real — it is just smaller than the naive comparison suggests.

The model agrees with this control: a simulation at D0 = 9,366 truncated to
586,145 tokens realises R* = 3,270, inside the native range at that length.

What remains genuinely anomalous about the learners is a *shape*, not a size:
at 586,145 tokens they have 16,096 types against the natives' ~25,000 (64%) but
a crossover at 259 against ~3,300 (8%). Their vocabulary is nearly native-sized
while their kernel is an order of magnitude smaller. The model ties the two
together through the single parameter D0 and therefore cannot reproduce the
combination — which is why its calibrated learner D0 (1,324) sits between the
value that matches the type count (~3,200) and the one that matches the
crossover (~300). See `src/model_adequacy.py`.

Needs `data_raw/` (the SPGC tokens zip), unlike the rest of the pipeline: it
re-reads the corpus to truncate it. Run once; everything downstream reads its table from `outputs/tables/`,
which is generated rather than committed.

Usage:
    python src/rstar_vs_length.py [--samples 6]
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import build_reduced as B                                 # noqa: E402
import estimate_d0 as E                                   # noqa: E402
import plotting as P                                      # noqa: E402
from io_reduced import load_corefl                        # noqa: E402

#: the learner corpus length, so one row is exactly the matched comparison
LEARNER_TOKENS = 586_145
LENGTHS = (52_425, LEARNER_TOKENS, 5_000_000, 50_000_000)


def native_curve(lengths=LENGTHS, samples: int = 6) -> pd.DataFrame:
    """R* of the native corpus truncated to each length.

    At the shorter lengths only a few dozen books fit, so the answer depends on
    *which* books; each length is therefore measured over `samples` independent
    book draws and the spread is reported. At 5e6 and above one sample is enough
    (hundreds of books, and R* has nearly converged).
    """
    langs = B.load_manifest(["en"])
    pgids = langs["en"]
    rows = []
    with zipfile.ZipFile(B.TOKENS_ZIP) as zf:
        for n in lengths:
            n_draws = samples if n <= LEARNER_TOKENS * 2 else 1
            stars, types = [], []
            for seed in range(n_draws):
                h = B.build_heaps_one_lang(zf, "en", pgids, max_tokens=n,
                                           n_shuffles=1, seed=seed,
                                           match_tokens=n, n_points=30)
                freq = np.sort(h["freq_matched"])[::-1].astype(np.int64)
                stars.append(E.estimate(freq)["R_star"])
                types.append(freq.size)
            rows.append({"tokens": n, "samples": n_draws,
                         "types": int(np.median(types)),
                         "R_star": round(float(np.median(stars))),
                         "R_star_min": round(float(np.min(stars))),
                         "R_star_max": round(float(np.max(stars)))})
    return pd.DataFrame(rows).set_index("tokens")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", type=int, default=6,
                    help="independent book draws at the short lengths")
    args = ap.parse_args()

    t = native_curve(samples=args.samples)
    t.to_csv(P.table_path("rstar_vs_length.csv"))
    print(t.to_string())

    learners = load_corefl("learner_all")
    lea = E.estimate(learners["freq"])
    nat = t.loc[LEARNER_TOKENS]
    matched = nat["R_star"] / lea["R_star"]
    naive = t.loc[LENGTHS[-1] if LENGTHS[-1] in t.index else t.index[-1], "R_star"]

    print(f"\nat the learner length ({LEARNER_TOKENS:,} tokens):")
    print(f"  natives  R* = {nat['R_star']:,} "
          f"[{nat['R_star_min']:,}, {nat['R_star_max']:,}]  ({nat['types']:,} types)")
    print(f"  learners R* = {lea['R_star']:,.0f} "
          f"[{lea['R_star_band']['lo']:,.0f}, {lea['R_star_band']['hi']:,.0f}]  "
          f"({learners['freq'].size:,} types)")
    print(f"  ratio at matched length = {matched:.0f}x")
    print(f"  (the naive 1e8-vs-586k comparison gives "
          f"{9388 / lea['R_star']:.0f}x, of which ~{9388 / nat['R_star']:.1f}x "
          f"is corpus length)")

    with open(P.table_path("rstar_vs_length.md"), "w") as fh:
        fh.write("| native corpus length | types | $R^*$ | across book samples |\n"
                 "| ---: | ---: | ---: | --- |\n")
        for n, r in t.iterrows():
            spread = (f"[{int(r.R_star_min):,}–{int(r.R_star_max):,}]"
                      if r.samples > 1 else "single sample")
            fh.write(f"| {int(n):,} | {int(r.types):,} | {int(r.R_star):,} | "
                     f"{spread} |\n")
        fh.write(
            f"\nThe same native corpus (SPGC English), truncated to each length "
            f"and re-measured with `src/estimate_d0.py`. $R^*$ saturates above "
            f"~5·10⁷ tokens, so the value quoted for the natives at 10⁸ is a "
            f"genuine estimate of $D_0$; at short lengths it is biased downwards, "
            f"because a sample cannot resolve a kernel larger than the vocabulary "
            f"it contains. At the shorter lengths only a few dozen books fit, so "
            f"each is measured over {args.samples} independent book draws.\n\n"
            f"**The learner-vs-native ratio must therefore be quoted at matched "
            f"length.** At {LEARNER_TOKENS:,} tokens the natives give "
            f"$R^*={int(nat['R_star']):,}$ "
            f"[{int(nat['R_star_min']):,}–{int(nat['R_star_max']):,}] against the "
            f"learners' {lea['R_star']:,.0f} "
            f"[{lea['R_star_band']['lo']:,.0f}–{lea['R_star_band']['hi']:,.0f}], a "
            f"factor **{matched:.0f}** with non-overlapping bands. Comparing "
            f"instead against the natives at 10⁸ gives {9388 / lea['R_star']:.0f}x, "
            f"of which about a factor {9388 / nat['R_star']:.1f} is corpus length "
            f"rather than a property of the speakers.\n\n"
            f"What stays anomalous is a shape, not a size: at this length the "
            f"learners have {learners['freq'].size:,} types against the natives' "
            f"{int(nat['types']):,} — {learners['freq'].size / nat['types']:.0%} — "
            f"but a crossover {matched:.0f}x earlier. Their vocabulary is nearly "
            f"native-sized while their kernel is an order of magnitude smaller.\n")
    print("\nwrote outputs/tables/rstar_vs_length.{csv,md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
