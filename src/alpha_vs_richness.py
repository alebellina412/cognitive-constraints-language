#!/usr/bin/env python3
"""Does the tail exponent follow how much vocabulary a language spends per token?

The claim of the paper, stated as a measurement: a language that expresses new
concepts by recombining words it already has (few new types per token) is pushed
into the constrained regime and its rank-frequency tail bends more steeply;
a language that mints a new lexical unit instead (many new types per token) keeps
a milder tail.

Everything is measured at one common corpus length T, so the type counts and the
exponents refer to the same point on every curve and nothing has to be assumed
about how either quantity scales.

Reported per language:
  D(T)          vocabulary at the common length
  D(T)/T        types minted per token: the direct measure of lexical spending
  b             Heaps exponent, fitted over the decade below T (scale-free)
  alpha1, alpha2   the two regimes at T
and the rank correlation of each richness measure with alpha2.

Needs `data_raw/` (the SPGC tokens zip). Run once; everything downstream reads its table from `outputs/tables/`,
which is generated rather than committed.

Usage:
    python src/alpha_vs_richness.py [--tokens 30000000] [--seeds 2]
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import sys
import time
import zipfile

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import build_reduced as B                                  # noqa: E402
import estimate_d0 as E                                    # noqa: E402
import plotting as P                                       # noqa: E402
from fits import two_regime_ols_band                       # noqa: E402

LANGS = ["en", "fr", "it", "es", "de"]

#: Spanish is the shortest corpus (35,986,705 tokens), so this is the largest
#: length at which all five can be compared without resampling.
COMMON_T = 30_000_000

#: alpha2 on each language's own full corpus (SI Table 1)
ALPHA2_FULL = {"en": 1.99, "fr": 1.98, "it": 1.71, "es": 1.78, "de": 1.48}


def read_pool(zf, pgids) -> tuple[list[np.ndarray], int]:
    """Every book of one language as an int32 id array, read once."""
    vocab: dict[str, int] = {}
    pool: list[np.ndarray] = []
    for pgid in pgids:
        lines = B.read_member_lines(zf, f"{B.TOKENS_ROOT}/{pgid}_tokens.txt")
        if lines is None:
            continue
        toks = [t for t in lines if t]
        if not toks:
            continue
        assign = vocab.setdefault
        pool.append(np.fromiter((assign(t, len(vocab)) for t in toks),
                                dtype=np.int32, count=len(toks)))
    return pool, len(vocab)


def exact_spearman_p(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided permutation p-value for Spearman's rho.

    With five languages the asymptotic p-value scipy returns is not usable — it
    goes to zero at |rho| = 1, where the t statistic diverges. The exact value is
    the fraction of the 5! = 120 relabellings that reach |rho| at least as large,
    which for a perfect monotone relation is 2/120 = 0.0167.
    """
    from itertools import permutations
    r0 = abs(stats.spearmanr(a, b).statistic)
    hits = sum(1 for p in permutations(range(len(a)))
               if abs(stats.spearmanr(np.array(p), b).statistic) >= r0 - 1e-12)
    return hits / math.factorial(len(a))


def heaps_exponent(first_sorted: np.ndarray, t_hi: int, decades: float = 1.0) -> float:
    """Slope of log D(t) against log t over the decade below `t_hi`.

    D(t) is read straight off the sorted first-occurrence positions, so no
    binning or averaging is involved.
    """
    t_lo = t_hi / (10 ** decades)
    grid = np.unique(np.round(np.logspace(np.log10(t_lo), np.log10(t_hi), 40)).astype(np.int64))
    D = np.searchsorted(first_sorted, grid, side="left")
    ok = D > 0
    x, y = np.log10(grid[ok]), np.log10(D[ok])
    return float(np.polyfit(x, y, 1)[0])


def within_between(df: pd.DataFrame, TS: list[int]) -> None:
    """Is the cross-language relation just arithmetic? The falsification test.

    If alpha2 were fixed by the token/type bookkeeping alone — "fewer types at
    the same amount of text forces a steeper curve" — then alpha2 would be some
    function g(T, D), and the sign of its response to the type-token ratio could
    not depend on how that ratio was made to vary. So the same quantity is
    regressed twice:

      between  at each fixed length, across the five languages;
      within   inside each language, across lengths.

    A mechanical relation must give the same sign both ways. Opposite signs
    falsify it: the cross-language relation then carries information that the
    bookkeeping does not.
    """
    med = (df.groupby(["lang", "T"])
             .agg(D=("D", "median"), D_per_1e6=("D_per_1e6", "median"),
                  alpha2=("alpha2", "median")).reset_index())
    med.to_csv(P.table_path("alpha_richness_within_between.csv"), index=False)

    print("\n=== is the relation mechanical? between- against within-language ===")
    lines = []
    for xvar, label in (("D_per_1e6", "log10 D(t)/t"), ("D", "log10 D(t)")):
        print(f"\n  slope of alpha2 on {label}")
        btw = []
        for t in TS:
            s = med[med["T"] == t]
            if len(s) < 3:
                continue
            sl, _, r, p, se = stats.linregress(np.log10(s[xvar]), s["alpha2"])
            btw.append(sl)
            print(f"    between languages, T={t:>11,}:  {sl:+.3f} "
                  f"(R2={r ** 2:.3f}, se={se:.3f})")
        wit = []
        for code in LANGS:
            s = med[med["lang"] == code]
            if len(s) < 3:
                continue
            sl, _, r, p, se = stats.linregress(np.log10(s[xvar]), s["alpha2"])
            wit.append(sl)
            print(f"    within {P.LANG_NAMES[code]:<9}            :  {sl:+.3f} "
                  f"(R2={r ** 2:.3f})")
        if btw and wit:
            mb, mw = float(np.median(btw)), float(np.median(wit))
            same = "SAME sign" if mb * mw > 0 else "OPPOSITE signs"
            print(f"    -> median between {mb:+.3f}, median within {mw:+.3f}: {same}")
            lines.append((label, mb, mw, same))

    with open(P.table_path("alpha_richness_within_between.md"), "w") as fh:
        fh.write("| regression | between languages | within a language | |\n"
                 "| --- | ---: | ---: | --- |\n")
        for label, mb, mw, same in lines:
            fh.write(f"| slope of $\\alpha_2$ on {label} | {mb:+.2f} | "
                     f"{mw:+.2f} | {same} |\n")
        fh.write("\nMedian over the lengths (between) and over the five "
                 "languages (within). If $\\alpha_2$ were determined by the "
                 "token/type bookkeeping, the same quantity could not respond "
                 "with one sign when the type-token ratio is varied across "
                 "languages and the opposite sign when it is varied inside one "
                 "language.\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens", type=int, nargs="+", default=[COMMON_T],
                    help="one or more common corpus lengths")
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()
    TS = sorted(args.tokens)
    T = TS[-1]                      # the headline length, for the main table

    langs = B.load_manifest(LANGS)
    rows = []
    with zipfile.ZipFile(B.TOKENS_ZIP) as zf:
        for code in LANGS:
            t0 = time.time()
            pool, vsize = read_pool(zf, langs[code])
            total = sum(p.size for p in pool)
            if total < T:
                print(f"  [{code}] only {total:,} tokens, skipped")
                continue
            for seed in range(args.seeds):
                rng = np.random.default_rng(seed)
                full = np.concatenate([pool[i] for i in rng.permutation(len(pool))])
                for t in TS:
                    stream = full[:t]
                    first = B.first_occurrence(stream, vsize)
                    first = np.sort(first[first < stream.size])
                    counts = np.bincount(stream, minlength=vsize)
                    freq = np.sort(counts[counts > 0])[::-1].astype(np.int64)
                    z = two_regime_ols_band(freq)
                    # the crossover is measured here rather than in a script of
                    # its own because R* is length-dependent, so the only place
                    # it can be compared across languages is inside this loop,
                    # where every corpus is cut to the same T
                    r = E.estimate(freq)
                    rows.append({"lang": code, "seed": seed, "T": t,
                                 "D": int(freq.size),
                                 "D_per_1e6": 1e6 * freq.size / t,
                                 "heaps_b": heaps_exponent(first, t),
                                 "alpha1": z["alpha1"], "alpha2": z["alpha2"],
                                 "alpha2_lo": z["alpha2_band"]["lo"],
                                 "alpha2_hi": z["alpha2_band"]["hi"],
                                 "R_star": r["R_star"],
                                 "R_star_lo": r["R_star_band"]["lo"],
                                 "R_star_hi": r["R_star_band"]["hi"]})
                    del first, counts, freq
                del full
                gc.collect()
            print(f"  [{code}] {total:,} tokens read, {args.seeds} orderings x "
                  f"{len(TS)} lengths [{time.time() - t0:.0f}s]", flush=True)
            del pool
            gc.collect()

    df_all = pd.DataFrame(rows)
    df_all.to_csv(P.table_path("alpha_vs_richness_long.csv"), index=False)
    within_between(df_all, TS)
    df = df_all[df_all["T"] == T].drop(columns=["T"])
    agg = (df.groupby("lang")
             .agg(D=("D", "median"), D_per_1e6=("D_per_1e6", "median"),
                  heaps_b=("heaps_b", "median"), alpha1=("alpha1", "median"),
                  alpha2=("alpha2", "median"), alpha2_min=("alpha2", "min"),
                  alpha2_max=("alpha2", "max"))
             .reindex(LANGS))
    agg["alpha2_full"] = [ALPHA2_FULL[c] for c in agg.index]
    agg = agg.sort_values("D")
    agg.to_csv(P.table_path("alpha_vs_richness.csv"))
    print("\n" + agg.round(3).to_string())

    corrs = {}
    for measure in ("D", "D_per_1e6", "heaps_b"):
        for target in ("alpha2", "alpha2_full"):
            rho = stats.spearmanr(agg[measure], agg[target]).statistic
            p = exact_spearman_p(agg[measure].to_numpy(), agg[target].to_numpy())
            r, pr = stats.pearsonr(agg[measure], agg[target])
            corrs[(measure, target)] = (rho, p, r, pr)
            print(f"  {measure:10s} vs {target:11s}: Spearman rho={rho:+.2f} "
                  f"(exact p={p:.4f})   Pearson r={r:+.2f} (p={pr:.4f})")

    # the empirical law: alpha2 against log10 of the vocabulary
    lx = np.log10(agg["D"].to_numpy())
    for target in ("alpha2", "alpha2_full"):
        sl, ic, rr, pp, se = stats.linregress(lx, agg[target].to_numpy())
        corrs[("fit", target)] = (sl, ic, rr ** 2, pp, se)
        print(f"  {target:11s} = {ic:.2f} {sl:+.3f} log10 D   "
              f"R2={rr ** 2:.4f}  p={pp:.4f}  se(slope)={se:.3f}")

    with open(P.table_path("alpha_vs_richness.md"), "w") as fh:
        fh.write(f"**All five languages at the same corpus length, "
                 f"T = {T:,} tokens**\n\n"
                 f"| language | $D(T)$ | types per $10^6$ tokens | Heaps $b$ | "
                 f"$\\alpha_1$ | $\\alpha_2$ | $\\alpha_2$ on the full corpus |\n"
                 f"| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for code, r in agg.iterrows():
            fh.write(f"| {P.LANG_NAMES[code]} | {int(r.D):,} | "
                     f"{r.D_per_1e6:,.0f} | {r.heaps_b:.3f} | {r.alpha1:.2f} | "
                     f"{r.alpha2:.2f} [{r.alpha2_min:.2f}–{r.alpha2_max:.2f}] | "
                     f"{r.alpha2_full:.2f} |\n")
        rho, p, _, _ = corrs[("D", "alpha2")]
        rho_f, p_f, _, _ = corrs[("D", "alpha2_full")]
        sl, ic, r2, pp, se = corrs[("fit", "alpha2")]
        _, _, r2f, _, _ = corrs[("fit", "alpha2_full")]
        rho_b, p_b, _, _ = corrs[("heaps_b", "alpha2")]
        fh.write(f"\nRows are ordered by vocabulary. The tail exponent decreases "
                 f"monotonically as the vocabulary spent per token grows: "
                 f"Spearman rho = {rho:+.2f} (exact p = {p:.3f}) against "
                 f"$\\alpha_2$ measured here, and {rho_f:+.2f} "
                 f"(exact p = {p_f:.3f}) against $\\alpha_2$ measured "
                 f"independently on each language's own full corpus. The "
                 f"relation is log-linear, $\\alpha_2 = {ic:.2f} {sl:+.2f}"
                 f"\\log_{{10}} D(T)$, with $R^2 = {r2:.3f}$ "
                 f"($R^2 = {r2f:.3f}$ against the full-corpus exponents).\n\n"
                 f"What does *not* predict $\\alpha_2$ is the Heaps exponent "
                 f"(rho = {rho_b:+.2f}, p = {p_b:.2f}): the level of vocabulary "
                 f"spent per token separates the languages, its growth rate does "
                 f"not.\n\n"
                 f"Caveat for the text: five languages from two families "
                 f"(three Romance, two Germanic), so the effective number of "
                 f"independent points is smaller than five, and the relation is "
                 f"reported as a quantified cross-linguistic regularity rather "
                 f"than an established law.\n")
    print("\nwrote outputs/tables/alpha_vs_richness.{csv,md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
