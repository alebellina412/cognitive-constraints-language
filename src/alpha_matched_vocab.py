#!/usr/bin/env python3
"""Is the tail exponent set by the language or by the size of the vocabulary?

`src/alpha_vs_length.py` matches the *amount of text*: at equal token counts
every other language has a larger vocabulary than English (Spanish 1.6x, Italian
2.0x, German 3.3x), and the larger it is the smaller alpha2. That leaves the two
candidate explanations entangled, because a larger vocabulary also means a longer
rank axis, so a fixed tail window R >= 1e4 intercepts a different part of the
curve in each language.

This script matches the *vocabulary* instead. For a common target D*, each
language is truncated at the token position t* where its D(t) first reaches D*,
and alpha1 / alpha2 are re-measured there. Now the rank axes have the same
length by construction and the fit window sits in the same place on all five
curves.

  * if the exponents converge, alpha2 is a function of vocabulary size rather
    than of the language;
  * if they stay apart, the difference is structural.

t* is exact and needs no search: it is the position at which the D*-th distinct
type first occurs, which `first_occurrence` already gives.

Needs `data_raw/` (the SPGC tokens zip). Run once; everything downstream reads its table from `outputs/tables/`,
which is generated rather than committed.

Usage:
    python src/alpha_matched_vocab.py [--seeds 2] [--targets 200000 350000]
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import build_reduced as B                                  # noqa: E402
import plotting as P                                       # noqa: E402
from fits import two_regime_ols_band                       # noqa: E402

LANGS = ["en", "fr", "it", "es", "de"]

#: Spanish has the smallest full-corpus vocabulary (368,692 types), so every
#: language can reach these targets.
D_TARGETS = (200_000, 350_000)

#: each language on its own full corpus, for the reference column
OWN = {"en": (163_240_840, 552_243, 1.99), "fr": (170_521_076, 631_377, 1.98),
       "it": (42_876_256, 504_725, 1.71), "es": (35_986_705, 368_692, 1.78),
       "de": (72_111_282, 1_170_968, 1.48)}


def stream_for(zf, code: str, pgids, seed: int) -> tuple[np.ndarray, int]:
    """One shuffled token stream for a language, as int32 vocabulary ids."""
    t0 = time.time()
    rng = np.random.default_rng(seed)
    ids = list(pgids)
    rng.shuffle(ids)
    vocab: dict[str, int] = {}
    pool: list[np.ndarray] = []
    for pgid in ids:
        lines = B.read_member_lines(zf, f"{B.TOKENS_ROOT}/{pgid}_tokens.txt")
        if lines is None:
            continue
        toks = [t for t in lines if t]
        if not toks:
            continue
        assign = vocab.setdefault
        pool.append(np.fromiter((assign(t, len(vocab)) for t in toks),
                                dtype=np.int32, count=len(toks)))
    stream = np.concatenate(pool)
    del pool
    gc.collect()
    print(f"  [{code}] seed {seed}: {stream.size:,} tokens, {len(vocab):,} types "
          f"[{time.time() - t0:.0f}s]", flush=True)
    return stream, len(vocab)


def measure(stream: np.ndarray, vocab_size: int, targets) -> list[dict]:
    """alpha1/alpha2 at the token position where D(t) first reaches each target."""
    first = B.first_occurrence(stream, vocab_size)
    first = np.sort(first[first < stream.size])
    out = []
    for D in targets:
        if D > first.size:
            out.append({"D_target": D, "t_star": np.nan, "types": np.nan,
                        "alpha1": np.nan, "alpha2": np.nan})
            continue
        t_star = int(first[D - 1]) + 1
        counts = np.bincount(stream[:t_star], minlength=vocab_size)
        freq = np.sort(counts[counts > 0])[::-1].astype(np.int64)
        z = two_regime_ols_band(freq)
        out.append({"D_target": D, "t_star": t_star, "types": int(freq.size),
                    "alpha1": z["alpha1"], "alpha2": z["alpha2"],
                    "alpha2_lo": z["alpha2_band"]["lo"],
                    "alpha2_hi": z["alpha2_band"]["hi"]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--targets", type=int, nargs="+", default=list(D_TARGETS))
    args = ap.parse_args()

    langs = B.load_manifest(LANGS)
    rows = []
    with zipfile.ZipFile(B.TOKENS_ZIP) as zf:
        for code in LANGS:
            for seed in range(args.seeds):
                stream, vsize = stream_for(zf, code, langs[code], seed)
                for r in measure(stream, vsize, args.targets):
                    print(f"    D*={r['D_target']:,}  t*={r['t_star']:,}  "
                          f"a1={r['alpha1']:.2f}  a2={r['alpha2']:.2f}", flush=True)
                    rows.append({"lang": code, "seed": seed, **r})
                del stream
                gc.collect()

    df = pd.DataFrame(rows)
    agg = (df.groupby(["lang", "D_target"])
             .agg(t_star=("t_star", "median"), types=("types", "median"),
                  alpha1=("alpha1", "median"), alpha2=("alpha2", "median"),
                  alpha2_min=("alpha2", "min"), alpha2_max=("alpha2", "max"))
             .reset_index())
    agg.to_csv(P.table_path("alpha_matched_vocab.csv"), index=False)
    print("\n" + agg.round(2).to_string(index=False))

    with open(P.table_path("alpha_matched_vocab.md"), "w") as fh:
        for D in args.targets:
            sub = agg[agg.D_target == D]
            fh.write(f"**Every language truncated to a vocabulary of "
                     f"{D:,} types**\n\n"
                     f"| language | tokens needed | $\\alpha_1$ | $\\alpha_2$ | "
                     f"across orderings | $\\alpha_2$ on its full corpus |\n"
                     f"| --- | ---: | ---: | ---: | --- | ---: |\n")
            for _, r in sub.iterrows():
                fh.write(f"| {P.LANG_NAMES[r.lang]} | {int(r.t_star):,} | "
                         f"{r.alpha1:.2f} | {r.alpha2:.2f} | "
                         f"[{r.alpha2_min:.2f}–{r.alpha2_max:.2f}] | "
                         f"{OWN[r.lang][2]:.2f} |\n")
            fh.write("\n")
        fh.write("Each language is cut at the token position where its "
                 "vocabulary first reaches the target, so the rank axis has the "
                 "same length in all five and the tail window sits in the same "
                 "place on every curve. \"Tokens needed\" is itself a measure of "
                 "morphological productivity: the fewer the tokens, the faster "
                 "that language generates new forms.\n")
    print("\nwrote outputs/tables/alpha_matched_vocab.{csv,md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
