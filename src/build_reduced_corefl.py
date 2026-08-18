#!/usr/bin/env python3
"""Step 2 (COREFL): raw learner/native texts -> compact numeric artifacts.

    python src/build_reduced_corefl.py                 # all groups
    python src/build_reduced_corefl.py --shuffles 200  # tighter Heaps band

Reads `data_raw/corefl/<group>/*.txt` (see `import_corefl.py`) and writes
`data_reduced/corefl_<group>.npz`. As for SPGC, the output holds **numbers only**
— no words, no text — so it is tiny, committable, and redistributes nothing of a
CC BY-NC-ND corpus. After this step `data_raw/corefl/` can be deleted.

Per group:
  `freq`        descending int64 count per word type (the whole group pooled)
  `vocab`, `n_tokens`, `n_texts`
  `heaps_t`     log-spaced corpus lengths
  `heaps_mean`, `heaps_lo`, `heaps_hi`
                vocabulary growth D(t), averaged over random orderings of the
                texts, with the 5th-95th percentile band

Why the band: D(t) of a corpus assembled from many short texts depends on the
order they are concatenated in, so any single order is arbitrary. Averaging over
`--shuffles` random orders removes that arbitrariness and shows how much of the
learner/native gap could be sampling noise (very little).

Tokenisation is `text_norm.tokens`, i.e. the SPGC convention, so COREFL
vocabularies are directly comparable with the Gutenberg ones.
"""

from __future__ import annotations

import argparse
import os
import time
from collections import Counter

import numpy as np

from text_norm import tokens as spgc_tokens

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(REPO_ROOT, "data_raw", "corefl")
DATA_REDUCED = os.path.join(REPO_ROOT, "data_reduced")

#: pooled learner corpus = the paper's "L2 learners"
COMBINED = {"learner_all": ("learner_es", "learner_de")}


def read_group(group: str) -> list[list[str]]:
    """Token list of every text of a group, in deterministic (sorted) order."""
    d = os.path.join(DATA_RAW, group)
    if not os.path.isdir(d):
        return []
    texts = []
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".txt"):
            with open(os.path.join(d, fn), encoding="utf-8", errors="replace") as fh:
                texts.append(spgc_tokens(fh.read()))
    return texts


def heaps_curve(texts: list[list[str]], n_shuffles: int, seed: int,
                n_points: int = 60) -> dict:
    """D(t) over random orderings of the texts: grid, mean and 5-95 percentiles."""
    total = sum(len(t) for t in texts)
    grid = np.unique(np.round(np.logspace(1, np.log10(total), n_points)).astype(np.int64))
    rng = np.random.default_rng(seed)
    curves = np.zeros((n_shuffles, grid.size), dtype=np.int64)

    for s in range(n_shuffles):
        order = rng.permutation(len(texts))
        seen: set[str] = set()
        n = 0          # tokens consumed
        g = 0          # next grid point
        for idx in order:
            for tok in texts[idx]:
                seen.add(tok)
                n += 1
                while g < grid.size and grid[g] == n:
                    curves[s, g] = len(seen)
                    g += 1
            if g >= grid.size:
                break
        curves[s, g:] = len(seen)

    return {
        "heaps_t": grid,
        "heaps_mean": curves.mean(axis=0),
        "heaps_lo": np.percentile(curves, 5, axis=0),
        "heaps_hi": np.percentile(curves, 95, axis=0),
        "n_shuffles": np.int64(n_shuffles),
        "seed": np.int64(seed),
    }


def matched_freq(texts: list[list[str]], n_tokens: int, seed: int) -> np.ndarray:
    """Frequency vector of a random `n_tokens`-long prefix of the group.

    Rank-frequency curves cannot be compared across corpora of different sizes,
    so Figure S4 compares every group at the same length. The prefix is drawn by
    shuffling the texts with `seed` and concatenating until `n_tokens`.

    `build_reduced.build_heaps_one_lang` writes a `freq_matched` under the same
    definition for SPGC, but it is a *second implementation*, not this one: there
    the pool is an int32 id array and the vector is a `bincount` of the first
    replicate's ordering, here it is a Counter over a permutation drawn from a
    fresh `seed`. Same statistic, different RNG stream — do not assume one can be
    substituted for the other.
    """
    rng = np.random.default_rng(seed)
    counts, n = Counter(), 0
    for idx in rng.permutation(len(texts)):
        for tok in texts[idx]:
            counts[tok] += 1
            n += 1
            if n >= n_tokens:
                break
        if n >= n_tokens:
            break
    return np.array(sorted(counts.values(), reverse=True), dtype=np.int64)


def build_group(group: str, texts: list[list[str]], n_shuffles: int, seed: int,
                match_tokens: int) -> None:
    t0 = time.time()
    counts = Counter()
    for t in texts:
        counts.update(t)
    freq = np.array(sorted(counts.values(), reverse=True), dtype=np.int64)

    out = {
        "freq": freq,
        "vocab": np.int64(freq.size),
        "n_tokens": np.int64(freq.sum()),
        "n_texts": np.int64(len(texts)),
        "freq_matched": matched_freq(texts, match_tokens, seed),
        "match_tokens": np.int64(match_tokens),
    }
    out.update(heaps_curve(texts, n_shuffles, seed))

    path = os.path.join(DATA_REDUCED, f"corefl_{group}.npz")
    np.savez_compressed(path, **out)
    print(f"  [{group:<12}] texts={len(texts):>5}  tokens={int(out['n_tokens']):>9,}  "
          f"types={freq.size:>7,}  -> {os.path.relpath(path, REPO_ROOT)} "
          f"[{time.time() - t0:.0f}s]", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shuffles", type=int, default=100,
                    help="random text orderings for the Heaps band (default 100)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--groups", nargs="+", default=None)
    ap.add_argument("--match-tokens", type=int, default=1_000_000,
                    help="cap for the extra frequency vector `freq_matched`, used "
                         "when corpora have to be compared at equal length "
                         "(default 1e6: above every COREFL group, so each one "
                         "simply gets its whole corpus)")
    args = ap.parse_args()

    if not os.path.isdir(DATA_RAW):
        raise SystemExit(f"error: {os.path.relpath(DATA_RAW, REPO_ROOT)} not found — "
                         f"see python src/import_corefl.py")
    os.makedirs(DATA_REDUCED, exist_ok=True)

    found = sorted(d for d in os.listdir(DATA_RAW)
                   if os.path.isdir(os.path.join(DATA_RAW, d)))
    loaded = {g: read_group(g) for g in found}
    for name, parts in COMBINED.items():
        pooled = [t for p in parts for t in loaded.get(p, [])]
        if pooled:
            loaded[name] = pooled

    for group in sorted(loaded):
        if args.groups and group not in args.groups:
            continue
        if not loaded[group]:
            continue
        build_group(group, loaded[group], args.shuffles, args.seed,
                    args.match_tokens)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
