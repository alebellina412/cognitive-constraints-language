#!/usr/bin/env python3
"""Step 2 of the SPGC pipeline: build super-compact, committable reduced data.

Reads the raw bulk in ``data_raw/`` (Step 1) and the committed manifest, and
writes tiny numeric-only artifacts into ``data_reduced/`` (a few MB total, NO
strings). After Step 2 the raw zips can be deleted — nothing downstream needs
them.

Two independent modes (the raw zips are large; run them separately):

  --counts   From SPGC-counts (1.5 GB): per-language aggregated 1-gram
             descending count vector  ->  data_reduced/spgc_<lang>_1gram.npz
             (fast, minutes). Feeds Figure 1A + the exponent table.

  --tokens   From SPGC-tokens (6.4 GB): per-language, per-order (1..5) n-gram
             frequency-of-frequency histograms (feeds Figure 1B) and word
             co-occurrence-network degree/strength vectors (feeds Figures 2 and S1)
             ->  data_reduced/spgc_<lang>_ngram.npz
                 data_reduced/spgc_<lang>_wcn.npz
             Heavier: streams one language at a time; n-gram orders are counted
             sequentially so peak memory is one order's counter.

Common:
  --langs en de fr it es   subset of languages (default: all in the manifest)
  --max-tokens N           cap tokens per language (deterministic book order);
                           logged in the output. Default: no cap.

Run from the repo root:
    python src/build_reduced.py --counts
    python src/build_reduced.py --tokens --langs es it        # smallest first
    python src/build_reduced.py --tokens                      # all languages
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile
from collections import Counter

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(REPO_ROOT, "data_raw")
DATA_REDUCED = os.path.join(REPO_ROOT, "data_reduced")
MANIFEST = os.path.join(REPO_ROOT, "manifests", "corpus_manifest.json")

COUNTS_ZIP = os.path.join(DATA_RAW, "SPGC-counts-2018-07-18.zip")
TOKENS_ZIP = os.path.join(DATA_RAW, "SPGC-tokens-2018-07-18.zip")
COUNTS_ROOT = "SPGC-counts-2018-07-18"
TOKENS_ROOT = "SPGC-tokens-2018-07-18"

NGRAM_ORDERS = (1, 2, 3, 4, 5)
LANG_ORDER = ("es", "it", "en", "de", "fr")  # smallest-ish first for --tokens


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def load_manifest(langs: list[str] | None) -> dict[str, list[str]]:
    with open(MANIFEST) as fh:
        man = json.load(fh)
    out = {}
    for code, info in man["languages"].items():
        if langs and code not in langs:
            continue
        out[code] = list(info["pg_ids"])
    return out


def human_int(n: int) -> str:
    return f"{n:,}"


def read_member_lines(zf: zipfile.ZipFile, name: str) -> list[str] | None:
    """Return decoded lines of a zip member, or None if absent."""
    try:
        raw = zf.read(name)
    except KeyError:
        return None
    return raw.decode("utf-8", "replace").splitlines()


# --------------------------------------------------------------------------- #
# --counts : per-language 1-gram frequency vectors
# --------------------------------------------------------------------------- #
def build_counts(langs: dict[str, list[str]]) -> None:
    if not os.path.exists(COUNTS_ZIP):
        sys.exit(f"error: {COUNTS_ZIP} not found (run Step 1 download first).")
    os.makedirs(DATA_REDUCED, exist_ok=True)

    with zipfile.ZipFile(COUNTS_ZIP) as zf:
        for code, pgids in langs.items():
            t0 = time.time()
            agg: Counter[str] = Counter()
            n_books = n_missing = 0
            total_tokens = 0
            for j, pgid in enumerate(pgids, 1):
                name = f"{COUNTS_ROOT}/{pgid}_counts.txt"
                lines = read_member_lines(zf, name)
                if lines is None:
                    n_missing += 1
                    continue
                n_books += 1
                for line in lines:
                    if not line:
                        continue
                    w, _, c = line.rpartition("\t")
                    if not w:
                        continue
                    cnt = int(c)
                    agg[w] += cnt
                    total_tokens += cnt
                if j % 500 == 0:
                    print(f"    [{code}] {j}/{len(pgids)} books, "
                          f"vocab so far {human_int(len(agg))}", flush=True)

            freq = np.fromiter(agg.values(), dtype=np.int64, count=len(agg))
            freq.sort()
            freq = freq[::-1].copy()  # descending

            out = os.path.join(DATA_REDUCED, f"spgc_{code}_1gram.npz")
            np.savez_compressed(
                out,
                freq=freq,
                vocab=np.int64(freq.size),
                total_tokens=np.int64(total_tokens),
                n_books=np.int64(n_books),
                n_books_missing=np.int64(n_missing),
            )
            print(f"[{code}] vocab={human_int(freq.size)} "
                  f"tokens={human_int(total_tokens)} "
                  f"books={n_books} (missing {n_missing}) "
                  f"-> {os.path.relpath(out, REPO_ROOT)} "
                  f"[{time.time() - t0:.0f}s]", flush=True)


# --------------------------------------------------------------------------- #
# --tokens : n-gram freq-of-freq + WCN degree/strength
# --------------------------------------------------------------------------- #
def iter_language_tokens(zf, pgids, max_tokens):
    """Yield (pgid, tokens_list) per book; stop after max_tokens if set."""
    used = 0
    for pgid in pgids:
        name = f"{TOKENS_ROOT}/{pgid}_tokens.txt"
        lines = read_member_lines(zf, name)
        if lines is None:
            yield pgid, None
            continue
        toks = [t for t in lines if t]
        if max_tokens is not None and used + len(toks) > max_tokens:
            toks = toks[: max(0, max_tokens - used)]
        used += len(toks)
        yield pgid, toks
        if max_tokens is not None and used >= max_tokens:
            return


def freq_of_freq(counter_values) -> tuple[np.ndarray, np.ndarray]:
    """Compress a set of counts into (distinct_count_value, multiplicity)."""
    ff = Counter(counter_values)
    vals = np.array(sorted(ff), dtype=np.int64)
    mult = np.array([ff[v] for v in vals], dtype=np.int64)
    return vals, mult


def _counts_from_hashes(chunks: list[np.ndarray]) -> tuple[np.ndarray, int]:
    """Given per-book int64 hash arrays, return (counts_per_distinct, n_distinct).

    Sort-and-run-length on the concatenated hashes: memory peak is ~1-2x the
    hash array (8 bytes/position), vs ~180 bytes/distinct for a string Counter.
    """
    if not chunks:
        return np.empty(0, dtype=np.int64), 0
    allh = np.concatenate(chunks)
    del chunks[:]
    allh.sort()
    change = np.empty(allh.size, dtype=bool)
    change[0] = True
    np.not_equal(allh[1:], allh[:-1], out=change[1:])
    idx = np.flatnonzero(change)
    counts = np.diff(np.append(idx, allh.size)).astype(np.int64)
    return counts, counts.size


def build_ngrams_one_lang(zf, code, pgids, max_tokens) -> dict:
    """Per-order n-gram freq-of-freq via 64-bit hashing (memory-safe).

    Each n-gram is mapped to Python's 64-bit ``hash``. This keeps peak memory at
    ~2 GB even for 170M-token languages, where an exact string Counter would
    need ~27 GB.

    Counting is exact up to hash collisions, and at 64 bits there are none to
    speak of: the birthday estimate is N^2 / 2^65, which for the largest case
    here (~1.4e8 distinct 5-grams) is 5e-4 collisions *in total*, and 7e-6 for
    English bigrams. That is why the artifacts come out byte-identical from one
    run to the next even though ``hash`` is salted per process (PYTHONHASHSEED):
    only the *identities* depend on the salt, the multiplicities do not, and with
    no collisions the two are independent. Set PYTHONHASHSEED if you want that
    guaranteed rather than overwhelmingly likely.
    """
    result = {}
    for k in NGRAM_ORDERS:
        t0 = time.time()
        chunks: list[np.ndarray] = []
        n_books = n_missing = n_positions = 0
        for pgid, toks in iter_language_tokens(zf, pgids, max_tokens):
            if toks is None:
                n_missing += 1
                continue
            n_books += 1
            m = len(toks) - k + 1
            if m <= 0:
                continue
            if k == 1:
                arr = np.fromiter((hash(t) for t in toks),
                                  dtype=np.int64, count=len(toks))
            else:
                arr = np.fromiter(
                    (hash(" ".join(toks[i:i + k])) for i in range(m)),
                    dtype=np.int64, count=m)
            chunks.append(arr)
            n_positions += m
        counts, n_distinct = _counts_from_hashes(chunks)
        vals, mult = np.unique(counts, return_counts=True)
        result[f"ff_vals_{k}"] = vals.astype(np.int64)
        result[f"ff_mult_{k}"] = mult.astype(np.int64)
        result[f"n_distinct_{k}"] = np.int64(n_distinct)
        result[f"n_positions_{k}"] = np.int64(n_positions)
        print(f"    [{code}] {k}-gram: distinct={human_int(n_distinct)} "
              f"positions={human_int(n_positions)} books={n_books} "
              f"(missing {n_missing}) [{time.time() - t0:.0f}s]", flush=True)
        del counts
    return result


def build_wcn_one_lang(zf, code, pgids, max_tokens) -> dict:
    """Undirected word co-occurrence network from consecutive tokens.

    strength[w] = total incident bigram weight; degree[w] = distinct neighbours.
    Returns descending strength and degree vectors (numeric only).
    """
    t0 = time.time()
    edges: Counter[tuple] = Counter()   # (min,max) undirected -> weight
    for pgid, toks in iter_language_tokens(zf, pgids, max_tokens):
        if toks is None:
            continue
        prev = None
        for w in toks:
            if prev is not None and prev != w:
                e = (prev, w) if prev < w else (w, prev)
                edges[e] += 1
            prev = w

    strength: Counter[str] = Counter()
    degree: Counter[str] = Counter()
    for (a, b), wgt in edges.items():
        strength[a] += wgt
        strength[b] += wgt
        degree[a] += 1
        degree[b] += 1

    s = np.fromiter(strength.values(), dtype=np.int64, count=len(strength))
    d = np.fromiter(degree.values(), dtype=np.int64, count=len(degree))
    s.sort(); s = s[::-1].copy()
    d.sort(); d = d[::-1].copy()
    print(f"    [{code}] WCN: nodes={human_int(len(strength))} "
          f"edges={human_int(len(edges))} [{time.time() - t0:.0f}s]", flush=True)
    return {
        "strength": s,
        "degree": d,
        "n_nodes": np.int64(len(strength)),
        "n_edges": np.int64(len(edges)),
    }


def build_tokens(langs: dict[str, list[str]], max_tokens: int | None) -> None:
    if not os.path.exists(TOKENS_ZIP):
        sys.exit(f"error: {TOKENS_ZIP} not found (run Step 1 download first).")
    os.makedirs(DATA_REDUCED, exist_ok=True)

    # process in smallest-first order for whatever subset was requested
    ordered = [c for c in LANG_ORDER if c in langs] + \
              [c for c in langs if c not in LANG_ORDER]

    with zipfile.ZipFile(TOKENS_ZIP) as zf:
        for code in ordered:
            pgids = langs[code]
            print(f"\n=== tokens [{code}] {len(pgids)} books "
                  f"(max_tokens={max_tokens}) ===", flush=True)

            ng = build_ngrams_one_lang(zf, code, pgids, max_tokens)
            out_ng = os.path.join(DATA_REDUCED, f"spgc_{code}_ngram.npz")
            np.savez_compressed(out_ng, max_tokens=np.int64(max_tokens or -1), **ng)
            print(f"[{code}] n-grams -> {os.path.relpath(out_ng, REPO_ROOT)}",
                  flush=True)

            wcn = build_wcn_one_lang(zf, code, pgids, max_tokens)
            out_wcn = os.path.join(DATA_REDUCED, f"spgc_{code}_wcn.npz")
            np.savez_compressed(out_wcn, max_tokens=np.int64(max_tokens or -1), **wcn)
            print(f"[{code}] WCN -> {os.path.relpath(out_wcn, REPO_ROOT)}",
                  flush=True)


# --------------------------------------------------------------------------- #
# --heaps : vocabulary growth D(t), the native reference curve of Figure S4
# --------------------------------------------------------------------------- #
def first_occurrence(stream: np.ndarray, vocab_size: int) -> np.ndarray:
    """Index at which each vocabulary id first appears in `stream` (else len).

    Vectorised: a fancy-index assignment keeps the *last* write for duplicate
    indices, so scattering the positions in decreasing order leaves the
    smallest one — the first occurrence. O(n) and no argsort, which matters at
    n = 10^8 (an argsort would need a 0.8 GB index array and ~20 s per replicate).
    """
    n = stream.size
    first = np.full(vocab_size, n, dtype=np.int64)
    first[stream[::-1]] = np.arange(n - 1, -1, -1, dtype=np.int64)
    return first


def build_heaps_one_lang(zf, code, pgids, max_tokens, n_shuffles, seed,
                         match_tokens=40_000, n_points=60) -> dict:
    """D(t) up to `max_tokens`, averaged over random orderings of the books.

    Same statistic and same output keys as `build_reduced_corefl.py`, so the
    Gutenberg curve and the COREFL curves are directly comparable in Figure S4.
    A pool of books is read once (enough to cover `max_tokens` a few times over
    when the corpus allows it) and only the order is re-drawn per replicate.

    Tokens are held as int32 vocabulary ids, not Python strings: at
    max_tokens = 10^8 a list-of-strings pool would need tens of GB, while the
    id array needs 4 bytes per token. D(t) then follows from the sorted
    first-occurrence indices by a single `searchsorted`.
    """
    t0 = time.time()
    rng = np.random.default_rng(seed)
    ids = list(pgids)
    rng.shuffle(ids)

    vocab: dict[str, int] = {}
    pool: list[np.ndarray] = []
    pooled = 0
    for pgid in ids:
        lines = read_member_lines(zf, f"{TOKENS_ROOT}/{pgid}_tokens.txt")
        if lines is None:
            continue
        toks = [t for t in lines if t]
        if not toks:
            continue
        assign = vocab.setdefault
        pool.append(np.fromiter((assign(t, len(vocab)) for t in toks),
                                dtype=np.int32, count=len(toks)))
        pooled += len(toks)
        if pooled >= 3 * max_tokens:      # room to reshuffle without repeating
            break

    total = min(pooled, max_tokens)
    n_match = min(match_tokens, total)
    vocab_size = len(vocab)
    grid = np.unique(np.round(np.logspace(1, np.log10(total), n_points)).astype(np.int64))

    buf = np.empty(pooled, dtype=np.int32)
    curves = np.zeros((n_shuffles, grid.size), dtype=np.int64)
    freq_matched = None
    for s in range(n_shuffles):
        offset = 0
        for idx in rng.permutation(len(pool)):
            book = pool[idx]
            buf[offset:offset + book.size] = book
            offset += book.size
        stream = buf[:total]
        first = first_occurrence(stream, vocab_size)
        first.sort()
        # D(t) = how many ids first appeared strictly before position t
        curves[s] = np.searchsorted(first, grid, side="left")
        if freq_matched is None:
            # size-matched frequency vector: same statistic as
            # build_reduced_corefl.py, so the Gutenberg curve can be put next
            # to the COREFL ones at equal length. Taken from this same ordering.
            counts = np.bincount(stream[:n_match], minlength=vocab_size)
            freq_matched = np.sort(counts[counts > 0])[::-1].astype(np.int64)

    print(f"    [{code}] Heaps: pool={len(pool)} books ({pooled:,} tokens, "
          f"{vocab_size:,} types), t<={total:,}, {n_shuffles} shuffles "
          f"[{time.time() - t0:.0f}s]", flush=True)
    return {
        "heaps_t": grid,
        "heaps_mean": curves.mean(axis=0),
        "heaps_lo": np.percentile(curves, 5, axis=0),
        "heaps_hi": np.percentile(curves, 95, axis=0),
        "n_shuffles": np.int64(n_shuffles),
        "seed": np.int64(seed),
        "n_books_pool": np.int64(len(pool)),
        "n_tokens_pool": np.int64(pooled),
        "vocab_pool": np.int64(vocab_size),
        "max_tokens": np.int64(total),
        "freq_matched": freq_matched,
        "match_tokens": np.int64(n_match),
    }


def build_heaps(langs, max_tokens, n_shuffles, seed, match_tokens,
                n_points=60) -> None:
    if not os.path.exists(TOKENS_ZIP):
        sys.exit(f"error: {TOKENS_ZIP} not found (run Step 1 download first).")
    os.makedirs(DATA_REDUCED, exist_ok=True)
    with zipfile.ZipFile(TOKENS_ZIP) as zf:
        for code, pgids in langs.items():
            h = build_heaps_one_lang(zf, code, pgids, max_tokens, n_shuffles,
                                     seed, match_tokens, n_points)
            out = os.path.join(DATA_REDUCED, f"spgc_{code}_heaps.npz")
            np.savez_compressed(out, **h)
            print(f"[{code}] Heaps -> {os.path.relpath(out, REPO_ROOT)}", flush=True)


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--counts", action="store_true", help="build 1-gram vectors")
    ap.add_argument("--tokens", action="store_true", help="build n-gram + WCN")
    ap.add_argument("--langs", nargs="+", choices=["en", "de", "fr", "it", "es"],
                    help="subset of languages (default: all)")
    ap.add_argument("--heaps", action="store_true",
                    help="build the vocabulary-growth curve D(t) (Figure S4)")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="cap tokens per language (deterministic order)")
    ap.add_argument("--heaps-max-tokens", type=int, default=1_000_000,
                    help="upper end of the Heaps curve (default 1e6)")
    ap.add_argument("--heaps-points", type=int, default=60,
                    help="points on the Heaps log grid (default 60)")
    ap.add_argument("--shuffles", type=int, default=50,
                    help="random book orderings for the Heaps band (default 50)")
    ap.add_argument("--match-tokens", type=int, default=1_000_000,
                    help="length of the size-matched frequency vector (Figure S4)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not (args.counts or args.tokens or args.heaps):
        ap.error("choose --counts, --tokens and/or --heaps")

    langs = load_manifest(args.langs)
    print(f"languages: {', '.join(f'{k}({len(v)})' for k, v in langs.items())}")
    print(f"output   : {DATA_REDUCED}")

    if args.counts:
        print("\n### MODE: --counts (1-gram frequency vectors) ###")
        build_counts(langs)
    if args.tokens:
        print("\n### MODE: --tokens (n-gram freq-of-freq + WCN) ###")
        build_tokens(langs, args.max_tokens)
    if args.heaps:
        print("\n### MODE: --heaps (vocabulary growth D(t)) ###")
        build_heaps(langs, args.heaps_max_tokens, args.shuffles, args.seed,
                    args.match_tokens, args.heaps_points)
    return 0


if __name__ == "__main__":
    sys.exit(main())
