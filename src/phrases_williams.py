#!/usr/bin/env python3
"""Phrase (multi-word) frequency distribution — Williams et al. random partition.

Williams, Bagrow, Danforth & Dodds, *Zipf's law holds for phrases, not words*,
Sci. Rep. 5:12209 (2015). A text is cut into **non-overlapping** phrases by
breaking each inter-word slot independently with probability `q`. A phrase `s`
of `k` words that occurs `N(s)` times as an (overlapping) k-gram then has
expected frequency in the partition

    f_q(s) = N(s) * q^(2-b) * (1-q)^(k-1),

where `b` counts how many ends of `s` sit on a text boundary (b = 0 for interior
occurrences, the case used here). The two factors are: both slots flanking the
phrase must break (q^2) and none of its k-1 internal slots may break ((1-q)^(k-1)).

This is why the merge is *not* a plain concatenation of the per-order n-gram
counts, which over-counts long phrases: each order carries a geometric weight
(for q = 1/2, w_k = 2^-(k+1), i.e. proportional to 2^(1-k)), and the whole set of
orders is normalised **once**, not per order.

Everything here works on the committed frequency-of-frequency arrays
(`data_reduced/spgc_<lang>_ngram.npz`): the expected frequency of a phrase
depends on its occurrence count only, so the *identities* of the n-grams are not
needed to build the rank-frequency curve — only how many phrases of each order
share each count.

Caveats, to state in the caption:
  * orders are capped at k = 5 (the reduced data): a fraction (1-q)^5 = 3% of the
    partition's phrases (for q = 1/2) is missing from the tail of the length
    distribution;
  * boundaries are book-level, not sentence-level, so b = 0 is used throughout;
  * n-grams do not cross book boundaries (see `build_reduced.py`).
"""

from __future__ import annotations

import numpy as np

ORDERS = (1, 2, 3, 4, 5)


def williams_weights(q: float = 0.5, orders=ORDERS) -> dict[int, float]:
    """Expected-frequency weight per n-gram order: w_k = q^2 (1-q)^(k-1)."""
    if not 0.0 < q < 1.0:
        raise ValueError("q must be in (0, 1)")
    return {int(k): q ** 2 * (1.0 - q) ** (int(k) - 1) for k in orders}


def phrase_spectrum(ngram: dict, q: float = 0.5, orders=ORDERS) -> dict:
    """Merge the per-order n-gram spectra into one phrase-frequency spectrum.

    Parameters
    ----------
    ngram : dict as returned by `io_reduced.load_ngram` — `ff_vals_k`/`ff_mult_k`
        (occurrence count value and how many distinct k-grams have it).
    q : partition break probability (q = 1/2 is the canonical choice).

    Returns
    -------
    dict with, sorted by descending frequency:
        `freq`   expected phrase frequency of each block (float),
        `mult`   how many distinct phrases share that frequency,
        `rank`   rank of the *last* phrase of each block (cumulative `mult`),
        `prob`   `freq` normalised by the total expected phrase mass (one single
                 normalisation across all orders),
        `total_mass`, `n_phrases`, `q`, `orders`.
    """
    w = williams_weights(q, orders)
    freq, mult = [], []
    for k in orders:
        v = ngram[f"ff_vals_{k}"].astype(np.float64)
        m = ngram[f"ff_mult_{k}"].astype(np.int64)
        freq.append(v * w[int(k)])
        mult.append(m)
    freq = np.concatenate(freq)
    mult = np.concatenate(mult)

    # different (order, count) pairs can land on the same expected frequency
    freq, inv = np.unique(freq, return_inverse=True)
    mult = np.bincount(inv, weights=mult).astype(np.int64)
    order = np.argsort(freq)[::-1]
    freq, mult = freq[order], mult[order]

    total = float((freq * mult).sum())
    return {
        "freq": freq,
        "mult": mult,
        "rank": np.cumsum(mult),
        "prob": freq / total,
        "total_mass": total,
        "n_phrases": int(mult.sum()),
        "q": float(q),
        "orders": tuple(int(k) for k in orders),
    }


def rank_of_frequency(spectrum: dict, f: float) -> int:
    """Rank of a phrase with expected frequency `f` (ties resolved at block end).

    Used to place the annotated example phrases on the curve.
    """
    idx = int(np.searchsorted(-spectrum["freq"], -float(f), side="left"))
    idx = min(idx, spectrum["freq"].size - 1)
    return int(spectrum["rank"][idx])


def expand_for_fit(spectrum: dict, max_points: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(rank, prob) sampled one point per spectrum block — the full curve.

    The blocks are already the distinct levels of the staircase, so this is exact;
    `max_points` log-subsamples them for plotting.
    """
    rank = spectrum["rank"].astype(np.float64)
    prob = spectrum["prob"]
    if max_points is not None and rank.size > max_points:
        idx = np.unique(np.round(
            np.logspace(0, np.log10(rank.size), num=max_points)).astype(np.int64) - 1)
        idx = np.clip(idx, 0, rank.size - 1)
        rank, prob = rank[idx], prob[idx]
    return rank, prob
