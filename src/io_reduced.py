#!/usr/bin/env python3
"""Readers for the committed `data_reduced/` artifacts (Step 2 output).

Every figure notebook loads its data through here, so the on-disk schema is
described in exactly one place (mirrored in README.md). Nothing else in the
pipeline may touch `data_raw/`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_REDUCED = os.path.join(REPO_ROOT, "data_reduced")

LANGS = ["en", "fr", "it", "es", "de"]


def _path(lang: str, kind: str) -> str:
    """Path of a reduced artifact, for a corpus language or a simulation.

    A two-letter language code names a corpus file (`spgc_en_wcn.npz`); anything
    else is taken to be a full prefix, which is how a simulation reduced by
    `src/build_reduced_sim.py` is read (`sim_calibrated_wcn.npz`). The model
    therefore goes through *these* readers, not a parallel set of its own — the
    point of writing the simulation into the corpus schema in the first place.
    """
    stem = f"spgc_{lang}" if lang in LANGS else lang
    path = os.path.join(DATA_REDUCED, f"{stem}_{kind}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{os.path.relpath(path, REPO_ROOT)} missing — run Step 2 "
            f"(python src/build_reduced.py), src/build_reduced_sim.py for a "
            f"simulation, or pull the committed data_reduced/."
        )
    return path


@dataclass
class OneGram:
    """Aggregated 1-gram counts for one language."""
    lang: str
    freq: np.ndarray      # descending int64 count per word type
    vocab: int            # number of word types
    total_tokens: int
    n_books: int
    n_books_missing: int

    @property
    def prob(self) -> np.ndarray:
        """Normalized p(R) = f(R) / sum(f)."""
        return self.freq / self.freq.sum()


def load_1gram(lang: str) -> OneGram:
    """Load `data_reduced/spgc_<lang>_1gram.npz`."""
    with np.load(_path(lang, "1gram")) as d:
        return OneGram(
            lang=lang,
            freq=np.sort(d["freq"])[::-1].astype(np.int64),
            vocab=int(d["vocab"]),
            total_tokens=int(d["total_tokens"]),
            n_books=int(d["n_books"]),
            n_books_missing=int(d["n_books_missing"]),
        )


def load_ngram(lang: str) -> dict:
    """Load the n-gram frequency-of-frequency file as a plain dict of arrays.

    Keys: `ff_vals_k`, `ff_mult_k`, `n_distinct_k`, `n_positions_k` for k=1..5,
    plus `max_tokens`.
    """
    with np.load(_path(lang, "ngram")) as d:
        return {k: d[k] for k in d.files}


def rank_from_ff(vals: np.ndarray, mult: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rank-frequency staircase from a frequency-of-frequency pair.

    `vals` are distinct occurrence counts and `mult` how many items have each.
    Returns (rank, freq) with one point per distinct count value: `rank` is the
    rank of the *last* item sharing that count, so the pair traces the exact
    rank-frequency curve of the (unavailable) full vector.
    """
    order = np.argsort(vals)[::-1]
    freq = np.asarray(vals)[order].astype(np.int64)
    rank = np.cumsum(np.asarray(mult)[order]).astype(np.int64)
    return rank, freq


def load_ngram_rank_freq(lang: str, order: int) -> tuple[np.ndarray, np.ndarray]:
    """(rank, freq) rank-frequency curve of the `order`-grams of `lang`."""
    d = load_ngram(lang)
    return rank_from_ff(d[f"ff_vals_{order}"], d[f"ff_mult_{order}"])


def load_wcn(lang: str) -> dict:
    """Load the word co-occurrence network summary as a plain dict of arrays.

    Keys: `strength`, `degree` (both descending int64), `n_nodes`, `n_edges`,
    `max_tokens`.
    """
    with np.load(_path(lang, "wcn")) as d:
        return {k: d[k] for k in d.files}


def sim_freq_from_wcn(wcn: dict) -> np.ndarray:
    """Rank-ordered frequencies of a simulated run, recovered from its WCN.

    A node's strength counts its occurrences twice, once per neighbour, so
    `f = s/2` — except at the two ends of the simulated sequence. The first and
    the last token each have one neighbour instead of two, so those two nodes
    carry `s = 2f - 1`. (Excluding self-loops removes endpoints in pairs, so it
    cannot change the parity.) Every simulated run therefore has exactly two
    nodes of odd strength.

    Hence `(s + 1) // 2` rather than `s // 2`: it is the same value for every
    even strength, and it is the only one that recovers `f` at the two ends.
    Plain halving truncates them, which costs nothing on the initial hub — it
    has millions of occurrences — but sends the final token to zero whenever it
    happens to be a hapax, dropping a real type from the vocabulary. That is
    why a run's reconstructed length can come out one short of `n_nodes`; with
    the `+1` it matches for every run.
    """
    return (np.sort(wcn["strength"])[::-1].astype(np.int64) + 1) // 2


def load_corefl(group: str) -> dict:
    """Load `data_reduced/corefl_<group>.npz` (Figure S4).

    Groups, as `build_reduced_corefl.py` writes them: `learner_all` (the paper's
    L2 learners = the pooled `learner_es` + `learner_de`), the per-L1 groups
    `learner_{cn,cz,de,es,et,fr,gr,it,tr}`, and `native_en`. Only `learner_all`
    and `native_en` are read by anything; the rest fall out of the import and are
    kept because they are cheap.

    Keys: `freq`, `freq_matched`, `match_tokens`, `vocab`, `n_tokens`,
    `n_texts`, `heaps_t`, `heaps_mean`, `heaps_lo`, `heaps_hi`, `n_shuffles`,
    `seed`.
    """
    path = os.path.join(DATA_REDUCED, f"corefl_{group}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{os.path.relpath(path, REPO_ROOT)} missing — run "
            f"python src/build_reduced_corefl.py (see src/import_corefl.py)."
        )
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


def load_heaps(lang: str) -> dict:
    """Load the vocabulary-growth curve `spgc_<lang>_heaps.npz`.

    For a corpus this carries the same keys as `load_corefl`, so the Gutenberg
    reference and the COREFL groups are drawn and fitted identically in Figure
    S4. It also reads a *simulation* prefix (`sim_calibrated`, ...), and there
    the file is the smaller one written by `build_reduced_sim.build_heaps`:
    `heaps_t`, `heaps_mean`, `heaps_lo`, `heaps_hi`, `n_shuffles`, `vocab_pool`,
    `max_tokens` only. A simulation has no `freq`, no `freq_matched` and no
    `match_tokens` — its frequency vector comes from `sim_freq_from_wcn`.
    """
    with np.load(_path(lang, "heaps")) as d:
        return {k: d[k] for k in d.files}


def load_phrases(lang: str) -> dict:
    """Occurrence counts of the Figure 1B example phrases (Step 2b).

    Written by `src/count_phrases.py`; the only reduced artifact that carries
    strings, because those phrases have to be named on the figure.
    """
    path = os.path.join(DATA_REDUCED, f"spgc_{lang}_phrases.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{os.path.relpath(path, REPO_ROOT)} missing — run "
            f"python src/count_phrases.py (needs data_raw/)."
        )
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


if __name__ == "__main__":
    for c in LANGS:
        g = load_1gram(c)
        print(f"{c}: vocab={g.vocab:>9,}  tokens={g.total_tokens:>12,}  "
              f"books={g.n_books:>5,} (missing {g.n_books_missing})")
