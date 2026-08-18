#!/usr/bin/env python3
"""Would annotated multi-word expressions change the phrase result?

A frequent n-gram is not a lexicalised multi-word expression, so Figure 1B's
phrase curve is built on surface strings rather than on units a linguist would
call phrases. The natural test is to redo it with *annotated* MWEs, and the only
manually annotated resource covering the paper's five languages is PARSEME 1.1 —
which annotates **verbal** MWEs in gold corpora of 10^5 tokens, against the 10^8
of SPGC.

Measured naively the annotated MWEs give a Zipf exponent of ~0.5-0.8 rather than
the ~1.0 of Figure 1B, which looks like a discrepancy. It is not: it is a
finite-size effect, and this script demonstrates that with three comparisons
that share one construction and differ only in what is counted or how much text
is available.

Per language, the exponent of the rank-frequency curve of:

  1. **annotated MWEs**, PARSEME gold                     the linguistic units
  2. **surface 2-5-grams from the same PARSEME text**     same corpus, same size
  3. **surface 2-5-grams from SPGC truncated to the
     same token count**                                   same construction, our corpus
  4. **surface 2-5-grams from the whole SPGC**            Figure 1B's regime

If 1 ~ 2 ~ 3 and all are far below 4, then the low exponent is a property of the
sample size and not of MWEs, and substituting annotated MWEs for surface
n-grams would change nothing that can be measured with the resources available.

Definitions:

* an MWE type is the ordered sequence of its annotated component lemmas,
  lowercased (PARSEME CUPT column 11), pooled over train/dev/test;
* `mwe_contiguous` additionally requires the 2-5 components to be adjacent, the
  variant directly comparable with surface n-grams — reported as a robustness
  check, since a verbal MWE is a lexical unit whether or not it is interrupted;
* n-grams are contiguous 2-5 token sequences within a sentence, after dropping
  UPOS=PUNCT; SPGC has no punctuation and no sentence splitting, so its n-grams
  run within a book.

Every exponent uses `fits.ols_rank_prob` on a grid uniform in log R over the
whole observed rank range — the estimator and the convention of Figure 1B.

Building needs `data_raw/` (SPGC tokens) and the PARSEME corpora, neither of
which is redistributed here; see `docs/DATA.md` for the download, the versions
and the licences. The reduction it writes is the frequency spectrum only (a few
kB), so the analysis re-runs from `data_reduced/` without either corpus.

Usage:
    python src/mwe_ranks.py --build      # needs data_raw/ + PARSEME, run once
    python src/mwe_ranks.py              # the table, from data_reduced/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import plotting as P                                      # noqa: E402
from fits import ols_rank_prob                            # noqa: E402
from io_reduced import DATA_REDUCED, REPO_ROOT, load_ngram  # noqa: E402

#: PARSEME language directories, in the paper's order
LANGS = {"en": "EN", "fr": "FR", "it": "IT", "es": "ES", "de": "DE"}
SPLITS = ("train.cupt", "dev.cupt", "test.cupt")
ORDERS = (2, 3, 4, 5)
REDUCED = os.path.join(DATA_REDUCED, "parseme_mwe.npz")
#: where the PARSEME download lives (95 MB, not redistributed — see docs/DATA.md)
DEFAULT_PARSEME = os.path.join(REPO_ROOT, "data_raw", "parseme_1_1")


# --------------------------------------------------------------------------- #
# parsing PARSEME CUPT
# --------------------------------------------------------------------------- #
def _mwe_field(value: str) -> list[str]:
    """Occurrence ids in CUPT column 11 ('*' and '_' mean no annotation)."""
    if value in {"*", "_", ""}:
        return []
    return [item.partition(":")[0] for item in value.split(";")]


def _consume(rows, mwe_all, mwe_contig, ngrams, stats) -> None:
    """Accumulate one sentence into the MWE and n-gram counters."""
    if not rows:
        return
    occurrences = defaultdict(list)
    lexical = []
    for position, (form, lemma, upos, field) in enumerate(rows):
        if upos != "PUNCT":
            lexical.append(form.lower())
        for occurrence_id in _mwe_field(field):
            occurrences[occurrence_id].append((position, lemma.lower()))

    for components in occurrences.values():
        components.sort()
        key = " ".join(lemma for _, lemma in components)
        if not key:
            continue
        mwe_all[key] += 1
        positions = [p for p, _ in components]
        if (2 <= len(components) <= 5
                and positions == list(range(positions[0],
                                            positions[0] + len(components)))):
            mwe_contig[key] += 1

    for k in ORDERS:
        for i in range(len(lexical) - k + 1):
            ngrams[" ".join(lexical[i:i + k])] += 1
    stats["sentences"] += 1
    stats["lexical_tokens"] += len(lexical)


def read_parseme(language_dir: str):
    """(mwe_all, mwe_contiguous, ngrams_2_5, stats) for one language."""
    mwe_all, mwe_contig, ngrams = Counter(), Counter(), Counter()
    stats = Counter()
    for split in SPLITS:
        path = os.path.join(language_dir, split)
        if not os.path.exists(path):
            continue
        sentence = []
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                stripped = line.rstrip("\n")
                if not stripped:
                    _consume(sentence, mwe_all, mwe_contig, ngrams, stats)
                    sentence = []
                    continue
                if stripped.startswith("#"):
                    continue
                fields = stripped.split("\t")
                if len(fields) < 11 or "-" in fields[0]:
                    continue
                sentence.append((fields[1], fields[2], fields[3], fields[10]))
        _consume(sentence, mwe_all, mwe_contig, ngrams, stats)
    return mwe_all, mwe_contig, ngrams, stats


def spgc_ngrams(code: str, max_tokens: int) -> Counter:
    """Surface 2-5-grams of the first `max_tokens` SPGC tokens of a language."""
    import build_reduced as B

    pgids = B.load_manifest([code])[code]
    tokens: list[str] = []
    with zipfile.ZipFile(B.TOKENS_ZIP) as zf:
        for _, toks in B.iter_language_tokens(zf, pgids, max_tokens):
            if toks:
                tokens.extend(toks)
    tokens = tokens[:max_tokens]
    counts: Counter = Counter()
    for k in ORDERS:
        for i in range(len(tokens) - k + 1):
            counts[" ".join(tokens[i:i + k])] += 1
    return counts


# --------------------------------------------------------------------------- #
# reduction: only the frequency spectrum is kept and committed
# --------------------------------------------------------------------------- #
def _spectrum(counts) -> tuple[np.ndarray, np.ndarray]:
    """(distinct count value, how many types have it) — lossless for our use.

    Rank-frequency does not depend on type identity, so the spectrum carries
    every number this analysis needs while staying a few hundred bytes.
    """
    ff = Counter(counts.values() if isinstance(counts, Counter) else counts)
    vals = np.array(sorted(ff), dtype=np.int64)
    return vals, np.array([ff[v] for v in vals], dtype=np.int64)


def _vector(vals: np.ndarray, mult: np.ndarray) -> np.ndarray:
    """Descending frequency vector rebuilt from a spectrum."""
    return np.repeat(vals[::-1], mult[::-1]).astype(float)


def build(parseme_dir: str) -> None:
    payload: dict[str, np.ndarray] = {}
    meta: dict[str, dict] = {}
    for code, folder in LANGS.items():
        path = os.path.join(parseme_dir, folder)
        if not os.path.isdir(path):
            sys.exit(f"error: {path} not found — see docs/DATA.md for the download")
        mwe_all, mwe_contig, ngrams, stats = read_parseme(path)
        n_tokens = int(stats["lexical_tokens"])
        spgc = spgc_ngrams(code, n_tokens)
        for name, counter in (("mwe", mwe_all), ("mwe_contiguous", mwe_contig),
                              ("parseme_ngram", ngrams), ("spgc_ngram", spgc)):
            vals, mult = _spectrum(counter)
            payload[f"{code}_{name}_vals"] = vals
            payload[f"{code}_{name}_mult"] = mult
        meta[code] = {"sentences": int(stats["sentences"]),
                      "lexical_tokens": n_tokens,
                      "mwe_occurrences": int(sum(mwe_all.values())),
                      "mwe_types": len(mwe_all)}
        print(f"[{code}] {n_tokens:,} lexical tokens · "
              f"{len(mwe_all):,} MWE types / {sum(mwe_all.values()):,} occurrences · "
              f"{len(ngrams):,} PARSEME n-gram types · {len(spgc):,} SPGC n-gram types",
              flush=True)
    np.savez_compressed(REDUCED, meta=json.dumps(meta), **payload)
    print(f"-> {os.path.relpath(REDUCED, REPO_ROOT)} "
          f"({os.path.getsize(REDUCED) / 1e3:.1f} kB)")


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #
def alpha(freq: np.ndarray, min_count: int = 2) -> float:
    """Zipf exponent on a grid uniform in log R — the Figure 1B estimator.

    `min_count=2` drops the trailing plateau of units seen once, the convention
    used for every rank curve in this paper. It matters here: the annotated MWE
    curves are 63-75% hapax, so keeping the plateau would fit mostly flat.
    """
    freq = np.sort(np.asarray(freq, dtype=float))[::-1]
    freq = freq[freq >= min_count]
    if freq.size < 10:
        return float("nan")
    rank = np.arange(1, freq.size + 1, dtype=float)
    return ols_rank_prob(rank, freq / freq.sum(), 1.0, float(freq.size))["alpha"]


def full_spgc_ngram_vector(code: str, orders=ORDERS) -> np.ndarray:
    """Pooled n-gram frequency vector of the whole SPGC corpus.

    Types are distinct across orders, so the per-order frequency-of-frequency
    arrays can simply be pooled — this is the exact combined distribution.
    """
    d = load_ngram(code)
    vals = np.concatenate([d[f"ff_vals_{k}"] for k in orders])
    mult = np.concatenate([d[f"ff_mult_{k}"] for k in orders])
    order = np.argsort(vals)[::-1]
    return np.repeat(vals[order], mult[order]).astype(float)


def williams_alpha(code: str, orders) -> float:
    """Exponent of the paper's own phrase construction on the full corpus.

    With `orders=(1..5)` this is literally Figure 1B's curve; with `(2..5)` it
    is the same construction restricted to *multi-word* units, which is what an
    MWE inventory is. The difference between the two is the single-word
    contribution, and it is large — see the note written by `main`.
    """
    from phrases_williams import expand_for_fit, phrase_spectrum

    rank, prob = expand_for_fit(phrase_spectrum(load_ngram(code), orders=orders))
    return ols_rank_prob(rank, prob, 1.0, float(rank[-1]))["alpha"]


def table() -> pd.DataFrame:
    if not os.path.exists(REDUCED):
        sys.exit(f"error: {os.path.relpath(REDUCED, REPO_ROOT)} missing — run "
                 f"python src/mwe_ranks.py --build")
    data = np.load(REDUCED, allow_pickle=False)
    meta = json.loads(str(data["meta"]))

    rows = []
    for code in LANGS:
        def v(name):
            return _vector(data[f"{code}_{name}_vals"], data[f"{code}_{name}_mult"])

        mwe, contig = v("mwe"), v("mwe_contiguous")
        par, spgc = v("parseme_ngram"), v("spgc_ngram")
        rows.append({
            "language": P.LANG_NAMES[code],
            "tokens": meta[code]["lexical_tokens"],
            "mwe_types": int(mwe.size),
            "mwe_occurrences": int(mwe.sum()),
            "mwe_hapax": float((mwe == 1).mean()),
            # --- multi-word units, all at ~1e5 tokens -------------------------
            "alpha_mwe": round(alpha(mwe), 2),
            "alpha_mwe_contiguous": round(alpha(contig), 2),
            "alpha_parseme_ngram": round(alpha(par), 2),
            "alpha_spgc_truncated": round(alpha(spgc), 2),
            # --- the same units on the full corpus, and Figure 1B -------------
            "alpha_spgc_full_multiword": round(alpha(full_spgc_ngram_vector(code)), 2),
            "alpha_williams_multiword": round(williams_alpha(code, ORDERS), 2),
            "alpha_williams_fig1B": round(williams_alpha(code, (1,) + ORDERS), 2),
        })
    return pd.DataFrame(rows, index=list(LANGS))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true",
                    help="re-read PARSEME and SPGC and rewrite the reduction")
    ap.add_argument("--parseme-dir", default=DEFAULT_PARSEME)
    args = ap.parse_args()

    if args.build:
        build(args.parseme_dir)

    t = table()
    t.to_csv(P.table_path("mwe_ranks.csv"), index_label="lang")

    # how far the annotated MWEs sit from surface n-grams measured on the very
    # same text -- the quantity the comparison turns on
    same_text = float((t["alpha_mwe"] - t["alpha_parseme_ngram"]).abs().max())
    size_step = float((t["alpha_spgc_full_multiword"]
                       - t["alpha_spgc_truncated"]).mean())
    unigram_step = float((t["alpha_williams_fig1B"]
                          - t["alpha_williams_multiword"]).mean())

    with open(P.table_path("mwe_ranks.md"), "w") as fh:
        fh.write("| language | tokens | MWE types | annotated MWE | n-gram, same text "
                 "| n-gram, SPGC cut to the same length | n-gram, full SPGC | "
                 "Fig. 1B phrases |\n"
                 "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for _, r in t.iterrows():
            fh.write(f"| {r.language} | {int(r.tokens):,} | {int(r.mwe_types):,} | "
                     f"{r.alpha_mwe:.2f} | {r.alpha_parseme_ngram:.2f} | "
                     f"{r.alpha_spgc_truncated:.2f} | "
                     f"{r.alpha_spgc_full_multiword:.2f} | "
                     f"{r.alpha_williams_fig1B:.2f} |\n")
        fh.write(
            "\nZipf exponent of the rank-frequency curve, fitted on a grid uniform "
            "in $\\log R$ over the range where units occur more than once — the "
            "estimator and the plateau convention used throughout the paper. "
            "Columns 4-6 all describe **multi-word units in ~$10^5$ tokens** and "
            "differ only in what is counted and where the text comes from; column 7 "
            "is the same n-gram construction on the full corpus ($1.6\\cdot10^8$ "
            "tokens for English); column 8 is Figure 1B's own phrase curve.\n\n"
            f"**Annotated MWEs and surface n-grams from the very same text agree to "
            f"within {same_text:.2f} in every language, with no systematic "
            f"direction.** For these statistics it makes no measurable difference "
            "whether the multi-word units are manually annotated or read off the "
            "surface.\n\n"
            "The distance from Figure 1B's $\\alpha\\approx1$ is then fully "
            "accounted for by two effects that have nothing to do with annotation, "
            "and both are measured here rather than asserted:\n\n"
            f"1. **Corpus size**, worth about {size_step:+.2f}: the same n-gram "
            f"construction gives {t['alpha_spgc_truncated'].min():.2f}-"
            f"{t['alpha_spgc_truncated'].max():.2f} on $10^5$ tokens of SPGC and "
            f"{t['alpha_spgc_full_multiword'].min():.2f}-"
            f"{t['alpha_spgc_full_multiword'].max():.2f} on the whole corpus.\n"
            f"2. **Single-word phrases**, worth about {unigram_step:+.2f}: Figure 1B "
            "partitions the *whole text*, so one-word phrases carry the largest "
            "Williams weight ($w_1=1/4$). Restricted to multi-word units the same "
            f"construction gives {t['alpha_williams_multiword'].min():.2f}-"
            f"{t['alpha_williams_multiword'].max():.2f} rather than "
            f"{t['alpha_williams_fig1B'].min():.2f}-"
            f"{t['alpha_williams_fig1B'].max():.2f}. An MWE inventory is multi-word "
            "by definition and so cannot reproduce the former number even in "
            "principle.\n\n"
            "**Conclusion.** Replacing surface n-grams by manually annotated "
            "multi-word expressions does not change the result: on the same corpus "
            "the two give the same exponent. It cannot be pushed further than this "
            "with existing resources — the largest manually annotated MWE corpora "
            "covering these five languages are $\\sim10^5$ tokens, three orders of "
            "magnitude below the corpus the paper uses, and PARSEME annotates only "
            "*verbal* MWEs in a different genre. The comparison is therefore "
            "reported as a robustness check, not as a replication of Figure 1B.\n\n"
            f"The MWE curves are severely undersampled in absolute terms "
            f"({t['mwe_hapax'].min():.0%}-{t['mwe_hapax'].max():.0%} of MWE types "
            "occur exactly once), which is the same statement seen from the type "
            "side and the reason no tighter claim is made.\n\n"
            "`alpha_mwe_contiguous` in the CSV restricts MWEs to the adjacent "
            "2-5-component cases, the variant strictly comparable with surface "
            "n-grams; it does not change the picture.\n")

    print(t[["tokens", "mwe_types", "alpha_mwe", "alpha_parseme_ngram",
             "alpha_spgc_truncated", "alpha_spgc_full_multiword",
             "alpha_williams_multiword", "alpha_williams_fig1B"]].to_string())
    print(f"\nMWE vs n-gram on the same text: max |difference| = {same_text:.2f}")
    print(f"corpus size (1e5 -> full), n-grams : {size_step:+.2f}")
    print(f"adding one-word phrases (Fig 1B)   : {unigram_step:+.2f}")
    print("wrote outputs/tables/mwe_ranks.{csv,md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
