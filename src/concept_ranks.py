#!/usr/bin/env python3
"""Do the same concepts occupy the same position in every language?

The all-versus-all version of the annotated-concept analysis, replacing the
German-versus-each-language table of `src/phrase_ranks.py`.

Two questions, answered on the 22 concepts the paper reports (see
`reported_ranks` below for how that set is defined):

1. **How far apart are two languages?** For each unordered pair, the median over
   concepts of |log10(rank_A / rank_B)|, reported as a factor and in decades.
2. **How dispersed is one concept over the five languages?** max - min of
   log10(rank), in decades, and its median over concepts.

Both are computed under two rank constructions:

* **weighted** — the Williams expected frequency f_q = N q^2 (1-q)^(k-1) placed
  on the Williams phrase spectrum. This is the curve Figure 1B plots, and a
  k-word phrase is down-weighted by 2^(k-1) relative to a one-word compound.
* **unweighted** — the raw occurrence count N placed on the raw pooled n-gram
  spectrum (orders 1..5, no weight). This removes the 2^(k-1) factor.

The comparison isolates what the weighting does. German lexicalises as one
compound what the others spell out as a phrase, so it is the language the
weighting moves most; if the languages converge once the weighting is removed,
the residual discrepancy is a property of how n-grams are counted, not evidence
that the languages place the concept differently.

Inputs: `manifests/annotation_phrases.json`, `data_reduced/spgc_<lang>_phrases.json`,
`data_reduced/spgc_<lang>_ngram.npz`. No raw corpus.

Outputs:
  outputs/tables/concept_ranks_all5.{csv,md}    per concept, per language
  outputs/tables/concept_pairwise.{csv,md}      the 5x5 discrepancy matrices
  outputs/tables/concept_dispersion.{csv,md}    the summary

Cross-language rank *correlations* are deliberately not computed here. Several
English surface forms in the manifest are period variants (looking glass,
railway station), which suppresses every correlation involving English while
leaving the dispersion essentially unchanged. Dispersion is the robust statistic
and it is the one the paper reports.

Usage:
    python src/concept_ranks.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import plotting as P                                              # noqa: E402
from io_reduced import DATA_REDUCED, load_ngram, load_phrases      # noqa: E402
from phrases_williams import (ORDERS, phrase_spectrum,             # noqa: E402
                              rank_of_frequency, williams_weights)

REPO_ROOT = os.path.dirname(DATA_REDUCED)
MANIFEST = os.path.join(REPO_ROOT, "manifests", "annotation_phrases.json")
LANGS = ["en", "fr", "it", "es", "de"]
Q = 0.5

#: Anachronistic for a corpus that is largely nineteenth-century, where these
#: concepts appear a handful of times if at all and the rank is dominated by
#: noise. Excluded before the results were looked at.
MODERN = {"telephone", "motor car", "bus stop", "bus"}


def reported_ranks(df: pd.DataFrame):
    """The concept set the paper reports, defined once for every output.

    A concept is kept if it is attested in all five languages on both the
    weighted and the unweighted curve, and is not in `MODERN`. Of the 26
    concepts in the manifest this leaves 22.

    Defining the set here rather than inside a figure is what keeps the number
    of concepts the same everywhere: a table and a figure built on different
    subsets would report different dispersions for the same claim.

    Returns `(rank_weighted, rank_unweighted, k, order, dropped)`, the three
    frames indexed by concept in manifest order.
    """
    rank_w = df.pivot(index="concept", columns="lang", values="rank_weighted")[LANGS]
    rank_r = df.pivot(index="concept", columns="lang", values="rank_unweighted")[LANGS]
    attested = rank_w.notna().all(axis=1) & rank_r.notna().all(axis=1)
    dropped = sorted(set(rank_w.index[~attested]) | (set(rank_w.index) & MODERN))
    spec = json.load(open(MANIFEST))
    order = [c for c in spec["concepts"]
             if c in rank_w.index[attested] and c not in MODERN]
    kk = df.pivot(index="concept", columns="lang", values="k")[LANGS]
    return rank_w.loc[order], rank_r.loc[order], kk.loc[order], order, dropped


def raw_spectrum(ngram: dict, orders=ORDERS) -> dict:
    """The pooled n-gram spectrum with no Williams weight (w_k = 1).

    Same merge and same output keys as `phrases_williams.phrase_spectrum`, so
    `rank_of_frequency` applies unchanged. Every n-gram of order <= 5 competes
    on its raw occurrence count, which is what "without the 2^(k-1) rescaling"
    means operationally.
    """
    freq, mult = [], []
    for k in orders:
        freq.append(ngram[f"ff_vals_{k}"].astype(np.float64))
        mult.append(ngram[f"ff_mult_{k}"].astype(np.int64))
    freq = np.concatenate(freq)
    mult = np.concatenate(mult)
    freq, inv = np.unique(freq, return_inverse=True)
    mult = np.bincount(inv, weights=mult).astype(np.int64)
    order = np.argsort(freq)[::-1]
    freq, mult = freq[order], mult[order]
    total = float((freq * mult).sum())
    return {"freq": freq, "mult": mult, "rank": np.cumsum(mult),
            "prob": freq / total, "total_mass": total,
            "n_phrases": int(mult.sum()), "q": None, "orders": tuple(orders)}


def build() -> tuple[pd.DataFrame, dict]:
    """Long table with both rank constructions, plus the panel span per language."""
    spec = json.load(open(MANIFEST))
    era_of = {c: era for era, concepts in spec["_era"].items() for c in concepts}
    w = williams_weights(Q, ORDERS)

    rows, span = [], {}
    for code in LANGS:
        ng = load_ngram(code)
        sp_w = phrase_spectrum(ng, Q, ORDERS)
        sp_r = raw_spectrum(ng, ORDERS)
        span[code] = {"weighted": np.log10(sp_w["n_phrases"]),
                      "unweighted": np.log10(sp_r["n_phrases"])}
        for p in load_phrases(code)["phrases"]:
            k, N = p["k"], p["count"]
            if k not in w or N == 0:
                rank_w = rank_r = np.nan
            else:
                rank_w = rank_of_frequency(sp_w, N * w[k])
                rank_r = rank_of_frequency(sp_r, float(N))
            rows.append({"lang": code, "language": P.LANG_NAMES[code],
                         "concept": p["concept"],
                         "era": era_of.get(p["concept"], "?"),
                         "surface": p["surface"], "k": k, "N": N,
                         "rank_weighted": rank_w, "rank_unweighted": rank_r})
    return pd.DataFrame(rows), span


def pairwise_matrix(rank: pd.DataFrame) -> pd.DataFrame:
    """Median |log10 rank ratio| between every pair of languages, as a factor."""
    M = pd.DataFrame(np.nan, index=LANGS, columns=LANGS, dtype=float)
    for a in LANGS:
        for b in LANGS:
            if a == b:
                M.loc[a, b] = 1.0
                continue
            v = np.abs(np.log10(rank[a] / rank[b]).dropna().to_numpy())
            M.loc[a, b] = float(10 ** np.median(v)) if v.size else np.nan
    return M


def dispersion(rank: pd.DataFrame) -> pd.Series:
    """Spread of one concept over the five languages, in decades."""
    lg = np.log10(rank[LANGS])
    return (lg.max(axis=1) - lg.min(axis=1)).rename("decades")


def null_within_language(rank: pd.DataFrame) -> float:
    """How far apart are two *different* concepts inside the *same* language?

    The scale against which the cross-language agreement has to be read. Without
    it, "the same concept lands 0.5 decades apart across languages" is a number
    with no unit of comparison: the claim is only meaningful if two arbitrary
    concepts inside one language land much further apart than that.

    Returns the median over languages of the median over concept pairs of
    |log10(rank_i / rank_j)|, in decades.
    """
    per_lang = []
    for c in LANGS:
        v = np.log10(rank[c].dropna().to_numpy())
        d = np.abs(v[:, None] - v[None, :])
        per_lang.append(float(np.median(d[~np.eye(v.size, dtype=bool)])))
    return float(np.median(per_lang))


def main() -> int:
    df, span = build()
    rank_w, rank_r, kk, order, dropped = reported_ranks(df)
    era = df.drop_duplicates("concept").set_index("concept")["era"]

    print(f"{len(order)} concepts reported; "
          f"dropped {len(dropped)}: {', '.join(dropped)}\n")

    disp_w, disp_r = dispersion(rank_w), dispersion(rank_r)
    mean_span = float(np.mean([span[c]["weighted"] for c in LANGS]))

    per_concept = pd.DataFrame({
        "era": era[order],
        **{f"rank_{c}": rank_w[c].astype("Int64") for c in LANGS},
        **{f"k_{c}": kk[c].astype("Int64") for c in LANGS},
        "spread_dec_weighted": disp_w.round(2),
        "spread_dec_unweighted": disp_r.round(2),
    })
    per_concept.to_csv(P.table_path("concept_ranks_all5.csv"))

    M_w, M_r = pairwise_matrix(rank_w), pairwise_matrix(rank_r)
    pd.concat({"weighted": M_w, "unweighted": M_r}).to_csv(
        P.table_path("concept_pairwise.csv"))

    off_w = M_w.to_numpy()[~np.eye(len(LANGS), dtype=bool)]
    off_r = M_r.to_numpy()[~np.eye(len(LANGS), dtype=bool)]
    de_w = M_w.loc["de", [c for c in LANGS if c != "de"]].to_numpy()
    de_r = M_r.loc["de", [c for c in LANGS if c != "de"]].to_numpy()
    nde_w = M_w.loc[[c for c in LANGS if c != "de"], [c for c in LANGS if c != "de"]]
    nde_r = M_r.loc[[c for c in LANGS if c != "de"], [c for c in LANGS if c != "de"]]
    nde_w = nde_w.to_numpy()[~np.eye(4, dtype=bool)]
    nde_r = nde_r.to_numpy()[~np.eye(4, dtype=bool)]

    null_w, null_r = null_within_language(rank_w), null_within_language(rank_r)

    summary = pd.DataFrame({
        "weighted (Williams, Fig. 1B)": {
            "median pair discrepancy": np.median(off_w),
            "median pair, German excluded": np.median(nde_w),
            "median pair, German vs rest": np.median(de_w),
            "same concept across languages (decades)": np.log10(np.median(off_w)),
            "different concepts, same language (decades)": null_w,
            "median concept spread (decades)": float(disp_w.median()),
            "as % of the panel": 100 * float(disp_w.median()) / mean_span},
        "unweighted (raw n-gram counts)": {
            "median pair discrepancy": np.median(off_r),
            "median pair, German excluded": np.median(nde_r),
            "median pair, German vs rest": np.median(de_r),
            "same concept across languages (decades)": np.log10(np.median(off_r)),
            "different concepts, same language (decades)": null_r,
            "median concept spread (decades)": float(disp_r.median()),
            "as % of the panel": 100 * float(disp_r.median()) / mean_span},
    })
    summary.to_csv(P.table_path("concept_dispersion.csv"))
    print(summary.round(2).to_string(), "\n")
    print(f"mean panel span: {mean_span:.2f} decades\n")
    print("weighted (Williams):\n", M_w.round(2).to_string())
    print("\nunweighted (raw counts):\n", M_r.round(2).to_string())

    with open(P.table_path("concept_pairwise.md"), "w") as fh:
        for name, M in (("Weighted (Williams expected frequency, the Figure 1B curve)", M_w),
                        ("Unweighted (raw n-gram occurrence counts)", M_r)):
            fh.write(f"**{name}**\n\n| | "
                     + " | ".join(P.LANG_NAMES[c] for c in LANGS) + " |\n| --- | "
                     + " | ".join("---:" for _ in LANGS) + " |\n")
            for a in LANGS:
                fh.write(f"| {P.LANG_NAMES[a]} | "
                         + " | ".join("--" if a == b else f"{M.loc[a, b]:.2f}x"
                                      for b in LANGS) + " |\n")
            fh.write("\n")
        fh.write(f"Median over the {len(order)} concepts of the factor by which "
                 f"the rank of the same concept differs between two languages. "
                 f"The panel spans {mean_span:.1f} decades of rank.\n")

    with open(P.table_path("concept_dispersion.md"), "w") as fh:
        fh.write("| | weighted (Fig. 1B) | unweighted (raw counts) |\n"
                 "| --- | ---: | ---: |\n")
        for row in summary.index:
            a, b = summary.loc[row]
            unit = "x" if "discrepancy" in row or "German" in row else ""
            fh.write(f"| {row} | {a:.2f}{unit} | {b:.2f}{unit} |\n")
        fh.write(f"\nOver the {len(order)} concepts the paper reports: the "
                 f"manifest's 26 less the four that postdate a corpus which is "
                 f"largely nineteenth-century (`MODERN` in "
                 f"`src/concept_ranks.py`). Nothing is dropped for lack of "
                 f"attestation — all {len(order)} are attested in all five "
                 f"languages. The weighted construction down-weights a k-word "
                 f"phrase by 2^(k-1) relative to a one-word compound; the "
                 f"unweighted one does not.\n")

    with open(P.table_path("concept_ranks_all5.md"), "w") as fh:
        fh.write("| concept | " + " | ".join(P.LANG_NAMES[c] for c in LANGS)
                 + " | spread |\n| --- | "
                 + " | ".join("---:" for _ in LANGS) + " | ---: |\n")
        for c in order:
            cells = [f"{int(rank_w.loc[c, l]):,} ({int(kk.loc[c, l])}w)" for l in LANGS]
            fh.write(f"| {c} | " + " | ".join(cells)
                     + f" | {disp_w[c]:.2f} dec |\n")
        fh.write(f"\nRank of each concept on the weighted phrase curve, with the "
                 f"number of orthographic words of the surface form in that "
                 f"language in brackets. \"Spread\" is max - min of log10(rank) "
                 f"over the five languages.\n")

    print("\nwrote outputs/tables/concept_{ranks_all5,pairwise,dispersion}.{csv,md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
