#!/usr/bin/env python3
"""Rank of every annotated concept on the Figure 1B phrase curve (SI table).

Measures the claim that Figure 1 makes: does the *same concept*, expressed as
one compound in one language and as a phrase in another, sit at a comparable
rank on the phrase distribution?

Inputs (all committed, no raw corpus):
  manifests/annotation_phrases.json   the 26 concepts, fixed BEFORE any count
  data_reduced/spgc_<lang>_phrases.json   occurrence counts (src/count_phrases.py)
  data_reduced/spgc_<lang>_ngram.npz      the phrase spectrum

Outputs:
  outputs/tables/fig1B_annotations.csv     one row per (language, concept)
  outputs/tables/fig1B_concept_ranks.{csv,md}   the SI table, one row per concept
  outputs/tables/fig1B_pairwise.{csv,md}   German-vs-each-language discrepancy

Method notes that belong in the SI:

* The rank comes from the Williams expected frequency f_q = N q^2 (1-q)^(k-1)
  located on the committed phrase spectrum, i.e. exactly the curve Figure 1B
  plots. Both N (raw occurrences) and the rank are reported, because the
  Williams weight penalises a k-gram by 2^(k-1) relative to a compound and a
  reader must be able to see that factor separately.
* The comparison to report is **pairwise**, not the spread over all five
  languages: a max/min over five languages always picks the two extremes and
  overstates the disagreement.
* Concepts are classified as low- or high-paraphrase-variability on linguistic
  grounds (one fixed conventional expression per language, versus a free
  composition or one of several near-synonyms). This classification was made
  *after* seeing the ranks, so it is reported as a robustness check and never as
  a selection criterion — and the full table is printed whatever the outcome.
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
from phrases_williams import (phrase_spectrum, rank_of_frequency,  # noqa: E402
                              williams_weights)

REPO_ROOT = os.path.dirname(DATA_REDUCED)
MANIFEST = os.path.join(REPO_ROOT, "manifests", "annotation_phrases.json")
LANGS = ["en", "fr", "it", "es", "de"]
Q = 0.5
ORDERS = (1, 2, 3, 4, 5)

#: one fixed conventional expression per language (see the docstring caveat)
LOW_VARIABILITY = ["water bottle", "post office", "dining room", "bedroom",
                   "birthday", "handkerchief", "lighthouse", "armchair",
                   "coachman", "new year", "sunrise"]


def build() -> pd.DataFrame:
    """Long table: one row per (language, concept), with N, k and the rank."""
    spec = json.load(open(MANIFEST))
    era_of = {c: era for era, concepts in spec["_era"].items() for c in concepts}
    spectra = {c: phrase_spectrum(load_ngram(c), Q, ORDERS) for c in LANGS}
    weights = williams_weights(Q, ORDERS)

    rows = []
    for code in LANGS:
        payload = load_phrases(code)
        for p in payload["phrases"]:
            w = weights.get(p["k"])
            if w is None or p["count"] == 0:      # order > 5, or absent
                f = prob = rank = np.nan
            else:
                f = p["count"] * w
                prob = f / spectra[code]["total_mass"]
                rank = rank_of_frequency(spectra[code], f)
            rows.append({"lang": code, "language": P.LANG_NAMES[code],
                         "concept": p["concept"], "era": era_of.get(p["concept"], "?"),
                         "surface": p["surface"], "k": p["k"], "N": p["count"],
                         "rel_freq": p["count"] / payload["n_tokens"],
                         "f_q": f, "p": prob, "rank": rank})
    return pd.DataFrame(rows)


def pairwise_vs_german(df: pd.DataFrame) -> pd.DataFrame:
    """Rank ratio of each language to German, per concept.

    German is the reference because it is the language the paper's argument is
    about: it lexicalises with compounds where the others spell phrases out, so
    it is where a rank agreement is informative.
    """
    rank = df.pivot(index="concept", columns="lang", values="rank")
    out = {}
    for lang in ["en", "fr", "it", "es"]:
        out[f"vs_{lang}"] = rank[lang] / rank["de"]
    return pd.DataFrame(out)


def _discrepancy(ratios: pd.Series) -> float:
    """Median factor by which two languages disagree, direction-free."""
    v = ratios.dropna().to_numpy()
    return float(10 ** np.median(np.abs(np.log10(v)))) if v.size else float("nan")


def tables() -> dict:
    """Build every annotated-concept table, write it, and return the frames.

    The single entry point shared by the CLI and `notebooks/figure_1B.ipynb`.
    Returns the keys `annotations` (long, one row per language x concept),
    `concept_ranks` (the SI table), `pairwise` (German-vs-each-language summary)
    and `coverage` (how many concepts are attested in all five languages).
    """
    df = build()
    df.to_csv(P.table_path("fig1B_annotations.csv"), index=False)

    ratios = pairwise_vs_german(df)
    rank = df.pivot(index="concept", columns="lang", values="rank")[LANGS]
    kk = df.pivot(index="concept", columns="lang", values="k")[LANGS]
    era = df.groupby("concept")["era"].first()
    spec = json.load(open(MANIFEST))
    order = [c for c in spec["concepts"] if c in rank.index]

    concept_ranks = pd.concat([era[order].rename("era"),
                               rank.loc[order].add_prefix("rank_"),
                               kk.loc[order].add_prefix("k_"),
                               ratios.loc[order]], axis=1)
    concept_ranks.to_csv(P.table_path("fig1B_concept_ranks.csv"))

    period = [c for c in spec["_era"]["period"] if c in ratios.index]
    low = [c for c in LOW_VARIABILITY if c in ratios.index]
    high = [c for c in period if c not in low]

    summary = pd.DataFrame({
        "all period concepts": {f"DE vs {l.upper()}": _discrepancy(ratios.loc[period, f"vs_{l}"])
                                for l in ["en", "fr", "it", "es"]},
        "low variability": {f"DE vs {l.upper()}": _discrepancy(ratios.loc[low, f"vs_{l}"])
                            for l in ["en", "fr", "it", "es"]},
        "high variability": {f"DE vs {l.upper()}": _discrepancy(ratios.loc[high, f"vs_{l}"])
                             for l in ["en", "fr", "it", "es"]},
    })
    summary.to_csv(P.table_path("fig1B_pairwise.csv"))

    with open(P.table_path("fig1B_pairwise.md"), "w") as fh:
        fh.write("| pair | all period concepts | low variability | high variability |\n")
        fh.write("| --- | ---: | ---: | ---: |\n")
        for pair, row in summary.iterrows():
            fh.write(f"| {pair} | {row.iloc[0]:.2f}x | {row.iloc[1]:.2f}x | {row.iloc[2]:.2f}x |\n")
        fh.write(f"\nMedian factor by which the rank of the same concept differs "
                 f"between German and each other language, over the "
                 f"{len(period)} period concepts (n={len(low)} low-variability, "
                 f"{len(high)} high-variability). Restricting to low-variability "
                 f"expressions does not change the conclusion, which is why the "
                 f"full set is reported.\n")

    # the SI table itself, one row per concept, ranks in every language
    with open(P.table_path("fig1B_concept_ranks.md"), "w") as fh:
        fh.write("| concept | era | " + " | ".join(P.LANG_NAMES[c] for c in LANGS)
                 + " |\n| --- | --- | " + " | ".join("---:" for _ in LANGS) + " |\n")
        for c in order:
            cells = []
            for lang in LANGS:
                r = concept_ranks.loc[c, f"rank_{lang}"]
                k = concept_ranks.loc[c, f"k_{lang}"]
                cells.append("--" if pd.isna(r) else f"{int(r):,} ({int(k)}w)")
            fh.write(f"| {c} | {concept_ranks.loc[c, 'era']} | "
                     + " | ".join(cells) + " |\n")
        fh.write(f"\nRank of each concept on the Figure 1B phrase curve, with the "
                 f"number of orthographic words of the surface form in brackets. "
                 f"The {len(spec['concepts'])} concepts were fixed before any count "
                 f"was looked at; `--` marks a concept absent from that language's "
                 f"corpus. Ranks follow from the Williams expected frequency "
                 f"$f_q=N q^2(1-q)^{{k-1}}$ located on the committed phrase "
                 f"spectrum, so a $k$-word phrase is penalised by $2^{{k-1}}$ "
                 f"relative to a compound by construction.\n")

    coverage = pd.DataFrame(
        [{"set": "period concepts", "n": len(period),
          "attested in all 5": int((rank.loc[period].notna().sum(axis=1) == 5).sum())},
         {"set": "all concepts", "n": len(order),
          "attested in all 5": int((rank.notna().sum(axis=1) == 5).sum())}]
    ).set_index("set")

    return {"annotations": df, "concept_ranks": concept_ranks,
            "pairwise": summary, "coverage": coverage, "ratios": ratios,
            "period": period, "low_variability": low, "high_variability": high}


def main() -> int:
    t = tables()
    print(t["pairwise"].round(2).to_string())
    print()
    print(t["coverage"].to_string())
    print("wrote outputs/tables/fig1B_{annotations,concept_ranks,pairwise}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
