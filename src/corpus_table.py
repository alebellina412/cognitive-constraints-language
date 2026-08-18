#!/usr/bin/env python3
"""Composition of the five language corpora — books, tokens and types.

This is the first table of the supplementary information, and the only one that
describes the input rather than a result. Every number is read back from the
reduced data actually used by the figures, not from the manifest, so the table
cannot drift away from what was analysed: if a book is missing from the frozen
SPGC release, the token and type counts here reflect that.

English is a random subsample drawn with a fixed seed until the aggregated
vocabulary reaches 5.5e5 types; the other four languages use every
single-language book in the release. `n_books_missing` records manifest books
absent from the frozen zips — a handful per language, which is why the book
counts can sit slightly below the manifest length.

Usage:
    python src/corpus_table.py
"""

from __future__ import annotations

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import plotting as P                                               # noqa: E402
from io_reduced import load_1gram                                  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(REPO, "manifests", "corpus_manifest.json")

LANGS = P.LEGEND_ORDER


def build() -> pd.DataFrame:
    """One row per language, read back from the reduced 1-gram data."""
    manifest = json.load(open(MANIFEST))
    rows = []
    for c in LANGS:
        g = load_1gram(c)
        listed = len(manifest["languages"][c]["pg_ids"])
        rows.append({
            "language": P.LANG_NAMES[c],
            "books": int(g.n_books),
            "books_in_manifest": listed,
            "books_missing": int(g.n_books_missing),
            "tokens": int(g.total_tokens),
            "types": int(g.vocab),
            "mode": "subsample" if c == "en" else "all",
        })
    return pd.DataFrame(rows, index=LANGS)


def write_table(table: pd.DataFrame) -> str:
    """Write the corpus table as CSV and as the markdown of SI Table S1."""
    table.to_csv(P.table_path("corpus_composition.csv"), index_label="lang")

    head = ("| language | books | tokens | types |\n"
            "| --- | ---: | ---: | ---: |\n")
    body = "".join(f"| {r.language} | {r.books:,} | {r.tokens:,} | {r.types:,} |\n"
                   for r in table.itertuples())
    note = ("\nAll five corpora are drawn from SPGC release 2018-07-18. English is a "
            "random subsample drawn with a fixed seed "
            "(`numpy.random.default_rng(seed=0)`) until the aggregated vocabulary "
            "reaches $5.5\\cdot10^5$ types; the other four use every single-language "
            "book in the release. Analyses that require a common corpus size "
            "truncate the English stream to $10^8$ tokens.\n\n"
            "Counts are read back from `data_reduced/`, i.e. from the data the "
            "figures were actually computed on. A few manifest books are absent from "
            "the frozen release (`books_missing` in the CSV), which is why a book "
            "count can sit just below the manifest length.\n")
    md = head + body + note
    with open(P.table_path("corpus_composition.md"), "w") as fh:
        fh.write(md)
    return md


def main() -> int:
    table = build()
    print(table.to_string())
    print()
    print(write_table(table))
    print("wrote outputs/tables/corpus_composition.{csv,md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
