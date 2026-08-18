#!/usr/bin/env python3
"""Step 2b: count the Figure 1B example phrases in the raw SPGC tokens.

The reduced n-gram data is hashed and identity-free, which is what keeps it
small — but Figure 1B points at specific phrases ("water bottle",
"Wasserflasche", ...), so their occurrence counts have to be measured once from
the raw tokens. This script is the only place besides Step 2 that needs
`data_raw/`; its output is a few hundred bytes per language.

    python src/count_phrases.py                 # all languages in the manifest
    python src/count_phrases.py --langs es it

Reads  `manifests/annotation_phrases.json` (surface forms, as printed in the
paper) and writes `data_reduced/spgc_<lang>_phrases.json`:

    {"lang", "n_books", "n_books_missing", "n_tokens",
     "phrases": [{"concept", "surface", "tokens", "k", "count"}, ...]}

Counting matches `build_reduced.py` exactly: same manifest books, same book
order, n-grams never cross a book boundary. The rank of each phrase on the
Figure 1B curve is *not* computed here — it follows from the committed phrase
spectrum (`src/phrases_williams.py`), so it stays reproducible without the raw
corpus.

Tokenisation of the target phrases mirrors the SPGC pipeline: lowercase, and
split on whitespace and apostrophes, since SPGC stores "d'eau" as the two tokens
"d" and "eau" ("dell'autobus" -> "dell", "autobus").
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile

from build_reduced import (
    DATA_REDUCED,
    REPO_ROOT,
    TOKENS_ZIP,
    iter_language_tokens,
    load_manifest,
)
from text_norm import phrase_tokens

PHRASES = os.path.join(REPO_ROOT, "manifests", "annotation_phrases.json")

#: SPGC-style tokens of a printed phrase ("bouteille d'eau" -> d, eau)
tokenise = phrase_tokens


def count_one_lang(zf, code: str, pgids: list[str], targets: dict) -> dict:
    """Occurrence count of every target phrase, as an exact k-gram scan."""
    seqs = {concept: tokenise(surface) for concept, surface in targets.items()}
    counts = {concept: 0 for concept in seqs}
    # index by first token so most positions cost a single set lookup
    firsts: dict[str, list[str]] = {}
    for concept, seq in seqs.items():
        firsts.setdefault(seq[0], []).append(concept)

    t0 = time.time()
    n_books = n_missing = n_tokens = 0
    for _pgid, toks in iter_language_tokens(zf, pgids, None):
        if toks is None:
            n_missing += 1
            continue
        n_books += 1
        n_tokens += len(toks)
        for i, tok in enumerate(toks):
            hit = firsts.get(tok)
            if hit is None:
                continue
            for concept in hit:
                seq = seqs[concept]
                if len(seq) == 1 or tuple(toks[i:i + len(seq)]) == seq:
                    counts[concept] += 1

    phrases = [{"concept": concept,
                "surface": targets[concept],
                "tokens": list(seqs[concept]),
                "k": len(seqs[concept]),
                "count": counts[concept]}
               for concept in targets]
    print(f"  [{code}] {n_books} books, {n_tokens:,} tokens "
          f"[{time.time() - t0:.0f}s]", flush=True)
    for p in phrases:
        print(f"      {p['surface']:<24} k={p['k']}  N={p['count']:,}")
    return {"lang": code, "n_books": n_books, "n_books_missing": n_missing,
            "n_tokens": n_tokens, "phrases": phrases}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--langs", nargs="+", default=None)
    args = ap.parse_args()

    if not os.path.exists(TOKENS_ZIP):
        sys.exit(f"error: {TOKENS_ZIP} not found (run Step 1 download first).")
    with open(PHRASES) as fh:
        spec = json.load(fh)
    langs = load_manifest(args.langs)
    os.makedirs(DATA_REDUCED, exist_ok=True)

    with zipfile.ZipFile(TOKENS_ZIP) as zf:
        for code, pgids in langs.items():
            targets = spec["languages"].get(code)
            if not targets:
                print(f"  [{code}] no target phrases, skipped")
                continue
            out = count_one_lang(zf, code, pgids, targets)
            path = os.path.join(DATA_REDUCED, f"spgc_{code}_phrases.json")
            with open(path, "w") as fh:
                json.dump(out, fh, ensure_ascii=False, indent=1)
            print(f"  -> {os.path.relpath(path, REPO_ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
