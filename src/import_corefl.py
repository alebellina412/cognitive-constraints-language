#!/usr/bin/env python3
"""Step 1b (COREFL): import and verify the learner corpus used by Figure S4.

COREFL — the Corpus of English as a Foreign Language (Lozano et al.),
http://corefl.learnercorpora.com — is served through a web interface only:
there is no bulk-download URL and no API (checked: `/search` and `/search_simple`
return the same JavaScript shell, with no form action and no downloadable
archive), so the fetch itself cannot be scripted. Once per subcorpus:

    1. open  http://corefl.learnercorpora.com/search_simple
    2. pick the Subcorpus (Learners of L2 English / the native controls), L1 any
    3. leave the Words field EMPTY and press Search
    4. press Download, choose purpose and the "Texts only" format
    5. extract every archive into one folder

"Texts only" matters: the metadata variant puts a header inside each text file,
which the tokeniser would count as running text.

Everything after that is scripted. Point this script at the extracted folder and
it imports the texts into `data_raw/corefl/`, classifies them, and records a
manifest (sha256 per file + per-group totals) so any later copy can be checked
for identity:

    python src/import_corefl.py --from-dir ~/Downloads/corefl --write-manifest
    python src/import_corefl.py --check

File names carry the metadata, so groups are read off the name and not off the
folder layout (the old analysis relied on the folder and mixed native English
with native Spanish, which is what Figure S4 was built on):

    <L1>_<mode>[_<CEFR level>]_<age>_<...>.txt
    es_wr_b1_19_11_13_rn.txt   L1 Spanish, written, level B1  -> learner of English
    en_sp_19_14_pg.txt         native English, spoken (no level) -> native control

A CEFR level means the text is a learner production in English; no level means it
is a native control in the language of the prefix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(REPO_ROOT, "data_raw", "corefl")
MANIFEST = os.path.join(REPO_ROOT, "manifests", "corefl_manifest.json")

CORPUS_URL = "http://corefl.learnercorpora.com/search"
LEVELS = ("a1", "a2", "b1", "b2", "c1", "c2")

#: published corpus composition (http://corefl.learnercorpora.com/statistics),
#: for a sanity check on how complete a given download is
OFFICIAL = {
    "learner_es": (2873, 459902),
    "learner_de": (453, 121371),
    "native_en": (230, 51954),
    "native_es": (1202, 299320),
    "native_de": (103, 25996),
}


def classify(filename: str) -> tuple[str, str] | None:
    """(group, mode) of a COREFL file name, or None if it does not parse.

    A few names in the corpus have an empty medium field (`es__a2_25_2_3_mmr`),
    so the fields are read positionally rather than with one strict pattern: the
    L1 is the first field, `wr`/`sp` the second when present, and a CEFR level in
    the next position marks a learner text (no level = native control).
    """
    parts = os.path.basename(filename).lower()
    if parts.endswith(".txt"):
        parts = parts[:-4]
    fields = parts.split("_")
    if len(fields) < 3 or len(fields[0]) != 2 or not fields[0].isalpha():
        return None
    l1, rest = fields[0], fields[1:]
    mode = rest[0] if rest[0] in ("wr", "sp") else None
    tail = rest[1:] if (mode or rest[0] == "") else rest
    level = tail[0] if tail and tail[0] in LEVELS else None
    group = f"learner_{l1}" if level else f"native_{l1}"
    return group, mode or "unknown"


def scan(root: str) -> dict[str, list[str]]:
    """Group -> sorted list of .txt paths found under `root` (recursively)."""
    out: dict[str, list[str]] = {}
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if not fn.endswith(".txt"):
                continue
            kind = classify(fn)
            if kind is None:
                print(f"  ! unrecognised name, skipped: {fn}")
                continue
            out.setdefault(kind[0], []).append(os.path.join(dirpath, fn))
    return {g: sorted(v) for g, v in sorted(out.items())}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def report(groups: dict[str, list[str]]) -> dict:
    """Print the composition next to the published one; return the summary."""
    print(f"{'group':<12} {'files':>6} {'words':>10} | {'official':>16}  coverage")
    summary = {}
    for g, paths in groups.items():
        words = sum(len(open(p, encoding='utf-8', errors='replace').read().split())
                    for p in paths)
        off = OFFICIAL.get(g)
        cov = f"{words / off[1]:6.0%}" if off else "     -"
        off_s = f"{off[0]:>5} / {off[1]:>8,}" if off else "-"
        print(f"{g:<12} {len(paths):>6} {words:>10,} | {off_s:>16}  {cov}")
        summary[g] = {"files": len(paths), "words": words}
    return summary


def do_import(src: str) -> None:
    src = os.path.expanduser(src)
    if not os.path.exists(src):
        sys.exit(f"error: {src} does not exist")

    if not os.path.isdir(src):
        sys.exit(f"error: {src} is not a directory — extract the download first "
                 f"and pass the folder that holds the .txt files")

    groups = scan(src)
    if not groups:
        listing = []
        for dirpath, _dirs, files in os.walk(src):
            listing += [os.path.join(dirpath, f) for f in files][:10]
            if len(listing) >= 10:
                break
        sys.exit(f"error: no COREFL .txt files found under {src}\n"
                 f"       (looking for names like es_wr_b1_19_11_13_rn.txt)\n"
                 f"       what is there instead:\n         "
                 + "\n         ".join(listing[:10] or ["<nothing>"]))
    os.makedirs(DATA_RAW, exist_ok=True)
    n = 0
    for g, paths in groups.items():
        dest_dir = os.path.join(DATA_RAW, g)
        os.makedirs(dest_dir, exist_ok=True)
        for p in paths:
            dest = os.path.join(dest_dir, os.path.basename(p))
            if os.path.abspath(p) == os.path.abspath(dest):
                continue          # re-importing data_raw/corefl onto itself
            shutil.copy2(p, dest)
            n += 1
    print(f"imported {n} files into {os.path.relpath(DATA_RAW, REPO_ROOT)}/<group>/\n")
    report(scan(DATA_RAW))


def do_manifest(note: str | None = None, show: bool = True) -> None:
    groups = scan(DATA_RAW)
    summary = report(groups) if show else {
        g: {"files": len(p), "words": sum(len(open(f, encoding='utf-8', errors='replace').read().split()) for f in p)}
        for g, p in groups.items()}
    man = {
        "source": CORPUS_URL,
        "licence": "CC BY-NC-ND 3.0 ES",
        "download": note or "(version/date not recorded — pass --note)",
        "note": "file lists and checksums of the local COREFL copy; the corpus "
                "itself is not redistributed here",
        "groups": {g: {"files": summary[g]["files"], "words": summary[g]["words"],
                       "sha256": {os.path.basename(p): sha256(p) for p in paths}}
                   for g, paths in groups.items()},
    }
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    with open(MANIFEST, "w") as fh:
        json.dump(man, fh, indent=1, sort_keys=True)
    print(f"\nwrote {os.path.relpath(MANIFEST, REPO_ROOT)}")


def do_check() -> int:
    if not os.path.exists(MANIFEST):
        sys.exit(f"error: {os.path.relpath(MANIFEST, REPO_ROOT)} not found "
                 f"(run --write-manifest first)")
    with open(MANIFEST) as fh:
        man = json.load(fh)
    groups = scan(DATA_RAW)
    problems = Counter()
    for g, info in man["groups"].items():
        have = {os.path.basename(p): p for p in groups.get(g, [])}
        for name, digest in info["sha256"].items():
            if name not in have:
                problems["missing"] += 1
            elif sha256(have[name]) != digest:
                problems["changed"] += 1
        problems["extra"] += len(set(have) - set(info["sha256"]))
    report(groups)
    problems = {k: v for k, v in problems.items() if v}   # Counter keys with 0 don't count
    if problems:
        print("\nMISMATCH vs manifest:", problems)
        return 1
    print("\nOK - local copy matches the manifest exactly")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-dir", help="import an extracted COREFL download")
    ap.add_argument("--write-manifest", action="store_true",
                    help="record file lists + checksums of the local copy")
    ap.add_argument("--note", default=None,
                    help="what exactly was downloaded — corpus version, date, "
                         "subcorpora and format; stored in the manifest, e.g. "
                         "--note 'v2.0 Oct 2025, Texts only, 2026-07-26'")
    ap.add_argument("--check", action="store_true",
                    help="verify the local copy against the manifest")
    args = ap.parse_args()

    if args.from_dir:
        do_import(args.from_dir)
    if args.write_manifest:
        do_manifest(args.note, show=not args.from_dir)
    if args.check:
        return do_check()
    if not (args.from_dir or args.write_manifest):
        print(__doc__)
        if os.path.isdir(DATA_RAW):
            print("Local copy:\n")
            report(scan(DATA_RAW))
        else:
            print(f"(nothing in {os.path.relpath(DATA_RAW, REPO_ROOT)} yet)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
