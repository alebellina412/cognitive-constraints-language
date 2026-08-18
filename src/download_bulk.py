#!/usr/bin/env python3
"""Step 1 of the SPGC pipeline: download the raw bulk from Zenodo.

Corpus: Standardized Project Gutenberg Corpus (SPGC), version SPGC-2018-07-18.
Data record: Zenodo 2422561 (doi:10.5281/zenodo.2422561).
    Gerlach & Font-Clos, Entropy 22(1):126 (2020), arXiv:1812.08092.

Downloads three files into ``data_raw/`` (gitignored — the raw bulk must never
reach GitHub):

    SPGC-metadata-2018-07-18.csv    ~10 MB   per-book id/language/author/year
    SPGC-counts-2018-07-18.zip      ~1.5 GB  word\tcount per book (1-grams)
    SPGC-tokens-2018-07-18.zip      ~6.4 GB  ordered token stream (n-grams + WCN)

Properties:
  * Resumable   — uses ``curl -L -C -`` (HTTP Range). A dropped connection or a
                  network switch loses nothing; re-run and it continues.
  * Idempotent  — a file already present and valid (size matches Zenodo + zips
                  pass ``zipfile.testzip``) is skipped, not re-downloaded.
  * Verified    — the size is checked against Zenodo's Content-Length, zips are
                  integrity-tested, and the MD5 of each file is compared with
                  ``manifests/spgc_checksums.json``. The MD5 is what pins the
                  *release*: size and testzip only say the file is a complete
                  zip, not that it is the one this paper was computed on.
  * Selective   — ``--only`` fetches a subset, e.g. one file at a time.

Usage (run from the repo root):
    python src/download_bulk.py                      # all three
    python src/download_bulk.py --only metadata counts
    python src/download_bulk.py --only tokens
    python src/download_bulk.py --check              # verify only, download nothing

Disk: metadata + counts + tokens together are ~8 GB. Make sure there is room
(the reduced-data step downstream lets you delete data_raw/ afterwards).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile

# --- Zenodo source -----------------------------------------------------------
ZENODO_RECORD = "2422561"
SPGC_VERSION = "SPGC-2018-07-18"
BASE_URL = f"https://zenodo.org/records/{ZENODO_RECORD}/files"

# key -> (filename, is_zip, human-readable approx size)
FILES: dict[str, tuple[str, bool, str]] = {
    "metadata": (f"SPGC-metadata-{SPGC_VERSION.split('SPGC-')[1]}.csv", False, "~10 MB"),
    "counts": (f"SPGC-counts-{SPGC_VERSION.split('SPGC-')[1]}.zip", True, "~1.5 GB"),
    "tokens": (f"SPGC-tokens-{SPGC_VERSION.split('SPGC-')[1]}.zip", True, "~6.4 GB"),
}

# Resolve paths relative to this file so it works from any CWD.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(REPO_ROOT, "data_raw")
CHECKSUMS = os.path.join(REPO_ROOT, "manifests", "spgc_checksums.json")


def published_checksums() -> dict:
    """`{filename: {"bytes": int, "md5": str}}` from the committed manifest."""
    try:
        with open(CHECKSUMS) as fh:
            return json.load(fh)["files"]
    except (OSError, KeyError, ValueError) as exc:
        print(f"    ! could not read {os.path.relpath(CHECKSUMS, REPO_ROOT)}: "
              f"{exc}", file=sys.stderr)
        return {}


def file_md5(path: str, chunk: int = 1 << 22) -> str:
    """MD5 of a file, read in chunks so the 6.4 GB zip never enters memory."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def file_url(filename: str) -> str:
    return f"{BASE_URL}/{filename}?download=1"


def human(n: int) -> str:
    x = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if x < 1024 or unit == "TB":
            return f"{x:.1f} {unit}"
        x /= 1024
    return f"{n} B"  # unreachable


def remote_size(url: str) -> int | None:
    """Content-Length from Zenodo, following redirects. None if unavailable."""
    try:
        out = subprocess.run(
            ["curl", "-sIL", url],
            capture_output=True,
            text=True,
            timeout=60,
        ).stdout
    except (subprocess.SubprocessError, OSError) as exc:
        print(f"    ! could not query remote size: {exc}", file=sys.stderr)
        return None
    size = None  # take the LAST Content-Length seen (after all redirects)
    for line in out.splitlines():
        if line.lower().startswith("content-length:"):
            try:
                size = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
    return size


def zip_ok(path: str) -> bool:
    """True if `path` is a complete, uncorrupted zip (zipfile.testzip)."""
    try:
        with zipfile.ZipFile(path) as zf:
            bad = zf.testzip()  # returns name of first bad member, else None
        if bad is not None:
            print(f"    ! corrupt member in zip: {bad}")
            return False
        return True
    except zipfile.BadZipFile:
        print("    ! not a valid/complete zip yet")
        return False


def is_complete(path: str, is_zip: bool, rsize: int | None) -> bool:
    """Is the local file already a valid, complete copy? (idempotency check)"""
    if not os.path.exists(path):
        return False
    lsize = os.path.getsize(path)
    if rsize is not None and lsize != rsize:
        print(f"    local {human(lsize)} != remote {human(rsize)} (incomplete)")
        return False
    if is_zip:
        return zip_ok(path)
    # non-zip (metadata csv): with matching size we accept it; without a known
    # remote size, a non-empty file is treated as present.
    return lsize > 0


def curl_download(url: str, dest: str, rsize: int | None) -> int:
    """Resumable curl download. Returns the curl exit code.

    curl's own progress meter (percentage, current speed, ETA) is shown live
    because stdout/stderr are inherited and we do NOT pass ``-s``. Before it
    starts we print where the resume picks up, so a re-run makes clear how much
    is already on disk vs how much is left.
    """
    have = os.path.getsize(dest) if os.path.exists(dest) else 0
    if have and rsize:
        pct = 100.0 * have / rsize
        print(f"    resuming from {human(have)} / {human(rsize)} "
              f"({pct:.1f}%) — {human(rsize - have)} left")
    elif have:
        print(f"    resuming from {human(have)} already on disk")
    else:
        print(f"    starting fresh (0 / {human(rsize) if rsize else '?'})")
    print("    live meter below: %  |  Dload = speed  |  Time Left = ETA")

    cmd = [
        "curl",
        "-L",              # follow Zenodo's redirect to the storage backend
        "-C", "-",         # resume from wherever the local file left off
        "--fail",          # non-zero exit on HTTP >= 400
        "--retry", "5",    # retry transient failures
        "--retry-delay", "5",
        "-o", dest,
        url,
    ]
    # Inherit stdout/stderr so curl's own progress meter + speed are shown live.
    return subprocess.run(cmd).returncode


def verify_md5(dest: str, filename: str, published: dict) -> bool:
    """Compare the local MD5 with the committed one. True if it matches or is
    not pinned; False on a mismatch, which is a different corpus release."""
    want = published.get(filename, {}).get("md5")
    if not want:
        return True
    print("    computing MD5...", flush=True)
    got = file_md5(dest)
    if got == want:
        print(f"    [ok] MD5 {got} matches manifests/spgc_checksums.json")
        return True
    print(f"    [FAIL] MD5 {got}\n"
          f"           expected {want} (manifests/spgc_checksums.json)\n"
          f"           This is not the SPGC release the paper used. Numbers "
          f"computed from it will not reproduce.", file=sys.stderr)
    return False


def process(key: str, check_only: bool) -> bool:
    filename, is_zip, approx = FILES[key]
    dest = os.path.join(DATA_RAW, filename)
    url = file_url(filename)
    published = published_checksums()
    print(f"\n=== {key}: {filename} ({approx}) ===")

    rsize = remote_size(url)
    if rsize is not None:
        print(f"    remote size: {human(rsize)}")

    if is_complete(dest, is_zip, rsize):
        print("    [skip] already present and valid.")
        return verify_md5(dest, filename, published)

    if check_only:
        print("    [check] missing or incomplete — would download.")
        return False

    code = curl_download(url, dest, rsize)
    if code == 33:  # curl: server does not support resume but file may be done
        print("    (server reported no-resume; re-checking local file)")
    elif code != 0:
        print(f"    ! curl exited with code {code}", file=sys.stderr)

    # Verify after download.
    if is_zip:
        print("    verifying zip integrity (testzip) — may take a few minutes "
              "for the big files, no progress bar here...")
    rsize = rsize if rsize is not None else remote_size(url)
    if is_complete(dest, is_zip, rsize):
        print(f"    [ok] verified {human(os.path.getsize(dest))}.")
        return verify_md5(dest, filename, published)
    print("    [FAIL] file is missing/incomplete/corrupt after download.")
    print("           Re-run the command — curl -C - resumes where it stopped.")
    return False


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Resumable download of the SPGC raw bulk from Zenodo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--only",
        nargs="+",
        choices=list(FILES),
        metavar="FILE",
        help="download only these: %(choices)s (default: all)",
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="verify what is present; download nothing",
    )
    args = ap.parse_args()

    if shutil.which("curl") is None:
        print("error: 'curl' not found on PATH.", file=sys.stderr)
        return 2

    os.makedirs(DATA_RAW, exist_ok=True)
    keys = args.only if args.only else list(FILES)

    print(f"SPGC bulk downloader — {SPGC_VERSION} (Zenodo {ZENODO_RECORD})")
    print(f"Target: {DATA_RAW}")
    print(f"Files : {', '.join(keys)}")

    results = {k: process(k, args.check) for k in keys}

    print("\n=== summary ===")
    for k in keys:
        print(f"    {k:9s} {'OK' if results[k] else 'MISSING/INCOMPLETE'}")
    ok = all(results.values())
    if not ok and not args.check:
        print("\nSome files are not complete. Re-run to resume.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
