#!/usr/bin/env bash
set -euo pipefail

# Bounded regression test: two sequential 10,000-step runs only.
ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
LEGACY_RUN="$ROOT_DIR/tests/legacy_run"
COMPRESSED_RUN="$ROOT_DIR/tests/compressed_run"
NAME="UMT_dynamic_rho=1.0_nu=1.0_p=0.50_N0=1_D0=8000"

make -C "$ROOT_DIR" all
rm -rf "$LEGACY_RUN" "$COMPRESSED_RUN"
mkdir -p "$LEGACY_RUN/data" "$COMPRESSED_RUN/data"
(cd "$LEGACY_RUN" && ../umt_legacy 1 10000 1.0 1.0 0.50 8000 > run.log)
(cd "$COMPRESSED_RUN" && ../umt_zstd 1 10000 1.0 1.0 0.50 8000 > run.log)
python3 "$ROOT_DIR/compressed/read_cltraj.py" "$COMPRESSED_RUN/data/$NAME.cltraj.zst" --tsv "$COMPRESSED_RUN/decoded.tsv"
cmp "$LEGACY_RUN/data/$NAME.dat" "$COMPRESSED_RUN/decoded.tsv"
cmp "$LEGACY_RUN/data/n_$NAME.dat" "$COMPRESSED_RUN/data/n_$NAME.dat"
echo "PASS: lossless compressed writer reproduces the archived C output exactly."
