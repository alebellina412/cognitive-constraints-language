#!/usr/bin/env bash
set -euo pipefail

# Deterministic bounded regression: with p=0 the stream is never selected, so
# the Fenwick implementation must reproduce every legacy urn event exactly.
ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
WORK_DIR=$(mktemp -d /tmp/umt_fast_equivalence_XXXXXX)
trap 'rm -rf "$WORK_DIR"' EXIT
NAME="UMT_dynamic_rho=1.0_nu=1.0_p=0.00_N0=1_D0=8000"

make -C "$ROOT_DIR" tests/umt_legacy tests/umt_fast_zstd
mkdir -p "$WORK_DIR/legacy/data" "$WORK_DIR/fast/data"
(cd "$WORK_DIR/legacy" && "$ROOT_DIR/tests/umt_legacy" 1 10000 1.0 1.0 0.0 8000 >/dev/null)
(cd "$WORK_DIR/fast" && "$ROOT_DIR/tests/umt_fast_zstd" 1 10000 1.0 1.0 0.0 8000 >/dev/null)
python3 "$ROOT_DIR/compressed/read_cltraj.py" \
    "$WORK_DIR/fast/data/UMT_dynamic_fast_rho=1.0_nu=1.0_p=0.00_N0=1_D0=8000.cltraj.zst" \
    --tsv "$WORK_DIR/fast/decoded.tsv"
cmp "$WORK_DIR/legacy/data/$NAME.dat" "$WORK_DIR/fast/decoded.tsv"
echo "PASS: fast Fenwick urn reproduces the 10,000-step legacy trajectory when p=0."
