#!/usr/bin/env python3
"""Stream a lossless CLTRJ1 trajectory as (t, D, estr) rows."""

from __future__ import annotations

import argparse
import struct
import subprocess
from pathlib import Path


HEADER = struct.Struct("<8sIQIIII3d")


def read_exact(stream, count: int) -> bytes:
    data = stream.read(count)
    if len(data) != count:
        raise ValueError("Unexpected end of compressed trajectory")
    return data


def iter_rows(path: Path):
    process = subprocess.Popen(["zstd", "-q", "-dc", str(path)], stdout=subprocess.PIPE)
    assert process.stdout is not None
    stream = process.stdout
    magic, version, total_steps, n0, d0, seed, block_capacity, rho, nu, p = HEADER.unpack(read_exact(stream, HEADER.size))
    if magic != b"CLTRJ1\0\0" or version != 1:
        raise ValueError("Unsupported trajectory format")
    current_d = 0
    t = 0
    while t < total_steps:
        count = struct.unpack("<I", read_exact(stream, 4))[0]
        estr = struct.unpack(f"<{count}I", read_exact(stream, 4 * count))
        bits = read_exact(stream, (count + 7) // 8)
        for index, selected in enumerate(estr):
            yield t, current_d, selected
            current_d += (bits[index // 8] >> (index % 8)) & 1
            t += 1
    if process.wait() != 0:
        raise RuntimeError("zstd decompression failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--tsv", type=Path)
    args = parser.parse_args()
    destination = args.tsv.open("w") if args.tsv else None
    try:
        for t, d, estr in iter_rows(args.trajectory):
            line = f"{t}\t{d}\t{estr}\n"
            if destination: destination.write(line)
            else: print(line, end="")
    finally:
        if destination: destination.close()


if __name__ == "__main__":
    main()
