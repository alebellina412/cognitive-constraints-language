#!/usr/bin/env python3
"""Step 2 for the model: reduce a simulated trajectory to the same schema as the corpus.

The C simulator writes a full `.cltraj.zst` trajectory (166 MB at T = 10^8).
That file is an intermediate, never committed. This script turns it into the
*same* compact artifacts that `build_reduced.py` produces from the raw corpus:

    data_reduced/<name>_wcn.npz     strength, degree, n_nodes, n_edges
    data_reduced/<name>_ngram.npz   1-gram and 2-gram frequency-of-frequency
    data_reduced/<name>_heaps.npz   vocabulary growth D(t)
    data_reduced/<name>_meta.json   T, D0, n0, rho, nu, p, seed, timings

Because the schema is identical, `notebooks/09_model_and_d0.ipynb` can push the
model through exactly the same `io_reduced` readers and `fits` functions that
Figures 2 and S1 use on the data: the comparison is then guaranteed to be
like-for-like, not two similar-looking pipelines.

Conventions copied from `build_reduced.py` so the numbers are comparable:
  * the co-occurrence network is **undirected** and **excludes self-loops**
    (consecutive identical items), exactly as the corpus WCN does;
  * D(t) is computed with the same `first_occurrence` routine as the corpus
    Heaps curve — one shared implementation, so no convention can drift;
  * 2-grams are **ordered** pairs (as in the corpus n-gram counts), while the
    network edges are unordered — the two differ and both are needed.

    # reduce an existing trajectory
    python src/build_reduced_sim.py --trajectory work/run.cltraj.zst --name sim_d0_8000

    # run the simulator and reduce in one go (trajectory deleted afterwards)
    python src/build_reduced_sim.py --run --d0 8000 --T 100000000 --seed 1
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np

from build_reduced import DATA_REDUCED, REPO_ROOT, first_occurrence

SIM_DIR = os.path.join(REPO_ROOT, "sim")
SIM_BINARY = os.path.join(SIM_DIR, "tests", "umt_fast_zstd")
WORK = os.path.join(REPO_ROOT, "work")

HEADER = struct.Struct("<8sIQIIII3d")
CHUNK = 1 << 23          # elements per vectorised chunk when packing pair keys


# --------------------------------------------------------------------------- #
# reading
# --------------------------------------------------------------------------- #
def read_trajectory(path: str) -> tuple[dict, np.ndarray]:
    """Decode a CLTRJ1 trajectory into (header, events).

    `events[t]` is the item drawn at step t, as a uint32 id. The per-step
    "vocabulary grew" bit is *not* returned: D(t) is recomputed from the events
    themselves with the same routine used for the corpus, which is a stronger
    guarantee of comparability (and is cross-checked against the bits here).
    """
    t0 = time.time()
    proc = subprocess.Popen(["zstd", "-q", "-dc", path], stdout=subprocess.PIPE)
    assert proc.stdout is not None
    stream = proc.stdout
    magic, version, total, n0, d0, seed, block_capacity, rho, nu, p = \
        HEADER.unpack(stream.read(HEADER.size))
    if magic != b"CLTRJ1\0\0" or version != 1:
        raise ValueError(f"{path}: not a CLTRJ1 v1 trajectory")

    events = np.empty(total, dtype=np.uint32)
    grew = 0
    t = 0
    while t < total:
        count = struct.unpack("<I", stream.read(4))[0]
        events[t:t + count] = np.frombuffer(stream.read(4 * count), dtype="<u4")
        bits = stream.read((count + 7) // 8)
        grew += int(np.unpackbits(np.frombuffer(bits, dtype=np.uint8),
                                  bitorder="little", count=count).sum())
        t += count
    if proc.wait() != 0:
        raise RuntimeError("zstd decoding failed")

    header = {"T": int(total), "N0": int(n0), "D0": int(d0), "seed": int(seed),
              "rho": float(rho), "nu": float(nu), "p": float(p),
              "block_capacity": int(block_capacity), "D_final_from_bits": grew}
    print(f"  read {total:,} steps [{time.time() - t0:.0f}s]", flush=True)
    return header, events


# --------------------------------------------------------------------------- #
# reductions
# --------------------------------------------------------------------------- #
def _pair_keys(events: np.ndarray, undirected: bool) -> np.ndarray:
    """Pack consecutive pairs into uint64 keys, dropping self-loops.

    Chunked so the intermediate uint64 arrays stay small: at T = 10^8 a
    non-chunked version would allocate several GB of temporaries.
    """
    n = events.size - 1
    keys = np.empty(n, dtype=np.uint64)
    shift = np.uint64(32)
    for i in range(0, n, CHUNK):
        j = min(i + CHUNK, n)
        a = events[i:j].astype(np.uint64)
        b = events[i + 1:j + 1].astype(np.uint64)
        if undirected:
            lo = np.minimum(a, b)
            hi = np.maximum(a, b)
        else:
            lo, hi = a, b
        keys[i:j] = (lo << shift) | hi
    self_loop = events[:-1] == events[1:]
    return keys[~self_loop]


def build_wcn(events: np.ndarray, vocab_size: int) -> dict:
    """Undirected co-occurrence network summary, self-loops excluded."""
    t0 = time.time()
    keys = _pair_keys(events, undirected=True)
    uniq, weight = np.unique(keys, return_counts=True)
    del keys
    a = (uniq >> np.uint64(32)).astype(np.int64)
    b = (uniq & np.uint64(0xFFFFFFFF)).astype(np.int64)
    w = weight.astype(np.int64)
    strength = (np.bincount(a, weights=w, minlength=vocab_size)
                + np.bincount(b, weights=w, minlength=vocab_size))
    degree = (np.bincount(a, minlength=vocab_size)
              + np.bincount(b, minlength=vocab_size))
    strength = strength[strength > 0].astype(np.int64)
    degree = degree[degree > 0].astype(np.int64)
    strength.sort()
    degree.sort()
    print(f"  WCN: nodes={degree.size:,} edges={uniq.size:,} "
          f"[{time.time() - t0:.0f}s]", flush=True)
    return {"strength": strength[::-1].copy(), "degree": degree[::-1].copy(),
            "n_nodes": np.int64(degree.size), "n_edges": np.int64(uniq.size)}


def _freq_of_freq(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(distinct count value, multiplicity) of a positive count vector."""
    vals, mult = np.unique(counts[counts > 0], return_counts=True)
    return vals.astype(np.int64), mult.astype(np.int64)


def build_ngrams(events: np.ndarray, vocab_size: int) -> dict:
    """1-gram and 2-gram frequency-of-frequency, same keys as the corpus file.

    Only orders 1 and 2 exist for the model: the simulator generates a sequence
    of items with pairwise reinforcement, so higher orders carry no mechanism of
    their own and enter no model/data comparison.
    """
    t0 = time.time()
    out: dict[str, np.ndarray] = {}

    counts1 = np.bincount(events, minlength=vocab_size)
    vals, mult = _freq_of_freq(counts1)
    out["ff_vals_1"], out["ff_mult_1"] = vals, mult
    out["n_distinct_1"] = np.int64(mult.sum())
    out["n_positions_1"] = np.int64(events.size)

    keys = _pair_keys(events, undirected=False)
    _, counts2 = np.unique(keys, return_counts=True)
    n_positions_2 = keys.size
    del keys
    vals, mult = _freq_of_freq(counts2)
    out["ff_vals_2"], out["ff_mult_2"] = vals, mult
    out["n_distinct_2"] = np.int64(mult.sum())
    out["n_positions_2"] = np.int64(n_positions_2)

    print(f"  n-grams: 1-gram types={int(out['n_distinct_1']):,} "
          f"2-gram types={int(out['n_distinct_2']):,} "
          f"[{time.time() - t0:.0f}s]", flush=True)
    return out


def build_heaps(events: np.ndarray, vocab_size: int, n_points: int = 120) -> dict:
    """D(t) on a log grid, via the same first-occurrence routine as the corpus.

    A simulation is one realisation, so there is no ordering to average over and
    no band: the seed-to-seed spread is reported by the calibration instead.
    """
    t0 = time.time()
    total = events.size
    grid = np.unique(np.round(np.logspace(1, np.log10(total), n_points)).astype(np.int64))
    first = first_occurrence(events, vocab_size)
    first.sort()
    D = np.searchsorted(first, grid, side="left").astype(float)
    print(f"  Heaps: D({total:,}) = {D[-1]:,.0f} [{time.time() - t0:.0f}s]", flush=True)
    return {"heaps_t": grid, "heaps_mean": D, "heaps_lo": D, "heaps_hi": D,
            "n_shuffles": np.int64(1), "vocab_pool": np.int64(vocab_size)}


# --------------------------------------------------------------------------- #
def run_simulator(d0: int, T: int, seed: int, model: dict, out: str,
                  cwd: str | None = None) -> None:
    """Run the fast C simulator into `out` (built via sim/Makefile if needed).

    `CLTRAJ_OUTPUT` redirects the trajectory but not the final frequency file,
    which the simulator always writes to `data/` relative to its working
    directory — 12 MB per run at T = 10^8. Running it inside the scratch
    directory therefore keeps that file with the trajectory, where the caller
    deletes both, instead of dropping a few hundred MB into whichever directory
    the notebook happened to start from. `calibrate_d0.run_pair` already does
    the same thing.
    """
    if not os.path.exists(SIM_BINARY):
        subprocess.run(["make", "-C", SIM_DIR, "tests/umt_fast_zstd"], check=True)
    if seed != 1:
        raise SystemExit(
            "the committed fast simulator hard-codes seed 1; use "
            "src/calibrate_d0.py, which compiles an auditable seed-aware copy")
    env = {**os.environ, "CLTRAJ_OUTPUT": out}
    t0 = time.time()
    subprocess.run([SIM_BINARY, str(model["N0"]), str(T), str(model["rho"]),
                    str(model["nu"]), str(model["p"]), str(d0)],
                   check=True, env=env, cwd=cwd, stdout=subprocess.DEVNULL)
    print(f"  simulated T={T:,} D0={d0:,} [{time.time() - t0:.0f}s]", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trajectory", help="existing .cltraj.zst to reduce")
    ap.add_argument("--run", action="store_true", help="run the simulator first")
    ap.add_argument("--d0", type=int, default=8000)
    ap.add_argument("--T", type=int, default=100_000_000)
    ap.add_argument("--seed", type=int, default=1,
                    help="must be 1: the committed simulator hard-codes it. "
                         "src/calibrate_d0.py compiles a seed-aware copy")
    ap.add_argument("--N0", type=int, default=1)
    ap.add_argument("--rho", type=float, default=1.0)
    ap.add_argument("--nu", type=float, default=1.0)
    ap.add_argument("--p", type=float, default=0.5)
    ap.add_argument("--name", help="output prefix (default sim_D0=<d0>_T=<T>)")
    ap.add_argument("--keep-trajectory", action="store_true",
                    help="do not delete a trajectory produced by --run")
    args = ap.parse_args()

    if not (args.trajectory or args.run):
        ap.error("give --trajectory PATH or --run")
    model = {"N0": args.N0, "rho": args.rho, "nu": args.nu, "p": args.p}
    os.makedirs(WORK, exist_ok=True)
    os.makedirs(DATA_REDUCED, exist_ok=True)

    tmpdir = None
    path = args.trajectory
    if args.run:
        tmpdir = tempfile.mkdtemp(prefix="sim_", dir=WORK)
        path = os.path.join(tmpdir, "trajectory.cltraj.zst")
        run_simulator(args.d0, args.T, args.seed, model, path, cwd=tmpdir)

    try:
        header, events = read_trajectory(path)
        vocab_size = int(events.max()) + 1
        name = args.name or f"sim_D0={header['D0']}_T={header['T']}"

        heaps = build_heaps(events, vocab_size)
        if int(heaps["heaps_mean"][-1]) != header["D_final_from_bits"]:
            raise RuntimeError(
                f"D(T) mismatch: first-occurrence gives "
                f"{int(heaps['heaps_mean'][-1]):,}, the trajectory's novelty bits "
                f"give {header['D_final_from_bits']:,}")
        ngram = build_ngrams(events, vocab_size)
        wcn = build_wcn(events, vocab_size)
        del events

        for kind, payload in (("heaps", heaps), ("ngram", ngram), ("wcn", wcn)):
            out = os.path.join(DATA_REDUCED, f"{name}_{kind}.npz")
            np.savez_compressed(out, max_tokens=np.int64(header["T"]), **payload)
            print(f"-> {os.path.relpath(out, REPO_ROOT)} "
                  f"({os.path.getsize(out) / 1e6:.1f} MB)")
        meta = {**header, "name": name, "vocab_size": vocab_size,
                "source_trajectory": os.path.basename(path)}
        with open(os.path.join(DATA_REDUCED, f"{name}_meta.json"), "w") as fh:
            json.dump(meta, fh, indent=1)
        print(f"-> data_reduced/{name}_meta.json")
    finally:
        # rmtree, not unlink-then-rmdir: the simulator creates a `data/`
        # subdirectory here for its frequency file (see run_simulator)
        if tmpdir and not args.keep_trajectory:
            shutil.rmtree(tmpdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
