# Lossless UMT trajectory exporter

## What is archival and what is new

`legacy/UMT_dynamic.c` and its header `legacy/polya_adj.h` are the **archived
simulator, reproduced verbatim** — the program the earlier work was run with.
They are kept unmodified on purpose, so that the fast implementation can be
checked against them rather than against a tidied-up rewrite:

    sha256  da6437fc4ab11ee62435dfc0c3c8a50989c0fcfae7ed84a27abc74eb9ea5c82c  legacy/UMT_dynamic.c
    sha256  ed59651ec1624fa12895bacc5f2636bb7086f2de95f3e6228e2f601e0a22a251  legacy/polya_adj.h

Being archival, they are also **unaudited**, and they compile with warnings that
new code here would not be allowed to have: unused variables, two functions in
`polya_adj.h` that fall off the end without returning, and most of that header
unreachable from `UMT_dynamic.c` at all. `sample()` has no bounds check. None of
this is on the path the paper uses — the production runs go through
`optimized/`, and `legacy/` exists to be the reference the two test scripts
compare against. Read the warnings as the age of the file, not as the state of
the pipeline. `legacy/UMT_dynamic.c` also does not create its own `data/`
directory: create it before running that binary by hand, as the test scripts do.

`compressed/UMT_dynamic_zstd.c` includes that exact source and intercepts only
trajectory file I/O. The simulator dynamics, arguments, fixed seed,
random-number calls, parameter updates, and final frequency output are
unchanged.

## Requirements

`gcc`, `make`, and the **`zstd` command-line tool**, which every program here
pipes its trajectory through (`popen("zstd …")`) and which
`compressed/read_cltraj.py` shells out to when decoding. Without it the writers
fail on a broken pipe rather than with a clear message.

## Output format

The new trajectory has extension `.cltraj.zst` and is compressed directly by
one Zstandard process at level 3 and one thread. It retains every step.

- `t` is implicit from record position;
- each selected item `estr` is stored as `uint32`;
- the increment of `D` after each step is stored as one bit;
- the header stores format version, parameters, seed, and intended `T`.

`compressed/read_cltraj.py` reconstructs the exact historical TSV rows
`t, D, estr` as a stream, without loading a whole trajectory into memory.

Set `CLTRAJ_OUTPUT=/path/to/trajectory.cltraj.zst` when running the executable
to select a stable canonical output path. Without it, the historical
parameter-based filename is used.

## Regression test

Run `tests/verify_roundtrip.sh`. It compiles both programs and runs two
sequential 10,000-step simulations, then performs byte-for-byte comparisons of
the decoded trajectory and final frequency output. This is intentionally small
and bounded.

## Long production runs

The original model performs substantial work per step and allocates its fixed
working arrays even for a short run. A `T=10^8` production run must be launched
only after an explicit resource estimate and user approval. The compressed
writer reduces output storage, but does not change simulation compute time.

## Fast time-dependent implementation

`optimized/UMT_dynamic_fast_zstd.c` has the same arguments as the archived
time-dependent simulator:

```text
<N0> <T> <rho> <nu> <p> <D0>
```

It preserves its stochastic rules, fixed seed, lazy probability, and initial
urn (`N0` units on the initial frontier). It changes only the representation:

- a Fenwick tree stores unnormalised urn weights, so urn draws and updates are
  `O(log D)` rather than scans and global renormalisations;
- the realised trajectory is a `uint32_t` history, so stream draws are `O(1)`;
- trajectory output is direct `CLTRJ1` Zstandard, compatible with
  `compressed/read_cltraj.py` and with `CLTRAJ_OUTPUT`.

The fixed seed does not imply an identical trajectory once the stream branch
is selected: uniform sampling from the history and cumulative-frequency
sampling have the same distribution but map the same pseudorandom number to
different individual events. With `p=0`, where only the urn branch is used,
`tests/verify_fast_urn_equivalence.sh` verifies a byte-identical 10,000-step
trajectory against the legacy program.

### What "not identical" means in practice

At `p > 0` the two programs are therefore **different realisations of the same
process**, not two computations of the same realisation. That is expected, and
the thing to check is that they agree on the statistics the paper reports, not
on individual draws. On the canonical run they do:

| quantity | legacy | fast |
| --- | ---: | ---: |
| rank-frequency $\alpha_1$ | 1.074 | 1.076 |
| rank-frequency $\alpha_2$ | 1.514 | 1.511 |
| mean degree $\langle k\rangle$ | 24.34 | 24.39 |
| vocabulary $D(T)$ | identical | identical |

Two further checks tied the fast pipeline to the archived one when it was
written. Both were run against files of the pre-publication working repository
which are **not redistributed here**, so they are recorded rather than
reproducible from this repository alone: `src/build_reduced_sim.py` reproduced
that repository's `node_statistics.npz` for the canonical trajectory
bit-for-bit (30,065 types, identical frequency vector), and a run at
`D0=8000, T=10^8` yielded `D(T) = 307,128` types — exactly the node count of its
`degree_strength_UMT_p=0.5.csv`, the file the network figure of the first
submission was built from. The new pipeline is continuous with the old one, not
an approximate rebuild of it.

What *is* reproducible here is everything above: `make`, then the two scripts in
`tests/`.
