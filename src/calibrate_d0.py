#!/usr/bin/env python3
"""Calibrate the model scale D0 against the reduced corpora.

For every (D0, seed) the fast C simulator is run once and its trajectory is
streamed — never held in memory — into two summaries: the vocabulary growth
D(t) at the empirical sampling times, and the rank-frequency vector of the
prefix whose token count matches the empirical one. Each summary is compared
with the corresponding empirical curve; the two component losses and their
weighted sum are recorded per run.

Empirical inputs come only from ``data_reduced/`` (no raw corpus):

  native   spgc_en_heaps.npz     D(t) up to 1e8 tokens (mean over book orders)
                                 + freq_matched, the 1e8-token rank-frequency
  learner  corefl_learner_all.npz  D(t) over the pooled COREFL es+de learners
                                 + freq, the whole 586,145-token corpus

The empirical curve is never extrapolated: a population is compared only up to
the token count it actually has, whatever T the simulation ran for.

Uncertainty is reported as (i) the spread across seeds at fixed D0 and (ii) the
interval of D0 whose mean joint loss stays within one seed-SD of the minimum —
the flatness of the loss profile. Neither is a confidence interval, and the
script does not claim one: the reduced data provide no resampling unit here.

**``--config`` and ``--tag`` are both required, and neither is cosmetic.** The
grid lives in a manifest, not in this file: ``DEFAULTS`` below exists only so
that the loss-comparison settings can be imported as a library (by
``model_adequacy.py`` and ``umt_transient_n0.py``), and its ``d0_grid`` is *not*
the paper's — running without ``--config`` selects D0 = 9,902 instead of 9,366.
``--tag`` names the output directory, and the two sweeps the paper reports are
read from two different ones (see below), so an untagged run writes where
nothing looks.

The paper's three invocations, in the order notebook 08 runs them:

    python src/calibrate_d0.py --config manifests/d0_calibration.json \
        --tag dryrun --dry-run                       # plumbing check, one D0
    python src/calibrate_d0.py --config manifests/d0_calibration.json \
        --tag crossover --T 100000000 --allow-long-run
    python src/calibrate_d0.py --config manifests/d0_calibration_refined.json \
        --tag refined --T 100000000 --allow-long-run

The two production sweeps answer different questions and **neither replaces the
other**:

  crossover  14 values of D0 from 200 to 26,000, 2 seeds. Records ``rstar_sim``,
             the crossover of each simulated curve, which is what
             ``figure_SI5.py`` builds its second criterion from; the low end is
             load-bearing, since for the learners that criterion selects D0=275.
             Read by ``figure_SI5.py`` and ``model_vs_data.py``.
  refined    24 values narrowed onto each minimum, 6 seeds. Has **no**
             ``rstar_sim`` column. Six seeds are what make the competitive
             interval honest; with two, the learners' range collapses onto one
             grid point. Read by ``estimate_d0.py``.

Refining does not sharpen D0 and is not a "fine" version of the other: the
natives come out at 9,366 with the same [6,664-13,163] range on both grids.

Outputs go to ``outputs/d0_calibration_<tag>/``: runs.csv (one row per run),
summary_by_d0.csv (mean/SD over seeds), calibration_report.json and the
diagnostic figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SIMULATOR = REPO_ROOT / "sim" / "optimized" / "UMT_dynamic_fast_zstd.c"
DATA_REDUCED = REPO_ROOT / "data_reduced"
OUTPUTS = REPO_ROOT / "outputs" / "d0_calibration"
WORK = REPO_ROOT / "work"

HEADER = struct.Struct("<8sIQIIII3d")

#: Fixed model settings. n0=1 and rho=nu=1 are not calibrated — they are the
#: standard UMT choices used throughout the paper; only D0 is estimated here.
#:
#: **`d0_grid` here is NOT the paper's grid** and must not be used as one — the
#: paper's grids are `manifests/d0_calibration*.json` and `--config` is required
#: for exactly that reason. What this dict is for is the *loss-comparison*
#: settings (`rank_min`, `rank_max`, `rank_points`, `heaps_points`, `weights`),
#: which `model_adequacy.py` and `umt_transient_n0.py` import so that they score
#: a run the same way the calibration does.
DEFAULTS = {
    "model": {"N0": 1, "rho": 1.0, "nu": 1.0, "p": 0.5},
    "calibration": {
        "T": 100_000_000,
        # union of two log grids, dense where each population's optimum sits
        "d0_grid": {"learner": [50, 5000, 16], "native": [2000, 64000, 14]},
        "d0_values": [],
        "seeds": [101, 202, 303],
        "weights": {"heaps": 0.5, "zipf": 0.5},
        "heaps_points": 80,
        "heaps_spacing": "log",
        "rank_min": 10,
        "rank_max": 5000,
        "rank_points": 80,
        "rank_spacing": "log",
    },
}

#: population -> (reduced file, growth keys, rank-frequency key)
POPULATIONS = {
    "native": ("spgc_en_heaps.npz", "freq_matched"),
    "learner": ("corefl_learner_all.npz", "freq"),
}
POP_COLORS = {"native": "#7A5197", "learner": "#F0A202"}


# --------------------------------------------------------------------------- #
# empirical side
# --------------------------------------------------------------------------- #
def load_empirical(population: str, maximum_t: int) -> dict:
    """Growth curve and rank-frequency vector of one population.

    `maximum_t` truncates the empirical curve to the simulated horizon; it never
    extends it, so a short corpus is simply compared over its own range.
    """
    try:
        filename, freq_key = POPULATIONS[population]
    except KeyError:
        raise ValueError(f"unknown population: {population}") from None
    data = np.load(DATA_REDUCED / filename)
    t = data["heaps_t"].astype(np.int64)
    d = data["heaps_mean"].astype(float)
    keep = t <= maximum_t
    if not keep.any():
        raise ValueError(f"{population}: no empirical point at or below T={maximum_t}")
    return {
        "t": t[keep],
        "D": d[keep],
        "frequency": data[freq_key].astype(float),
        "source": filename,
    }


def sample_grid(low: int, high: int, points: int, spacing: str) -> np.ndarray:
    """Integer comparison positions in [low, high]; points=0 means every one."""
    if high < low:
        raise ValueError("comparison interval is empty")
    if points == 0:
        return np.arange(low, high + 1, dtype=np.int64)
    if points < 2:
        raise ValueError("points must be 0 (all positions) or at least 2")
    points = min(points, high - low + 1)
    if spacing == "log":
        values = np.geomspace(max(1, low), max(low, high), points)
    elif spacing == "linear":
        values = np.linspace(low, high, points)
    else:
        raise ValueError(f"unsupported spacing: {spacing}")
    selected = np.rint(values).astype(np.int64)
    # keep the requested count despite integer rounding at the dense end
    selected[0], selected[-1] = low, high
    for i in range(1, len(selected)):
        selected[i] = max(selected[i], selected[i - 1] + 1)
    selected[-1] = min(selected[-1], high)
    for i in range(len(selected) - 2, -1, -1):
        selected[i] = min(selected[i], selected[i + 1] - 1)
    return selected


def d0_grid(cfg: dict) -> list[int]:
    """Union of the per-population log grids, plus any explicit values."""
    values = set(int(v) for v in cfg.get("d0_values", []))
    for low, high, n in cfg.get("d0_grid", {}).values():
        values.update(int(round(v)) for v in np.geomspace(low, high, n))
    if not values:
        raise ValueError("empty D0 grid")
    return sorted(values)


# --------------------------------------------------------------------------- #
# simulator
# --------------------------------------------------------------------------- #
def patch_seeded_source(destination: Path) -> None:
    """Ephemeral, auditable seed-aware copy of the repository's fast C source.

    Only the two hard-coded seeds become environment-driven; the dynamics, the
    RNG calls and the output format are untouched.
    """
    original = SIMULATOR.read_text()
    #: (what to find, what to put there). Both must match exactly once. Checking
    #: them separately matters: if only the header one matched, the binary would
    #: keep srand48(1) while recording a varying seed, every seed would produce
    #: the same trajectory, and the seed spread the uncertainty rests on would
    #: silently collapse to zero.
    patches = (
        ("const uint32_t version = 1, seed = 1, block_capacity = BLOCK_CAPACITY;",
         'const uint32_t version = 1, seed = (uint32_t)strtoul(getenv("UMT_SEED") '
         '? getenv("UMT_SEED") : "1", NULL, 10), block_capacity = BLOCK_CAPACITY;'),
        ("    srand48(1);",
         '    srand48((long)strtol(getenv("UMT_SEED") ? getenv("UMT_SEED") '
         ': "1", NULL, 10));'),
    )
    text = original
    for find, replace in patches:
        if text.count(find) != 1:
            raise RuntimeError(
                f"fast simulator changed: the seed patch expects exactly one "
                f"occurrence of {find!r}, found {text.count(find)}")
        text = text.replace(find, replace)
    destination.write_text(text)


def compile_simulator(binary: Path, work: Path) -> None:
    source = work / "UMT_dynamic_fast_seeded.c"
    patch_seeded_source(source)
    subprocess.run(["gcc", "-O2", "-std=c11", "-D_DEFAULT_SOURCE", "-Wall",
                    "-Wextra", "-o", str(binary), str(source), "-lm"], check=True)
    source.unlink()


def stream_summary(path: Path, sample_times: np.ndarray,
                   frequency_tokens: int) -> tuple[np.ndarray, np.ndarray]:
    """Read a CLTRJ1 trajectory once, keeping only D(t) samples and counts.

    Blocks are decoded with numpy, so a 10^8-step trajectory costs a few
    seconds and a few MB rather than being materialised.
    """
    process = subprocess.Popen(["zstd", "-q", "-dc", str(path)], stdout=subprocess.PIPE)
    assert process.stdout is not None
    stream = process.stdout
    magic, version, total, *_ = HEADER.unpack(stream.read(HEADER.size))
    if magic != b"CLTRJ1\0\0" or version != 1:
        raise ValueError("unsupported trajectory format")
    targets = np.asarray(sample_times, dtype=np.int64) - 1
    sampled = np.empty(len(targets), dtype=float)
    target_i, D, t, seen = 0, 0, 0, 0
    counts = np.zeros(1024, dtype=np.uint64)
    while t < total:
        count = struct.unpack("<I", stream.read(4))[0]
        events = np.frombuffer(stream.read(4 * count), dtype="<u4")
        bits = stream.read((count + 7) // 8)
        increments = np.unpackbits(np.frombuffer(bits, dtype=np.uint8),
                                   bitorder="little", count=count).astype(np.int64)
        if t < frequency_tokens:
            usable = events[:min(count, frequency_tokens - t)]
            if len(usable):
                needed = int(usable.max()) + 1
                if needed > len(counts):
                    counts = np.pad(counts, (0, max(len(counts), needed - len(counts))))
                counts[:needed] += np.bincount(usable, minlength=needed).astype(np.uint64)
                seen = max(seen, needed)
        block_end = t + count
        target_end = np.searchsorted(targets, block_end, side="left")
        cumulative = D + np.cumsum(increments)
        if target_end > target_i:
            sampled[target_i:target_end] = cumulative[targets[target_i:target_end] - t]
            target_i = target_end
        D = int(cumulative[-1])
        t = block_end
    if process.wait() != 0:
        raise RuntimeError("zstd decoding failed")
    sampled[target_i:] = D
    nonzero = counts[:seen][counts[:seen] > 0]
    return sampled, np.sort(nonzero)[::-1].astype(float)


# --------------------------------------------------------------------------- #
# losses
# --------------------------------------------------------------------------- #
def metrics(empirical: dict, simulated_d: np.ndarray,
            simulated_freq: np.ndarray, cfg: dict) -> tuple[float, float, int]:
    """Heaps and Zipf log-RMSE. Zipf uses relative frequencies, so vectors of
    different total token counts remain comparable."""
    heaps = float(np.sqrt(np.mean(
        (np.log10(simulated_d) - np.log10(empirical["D"])) ** 2)))
    hi = min(cfg["rank_max"], len(empirical["frequency"]), len(simulated_freq))
    ranks = sample_grid(cfg["rank_min"], hi, cfg["rank_points"], cfg["rank_spacing"])
    sim_rel = simulated_freq / simulated_freq.sum()
    emp_rel = empirical["frequency"] / empirical["frequency"].sum()
    zipf = float(np.sqrt(np.mean(
        (np.log10(sim_rel[ranks - 1]) - np.log10(emp_rel[ranks - 1])) ** 2)))
    return heaps, zipf, hi


def simulated_rstar(frequency: np.ndarray) -> float:
    """Crossover the simulation itself produces, by the estimator of Appendix B.

    The second criterion for D0. The loss above asks which D0 brings the whole
    simulated curve closest to the whole empirical one; this asks only where the
    simulated curve bends, which is the quantity actually compared between
    populations. The two need not select the same D0, and on the learners they
    do not.
    """
    from fits import broken_power_law                        # noqa: PLC0415
    rank = np.arange(1, len(frequency) + 1, dtype=float)
    return float(broken_power_law(rank, frequency / frequency.sum())["R_star"])


FIELDS = ["population", "D0", "seed", "T", "N0", "rho", "nu", "p", "source",
          "loss_heaps", "loss_zipf", "loss_joint", "rstar_sim", "rank_min",
          "rank_max_used", "n_heaps_points", "status", "elapsed_s"]


def append_row(path: Path, row: dict) -> None:
    new = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow(row)


def prepare(populations, cfg) -> dict:
    """Empirical curves resampled onto their comparison grids, once."""
    out = {}
    for population in populations:
        emp = load_empirical(population, cfg["T"])
        times = sample_grid(int(emp["t"][0]), int(emp["t"][-1]),
                            cfg["heaps_points"], cfg["heaps_spacing"])
        out[population] = {**emp, "t": times,
                           "D": np.interp(times, emp["t"], emp["D"])}
    return out


def run_pair(binary: Path, populations, d0: int, seed: int, cfg: dict,
             model: dict, results: Path, keep: bool = False) -> list[dict]:
    """One simulation per (D0, seed), scored separately for each population."""
    prepared = prepare(populations, cfg)
    start = time.monotonic()
    rows = []
    with tempfile.TemporaryDirectory(prefix="d0cal_", dir=WORK) as tmp:
        trajectory = Path(tmp) / "trajectory.cltraj.zst"
        env = {**os.environ, "UMT_SEED": str(seed), "CLTRAJ_OUTPUT": str(trajectory)}
        try:
            subprocess.run([str(binary), str(model["N0"]), str(cfg["T"]),
                            str(model["rho"]), str(model["nu"]), str(model["p"]),
                            str(d0)],
                           cwd=tmp, env=env, check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            for population, emp in prepared.items():
                tokens = min(cfg["T"], int(emp["frequency"].sum()))
                ds, freq = stream_summary(trajectory, emp["t"], tokens)
                lh, lz, hi = metrics(emp, ds, freq, cfg)
                rows.append({
                    "population": population, "D0": d0, "seed": seed,
                    "T": cfg["T"], **model, "source": emp["source"],
                    "loss_heaps": lh, "loss_zipf": lz,
                    "loss_joint": cfg["weights"]["heaps"] * lh + cfg["weights"]["zipf"] * lz,
                    "rstar_sim": simulated_rstar(freq),
                    "rank_min": cfg["rank_min"], "rank_max_used": hi,
                    "n_heaps_points": len(emp["t"]), "status": "ok"})
                if keep:
                    np.savez_compressed(
                        OUTPUTS / f"best_{population}_D0={d0}_seed={seed}.npz",
                        t=emp["t"], D=ds, frequencies=freq)
        except Exception as exc:                              # noqa: BLE001
            for population in populations:
                rows.append({"population": population, "D0": d0, "seed": seed,
                             "T": cfg["T"], **model, "status": f"failed: {type(exc).__name__}",
                             "loss_heaps": "", "loss_zipf": "", "loss_joint": "",
                             "rank_min": cfg["rank_min"], "rank_max_used": "",
                             "n_heaps_points": "", "source": ""})
            print(f"  !! D0={d0} seed={seed}: {exc}", file=sys.stderr)
    elapsed = round(time.monotonic() - start, 3)
    for row in rows:
        row["elapsed_s"] = elapsed
        append_row(results, row)
    return rows


# --------------------------------------------------------------------------- #
# summaries
# --------------------------------------------------------------------------- #
LOSSES = ("loss_heaps", "loss_zipf", "loss_joint")


def summarise(results: Path, output: Path) -> list[dict]:
    groups: dict[tuple[str, int], list[dict]] = {}
    for row in csv.DictReader(results.open()):
        if row["status"] == "ok":
            groups.setdefault((row["population"], int(row["D0"])), []).append(row)
    fields = ["population", "D0", "n_seeds"] + [f"{n}_{s}" for n in LOSSES
                                                for s in ("mean", "sd")]
    out_rows = []
    for (pop, d0), values in sorted(groups.items()):
        row = {"population": pop, "D0": d0, "n_seeds": len(values)}
        for name in LOSSES:
            x = np.array([float(v[name]) for v in values])
            row[f"{name}_mean"] = x.mean()
            row[f"{name}_sd"] = x.std(ddof=1) if len(x) > 1 else 0.0
        out_rows.append(row)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)
    return out_rows


def competitive_interval(runs: list[dict], population: str) -> dict:
    """Best D0 and the range of D0 that fits the data comparably well.

    The comparison is **paired across seeds**, and that choice is the whole
    point of this function.

    Every seed is a different realisation of the process, and realisations sit
    at systematically different loss levels: over the refined grid the per-seed
    mean joint loss ranges from 0.16 to 0.31 for the natives. That offset is
    common to every D0 within a seed, so it says nothing about which D0 is
    best — and it cancels exactly when D0 values are compared inside a seed.
    Sorting seeds by their own argmin shows this directly: the levels differ by
    a factor two while the argmins cluster within one grid step.

    Using the raw seed-to-seed SD as the threshold (the unpaired statistic)
    therefore charges the calibration for variance that is irrelevant to it,
    and inflates the competitive range by an order of magnitude: it gave
    [2,193 – 26,000] for the natives where the paired statistic gives
    [6,664 – 13,163]. Both are reported below, `seed_sd` being the unpaired one,
    so the difference is on the record rather than hidden in a choice.

    Concretely: each seed's profile is centred on its own mean over the D0 grid,
    the centred profiles are averaged, and the competitive range is the set of
    D0 whose mean centred loss lies within one standard error of the minimum.

    This is a statement about the flatness of the loss profile — deliberately
    *not* called a confidence interval.
    """
    grid: dict[int, dict[int, float]] = {}
    for row in runs:
        if row["population"] == population and row["status"] == "ok":
            grid.setdefault(int(row["D0"]), {})[int(row["seed"])] = float(row["loss_joint"])
    if not grid:
        return {}

    d0s = sorted(grid)
    seeds = sorted(set.intersection(*(set(v) for v in grid.values())))
    losses = np.array([[grid[d][s] for d in d0s] for s in seeds])   # seeds x D0

    centred = losses - losses.mean(axis=1, keepdims=True)           # drop the offset
    profile = centred.mean(axis=0)
    stderr = centred.std(axis=0, ddof=1) / np.sqrt(len(seeds)) if len(seeds) > 1 \
        else np.zeros_like(profile)

    k = int(profile.argmin())
    inside = [d0s[i] for i in range(len(d0s)) if profile[i] <= profile[k] + stderr[k]]
    return {"population": population, "best_D0": int(d0s[k]),
            "loss": float(losses[:, k].mean()),
            "paired_se": float(stderr[k]),
            "seed_sd": float(losses[:, k].std(ddof=1)) if len(seeds) > 1 else 0.0,
            "per_seed_argmin_D0": [int(d0s[i]) for i in losses.argmin(axis=1)],
            "low": min(inside), "high": max(inside), "n_competitive": len(inside),
            "n_grid": len(d0s), "n_seeds": len(seeds)}


def plot_profiles(results: Path, summary_rows: list[dict]) -> None:
    runs = [r for r in csv.DictReader(results.open()) if r["status"] == "ok"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.25))
    titles = ("Heaps log-RMSE", "Zipf log-relative-frequency RMSE",
              "joint selection loss")
    for ax, metric, title in zip(axes, LOSSES, titles):
        for pop, colour in POP_COLORS.items():
            subset = sorted((r for r in summary_rows if r["population"] == pop),
                            key=lambda r: int(r["D0"]))
            if not subset:
                continue
            seeds = [r for r in runs if r["population"] == pop]
            ax.scatter([float(r["D0"]) for r in seeds],
                       [float(r[metric]) for r in seeds],
                       color=colour, alpha=.35, s=18, zorder=1)
            ax.errorbar([r["D0"] for r in subset],
                        [r[f"{metric}_mean"] for r in subset],
                        yerr=[r[f"{metric}_sd"] for r in subset],
                        marker="o", ms=4, color=colour, capsize=3,
                        label=pop, zorder=2)
        ax.set(xscale="log", xlabel="$D_0$", title=title)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("loss")
    fig.suptitle("D0 calibration: loss profiles (points = individual seeds, "
                 "bars = mean ± seed SD)", y=.99)
    fig.tight_layout()
    fig.savefig(OUTPUTS / "loss_vs_d0.png", dpi=180)
    plt.close(fig)


def plot_best(population: str, d0: int, seed: int, cfg: dict) -> None:
    saved = np.load(OUTPUTS / f"best_{population}_D0={d0}_seed={seed}.npz")
    emp = load_empirical(population, cfg["T"])
    colour = POP_COLORS[population]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(emp["t"], emp["D"], color="black", alpha=.5, lw=2, label="data")
    axes[0].plot(saved["t"], saved["D"], color=colour, label=f"model $D_0$={d0:,}")
    axes[0].set(xscale="log", yscale="log", xlabel="tokens $t$", ylabel="$D(t)$")
    axes[0].legend(fontsize=8)
    hi = min(len(emp["frequency"]), len(saved["frequencies"]), cfg["rank_max"])
    ranks = sample_grid(cfg["rank_min"], hi, cfg["rank_points"], cfg["rank_spacing"])
    axes[1].plot(ranks, emp["frequency"][ranks - 1] / emp["frequency"].sum(),
                 "o", ms=3, color="black", alpha=.6, label="data")
    axes[1].plot(ranks, saved["frequencies"][ranks - 1] / saved["frequencies"].sum(),
                 "o", ms=3, color=colour, label="model")
    axes[1].set(xscale="log", yscale="log", xlabel="rank $R$",
                ylabel="relative frequency")
    axes[1].legend(fontsize=8)
    fig.suptitle(f"{population}: best joint fit (D0={d0:,}, seed={seed})", y=.99)
    fig.tight_layout()
    fig.savefig(OUTPUTS / f"best_{population}_comparison.png", dpi=180)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True,
                    help="JSON holding the D0 grid and seeds; required, because "
                         "the built-in DEFAULTS grid is not the paper's "
                         "(manifests/d0_calibration.json or _refined.json)")
    ap.add_argument("--dry-run", action="store_true",
                    help="one D0 and one seed, to check the plumbing")
    ap.add_argument("--population", choices=tuple(POPULATIONS))
    ap.add_argument("--d0-values", help="comma-separated D0 grid (replaces the default)")
    ap.add_argument("--seeds", help="comma-separated integer seeds")
    ap.add_argument("--T", type=int)
    ap.add_argument("--heaps-weight", type=float)
    ap.add_argument("--zipf-weight", type=float)
    ap.add_argument("--tag", required=True,
                    help="suffix of the output directory "
                         "outputs/d0_calibration_<tag>; required, because the "
                         "two sweeps the paper reports are read from two "
                         "different directories and an untagged run writes "
                         "where nothing looks")
    ap.add_argument("--summarise-only", action="store_true",
                    help="skip the grid and rebuild the summary, plots and report "
                         "from an existing runs.csv")
    ap.add_argument("--allow-long-run", action="store_true",
                    help="required above T = 1e8 (a 1e8 run costs ~20 s and 380 MB)")
    args = ap.parse_args()

    conf = json.loads(json.dumps(DEFAULTS))
    if args.config:
        user = json.loads(args.config.read_text())
        conf["model"].update(user.get("model", {}))
        conf["calibration"].update(user.get("calibration", {}))
    model, cfg = conf["model"], conf["calibration"]

    if args.T is not None:
        cfg["T"] = args.T
    if args.seeds:
        cfg["seeds"] = [int(v) for v in args.seeds.split(",") if v]
    if args.d0_values:
        cfg["d0_grid"] = {}
        cfg["d0_values"] = [int(v) for v in args.d0_values.split(",") if v]
    if args.heaps_weight is not None:
        cfg["weights"]["heaps"] = args.heaps_weight
    if args.zipf_weight is not None:
        cfg["weights"]["zipf"] = args.zipf_weight

    if not (model["N0"] > 0 and model["rho"] >= 0 and model["nu"] >= 0
            and 0 <= model["p"] <= 1):
        ap.error("require N0 > 0, rho/nu >= 0, 0 <= p <= 1")
    if min(cfg["weights"].values()) < 0 or sum(cfg["weights"].values()) <= 0:
        ap.error("loss weights must be non-negative and not both zero")
    if not cfg["seeds"]:
        ap.error("provide at least one seed")
    if cfg["T"] > 100_000_000 and not args.allow_long_run:
        ap.error("T > 1e8 requires --allow-long-run after a resource estimate")

    global OUTPUTS
    if args.tag:
        OUTPUTS = OUTPUTS.parent / f"d0_calibration_{args.tag}"
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(exist_ok=True)

    values = d0_grid(cfg)
    seeds = cfg["seeds"]
    if args.dry_run:
        values, seeds = values[len(values) // 2:len(values) // 2 + 1], seeds[:1]
    populations = (args.population,) if args.population else tuple(POPULATIONS)

    print(f"T = {cfg['T']:,} | seeds {seeds} | {len(values)} D0 values "
          f"[{values[0]:,} .. {values[-1]:,}]")
    for pop in populations:
        emp = load_empirical(pop, cfg["T"])
        print(f"  {pop:<8s} <- {emp['source']}: D(t) to t={emp['t'][-1]:,}, "
              f"V={len(emp['frequency']):,}, {int(emp['frequency'].sum()):,} tokens")

    results = OUTPUTS / "runs.csv"
    done = ({(r["population"], int(r["D0"]), int(r["seed"]))
             for r in csv.DictReader(results.open())} if results.exists() else set())

    binary = WORK / "umt_fast_seeded"
    compile_simulator(binary, WORK)
    failed = 0
    try:
        total = len(values) * len(seeds)
        for i, d0 in enumerate(values, 1):
            if args.summarise_only:
                break
            for seed in seeds:
                missing = tuple(p for p in populations if (p, d0, seed) not in done)
                if not missing:
                    continue
                rows = run_pair(binary, missing, d0, seed, cfg, model, results)
                failed += sum(1 for row in rows if row["status"] != "ok")
                for row in rows:
                    if row["status"] == "ok":
                        print(f"  [{i:>3}/{total // len(seeds)}] {row['population']:<8s} "
                              f"D0={d0:>7,} seed={seed} "
                              f"heaps={row['loss_heaps']:.4f} zipf={row['loss_zipf']:.4f} "
                              f"joint={row['loss_joint']:.4f} ({row['elapsed_s']:.0f}s)",
                              flush=True)

        summary_rows = summarise(results, OUTPUTS / "summary_by_d0.csv")
        plot_profiles(results, summary_rows)

        report = []
        runs = [r for r in csv.DictReader(results.open()) if r["status"] == "ok"]
        for pop in populations:
            info = competitive_interval(runs, pop)
            if not info:
                continue
            candidates = [r for r in runs if r["population"] == pop
                          and int(r["D0"]) == info["best_D0"]]
            seed = int(min(candidates, key=lambda r: float(r["loss_joint"]))["seed"])
            kept = OUTPUTS / f"best_{pop}_D0={info['best_D0']}_seed={seed}.npz"
            if not kept.exists():
                run_pair(binary, (pop,), info["best_D0"], seed, cfg, model,
                         OUTPUTS / "best_runs.csv", keep=True)
            plot_best(pop, info["best_D0"], seed, cfg)
            info["representative_seed"] = seed
            report.append(info)
            print(f"\n{pop}: best D0 = {info['best_D0']:,}  "
                  f"(joint loss {info['loss']:.4f} over {info['n_seeds']} seeds)\n"
                  f"  competitive range [{info['low']:,}, {info['high']:,}] "
                  f"= {info['n_competitive']}/{info['n_grid']} grid points within one "
                  f"paired standard error ({info['paired_se']:.4f}) of the minimum\n"
                  f"  per-seed argmin D0: "
                  f"{', '.join(f'{v:,}' for v in info['per_seed_argmin_D0'])}\n"
                  f"  (the unpaired seed-SD is {info['seed_sd']:.4f}, "
                  f"{info['seed_sd'] / max(info['paired_se'], 1e-12):.0f}x larger: it "
                  f"measures how much realisations differ in level, not in argmin)")
        (OUTPUTS / "calibration_report.json").write_text(json.dumps(
            {"model": model, "calibration": cfg, "populations": report}, indent=2))
    finally:
        binary.unlink(missing_ok=True)

    # A failed run is written to runs.csv with status != ok and then dropped by
    # `summarise`, so without this the sweep can lose half its grid — or all of
    # it, leaving an empty report — and still exit 0. It is invoked from a
    # notebook, where a zero exit is the only thing anyone sees.
    if failed:
        print(f"\n{failed} of {len(values) * len(seeds) * len(populations)} runs "
              f"FAILED; see the status column of "
              f"{results.relative_to(REPO_ROOT)}", file=sys.stderr)
        return 1
    if not report and not args.summarise_only:
        print("\nno population could be summarised: every run failed or the "
              "grid was empty", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
