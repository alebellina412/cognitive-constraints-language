#!/usr/bin/env python3
"""Fitting utilities for rank-frequency / power-law analysis (single source of truth).

Two independent estimators, reported together; they agree within their bands in
all five languages (Table S2 of the supplementary information):

1. Two-regime rank-frequency OLS.
   Zipf form  p(R) ∝ R^(-alpha).  We fit log10 p(R) vs log10 R with ordinary
   least squares over two rank windows:
     * head  R in [1, 1e3]                 -> alpha1  (~1, universal)
     * tail  R in [1e4, ...], count >= 2   -> alpha2  ("of order 2")

2. Clauset-Shalizi-Newman discrete MLE.
   Fits the distribution of word counts  P(f) ∝ f^(-beta)  with a KS-optimal
   lower cutoff x_min (discrete power law, Hurwitz-zeta normalisation).
   The Zipf rank exponent is recovered as  alpha = 1/(beta - 1).

All functions take a *descending integer frequency vector* (one entry per word
type, the aggregated corpus count), as produced by build_reduced.py.

References: Clauset, Shalizi & Newman, SIAM Review 51, 661 (2009);
Moreno-Sanchez, Font-Clos & Corral, PLoS ONE (2016), beta = 1 + 1/alpha.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy import optimize, special


# --------------------------------------------------------------------------- #
# Two-regime rank-frequency OLS
# --------------------------------------------------------------------------- #
def _ols_loglog(rank: np.ndarray, prob: np.ndarray) -> tuple[float, float, float]:
    """OLS slope of log10(prob) on log10(rank). Returns (alpha, r_squared, se).

    alpha is the positive Zipf exponent (negative of the fitted slope); `se` is
    its nominal OLS standard error,

        se = sqrt( (SS_res / (n-2)) / sum (x - xbar)^2 ).

    IMPORTANT — how to read `se`. It assumes independent residuals, which a
    rank-frequency curve does not have: consecutive ranks are a smooth curve, so
    the residuals are strongly autocorrelated and `se` *understates* the real
    uncertainty, often by an order of magnitude. It is reported because it is
    the conventional number a reader expects, but the honest error bar on every
    exponent in this paper is the fit-window band (`*_band` helpers below) —
    the same choice already made for the x_min band of Figure 1A.
    """
    x = np.log10(rank.astype(float))
    y = np.log10(prob.astype(float))
    n = x.size
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    sxx = float(np.sum((x - x.mean()) ** 2))
    se = float(np.sqrt((ss_res / (n - 2)) / sxx)) if n > 2 and sxx > 0 else float("nan")
    return -slope, r2, se


def two_regime_ols(
    freq: np.ndarray,
    head: tuple[int, int] = (1, 1000),
    tail_start: int = 10_000,
    tail_min_count: int = 2,
) -> dict:
    """Fit alpha1 (head) and alpha2 (tail) of the rank-frequency curve.

    Parameters
    ----------
    freq : descending integer count per word type (rank 1 = most frequent).
    head : (Rmin, Rmax) rank window for alpha1.
    tail_start : first rank of the tail window for alpha2.
    tail_min_count : drop the freq==1 hapax staircase; keep counts >= this.
    """
    freq = np.asarray(freq)
    freq = freq[freq > 0]
    freq = np.sort(freq)[::-1]  # ensure descending
    V = freq.size
    rank = np.arange(1, V + 1)
    prob = freq / freq.sum()

    # head
    h0, h1 = head
    hmask = (rank >= h0) & (rank <= min(h1, V))
    alpha1, r2_1, se_1 = _ols_loglog(rank[hmask], prob[hmask])

    # tail: rank >= tail_start AND count >= tail_min_count
    tmask = (rank >= tail_start) & (freq >= tail_min_count)
    if tmask.sum() >= 5:
        alpha2, r2_2, se_2 = _ols_loglog(rank[tmask], prob[tmask])
        tail_n = int(tmask.sum())
    else:
        alpha2, r2_2, se_2 = float("nan"), float("nan"), float("nan")
        tail_n = int(tmask.sum())

    return {
        "alpha1": alpha1,
        "alpha1_r2": r2_1,
        "alpha1_se": se_1,
        "alpha2": alpha2,
        "alpha2_r2": r2_2,
        "alpha2_se": se_2,
        "head_window": (int(h0), int(min(h1, V))),
        "tail_start": int(tail_start),
        "tail_n_points": tail_n,
        "vocab": int(V),
    }


def ols_rank_prob(
    rank: np.ndarray,
    prob: np.ndarray,
    rmin: float,
    rmax: float,
    n_points: int = 400,
) -> dict:
    """Power-law exponent of a decreasing log-log curve over [rmin, rmax].

    Written for rank-frequency curves, but it fits any positive decreasing
    y(x) sampled on a log grid — including a log-binned density p(q), where the
    exponent returned is the tau of P(q) ~ q^-tau.

    For curves given as a staircase of blocks (e.g. the merged phrase spectrum,
    where one block can cover 10^8 phrases) fitting the blocks directly would
    weight the head enormously more than the tail. The curve is therefore
    resampled on a grid uniform in log R -- the same weighting the eye applies to
    the log-log plot -- and fitted with OLS.

    `rank` must be increasing and `prob` decreasing (the curve as plotted).
    """
    rank = np.asarray(rank, dtype=float)
    prob = np.asarray(prob, dtype=float)
    lo = max(float(rmin), rank[0])
    hi = min(float(rmax), rank[-1])
    if not hi > lo:
        return {"alpha": float("nan"), "r2": float("nan"), "alpha_se": float("nan"),
                "n_points": 0, "window": (lo, hi)}
    grid = np.logspace(np.log10(lo), np.log10(hi), n_points)
    idx = np.clip(np.searchsorted(rank, grid, side="left"), 0, rank.size - 1)
    alpha, r2, se = _ols_loglog(grid, prob[idx])
    return {"alpha": alpha, "r2": r2, "alpha_se": se, "n_points": int(grid.size),
            "window": (lo, hi)}


def heaps_exponent(t: np.ndarray, D: np.ndarray, tmin: float, tmax: float) -> dict:
    """Heaps exponent b of D(t) ~ t^b over [tmin, tmax] (OLS in log-log).

    The counterpart of `ols_rank_prob` for a *growing* curve: the vocabulary
    growth of Figure S4. Returns b (positive), R^2 and the number of points used.
    """
    t = np.asarray(t, dtype=float)
    D = np.asarray(D, dtype=float)
    m = (t >= tmin) & (t <= tmax) & (D > 0)
    if m.sum() < 3:
        return {"b": float("nan"), "r2": float("nan"), "b_se": float("nan"),
                "n_points": int(m.sum()), "window": (float(tmin), float(tmax))}
    alpha, r2, se = _ols_loglog(t[m], D[m])
    return {"b": -alpha, "r2": r2, "b_se": se, "n_points": int(m.sum()),
            "window": (float(tmin), float(tmax))}


# --------------------------------------------------------------------------- #
# Broken power law — the crossover scale D0 read straight off the data
# --------------------------------------------------------------------------- #
def broken_power_law(
    rank: np.ndarray,
    prob: np.ndarray,
    rmin: float = 10.0,
    rmax: float | None = None,
    n_points: int = 400,
    n_breaks: int = 240,
    break_margin: float = 0.5,
) -> dict:
    """Continuous two-slope fit of a rank-frequency curve with a FREE breakpoint.

    Fits, in log10-log10 space,

        log p = c - a1 * log R                        for R <= R*
        log p = c - a1 * log R* - a2 * (log R - log R*)   for R >  R*

    which is linear in (c, a1, a2) once R* is fixed. The breakpoint is therefore
    estimated by *profile* least squares: scan log10 R* on a grid, solve the
    linear problem at each, keep the minimum residual sum of squares.

    Why this matters here. In the model, D0 is a vocabulary size — the number of
    types beyond which novelty is attenuated. On a rank-frequency curve the rank
    *is* a position in the type ranking, so the fitted R* is directly comparable
    with the calibrated D0, with no conversion and no free normalisation. This
    gives an estimate of D0 from the data alone, independent of any simulation.

    The curve is resampled on a log-uniform grid before fitting, so the head and
    the tail carry equal weight (the raw vector has millions of tail ranks and a
    handful of head ranks; fitting it directly would fit the tail only).

    Parameters
    ----------
    rmin, rmax : fit range in rank; `rmax` defaults to the last rank.
    n_points   : points on the log-uniform sampling grid.
    n_breaks   : candidate breakpoints scanned.
    break_margin : decades of the fit range excluded at each end, so a candidate
        breakpoint always keeps enough points on both sides to define a slope.

    Returns `R_star`, `alpha1`, `alpha2`, `rss`, `r2`, the profile curve
    (`break_grid`, `break_rss`) and `R_star_interval`, the range of breakpoints
    whose RSS lies within the standard 1-sigma likelihood-ratio threshold
    RSS_min * (1 + 1/(n-3)) — the profile-likelihood interval of the breakpoint.
    """
    rank = np.asarray(rank, dtype=float)
    prob = np.asarray(prob, dtype=float)
    lo = max(float(rmin), float(rank[0]))
    hi = float(rank[-1]) if rmax is None else min(float(rmax), float(rank[-1]))
    if not hi > lo * 10:
        raise ValueError("fit range must span at least one decade")

    grid = np.logspace(np.log10(lo), np.log10(hi), n_points)
    idx = np.clip(np.searchsorted(rank, grid, side="left"), 0, rank.size - 1)
    x = np.log10(grid)
    y = np.log10(prob[idx])
    ok = np.isfinite(y)
    x, y = x[ok], y[ok]
    n = x.size

    breaks = np.linspace(x[0] + break_margin, x[-1] - break_margin, n_breaks)
    rss = np.empty(breaks.size)
    coef = np.empty((breaks.size, 3))
    ones = np.ones_like(x)
    for i, xb in enumerate(breaks):
        design = np.column_stack([ones, x, np.maximum(0.0, x - xb)])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        coef[i] = beta
        rss[i] = float(np.sum((y - design @ beta) ** 2))

    best = int(np.argmin(rss))
    c, b1, b2 = coef[best]
    ss_tot = float(np.sum((y - y.mean()) ** 2))

    # profile-likelihood interval: RSS(R*) <= RSS_min * (1 + 1/(n - 3))
    threshold = rss[best] * (1.0 + 1.0 / max(n - 3, 1))
    inside = breaks[rss <= threshold]

    return {
        "R_star": float(10.0 ** breaks[best]),
        "R_star_interval": (float(10.0 ** inside.min()), float(10.0 ** inside.max())),
        "alpha1": float(-b1),
        "alpha2": float(-(b1 + b2)),
        "rss": float(rss[best]),
        "r2": 1.0 - rss[best] / ss_tot if ss_tot > 0 else float("nan"),
        "n_points": int(n),
        "window": (lo, hi),
        "break_grid": 10.0 ** breaks,
        "break_rss": rss,
    }


def broken_power_law_band(
    rank: np.ndarray,
    prob: np.ndarray,
    rmin: float = 10.0,
    rmax: float | None = None,
    rmins=(3.0, 10.0, 30.0, 100.0),
    n_points: int = 400,
) -> dict:
    """`broken_power_law` plus the band swept by the lower end of the fit range.

    Where the fit starts is the one choice a reader could most reasonably
    disagree about (how much of the very-high-frequency head belongs to the
    scaling regime), so it is swept explicitly and reported next to the
    profile-likelihood interval.
    """
    central = broken_power_law(rank, prob, rmin, rmax, n_points)
    stars = []
    for r in rmins:
        try:
            stars.append(broken_power_law(rank, prob, r, rmax, n_points)["R_star"])
        except ValueError:
            continue
    return {**central, "R_star_band": _band(stars), "rmins": tuple(rmins)}


# --------------------------------------------------------------------------- #
# Fit-window bands — the reported uncertainty on every exponent
# --------------------------------------------------------------------------- #
# Why a band and not the OLS standard error: see the note in `_ols_loglog`.
# A rank-frequency curve is smooth, so the nominal se is far too small; what
# actually moves an exponent is where the fit window is placed. Sweeping a set
# of a-priori-reasonable windows and reporting the resulting spread is the same
# convention already used for the x_min band of the Figure 1A spectrum MLE, so
# every exponent in the paper carries the same kind of error bar.

#: the swept windows are the reported one scaled by these factors, so a band is
#: always centred on the fit it accompanies -- including on the much smaller
#: corpora of Figure S4, where the reported windows are two decades lower
WINDOW_FACTORS = (0.5, 1.0, 2.0)


def _band(values) -> dict:
    """Central value, min, max and half-spread of a set of refits."""
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return {"lo": float("nan"), "hi": float("nan"), "half_spread": float("nan"),
                "n": 0}
    return {"lo": float(v.min()), "hi": float(v.max()),
            "half_spread": float((v.max() - v.min()) / 2), "n": int(v.size)}


def two_regime_ols_band(
    freq: np.ndarray,
    head: tuple[int, int] = (1, 1000),
    tail_start: int = 10_000,
    tail_min_count: int = 2,
    head_windows=None,
    tail_starts=None,
    factors=WINDOW_FACTORS,
) -> dict:
    """`two_regime_ols` at the reported window, plus the band over all windows.

    Adds `alpha1_band` / `alpha2_band` (each with lo, hi, half_spread, n) to the
    central fit, so a table can print e.g. alpha2 = 1.99 [1.94-2.05].

    By default the swept windows are *derived from the reported ones*: the head
    upper end and the tail start are each halved and doubled. That is what makes
    the band meaningful — it brackets the fit it is quoted next to, whatever the
    corpus size. Passing explicit `head_windows` / `tail_starts` overrides this.
    """
    if head_windows is None:
        head_windows = tuple((head[0], int(round(head[1] * f))) for f in factors)
    if tail_starts is None:
        tail_starts = tuple(int(round(tail_start * f)) for f in factors)
    central = two_regime_ols(freq, head, tail_start, tail_min_count)
    a1 = [two_regime_ols(freq, w, tail_start, tail_min_count)["alpha1"]
          for w in head_windows]
    a2 = [two_regime_ols(freq, head, s, tail_min_count)["alpha2"]
          for s in tail_starts]
    return {**central,
            "alpha1_band": _band(a1), "alpha2_band": _band(a2),
            "head_windows": tuple(head_windows), "tail_starts": tuple(tail_starts)}


def ols_rank_prob_band(
    rank: np.ndarray,
    prob: np.ndarray,
    rmin: float,
    rmax: float,
    n_points: int = 400,
    shifts=(0.5, 0.0, -0.5),
) -> dict:
    """`ols_rank_prob` plus the band swept by moving the window in log10 R.

    Each shift s moves both ends of the window by s decades inward/outward
    (rmin/10^s, rmax*10^s), which is how much a reader could reasonably disagree
    about where the scaling regime starts and stops.
    """
    central = ols_rank_prob(rank, prob, rmin, rmax, n_points)
    alphas = [ols_rank_prob(rank, prob, rmin / 10.0 ** s, rmax * 10.0 ** s,
                            n_points)["alpha"] for s in shifts]
    return {**central, "alpha_band": _band(alphas), "shifts": tuple(shifts)}


def heaps_exponent_band(
    t: np.ndarray,
    D: np.ndarray,
    tmin: float,
    tmax: float,
    shifts=(0.5, 0.0, -0.5),
) -> dict:
    """`heaps_exponent` plus the band swept by moving the fit window."""
    central = heaps_exponent(t, D, tmin, tmax)
    bs = [heaps_exponent(t, D, tmin / 10.0 ** s, tmax * 10.0 ** s)["b"]
          for s in shifts]
    return {**central, "b_band": _band(bs), "shifts": tuple(shifts)}


#: split points swept for the two-regime fit of a log-binned density
SPLIT_POINTS = (3e2, 1e3, 3e3)


def logbin_split_band(
    centre: np.ndarray,
    density: np.ndarray,
    split: float = 1e3,
    splits=SPLIT_POINTS,
) -> dict:
    """Exponents of a log-binned density below/above `split`, plus their bands.

    The counterpart of `two_regime_ols_band` for a density p(q) rather than a
    rank curve. A log-binned density spans a fixed, fairly short range, so
    sliding the window ends by decades (what `ols_rank_prob_band` does) mostly
    runs off the data and gets clipped. What a reader can actually disagree
    about is **where the two regimes are split**, so that is what is swept.

    Returns `alpha_low` / `alpha_high` at the reported split with their R^2, and
    `alpha_low_band` / `alpha_high_band` over `splits`.
    """
    centre = np.asarray(centre, dtype=float)
    density = np.asarray(density, dtype=float)
    lo = ols_rank_prob(centre, density, centre[0], split)
    hi = ols_rank_prob(centre, density, split, centre[-1])
    los, his = [], []
    for q in splits:
        if not centre[0] < q < centre[-1]:
            continue
        los.append(ols_rank_prob(centre, density, centre[0], q)["alpha"])
        his.append(ols_rank_prob(centre, density, q, centre[-1])["alpha"])
    return {"alpha_low": lo["alpha"], "alpha_low_r2": lo["r2"],
            "alpha_low_se": lo["alpha_se"],
            "alpha_high": hi["alpha"], "alpha_high_r2": hi["r2"],
            "alpha_high_se": hi["alpha_se"],
            "alpha_low_band": _band(los), "alpha_high_band": _band(his),
            "split": float(split), "splits": tuple(splits)}


def logbin_pdf(values: np.ndarray, n_bins: int = 20) -> dict:
    """Log-binned probability density of a positive sample (degrees, strengths).

    Heavy-tailed integer samples are unreadable as a raw histogram: the tail is
    one count per bin. Bins uniform in log10 fix that; each bin's probability
    mass is divided by its *width*, so what is plotted is a density and the
    slope is the exponent of P(q) ~ q^-tau.

    Returns `centre` (geometric bin centre), `density`, `edges`, `n_per_bin`.
    Empty bins are dropped.
    """
    v = np.asarray(values)
    v = v[v > 0]
    val, cnt = np.unique(v, return_counts=True)
    prob = cnt / cnt.sum()

    edges = np.logspace(np.log10(val[0]), np.log10(val[-1]), n_bins + 1)
    edges[-1] *= 1.000001                      # keep the largest value inside
    idx = np.digitize(val, edges) - 1
    mass = np.bincount(idx, weights=prob, minlength=n_bins)[:n_bins]
    n_per_bin = np.bincount(idx, weights=cnt, minlength=n_bins)[:n_bins]
    width = np.diff(edges)
    centre = np.sqrt(edges[:-1] * edges[1:])

    keep = mass > 0
    return {"centre": centre[keep], "density": (mass / width)[keep],
            "edges": edges, "n_per_bin": n_per_bin[keep].astype(np.int64)}


def logbin_rank_curve(freq: np.ndarray, n_bins: int = 20,
                      min_count: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Rank-frequency curve averaged in log-spaced rank bins.

    Small corpora end in a long staircase of types seen once or twice; plotting
    it raw (especially as R f(R), where the f = 1 plateau turns into a rising
    line) hides the shape. Ranks with count < `min_count` are dropped, as in the
    rest of the paper, and what is left is averaged over log-spaced rank bins.

    Returns (rank centre, mean frequency in the bin).
    """
    f = np.asarray(freq, dtype=float)
    f = np.sort(f)[::-1]
    keep = int(np.searchsorted(-f, -min_count, side="right"))
    if keep < 2:
        return np.array([]), np.array([])
    f = f[:keep]
    edges = np.unique(np.round(
        np.logspace(0, np.log10(keep), n_bins + 1)).astype(np.int64))
    centre, mean = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        if b > a:
            centre.append(np.sqrt(a * b))
            mean.append(f[a - 1:b - 1].mean())
    return np.array(centre), np.array(mean)


def local_slope(rank: np.ndarray, prob: np.ndarray, n_points: int = 60,
                half_width: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Local Zipf exponent: OLS slope in a sliding log-window of +-`half_width`.

    Returns (rank_centre, alpha). Used to show over how many decades a curve
    stays Zipfian (alpha ~ 1) instead of asserting it by eye.
    """
    rank = np.asarray(rank, dtype=float)
    prob = np.asarray(prob, dtype=float)
    lr, lp = np.log10(rank), np.log10(prob)
    centres = np.linspace(lr[0] + half_width, lr[-1] - half_width, n_points)
    out = []
    for c in centres:
        m = (lr >= c - half_width) & (lr <= c + half_width)
        out.append(-np.polyfit(lr[m], lp[m], 1)[0] if m.sum() >= 5 else np.nan)
    return 10.0 ** centres, np.array(out)


# --------------------------------------------------------------------------- #
# Clauset discrete-MLE power law on the counts P(f) ~ f^-beta
# --------------------------------------------------------------------------- #
def _discrete_loglik_beta(beta: float, x: np.ndarray, xmin: int) -> float:
    """Negative log-likelihood of a discrete power law with cutoff xmin."""
    n = x.size
    # Hurwitz zeta zeta(beta, xmin) normalises the tail x>=xmin.
    norm = special.zeta(beta, xmin)
    if norm <= 0 or not np.isfinite(norm):
        return np.inf
    return n * np.log(norm) + beta * np.sum(np.log(x))


def _mle_beta(x: np.ndarray, xmin: int) -> float:
    """MLE of beta for x >= xmin (bounded scalar minimisation)."""
    res = optimize.minimize_scalar(
        _discrete_loglik_beta,
        bounds=(1.01, 5.0),
        args=(x, xmin),
        method="bounded",
    )
    return float(res.x)


def _discrete_cdf_model(xs: np.ndarray, beta: float, xmin: int) -> np.ndarray:
    """Model CDF P(X <= x) for a discrete power law, evaluated at sorted xs."""
    z = special.zeta(beta, xmin)
    # P(X >= k) = zeta(beta, k) / zeta(beta, xmin); CDF = 1 - P(X >= k+1)
    return 1.0 - special.zeta(beta, xs + 1) / z


def clauset_discrete_mle(
    freq: np.ndarray,
    xmin_candidates: np.ndarray | None = None,
    max_candidates: int = 60,
) -> dict:
    """KS-optimal discrete power-law fit to the word counts.

    Scans candidate x_min values, fits beta by MLE for each, computes the KS
    distance between the empirical and model tail CDFs, and keeps the x_min with
    the smallest KS. Returns beta, x_min, KS, n_tail, and the Zipf exponent
    alpha = 1/(beta - 1).
    """
    x_all = np.asarray(freq)
    x_all = x_all[x_all > 0].astype(np.int64)

    uniq = np.unique(x_all)
    if xmin_candidates is None:
        # candidate x_min: unique count values, log-spaced if too many
        if uniq.size > max_candidates:
            idx = np.unique(
                np.round(np.linspace(0, uniq.size - 1, max_candidates)).astype(int)
            )
            xmin_candidates = uniq[idx]
        else:
            xmin_candidates = uniq
        # never let x_min reach the very top of the tail
        xmin_candidates = xmin_candidates[xmin_candidates <= uniq[-1] // 2 + 1]
        if xmin_candidates.size == 0:
            xmin_candidates = uniq[:1]

    best = None
    for xmin in xmin_candidates:
        xmin = int(xmin)
        tail = x_all[x_all >= xmin]
        if tail.size < 50:
            continue
        beta = _mle_beta(tail, xmin)
        xs = np.unique(tail)
        emp = np.searchsorted(np.sort(tail), xs, side="right") / tail.size
        mod = _discrete_cdf_model(xs, beta, xmin)
        ks = float(np.max(np.abs(emp - mod)))
        if best is None or ks < best["ks"]:
            best = {
                "beta": beta,
                "xmin": xmin,
                "ks": ks,
                "n_tail": int(tail.size),
                "alpha": 1.0 / (beta - 1.0) if beta > 1.0 else float("nan"),
            }
    if best is None:
        return {"beta": float("nan"), "xmin": -1, "ks": float("nan"),
                "n_tail": 0, "alpha": float("nan")}
    return best


# --------------------------------------------------------------------------- #
# Frequency-spectrum MLE -> tail Zipf exponent (the paper's "alpha (MLE)")
# --------------------------------------------------------------------------- #
def zipf_mle_spectrum(freq: np.ndarray, xmin: int = 2, xmax: int | None = None) -> dict:
    """Independent MLE estimate of the *tail* Zipf exponent alpha2.

    The paper's Fig 1A "alpha (MLE)" column is NOT the KS-optimal Clauset fit
    (that fits the upper tail of the counts = the most frequent words = the HEAD
    of the Zipf plot, giving alpha ~ 1). Instead it is a discrete power-law MLE
    on the frequency *spectrum* P(f) ∝ f^-beta over the full body of counts with
    a small lower cutoff (hapax f=1 dropped by default, as singletons are
    undersampled). Because the spectrum is dominated by rare words, this recovers
    the tail rank exponent via alpha = 1/(beta-1) — and reproduces the paper's
    column (which agrees with the OLS tail alpha2).

    Parameters
    ----------
    xmin : drop counts below this (default 2 -> exclude hapax).
    xmax : optional upper bound on counts included in the fit.
    """
    x = np.asarray(freq)
    x = x[x >= xmin]
    if xmax is not None:
        x = x[x <= xmax]
    x = x.astype(np.int64)
    if x.size < 50:
        return {"beta": float("nan"), "xmin": xmin, "alpha": float("nan"), "n": int(x.size)}
    beta = _mle_beta(x, xmin)
    return {
        "beta": beta,
        "xmin": xmin,
        "alpha": 1.0 / (beta - 1.0) if beta > 1.0 else float("nan"),
        "n": int(x.size),
    }


def zipf_mle_spectrum_band(
    freq: np.ndarray,
    xmin: int = 2,
    xmins: tuple[int, ...] = (2, 3, 4, 5),
) -> dict:
    """Reported spectrum-MLE alpha plus its two uncertainties.

    `xmin` is the *reported* cutoff (2 = "drop only the hapax", the one choice
    that needs no tuning). `xmins` is the sensitivity range: the spectrum is not
    an exact power law, so alpha drifts slowly with the cutoff, and that drift —
    not the sampling error — dominates the uncertainty. Both are returned so the
    SI can state them separately:

      * `sigma_stat` — MLE standard error. For a discrete power law
        sigma_beta ~= (beta-1)/sqrt(n) (Clauset et al. 2009, eq. 3.2); with
        alpha = 1/(beta-1) and |d alpha / d beta| = alpha^2 this collapses to
        sigma_alpha ~= alpha/sqrt(n).
      * `alpha_min`, `alpha_max` — the spread of alpha over `xmins`.
    """
    fits = {int(x): zipf_mle_spectrum(freq, xmin=int(x)) for x in sorted(set(xmins) | {xmin})}
    ref = fits[int(xmin)]
    alphas = [fits[x]["alpha"] for x in sorted(xmins)]
    return {
        "alpha": ref["alpha"],
        "beta": ref["beta"],
        "xmin": int(xmin),
        "n": ref["n"],
        "sigma_stat": ref["alpha"] / np.sqrt(ref["n"]) if ref["n"] else float("nan"),
        "alpha_min": float(np.min(alphas)),
        "alpha_max": float(np.max(alphas)),
        "per_xmin": {x: fits[x]["alpha"] for x in sorted(xmins)},
    }


# --------------------------------------------------------------------------- #
# Convenience: both estimators at once
# --------------------------------------------------------------------------- #
@dataclass
class FitResult:
    vocab: int
    alpha1: float          # head OLS
    alpha2: float          # tail OLS
    alpha_mle: float       # 1/(beta-1)
    beta: float            # Clauset discrete MLE
    xmin: int
    ks: float
    alpha1_r2: float
    alpha2_r2: float

    def as_dict(self) -> dict:
        return asdict(self)


def fit_all(freq: np.ndarray) -> FitResult:
    """Run both estimators on a descending frequency vector."""
    ols = two_regime_ols(freq)
    mle = clauset_discrete_mle(freq)
    return FitResult(
        vocab=ols["vocab"],
        alpha1=round(ols["alpha1"], 3),
        alpha2=round(ols["alpha2"], 3),
        alpha_mle=round(mle["alpha"], 3),
        beta=round(mle["beta"], 4),
        xmin=mle["xmin"],
        ks=round(mle["ks"], 4),
        alpha1_r2=round(ols["alpha1_r2"], 4),
        alpha2_r2=round(ols["alpha2_r2"], 4),
    )


if __name__ == "__main__":
    # Smoke test on a synthetic Zipf sample (alpha ~ 1).
    rng = np.random.default_rng(0)
    ranks = np.arange(1, 200_000)
    counts = np.maximum(1, np.round(1e7 / ranks**1.05)).astype(np.int64)
    print(fit_all(counts).as_dict())
