#!/usr/bin/env python3
"""Appendix figure S4 — learners of English against native speakers.

Two panels:

  (A) vocabulary growth D(t) for the two populations, each averaged over random
      orderings of its constituent texts, with a 5-95% band;
  (B) the rank-frequency curves as R p(R), in relative frequency, so that a
      Zipfian regime is flat and the end of the plateau is the crossover itself.

Panel B is normalised because the two corpora differ in size by a factor of 170:
in raw counts they would sit two decades apart for that reason alone, and the
shapes — which is what the panel compares — could not be read against each other.
The fitted crossover R* is marked on each curve, so the edge of the plateau is
measured rather than eyeballed.

Three groups are tabulated but only two are drawn. The COREFL native control is
genre-matched to the learners and is listed for completeness, but at 52 k tokens
it is two orders of magnitude too small to serve as the native reference; the
reference is SPGC English at 10^8 tokens, the corpus and length used throughout
the paper.

**R* is not length-independent**, so the two drawn values cannot simply be
divided: the same native corpus truncated to the learners' length gives a much
smaller R*. `rstar_vs_length.py` measures that, and this module asserts that the
learner kernel is still the smaller one once length is controlled for. Run
`python src/rstar_vs_length.py` first — its table is read here.

Usage:
    python src/rstar_vs_length.py      # writes outputs/tables/rstar_vs_length.csv
    python src/figure_SI4.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.lines import Line2D                                # noqa: E402

import estimate_d0 as D0                                           # noqa: E402
import plotting as P                                               # noqa: E402
from fits import (heaps_exponent_band, logbin_rank_curve,          # noqa: E402
                  two_regime_ols_band)
from io_reduced import load_corefl, load_heaps                     # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NAT = "L1 natives (Gutenberg)"
LEA = "L2 learners (COREFL)"
CTL = "L1 control (COREFL)"


def load():
    """The three populations, as (name, colour key, freq, heaps dict, drawn)."""
    learners = load_corefl("learner_all")      # L1 Spanish + L1 German, English L2
    natives_corefl = load_corefl("native_en")  # COREFL native controls, same tasks
    gutenberg = load_heaps("en")               # SPGC English, 1e8 tokens

    groups = [
        (LEA, "learners", learners["freq"], learners, True),
        (NAT, "gutenberg", gutenberg["freq_matched"], gutenberg, True),
        (CTL, "natives", natives_corefl["freq"], natives_corefl, False),
    ]

    assert int(gutenberg["match_tokens"]) == 100_000_000, (
        "the native reference must be the 1e8 curve - rebuild with\n"
        "  python src/build_reduced.py --heaps --langs en "
        "--heaps-max-tokens 100000000 --match-tokens 100000000 "
        "--heaps-points 120 --shuffles 30\n"
        "Note the option names: --max-tokens belongs to --tokens and is a "
        "silent no-op here, and it is --match-tokens that sets the quantity "
        "this assert reads.")
    return groups, learners


def composition(groups) -> pd.DataFrame:
    """Size of each population, as a sanity view before the fits."""
    return pd.DataFrame([{"group": n,
                          "texts/books": int(g.get("n_texts", g.get("n_books_pool", -1))),
                          "tokens": int(f.sum()),
                          "types": int(f.size),
                          "in figure": "yes" if drawn else "table only"}
                         for n, _, f, g, drawn in groups])


def exponents(groups) -> pd.DataFrame:
    """Heaps, Zipf and crossover fits for each population.

    The fit windows are two decades below those of Table S2, because they have to
    fit inside the learner corpus; the same windows are then applied to every
    group, so only the comparison between rows is meaningful.
    """
    rows = []
    for name, _, freq, g, _ in groups:
        t, D = g["heaps_t"], g["heaps_mean"]
        lo = heaps_exponent_band(t, D, 1e2, 1e4)
        hi = heaps_exponent_band(t, D, 1e4, t[-1])
        z = two_regime_ols_band(freq, head=(1, 100), tail_start=300, tail_min_count=2)
        d0 = D0.estimate(freq)                       # free-breakpoint crossover
        rows.append({
            "group": name,
            "tokens": int(freq.sum()),
            "types": int(freq.size),
            "b (1e2-1e4)": round(lo["b"], 3),
            "b_lo": round(lo["b_band"]["lo"], 3),
            "b_hi": round(lo["b_band"]["hi"], 3),
            "R2": round(lo["r2"], 3),
            "b (>1e4)": round(hi["b"], 3) if hi["n_points"] >= 3 else np.nan,
            "D(1e4)": int(np.interp(1e4, t, D)),
            "D(1e5)": int(np.interp(1e5, t, D)) if t[-1] >= 1e5 else np.nan,
            "alpha1": round(z["alpha1"], 2),
            "alpha1_lo": round(z["alpha1_band"]["lo"], 2),
            "alpha1_hi": round(z["alpha1_band"]["hi"], 2),
            "alpha2": round(z["alpha2"], 2),
            "alpha2_lo": round(z["alpha2_band"]["lo"], 2),
            "alpha2_hi": round(z["alpha2_band"]["hi"], 2),
            "R_star": round(d0["R_star"]),
            "R_star_lo": round(d0["R_star_band"]["lo"]),
            "R_star_hi": round(d0["R_star_band"]["hi"]),
        })
    return pd.DataFrame(rows).set_index("group")


def matched_length_check(table: pd.DataFrame, verbose: bool = True) -> pd.Series:
    """Compare the two kernels at matched corpus length, and assert the gap holds."""
    path = P.table_path("rstar_vs_length.csv")
    if not os.path.exists(path):
        raise SystemExit(
            f"{os.path.relpath(path, REPO)} missing.\n"
            f"It is the length control this comparison depends on, and it is the "
            f"one thing in this figure that cannot be computed from "
            f"data_reduced/ alone: measuring R* on the native corpus truncated "
            f"to a range of lengths means re-reading the raw tokens.\n"
            f"Run notebook 3 (section 3.4), or directly:\n"
            f"    python src/rstar_vs_length.py       # needs data_raw/")
    matched = pd.read_csv(path).set_index("tokens")
    mrow = matched.loc[int(table.loc[LEA, "tokens"])]

    if verbose:
        print(f"naive ratio (different lengths): "
              f"{table.loc[NAT, 'R_star'] / table.loc[LEA, 'R_star']:.0f}x  "
              f"[{table.loc[NAT, 'R_star']:,} at 1e8 vs "
              f"{table.loc[LEA, 'R_star']:,} at {int(table.loc[LEA, 'tokens']):,}]")
        print(f"MATCHED length ({int(mrow.name):,} tokens): natives R* = "
              f"{int(mrow.R_star):,} [{int(mrow.R_star_min):,}, "
              f"{int(mrow.R_star_max):,}] ({int(mrow.types):,} types)")
        print(f"  -> ratio at matched length = "
              f"{mrow.R_star / table.loc[LEA, 'R_star']:.0f}x, bands disjoint: "
              f"{int(mrow.R_star_min) > table.loc[LEA, 'R_star_hi']}")

    assert int(mrow.R_star_min) > table.loc[LEA, "R_star_hi"], \
        "the learner kernel must still be smaller once length is controlled for"
    return mrow


def write_table(table: pd.DataFrame, learners) -> str:
    """Write the population table as CSV and markdown."""
    table.to_csv(P.table_path("SI4_exponents.csv"))

    head_md = ("| group | tokens | types | $b$ ($10^2$–$10^4$) | $D(10^4)$ | "
               "$\\alpha_1$ | $\\alpha_2$ | $R^*$ |\n"
               "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")

    def band(r, col):
        # iterrows() upcasts a mixed row to float, so integers are cast back here
        return f"{r[col]:.2f} [{r[col + '_lo']:.2f}–{r[col + '_hi']:.2f}]"

    body = "".join(
        f"| {i} | {int(r.tokens):,} | {int(r.types):,} | "
        f"{r['b (1e2-1e4)']:.3f} [{r.b_lo:.3f}–{r.b_hi:.3f}] | "
        f"{int(r['D(1e4)']):,} | {band(r, 'alpha1')} | {band(r, 'alpha2')} | "
        f"{int(r.R_star):,} [{int(r.R_star_lo):,}–{int(r.R_star_hi):,}] |\n"
        for i, r in table.iterrows())
    note = (f"\nExponents are *central value* [*fit-window band*], the convention of "
            f"Appendix B. Heaps exponent $b$ of $D(t)\\propto t^b$ over "
            f"$t\\in[10^2,10^4]$, from curves averaged over "
            f"{int(learners['n_shuffles'])} random orderings of the texts (books, "
            f"for Gutenberg), band over window ends moved by $\\pm0.5$ decades. Zipf "
            f"exponents: two-regime OLS, head $R\\in[1,100]$, tail $R\\ge300$ with "
            f"count $\\ge2$, bands over the same windows halved and doubled. $R^*$: "
            f"crossover of a two-slope power law with a free breakpoint "
            f"(`src/estimate_d0.py`), band over the lower end of the fit range.\n\n"
            f"These fit windows are two decades below those of Table S2, because they "
            f"have to fit inside the learner corpus and the same windows are then "
            f"applied to every group. The Gutenberg $\\alpha_2$ here "
            # 1.99 is English alpha2 as Table S2 reports it; fig1a_table.py
            # computes it and asserts it against the same literal, so the two
            # cannot drift apart silently
            f"({table.loc[NAT, 'alpha2']:.2f}) is therefore *not* the Table S2 value "
            f"({1.99:.2f}) for the same corpus: it is measured over $R\\ge300$ rather "
            f"than $R\\ge10^4$, i.e. it still straddles the crossover. Only the "
            f"like-for-like comparison between the rows is meaningful.\n\n"
            f"The native reference is SPGC English at $10^8$ tokens, the corpus and "
            f"length used throughout the paper. The COREFL native control (52 k "
            f"tokens) is listed for completeness but not drawn: it is genre-matched "
            f"to the learners yet two orders of magnitude too small to serve as the "
            f"reference. Its $R^*$ nonetheless lands within a factor "
            f"{table.loc[LEA, 'R_star'] / table.loc[CTL, 'R_star']:.1f} of the "
            f"learners', which is what the length control below makes precise: at "
            f"equal, very short length the two groups look alike, and the separation "
            f"the figure shows is between the learner corpus and mature native "
            f"usage.\n")
    md = head_md + body + note
    with open(P.table_path("SI4_exponents.md"), "w") as fh:
        fh.write(md)
    return md


def draw(groups, table):
    """Draw the two panels and return the figure."""
    drawn = [g for g in groups if g[4]]
    LW = P.LW["curve"]
    fig, (axA, axB) = plt.subplots(1, 2, figsize=P.figsize(aspect=0.70, n_cols=2))

    for name, key, freq, g, _ in drawn:
        c = P.GROUP_COLORS[key]
        t = g["heaps_t"]
        axA.plot(t, g["heaps_mean"], "-", color=c, lw=LW, zorder=3)
        axA.fill_between(t, g["heaps_lo"], g["heaps_hi"], color=c, alpha=0.25, lw=0)

    axA.set(xscale="log", yscale="log", xlabel=r"corpus length $t$",
            ylabel=r"vocabulary size $D(t)$")
    axA.set_xlim(1e2, 1.5e8)
    axA.set_ylim(20, 6e5)

    for name, key, freq, g, _ in drawn:
        c = P.GROUP_COLORS[key]
        R, f = logbin_rank_curve(freq, n_bins=26, min_count=2)
        axB.plot(R, R * f / freq.sum(), "-", color=c, lw=LW)

    axB.set(xscale="log", yscale="log", xlabel=r"rank $R$", ylabel=r"$R\,p(R)$")
    axB.set_xlim(0.8, 5e5)

    # the R* labels go in last, stacked in the free lower-left corner as in
    # main-text Figure 4C, which draws the same two curves: hung on their own
    # verticals they overprint each other and the data both
    P.mark_scales(axB, [(table.loc[name, "R_star"],
                         rf"$R^*={table.loc[name, 'R_star']:,.0f}$",
                         P.GROUP_COLORS[key])
                        for name, key, _, _, _ in drawn],
                  corner="lower left", fontsize=P.FONT["legend"])

    handles = [Line2D([0], [0], color=P.GROUP_COLORS[k], lw=LW * 1.4, label=n)
               for n, k, _, _, _ in drawn]
    fig.legend(handles=handles, loc="outside upper center", ncol=2,
               frameon=False, handlelength=1.6, columnspacing=1.4)

    for ax, letter in zip((axA, axB), "AB"):
        P.panel_label(ax, letter)

    return fig


def main() -> int:
    P.setup_style()
    groups, learners = load()
    print(composition(groups).to_string(index=False))
    table = exponents(groups)
    print()
    matched_length_check(table)
    write_table(table, learners)
    fig = draw(groups, table)
    for path in P.save_figure(fig, "SI4"):
        print("wrote", os.path.relpath(path, REPO))
    return 0


if __name__ == "__main__":
    sys.exit(main())
