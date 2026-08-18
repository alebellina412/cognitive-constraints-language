#!/usr/bin/env python3
"""Figure 4 of the main text — the model against the data, and D0 across groups.

Three panels: the 1-gram rank-frequency curve of corpus against calibrated
model with the fitted crossover of each; the degree and strength densities of
the corresponding Word Co-occurrence Networks; and the rank-frequency curve of
native against learner English, where the same crossover separates two
populations of speakers.

Some of these panels also appear in the appendix figures, which are meant to be
readable on their own; the duplication is deliberate, and the main text carries
the summary while the appendices carry the detail.

Vocabulary growth is *not* a panel here. One of the two lexical statistics is
enough to show that the calibrated model tracks the corpus, and the Heaps
mismatch (0.589 against 0.547) is stated in the body text instead, where it can
be explained in a sentence.

One colour per source, across all three panels: the SPGC English curve is the
same slate blue in panel C as in panels A and B, even though its twin in the
appendix draws it grey. Within one figure a reader must not meet the same corpus
in two colours; between figures, each carries its own legend.

Nothing here computes science: the curves come from `io_reduced`, the estimators
from `fits` and `estimate_d0`, the style from `plotting`.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.lines import Line2D                                # noqa: E402

import estimate_d0 as D0                                           # noqa: E402
import plotting as P                                               # noqa: E402
from fits import logbin_pdf, logbin_rank_curve                     # noqa: E402
from io_reduced import load_corefl, load_heaps, load_wcn, sim_freq_from_wcn  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM = "sim_calibrated"
T_TOKENS = 100_000_000

#: the learner curve keeps the orange of appendix Figure S5; the native and
#: model curves keep the two colours of the model-versus-data comparison
C_LEARN = P.GROUP_COLORS["learners"]


def load():
    """The model run, the native corpus, the learner corpus, and their R*."""
    meta = json.load(open(os.path.join(REPO, "data_reduced", f"{SIM}_meta.json")))

    sim_heaps, sim_wcn = load_heaps(SIM), load_wcn(SIM)
    dat_heaps, dat_wcn = load_heaps("en"), load_wcn("en")
    learners = load_corefl("learner_all")      # L1 Spanish + L1 German, English L2

    dat_freq = np.sort(dat_heaps["freq_matched"])[::-1].astype(np.int64)
    sim_freq = sim_freq_from_wcn(sim_wcn)
    lea_freq = np.asarray(learners["freq"])

    assert int(sim_heaps["max_tokens"]) == int(dat_heaps["match_tokens"]) == T_TOKENS
    assert meta["D0"] == 9366 and meta["p"] == 0.5

    r = {k: round(D0.estimate(f)["R_star"])
         for k, f in (("dat", dat_freq), ("sim", sim_freq), ("lea", lea_freq))}
    return meta, dat_freq, dat_wcn, sim_freq, sim_wcn, lea_freq, r


def main() -> int:
    P.setup_style()
    meta, dat_freq, dat_wcn, sim_freq, sim_wcn, lea_freq, r = load()

    LW = P.LW["curve"]
    C_DAT, C_MOD = P.MODEL_COLORS["data"], P.MODEL_COLORS["model"]

    # The drawn box is 104 x 109 pt here. Its WIDTH is not a choice: three panels
    # across 468 bp leave 156 bp each, and the y label and its tick labels take a
    # third of that at the current type size -- tightening wspace, the layout pad
    # and the tick set together buy 2.6%. Only the height is free, and 1.10 is
    # what makes the box square-ish and the same size as Figure 2's (104 x 114),
    # which carries its legends inside the panels and so loses no strip to one.
    # Bigger than this means fewer panels per row: 2 x 2 gives 184 x 144 pt but a
    # figure 398 pt tall instead of 172.
    fig, (axA, axB, axC) = plt.subplots(
        1, 3, figsize=P.figsize(aspect=1.10, n_cols=3), layout="constrained")

    # --- A: rank-frequency, model against data, both crossovers marked ------ #
    for freq, colour in ((dat_freq, C_DAT), (sim_freq, C_MOD)):
        rank, prob = P.rank_prob(freq)
        i = P.log_indices(int((freq >= 2).sum()), 260)
        axA.plot(rank[i], prob[i], "-", color=colour, lw=LW, solid_capstyle="round")
    axA.set(xscale="log", yscale="log", xlabel=r"rank $R$", ylabel=r"$p(R)$")
    axA.set_xlim(0.6, 1e6)
    axA.set_yticks([1e-8, 1e-6, 1e-4, 1e-2])
    # the two crossovers differ by 6%, so labels hung on the lines would
    # overprint each other, and at any height they would land on the data: a
    # rank-frequency curve crosses the whole panel diagonally. They go to the
    # lower-left corner instead, which is the free one, while the verticals
    # stay on their values
    P.mark_scales(axA, [(r["dat"], rf"$R^*={r['dat']:,}$", C_DAT),
                        (r["sim"], rf"$R^*={r['sim']:,}$", C_MOD)],
                  corner="lower left", fontsize=P.FONT["legend"], dy=0.105)

    # --- B: degree and strength densities ----------------------------------- #
    for wcn, colour in ((dat_wcn, C_DAT), (sim_wcn, C_MOD)):
        # a dotted line at this width is barely visible in print, but the long
        # dash first used here read as a broken solid; this is the middle
        for key, style in (("degree", "-"), ("strength", (0, (2.8, 1.25)))):
            b = logbin_pdf(wcn[key], 24)
            axB.plot(b["centre"], b["density"], color=colour, lw=LW * 0.9,
                     ls=style, dash_capstyle="round")
    axB.set(xscale="log", yscale="log", xlabel=r"quantity $q$", ylabel=r"$p(q)$")
    # the densities reach into the lower-left corner, so the floor is dropped to
    # open the strip the legend needs instead of moving the legend onto a curve
    axB.set_ylim(2e-14, 3e0)
    axB.legend(handles=[Line2D([0], [0], color="0.35", lw=P.LW["aux"], ls=s,
                               label=l, dash_capstyle="round")
                        for s, l in (("-", r"degree $k$"),
                                     ((0, (2.8, 1.25)), r"strength $s$"))],
               fontsize=P.FONT["legend"], frameon=False, loc="lower left",
               borderaxespad=0.15, labelspacing=0.3)

    # --- C: natives against learners, as R p(R) ----------------------------- #
    # a Zipfian regime is flat in R p(R), so the plateau IS the kernel lexicon
    # and its edge is the fitted R*. Normalised to probabilities: the two corpora
    # differ in size by a factor 170 and the panel compares shapes.
    for freq, colour in ((dat_freq, C_DAT), (lea_freq, C_LEARN)):
        R, f = logbin_rank_curve(freq, n_bins=26, min_count=2)
        axC.plot(R, R * f / freq.sum(), "-", color=colour, lw=LW)
    axC.set(xscale="log", yscale="log", xlabel=r"rank $R$", ylabel=r"$R\,p(R)$")
    axC.set_xlim(0.8, 5e5)
    # same as panel A: each label is a third of a 156 bp panel wide, so hung on
    # its own line either one crosses the curves. Stacked in the lower-left
    # corner they read as a key, and the numbers are the result of the panel
    P.mark_scales(axC, [(r["dat"], rf"$R^*={r['dat']:,}$", C_DAT),
                        (r["lea"], rf"$R^*={r['lea']:,}$", C_LEARN)],
                  corner="lower left", fontsize=P.FONT["legend"], dy=0.105)

    handles = [Line2D([0], [0], color=c, lw=LW * 1.4, label=l) for c, l in (
        (C_DAT, "data (SPGC English)"),
        (C_MOD, f"model ($D_0={meta['D0']:,}$)"),
        (C_LEARN, "L2 learners (COREFL)"))]
    fig.legend(handles=handles, loc="outside upper center", ncol=3,
               frameon=False, handlelength=1.6, columnspacing=1.4)

    for ax, letter in zip((axA, axB, axC), "ABC"):
        P.panel_label(ax, letter)

    print(f"R* data {r['dat']:,} | R* model {r['sim']:,} | "
          f"R* learners {r['lea']:,} | D0 calibrated {meta['D0']:,}")
    for path in P.save_figure(fig, "fig4"):
        print("wrote", os.path.relpath(path, REPO))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
