"""Figures for v3 (frozen) against the two coders, Coder 1 and GPT, on the 200
hand-coded tweets. No other scorer versions are shown.

Inputs (apsi folder): li_validation_scores.csv, li_v3.margins.npz.
Writes PNG + PDF to figures/liberal_illiberal_v3/.

  v3_1_scatter        v3 vs coders' mean, vs C1, vs GPT; C1 vs GPT as the coder benchmark
  v3_2_agreement      the same four pairings: ICC, Spearman, direction kappa, MAE with 95% CIs
  v3_3_distributions  score distributions of C1, GPT and v3 on the same tweets
  v3_4_relevance      share marked relevant, and pairwise kappa on relevance
  v3_5_pairs          per pair: one-sidedness and whether it separates the coders' scores
  v3_6_levels         v3 V-Party level by the coders' score band

All in-sample: v3's design choices were made on these tweets.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classifier import alternativeLib_v3 as L

OUT = "figures/liberal_illiberal_v3"
os.makedirs(OUT, exist_ok=True)

# Okabe-Ito, colour-blind safe; one colour per rater throughout
C = {"v3": "#0072B2", "C1": "#E69F00", "GPT": "#009E73", "mean": "#555555"}
FAIL = "#D55E00"
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e4e4e4"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "savefig.dpi": 300,
})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- data
g = pd.read_csv("li_validation_scores.csv")
z = np.load("li_v3.margins.npz"); M = z["margins"]
S = L.LiberalIlliberalScorer(margin_fn=lambda p, h: None)
assert list(z["hypotheses"]) == S.hypotheses
res = [S.score_from_margins(m, t) for m, t in zip(M, z["texts"])]   # text needed for the length gate
g["v3"] = [r["score"] if r["is_relevant"] else np.nan for r in res]
g["v3_rel"] = [int(r["is_relevant"]) for r in res]
g["v3_level"] = [r.get("vparty_level", np.nan) for r in res]
g["C1"] = g.score_c1 / 10.0
g["GPT"] = g.score_gpt / 10.0
g["mean"] = g.gold
both = (g.rel_both == 1).to_numpy()
rng = np.random.default_rng(0)


def icc21(x, y):
    X = np.c_[x, y]; n, k = X.shape; gm = X.mean()
    msr = k * ((X.mean(1) - gm) ** 2).sum() / (n - 1); msc = n * ((X.mean(0) - gm) ** 2).sum() / (k - 1)
    mse = ((X - X.mean(1, keepdims=True) - X.mean(0, keepdims=True) + gm) ** 2).sum() / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)


def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    po = (a == b).mean(); pe = sum((a == k).mean() * (b == k).mean() for k in (0, 1)); return (po - pe) / (1 - pe)


METRICS = {"ICC(2,1)": icc21, "Spearman ρ": lambda x, y: stats.spearmanr(x, y)[0],
           "Direction κ (≥ 5 liberal)": lambda x, y: kappa(x >= 5, y >= 5),
           "Mean abs. error": lambda x, y: np.abs(x - y).mean()}
# (y rater, x rater, label). The scored set is always: both coders relevant + both raters scored.
PAIRINGS = [("v3", "mean", "v3 vs coders' mean"), ("v3", "C1", "v3 vs Coder 1"),
            ("v3", "GPT", "v3 vs GPT coder"), ("C1", "GPT", "Coder 1 vs GPT coder (benchmark)")]


def paired(a, b):
    m = both & np.isfinite(g[a].to_numpy()) & np.isfinite(g[b].to_numpy())
    return g.loc[m, a].to_numpy(float), g.loc[m, b].to_numpy(float)


# ---------------------------------------------------------------- 1 scatter
fig, axes = plt.subplots(2, 2, figsize=(7.4, 7.2), sharex=True, sharey=True)
for ax, (ya, xb, title) in zip(axes.ravel(), PAIRINGS):
    y, x = paired(ya, xb)
    jx, jy = rng.uniform(-0.08, 0.08, len(x)), rng.uniform(-0.08, 0.08, len(y))
    bad = (x < 3) & (y >= 5) if ya == "v3" else np.zeros(len(x), bool)
    ax.plot([0, 10], [0, 10], color=MUTED, lw=1, ls="--", zorder=1)
    ax.axhline(5, color=GRID, lw=1, zorder=0); ax.axvline(5, color=GRID, lw=1, zorder=0)
    col = C[ya] if ya == "v3" else C["mean"]
    ax.scatter(x[~bad] + jx[~bad], y[~bad] + jy[~bad], s=16, color=col, alpha=0.75,
               edgecolor="white", linewidth=0.6, zorder=2)
    if bad.any():
        ax.scatter(x[bad] + jx[bad], y[bad] + jy[bad], s=24, color=FAIL, edgecolor="white", linewidth=0.6,
                   zorder=3, label=f"coder(s) < 3, v3 ≥ 5  ({bad.sum()})")
        ax.legend(loc="upper left", fontsize=7.5, handletextpad=0.2)
    ax.set_title(title, loc="left", color=INK, fontweight="bold", fontsize=9)
    ax.text(0.97, 0.04, f"ICC {icc21(x, y):.2f}   ρ {stats.spearmanr(x, y)[0]:.2f}   n = {len(x)}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=INK)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_aspect("equal")
    ax.set_xticks([0, 2.5, 5, 7.5, 10]); ax.set_yticks([0, 2.5, 5, 7.5, 10])
    ax.set_xlabel({"mean": "Coders' mean", "C1": "Coder 1", "GPT": "GPT coder"}[xb] + " (0–10)")
    ax.set_ylabel({"v3": "v3 score", "C1": "Coder 1"}[ya] + " (0–10)")
for ax in axes[0]:
    ax.set_xlabel("")
fig.suptitle("v3 and the coders, tweets both coders marked relevant", x=0.01, ha="left", fontsize=11, color=INK)
fig.tight_layout()
save(fig, "v3_1_scatter")

# ---------------------------------------------------------------- 2 agreement
B = 2000
est = {}
for ya, xb, title in PAIRINGS:
    y, x = paired(ya, xb)
    boot = np.array([[f(y[b], x[b]) for f in METRICS.values()]
                     for b in (rng.integers(0, len(x), len(x)) for _ in range(B))])
    est[title] = ([f(y, x) for f in METRICS.values()], np.percentile(boot, [2.5, 97.5], axis=0))
fig, axes = plt.subplots(1, 4, figsize=(9.6, 2.5), sharey=True)
labels = [t for _, _, t in PAIRINGS]
colors = [C["v3"], C["C1"], C["GPT"], C["mean"]]
for j, (ax, name) in enumerate(zip(axes, METRICS)):
    bench = est[labels[-1]][0][j]
    ax.axvline(bench, color=C["mean"], lw=1, ls=":", zorder=0)
    for i, (lab, col) in enumerate(zip(labels, colors)):
        e, lo, hi = est[lab][0][j], est[lab][1][0][j], est[lab][1][1][j]
        ax.plot([lo, hi], [i, i], color=col, lw=2, solid_capstyle="round")
        ax.scatter(e, i, s=40, color=col, edgecolor="white", linewidth=1.2, zorder=3,
                   marker="D" if i == len(labels) - 1 else "o")
        ax.text(e, i + 0.3, f"{e:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_title(name + ("\n(lower is better)" if "error" in name else ""), loc="left", fontsize=9, color=INK)
    ax.grid(axis="y", visible=False)
axes[0].set_yticks(range(len(labels))); axes[0].set_yticklabels(labels); axes[0].set_ylim(-0.6, len(labels) - 0.1)
axes[0].invert_yaxis()
fig.suptitle("Agreement on the score, bootstrap 95% CIs. Dotted line = Coder 1 vs GPT coder", x=0.01, ha="left",
             fontsize=11, color=INK, y=1.08)
save(fig, "v3_2_agreement")

# ---------------------------------------------------------------- 3 distributions
m = both & np.isfinite(g.v3.to_numpy())
bins = np.arange(0, 10.5, 0.5)
fig, axes = plt.subplots(3, 1, figsize=(6.4, 4.6), sharex=True, sharey=True)
for ax, (r, name) in zip(axes, [("C1", "Coder 1"), ("GPT", "GPT coder"), ("v3", "v3")]):
    v = g.loc[m, r].to_numpy(float)
    ax.hist(v, bins=bins, color=C[r], edgecolor="white", linewidth=1.5)
    ax.axvline(np.median(v), color=INK, lw=1)
    ax.text(0.01, 0.92, f"{name}   median {np.median(v):.1f}   sd {v.std(ddof=1):.2f}   "
                        f"share < 5: {(v < 5).mean():.0%}", transform=ax.transAxes, va="top", fontsize=8,
            color=INK)
    ax.grid(axis="x", visible=False); ax.set_ylabel("tweets")
axes[-1].set_xlabel("Score (0 illiberal – 10 liberal); vertical line = median")
axes[-1].set_xticks(range(0, 11))
fig.suptitle(f"Score distributions on the same {m.sum()} tweets", x=0.01, ha="left", fontsize=11, color=INK)
fig.tight_layout()
save(fig, "v3_3_distributions")

# ---------------------------------------------------------------- 4 relevance
R = {"Coder 1": g.rel_c1.to_numpy(), "GPT coder": g.rel_gpt.to_numpy(), "v3": g.v3_rel.to_numpy()}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 2.6), gridspec_kw={"width_ratios": [1, 1.1]})
names = list(R)
cols = [C["C1"], C["GPT"], C["v3"]]
for i, (n, col) in enumerate(zip(names, cols)):
    a1.barh(i, R[n].mean() * 100, color=col, height=0.6)
    a1.text(R[n].mean() * 100 + 1.5, i, f"{R[n].mean():.0%}", va="center", fontsize=8, color=INK)
a1.set_yticks(range(3)); a1.set_yticklabels(names); a1.invert_yaxis(); a1.set_xlim(0, 108)
a1.set_xlabel("Tweets marked relevant (%, n = 200)"); a1.grid(axis="y", visible=False)
a1.set_title("A. How much each rater lets in", loc="left", color=INK)
pairs_k = [("Coder 1", "GPT coder"), ("v3", "Coder 1"), ("v3", "GPT coder")]
for i, (a, b) in enumerate(pairs_k):
    k = kappa(R[a], R[b])
    boot = [kappa(R[a][s], R[b][s]) for s in (rng.integers(0, 200, 200) for _ in range(B))]
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    col = C["mean"] if i == 0 else C["v3"]
    a2.plot([lo, hi], [i, i], color=col, lw=2, solid_capstyle="round")
    a2.scatter(k, i, s=40, color=col, edgecolor="white", lw=1.2, zorder=3, marker="D" if i == 0 else "o")
    a2.text(k, i + 0.3, f"{k:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK)
a2.axvline(0, color=MUTED, lw=0.8)
a2.set_yticks(range(3)); a2.set_yticklabels([f"{a} vs {b}" for a, b in pairs_k]); a2.invert_yaxis()
a2.set_ylim(2.6, -0.5); a2.set_xlabel("Cohen's κ on relevance, 95% CI"); a2.grid(axis="y", visible=False)
a2.set_title("B. Agreement on what is relevant", loc="left", color=INK)
fig.suptitle("Relevance: these tweets were pre-selected on democracy, so v3's gate passes almost all",
             x=0.01, ha="left", fontsize=10, color=INK, y=1.04)
fig.tight_layout()
save(fig, "v3_4_relevance")

# ---------------------------------------------------------------- 5 pairs
gold = g.gold.to_numpy()
nt = S.n_topic; lib, ill = M[:, nt + 1::2], M[:, nt + 2::2]
act = np.fmax(lib, ill) >= S.active_margin; CON = (lib - ill) / 2
rows = []
for p, (d, comp, w, _, _) in enumerate(L.BATTERY):
    a = act[:, p]; ml, mi = a & both & (CON[:, p] > 0), a & both & (CON[:, p] < 0)
    rows.append(dict(pair=d, comp=comp, dropped=d in L.DROP_DOMAINS,
                     lib_share=(CON[a, p] > 0).mean() if a.any() else np.nan,
                     gold_lib=gold[ml].mean() if ml.any() else np.nan,
                     gold_ill=gold[mi].mean() if mi.any() else np.nan, n_lib=int(ml.sum()), n_ill=int(mi.sum())))
P = pd.DataFrame(rows)
y = np.arange(len(P))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 5.8), sharey=True, gridspec_kw={"width_ratios": [1, 1.25]})
a1.barh(y, P.lib_share * 100, color=[FAIL if d else C["v3"] for d in P.dropped], height=0.62)
a1.axvline(95, color=INK, lw=1, ls="--")
a1.text(95, -0.9, "drop rule: > 95%", ha="right", va="bottom", fontsize=7.5, color=INK)
for yi, r in zip(y, P.itertuples()):
    a1.text(r.lib_share * 100 + 1, yi, f"{r.lib_share:.0%}", va="center", fontsize=7, color=MUTED,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
a1.set_xlim(0, 112); a1.set_xlabel("Active cases decided on the liberal side (%)")
a1.set_title("A. One-sidedness (red = dropped from the score)", loc="left", color=INK, pad=22)
a1.set_yticks(y); a1.set_yticklabels([f"{r.pair}  ·  {r.comp}" for r in P.itertuples()], fontsize=8)
a1.invert_yaxis(); a1.grid(axis="y", visible=False)
for yi, r in zip(y, P.itertuples()):
    if np.isfinite(r.gold_lib) and np.isfinite(r.gold_ill):
        a2.plot([r.gold_ill, r.gold_lib], [yi, yi], color=GRID, lw=2.5, zorder=1)
    if np.isfinite(r.gold_ill):
        a2.scatter(r.gold_ill, yi, s=12 + 2.2 * r.n_ill, color=FAIL, edgecolor="white", lw=0.8, zorder=2)
    if np.isfinite(r.gold_lib):
        a2.scatter(r.gold_lib, yi, s=12 + 2.2 * r.n_lib, color=C["v3"], edgecolor="white", lw=0.8, zorder=3)
    a2.text(10.25, yi, f"{r.n_ill} / {r.n_lib}", va="center", fontsize=7, color=MUTED)
a2.text(10.25, -0.9, "n ill / lib", fontsize=7, color=MUTED, va="bottom")
a2.scatter([], [], s=30, color=FAIL, label="pair says illiberal")
a2.scatter([], [], s=30, color=C["v3"], label="pair says liberal")
a2.legend(loc="lower left", bbox_to_anchor=(0, 1.04), ncol=2, fontsize=7.5, handletextpad=0.2)
a2.set_xlim(0, 10); a2.set_xlabel("Coders' mean score of those tweets (0–10); dot size = n")
a2.set_title("B. Does the pair separate the coders' scores?", loc="left", color=INK, pad=22)
a2.grid(axis="y", visible=False)
fig.tight_layout()
save(fig, "v3_5_pairs")

# ---------------------------------------------------------------- 6 levels
band = pd.cut(g.gold, [0, 3, 4.5, 5.5, 7, 10.01], labels=["< 3", "3–4.5", "4.5–5.5", "5.5–7", "≥ 7"], right=False)
T = pd.crosstab(band, g.v3_level).reindex(columns=range(5), fill_value=0)
share = T.div(T.sum(1), axis=0)
fig, ax = plt.subplots(figsize=(6.2, 3.0))
im = ax.imshow(share.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
for i in range(T.shape[0]):
    for j in range(T.shape[1]):
        if T.values[i, j]:
            ax.text(j, i, f"{T.values[i, j]}", ha="center", va="center", fontsize=8,
                    color="white" if share.values[i, j] > 0.55 else INK)
ax.set_xticks(range(5))
ax.set_xticklabels(["0\nNot at all", "1\nNot", "2\nWeakly", "3\nCommitted", "4\nFully"], fontsize=7.5)
ax.set_xlabel("v3 V-Party level (uncalibrated): … committed")
ax.set_yticks(range(len(T))); ax.set_yticklabels(T.index); ax.set_ylabel("Coders' mean score"); ax.grid(False)
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("share of row", fontsize=8)
cb.outline.set_visible(False)
ax.set_title("v3 level by the coders' score; counts in cells", loc="left", fontsize=9, color=INK)
save(fig, "v3_6_levels")

# ---------------------------------------------------------------- 7 all raters in one scatter
# x = the coders' mean, y = each rater. The two coders' own dots are partly
# circular (each is half of x), so their spread around the diagonal is the
# coder disagreement, and v3's is the comparison of interest.
m = both & np.isfinite(g.v3.to_numpy())
x = g.loc[m, "mean"].to_numpy(float)
fig, ax = plt.subplots(figsize=(6.0, 5.6))
ax.plot([0, 10], [0, 10], color=MUTED, lw=1, ls="--", zorder=1)
ax.axhline(5, color=GRID, lw=1, zorder=0); ax.axvline(5, color=GRID, lw=1, zorder=0)
for r, name, mk, sz in [("C1", "Coder 1", "^", 26), ("GPT", "GPT coder", "s", 20), ("v3", "v3", "o", 24)]:
    yv = g.loc[m, r].to_numpy(float)
    ax.scatter(x + rng.uniform(-0.07, 0.07, len(x)), yv, s=sz, marker=mk, color=C[r], alpha=0.8,
               edgecolor="white", linewidth=0.6, zorder=3 if r == "v3" else 2,
               label=f"{name}   ICC {icc21(x, yv):.2f}   ρ {stats.spearmanr(x, yv)[0]:.2f}" if r == "v3" else name)
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_aspect("equal")
ax.set_xticks([0, 2.5, 5, 7.5, 10]); ax.set_yticks([0, 2.5, 5, 7.5, 10])
ax.set_xlabel("Coders' mean score (0–10)"); ax.set_ylabel("Score given by each rater (0–10)")
ax.legend(loc="upper left", fontsize=8, handletextpad=0.3, borderaxespad=0.2)
ax.set_title(f"v3, Coder 1 and the GPT coder on the same {m.sum()} tweets\n"
             "agreement with the coders' mean (circular for the coders, who make up x)",
             loc="left", fontsize=9, color=INK)
save(fig, "v3_7_scatter_all")

# ---------------------------------------------------------------- 8 each rater vs the other two
# Leave-one-out: each rater is compared with the mean of the OTHER two, so no
# rater is part of its own reference and all three numbers can be reported side
# by side. If v3 were as good as a coder, its numbers would match theirs.
m = both & np.isfinite(g.v3.to_numpy())
R3 = {r: g.loc[m, r].to_numpy(float) for r in ("C1", "GPT", "v3")}
fig, ax = plt.subplots(figsize=(6.0, 5.6))
ax.plot([0, 10], [0, 10], color=MUTED, lw=1, ls="--", zorder=1)
ax.axhline(5, color=GRID, lw=1, zorder=0); ax.axvline(5, color=GRID, lw=1, zorder=0)
loo = {}
for r, name, mk, sz in [("C1", "Coder 1", "^", 26), ("GPT", "GPT coder", "s", 20), ("v3", "v3", "o", 24)]:
    yv = R3[r]; xv = np.mean([R3[o] for o in R3 if o != r], axis=0)
    boot = [icc21(xv[b], yv[b]) for b in (rng.integers(0, len(yv), len(yv)) for _ in range(B))]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    loo[r] = (icc21(xv, yv), lo, hi, stats.spearmanr(xv, yv)[0])
    ax.scatter(xv + rng.uniform(-0.07, 0.07, len(xv)), yv, s=sz, marker=mk, color=C[r], alpha=0.8,
               edgecolor="white", linewidth=0.6, zorder=3 if r == "v3" else 2,
               label=f"{name}   ICC {loo[r][0]:.2f} [{lo:.2f}, {hi:.2f}]   ρ {loo[r][3]:.2f}")
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_aspect("equal")
ax.set_xticks([0, 2.5, 5, 7.5, 10]); ax.set_yticks([0, 2.5, 5, 7.5, 10])
ax.set_xlabel("Mean of the other two raters (0–10)"); ax.set_ylabel("Score given by the rater (0–10)")
ax.legend(loc="upper left", fontsize=8, handletextpad=0.3, borderaxespad=0.2)
ax.set_title(f"Each rater against the other two, same {m.sum()} tweets\n"
             "ICC with bootstrap 95% CI; no rater is part of its own reference",
             loc="left", fontsize=9, color=INK)
save(fig, "v3_8_scatter_vs_others")
print("leave-one-out ICC [CI], rho:", {k: tuple(round(v, 3) for v in val) for k, val in loo.items()})

print("wrote", sorted(f for f in os.listdir(OUT) if f.endswith(".png")))
