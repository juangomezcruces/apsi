"""Figures for the liberal-illiberal scorer validation (200 hand-coded tweets).

Inputs (apsi folder): li_validation_scores.csv, li_v3.margins.npz, li_fixes_cv.csv
(from cv_liberal_illiberal_fixes.py). Writes PNG + PDF to figures/liberal_illiberal/.

  fig1_scatter      scorer vs coders, v1 / v2 / v3 frozen
  fig2_metrics      agreement metrics with bootstrap 95% CIs, all versions
  fig3_pairs        per pair: one-sidedness and whether it separates the coders' scores
  fig4_fixes_cv     held-out ICC of fixes 1-4 across 100 half-splits
  fig5_levels       v3 V-Party level by the coders' score band

"v3 frozen" = v3 with the three one-sided pairs dropped (DROP_DOMAINS), rescored
from the cached margins. "v3 initial" = the Kaggle run before that drop.
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

OUT = "figures/liberal_illiberal"
os.makedirs(OUT, exist_ok=True)

# Okabe-Ito: colour-blind safe, fixed order per version
VERSION_COLOR = {"v1": "#E69F00", "v2": "#56B4E9", "v3 initial": "#999999", "v3 frozen": "#0072B2"}
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
g["v3f_score"] = [r["score"] if r["is_relevant"] else np.nan for r in res]
g["v3f_level"] = [r.get("vparty_level", np.nan) for r in res]
COLS = {"v1": "v1_score", "v2": "v2_score", "v3 initial": "v3_score", "v3 frozen": "v3f_score"}
rel = (g.rel_both == 1).to_numpy(); gold = g.gold.to_numpy()


def icc21(x, y):
    X = np.c_[x, y]; n, k = X.shape; gm = X.mean()
    msr = k * ((X.mean(1) - gm) ** 2).sum() / (n - 1); msc = n * ((X.mean(0) - gm) ** 2).sum() / (k - 1)
    mse = ((X - X.mean(1, keepdims=True) - X.mean(0, keepdims=True) + gm) ** 2).sum() / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)


def kappa(a, b):
    po = (a == b).mean(); pe = sum((a == k).mean() * (b == k).mean() for k in (0, 1)); return (po - pe) / (1 - pe)


METRICS = {"ICC(2,1)": icc21, "Spearman ρ": lambda x, y: stats.spearmanr(x, y)[0],
           "Direction κ": lambda x, y: kappa(x >= 5, y >= 5), "Mean abs. error": lambda x, y: np.abs(x - y).mean()}


def pairs(col):
    m = rel & np.isfinite(g[col].to_numpy())
    return g.loc[m, col].to_numpy(float), gold[m]


# ---------------------------------------------------------------- fig 1
fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.3), sharex=True, sharey=True)
rng = np.random.default_rng(0)
for ax, v in zip(axes, ["v1", "v2", "v3 frozen"]):
    x, y = pairs(COLS[v])
    bad = (y < 3) & (x >= 5)
    jit = rng.uniform(-0.08, 0.08, len(y))
    ax.fill_between([0, 3], 5, 10, color=FAIL, alpha=0.06, lw=0)
    ax.plot([0, 10], [0, 10], color=MUTED, lw=1, ls="--", zorder=1)
    ax.axhline(5, color=GRID, lw=1, zorder=0); ax.axvline(5, color=GRID, lw=1, zorder=0)
    ax.scatter(y[~bad] + jit[~bad], x[~bad], s=16, color=VERSION_COLOR[v], alpha=0.75,
               edgecolor="white", linewidth=0.6, zorder=2)
    ax.scatter(y[bad] + jit[bad], x[bad], s=22, color=FAIL, edgecolor="white", linewidth=0.6, zorder=3,
               label="coders < 3, scorer ≥ 5")
    ax.set_title(v, color=INK, loc="left", fontweight="bold")
    ax.text(0.97, 0.04, f"ICC {icc21(x, y):.2f}   ρ {stats.spearmanr(x, y)[0]:.2f}\n"
                        f"n = {len(x)}   failures {bad.sum()}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=INK)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_xticks([0, 2.5, 5, 7.5, 10]); ax.set_yticks([0, 2.5, 5, 7.5, 10])
    ax.set_aspect("equal")
    ax.set_xlabel("Coders' mean score (0–10)")
axes[0].set_ylabel("Scorer (0 illiberal – 10 liberal)")
axes[0].legend(loc="upper left", fontsize=8, handletextpad=0.3)
fig.suptitle("Scorer vs hand coding, tweets both coders marked relevant", x=0.01, ha="left",
             fontsize=11, color=INK)
save(fig, "fig1_scatter")

# ---------------------------------------------------------------- fig 2
B = 2000
vals = {}
for v, col in COLS.items():
    x, y = pairs(col)
    boot = np.array([[f(x[b], y[b]) for f in METRICS.values()]
                     for b in (rng.integers(0, len(y), len(y)) for _ in range(B))])
    vals[v] = ([f(x, y) for f in METRICS.values()], np.percentile(boot, [2.5, 97.5], axis=0))
fig, axes = plt.subplots(1, 4, figsize=(9.2, 2.4), sharey=True)
order = list(COLS)
for j, (ax, name) in enumerate(zip(axes, METRICS)):
    for i, v in enumerate(order):
        est, (lo, hi) = vals[v][0][j], (vals[v][1][0][j], vals[v][1][1][j])
        ax.plot([lo, hi], [i, i], color=VERSION_COLOR[v], lw=2, solid_capstyle="round")
        ax.scatter(est, i, s=40, color=VERSION_COLOR[v], edgecolor="white", linewidth=1.2, zorder=3)
        ax.text(est, i + 0.28, f"{est:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_title(name + (" (lower is better)" if "error" in name else ""), loc="left", fontsize=9, color=INK)
    ax.grid(axis="y", visible=False)
axes[0].set_yticks(range(len(order))); axes[0].set_yticklabels(order); axes[0].set_ylim(-0.6, len(order) - 0.2)
axes[0].invert_yaxis()
fig.suptitle("Agreement with the coders' mean, bootstrap 95% CIs (in-sample)", x=0.01, ha="left",
             fontsize=11, color=INK, y=1.06)
save(fig, "fig2_metrics")

# ---------------------------------------------------------------- fig 3
nt = S.n_topic; lib, ill = M[:, nt + 1::2], M[:, nt + 2::2]
act = np.fmax(lib, ill) >= S.active_margin; C = (lib - ill) / 2
rows = []
for p, (d, comp, w, _, _) in enumerate(L.BATTERY):
    a = act[:, p]
    rows.append(dict(pair=d, comp=comp, dropped=d in L.DROP_DOMAINS, active=a.mean(),
                     lib_share=(C[a, p] > 0).mean() if a.any() else np.nan,
                     gold_lib=np.nanmean(gold[a & rel & (C[:, p] > 0)]) if (a & rel & (C[:, p] > 0)).any() else np.nan,
                     gold_ill=np.nanmean(gold[a & rel & (C[:, p] < 0)]) if (a & rel & (C[:, p] < 0)).any() else np.nan,
                     n_lib=int((a & rel & (C[:, p] > 0)).sum()), n_ill=int((a & rel & (C[:, p] < 0)).sum())))
P = pd.DataFrame(rows)
P["label"] = [f"{r.pair}  ·  {r.comp}" for r in P.itertuples()]
y = np.arange(len(P))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 5.6), sharey=True, gridspec_kw={"width_ratios": [1, 1.25]})
colors = [FAIL if r.dropped else VERSION_COLOR["v3 frozen"] for r in P.itertuples()]
a1.barh(y, P.lib_share * 100, color=colors, height=0.62)
a1.axvline(95, color=INK, lw=1, ls="--")
a1.text(95, -0.9, "drop rule: > 95%", ha="right", va="bottom", fontsize=7.5, color=INK)
for yi, r in zip(y, P.itertuples()):
    a1.text(r.lib_share * 100 + 1, yi, f"{r.lib_share:.0%}", va="center", fontsize=7, color=MUTED,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
a1.set_xlim(0, 112); a1.set_xlabel("Active cases decided on the liberal side (%)")
a1.set_title("A. One-sidedness (all 200 tweets)", loc="left", color=INK, pad=22)
a1.set_yticks(y); a1.set_yticklabels(P.label, fontsize=8); a1.invert_yaxis(); a1.grid(axis="y", visible=False)
for yi, r in zip(y, P.itertuples()):
    if np.isfinite(r.gold_lib) and np.isfinite(r.gold_ill):
        a2.plot([r.gold_ill, r.gold_lib], [yi, yi], color=GRID, lw=2.5, zorder=1)
    if np.isfinite(r.gold_ill):
        a2.scatter(r.gold_ill, yi, s=12 + 2.2 * r.n_ill, color=FAIL, edgecolor="white", lw=0.8, zorder=2)
    if np.isfinite(r.gold_lib):
        a2.scatter(r.gold_lib, yi, s=12 + 2.2 * r.n_lib, color=VERSION_COLOR["v3 frozen"], edgecolor="white",
                   lw=0.8, zorder=3)
    a2.text(10.25, yi, f"{r.n_ill} / {r.n_lib}", va="center", fontsize=7, color=MUTED)
a2.text(10.25, -0.9, "n ill / lib", fontsize=7, color=MUTED, va="bottom")
a2.scatter([], [], s=30, color=FAIL, label="pair says illiberal")
a2.scatter([], [], s=30, color=VERSION_COLOR["v3 frozen"], label="pair says liberal")
a2.legend(loc="lower left", bbox_to_anchor=(0, 1.04), ncol=2, fontsize=7.5, handletextpad=0.2)
a2.set_xlim(0, 10); a2.set_xlabel("Coders' mean score of those tweets (0–10); dot size = n")
a2.set_title("B. Does the pair separate the coders' scores?", loc="left", color=INK, pad=22)
a2.grid(axis="y", visible=False)
fig.suptitle("v3 pairs. Red bars in A: one-sided pairs dropped from the score", x=0.01, ha="left",
             fontsize=11, color=INK)
fig.tight_layout()
save(fig, "fig3_pairs")

# ---------------------------------------------------------------- fig 4
cv = pd.read_csv("li_fixes_cv.csv")
ARMS = [("current v3 (fixed)", "v3 frozen (no re-tuning)"), ("3. settings re-tuned", "3. settings re-tuned"),
        ("1. parts weighted equally", "1. parts weighted equally"),
        ("2. defeat cap 4.5", "2. defeat cap at 4.5"), ("2. defeat cap 3.0", "2. defeat cap at 3.0"),
        ("4. fire at margin 2", "4. fire at margin ≥ 2"), ("4. fire at margin 4", "4. fire at margin ≥ 4"),
        ("all free", "all four tuned together")]
base = cv[cv.arm == "current v3 (fixed)"].icc_na5.mean()
fig, ax = plt.subplots(figsize=(7.2, 3.4))
for i, (arm, label) in enumerate(ARMS):
    d = cv[cv.arm == arm].icc_na5.to_numpy()
    lo, q1, med, q3, hi = np.percentile(d, [2.5, 25, 50, 75, 97.5])
    col = VERSION_COLOR["v3 frozen"] if i == 0 else MUTED
    ax.plot([lo, hi], [i, i], color=col, lw=1, alpha=0.6)
    ax.plot([q1, q3], [i, i], color=col, lw=4, solid_capstyle="round", alpha=0.9)
    ax.scatter(d.mean(), i, s=36, color="white", edgecolor=col, lw=1.6, zorder=3)
    ax.text(hi + 0.005, i, f"{d.mean():.3f}", va="center", fontsize=7.5, color=INK)
ax.axvline(base, color=VERSION_COLOR["v3 frozen"], lw=1, ls="--")
ax.set_yticks(range(len(ARMS))); ax.set_yticklabels([l for _, l in ARMS]); ax.invert_yaxis()
ax.grid(axis="y", visible=False)
ax.set_xlabel("Held-out ICC with the coders (NA counted as 5)")
ax.set_title("Fixes 1–4: 100 half-splits, chosen on one half, scored on the other\n"
             "thick = middle 50% of splits, thin = middle 95%, circle = mean", loc="left", fontsize=9, color=INK)
save(fig, "fig4_fixes_cv")

# ---------------------------------------------------------------- fig 5
band = pd.cut(g.gold, [0, 3, 4.5, 5.5, 7, 10.01], labels=["< 3", "3–4.5", "4.5–5.5", "5.5–7", "≥ 7"], right=False)
T = pd.crosstab(band, g.v3f_level).reindex(columns=range(5), fill_value=0)
share = T.div(T.sum(1), axis=0)
fig, ax = plt.subplots(figsize=(6.2, 3.0))
im = ax.imshow(share.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
for i in range(T.shape[0]):
    for j in range(T.shape[1]):
        n = T.values[i, j]
        if n:
            ax.text(j, i, f"{n}", ha="center", va="center", fontsize=8,
                    color="white" if share.values[i, j] > 0.55 else INK)
ax.set_xticks(range(5)); ax.set_xticklabels([f"{k}\n{lab.replace(' committed', '')}" for k, lab in enumerate(L.VPARTY_LABELS)], fontsize=7.5)
ax.set_xlabel("v3 level: … committed")
ax.set_yticks(range(len(T))); ax.set_yticklabels(T.index)
ax.set_ylabel("Coders' mean score"); ax.grid(False)
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("share of row", fontsize=8)
cb.outline.set_visible(False)
ax.set_title("v3 frozen: V-Party level (uncalibrated) by coders' score; counts in cells",
             loc="left", fontsize=9, color=INK)
save(fig, "fig5_levels")

print("wrote", sorted(os.listdir(OUT)))
