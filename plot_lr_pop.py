"""Validation figures for the other two dimensions, from the final audited runs.

ECONOMIC LEFT-RIGHT   classify_economic.py, battery 90a23fbedce3 (veto on)
POPULISM-PLURALISM    classify_populism.py, battery 893cac4745d1 (split halves)
Both on the PopBERT sample of 1,759 Bundestag sentences, translated to English.

Inputs (validation_data/, copied from the Kaggle outputs):
    popbert_lr_90a23fbedce3.csv   lr scores, all 1,759 sentences
    econ_gold.csv                 256 hand-coded sentences: Economic (0/1), w
    popbert_pop_893cac4745d1.csv  anti-elitism half (_ae), people-centrism half (_pc), pop_score
    auth_gold_v2.csv              662 recoded sentences: AntiElitism, PeopleCentrism, OnTopic, w
Writes PNG + PDF to figures/left_right/ and figures/populism/.

Two kinds of check, never mixed. GATE checks use the hand-coded "is it about
this dimension" columns (non-directional). SCORE checks use PopBERT's own
directional annotator labels, or CHES for party placement.
The coded samples are stratified, so every gold-based figure is weighted by w.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

D = "validation_data"
OUT_LR, OUT_POP = "figures/left_right", "figures/populism"
for d in (OUT_LR, OUT_POP):
    os.makedirs(d, exist_ok=True)

# Okabe-Ito, colour-blind safe
LEFT, RIGHT = "#D55E00", "#0072B2"
AE, PC, POP = "#CC79A7", "#009E73", "#0072B2"
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e4e4e4"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "savefig.dpi": 300,
})
rng = np.random.default_rng(0)
B = 2000

# Approximate CHES 2019 lrecon, as in classify_economic.py (CHES_LRECON_DE). The
# script itself says to replace these with exact figures before reporting.
CHES = {"DIE LINKE.": 1.2, "Grüne": 3.5, "SPD": 3.7, "CDU/CSU": 6.5, "AfD": 7.2, "FDP": 8.3}
PARTY_ORDER = list(CHES)


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


def wauc(score, y, w):
    score, y, w = np.asarray(score, float), np.asarray(y).astype(bool), np.asarray(w, float)
    ok = np.isfinite(score); score, y, w = score[ok], y[ok], w[ok]
    if y.sum() == 0 or (~y).sum() == 0:
        return np.nan
    o = np.argsort(-score)
    tp = np.r_[0, np.cumsum(w[o] * y[o])]; fp = np.r_[0, np.cumsum(w[o] * ~y[o])]
    return float(np.trapezoid(tp / tp[-1], fp / fp[-1]))


def wroc(score, y, w):
    o = np.argsort(-np.asarray(score, float)); y = np.asarray(y).astype(bool)[o]; w = np.asarray(w, float)[o]
    tp = np.r_[0, np.cumsum(w * y)]; fp = np.r_[0, np.cumsum(w * ~y)]
    return fp / fp[-1], tp / tp[-1]


def boot_ci(f, n, B=B):
    vals = [f(rng.integers(0, n, n)) for _ in range(B)]
    return np.nanpercentile(vals, [2.5, 97.5])


def mean_ci(v):
    v = np.asarray(v, float)
    lo, hi = boot_ci(lambda b: v[b].mean(), len(v))
    return v.mean(), lo, hi


# =============================================================== LEFT-RIGHT
lr = pd.read_csv(f"{D}/popbert_lr_90a23fbedce3.csv", low_memory=False)
eg = pd.read_csv(f"{D}/econ_gold.csv")
lr["score"] = lr.lr_score_en
lr["admitted"] = lr.lr_gate_reason_en == "admitted"
gj = eg.merge(lr[["id", "admitted", "score"]], on="id")

# ---- lr_1 gate: coded sample projected onto the gate's known margins
N_ADM, N_DROP = int(lr.admitted.sum()), int((~lr.admitted).sum())


def gate_stats(d):
    a, y, w = d.admitted.values, d.Economic.astype(bool).values, d.w.values
    prec = (w * (a & y)).sum() / (w * a).sum()            # share economic among admitted
    miss = (w * (~a & y)).sum() / (w * ~a).sum()          # share economic among dropped
    tp, fp, fn, tn = N_ADM * prec, N_ADM * (1 - prec), N_DROP * miss, N_DROP * (1 - miss)
    rec, spec = tp / (tp + fn), tn / (tn + fp)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, Precision=prec, Recall=rec, Specificity=spec,
                F1=2 * prec * rec / (prec + rec))


G = gate_stats(gj)
idx_a, idx_d = np.flatnonzero(gj.admitted.values), np.flatnonzero(~gj.admitted.values)
GB = [gate_stats(gj.iloc[np.r_[rng.choice(idx_a, len(idx_a)), rng.choice(idx_d, len(idx_d))]])
      for _ in range(B)]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 2.9), gridspec_kw={"width_ratios": [1, 1.15]})
cells = [[G["tp"], G["fp"]], [G["fn"], G["tn"]]]
coded = [[((gj.admitted) & (gj.Economic == 1)).sum(), ((gj.admitted) & (gj.Economic == 0)).sum()],
         [((~gj.admitted) & (gj.Economic == 1)).sum(), ((~gj.admitted) & (gj.Economic == 0)).sum()]]
good = np.array([[1, 0], [0, 1]])
a1.imshow(good, cmap=matplotlib.colors.ListedColormap(["#f6e3d6", "#dce9f5"]), aspect="auto")
names = [["Correctly scored", "False positive"], ["False negative", "Correctly dropped"]]
for i in range(2):
    for j in range(2):
        a1.text(j, i - 0.12, f"{cells[i][j]:,.0f}", ha="center", va="center", fontsize=13, color=INK)
        a1.text(j, i + 0.22, f"{names[i][j]}\n{coded[i][j]} coded", ha="center", va="center",
                fontsize=7.5, color=MUTED)
a1.set_xticks([0, 1]); a1.set_xticklabels(["Economic", "Not economic"]); a1.xaxis.tick_top()
a1.set_yticks([0, 1]); a1.set_yticklabels(["Gate admits", "Gate drops"]); a1.grid(False)
for s in a1.spines.values():
    s.set_visible(False)
a1.set_title(f"A. Projected onto all {N_ADM + N_DROP:,} sentences", loc="left", color=INK, pad=22, fontsize=9)
for i, k in enumerate(["Precision", "Recall", "Specificity", "F1"]):
    lo, hi = np.percentile([b[k] for b in GB], [2.5, 97.5])
    a2.plot([lo, hi], [i, i], color=RIGHT, lw=2, solid_capstyle="round")
    a2.scatter(G[k], i, s=40, color=RIGHT, edgecolor="white", lw=1.2, zorder=3)
    a2.text(hi + 0.01, i, f"{G[k]:.3f}  [{lo:.3f}, {hi:.3f}]", va="center", fontsize=7.5, color=INK)
a2.set_yticks(range(4)); a2.set_yticklabels(["Precision", "Recall", "Specificity", "F1"]); a2.invert_yaxis()
a2.set_xlim(0.6, 1.08); a2.grid(axis="y", visible=False); a2.set_xlabel("Weighted, bootstrap 95% CI")
a2.set_title("B. Gate accuracy against hand coding", loc="left", color=INK, pad=22, fontsize=9)
fig.suptitle(f"Economic left-right: does the gate keep economic sentences? ({len(gj)} coded)",
             x=0.01, ha="left", fontsize=11, color=INK, y=1.07)
save(fig, OUT_LR, "lr_1_gate")

# ---- lr_2 score by PopBERT annotator agreement (left-wing, right-wing)
sc = lr[lr.score.notna()]
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0), sharey=True)
for ax, col, colr, lab in [(axes[0], "left_wing_n_true", LEFT, "left-wing"),
                           (axes[1], "right_wing_n_true", RIGHT, "right-wing")]:
    for k in range(6):
        v = sc.loc[sc[col] == k, "score"].to_numpy()
        if len(v) < 3:
            continue
        m_, lo, hi = mean_ci(v)
        ax.plot([k, k], [lo, hi], color=colr, lw=2, solid_capstyle="round")
        ax.scatter(k, m_, s=14 + 1.2 * np.sqrt(len(v)) * 6, color=colr, edgecolor="white", lw=1, zorder=3)
        ax.text(k, hi + 0.15, f"n={len(v)}", ha="center", fontsize=7, color=MUTED)
    r = stats.spearmanr(sc[col], sc.score)[0]
    ax.set_title(f"PopBERT annotators calling it {lab}  (ρ = {r:+.2f})", loc="left", fontsize=9, color=INK)
    ax.set_xlabel(f"Annotators (of 5) who coded the sentence {lab}"); ax.set_xticks(range(6))
    ax.axhline(5, color=MUTED, lw=0.8, ls=":")
axes[0].set_ylabel("Mean scorer score, 95% CI\n(0 left – 10 right)")
fig.suptitle(f"Economic left-right: score against the sentence-level human labels ({len(sc)} scored)",
             x=0.01, ha="left", fontsize=11, color=INK, y=1.04)
fig.tight_layout()
save(fig, OUT_LR, "lr_2_annotators")

# ---- lr_3 party placement vs CHES
fig, ax = plt.subplots(figsize=(5.4, 4.4))
LABEL_OFFSET = {"DIE LINKE.": (7, 0, "left"), "SPD": (7, -10, "left"),
                "CDU/CSU": (-7, 10, "right"), "FDP": (7, 0, "left"), "AfD": (-7, 6, "right")}
pm = {}
for p in PARTY_ORDER:
    v = sc.loc[sc.faction == p, "score"].to_numpy()
    pm[p] = mean_ci(v) + (len(v),)
rho = stats.spearmanr([CHES[p] for p in PARTY_ORDER], [pm[p][0] for p in PARTY_ORDER])[0]
for p in PARTY_ORDER:
    m_, lo, hi, n = pm[p]
    ax.plot([CHES[p], CHES[p]], [lo, hi], color=RIGHT, lw=2, solid_capstyle="round")
    ax.scatter(CHES[p], m_, s=46, color=RIGHT, edgecolor="white", lw=1.2, zorder=3)
    dx, dy, ha = LABEL_OFFSET.get(p, (7, -3, "left"))
    txt = f"{p}\n(n={n})" if p == "DIE LINKE." else f"{p}  (n={n})"
    ax.annotate(txt, (CHES[p], m_), xytext=(dx, dy), textcoords="offset points", va="center",
                fontsize=8, color=INK, ha=ha)
ax.set_xlim(0, 10); ax.set_ylim(2.5, 8.5)
ax.set_xlabel("CHES 2019 economic left-right, lrecon (approximate values)")
ax.set_ylabel("Mean scorer score, 95% CI (0 left – 10 right)")
ax.set_title(f"Party placement: Spearman ρ = {rho:+.3f} over 6 parties\n"
             "ranks are meaningful; the scorer is compressed, so absolute values are not",
             loc="left", fontsize=9, color=INK)
save(fig, OUT_LR, "lr_3_ches")

# ---- lr_4 score distributions by party
fig, ax = plt.subplots(figsize=(7.2, 3.4))
for i, p in enumerate(PARTY_ORDER):
    v = sc.loc[sc.faction == p, "score"].to_numpy()
    ax.scatter(i + rng.uniform(-0.22, 0.22, len(v)), v, s=7, color=RIGHT, alpha=0.35, lw=0)
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    ax.plot([i - 0.3, i + 0.3], [med, med], color=INK, lw=2)
    ax.plot([i, i], [q1, q3], color=INK, lw=1)
ax.axhline(5, color=MUTED, lw=0.8, ls=":")
ax.set_xticks(range(len(PARTY_ORDER)))
ax.set_xticklabels([f"{p}\nCHES {CHES[p]}" for p in PARTY_ORDER], fontsize=8)
ax.set_ylabel("Scorer score (0 left – 10 right)"); ax.set_ylim(0, 10); ax.grid(axis="x", visible=False)
ax.set_title("Sentence scores by party, ordered by CHES; bar = median, line = middle 50%",
             loc="left", fontsize=9, color=INK)
save(fig, OUT_LR, "lr_4_party_distributions")

# =============================================================== POPULISM
pp = pd.read_csv(f"{D}/popbert_pop_893cac4745d1.csv", low_memory=False)
ag = pd.read_csv(f"{D}/auth_gold_v2.csv")
pj = ag.merge(pp[["id", "lr_score_ae", "lr_score_pc", "pop_score", "lr_topic_margin_ae",
                  "lr_topic_margin_pc"]], on="id")
pj["ae_y"] = pj.anti_elitism_n_true > 0
pj["pc_y"] = pj.people_centrism_n_true > 0

# ---- pop_1 each half against each component (the discriminant check)
ROWS = [("lr_score_ae", "ae_y", None, AE, "Anti-elitism half → anti-elitism", True),
        ("lr_score_ae", "ae_y", "pc_y", AE, "   … where people-centrism is absent", False),
        ("lr_score_ae", "pc_y", None, AE, "Anti-elitism half → people-centrism", False),
        ("lr_score_pc", "pc_y", None, PC, "People-centrism half → people-centrism", True),
        ("lr_score_pc", "pc_y", "ae_y", PC, "   … where anti-elitism is absent", False),
        ("lr_score_pc", "ae_y", None, PC, "People-centrism half → anti-elitism", False),
        ("pop_score", "ae_y", None, POP, "Combined score → anti-elitism", True),
        ("pop_score", "pc_y", None, POP, "Combined score → people-centrism", True)]
fig, ax = plt.subplots(figsize=(7.6, 3.8))
for i, (sc_, y_, absent, col, lab, own) in enumerate(ROWS):
    d = pj[pj[sc_].notna()]
    if absent:
        d = d[~d[absent]]
    s_, yy, ww = d[sc_].to_numpy(), d[y_].to_numpy(), d.w.to_numpy()
    a = wauc(s_, yy, ww)
    lo, hi = boot_ci(lambda b: wauc(s_[b], yy[b], ww[b]), len(d))
    ax.plot([lo, hi], [i, i], color=col, lw=2, solid_capstyle="round", alpha=1 if own else 0.55)
    ax.scatter(a, i, s=44, color=col if own else "white", edgecolor=col, lw=1.6, zorder=3)
    ax.text(1.005, i, f"{a:.3f}  [{lo:.2f}, {hi:.2f}]  n={len(d)}", va="center", fontsize=7.5, color=INK,
            transform=ax.get_yaxis_transform())
ax.axvline(0.5, color=MUTED, lw=1, ls="--")
ax.text(0.505, len(ROWS) - 0.55, "chance", ha="left", fontsize=7.5, color=MUTED)
ax.set_yticks(range(len(ROWS))); ax.set_yticklabels([r[4] for r in ROWS], fontsize=8); ax.invert_yaxis()
ax.set_xlim(0.35, 1.0); ax.grid(axis="y", visible=False)
ax.set_xlabel("Weighted AUC against PopBERT's annotator labels, bootstrap 95% CI")
ax.set_title("Populism: each half against each component. Filled = the half's own target", loc="left",
             fontsize=9, color=INK)
save(fig, OUT_POP, "pop_1_components")

# ---- pop_2 ROC curves
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6), sharey=True)
for ax, y_, title in [(axes[0], "ae_y", "Anti-elitism label"), (axes[1], "pc_y", "People-centrism label")]:
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    for sc_, col, lab in [("lr_score_ae", AE, "anti-elitism half"), ("lr_score_pc", PC, "people-centrism half"),
                          ("pop_score", POP, "combined")]:
        d = pj[pj[sc_].notna()]
        fx, ty = wroc(d[sc_], d[y_], d.w)
        ax.plot(fx, ty, color=col, lw=2, label=f"{lab}  {wauc(d[sc_], d[y_], d.w):.2f}")
    ax.set_aspect("equal"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(title, loc="left", fontsize=9, color=INK); ax.set_xlabel("False positive rate (weighted)")
    ax.legend(loc="lower right", fontsize=7.5, title="AUC", title_fontsize=7.5)
axes[0].set_ylabel("True positive rate (weighted)")
fig.suptitle("Populism: ROC of each score against each PopBERT component", x=0.01, ha="left",
             fontsize=11, color=INK)
fig.tight_layout()
save(fig, OUT_POP, "pop_2_roc")

# ---- pop_3 party ordering
on_topic = set(ag.loc[ag.OnTopic == 1, "id"])
fig, ax = plt.subplots(figsize=(6.4, 3.4))
order = sorted(PARTY_ORDER, key=lambda p: pp.loc[(pp.faction == p) & pp.pop_score.notna(), "pop_score"].mean())
for k, (subset, off, mk, lab) in enumerate([(pp.pop_score.notna(), -0.13, "o", "all scored sentences"),
                                            (pp.pop_score.notna() & pp.id.isin(on_topic), 0.13, "D",
                                             "hand-coded on-topic only")]):
    for i, p in enumerate(order):
        v = pp.loc[subset & (pp.faction == p), "pop_score"].to_numpy()
        if len(v) < 3:
            continue
        m_, lo, hi = mean_ci(v)
        ax.plot([lo, hi], [i + off, i + off], color=POP, lw=2, alpha=1 if k == 0 else 0.6, solid_capstyle="round")
        ax.scatter(m_, i + off, s=36, marker=mk, color=POP if k == 0 else "white", edgecolor=POP, lw=1.4,
                   zorder=3, label=lab if i == 0 else None)
        ax.text(10.2, i + off, f"n={len(v)}", va="center", fontsize=7, color=MUTED)
ax.axvline(5, color=MUTED, lw=0.8, ls=":")
ax.set_yticks(range(len(order))); ax.set_yticklabels(order); ax.set_xlim(2, 10)
ax.set_xlabel("Mean combined score, 95% CI (0 pluralist – 10 populist)"); ax.grid(axis="y", visible=False)
ax.legend(loc="lower right", fontsize=7.5)
ax.set_title("Populism: party ordering", loc="left", fontsize=9, color=INK)
save(fig, OUT_POP, "pop_3_parties")

# ---- pop_4 combined score by PopBERT label group (all scored sentences)
grp = np.select([(pp.anti_elitism == 1) & (pp.people_centrism == 1), pp.anti_elitism == 1,
                 pp.people_centrism == 1], ["both", "anti-elitism only", "people-centrism only"], "neither")
pp["grp"] = grp
GROUPS = ["neither", "people-centrism only", "anti-elitism only", "both"]
fig, ax = plt.subplots(figsize=(6.8, 3.3))
for i, gname in enumerate(GROUPS):
    v = pp.loc[(pp.grp == gname) & pp.pop_score.notna(), "pop_score"].to_numpy()
    ax.scatter(i + rng.uniform(-0.22, 0.22, len(v)), v, s=8, color=POP, alpha=0.35, lw=0)
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    ax.plot([i - 0.3, i + 0.3], [med, med], color=INK, lw=2); ax.plot([i, i], [q1, q3], color=INK, lw=1)
    ax.text(i, 10.15, f"n={len(v)}", ha="center", fontsize=7.5, color=MUTED)
ax.axhline(5, color=MUTED, lw=0.8, ls=":")
ax.set_xticks(range(4)); ax.set_xticklabels(GROUPS); ax.set_ylim(0, 10.6)
ax.set_ylabel("Combined score (0 pluralist – 10 populist)"); ax.grid(axis="x", visible=False)
ax.set_title("Populism: combined score by PopBERT label group; bar = median, line = middle 50%",
             loc="left", fontsize=9, color=INK)
save(fig, OUT_POP, "pop_4_label_groups")

# ---- pop_5 gate sweep per half against the recoded non-directional columns
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0), sharey=True)
TH = np.arange(-4, 15, 1)
for ax, tm, gcol, col, lab in [(axes[0], "lr_topic_margin_ae", "AntiElitism", AE, "Anti-elitism half"),
                               (axes[1], "lr_topic_margin_pc", "PeopleCentrism", PC, "People-centrism half")]:
    y, w, m = pj[gcol].astype(bool).values, pj.w.values, pj[tm].values
    P, R, F = [], [], []
    for t in TH:
        a = m >= t
        tp, fp, fn = (w * (a & y)).sum(), (w * (a & ~y)).sum(), (w * (~a & y)).sum()
        p = tp / (tp + fp) if tp + fp else np.nan; r = tp / (tp + fn)
        P.append(p); R.append(r); F.append(2 * p * r / (p + r) if p + r else np.nan)
    ax.plot(TH, P, color=col, lw=2, label="precision")
    ax.plot(TH, R, color=col, lw=2, ls="--", label="recall")
    ax.plot(TH, F, color=INK, lw=1.2, label="F1")
    ax.axvline(0, color=MUTED, lw=0.8, ls=":"); ax.text(0.2, 0.04, "used: 0", fontsize=7.5, color=MUTED)
    ax.set_title(f"{lab} vs hand-coded {gcol}", loc="left", fontsize=9, color=INK)
    ax.set_xlabel("Gate threshold (topic margin)"); ax.set_ylim(0, 1.02)
axes[0].set_ylabel("Weighted rate"); axes[0].legend(loc="lower left", fontsize=7.5)
fig.suptitle("Populism: gate threshold sweep (in-sample)", x=0.01, ha="left", fontsize=11, color=INK, y=1.03)
fig.tight_layout()
save(fig, OUT_POP, "pop_5_gate")

print("left-right:", sorted(f for f in os.listdir(OUT_LR) if f.endswith(".png")))
print("populism:  ", sorted(f for f in os.listdir(OUT_POP) if f.endswith(".png")))
print("gate", {k: round(v, 3) for k, v in G.items()})
