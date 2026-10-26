"""Liberal-illiberal scorers v1 (pasted original), v2 (the app's) and v3 (paper
version) against the two most reliable hand-codings of the 200 democracy tweets
(Coder 1 and GPT; alpha 0.92 on the score, 0.47 on relevance).

Run where torch + transformers exist (Kaggle GPU):
    python validate_liberal_illiberal.py --c1 coder1.csv --gpt coder_gpt.csv
Paths default to the "Liberal Illiberal HandCoding" folder, else the first match
under /kaggle/input. The repo root (holding classifier/) must be importable:
it is the directory of this script, or set --repo. --versions picks which to run
(default v1,v2,v3).

Outputs (in --out, default .):
    li_validation_scores.csv   per tweet: gold, and each version's score and relevance
    li_<v>.margins.npz         v2 / v3 margin matrices (v3 now with the second wording): re-score any knob on CPU via
                               LiberalIlliberalScorer(margin_fn=...) / score_from_margins
    li_validation_report.txt   the tables below

What is compared
  RELEVANCE  kappa of each scorer's is_relevant vs C1, vs GPT, vs "both coders".
             The coders disagree with each other here (kappa 0.47; C1 is the
             broader one), so read the scorer against each, not one truth.
  POSITION   on tweets BOTH coders called relevant AND the scorer scored:
             Spearman, Pearson, ICC(2,1) and MAE against the coders' mean
             (rescaled 0-100 -> 0-10), and kappa on direction (>=5 liberal).
             Coverage = the share of those tweets the scorer scored at all.
  VS v1      paired bootstrap of the ICC and Spearman difference on the same tweets.
  FAILURES   tweets the coders put below 3 that a version places at 5 or above.
  PAIRS      per pair of the last version: activation rate and liberal share of
             active cells. A pair liberal in >95% of its cells is measuring
             "mentions the topic", not position.
  GATE SWEEP relevance kappa across topic_margin_min, IN-SAMPLE: a guide to
             where the threshold sits, not a result to report.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats


def find(explicit, pattern):
    if explicit:
        return explicit
    for root in ("Liberal Illiberal HandCoding", "/kaggle/input"):
        hits = sorted(glob.glob(os.path.join(root, "**", pattern), recursive=True))
        if hits:
            return hits[0]
    raise SystemExit(f"no file matching {pattern}; pass it explicitly")


def load_gold(c1_path, gpt_path):
    c1, gp = pd.read_csv(c1_path), pd.read_csv(gpt_path)
    norm = lambda s: " ".join(str(s).split())
    assert (c1.text.map(norm) == gp.text.map(norm)).all(), "coder files are not row-aligned"
    rel = lambda s: s.astype(str).str.strip().str.lower().map({"true": 1, "false": 0})
    sc = lambda s: pd.to_numeric(s.replace({"null": np.nan}), errors="coerce")
    g = pd.DataFrame({"text": c1.text, "rel_c1": rel(c1.relevant), "rel_gpt": rel(gp.relevant),
                      "score_c1": sc(c1.score), "score_gpt": sc(gp.score)})
    g.loc[g.rel_c1 == 0, "score_c1"] = np.nan
    g.loc[g.rel_gpt == 0, "score_gpt"] = np.nan
    g["rel_both"] = ((g.rel_c1 == 1) & (g.rel_gpt == 1)).astype(int)
    g["gold"] = g[["score_c1", "score_gpt"]].mean(axis=1, skipna=False) / 10.0
    return g


def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    po = (a == b).mean()
    pe = sum((a == k).mean() * (b == k).mean() for k in np.unique(np.r_[a, b]))
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def icc21(x, y):
    X = np.c_[x, y]; n, k = X.shape; g = X.mean()
    msr = k * ((X.mean(1) - g) ** 2).sum() / (n - 1)
    msc = n * ((X.mean(0) - g) ** 2).sum() / (k - 1)
    mse = ((X - X.mean(1, keepdims=True) - X.mean(0, keepdims=True) + g) ** 2).sum() / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)


def boot(f, *cols, B=2000, seed=0):
    rng = np.random.default_rng(seed); n = len(cols[0])
    vals = [f(*(c[idx] for c in cols)) for idx in (rng.integers(0, n, n) for _ in range(B))]
    return np.nanpercentile(vals, [2.5, 97.5])


def position_table(g, col):
    base = g[g.rel_both == 1]
    m = base[np.isfinite(base[col])]
    x, y = m[col].to_numpy(float), m.gold.to_numpy(float)
    rows = {"coverage": f"{len(m)}/{len(base)} ({len(m) / max(len(base), 1):.0%})"}
    if len(m) < 5:
        return rows
    for name, f in [("Spearman rho", lambda a, b: stats.spearmanr(a, b)[0]),
                    ("Pearson r", lambda a, b: stats.pearsonr(a, b)[0]),
                    ("ICC(2,1)", icc21),
                    ("MAE (0-10)", lambda a, b: np.abs(a - b).mean()),
                    ("direction kappa", lambda a, b: kappa(a >= 5, b >= 5))]:
        lo, hi = boot(f, x, y)
        rows[name] = f"{f(x, y):6.3f}  [{lo:.3f}, {hi:.3f}]"
    rows["mean / sd"] = f"{x.mean():.2f} / {x.std(ddof=1):.2f}  (gold {y.mean():.2f} / {y.std(ddof=1):.2f})"
    return rows


def relevance_table(g, col):
    return {f"kappa vs {who}": f"{kappa(g[col], g[ref]):6.3f}  (scorer {g[col].mean():.0%} relevant, "
                               f"{who} {g[ref].mean():.0%})"
            for who, ref in [("C1", "rel_c1"), ("GPT", "rel_gpt"), ("both", "rel_both")]}


MODULES = {"v1": "alternativeLib_v1", "v2": "alternativeLib", "v3": "alternativeLib_v3"}


def paired_vs(g, col, ref="v1_score", B=2000, seed=1):
    m = (g.rel_both == 1) & np.isfinite(g[col]) & np.isfinite(g[ref])
    x, r, y = (g.loc[m, c].to_numpy(float) for c in (col, ref, "gold"))
    rng = np.random.default_rng(seed); d = []
    for _ in range(B):
        b = rng.integers(0, len(y), len(y))
        d.append((icc21(x[b], y[b]) - icc21(r[b], y[b]),
                  stats.spearmanr(x[b], y[b])[0] - stats.spearmanr(r[b], y[b])[0]))
    lo, md, hi = np.percentile(np.array(d), [2.5, 50, 97.5], axis=0)
    return f"dICC {md[0]:+.3f} [{lo[0]:+.3f}, {hi[0]:+.3f}] | drho {md[1]:+.3f} [{lo[1]:+.3f}, {hi[1]:+.3f}]  (n={m.sum()})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c1"); ap.add_argument("--gpt")
    ap.add_argument("--repo", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--out", default=".")
    ap.add_argument("--versions", default="v1,v2,v3")
    a = ap.parse_args()
    sys.path.insert(0, a.repo)
    import importlib
    versions = a.versions.split(",")

    g = load_gold(find(a.c1, "*CODER 1*.csv"), find(a.gpt, "*GPT*.csv"))
    texts = g.text.astype(str).tolist()
    os.makedirs(a.out, exist_ok=True)

    last = None
    REL = {}
    for v in versions:
        mod = importlib.import_module("classifier." + MODULES[v])
        sc = mod.LiberalIlliberalScorer()
        if v == "v1":
            res = [sc.score_liberal_illiberal(t) for t in texts]
        else:
            M = sc.margin_matrix(texts)
            np.savez_compressed(os.path.join(a.out, f"li_{v}.margins.npz"), margins=M,
                                hypotheses=np.array(sc.hypotheses), texts=np.array(texts))
            res = [sc.score_from_margins(M[i], t) for i, t in enumerate(texts)]
            g[f"{v}_level"] = [r.get("vparty_level", np.nan) for r in res]
            g[f"{v}_conf"] = [r["confidence"] for r in res]
            for k in ("score_sd", "score_lo", "score_hi"):
                g[f"{v}_{k}"] = [r.get(k, np.nan) for r in res]
            g[f"{v}_topic_margin"] = [r["topic_margin"] for r in res]
            last = (v, sc, M)
            if getattr(sc, "_para", None) is not None:      # v3: two wordings
                REL[v] = sc.reliability(M, texts)
        g[f"{v}_rel"] = [int(r["is_relevant"]) for r in res]
        g[f"{v}_score"] = [r["score"] if r["is_relevant"] else np.nan for r in res]

    lines = [f"n = {len(g)} tweets | both coders relevant: {g.rel_both.sum()}", ""]
    bad = (g.rel_both == 1) & (g.gold < 3)
    for v in versions:
        lines.append(f"=== {v} ===")
        for k, val in {**relevance_table(g, f"{v}_rel"), **position_table(g, f"{v}_score")}.items():
            lines.append(f"  {k:<18}{val}")
        if v != "v1" and "v1" in versions:
            lines.append(f"  {'vs v1 (paired)':<18}{paired_vs(g, f'{v}_score')}")
        lines.append(f"  {'gold<3 at >=5':<18}{int((g.loc[bad, f'{v}_score'] >= 5).sum())} of {int(bad.sum())}")
        if v in REL:
            r = REL[v]
            lines.append(f"  {'two wordings':<18}pearson {r['pearson']:.3f} | spearman {r['spearman']:.3f} | "
                         f"mean |diff| {r['mean_abs_diff']:.2f} (n={r['n']}); averaged score used above")
            for w in ("main", "paraphrase"):
                g[f"{v}_{w}"] = r[w]
                lines.append(f"  {w + ' alone':<18}{position_table(g, f'{v}_{w}').get('ICC(2,1)', 'n/a')}  (ICC)")
        lines.append("")
    if last is None:
        report = "\n".join(lines); print(report); return

    v, sc, M = last
    nt = sc.n_topic
    bat = M[:, sc._bat]
    lib, ill = bat[:, 0::2], bat[:, 1::2]
    act = np.fmax(lib, ill) >= sc.active_margin
    lines.append(f"=== {v} pairs: active share | liberal share of active | mean margin L / I ===")
    for p, d in enumerate(sc.domains):
        ls = ((lib[act[:, p], p] - ill[act[:, p], p]) > 0).mean() if act[:, p].any() else np.nan
        flag = "  <- one-sided" if ls > 0.95 else ""
        lines.append(f"  {d:<22}{act[:, p].mean():5.0%} | {ls:5.0%} | {lib[:, p].mean():6.1f} / {ill[:, p].mean():6.1f}{flag}")
    md = 1 / (1 + np.exp(-M[:, nt]))
    lines.append(f"  militant probe: mean sigmoid {md.mean():.2f}, share > 0.5: {(md > 0.5).mean():.0%}")
    lines.append("")

    lines.append(f"=== {v} gate sweep (IN-SAMPLE, guide only) ===")
    tm = g[f"{v}_topic_margin"].to_numpy()
    any_active = act.any(axis=1)
    for thr in [-4, 0, 4, 8, 12, 14]:
        rel = ((tm >= thr) & any_active).astype(int)
        lines.append(f"  topic_margin_min {thr:+d}: relevant {rel.mean():4.0%} | kappa C1 {kappa(rel, g.rel_c1):.3f}"
                     f"  GPT {kappa(rel, g.rel_gpt):.3f}  both {kappa(rel, g.rel_both):.3f}")
    lines += ["", f"=== {v} level (uncalibrated) by gold band ==="]
    band = pd.cut(g.gold, [0, 3, 4.5, 5.5, 7, 10.01], labels=["<3", "3-4.5", "4.5-5.5", "5.5-7", ">=7"], right=False)
    lines.append(pd.crosstab(band, g[f"{v}_level"], dropna=True).to_string())

    report = "\n".join(lines)
    print(report)
    open(os.path.join(a.out, "li_validation_report.txt"), "w").write(report)
    g.to_csv(os.path.join(a.out, "li_validation_scores.csv"), index=False)


if __name__ == "__main__":
    main()
