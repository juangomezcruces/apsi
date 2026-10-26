"""End-to-end phase-2 test on a synthetic corpus with known ground truth.

Reproduces the pathology found on the real PopBERT sample: descriptive
policy-delivery sentences ("we are extending unemployment benefit I") read as
far left, and one party uses that register far more than the others, which
inverted its rank. The stance gate must remove it.
"""
import os
import numpy as np, pandas as pd, importlib.util, sys, os

spec = importlib.util.spec_from_file_location("lr", os.environ.get("SCORER", "classify_economic.py"))
lr = importlib.util.module_from_spec(spec); sys.modules["lr"] = lr
spec.loader.exec_module(lr)

rng = np.random.default_rng(7)
P, NT, NS = len(lr.BATTERY), len(lr.TOPIC_HYPOTHESES), len(lr.STANCE_PAIRS)
# Linke and SPD are close on the true scale (as they are on CHES), and SPD uses
# the policy-delivery register far more -- exactly the configuration that
# inverted their ranks on the real sample.
PARTIES = {"Linke": -0.60, "SPD": -0.42, "FDP": 0.45, "AfD": 0.75}
DESC_SHARE = {"Linke": 0.15, "SPD": 0.70, "FDP": 0.20, "AfD": 0.20}
DESC_LEFT_BIAS = -0.45                       # descriptive text reads too left
N = 1600

ids, groups, texts, truth, is_desc, labels = [], [], [], [], [], []
batt = np.full((N, 2 * P), -5.0, dtype=np.float32)
topic = np.full((N, NT), -4.0, dtype=np.float32)
stance = np.zeros((N, 2 * NS), dtype=np.float32)

for i in range(N):
    party = list(PARTIES)[i % len(PARTIES)]
    q = PARTIES[party]
    desc = rng.random() < DESC_SHARE[party]
    q_eff = q + DESC_LEFT_BIAS if desc else q
    # descriptive -> normative margins below descriptive ones, and vice versa
    lvl = rng.normal(-3.5 if desc else 3.5, 1.5, NS)
    stance[i, 0::2], stance[i, 1::2] = 3.0 - lvl, 3.0 + lvl
    n_act = int(rng.integers(1, 13))
    n_tok = int(10 + 8 * n_act + rng.normal(0, 5))
    if rng.random() > 0.10:
        topic[i, rng.choice(NT, 3, replace=False)] = rng.normal(4, 1, 3)
    for p in rng.choice(P, n_act, replace=False):
        c = q_eff * 4.0 + rng.normal(0, 2.0)
        batt[i, 2 * p], batt[i, 2 * p + 1] = 3.0 - c, 3.0 + c
    ids.append(f"t{i:04d}"); groups.append(party); truth.append(q); is_desc.append(desc)
    labels.append(int(np.clip(round(2 - 2.5 * q + rng.normal(0, .6)), 0, 4)))  # 0..4, higher=left
    texts.append(" ".join(["wort"] * max(n_tok, 3)))

df = pd.DataFrame({"id": ids, "party": groups, "text_en": texts, "true_q": truth,
                   "descriptive": is_desc, "left_label": labels})
path = "test.margins.npz"
lr.save_margins(path, ids, topic, batt,
                {"battery_hash": lr.battery_hash(), "stance_hash": lr.stance_hash(),
                 "text_col": "text_en", "model": "synthetic", "n_pairs": P},
                stance=stance)

true_order = [p for p, _ in sorted(PARTIES.items(), key=lambda kv: kv[1])]
fails = []
def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not cond: fails.append(name)

res = {}
for mode in ("off", "gate", "shrink"):
    out, agg = lr.rescore(path, df=df, text_col="text_en", id_col="id",
                          group_col="party", stance_mode=mode, bootstrap=0,
                          run_diagnostics=False)
    s = out[out["lr_score"].notna()]
    gm = s.groupby("party")["lr_score"].mean()
    res[mode] = (s, gm)
    print(f"\nstance_mode={mode:<7} n={len(s):4d}  order={list(gm.sort_values().index)}")
    print("   " + "  ".join(f"{p}={gm[p]:.2f}" for p in true_order if p in gm))

print("\nstance gate:")
check("without the gate, SPD's rank is wrong",
      list(res['off'][1].sort_values().index) != true_order,
      f"got {list(res['off'][1].sort_values().index)}")
check("the gate restores the true order",
      list(res['gate'][1].sort_values().index) == true_order)
check("shrink also restores it",
      list(res['shrink'][1].sort_values().index) == true_order)
s_off, s_on = res['off'][0], res['gate'][0]
gated = lr.rescore(path, df=df, text_col="text_en", id_col="id",
                   stance_mode="gate", bootstrap=0, run_diagnostics=False)[0]
purity = gated.query("lr_status == 'descriptive'")["descriptive"].mean()
recall = gated.query("descriptive")["lr_status"].eq("descriptive").mean()
check("rows the gate drops really are descriptive", purity > 0.95, f"purity {purity:.1%}")
check("it catches most descriptive rows", recall > 0.80, f"recall {recall:.1%}")
check("the gate shrinks the spurious left tail",
      (s_on["lr_score"] < 1.43).mean() < (s_off["lr_score"] < 1.43).mean(),
      f"{(s_off['lr_score']<1.43).mean():.1%} -> {(s_on['lr_score']<1.43).mean():.1%}")
check("recovery of the latent position improves",
      abs(lr._pearson(s_on["lr_score"], s_on["true_q"]))
      > abs(lr._pearson(s_off["lr_score"], s_off["true_q"])),
      f"r {lr._pearson(s_off['lr_score'], s_off['true_q']):+.3f} -> "
      f"{lr._pearson(s_on['lr_score'], s_on['true_q']):+.3f}")

print("\nother guards:")
s = res['gate'][0]
check("no row scored on fewer than min_active pairs", s["lr_n_active"].min() >= 2)
check("thin_evidence rows are recorded",
      (res['gate'][0].shape[0] < N) and "lr_stance" in s.columns)
ntok = s["text_en"].str.split().str.len()
check("no length confound", abs(lr._pearson((s["lr_score"] - 5).abs(), ntok)) < 0.15,
      f"r={lr._pearson((s['lr_score']-5).abs(), ntok):+.3f}")

print("\nfull diagnostics with label_col + anchor:")
lr.rescore(path, df=df, text_col="text_en", id_col="id", group_col="party",
           label_col="left_label", anchor={k: 5 + 5 * v for k, v in PARTIES.items()},
           output_csv="test_scores.csv", run_diagnostics=True)
check("runmeta written", os.path.exists("test_scores.runmeta.json"))
lr.compare_settings(path)

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)
