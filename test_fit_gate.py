"""Fitting the gate instead of hand-picking its rule.

The hand rule is "2nd-highest topic margin >= 0". Here the truth depends on two
SPECIFIC probes, so that rule is systematically wrong and a fit should beat it.
Also checks the weighted estimator: the gold set is stratified, so weighted sums
on the sample must reproduce corpus quantities.
"""
import os
import numpy as np, pandas as pd, importlib.util, sys, os
spec = importlib.util.spec_from_file_location("lr", os.environ.get("SCORER", "classify_economic.py"))
lr = importlib.util.module_from_spec(spec); sys.modules["lr"] = lr; spec.loader.exec_module(lr)

rng = np.random.default_rng(5)
N, NT, NV = 2000, len(lr.TOPIC_HYPOTHESES), len(lr.NEGATIVE_TOPIC_HYPOTHESES)
P = len(lr.BATTERY)
SIGNAL = [3, 7]                       # only these two probes carry the truth

topic = rng.normal(-4, 7, (N, NT)).astype(np.float32)
veto = rng.normal(-4, 6, (N, NV)).astype(np.float32)
lat = topic[:, SIGNAL].sum(1) + rng.normal(0, 4, N)
gold = (lat > np.percentile(lat, 60)).astype(int)          # ~40% prevalence
# the noise probes fire a lot, so "2nd-highest >= 0" is a poor rule
batt = np.full((N, 2*P), -6.0, np.float32)
for i in range(N):
    for p in rng.choice(P, 6, replace=False):
        c = rng.normal(0, 5)
        batt[i, 2*p], batt[i, 2*p+1] = 3.0 - c, 3.0 + c
ids = [f"t{i:04d}" for i in range(N)]
lr.save_margins("gate.npz", ids, topic, batt,
                {"battery_hash": lr.battery_hash(lr.BATTERY,
                                                 lr.TOPIC_HYPOTHESES),
                 "veto_hash": lr.veto_hash(lr.NEGATIVE_TOPIC_HYPOTHESES),
                 "text_col": "text_en"}, veto=veto)

# stratified gold: all of stratum A, a quarter of stratum B
inA = np.zeros(N, bool); inA[rng.choice(N, 400, replace=False)] = True
poolB = np.where(~inA)[0]; sampB = rng.choice(poolB, 400, replace=False)
sel = np.where(inA)[0].tolist() + sampB.tolist()
w = [1.0]*400 + [len(poolB)/400]*400
gdf = pd.DataFrame({"id": [ids[i] for i in sel],
                    "PoliticalAuthority": [gold[i] for i in sel], "w": w})
df = pd.DataFrame({"id": ids, "text_en": ["wort "*30]*N})

fails = []
def check(n, c, d=""):
    print(f"  {'PASS' if c else 'FAIL'}  {n}{('  '+d) if d else ''}")
    if not c: fails.append(n)

# --- the weighted estimator must reproduce corpus truth ---------------------
agg = lr.ContrastAggregator(lr.BATTERY, lr.TOPIC_HYPOTHESES,
                            bootstrap=0, stance_mode="off")
hand_all = (agg._topic(topic) >= 0)
true = lr._prf(hand_all.astype(int), gold, np.ones(N))
est = lr._prf(hand_all[sel].astype(int), gdf.PoliticalAuthority.values, np.array(w))
print("weighted estimator (stratified sample vs the whole corpus):")
for k in ("precision", "recall", "f1"):
    check(f"{k} recovered within 0.05", abs(true[k]-est[k]) < 0.05,
          f"corpus {true[k]:.3f}  estimated {est[k]:.3f}")

res = lr.fit_gate("gate.npz", gdf, "PoliticalAuthority", weight_col="w",
                  battery=lr.BATTERY,
                  topic_hypotheses=lr.TOPIC_HYPOTHESES,
                  veto_hypotheses=lr.NEGATIVE_TOPIC_HYPOTHESES, seed=1)
print("fitted gate:")
check("beats the hand rule out of fold",
      res["fitted"]["f1"] > res["hand"]["f1"] + 0.05,
      f"hand {res['hand']['f1']:.3f} -> fitted {res['fitted']['f1']:.3f}")
coef = res["coef"]
top2 = sorted(coef, key=lambda k: -abs(coef[k]))[:2]
check("recovers the two probes that carry the truth",
      set(top2) == {f"topic{SIGNAL[0]:02d}", f"topic{SIGNAL[1]:02d}"}, str(top2))
check("admit_mask covers every cached row", len(res["admit_mask"]) == N)

# --- and it must flow through rescore ---------------------------------------
out, _ = lr.rescore("gate.npz", df=df, text_col="text_en", id_col="id",
                    battery=lr.BATTERY,
                    topic_hypotheses=lr.TOPIC_HYPOTHESES,
                    veto_hypotheses=lr.NEGATIVE_TOPIC_HYPOTHESES,
                    admit_mask=res["admit_mask"], min_active=1,
                    bootstrap=0, run_diagnostics=False)
adm = (out.lr_status != lr.OFF_TOPIC_LABEL).values
print("through rescore:")
check("the fitted gate decides admission",
      (adm == res["admit_mask"]).mean() > 0.98,
      f"{(adm == res['admit_mask']).mean():.1%} agreement")
check("dropped rows are labelled 'fitted'",
      set(out.loc[~adm, "lr_gate_reason"].unique()) == {"fitted"})
os.remove("gate.npz")
print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)
