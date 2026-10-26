"""Supervised calibration: does it beat the hand-weighted heuristic without
fooling itself? Ground truth depends on 5 pairs with weights unlike the
battery's, and 4 pairs are pure noise."""
import os
import numpy as np, pandas as pd, importlib.util, sys, os
spec = importlib.util.spec_from_file_location("lr", os.environ.get("SCORER", "classify_economic.py"))
lr = importlib.util.module_from_spec(spec); sys.modules["lr"] = lr; spec.loader.exec_module(lr)

rng = np.random.default_rng(4)
P, NT = len(lr.BATTERY), len(lr.TOPIC_HYPOTHESES)
DOM = [d for d, _, _, _ in lr.BATTERY]
TRUE_W = np.zeros(P)
for d, w in [(DOM[0], 2.0), (DOM[1], 1.6), (DOM[2], 1.2), (DOM[3], 0.8),
             ]:
    TRUE_W[DOM.index(d)] = w
NOISE = [len(DOM) - 4, len(DOM) - 3, len(DOM) - 2, len(DOM) - 1]
N = 900

batt = np.full((N, 2 * P), -6.0, np.float32)
topic = np.full((N, NT), 5.0, np.float32)
ids, lab = [], []
for i in range(N):
    act = rng.choice(P, int(rng.integers(4, 12)), replace=False)
    contrib = 0.0
    for p in act:
        c = rng.normal(0, 5.0)
        batt[i, 2*p], batt[i, 2*p+1] = 3.0 - c, 3.0 + c
        if p not in NOISE:
            contrib += TRUE_W[p] * np.tanh(c / 5.0)
    ids.append(f"t{i:04d}")
    lab.append(contrib + rng.normal(0, 0.8))
lab = np.array(lab)
lab = np.clip(np.round((lab - lab.min()) / (lab.max() - lab.min()) * 5), 0, 5)
df = pd.DataFrame({"id": ids, "text_en": ["wort "*30]*N, "gold": lab})
path = "fit.margins.npz"
lr.save_margins(path, ids, topic, batt,
                {"battery_hash": lr.battery_hash(), "text_col": "text_en",
                 "model": "synthetic"})

fails = []
def check(n, c, d=""):
    print(f"  {'PASS' if c else 'FAIL'}  {n}{('  '+d) if d else ''}")
    if not c: fails.append(n)

res = lr.fit_scale(path, df, "gold", rows="scored", seed=0)
print("checks:")
check("fitted beats the hand-weighted heuristic",
      abs(res["r_fit"]) > abs(res["r_heur"]) + 0.05,
      f"heuristic {res['r_heur']:+.3f} -> fitted {res['r_fit']:+.3f}")
check("out-of-fold predictions exist for every row", np.isfinite(res["oof"]).all())

# the fit must recover which pairs matter
w = res["weights"]
sig = {d: abs(w.get(f"pos::{d}", 0)) for d in DOM}
top5 = sorted(sig, key=sig.get, reverse=True)[:5]
truth = [d for d in DOM if TRUE_W[DOM.index(d)] != 0]
hit = len(set(top5) & set(truth))
check("recovers the pairs that actually drive the label", hit >= 4,
      f"{hit}/5 of the true drivers in the top 5: {top5}")
drv = np.mean([abs(w.get(f"pos::{d}", 0)) for d in truth])
noi = np.mean([abs(w.get(f"pos::{DOM[p]}", 0)) for p in NOISE])
check("planted noise pairs get near-zero weight", noi < 0.25 * drv,
      f"mean |w| drivers {drv:.3f} vs noise {noi:.3f}")

# out-of-fold must be honest: an in-sample refit should look better
ids_, topic_, batt_, stance_, veto_, meta_, _b2 = lr.load_margins(path)
agg_ = lr.ContrastAggregator(bootstrap=0, stance_mode="off").fit(batt_, topic_)
X, _ = lr.build_features(batt_, topic_, stance_, agg_)
y, m = res["y"], res["mask"]
with np.errstate(all="ignore"):
    wv, c = lr._ridge(X[m], y, 30.0)
    r_in = lr._pearson(X[m] @ wv + c, y)
check("in-sample r exceeds out-of-fold r (no leakage)", abs(r_in) > abs(res["r_fit"]),
      f"in-sample {r_in:+.3f} vs OOF {res['r_fit']:+.3f}")

os.remove(path)
print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)
