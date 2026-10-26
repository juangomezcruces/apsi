"""Does centring zero the mean position on SATURATED, BIMODAL contrasts?
Reproduces the 'redistribution' anomaly: n_ctr=29 yet mean pos|act = +0.491."""
import os
import numpy as np, importlib.util, sys
spec = importlib.util.spec_from_file_location("lr", os.environ.get("SCORER", "classify_economic.py"))
lr = importlib.util.module_from_spec(spec); sys.modules["lr"] = lr; spec.loader.exec_module(lr)

rng = np.random.default_rng(11)
P, NT, N = len(lr.BATTERY), len(lr.TOPIC_HYPOTHESES), 4000
batt = np.full((N, 2 * P), -12.0, dtype=np.float32)
topic = np.full((N, NT), 6.0, dtype=np.float32)
# each pair: active at its own rate, active contrasts BIMODAL at +-10 logits
rates = np.linspace(0.06, 0.55, P)
skews = rng.uniform(0.25, 0.85, P)          # template bias = share on the right mode
for p in range(P):
    act = rng.random(N) < rates[p]
    idx = np.where(act)[0]
    right = rng.random(idx.size) < skews[p]
    c = np.where(right, rng.normal(11, 1.5, idx.size), rng.normal(-9, 1.5, idx.size))
    batt[idx, 2 * p] = 3.0 - c
    batt[idx, 2 * p + 1] = 3.0 + c

for stat in ("median (old)", "solve (new)"):
    agg = lr.ContrastAggregator(standardize="center", contrast_scale="auto",
                                bootstrap=0, min_center_n=25)
    contrast, active = agg._contrasts(batt)
    if stat.startswith("median"):
        agg.fit_ = {"centre": [float(np.median(contrast[active[:, p], p])) for p in range(P)],
                    "spread": [1.0] * P, "scale": 10.0}
    else:
        agg.fit(batt, topic)
    pos = agg._positions(contrast)
    mp = np.array([pos[active[:, p], p].mean() for p in range(P)])
    print(f"{stat:<14} mean|pos| across pairs = {np.abs(mp).mean():.4f}   "
          f"worst pair = {np.abs(mp).max():.4f}")
    if stat.startswith("solve"):
        assert np.abs(mp).max() < 0.02, f"centring failed: worst={np.abs(mp).max():.3f}"

# and the score must still track the underlying skew
rows = agg.transform(batt, topic)
sc = np.array([r["score"] for r in rows], float)
print(f"\nscore mean {np.nanmean(sc):.2f}  sd {np.nanstd(sc):.2f}  "
      f"(centred corpus should sit at ~5.0)")
assert abs(np.nanmean(sc) - 5.0) < 0.25
print("PASS  bimodal centring zeroes mean position and leaves the corpus at centre")
