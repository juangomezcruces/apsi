"""Paraphrase averaging and label-free balance weighting."""
import os
import numpy as np, pandas as pd, importlib.util, sys, os
spec = importlib.util.spec_from_file_location("lr", os.environ.get("SCORER", "classify_economic.py"))
lr = importlib.util.module_from_spec(spec); sys.modules["lr"] = lr; spec.loader.exec_module(lr)

rng = np.random.default_rng(9)
P, NT = len(lr.BATTERY), len(lr.TOPIC_HYPOTHESES)
DOM = [d for d, _, _, _ in lr.BATTERY]
N = 1200
ONESIDED = [len(DOM) - 3, len(DOM) - 2, len(DOM) - 1]   # any three domains

def build(noise_sd):
    """Two wordings of the same pair, each with independent template noise."""
    b1 = np.full((N, 2*P), -6.0, np.float32); b2 = b1.copy()
    topic = np.full((N, NT), 5.0, np.float32)
    q = rng.uniform(-1, 1, N); ids = [f"t{i:04d}" for i in range(N)]
    for i in range(N):
        for p in rng.choice(P, int(rng.integers(4, 11)), replace=False):
            true_c = q[i] * 5.0
            if p in ONESIDED:                     # fires right almost always
                true_c = abs(true_c) + 4.0
            for arr in (b1, b2):
                c = true_c + rng.normal(0, noise_sd)
                arr[i, 2*p], arr[i, 2*p+1] = 3.0 - c, 3.0 + c
    return ids, topic, b1, b2, q

ids, topic, b1, b2, q = build(noise_sd=4.0)
df = pd.DataFrame({"id": ids, "text_en": ["wort "*30]*N, "q": q})
lr.save_margins("ps.npz", ids, topic, b1,
                {"battery_hash": lr.battery_hash(), "text_col": "text_en"}, battery2=b2)

fails = []
def check(n, c, d=""):
    print(f"  {'PASS' if c else 'FAIL'}  {n}{('  '+d) if d else ''}")
    if not c: fails.append(n)

def r_vs_truth(**kw):
    out, agg = lr.rescore("ps.npz", df=df, text_col="text_en", id_col="id",
                          bootstrap=0, run_diagnostics=False, **kw)
    s = out[out.lr_score.notna()]
    return abs(lr._pearson(s.lr_score, s.q)), agg, len(s)

r_single, _, n1 = r_vs_truth(average_paraphrase=False)
r_avg, _, n2 = r_vs_truth(average_paraphrase=True)
print("\nparaphrase averaging (both wordings measure the same truth + noise):")
check("averaging recovers the latent position better", r_avg > r_single + 0.01,
      f"single r={r_single:.3f} -> averaged r={r_avg:.3f}")
check("it scores the same rows, it does not filter", n1 == n2, f"{n1} vs {n2}")

r_b0, a0, _ = r_vs_truth(average_paraphrase=True, balance_weight=0.0)
r_b5, a5, _ = r_vs_truth(average_paraphrase=True, balance_weight=0.5)
rr = np.array(a5.fit_["raw_right_share"])
w0, w5 = np.array(a0.fit_["weights_eff"]), np.array(a5.fit_["weights_eff"])
print("\nbalance weighting (three pairs wired to fire right almost always):")
check("one-sided pairs lose weight",
      all(w5[p] < 0.75 * w0[p] for p in ONESIDED),
      "  ".join(f"{DOM[p]} {100*rr[p]:.0f}%R w{w0[p]:.2f}->{w5[p]:.2f}" for p in ONESIDED))
bal = [p for p in range(P) if p not in ONESIDED and np.isfinite(rr[p])
       and abs(rr[p]-0.5) < 0.15]
check("balanced pairs keep theirs", all(w5[p] > 0.85 * w0[p] for p in bal),
      f"{len(bal)} balanced pairs")
check("it uses no labels", "q" not in str(a5.config()) and "label" not in str(a5.config()))
check("recovery improves once they are down-weighted", r_b5 > r_b0,
      f"r {r_b0:.3f} -> {r_b5:.3f}")

os.remove("ps.npz")
print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)
