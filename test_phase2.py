"""Synthetic tests for phase 2. No model, no GPU."""
import os
import numpy as np, pandas as pd, importlib.util, sys, os, tempfile

spec = importlib.util.spec_from_file_location("lr", os.environ.get("SCORER", "classify_economic.py"))
lr = importlib.util.module_from_spec(spec); sys.modules["lr"] = lr
spec.loader.exec_module(lr)

P, NT = len(lr.BATTERY), len(lr.TOPIC_HYPOTHESES)
STRONG, WEAK = 6.0, -6.0

def make(cases):
    """cases: list of (left_pairs, right_pairs, on_topic)."""
    n = len(cases)
    b = np.full((n, 2 * P), WEAK, dtype=np.float32)
    t = np.full((n, NT), -5.0, dtype=np.float32)
    for i, (lp, rp, topic) in enumerate(cases):
        for p in lp:
            b[i, 2 * p], b[i, 2 * p + 1] = STRONG, WEAK
        for p in rp:
            b[i, 2 * p], b[i, 2 * p + 1] = WEAK, STRONG
        if topic:
            t[i, 0], t[i, 1] = 5.0, 4.0
    return b, t

cases = [
    ([0, 1, 2], [], True),                 # A: 3 pairs, left
    ([], [0, 1, 2], True),                 # B: mirror of A
    (list(range(10)), [], True),           # C: same intensity, 10 pairs
    ([0, 1, 2], [3, 4, 5], True),          # D: mixed
    ([0, 1, 2], [], False),                # E: off topic
    ([], [], True),                        # F: on topic, nothing active
]
labels = list("ABCDEF")
b, t = make(cases)
agg = lr.ContrastAggregator(standardize="none", contrast_scale=4.0,
                            bootstrap=200, seed=0,
                            min_active=1, stance_mode="off").fit(b, t)
rows = agg.transform(b, t)

print(f"{'':3}{'status':<18}{'score':>7}{'sd':>7}{'disp':>7}{'n_act':>7}")
for lab, r in zip(labels, rows):
    s = "  nan" if not np.isfinite(r["score"]) else f"{r['score']:7.3f}"
    sd = f"{r['score_sd']:7.3f}" if np.isfinite(r["score_sd"]) else "    nan"
    dp = f"{r['dispersion']:7.3f}" if np.isfinite(r["dispersion"]) else "    nan"
    print(f"{lab:<3}{r['status']:<18}{s}{sd}{dp}{r['n_active']:7d}")

A, B, C, D = (rows[i]["score"] for i in range(4))
fails = []
def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not cond: fails.append(name)

print("\nproperties v1 violated:")
check("left/right symmetry", abs((5 - A) - (B - 5)) < 1e-9,
      f"A={A:.4f} B={B:.4f} -> |offset|={abs((5-A)-(B-5)):.2e}")
check("breadth invariance", abs(A - C) < 1e-9,
      f"3 pairs={A:.4f}  10 pairs={C:.4f}  (v1: more pairs -> more extreme)")
check("mixed text lands near centre", abs(D - 5) < 0.6, f"D={D:.3f}")
check("mixed text flagged by dispersion",
      rows[3]["dispersion"] > 3 * max(rows[0]["dispersion"], 1e-9),
      f"disp mixed={rows[3]['dispersion']:.3f} vs pure={rows[0]['dispersion']:.3f}")
check(f"off-topic -> {lr.OFF_TOPIC_LABEL}", rows[4]["status"] == lr.OFF_TOPIC_LABEL)
check("no evidence -> no_active_pair", rows[5]["status"] == "no_active_pair")
check("left of centre", A < 5 and B > 5, f"A={A:.3f} B={B:.3f}")

# full reachable range, the thing v1 could not reach symmetrically
allL, allR = make([(list(range(P)), [], True), ([], list(range(P)), True)])
ext = agg.transform(allL, allR[1] if False else make([(list(range(P)), [], True), ([], list(range(P)), True)])[1])
lo, hi = ext[0]["score"], ext[1]["score"]
print()
check("scale endpoints symmetric", abs((5 - lo) - (hi - 5)) < 1e-9,
      f"floor={lo:.3f} ceiling={hi:.3f}  (v1 at k=10: 0.40 / 9.83)")

# cross-side independence: adding right-side evidence must not change the
# left side's own magnitude, only the balance
b2, t2 = make([([0], [], True), ([0], [5], True)])
r2 = agg.transform(b2, t2)
print()
print(f"  one left pair            -> {r2[0]['score']:.3f}")
print(f"  same + one right pair    -> {r2[1]['score']:.3f}  (v1: the extra right pair"
      " also rescaled the left side)")

# monotone in intensity
grid = np.full((5, 2 * P), WEAK, dtype=np.float32)
tg = np.full((5, NT), 5.0, dtype=np.float32)
for i, m in enumerate([0.5, 1, 2, 4, 8]):
    grid[i, 0], grid[i, 1] = m, -m
sc = [r["score"] for r in agg.transform(grid, tg)]
print("\nmonotone in evidence strength (margin 0.5 -> 8):",
      " ".join(f"{s:.2f}" for s in sc))
check("monotone decreasing", all(sc[i] > sc[i + 1] for i in range(4)))

# min_active is off above so the maths can be probed on single pairs; check it
# separately here
strict = lr.ContrastAggregator(standardize="none", contrast_scale=4.0, bootstrap=0,
                               min_active=2, stance_mode="off").fit(b, t)
one, _ = make([([0], [], True), ([0, 1], [], True)])
r_one = strict.transform(one, make([([0], [], True), ([0, 1], [], True)])[1])
print()
check("min_active rejects a single active pair",
      r_one[0]["status"] == "thin_evidence" and r_one[1]["status"] == "scored")

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)
