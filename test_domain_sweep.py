"""drop_domains and the split-half domain sweep.

Positive control: two pairs are wired to pure noise, so the sweep must find them
and the gain must survive on the confirm half.
Negative control: every pair carries signal, so the sweep must NOT manufacture
a confirm-half gain out of selection noise.
"""
import os
import numpy as np, pandas as pd, importlib.util, sys, os
spec = importlib.util.spec_from_file_location("lr", os.environ.get("SCORER", "classify_economic.py"))
lr = importlib.util.module_from_spec(spec); sys.modules["lr"] = lr; spec.loader.exec_module(lr)

P, NT = len(lr.BATTERY), len(lr.TOPIC_HYPOTHESES)
DOMAINS = [d for d, _, _, _ in lr.BATTERY]
NOISE = [d for d, *_ in lr.BATTERY][:2]              # first two domains of this battery
N = 2400

def build(noise_domains, seed=5):
    rng = np.random.default_rng(seed)
    batt = np.full((N, 2 * P), -6.0, np.float32)
    topic = np.full((N, NT), 5.0, np.float32)
    ids, q_all, lab = [], [], []
    ni = {DOMAINS.index(d) for d in noise_domains}
    for i in range(N):
        q = rng.uniform(-1, 1)
        for p in rng.choice(P, int(rng.integers(3, 10)), replace=False):
            c = (rng.normal(0, 4.0) if p in ni else q * 4.0 + rng.normal(0, 1.5))
            batt[i, 2*p], batt[i, 2*p+1] = 3.0 - c, 3.0 + c
        ids.append(f"t{i:04d}"); q_all.append(q)
        lab.append(float(np.clip(round(2 - 2.5*q + rng.normal(0, .4)), 0, 4)))
    df = pd.DataFrame({"id": ids, "text_en": ["wort "*30]*N, "lab": lab, "q": q_all})
    p = f"sweep_{len(noise_domains)}.npz"
    lr.save_margins(p, ids, topic, batt,
                    {"battery_hash": lr.battery_hash(), "text_col": "text_en",
                     "model": "synthetic"})
    return p, df

fails = []
def check(n, c, d=""):
    print(f"  {'PASS' if c else 'FAIL'}  {n}{('  '+d) if d else ''}")
    if not c: fails.append(n)

# ---- drop_domains mechanics -------------------------------------------------
path, df = build(NOISE)
ids, topic, batt, stance, veto, meta, _b2 = lr.load_margins(path)
full = lr.ContrastAggregator(bootstrap=0).fit(batt, topic)
cut  = lr.ContrastAggregator(bootstrap=0, drop_domains=NOISE).fit(batt, topic)
rf, rc = full.transform(batt, topic), cut.transform(batt, topic)
print("drop_domains mechanics:")
check("dropped pairs are never active",
      all(r["top"] == [] or all(t["domain"] not in NOISE for t in r["top"]) for r in rc))
check("n_active falls when pairs are dropped",
      np.mean([r["n_active"] for r in rc]) < np.mean([r["n_active"] for r in rf]),
      f"{np.mean([r['n_active'] for r in rf]):.2f} -> {np.mean([r['n_active'] for r in rc]):.2f}")
try:
    lr.ContrastAggregator(drop_domains=["not_a_domain"]); ok = False
except ValueError: ok = True
check("unknown domain is rejected", ok)
q = df.set_index("id").loc[[str(i) for i in ids], "q"].to_numpy()
sf = np.array([r["score"] for r in rf], float); scut = np.array([r["score"] for r in rc], float)
check("recovery improves once the noise pairs are gone",
      abs(lr._pearson(scut, q)) > abs(lr._pearson(sf, q)),
      f"r {lr._pearson(sf,q):+.3f} -> {lr._pearson(scut,q):+.3f}")

# ---- positive control -------------------------------------------------------
print("\npositive control: two pairs wired to noise")
res = lr.domain_sweep(path, df, "lab", seed=1)
found = set(res["chosen"]) & set(NOISE)
check("the sweep finds at least one planted noise pair", len(found) >= 1,
      f"chose {res['chosen']}")

# ---- negative control -------------------------------------------------------
print("\nnegative control: every pair carries signal")
path2, df2 = build([])
res2 = lr.domain_sweep(path2, df2, "lab", seed=1)
check("the sweep does not invent a large drop set", len(res2["chosen"]) <= 2,
      f"chose {res2['chosen']}")

for f in (path, path2):
    if os.path.exists(f): os.remove(f)
print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)
