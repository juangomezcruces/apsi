"""The populism/pluralism dimension runs on the same machinery as left-right."""
import os
import numpy as np, pandas as pd, importlib.util, sys, os
spec = importlib.util.spec_from_file_location("lr", os.environ.get("SCORER", "classify_economic.py"))
lr = importlib.util.module_from_spec(spec); sys.modules["lr"] = lr; spec.loader.exec_module(lr)
if lr.BANDS[0][1] != "Strong Pluralist":
    print("test_populism: SKIP (not the populism dimension)"); raise SystemExit

B, T, V = lr.BATTERY, lr.TOPIC_HYPOTHESES, lr.NEGATIVE_TOPIC_HYPOTHESES
P, NT, NV = len(B), len(T), len(V)
fails = []
def check(n, c, d=""):
    print(f"  {'PASS' if c else 'FAIL'}  {n}{('  '+d) if d else ''}")
    if not c: fails.append(n)

print("structure:")
check("battery is paired, so the sides are balanced by construction",
      all(len(p) == 4 for p in B) and len({d for d,_,_,_ in B}) == P, f"{P} pairs")
check("no position statement leaked into the topic gate",
      not (set(T) & {h for _,_,l,r in B for h in (l,r)}))
check("no topic probe leaked into the negatives", not (set(T) & set(V)))
check("populism bands name their own ends",
      lr.interpret(0.5, lr.BANDS) == "Strong Pluralist"
      and lr.interpret(9.5, lr.BANDS) == "Strong Populist")

# the v1 asymmetry this replaces: 12 populist hyps at weight 10.15 vs 11 at 8.65
STRONG, WEAK = 7.0, -7.0
def make(cases):
    n = len(cases)
    b = np.full((n, 2*P), WEAK, np.float32)
    t = np.full((n, NT), -6.0, np.float32)
    v = np.full((n, NV), -6.0, np.float32)
    for i, (plu, pop, topic) in enumerate(cases):
        for p in plu: b[i, 2*p], b[i, 2*p+1] = STRONG, WEAK
        for p in pop: b[i, 2*p], b[i, 2*p+1] = WEAK, STRONG
        if topic: t[i, 0], t[i, 1] = 6.0, 5.0
    return b, t, v

allp = list(range(P))
b, t, v = make([(allp, [], True), ([], allp, True), ([0,1,2], [], True), ([], [0,1,2], True)])
agg = lr.ContrastAggregator(B, T, standardize="none", contrast_scale=4.0, bootstrap=0,
                            min_active=1, veto_hypotheses=V,
                            bands=lr.BANDS).fit(b, t)
r = agg.transform(b, t, veto_margins=v)
lo, hi, lo3, hi3 = (x["score"] for x in r)
print("\nscale symmetry (the v1 battery gave the populist end ~0.46 points more headroom):")
check("the two ends are equidistant from centre", abs((5-lo) - (hi-5)) < 1e-9,
      f"pluralist floor {lo:.3f}  populist ceiling {hi:.3f}")
check("3 pairs and 22 pairs at equal intensity score the same", abs(lo-lo3) < 1e-9,
      f"{lo3:.3f} vs {lo:.3f}")
check("pluralist scores low, populist scores high", lo < 5 < hi)
check("bands applied", r[0]["interpretation"] == "Strong Pluralist"
      and r[1]["interpretation"] == "Strong Populist")

# end to end through rescore, with the veto
ids = [f"t{i}" for i in range(4)]
df = pd.DataFrame({"id": ids, "text_en": ["wort "*30]*4})
lr.save_margins("pop.npz", ids, t, b,
                {"battery_hash": lr.battery_hash(B, T), "veto_hash": lr.veto_hash(V),
                 "text_col": "text_en"}, veto=v)
out, _ = lr.rescore("pop.npz", df=df, text_col="text_en", id_col="id",
                    battery=B, topic_hypotheses=T, veto_hypotheses=V,
                    bands=lr.BANDS, standardize="none", contrast_scale=4.0,
                    min_active=1, bootstrap=0, run_diagnostics=False)
print("\nend to end:")
check("rescore accepts the second dimension", out.lr_score.notna().all())
check("labels come from the populism bands",
      set(out.lr_interpretation) <= {n for _, n in lr.BANDS},
      str(list(out.lr_interpretation)))
os.remove("pop.npz")
print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)
