"""The negative-probe contrast gate.

The point of a contrast rather than a hard veto: a sentence that is genuinely
both economic and rhetorical -- welfare chauvinism, "they squander your taxes on
migrants" -- must still get in on the strength of its economic side.
"""
import os
import numpy as np, pandas as pd, importlib.util, sys, os
spec = importlib.util.spec_from_file_location("lr", os.environ.get("SCORER", "classify_economic.py"))
lr = importlib.util.module_from_spec(spec); sys.modules["lr"] = lr; spec.loader.exec_module(lr)

P, NT, NV = len(lr.BATTERY), len(lr.TOPIC_HYPOTHESES), len(lr.NEGATIVE_TOPIC_HYPOTHESES)
CASES = [  # name, topic max, topic 2nd, veto max, has economic content
 ("economic, no rhetoric",      12.0,  10.0, -9.0, True),
 ("rhetoric only",             -12.0, -13.0,  11.0, False),
 ("both (welfare chauvinism)",  12.0,  10.0,  11.0, True),
 ("weak economic + rhetoric",    3.0,   2.0,   9.0, True),
]
N = len(CASES)
topic = np.full((N, NT), -14.0, np.float32)
veto  = np.full((N, NV), -10.0, np.float32)
batt  = np.full((N, 2*P), -6.0, np.float32)
for i, (_, tmax, t2, vmax, econ) in enumerate(CASES):
    topic[i, 0], topic[i, 1] = tmax, t2
    veto[i, 0] = vmax
    if econ:
        for p in (0, 1, 2, 3):
            batt[i, 2*p], batt[i, 2*p+1] = 3.0 - 8.0, 3.0 + 8.0   # right-leaning
ids = [f"t{i}" for i in range(N)]
df = pd.DataFrame({"id": ids, "text_en": ["wort "*30]*N})
path = "veto.margins.npz"
lr.save_margins(path, ids, topic, batt,
                {"battery_hash": lr.battery_hash(), "veto_hash": lr.veto_hash(),
                 "text_col": "text_en", "model": "synthetic"}, veto=veto)

fails = []
def check(n, c, d=""):
    print(f"  {'PASS' if c else 'FAIL'}  {n}{('  '+d) if d else ''}")
    if not c: fails.append(n)

out_on,  _ = lr.rescore(path, df=df, text_col="text_en", id_col="id",
                        use_veto=True,  bootstrap=0, run_diagnostics=False)
out_off, _ = lr.rescore(path, df=df, text_col="text_en", id_col="id",
                        use_veto=False, bootstrap=0, run_diagnostics=False)
print(f"\n  {'case':<28}{'veto off':>12}{'veto on':>12}{'reason':>10}")
for i, (name, *_ ) in enumerate(CASES):
    a = out_off.iloc[i]; b = out_on.iloc[i]
    print(f"  {name:<28}{a.lr_status:>12}{b.lr_status:>12}{str(b.lr_gate_reason):>10}")

r = {c[0]: out_on[out_on.id == f"t{i}"].iloc[0] for i, c in enumerate(CASES)}
o = {c[0]: out_off[out_off.id == f"t{i}"].iloc[0] for i, c in enumerate(CASES)}
print("\nchecks:")
check("clean economic text is admitted", r["economic, no rhetoric"].lr_status == "scored")
check("pure rhetoric is dropped on weak topic evidence",
      r["rhetoric only"].lr_gate_reason == "topic")
check("the veto fires on strong rhetoric with weak economics",
      r["weak economic + rhetoric"].lr_gate_reason == "veto")
check("without the veto that row WOULD have been scored",
      o["weak economic + rhetoric"].lr_status == "scored")
check("a sentence strong on BOTH still gets in (contrast, not hard veto)",
      r["both (welfare chauvinism)"].lr_status == "scored")
check(f"status stays '{lr.OFF_TOPIC_LABEL}' for vetoed rows (downstream filters keep working)",
      r["weak economic + rhetoric"].lr_status == lr.OFF_TOPIC_LABEL)
check("the reason column distinguishes the two drop rules",
      set(out_on.lr_gate_reason) == {"admitted", "topic", "veto"})
check("veto margin is recorded", np.isfinite(r["weak economic + rhetoric"].lr_veto))

# an old cache with no veto columns must still work
lr.save_margins("noveto.margins.npz", ids, topic, batt,
                {"battery_hash": lr.battery_hash(), "text_col": "text_en"})
old, _ = lr.rescore("noveto.margins.npz", df=df, text_col="text_en", id_col="id",
                    use_veto=True, bootstrap=0, run_diagnostics=False)
check("a cache without veto columns degrades gracefully",
      (old.lr_status == "scored").sum() == (out_off.lr_status == "scored").sum())

for f in (path, "noveto.margins.npz"):
    if os.path.exists(f): os.remove(f)
print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)
