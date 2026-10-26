"""Stratified coding sheet + gate evaluation. The estimator must recover known
precision and recall despite the sample being deliberately non-proportional."""
import os
import numpy as np, pandas as pd, importlib.util, sys, os
spec = importlib.util.spec_from_file_location("lr", os.environ.get("SCORER", "classify_economic.py"))
lr = importlib.util.module_from_spec(spec); sys.modules["lr"] = lr; spec.loader.exec_module(lr)

rng = np.random.default_rng(2)
N, N_ADM = 2000, 600
P_ECON_ADM, P_ECON_DROP = 0.80, 0.10          # ground truth
TRUE_PREC = P_ECON_ADM
TRUE_REC = (N_ADM*P_ECON_ADM) / (N_ADM*P_ECON_ADM + (N-N_ADM)*P_ECON_DROP)

adm = np.zeros(N, bool); adm[:N_ADM] = True
econ = np.where(adm, rng.random(N) < P_ECON_ADM, rng.random(N) < P_ECON_DROP)
pos = np.where(econ, rng.integers(-2, 3, N), np.nan)
# model score tracks gold position with noise, only for admitted rows
score = np.where(adm, np.clip(5 + 2.5*np.nan_to_num(pos) + rng.normal(0, 1.6, N), 0, 10), np.nan)
res = pd.DataFrame({"id": [f"t{i:04d}" for i in range(N)],
                    "text_en": [f"sentence {i}" for i in range(N)],
                    "lr_status": np.where(adm, "scored", lr.OFF_TOPIC_LABEL),
                    "lr_score": score})
res.to_csv("gate_results.csv", index=False)

lr.make_coding_sheet("gate_results.csv", "gate_sheet.csv", n_admit=150, n_drop=150, seed=1)

# "hand-code" the sheet from the ground truth
truth = dict(zip(res.id, econ)); tpos = dict(zip(res.id, pos))
sheet = pd.read_csv("gate_sheet.csv")
sheet["gold_economic"] = [int(truth[i]) for i in sheet.id]
sheet["gold_position"] = [tpos[i] if truth[i] else np.nan for i in sheet.id]
sheet.to_csv("gate_sheet.csv", index=False)

out = lr.evaluate_gate("gate_sheet.csv")

fails = []
def check(n, c, d=""):
    print(f"  {'PASS' if c else 'FAIL'}  {n}{('  '+d) if d else ''}")
    if not c: fails.append(n)
print("checks:")
check("recovers true precision", abs(out["precision"] - TRUE_PREC) < 0.07,
      f"true {TRUE_PREC:.1%} vs estimated {out['precision']:.1%}")
check("recovers true recall despite non-proportional sampling",
      abs(out["recall"] - TRUE_REC) < 0.07,
      f"true {TRUE_REC:.1%} vs estimated {out['recall']:.1%}")
naive = out["n_admitted_coded"] / (out["n_admitted_coded"] + out["n_dropped_coded"])
check("reweighting matters (naive sample ratio would mislead)",
      abs(naive - TRUE_REC) > 0.15,
      f"unweighted sample is {naive:.1%} admitted vs true recall {TRUE_REC:.1%}")
check("blind sheet carries no model output",
      set(pd.read_csv("gate_sheet.csv").columns) >=
      {"id","text_en","gold_economic","gold_position"} and
      not any(c.startswith("lr_") for c in
              pd.read_csv("gate_sheet.csv").columns))

for f in ("gate_results.csv","gate_sheet.csv","gate_sheet.strata.json"):
    if os.path.exists(f): os.remove(f)
print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)
