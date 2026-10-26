"""evaluate_populism must report the two halves SEPARATELY and must not be
foolable by an instrument that is strong on one half and blind on the other --
which is exactly the failure it exists to catch."""
import os, importlib.util, numpy as np, pandas as pd, io, contextlib
s=importlib.util.spec_from_file_location("lr", os.environ.get("SCORER","classify_populism.py"))
lr=importlib.util.module_from_spec(s); lr.__name__="lr"; s.loader.exec_module(lr)
if not hasattr(lr,"evaluate_populism"):
    print("test_evaluate_populism: SKIP (single-scale dimension)"); raise SystemExit

rng=np.random.default_rng(0); n=400
gold=pd.DataFrame({"id":[str(i) for i in range(n)],
                   "AntiElitism":rng.integers(0,2,n),
                   "PeopleCentrism":rng.integers(0,2,n),
                   "w":rng.choice([1.0,4.3],n)})
gold["OnTopic"]=((gold.AntiElitism+gold.PeopleCentrism)>0).astype(int)

def make(ae_signal, pc_signal):  # one scale; pc_signal shifts how much it also tracks PC
    # directional scores: 0 = pluralist pole, 10 = populist pole
    return pd.DataFrame({
        "id":gold.id,
        "anti_elitism_n_true":gold.AntiElitism*3,
        "people_centrism_n_true":gold.PeopleCentrism*3,
        "lr_score":5+ae_signal*3*(gold.AntiElitism*2-1)+pc_signal*3*(gold.PeopleCentrism*2-1)+rng.normal(0,1,n),
        "lr_score_unused":rng.normal(5,1,n),
        "lr_topic_margin":rng.normal(6,6,n),
        "faction":rng.choice(list("AB"),n)})

def run(res):
    buf=io.StringIO()
    with contextlib.redirect_stdout(buf):
        lr.evaluate_populism(res, gold)
    return buf.getvalue()

# 1. an instrument good on BOTH halves
out=run(make(1,1))
assert "anti-elitism" in out and "people-centrism" in out
assert "Both components detected" in out or "LOPSIDED" in out, out[-500:]

# 2. THE FAILURE CASE: strong anti-elitism, blind on people-centrism.
#    This is what the previous instrument did and what a combined number hid.
out=run(make(1,0))
assert "PEOPLE-CENTRISM NOT DETECTED" in out, \
    "must flag a half-blind scale rather than averaging the failure away"
assert "Report it under that name" in out

# 3. both halves reported on their own line, against the prior benchmark
for frag in ("was 0.806","was 0.480","PARTIALLED","THE GATE","BOTH COMPONENTS"):
    assert frag in out, frag

# 4. the gate sweep shows F1 and validity together so validity can win
assert "Choose on VALIDITY, not F1" in out
hdr=[l for l in out.split("\n") if "AUC(AE)" in l]
assert hdr and "F1" in hdr[0], "sweep must show F1 and both AUCs side by side"

# 5. REGRESSION: score validity must use the DIRECTIONAL PopBERT targets, not the
#    non-directional gold columns. Build a frame whose ae_score tracks the
#    directional label but is deliberately uncorrelated with the gold column;
#    a correct evaluator reports high AUC, the category-error version reports ~0.5.
gold2=gold.copy()
gold2["AntiElitism"]=rng.integers(0,2,n)          # decoupled from the score
res2=pd.DataFrame({"id":gold.id,
    "anti_elitism_n_true":gold.AntiElitism*3,
    "people_centrism_n_true":gold.PeopleCentrism*3,
    "lr_score":5+3*(gold.AntiElitism*2-1)+rng.normal(0,1,n),
    "lr_score_unused":5+3*(gold.PeopleCentrism*2-1)+rng.normal(0,1,n),
    "lr_topic_margin":rng.normal(6,6,n),"faction":rng.choice(list("AB"),n)})
buf=io.StringIO()
with contextlib.redirect_stdout(buf):
    lr.evaluate_populism(res2, gold2)
line=[l for l in buf.getvalue().split("\n") if l.strip().startswith("anti-elitism")][0]
auc=float(line.split()[2])
assert auc > 0.9, f"score validity must use the directional target (got AUC {auc})"

# 6. it returns the merged frame for further work
res=lr.evaluate_populism(make(1,1), gold)
assert len(res)==n and {"AntiElitism","PeopleCentrism","OnTopic"} <= set(res.columns)
print("test_evaluate_populism: PASS")
