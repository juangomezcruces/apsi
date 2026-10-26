"""Populism as the conjunction of people-centrism and anti-elitism.

The failure this guards against: a sentence that is strongly anti-elitist and
not at all people-centric scoring as populist. That is what the previous
instrument did, and it is why every populism figure it produced was really an
anti-elitism figure."""
import os, importlib.util, numpy as np, pandas as pd, tempfile
s=importlib.util.spec_from_file_location("lr", os.environ.get("SCORER","classify_populism.py"))
lr=importlib.util.module_from_spec(s); lr.__name__="lr"; s.loader.exec_module(lr)
if not hasattr(lr,"AE_DOMAINS"):
    print("test_two_halves: SKIP (single-scale dimension)"); raise SystemExit

# --- partition ------------------------------------------------------------
dom=[d for d,*_ in lr.BATTERY]
assert len(set(lr.AE_DOMAINS)|set(lr.PC_DOMAINS)|set(lr.SHARED_DOMAINS))==len(dom)
assert not (set(lr.AE_DOMAINS)&set(lr.PC_DOMAINS)), "a pair cannot be in both halves"
assert len(lr.PC_DOMAINS)>=len(lr.AE_DOMAINS)-2, "people-centrism half must not be thin"

# --- the combination rule encodes the definition --------------------------
c=lr.combine_populism
assert abs(float(c(9,9))-9.0)<1e-9,        "both halves populist -> populist"
assert abs(float(c(1,1))-1.0)<1e-9,        "both halves pluralist -> pluralist"
assert abs(float(c(5,5))-5.0)<1e-9,        "neutral stays neutral"
# the headline property: one half alone is NOT populism
assert float(c(9,5)) < 6.0, "anti-elitism alone must not read as populist"
assert float(c(5,9)) < 6.0, "people-centrism alone must not read as populist"
assert float(c(9,2)) < 6.0, "opposed halves must not read as populist"
assert float(c(9,9)) > float(c(9,7)) > float(c(9,5.5)), "monotone in the weaker half"
# and it must be strictly stricter than averaging on the mixed case
assert float(c(9,5.5)) < float(c(9,5.5,rule="mean")), "conjunctive must damp vs mean"
assert np.isnan(float(c(np.nan,7))), "missing half -> missing score"
assert 0.0<=float(c(0,0))<=10.0 and 0.0<=float(c(10,10))<=10.0, "stays in range"
sym=[float(c(a,b))-float(c(b,a)) for a,b in [(9,6),(3,1),(8,2)]]
assert max(abs(x) for x in sym)<1e-9, "the two halves must be interchangeable"

# --- end to end on a synthetic cache --------------------------------------
d=tempfile.mkdtemp(); rng=np.random.default_rng(0); n=260
P=len(lr.BATTERY); nt=len(lr.TOPIC_HYPOTHESES); nv=len(lr.NEGATIVE_TOPIC_HYPOTHESES)
ids=[str(i) for i in range(n)]
topic=rng.normal(9,4,(n,nt))                       # everything passes the gate
ae_i=[dom.index(x) for x in lr.AE_DOMAINS]; pc_i=[dom.index(x) for x in lr.PC_DOMAINS]
batt=rng.normal(0,3,(n,2*P))
half=n//2
for i in ae_i:                                     # every row strongly anti-elitist
    batt[:,2*i]-=9; batt[:,2*i+1]+=9
for i in pc_i:                                     # only the FIRST half people-centric
    batt[:half,2*i]-=9; batt[:half,2*i+1]+=9
    batt[half:,2*i]+=9; batt[half:,2*i+1]-=9       # the rest are pluralist on that half
path=os.path.join(d,"c.npz")
lr.save_margins(path, ids, topic, batt,
                {"model":"t", "battery_hash": lr.battery_hash(),
                 "veto_hash": lr.veto_hash()}, veto=rng.normal(-9,2,(n,nv)))
src=pd.DataFrame({"id":ids, "text_en":["alpha beta gamma delta epsilon zeta eta"]*n, "faction":["A","B"]*(n//2)})
out=lr.score_populism(path, output_csv=None, df=src, min_active=1)
ae=next(c_ for c_ in out.columns if c_.startswith("lr_score") and c_.endswith("_ae"))
pc=next(c_ for c_ in out.columns if c_.startswith("lr_score") and c_.endswith("_pc"))
assert out[ae].notna().sum()>n*0.8 and out[pc].notna().sum()>n*0.8
both=out.iloc[:half]; onlyae=out.iloc[half:]
assert both[ae].mean()>7 and both[pc].mean()>7, "both halves should read populist"
assert onlyae[ae].mean()>7 and onlyae[pc].mean()<3, "second group: anti-elite, pluralist people-half"
assert both.pop_score.mean()>7, f"true populists scored {both.pop_score.mean():.2f}"
assert onlyae.pop_score.mean()<6, \
    f"anti-elitism-only scored {onlyae.pop_score.mean():.2f} -- the old bug is back"
assert both.pop_both_halves.mean()>0.8 and onlyae.pop_both_halves.mean()<0.2
assert {"pop_score","pop_interpretation","pop_both_halves","pop_rule"} <= set(out.columns)
print(f"test_two_halves: PASS  (both halves {both.pop_score.mean():.2f} vs "
      f"anti-elitism only {onlyae.pop_score.mean():.2f})")
