"""domain_sweep must be restrictable to one half of a split battery."""
import os, importlib.util, numpy as np, pandas as pd, tempfile, io, contextlib
s=importlib.util.spec_from_file_location("lr", os.environ.get("SCORER","classify_economic.py"))
lr=importlib.util.module_from_spec(s); lr.__name__="lr"; s.loader.exec_module(lr)
doms=[d for d,*_ in lr.BATTERY]
half=tuple(doms[:len(doms)//2]); other=tuple(doms[len(doms)//2:])
rng=np.random.default_rng(0); n=400
nb=len(lr.BATTERY)*2; nt=len(lr.TOPIC_HYPOTHESES); nv=len(lr.NEGATIVE_TOPIC_HYPOTHESES)
ids=[str(i) for i in range(n)]
topic=rng.normal(4,6,(n,nt)); batt=rng.normal(0,7,(n,nb)); veto=rng.normal(-6,4,(n,nv))
y=rng.integers(0,4,n)
i=doms.index(half[0]); batt[:,2*i+1]+=9.0*(y>0)     # one informative pair, in `half`
d=tempfile.mkdtemp(); p=os.path.join(d,"c.npz")
lr.save_margins(p, ids, topic, batt,
                {"model":"t","battery_hash":lr.battery_hash(),"veto_hash":lr.veto_hash()},
                veto=veto)
df=pd.DataFrame({"id":ids,"y":y,"text_en":["x"]*n})
buf=io.StringIO()
with contextlib.redirect_stdout(buf):
    lr.domain_sweep(p, df, label_col="y", fixed_drop=other, n_splits=3, max_drop=2)
out=buf.getvalue()
# the other half must never appear as a drop candidate
for dom in other:
    assert f" {dom} " not in out.replace("\n"," "), f"{dom} was offered for dropping"
# at least one of this half's domains should be considered
assert any(dom in out for dom in half), "no candidate from the swept half"
# and it must still run unrestricted
with contextlib.redirect_stdout(io.StringIO()):
    lr.domain_sweep(p, df, label_col="y", n_splits=3, max_drop=1)
print("test_domain_sweep_half: PASS")
