"""slice_margins(): one GPU pass over a wide probe list can serve several gate
designs. Dimension-agnostic -- it only needs a cache."""
import os, importlib.util, numpy as np, tempfile
s=importlib.util.spec_from_file_location("lr", os.environ.get("SCORER","classify_economic.py"))
lr=importlib.util.module_from_spec(s); lr.__name__="lr"; s.loader.exec_module(lr)

d=tempfile.mkdtemp(); rng=np.random.default_rng(0); n=120
NT=len(lr.TOPIC_HYPOTHESES); nb=len(lr.BATTERY)*2; nv=len(lr.NEGATIVE_TOPIC_HYPOTHESES)
WIDE=NT+6
ids=[str(i) for i in range(n)]
topic=rng.normal(0,6,(n,WIDE)); batt=rng.normal(0,6,(n,nb)); veto=rng.normal(0,6,(n,nv))
comb=os.path.join(d,"c.npz"); lr.save_margins(comb, ids, topic, batt, {"model":"t"}, veto=veto)

lo=lr.slice_margins(comb, os.path.join(d,"lo.npz"), 0, NT)
hi=lr.slice_margins(comb, os.path.join(d,"hi.npz"), NT, WIDE)
i1,t1,b1,_,v1,_,_ = lr.load_margins(lo)
i2,t2,_,_,_,_,_   = lr.load_margins(hi)
assert t1.shape==(n,NT) and t2.shape==(n,WIDE-NT)
assert np.allclose(t1, topic[:,:NT]) and np.allclose(t2, topic[:,NT:])
assert [str(x) for x in i1]==ids and np.allclose(b1,batt) and np.allclose(v1,veto)

agg=lr.ContrastAggregator(lr.BATTERY, lr.TOPIC_HYPOTHESES, bootstrap=0,
                          stance_mode="off", veto_hypotheses=lr.NEGATIVE_TOPIC_HYPOTHESES)
assert np.allclose(agg._topic(t1), agg._topic(topic[:,:NT])), "gate rule must survive the slice"
# slicing is lossless: the two halves reassemble the original
assert np.allclose(np.hstack([t1,t2]), topic)
print("test_slice_margins: PASS")
