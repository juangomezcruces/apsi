"""Fixes 1-4 for the v3 liberal-illiberal scorer, cross-validated on the cached
v3 margins (li_v3.margins.npz) and the 200 hand-coded tweets
(li_validation_scores.csv). CPU only.

  1. weight the parts of the concept equally   2. cap the score when defeat is rejected
  3. re-tune negative_weight / contrast_scale / shrinkage   4. stricter firing margin

Selection inside each half uses ICC with NA counted as 5, so a variant cannot
win by declining hard tweets. Writes li_fixes_cv.csv (one row per fold x arm)."""
import itertools, sys
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, ".")
from classifier import alternativeLib_v3 as L
z = np.load("li_v3.margins.npz"); M = z["margins"]; g = pd.read_csv("li_validation_scores.csv")
S = L.LiberalIlliberalScorer(margin_fn=lambda p, h: None)
nt = S.n_topic; LIB, ILL = M[:, nt+1::2], M[:, nt+2::2]; MD = 1/(1+np.exp(-M[:, nt]))
W0, KEEP, DEF = S.weights, S.keep, S.defensive_mask
COMP = np.array(S.components); COMPS = L.COMPONENTS
rel = (g.rel_both == 1).to_numpy(); gold = g.gold.to_numpy()

def score(comp_equal, cap, am, neg, scale, k0):
    act = (np.fmax(LIB, ILL) >= am) & KEEP
    c = (LIB - ILL) / 2
    c = c * np.where(DEF & (c < 0), 1 - S.defensive_discount * MD[:, None], 1.0)
    pos = np.where(act, np.tanh(c / scale), 0.0)
    w = W0 * act * np.where(pos < 0, neg, 1.0)
    cp = {}
    for k in COMPS:
        sel = COMP == k; ws = w[:, sel].sum(1)
        cp[k] = np.where(ws > 0, (w[:, sel] * pos[:, sel]).sum(1) / np.where(ws > 0, ws, 1), np.nan)
    if comp_equal:
        P = np.column_stack([cp[k] for k in COMPS]); has = np.isfinite(P); P0 = np.where(has, P, 0)
        cw = has * np.where(P0 < 0, neg, 1.0)
        s = 5 + 5 * (cw * P0).sum(1) / (cw.sum(1) + k0)
    else:
        s = 5 + 5 * (w * pos).sum(1) / (w.sum(1) + k0)
    if cap is not None:
        s = np.where(np.nan_to_num(cp["defeat"], nan=0) < -0.5, np.minimum(s, cap), s)
    gate = np.sort(M[:, :nt], 1)[:, -2] >= S.topic_margin_min   # shared engine gate: 2nd-highest probe
    long_enough = np.array([len(str(t).split()) for t in z["texts"]]) >= S.min_tokens   # shared length gate
    return np.where(act.any(1) & gate & long_enough, np.clip(s, 0, 10), np.nan)

# sanity: the defaults reproduce the module
mod = np.array([r["score"] if r["is_relevant"] else np.nan for r in (S.score_from_margins(m, t) for m, t in zip(M, z["texts"]))])
base = score(False, None, 0.0, 3.0, 8.0, 2.0)
assert np.allclose(np.nan_to_num(mod, nan=-1), np.nan_to_num(base, nan=-1)), "vectorised scorer differs"

def icc(x, y):
    X = np.c_[x, y]; n, k = X.shape; gm = X.mean()
    msr = k*((X.mean(1)-gm)**2).sum()/(n-1); msc = n*((X.mean(0)-gm)**2).sum()/(k-1)
    mse = ((X-X.mean(1, keepdims=True)-X.mean(0, keepdims=True)+gm)**2).sum()/((n-1)*(k-1))
    return (msr-mse)/(msr+(k-1)*mse+k*(msc-mse)/n)
def kappa(a, b):
    po = (a == b).mean(); pe = sum((a == k).mean()*(b == k).mean() for k in (0, 1)); return (po-pe)/(1-pe)
def metrics(s, idx):
    y = gold[idx]; x = s[idx]; m = np.isfinite(x)
    return dict(icc_na5=icc(np.where(m, x, 5.0), y), icc=icc(x[m], y[m]), rho=stats.spearmanr(x[m], y[m])[0],
                mae=np.abs(x[m]-y[m]).mean(), dirk=kappa(x[m] >= 5, y[m] >= 5), cov=m.mean(),
                fail=int((x[m & (y < 3)] >= 5).sum()))

GRID = list(itertools.product([False, True], [None, 4.5, 3.0], [0.0, 2.0, 4.0],
                              [1.0, 2.0, 3.0, 4.0], [4.0, 8.0, 12.0], [0.5, 1.0, 2.0, 4.0]))
SC = {cfg: score(*cfg) for cfg in GRID}
idx_all = np.flatnonzero(rel)
SEL = lambda s, idx: metrics(s, idx)["icc_na5"]   # coverage-aware: an NA counts as 5

def cv(allowed, reps=50):
    out = []
    for rep in range(reps):
        p = np.random.default_rng(rep).permutation(idx_all); A, B = p[:len(p)//2], p[len(p)//2:]
        for tr, te in [(A, B), (B, A)]:
            best = max(allowed, key=lambda c: SEL(SC[c], tr))
            out.append({**metrics(SC[best], te), "cfg": best})
    return pd.DataFrame(out)

DEFAULT = (False, None, 0.0, 3.0, 8.0, 2.0)
cols = ["icc_na5", "icc", "rho", "mae", "dirk", "cov", "fail"]
rows = {}
rows["current v3 (fixed)"] = cv([DEFAULT])
rows["3. settings re-tuned"] = cv([c for c in GRID if c[:3] == (False, None, 0.0)])
for name, f in [("1. parts weighted equally", lambda c: c[0]), ("1. off", lambda c: not c[0]),
                ("2. defeat cap 4.5", lambda c: c[1] == 4.5), ("2. defeat cap 3.0", lambda c: c[1] == 3.0),
                ("2. off", lambda c: c[1] is None),
                ("4. fire at margin 2", lambda c: c[2] == 2.0), ("4. fire at margin 4", lambda c: c[2] == 4.0),
                ("4. off (margin 0)", lambda c: c[2] == 0.0)]:
    rows[name] = cv([c for c in GRID if f(c)])
rows["all free"] = cv(GRID)
T = pd.DataFrame({k: v[cols].mean() for k, v in rows.items()}).T
print("HELD-OUT, mean over 100 half-splits (fail = per held-out half, ~19 gold<3 tweets)")
print(T.round(3).to_string())
print("\nconfigs chosen when all free (comp_equal, cap, margin, neg, scale, k0):")
print(rows["all free"].cfg.value_counts().head(8).to_string())

# paired fold-level differences for each fix vs its 'off' arm
def diff(a, b, col="icc_na5"):
    d = rows[a][col].values - rows[b][col].values
    return f"{d.mean():+.3f}  (better in {np.mean(d > 0):.0%} of folds, worse in {np.mean(d < 0):.0%})"
print("\nfix vs off, held-out ICC (NA=5), same folds:")
for a, b in [("1. parts weighted equally", "1. off"), ("2. defeat cap 4.5", "2. off"), ("2. defeat cap 3.0", "2. off"),
             ("4. fire at margin 2", "4. off (margin 0)"), ("4. fire at margin 4", "4. off (margin 0)"),
             ("3. settings re-tuned", "current v3 (fixed)"), ("all free", "current v3 (fixed)")]:
    print(f"  {a:<28} vs {b:<20} {diff(a, b)}")
print("\nfull sample at fixed settings (3, 8, 2), structural switches only:")
for cfg in [DEFAULT, (True, None, 0.0, 3., 8., 2.), (False, 4.5, 0.0, 3., 8., 2.), (False, 3.0, 0.0, 3., 8., 2.),
            (False, None, 2.0, 3., 8., 2.), (True, 4.5, 0.0, 3., 8., 2.)]:
    print(" ", cfg, {k: round(v, 3) for k, v in metrics(SC[cfg], idx_all).items()})

pd.concat([v.drop(columns="cfg").assign(arm=k, fold=np.arange(len(v))) for k, v in rows.items()]).to_csv(
    "li_fixes_cv.csv", index=False)
print("\nwrote li_fixes_cv.csv")
