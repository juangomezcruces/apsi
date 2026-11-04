"""Inferential statistics for the report.

    python -m ltg stats              # -> results/text/stats_*.csv

Unit of analysis = the book (national corpus of student books). Eras group the
editions by the text-reuse families (see reuse.py); 2008-22 is the reference.

  rate ratios   Poisson GLM, count of mentions per book ~ era + offset(log words)
  trend         the same over 1960-2022 with years since 1960 (per decade)
  logits        binary outcomes per mention or illustration ~ era

Standard errors are clustered by book title (`code`): the 2023-2025 reprints of one
NEM title share a code, so they count as one cluster, not three. Tests are
exploratory; Holm-adjusted p-values are given across each table.
"""
import logging

import numpy as np
import pandas as pd

from .config import PATHS
from .text_analysis import _in_corpus, out_dir

log = logging.getLogger("ltg.stats")

ERA = {"1960": "1960-62", "1962": "1960-62", "1972": "1972-88", "1982": "1972-88", "1988": "1972-88",
       "1993": "1993", "2008": "2008-22", "2011": "2008-22", "2014": "2008-22", "2018": "2008-22",
       "2019": "2008-22", "2022": "2008-22", "2023": "NEM", "2024": "NEM", "2025": "NEM"}
ERAS = ["1960-62", "1972-88", "1993", "2008-22", "NEM"]
REF = "2008-22"
FAMILIES = ["people_all", "pueblo", "nacion", "ciudadania", "comunidad", "clase", "popular", "etnico",
            "agravio", "elite", "pluralismo", "unidad", "instituciones"]


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def book_counts():
    """Book-level words and mentions per family (settlement / classroom uses removed)."""
    td = out_dir()
    pt = _in_corpus(pd.read_csv(td / "page_tokens.csv", dtype={"edition": str}, keep_default_na=False), "national")
    pt = pt[pt["edition"].isin(ERA)]
    keep = set(zip(pt["edition"], pt["code"]))
    inside = lambda df: df[[k in keep for k in zip(df["edition"], df["code"])]]
    ptok = inside(pd.read_csv(td / "people_tokens.csv.gz", dtype={"edition": str}, keep_default_na=False))
    ptok["counted"] = ptok["counted"].astype(str) == "True"
    m = inside(pd.read_csv(td / "matches.csv.gz", dtype={"edition": str}))
    dropped = ptok[~ptok["counted"]]
    drop = set(zip(dropped["sent_id"], dropped["token_i"].astype(int)))
    m = m[[not (f in ("pueblo", "comunidad") and (s, int(i)) in drop)
           for f, s, i in zip(m["family"], m["sent_id"], m["token_i"])]]
    books = pt.groupby(["edition", "code"])["n_tokens"].sum().astype(float).rename("words").to_frame()
    books = books.join(m.groupby(["edition", "code", "family"]).size().unstack(fill_value=0), how="left")
    books = books.join(ptok[ptok["counted"]].groupby(["edition", "code"]).size().rename("people_all"), how="left")
    books = books.fillna(0).reset_index()
    books["era"] = books["edition"].map(ERA)
    return books[books["words"] > 0], ptok


def _glm(formula, data, family, groups):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    fam = sm.families.Poisson() if family == "poisson" else sm.families.Binomial()
    kw = dict(offset=np.log(data["words"])) if family == "poisson" else {}
    return smf.glm(formula, data=data, family=fam, **kw).fit(cov_type="cluster", cov_kwds={"groups": groups})


def _era_terms(res, outcome, label, model):
    rows = []
    ci = res.conf_int()
    for era in ERAS:
        if era == REF:
            continue
        term = f"C(era, Treatment('{REF}'))[T.{era}]"
        if term not in res.params:
            continue
        b, (lo, hi), p = res.params[term], ci.loc[term], res.pvalues[term]
        rows.append(dict(outcome=outcome, label=label, model=model, era=era, ratio=np.exp(b),
                         ci_low=np.exp(lo), ci_high=np.exp(hi), p=p))
    return rows


def rate_ratios(books):
    rows = []
    for fam in FAMILIES:
        if fam not in books:
            continue
        res = _glm(f"{fam} ~ C(era, Treatment('{REF}'))", books, "poisson", pd.factorize(books["code"])[0])
        rows += _era_terms(res, fam, fam, "poisson, offset log words")
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p"])
    return out


def trends(books):
    d = books[books["era"] != "NEM"].copy()
    d["decades"] = (d["edition"].astype(int) - 1960) / 10
    rows = []
    for fam in FAMILIES:
        if fam not in d:
            continue
        res = _glm(f"{fam} ~ decades", d, "poisson", pd.factorize(d["code"])[0])
        b, (lo, hi), p = res.params["decades"], res.conf_int().loc["decades"], res.pvalues["decades"]
        rows.append(dict(outcome=fam, ratio_per_decade=np.exp(b), ci_low=np.exp(lo), ci_high=np.exp(hi), p=p))
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p"])
    return out


def logits(ptok, books):
    era = lambda df: df.assign(era=df["edition"].map(ERA)).dropna(subset=["era"])
    pb = era(ptok[ptok["lemma"] == "pueblo"])
    pp = pb[pb["sense"].str.startswith("people")]
    counted = era(ptok[ptok["counted"]])
    specs = [("commons", "pueblo = common people (of all uses)", pb, (pb["sense"] == "people_commons")),
             ("nation", "pueblo = nation (of all uses)", pb, (pb["sense"] == "people_nation")),
             ("ethnic", "pueblo = ethnic people (of all uses)", pb, (pb["sense"] == "people_ethnic")),
             ("plural", "plural pueblos (of people-sense uses)", pp, (pp["number"] == "Plur")),
             ("agent", "people noun as agent (of counted mentions)", counted, (counted["role"] == "agent"))]
    det = PATHS["results"] / "visual" / "people_detect.csv"
    if det.exists():
        d = pd.read_csv(det).merge(pd.read_csv(PATHS["results"] / "visual" / "regions_sample.csv",
                                               dtype={"edition": str})[["region_id", "edition", "code"]], on="region_id")
        d = era(d)
        specs.append(("people_image", "illustration shows people (sampled)", d, (d["people_det"] != "no_people")))
    rows, shares = [], []
    for key, label, df, y in specs:
        df = df.assign(y=y.astype(int).values)
        res = _glm("y ~ C(era, Treatment('%s'))" % REF, df, "binomial", pd.factorize(df["code"])[0])
        rows += _era_terms(res, key, label, "logit")
        for e in ERAS:
            sub = df[df["era"] == e]
            if len(sub):
                shares.append(dict(outcome=key, label=label, era=e, n=len(sub), share=100 * sub["y"].mean()))
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p"])
    return out, pd.DataFrame(shares)


def sense_chi2(ptok):
    from scipy.stats import chi2_contingency
    pb = ptok[ptok["lemma"] == "pueblo"].assign(era=lambda d: d["edition"].map(ERA)).dropna(subset=["era"])
    t = pd.crosstab(pb["sense"], pb["era"])
    chi2, p, dof, _ = chi2_contingency(t)
    n = t.values.sum()
    v = np.sqrt(chi2 / (n * (min(t.shape) - 1)))
    return dict(chi2=chi2, dof=dof, p=p, n=int(n), cramers_v=v)


def run():
    books, ptok = book_counts()
    out = PATHS["results"] / "text"
    rr = rate_ratios(books)
    tr = trends(books)
    lg, shares = logits(ptok, books)
    chi = sense_chi2(ptok)
    rr.to_csv(out / "stats_rate_ratios.csv", index=False)
    tr.to_csv(out / "stats_trends.csv", index=False)
    lg.to_csv(out / "stats_logits.csv", index=False)
    shares.to_csv(out / "stats_shares.csv", index=False)
    pd.DataFrame([chi]).to_csv(out / "stats_sense_chi2.csv", index=False)
    books.groupby("era").agg(books=("code", "size"), titles=("code", "nunique"), words=("words", "sum")).to_csv(
        out / "stats_era_sizes.csv")
    log.info("wrote stats_*.csv: %d rate ratios, %d trends, %d logit terms; chi2 %.0f (V=%.2f)",
             len(rr), len(tr), len(lg), chi["chi2"], chi["cramers_v"])
    return rr, tr, lg, shares, chi
