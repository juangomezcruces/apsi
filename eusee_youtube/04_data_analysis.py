#!/usr/bin/env python3
"""
Step 4 - Data analysis: every number in the report, as CSV tables.

Reads the main dataset from step 3 and writes outputs/tables/*.csv plus a
short outputs/summary.md with the headline numbers. Works for any dataset in the
same schema, so a Facebook table with the same columns (platform, country,
published_at, topic, toxic, narrative, likes, comments, shares, views,
channel_title) can be concatenated with --extra.

Tables (report section in brackets)
  01 observations by country / platform                        [2.4, Table 2.4.1]
  02 topic distribution, overall                               [4.1, Fig 4.1.1]
  03 topic distribution by country, all vs toxic               [4.1, 5.x.1]
  04 monthly volume by topic                                   [4.1, Fig 4.1.2]
  05 toxic share: overall, year, pre/post EO, country, platform[4.2]
  06 toxicity rate within topics, by year                      [4.4]
  07 engagement toxic vs non-toxic (mean, median, ratio, test) [4.5]
  08 engagement by topic within the toxic subset               [4.5]
  09 narrative distribution: overall, by year, by country      [4.7]
  10 engagement by narrative                                   [4.7, Fig 4.7.2]
  11 country comparison at a glance                            [Table 5.1]
  12 topic growth vs toxic-share change (research question 4)  [1, 8.2]
  13 monthly toxic volume by country and topic                 [5.x.3]
  14 top-20 toxic spreaders per country, with account type     [5.x.2, 7.6]
  15 spreader-type shares per country                          [5.x.2]
  16 monthly volume / toxic share per country (time series)    [research question 1]
  17 interrupted time series around 20 Jan 2025                [research question 1]

Usage
-----
    python 04_data_analysis.py
    python 04_data_analysis.py --input other_main.csv --extra facebook_main.csv
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

ENGAGEMENT = ["views", "likes", "comments", "shares"]
T = C.TABLE_DIR


def pct(a, b):
    return np.where(b > 0, 100 * a / b, np.nan).round(2) if np.ndim(a) else (
        round(100 * a / b, 2) if b else np.nan)


def save(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df.to_csv(T / f"{name}.csv", index=False)
    return df


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load(paths: list[Path]) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df["year"] = df["published_at"].dt.year
    df["month"] = df["published_at"].dt.tz_localize(None).dt.to_period("M").astype(str)
    df["period"] = np.where(df["published_at"] >= pd.Timestamp(C.EO_DATE, tz="UTC"),
                            "post-EO", "pre-EO")
    df["toxic"] = df["toxic"].astype(str).str.lower().isin(["true", "1"])
    for col in ENGAGEMENT:
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    if "platform" not in df:
        df["platform"] = "YouTube"
    df["topic"] = pd.Categorical(df["topic"], categories=list(C.TOPICS), ordered=True)
    return df


def years(df):
    return sorted(int(y) for y in df["year"].dropna().unique())


# --------------------------------------------------------------------------
# Account typing (Section 7.6)
# --------------------------------------------------------------------------
_RULES = [(name, re.compile("|".join(pats), re.IGNORECASE)) for name, pats in C.SPREADER_RULES]
_PERSON = re.compile(r"^[A-ZÁÉÍÓÚÑ][a-záéíóúñ'’\-]+( [A-ZÁÉÍÓÚÑ][a-záéíóúñ'’\-]+){1,3}$")


def account_type(title: str, handle: str = "", description: str = "") -> str:
    title, handle, description = (str(x) if isinstance(x, str) else "" for x in (title, handle, description))
    primary = f"{title} {handle}".lower()
    for name, rx in _RULES:
        if rx.search(primary):
            return name
    if _PERSON.match(title.strip()):
        return C.SPREADER_INDIVIDUAL
    for name, rx in _RULES:              # fall back to the channel description
        if rx.search(description.lower()):
            return name
    return C.SPREADER_OTHER


# --------------------------------------------------------------------------
# Interrupted time series (segmented OLS on monthly series)
# --------------------------------------------------------------------------
def segmented_ols(y: np.ndarray, post: np.ndarray) -> dict:
    t = np.arange(len(y), dtype=float)
    t0 = np.argmax(post) if post.any() else len(y)
    X = np.column_stack([np.ones_like(t), t, post, (t - t0) * post])
    ok = ~np.isnan(y)
    X, y = X[ok], y[ok]
    if len(y) <= X.shape[1] or post[ok].sum() < 2 or (~post[ok].astype(bool)).sum() < 2:
        return {}
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    sigma2 = resid @ resid / dof
    se = np.sqrt(np.diag(sigma2 * np.linalg.pinv(X.T @ X)))
    z = beta / np.where(se > 0, se, np.nan)
    p = [math.erfc(abs(v) / math.sqrt(2)) if not np.isnan(v) else np.nan for v in z]
    names = ["intercept", "pre_trend", "level_change", "slope_change"]
    out = {}
    for n, b, s, pv in zip(names, beta, se, p):
        out[n], out[f"{n}_se"], out[f"{n}_p"] = round(b, 4), round(s, 4), round(pv, 4)
    out["n_months"] = int(len(y))
    return out


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=C.MAIN_CSV)
    ap.add_argument("--extra", type=Path, nargs="*", default=[],
                    help="more main-dataset CSVs in the same schema (e.g. Facebook)")
    ap.add_argument("--top-n", type=int, default=20, help="spreaders per country")
    args = ap.parse_args()

    T.mkdir(parents=True, exist_ok=True)
    df = load([args.input, *args.extra])
    tox = df[df["toxic"]]
    countries = [c for c in C.COUNTRY_ORDER if c in set(df["country"])]
    yrs = years(df)
    y0, y1 = (yrs[0], yrs[-1]) if len(yrs) >= 2 else (None, None)
    print(f"Main dataset: {len(df):,} rows, toxic {len(tox):,}; countries {countries}; years {yrs}")

    # 01 ---------------------------------------------------------------------
    t1 = df.pivot_table(index="country", columns="platform", values="id", aggfunc="count", fill_value=0)
    t1 = t1.reindex(countries)
    t1["Total"] = t1.sum(axis=1)
    t1.loc["Total"] = t1.sum()
    t1["share_pct"] = pct(t1["Total"], t1.loc["Total", "Total"])
    save(t1.reset_index(), "01_observations_by_country")

    # 02 ---------------------------------------------------------------------
    t2 = df["topic"].value_counts(sort=False).rename_axis("topic").reset_index(name="n")
    t2["pct"] = pct(t2["n"], t2["n"].sum())
    t2["n_toxic"] = tox["topic"].value_counts(sort=False).reindex(t2["topic"]).to_numpy()
    t2["pct_of_toxic"] = pct(t2["n_toxic"], len(tox))
    save(t2, "02_topic_distribution_overall")

    # 03 ---------------------------------------------------------------------
    rows = []
    for c in countries:
        for subset, d in (("all", df[df.country == c]), ("toxic", tox[tox.country == c])):
            vc = d["topic"].value_counts(sort=False)
            for topic, n in vc.items():
                rows.append({"country": c, "subset": subset, "topic": topic, "n": int(n),
                             "pct": pct(n, len(d))})
    save(pd.DataFrame(rows), "03_topic_distribution_by_country")

    # 04 ---------------------------------------------------------------------
    t4 = (df.pivot_table(index="month", columns="topic", values="id", aggfunc="count",
                         fill_value=0, observed=False).sort_index())
    t4["Total"] = t4.sum(axis=1)
    save(t4.reset_index(), "04_monthly_volume_by_topic")

    # 05 ---------------------------------------------------------------------
    def share_rows(d, scope, keys):
        g = d.groupby(keys, observed=True)["toxic"].agg(n="size", n_toxic="sum").reset_index()
        g.insert(0, "scope", scope)
        g["toxic_pct"] = pct(g["n_toxic"], g["n"])
        return g

    overall = pd.DataFrame([{"scope": "overall", "n": len(df), "n_toxic": int(df.toxic.sum()),
                             "toxic_pct": pct(df.toxic.sum(), len(df))}])
    t5 = pd.concat([overall,
                    share_rows(df, "year", ["year"]),
                    share_rows(df, "period", ["period"]),
                    share_rows(df, "country", ["country"]),
                    share_rows(df, "country x year", ["country", "year"]),
                    share_rows(df, "platform", ["platform"])], ignore_index=True)
    save(t5, "05_toxic_share")

    # 06 ---------------------------------------------------------------------
    t6 = share_rows(df, "topic", ["topic"]).drop(columns="scope")
    for y in yrs:
        s = share_rows(df[df.year == y], "topic", ["topic"]).set_index("topic")
        t6[f"n_{y}"] = t6["topic"].map(s["n"])
        t6[f"toxic_pct_{y}"] = t6["topic"].map(s["toxic_pct"])
    t6["share_of_all_pct"] = pct(t6["n"], len(df))
    t6["share_of_toxic_pct"] = pct(t6["n_toxic"], len(tox))
    t6["overall_toxic_pct"] = pct(len(tox), len(df))
    save(t6.sort_values("toxic_pct", ascending=False), "06_toxicity_within_topic")

    # 07 ---------------------------------------------------------------------
    try:
        from scipy.stats import mannwhitneyu
    except ImportError:
        mannwhitneyu = None
    rows = []
    for m in ENGAGEMENT:
        a, b = df.loc[df.toxic, m].dropna(), df.loc[~df.toxic, m].dropna()
        if len(a) == 0 or len(b) == 0:
            continue
        r = {"metric": m, "n_toxic": len(a), "n_non_toxic": len(b),
             "mean_toxic": round(a.mean(), 2), "mean_non_toxic": round(b.mean(), 2),
             "ratio_mean": round(a.mean() / b.mean(), 2) if b.mean() else np.nan,
             "median_toxic": a.median(), "median_non_toxic": b.median()}
        if mannwhitneyu is not None:
            r["mannwhitney_p"] = round(mannwhitneyu(a, b, alternative="two-sided").pvalue, 5)
        rows.append(r)
    t7 = save(pd.DataFrame(rows), "07_engagement_toxic_vs_nontoxic")

    # 08 ---------------------------------------------------------------------
    metrics = [m for m in ENGAGEMENT if tox[m].notna().any()]
    t8 = tox.groupby("topic", observed=True)[metrics].mean().round(2)
    t8.insert(0, "n_toxic", tox.groupby("topic", observed=True).size())
    save(t8.reset_index().sort_values(metrics[0] if "likes" not in metrics else "likes",
                                      ascending=False), "08_engagement_by_topic_toxic")

    # 09 ---------------------------------------------------------------------
    narr_order = list(C.NARRATIVES) + [C.NARRATIVE_OTHER]

    def narr_rows(d, scope, key=None):
        groups = [(None, d)] if key is None else list(d.groupby(key))
        out = []
        for g, sub in groups:
            vc = sub["narrative"].value_counts()
            for nm in narr_order:
                out.append({"scope": scope, "group": g, "narrative": nm, "n": int(vc.get(nm, 0)),
                            "pct": pct(vc.get(nm, 0), len(sub))})
        return out

    t9 = pd.DataFrame(narr_rows(tox, "overall") + narr_rows(tox, "year", "year")
                      + narr_rows(tox, "country", "country"))
    t9["group"] = t9["group"].astype("string")
    save(t9, "09_narrative_distribution")

    # 10 ---------------------------------------------------------------------
    t10 = tox.groupby("narrative")[metrics].agg(["mean", "median"]).round(2)
    t10.columns = [f"{a}_{b}" for a, b in t10.columns]
    t10.insert(0, "n", tox.groupby("narrative").size())
    save(t10.reindex([n for n in narr_order if n in t10.index]).reset_index(), "10_engagement_by_narrative")

    # 11 ---------------------------------------------------------------------
    rows = []
    for c in countries:
        d, dt = df[df.country == c], tox[tox.country == c]
        tc = d["topic"].value_counts()
        nc = dt["narrative"].value_counts()
        r = {"country": c, "relevant_n": len(d), "toxic_n": len(dt), "toxic_pct": pct(len(dt), len(d)),
             "leading_topic": tc.index[0] if len(tc) else None,
             "leading_topic_pct": pct(tc.iloc[0], len(d)) if len(tc) else None,
             "dominant_narrative": nc.index[0] if len(nc) else None,
             "dominant_narrative_pct": pct(nc.iloc[0], len(dt)) if len(nc) else None}
        if y0 is not None:
            n0, n1 = (d.year == y0).sum(), (d.year == y1).sum()
            r[f"n_{y0}"], r[f"n_{y1}"] = int(n0), int(n1)
            r["volume_change_pct"] = pct(n1 - n0, n0)
            r[f"toxic_pct_{y0}"] = pct(d.loc[d.year == y0, "toxic"].sum(), n0)
            r[f"toxic_pct_{y1}"] = pct(d.loc[d.year == y1, "toxic"].sum(), n1)
        rows.append(r)
    t11 = save(pd.DataFrame(rows), "11_country_summary")

    # 12 ---------------------------------------------------------------------
    rows = []
    if y0 is not None:
        for c in ["All countries", *countries]:
            d = df if c == "All countries" else df[df.country == c]
            for topic in C.TOPICS:
                a, b = d[(d.topic == topic) & (d.year == y0)], d[(d.topic == topic) & (d.year == y1)]
                rows.append({"country": c, "topic": topic, f"n_{y0}": len(a), f"n_{y1}": len(b),
                             "growth_pct": pct(len(b) - len(a), len(a)),
                             f"toxic_pct_{y0}": pct(a.toxic.sum(), len(a)),
                             f"toxic_pct_{y1}": pct(b.toxic.sum(), len(b))})
        t12 = pd.DataFrame(rows)
        t12["toxic_change_pp"] = (t12[f"toxic_pct_{y1}"] - t12[f"toxic_pct_{y0}"]).round(2)
        save(t12, "12_topic_growth_vs_toxicity")
    else:
        t12 = pd.DataFrame()

    # 13 ---------------------------------------------------------------------
    t13 = (tox.groupby(["country", "month", "topic"], observed=True).size()
              .rename("n_toxic").reset_index())
    save(t13, "13_monthly_toxic_by_country_topic")

    # 14 / 15 ----------------------------------------------------------------
    name_col = "channel_title" if "channel_title" in tox else "author"
    if name_col in tox:
        acc = tox.dropna(subset=[name_col])
        top = (acc.groupby(["country", name_col])
                  .agg(n_toxic=("id", "size"),
                       handle=("channel_handle", "first") if "channel_handle" in acc else (name_col, "first"),
                       description=("channel_description", "first") if "channel_description" in acc
                       else (name_col, "first"))
                  .reset_index())
        top["account_type"] = [account_type(t, h, d) for t, h, d in
                               zip(top[name_col], top["handle"], top["description"])]
        top = (top.sort_values(["country", "n_toxic"], ascending=[True, False])
                  .groupby("country").head(args.top_n))
        top["rank"] = top.groupby("country").cumcount() + 1
        save(top[["country", "rank", name_col, "handle", "n_toxic", "account_type"]],
             "14_top_toxic_spreaders")
        t15 = top.groupby(["country", "account_type"]).size().rename("n_accounts").reset_index()
        t15["pct_of_top"] = t15.groupby("country")["n_accounts"].transform(lambda s: pct(s, s.sum()))
        save(t15, "15_spreader_types")
        coverage = pct(tox[name_col].notna().sum(), len(tox))
    else:
        coverage = 0.0

    # 16 / 17 ----------------------------------------------------------------
    months = pd.period_range(C.START_DATE, C.END_DATE, freq="M").astype(str)
    post = np.array([m >= f"{C.EO_DATE:%Y-%m}" for m in months], dtype=float)
    rows, series = [], []
    for c in ["All countries", *countries]:
        d = df if c == "All countries" else df[df.country == c]
        g = d.groupby("month")["toxic"].agg(n="size", n_toxic="sum").reindex(months)
        g["n"] = g["n"].fillna(0)
        g["n_toxic"] = g["n_toxic"].fillna(0)
        share = np.where(g["n"] > 0, 100 * g["n_toxic"] / g["n"].where(g["n"] > 0), np.nan)
        series.append(pd.DataFrame({"country": c, "month": months, "n": g["n"].to_numpy(int),
                                    "n_toxic": g["n_toxic"].to_numpy(int), "toxic_pct": share.round(2),
                                    "post_eo": post.astype(bool)}))
        for outcome, y in (("volume", g["n"].to_numpy(float)),
                           ("toxic_volume", g["n_toxic"].to_numpy(float)),
                           ("toxic_share_pct", share.astype(float))):
            res = segmented_ols(y, post)
            if res:
                rows.append({"country": c, "outcome": outcome, **res})
    save(pd.concat(series, ignore_index=True), "16_monthly_series_by_country")
    t16 = save(pd.DataFrame(rows), "17_interrupted_time_series")

    # ---- summary -----------------------------------------------------------
    lines = ["# YouTube discourse on civil society: headline numbers", "",
             f"- Main dataset: **{len(df):,}** videos; toxic: **{len(tox):,}** "
             f"({pct(len(tox), len(df))}%)."]
    if y0 is not None:
        s = t5[t5.scope == "year"].set_index("year")
        lines.append(f"- Toxic share {y0} -> {y1}: {s.loc[y0, 'toxic_pct']}% -> {s.loc[y1, 'toxic_pct']}%; "
                     f"volume {int(s.loc[y0, 'n']):,} -> {int(s.loc[y1, 'n']):,} "
                     f"({pct(s.loc[y1, 'n'] - s.loc[y0, 'n'], s.loc[y0, 'n'])}%).")
    top2 = t2.sort_values("n", ascending=False).head(2)
    lines.append("- Largest topics: " + "; ".join(f"{r.topic} {r.pct}%" for r in top2.itertuples()) + ".")
    t6s = t6.dropna(subset=["toxic_pct"]).sort_values("toxic_pct", ascending=False)
    if len(t6s):
        lines.append(f"- Most toxic topic: {t6s.iloc[0].topic} ({t6s.iloc[0].toxic_pct}% toxic); "
                     f"least: {t6s.iloc[-1].topic} ({t6s.iloc[-1].toxic_pct}%).")
    for r in t7.itertuples():
        lines.append(f"- {r.metric}: toxic videos average {r.mean_toxic:,} vs {r.mean_non_toxic:,} "
                     f"(x{r.ratio_mean}); medians {r.median_toxic:,} vs {r.median_non_toxic:,}.")
    nv = t9[t9.scope == "overall"].sort_values("n", ascending=False)
    if len(nv):
        lines.append("- Narratives (toxic subset): " +
                     "; ".join(f"{r.narrative} {r.pct}%" for r in nv.itertuples() if r.n) + ".")
    try:
        country_table = t11.to_markdown(index=False)          # needs `tabulate`
    except ImportError:
        country_table = "```\n" + t11.to_string(index=False) + "\n```"
    lines += ["", "## Countries", "", country_table]
    if len(t16):
        its = t16[(t16.outcome == "toxic_share_pct")]
        lines += ["", "## Interrupted time series (monthly toxic share, %)", "",
                  "Level change = jump at January 2025; slope change = change in monthly trend after it.", ""]
        for r in its.itertuples():
            lines.append(f"- {r.country}: level change {r.level_change:+.2f} pp (p={r.level_change_p}), "
                         f"slope change {r.slope_change:+.2f} pp/month (p={r.slope_change_p}), "
                         f"{r.n_months} months")
    lines += ["", f"Account names available for {coverage}% of toxic videos.",
              "YouTube has no share counts; YouTube-only results are directional (small N)."]
    (C.OUTPUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nTables -> {T}\nSummary -> {C.OUTPUT_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
