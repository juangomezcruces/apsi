#!/usr/bin/env python3
"""
Step 5 - Figures. Reads only the CSV tables written by step 4, so figures can
be restyled without re-running any analysis.

Figures (report numbering)
  fig_4_1_1  topic distribution, overall
  fig_4_1_2  monthly volume stacked by topic, EO line
  fig_4_2_1  toxic vs non-toxic: overall and by year
  fig_4_2_2  toxic share by country and year
  fig_4_4_1  toxicity rate within each topic, overall-average line
  fig_4_5_1  mean engagement, toxic vs non-toxic (one panel per metric)
  fig_4_7_1  narrative distribution (toxic subset)
  fig_4_7_2  engagement by narrative (one panel per metric)
  fig_5_<country>_1  topics: all vs toxic
  fig_5_<country>_2  top toxic spreaders by account type
  fig_5_<country>_3  monthly toxic volume by topic, EO line
  fig_8_1    topic growth vs change in toxic share (research question 4)
  fig_8_2    monthly toxic share per country, EO line (research question 1)

Usage
-----
    python 05_make_figures.py
    python 05_make_figures.py --formats png svg --theme dark
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402

import config as C  # noqa: E402

T = C.TABLE_DIR

# --------------------------------------------------------------------------
# Palette (validated categorical order; light and dark steps of the same hues)
# --------------------------------------------------------------------------
THEMES = {
    "light": dict(surface="#ffffff", ink="#191919", ink2="#666666", grid="#e7e7e7", axis="#c2c2c2",
                  other="#9c9c9c",
                  series=["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17",
                          "#6a4ce2", "#ec4647"]),
    "dark": dict(surface="#191919", ink="#efefef", ink2="#9c9c9c", grid="#303030", axis="#474747",
                 other="#666666",
                 series=["#338ced", "#e76624", "#00a888", "#d28e05", "#e24d8b", "#0aac23",
                         "#9887ef", "#eb6263"]),
}
P: dict = {}


def setup(theme: str) -> None:
    P.update(THEMES[theme])
    s = P["series"]
    P["all"], P["toxic"], P["non_toxic"] = s[0], s[7], s[0]
    P["year"] = {2024: s[0], 2025: s[1]}
    P["topic"] = dict(zip(C.TOPICS, s))
    P["narrative"] = {**dict(zip(C.NARRATIVES, s[:5])), C.NARRATIVE_OTHER: P["other"]}
    types = [n for n, _ in C.SPREADER_RULES] + [C.SPREADER_INDIVIDUAL]
    P["account"] = {**dict(zip(types, s)), C.SPREADER_OTHER: P["other"]}
    plt.rcParams.update({
        "figure.facecolor": P["surface"], "axes.facecolor": P["surface"],
        "savefig.facecolor": P["surface"], "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 9, "text.color": P["ink"], "axes.labelcolor": P["ink2"],
        "axes.edgecolor": P["axis"], "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.titlecolor": P["ink"], "axes.titlepad": 10,
        "xtick.color": P["ink2"], "ytick.color": P["ink2"], "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.grid": True, "grid.color": P["grid"], "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "legend.frameon": False, "legend.fontsize": 8, "legend.labelcolor": P["ink2"],
        "patch.linewidth": 0, "axes.axisbelow": True,
    })


def read(name: str) -> pd.DataFrame | None:
    path = T / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else None


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def finish(fig, ax, name: str, args, caption: str | None = None, legend: bool = False,
           legend_cols: int = 4) -> None:
    if legend:
        # legend sits in its own band between the title and the plot area
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc="lower left", bbox_to_anchor=(0, 1.01), ncol=legend_cols,
                  handlelength=1, handleheight=1, columnspacing=1.2, borderaxespad=0)
        rows = -(-len(labels) // legend_cols)
        ax.set_title(ax.get_title(loc="left"), loc="left", pad=10 + 15 * rows)
    if caption:
        fig.text(0.01, 0.01, caption, fontsize=7, color=P["ink2"], ha="left", va="bottom", wrap=True)
    args.out.mkdir(parents=True, exist_ok=True)
    for fmt in args.formats:
        fig.savefig(args.out / f"{name}.{fmt}", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name)


def gap():
    """2px surface-coloured separation between adjacent / stacked fills."""
    return dict(edgecolor=P["surface"], linewidth=0.8)


def eo_line(ax, months: list[str]) -> None:
    eo = f"{C.EO_DATE:%Y-%m}"
    if eo in months:
        x = months.index(eo) - 0.5 + C.EO_DATE.day / 31
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + (hi - lo) * 0.14)          # headroom so the label clears the data
        ax.axvline(x, color=P["ink2"], linestyle=(0, (4, 3)), linewidth=1, zorder=5)
        ax.text(x + 0.3, hi + (hi - lo) * 0.13, "EO 14169, 20 Jan 2025", fontsize=7,
                color=P["ink2"], va="top")


def month_ticks(ax, months: list[str]) -> None:
    idx = [i for i, m in enumerate(months) if m.endswith(("-01", "-04", "-07", "-10"))]
    ax.set_xticks(idx)
    ax.set_xticklabels([pd.Period(months[i]).strftime("%b\n%Y") if months[i].endswith("-01")
                        else pd.Period(months[i]).strftime("%b") for i in idx])
    ax.grid(axis="x", visible=False)
    ax.set_xlim(-0.6, len(months) - 0.4)


def hbar_labels(ax, bars, values, fmt="{:.1f}%"):
    xmax = ax.get_xlim()[1]
    for b, v in zip(bars, values):
        if pd.notna(v):
            ax.text(b.get_width() + xmax * 0.01, b.get_y() + b.get_height() / 2, fmt.format(v),
                    va="center", fontsize=7.5, color=P["ink"])


def repel_labels(fig, texts, max_iter: int = 200) -> None:
    """Nudge annotation labels downward until their boxes stop overlapping.

    Works in display pixels from the label sizes and anchor points, then sets
    each label's offset once (matplotlib's extents only refresh on redraw).
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    px_per_pt = fig.dpi / 72
    boxes = []   # [x0, y0, w, h] in pixels
    for t in texts:
        bb = t.get_window_extent(renderer)
        ax_ = t.axes
        x, y = ax_.transData.transform(t.xy)
        ox, oy = t.xyann
        boxes.append([x + ox * px_per_pt, y + oy * px_per_pt, bb.width, bb.height])
    for _ in range(max_iter):
        moved = False
        for i in range(len(boxes)):
            for j in range(i):
                xi, yi, wi, hi = boxes[i]
                xj, yj, wj, hj = boxes[j]
                if xi < xj + wj and xj < xi + wi and yi < yj + hj and yj < yi + hi:
                    boxes[i][1] = yj - hi - 1          # move just below the label it hits
                    moved = True
        if not moved:
            break
    for t, (bx, by, _, _) in zip(texts, boxes):
        x, y = t.axes.transData.transform(t.xy)
        t.xyann = ((bx - x) / px_per_pt, (by - y) / px_per_pt)

def stacked_monthly(ax, wide: pd.DataFrame, months: list[str]) -> None:
    bottom = np.zeros(len(months))
    for topic in C.TOPICS:
        if topic in wide:
            vals = wide[topic].reindex(months).fillna(0).to_numpy()
            if vals.sum():
                ax.bar(range(len(months)), vals, 0.8, bottom=bottom, color=P["topic"][topic],
                       label=topic, **gap())
                bottom += vals
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    month_ticks(ax, months)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def fig_topics_overall(args):
    t = read("02_topic_distribution_overall")
    if t is None:
        return
    t = t.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    bars = ax.barh(t["topic"], t["pct"], 0.62, color=P["all"])
    ax.set_xlim(0, t["pct"].max() * 1.15)
    hbar_labels(ax, bars, t["pct"])
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_title(f"Topic distribution (N = {t['n'].sum():,})")
    finish(fig, ax, "fig_4_1_1_topic_distribution", args)


def fig_monthly_volume(args):
    t = read("04_monthly_volume_by_topic")
    if t is None:
        return
    months = list(pd.period_range(C.START_DATE, C.END_DATE, freq="M").astype(str))
    fig, ax = plt.subplots(figsize=(9, 4.6))
    stacked_monthly(ax, t.set_index("month"), months)
    eo_line(ax, months)
    ax.set_ylabel("Videos per month")
    ax.set_title("Monthly volume by topic")
    finish(fig, ax, "fig_4_1_2_monthly_volume_by_topic", args, legend=True)


def fig_toxic_share(args):
    t = read("05_toxic_share")
    if t is None:
        return
    sel = pd.concat([t[t.scope == "overall"].assign(label="Overall"),
                     t[t.scope == "year"].assign(label=lambda d: d["year"].astype(float).astype(int).astype(str))])
    sel = sel.iloc[::-1]
    labels, tox, ns = list(sel["label"]), sel["toxic_pct"].to_numpy(float), sel["n"].astype(int).tolist()
    fig, ax = plt.subplots(figsize=(7.5, 2.6))
    ax.barh(labels, tox, 0.55, color=P["toxic"], label="Toxic", **gap())
    ax.barh(labels, 100 - tox, 0.55, left=tox, color=P["non_toxic"], label="Non-toxic", **gap())
    for i, (v, n) in enumerate(zip(tox, ns)):
        ax.text(v / 2, i, f"{v:.1f}%", ha="center", va="center", color="#ffffff", fontsize=8,
                fontweight="bold")
        ax.text(101, i, f"n = {n:,}", va="center", fontsize=7.5, color=P["ink2"])
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.grid(axis="y", visible=False)
    ax.set_title("Toxic vs non-toxic videos")
    finish(fig, ax, "fig_4_2_1_toxic_share", args, legend=True)

    cy = t[t.scope == "country x year"].copy()
    if cy.empty:
        return
    cy["year"] = cy["year"].astype(float).astype(int)
    countries = [c for c in C.COUNTRY_ORDER if c in set(cy["country"])]
    yrs = sorted(cy["year"].unique())
    w = 0.8 / len(yrs)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for j, y in enumerate(yrs):
        sub = cy[cy.year == y].set_index("country").reindex(countries)
        x = np.arange(len(countries)) + (j - (len(yrs) - 1) / 2) * w
        bars = ax.bar(x, sub["toxic_pct"], w * 0.92, color=P["year"].get(y, P["series"][j]), label=str(y))
        for b, v in zip(bars, sub["toxic_pct"]):
            if pd.notna(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.0f}%", ha="center", fontsize=7.5)
    ax.set_xticks(range(len(countries)))
    ax.set_xticklabels(countries)
    ax.grid(axis="x", visible=False)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_ylim(0, cy["toxic_pct"].max() * 1.2)
    ax.set_title("Toxic share by country and year")
    finish(fig, ax, "fig_4_2_2_toxic_share_by_country_year", args, legend=True)


def fig_toxicity_within_topic(args):
    t = read("06_toxicity_within_topic")
    if t is None:
        return
    t = t.dropna(subset=["toxic_pct"]).sort_values("toxic_pct")
    overall = t["overall_toxic_pct"].iloc[0]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    bars = ax.barh(t["topic"], t["toxic_pct"], 0.62, color=P["toxic"])
    ax.set_xlim(0, max(t["toxic_pct"].max(), overall) * 1.18)
    hbar_labels(ax, bars, t["toxic_pct"])
    ax.axvline(overall, color=P["ink2"], linestyle=(0, (4, 3)), linewidth=1)
    ax.text(overall, len(t) - 0.35, f" overall {overall:.1f}%", fontsize=7.5, color=P["ink2"])
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_title("Share of toxic videos within each topic")
    finish(fig, ax, "fig_4_4_1_toxicity_within_topic", args)


def fig_engagement(args):
    t = read("07_engagement_toxic_vs_nontoxic")
    if t is None or t.empty:
        return
    fig, axes = plt.subplots(1, len(t), figsize=(2.4 * len(t) + 0.6, 3.2), squeeze=False)
    for ax, r in zip(axes[0], t.itertuples()):
        vals = [r.mean_toxic, r.mean_non_toxic]
        bars = ax.bar(["Toxic", "Non-toxic"], vals, 0.6, color=[P["toxic"], P["non_toxic"]])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v * 1.02, f"{v:,.1f}", ha="center", fontsize=7.5)
        ax.set_ylim(0, max(vals) * 1.2)
        ax.set_title(f"{r.metric.capitalize()}  (×{r.ratio_mean})", fontsize=9)
        ax.grid(axis="x", visible=False)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    fig.suptitle("Mean engagement per video, toxic vs non-toxic", x=0.01, ha="left",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    finish(fig, axes[0][0], "fig_4_5_1_engagement_toxic_vs_nontoxic", args,
           caption="Means are pulled up by a few viral videos; medians are in table 07. "
                   "YouTube exposes no share counts.")


def fig_narratives(args):
    t = read("09_narrative_distribution")
    if t is None:
        return
    o = t[t.scope == "overall"]
    o = o[o.n > 0].sort_values("n")
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    bars = ax.barh(o["narrative"], o["pct"], 0.6, color=P["toxic"])
    ax.set_xlim(0, o["pct"].max() * 1.2)
    hbar_labels(ax, bars, o["pct"])
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_title(f"Negative narratives in the toxic subset (N = {o['n'].sum():,})")
    finish(fig, ax, "fig_4_7_1_narrative_distribution", args)

    e = read("10_engagement_by_narrative")
    if e is None or e.empty:
        return
    metrics = [c[:-5] for c in e.columns if c.endswith("_mean")]
    e = e.set_index("narrative")
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.3 * len(metrics), 3.4), squeeze=False, sharey=True)
    for ax, m in zip(axes[0], metrics):
        s = e[f"{m}_mean"][::-1]
        bars = ax.barh(s.index, s.values, 0.6, color=P["toxic"])
        ax.set_xlim(0, s.max() * 1.3 if s.max() > 0 else 1)
        hbar_labels(ax, bars, s.values, "{:,.1f}")
        ax.set_title(f"Mean {m}", fontsize=9)
        ax.grid(axis="y", visible=False)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    fig.suptitle("Engagement per video by narrative (toxic subset)", x=0.01, ha="left",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    finish(fig, axes[0][0], "fig_4_7_2_engagement_by_narrative", args)


def fig_country(args):
    topics = read("03_topic_distribution_by_country")
    monthly = read("13_monthly_toxic_by_country_topic")
    spreaders = read("14_top_toxic_spreaders")
    months = list(pd.period_range(C.START_DATE, C.END_DATE, freq="M").astype(str))
    for c in C.COUNTRY_ORDER:
        s = slug(c)
        if topics is not None and (topics.country == c).any():
            d = topics[topics.country == c].pivot(index="topic", columns="subset", values="pct")
            d = d.reindex(list(C.TOPICS)[::-1]).fillna(0)
            n_all = topics[(topics.country == c) & (topics.subset == "all")]["n"].sum()
            n_tox = topics[(topics.country == c) & (topics.subset == "toxic")]["n"].sum()
            y = np.arange(len(d))
            fig, ax = plt.subplots(figsize=(7.5, 4.4))
            b1 = ax.barh(y + 0.2, d.get("all", 0), 0.38, color=P["all"], label=f"All videos (n = {n_all:,})")
            b2 = ax.barh(y - 0.2, d.get("toxic", 0), 0.38, color=P["toxic"], label=f"Toxic videos (n = {n_tox:,})")
            ax.set_yticks(y)
            ax.set_yticklabels(d.index)
            ax.set_xlim(0, d.to_numpy().max() * 1.18)
            hbar_labels(ax, b1, d.get("all", pd.Series(0, index=d.index)))
            hbar_labels(ax, b2, d.get("toxic", pd.Series(0, index=d.index)))
            ax.grid(axis="y", visible=False)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
            ax.set_title(f"{c}: topic distribution, all vs toxic")
            finish(fig, ax, f"fig_5_{s}_1_topics_all_vs_toxic", args, legend=True, legend_cols=2)

        if spreaders is not None and (spreaders.country == c).any():
            d = spreaders[spreaders.country == c].sort_values("rank", ascending=False)
            name_col = [col for col in d.columns if col not in
                        ("country", "rank", "handle", "n_toxic", "account_type")][0]
            fig, ax = plt.subplots(figsize=(7.5, 0.26 * len(d) + 1.4))
            seen = set()
            for i, r in enumerate(d.itertuples()):
                lab = r.account_type if r.account_type not in seen else None
                seen.add(r.account_type)
                ax.barh(i, r.n_toxic, 0.7, color=P["account"].get(r.account_type, P["other"]), label=lab)
                ax.text(r.n_toxic + d["n_toxic"].max() * 0.01, i, str(r.n_toxic), va="center", fontsize=7)
            ax.set_yticks(range(len(d)))
            ax.set_yticklabels([str(x)[:40] for x in d[name_col]], fontsize=7.5)
            ax.set_xlim(0, d["n_toxic"].max() * 1.1)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(axis="y", visible=False)
            ax.set_xlabel("Toxic videos")
            ax.set_title(f"{c}: most active toxic-spreading channels")
            finish(fig, ax, f"fig_5_{s}_2_top_spreaders", args, legend=True, legend_cols=3)

        if monthly is not None and (monthly.country == c).any():
            d = monthly[monthly.country == c].pivot_table(index="month", columns="topic",
                                                          values="n_toxic", aggfunc="sum")
            fig, ax = plt.subplots(figsize=(9, 4.6))
            stacked_monthly(ax, d, months)
            eo_line(ax, months)
            ax.set_ylabel("Toxic videos per month")
            ax.set_title(f"{c}: monthly toxic videos by topic (N = {int(np.nansum(d.to_numpy())):,})")
            finish(fig, ax, f"fig_5_{s}_3_monthly_toxic_by_topic", args, legend=True)


def fig_growth_vs_toxicity(args):
    t = read("12_topic_growth_vs_toxicity")
    if t is None or t.empty:
        return
    d = t[t.country == "All countries"].dropna(subset=["growth_pct", "toxic_change_pp"])
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ncol = [c for c in d.columns if c.startswith("n_")][-1]
    size = 30 + 250 * d[ncol] / d[ncol].max()
    ax.scatter(d["growth_pct"], d["toxic_change_pp"], s=size, color=P["all"], alpha=0.85,
               edgecolor=P["surface"], linewidth=1.5, zorder=3)
    texts = [ax.annotate(r.topic, (r.growth_pct, r.toxic_change_pp), xytext=(6, 4),
                         textcoords="offset points", fontsize=7.5, color=P["ink"])
             for r in d.sort_values(ncol, ascending=False).itertuples()]
    repel_labels(fig, texts)
    ax.axhline(0, color=P["axis"], linewidth=1)
    ax.axvline(0, color=P["axis"], linewidth=1)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.0f}%"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.0f} pp"))
    ax.set_xlabel("Growth in volume, 2024 → 2025")
    ax.set_ylabel("Change in toxic share")
    ax.set_title("Does a topic's growth go with a rise in its toxic share?")
    finish(fig, ax, "fig_8_1_topic_growth_vs_toxicity", args,
           caption="Each dot is a topic; dot area is proportional to 2025 volume.")


def fig_time_series(args):
    t = read("16_monthly_series_by_country")
    if t is None:
        return
    months = list(pd.period_range(C.START_DATE, C.END_DATE, freq="M").astype(str))
    countries = [c for c in C.COUNTRY_ORDER if c in set(t["country"])]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True, squeeze=False)
    for ax, c in zip(axes.flat, countries):
        d = t[t.country == c].set_index("month").reindex(months)
        y = d["toxic_pct"].to_numpy(float)
        x = np.arange(len(months))
        ax.plot(x, y, color=P["toxic"], linewidth=2, marker="o", markersize=3.5, zorder=3)
        ax.set_ylim(0, 100)
        month_ticks(ax, months)
        eo_line(ax, months)
        ax.set_title(f"{c}  (n = {int(d['n'].sum()):,})", fontsize=9)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    for ax in list(axes.flat)[len(countries):]:
        ax.set_visible(False)
    fig.suptitle("Monthly toxic share of videos, by country", x=0.01, ha="left", fontweight="bold",
                 fontsize=11)
    fig.tight_layout()
    finish(fig, axes[0][0], "fig_8_2_monthly_toxic_share_by_country", args,
           caption="Months with few videos swing widely; see table 17 for the segmented-regression test.")


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=C.FIGURE_DIR)
    ap.add_argument("--formats", nargs="+", default=["png"], help="png, svg, pdf ...")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--theme", choices=list(THEMES), default="light")
    args = ap.parse_args()
    if not T.exists():
        raise SystemExit(f"{T} not found - run 04_data_analysis.py first")
    setup(args.theme)
    for f in (fig_topics_overall, fig_monthly_volume, fig_toxic_share, fig_toxicity_within_topic,
              fig_engagement, fig_narratives, fig_country, fig_growth_vs_toxicity, fig_time_series):
        f(args)
    print(f"Figures -> {args.out}")


if __name__ == "__main__":
    main()
