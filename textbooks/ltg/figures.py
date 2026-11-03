"""Figures for the paper, from the report tables.

    python -m ltg figures            # -> results/figures/*.png and *.svg

Static figures (print), light surface. Forms: dot + 95% interval small multiples
for rates (one series per panel, so no legend), 100% stacked bars for
compositions (legend + direct percentage labels, since two of the categorical
colours are under 3:1 contrast on white). Colours are the validated reference
palette (blue, orange, aqua, yellow; adjacent CVD dE >= 9.1); 'not the people'
categories are neutral grey so the people categories carry the colour.
"""
import logging

import numpy as np
import pandas as pd

from .config import PATHS
from .text_analysis import edition_order, out_dir as text_dir

log = logging.getLogger("ltg.figures")

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e5e4e0", "#ffffff"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
NEUTRAL = "#c9c8c3"
ORDINAL = ["#86b6ef", "#3987e5", "#1c5cab"]          # blue 250 / 400 / 550: one / group / crowd

FAMILY_TITLES = {
    "people_all": "All people nouns", "pueblo": "pueblo (people senses)", "nacion": "nation / patria",
    "ciudadania": "citizens / society", "comunidad": "community (not classroom)", "popular": "ordinary / poor",
    "clase": "workers / peasants", "etnico": "indigenous / ethnic", "personas": "personas (generic)",
    "elite": "elites / antagonists", "agravio": "grievance", "unidad": "unity", "pluralismo": "pluralism",
}
FLAG_TITLES = {"has_elite": "with elite / antagonist", "has_agravio": "with grievance", "has_unidad": "with unity",
               "has_pluralismo": "with pluralism", "has_instituciones": "with institutions",
               "has_nosotros": "with 'nosotros'"}
SENSES = [("people_commons", "common people (vs rulers)"), ("people_nation", "nation / peoples"),
          ("people_ethnic", "ethnic / ancient people"), ("settlement", "town (not counted)")]


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 9.5,
                         "axes.titlecolor": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                         "svg.fonttype": "none"})
    return plt


def _style(ax):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def _save(fig, name):
    d = PATHS["results"] / "figures"
    d.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(d / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    log.info("wrote %s.png/.svg", d / name)


def dot_ci_panels(df, measures, titles, ylabel, name, ncols=4, suptitle=""):
    """df: edition, measure, estimate, ci_low, ci_high (long). One panel per measure."""
    plt = _plt()
    measures = [m for m in measures if m in set(df["measure"])]
    eds = edition_order(df["edition"].unique())
    nrows = int(np.ceil(len(measures) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.3 * ncols, 2.0 * nrows), sharex=True, squeeze=False)
    for ax, m in zip(axes.flat, measures):
        d = df[df["measure"] == m].set_index("edition").reindex(eds)
        x = np.arange(len(eds))
        lo = (d["estimate"] - d["ci_low"]).clip(lower=0).fillna(0)
        hi = (d["ci_high"] - d["estimate"]).clip(lower=0).fillna(0)
        ax.errorbar(x, d["estimate"], yerr=[lo, hi], fmt="o", color=SERIES[0], ecolor=SERIES[0],
                    elinewidth=1.5, capsize=0, markersize=5.5, markeredgecolor=SURFACE, markeredgewidth=1)
        ax.set_title(titles.get(m, m), loc="left")
        ax.set_xticks(x, eds, rotation=0)
        ax.set_ylim(bottom=0)
        _style(ax)
    for ax in list(axes.flat)[len(measures):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    if suptitle:
        fig.suptitle(suptitle, x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    _save(fig, name)
    plt.close(fig)


def stacked_shares(table, order, labels, colors, name, title, note=""):
    """table: index = edition, columns = categories, values = counts. 100% horizontal bars."""
    plt = _plt()
    eds = [e for e in edition_order(table.index) if e in table.index][::-1]
    share = table.reindex(eds).reindex(columns=order).fillna(0)
    n = share.sum(axis=1)
    share = 100 * share.div(n.replace(0, np.nan), axis=0)
    fig, ax = plt.subplots(figsize=(7.2, 0.45 * len(eds) + 1.3))
    left = np.zeros(len(eds))
    for cat, col in zip(order, colors):
        v = share[cat].fillna(0).values
        ax.barh(eds, v, left=left, color=col, edgecolor=SURFACE, linewidth=1.5, height=0.62, label=labels.get(cat, cat))
        for i, (l, w) in enumerate(zip(left, v)):
            if w >= 8:                                     # label only segments wide enough to hold it
                dark = col in (SERIES[0], ORDINAL[1], ORDINAL[2])
                ax.text(l + w / 2, i, f"{w:.0f}%", ha="center", va="center", fontsize=8,
                        color=SURFACE if dark else INK)
        left += v
    for i, k in enumerate(n.values):
        ax.text(101, i, f"n={int(k)}", va="center", fontsize=8, color=MUTED)
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)
    ax.legend(ncols=min(4, len(order)), loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, fontsize=8,
              handlelength=1.2, columnspacing=1.2)
    ax.set_title(title, loc="left", pad=24, fontsize=10.5)
    if note:
        fig.text(0.01, -0.02, note, fontsize=7.5, color=MUTED, ha="left", va="top")
    fig.tight_layout()
    _save(fig, name)
    plt.close(fig)


def run():
    td = text_dir()
    made = []
    rates = td / "rates_per_10k.csv"
    if rates.exists():
        r = pd.read_csv(rates, dtype={"edition": str})
        dot_ci_panels(r, ["people_all", "pueblo", "nacion", "ciudadania", "comunidad", "popular", "clase", "etnico"],
                      FAMILY_TITLES, "per 10,000 words", "people_rates",
                      suptitle="How often the books talk about the people (per 10,000 words, 95% CI over books)")
        dot_ci_panels(r, ["elite", "agravio", "unidad", "pluralismo"], FAMILY_TITLES, "per 10,000 words",
                      "relation_rates", suptitle="Antagonists, grievance, unity and pluralism (per 10,000 words)")
        made += ["people_rates", "relation_rates"]
    shares = td / "people_sentence_shares.csv"
    if shares.exists():
        s = pd.read_csv(shares, dtype={"edition": str})
        dot_ci_panels(s, list(FLAG_TITLES), FLAG_TITLES, "% of people-talk sentences", "people_talk_context",
                      ncols=3, suptitle="What sentences about the people also contain (%, 95% CI over books)")
        made.append("people_talk_context")
    ptok_path = td / "people_tokens.csv.gz"
    if ptok_path.exists():
        ptok = pd.read_csv(ptok_path, dtype={"edition": str}, keep_default_na=False)
        if "corpus" in ptok:
            ptok = ptok[ptok["corpus"].isin(["", "national"])]
        pb = ptok[ptok["lemma"] == "pueblo"]
        if pb["sense"].isin([s for s, _ in SENSES]).any():
            t = pd.crosstab(pb["edition"], pb["sense"])
            stacked_shares(t, [s for s, _ in SENSES], dict(SENSES), SERIES[:3] + [NEUTRAL], "pueblo_senses",
                           "What 'pueblo' means, by edition (% of uses)",
                           note="Senses from the BETO classifier (`ltg senses`); validate against hand coding before citing.")
            made.append("pueblo_senses")
    det = PATHS["results"] / "visual" / "people_detect.csv"
    if det.exists():
        d = pd.read_csv(det).merge(pd.read_csv(PATHS["results"] / "visual" / "regions_sample.csv",
                                               dtype={"edition": str})[["region_id", "edition"]], on="region_id")
        t = pd.crosstab(d["edition"], d["people_det"])
        stacked_shares(t, ["no_people", "one_person", "small_group", "crowd"],
                       {"no_people": "no people", "one_person": "one person", "small_group": "2-5 people",
                        "crowd": "6+ people"}, [NEUTRAL] + ORDINAL, "illustration_people",
                       "People in the illustrations (% of sampled illustrations)",
                       note="Person detector (Faster R-CNN, COCO) on a random sample of illustrations per edition; validate before citing.")
        made.append("illustration_people")
    log.info("figures: %s", made)
    return made
