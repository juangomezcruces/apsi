"""Agreement between hand codes and model labels, weighted to the books.

    python -m ltg validate --kind sense          # pueblo, and comunidad / asamblea, separately
    python -m ltg validate --kind role
    python -m ltg validate --kind visual [--dim people|activity|figure|setting|medium]
    python -m ltg validate --kind seed           # your codes against Claude's, on the 100-seed check
    python -m ltg validate --strata              # recount the strata from data/ (needs the corpus)

The validation samples are stratified: up to n items from every era x predicted-label
cell (pueblo: era x sense; comunidad / asamblea: era x word x sense; roles: era x role;
illustrations: era x detector people label). Rare labels are over-represented, so an
unweighted accuracy describes the sample, not the books. Each coded item is weighted
by N_h / n_h, the number of corpus items in its cell over the items coded in that cell.
Coding only part of a sample is therefore fine; cells with nothing coded yet are listed,
and the estimates cover the rest.

Items marked "left out" count as coded (they enter n_h) but have no code, so the
estimates are for the items a coder could decide, and the share left out is reported.
Standard errors use the stratified formula with a finite-population correction.
The seed check is a simple random sample of the 451 seeds, so it is not weighted.

Hand-coded files are read from results/validation/ (the app's exports); strata sizes
from results/validation/strata.csv. Writes results/validation/validate_<kind>.md.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ERA

log = logging.getLogger("ltg.validation")
VDIR = Path(__file__).resolve().parents[1] / "results" / "validation"
ERAS = ["1960s", "1970s-80s", "1988-93", "2008-14", "2018-22", "NEM"]
VISUAL_DIMS = ["people", "activity", "figure", "setting", "medium"]
CONDITIONAL = {"activity", "figure", "setting"}          # coded only when people are shown


# --------------------------------------------------------------------------- strata
def count_strata():
    """Corpus size of every sampling cell, from the files the samples were drawn from."""
    from .config import PATHS
    from .text_analysis import _in_corpus, out_dir
    pt = _in_corpus(pd.read_csv(out_dir() / "people_tokens.csv.gz", dtype={"edition": str}, keep_default_na=False,
                                usecols=["edition", "lemma", "sense", "role", "counted", "corpus"]), "national")
    pt["era"] = pt["edition"].map(ERA).fillna(pt["edition"])
    parts = []
    pb = pt[pt["lemma"] == "pueblo"]
    parts.append(pb.groupby(["era", "sense"]).size().rename("N").reset_index().assign(kind="sense", lemma="pueblo"))
    cm = pt[pt["lemma"].isin(["comunidad", "asamblea"])]
    parts.append(cm.groupby(["era", "lemma", "sense"]).size().rename("N").reset_index().assign(kind="sense"))
    ro = pt[pt["counted"].astype(str) == "True"]
    parts.append(ro.groupby(["era", "role"]).size().rename("N").reset_index().assign(kind="role", lemma=""))
    v = PATHS["results"] / "visual"
    reg = pd.read_csv(v / "regions_sample.csv", dtype={"edition": str})[["region_id", "edition"]]
    reg = reg.merge(pd.read_csv(v / "people_detect.csv")[["region_id", "people_det"]], on="region_id")
    reg["era"] = reg["edition"].map(ERA).fillna(reg["edition"])
    parts.append(reg.groupby(["era", "people_det"]).size().rename("N").reset_index().assign(kind="visual", lemma=""))
    out = pd.concat([p.rename(columns={"sense": "pred", "role": "pred", "people_det": "pred"}) for p in parts])
    return out[["kind", "era", "lemma", "pred", "N"]].reset_index(drop=True)


def write_strata(path=VDIR / "strata.csv"):
    s = count_strata()
    s.to_csv(path, index=False)
    log.info("wrote %s (%d cells)", path, len(s))
    return s


def load_strata(path=VDIR / "strata.csv"):
    if path.exists():
        return pd.read_csv(path, keep_default_na=False, dtype={"N": int})
    log.warning("%s not found: trying to count the strata from data/", path)
    return count_strata()


# --------------------------------------------------------------------------- estimation
def agreement(y, p, w=None):
    """Accuracy, Cohen's kappa, per-label precision / recall and the confusion matrix,
    from (optionally weighted) gold and predicted labels."""
    y, p = pd.Series(y, dtype=str).reset_index(drop=True), pd.Series(p, dtype=str).reset_index(drop=True)
    w = pd.Series(np.ones(len(y)) if w is None else np.asarray(w, float))
    labels = sorted(set(y) | set(p))
    cm = (pd.DataFrame({"gold": y, "pred": p, "w": w}).pivot_table(index="gold", columns="pred", values="w", aggfunc="sum")
          .reindex(index=labels, columns=labels).fillna(0.0))
    tot = cm.values.sum()
    diag = np.diag(cm.values)
    acc = diag.sum() / tot
    pe = float((cm.sum(axis=1).values / tot) @ (cm.sum(axis=0).values / tot))
    kappa = (acc - pe) / (1 - pe) if pe < 1 else np.nan
    per = pd.DataFrame(dict(label=labels, n_gold=[int((y == l).sum()) for l in labels],
                            n_pred=[int((p == l).sum()) for l in labels],
                            precision=diag / np.where(cm.sum(axis=0).values > 0, cm.sum(axis=0).values, np.nan),
                            recall=diag / np.where(cm.sum(axis=1).values > 0, cm.sum(axis=1).values, np.nan)))
    return float(acc), float(kappa), per, cm


def stratified_se(correct, cell, N_cell):
    """SE of a stratified mean of 0/1 `correct` (rows with a code), cells `cell`, sizes `N_cell`."""
    d = pd.DataFrame(dict(c=np.asarray(correct, float), h=np.asarray(cell), N=np.asarray(N_cell, float)))
    g = d.groupby("h").agg(p=("c", "mean"), n=("c", "size"), N=("N", "first"))
    W = g["N"] / g["N"].sum()
    s2 = np.where(g["n"] > 1, g["p"] * (1 - g["p"]) * g["n"] / (g["n"] - 1).clip(lower=1), 0.25)
    fpc = (1 - g["n"] / g["N"]).clip(lower=0)
    return float(np.sqrt((W ** 2 * s2 * fpc / g["n"]).sum()))


def _weights(df, strata, keys):
    """N_h / n_h for every coded row (left-out rows included), and the uncoded cells."""
    coded = df[df["_coded"]]
    n_h = coded.groupby(keys).size().rename("n_h")
    cells = strata.set_index(keys)["N"].rename("N_h")
    df = df.join(n_h, on=keys).join(cells, on=keys)
    if df.loc[df["_coded"], "N_h"].isna().any():
        bad = df.loc[df["_coded"] & df["N_h"].isna(), keys].drop_duplicates().to_dict("records")
        raise SystemExit(f"cells missing from strata.csv: {bad[:5]}; run `python -m ltg validate --strata`")
    df["w"] = np.where(df["_coded"], df["N_h"] / df["n_h"], 0.0)
    df["_cell"] = df[keys].astype(str).agg("|".join, axis=1)
    sampled = set(map(tuple, coded[keys].astype(str).values))
    empty = strata[[tuple(r) not in sampled for r in strata[keys].astype(str).values]]
    return df, empty


def _fmt(x, pct=True, d=0):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{100 * x:.{d}f}" if pct else f"{x:.2f}"


def _md(df):
    cols = list(df.columns)
    rows = [cols, ["---"] * len(cols)] + [[str(v) for v in r] for r in df.values]
    return "\n".join("| " + " | ".join(r) + " |" for r in rows)


def evaluate(df, gold, pred, strata=None, keys=("era", "pred"), title="", n_pop=None):
    """df: one row per sampled item with columns `gold`, `pred`, era, _coded, _out, (lemma).
    strata: kind's rows of strata.csv, or None for a simple random sample (unweighted)
    of a population of n_pop items."""
    keys = list(keys)
    df = df.copy()
    df["_coded"] = df["_coded"].astype(bool)
    if strata is not None:
        df, empty = _weights(df, strata, keys)
    else:
        df["w"] = np.where(df["_coded"], 1.0, 0.0)
        df["_cell"] = "all"
        df["N_h"] = float(n_pop or len(df))
        empty = pd.DataFrame()
    coded = df[df["_coded"]]
    rated = coded[coded[gold].astype(str).str.strip() != ""]
    if rated.empty:
        return dict(title=title, n_coded=len(coded), md=f"## {title}\n\nNothing coded yet.\n")
    y, p = rated[gold].astype(str).str.strip(), rated[pred].astype(str).str.strip()
    acc, kappa, per, cm = agreement(y, p, rated["w"])
    se = stratified_se((y == p).values, rated["_cell"], rated["N_h"])
    out_flag = coded["_out"].astype(bool) if "_out" in coded else pd.Series(False, index=coded.index)
    left_out = coded.loc[out_flag, "w"].sum() / coded["w"].sum()
    n_out, n_other = int(out_flag.sum()), len(coded) - len(rated) - int(out_flag.sum())
    weighted = strata is not None

    lines = [f"## {title}", ""]
    covered = f" Cells with nothing coded yet: {len(empty)} ({empty['N'].sum():,} corpus items not covered)." if len(empty) else ""
    other = f", {n_other} without people" if n_other else ""
    lines += [f"Coded {len(coded)} of {len(df)} sampled items ({n_out} left out{other}). "
              + (f"Weighted to {coded.drop_duplicates('_cell')['N_h'].sum():,.0f} corpus items in the coded cells.{covered}"
                 if weighted else "Simple random sample: not weighted."), ""]
    lines += ["| | estimate | 95% CI |", "|---|---|---|",
              f"| accuracy | {_fmt(acc, d=1)}% | {_fmt(max(0, acc - 1.96 * se), d=1)}–{_fmt(min(1, acc + 1.96 * se), d=1)}% |",
              f"| Cohen's kappa | {kappa:.2f} | |",
              f"| left out (share of items) | {_fmt(left_out, d=1)}% | |", ""]

    per_t = per.assign(precision=per["precision"].map(lambda v: _fmt(v)), recall=per["recall"].map(lambda v: _fmt(v)))
    lines += ["**By label** (precision: of items the model gave this label, % right; recall: of items that "
              "truly have it, % the model found; both in %, weighted)", "", _md(per_t), ""]

    era_rows, share_rows = [], []
    present = set(df["era"])
    eras = [e for e in ERAS if e in present] + sorted(present - set(ERAS))
    for era in eras:
        r = rated[rated["era"] == era]
        if r.empty:
            era_rows.append([era, 0, "–", "–", "–"])
            continue
        ry, rp = r[gold].astype(str).str.strip(), r[pred].astype(str).str.strip()
        a, k, _, _ = agreement(ry, rp, r["w"])
        s = stratified_se((ry == rp).values, r["_cell"], r["N_h"])
        era_rows.append([era, len(r), _fmt(a, d=1), f"{_fmt(max(0, a - 1.96 * s), d=1)}–{_fmt(min(1, a + 1.96 * s), d=1)}",
                         f"{k:.2f}" if not np.isnan(k) else "–"])
        model = r.groupby(rp.values)["w"].sum() / r["w"].sum()
        hand = r.groupby(ry.values)["w"].sum() / r["w"].sum()
        for lab in sorted(set(model.index) | set(hand.index)):
            share_rows.append(dict(era=era, label=lab, model=model.get(lab, 0.0), hand=hand.get(lab, 0.0)))
    lines += ["**By era** (accuracy in %, weighted)", "",
              _md(pd.DataFrame(era_rows, columns=["era", "n coded", "accuracy", "95% CI", "kappa"])), ""]
    shares = pd.DataFrame(share_rows)
    if not shares.empty:
        wide = shares.assign(cell=[f"{_fmt(m)} → {_fmt(h)}" for m, h in zip(shares["model"], shares["hand"])]).pivot(
            index="label", columns="era", values="cell").reindex(columns=[e for e in eras if e in set(shares["era"])]).fillna("–")
        lines += ["**Label shares by era, model → hand-coded** (% of the coded items in the era, weighted to the "
                  "books: the hand-coded share estimates what the era looks like without model error)", "",
                  _md(wide.reset_index()), ""]
    cm_t = (100 * cm / cm.values.sum()).round(1)
    lines += ["**Confusion matrix** (rows: hand code, columns: model; % of items, weighted)", "",
              _md(cm_t.reset_index().rename(columns={"gold": "hand \\ model"})), ""]
    return dict(title=title, n_coded=len(coded), n_rated=len(rated), accuracy=acc, se=se, kappa=kappa, per=per,
                by_era=pd.DataFrame(era_rows, columns=["era", "n", "accuracy", "ci", "kappa"]), shares=shares,
                empty_cells=empty, md="\n".join(lines))


# --------------------------------------------------------------------------- per kind
def _flag(df, col):
    return df[col].astype(str).str.strip().isin(["1", "True", "true", "yes"]) if col in df else pd.Series(False, index=df.index)


def _filled(df, col):
    return df[col].astype(str).str.strip() != "" if col in df else pd.Series(False, index=df.index)


def run(kind, dim="people", path=None, strata_path=VDIR / "strata.csv", write=True):
    read = lambda p: pd.read_csv(p, keep_default_na=False, dtype=str)
    files = dict(sense="sense_sample.csv", role="role_sample.csv", visual="visual_sample.csv", seed="seed_check.csv")
    path = Path(path) if path else VDIR / files[kind]
    if not path.exists():
        raise SystemExit(f"{path} not found: export it from the coding app into {VDIR}")
    df = read(path)
    out = []
    if kind == "seed":
        df["era"] = df["edition"].map(ERA).fillna(df["edition"])
        df["_coded"] = _filled(df, "gold_label") | _flag(df, "excluded")
        df["_out"] = _flag(df, "excluded")
        n_pop = len(read(Path(__file__).resolve().parent / "seeds" / "pueblo_senses.csv"))
        out.append(evaluate(df, "gold_label", "claude_label", None, title="Seed check: your codes against Claude's",
                            n_pop=n_pop))
    else:
        strata = load_strata(strata_path)
        strata = strata[strata["kind"] == kind]
        df["_out"] = _flag(df, "excluded")
        if kind == "sense":
            df["_coded"] = _filled(df, "gold_sense") | _flag(df, "excluded")
            df["pred"] = df["predicted_sense"]
            for name, mask, keys in [("Senses of pueblo", df["lemma"] == "pueblo", ["era", "pred"]),
                                     ("Senses of comunidad and asamblea", df["lemma"] != "pueblo", ["era", "lemma", "pred"])]:
                st = strata[(strata["lemma"] == "pueblo") == (name == "Senses of pueblo")]
                out.append(evaluate(df[mask], "gold_sense", "pred", st, keys, title=name))
        elif kind == "role":
            df["_coded"] = _filled(df, "gold_role") | _flag(df, "excluded")
            df["pred"] = df["predicted_role"]
            out.append(evaluate(df, "gold_role", "pred", strata, ["era", "pred"], title="Grammatical roles"))
        else:
            if dim not in VISUAL_DIMS:
                raise SystemExit(f"--dim must be one of {VISUAL_DIMS}")
            df["era"] = df["edition"].map(ERA).fillna(df["edition"])
            df["pred"] = df["pred_people"]                          # the sampling cells
            g = f"gold_{dim}"
            if dim in CONDITIONAL:   # coded units: people decided, and this row coded unless there are no people
                df["_coded"] = _flag(df, "excluded") | (_filled(df, "gold_people") & (
                    (df["gold_people"].astype(str).str.strip() == "no_people") | _filled(df, g)))
            else:
                df["_coded"] = _filled(df, g) | _flag(df, "excluded")
            df["pred_dim"] = df[f"pred_{dim}"]
            note = " (illustrations with people)" if dim in CONDITIONAL else ""
            out.append(evaluate(df, g, "pred_dim", strata, ["era", "pred"], title=f"Illustrations: {dim}{note}"))
    md = "\n".join(r["md"] for r in out)
    print(md)
    if write:
        name = f"validate_{kind}{'_' + dim if kind == 'visual' else ''}.md"
        head = (f"# Validation: {kind}{' / ' + dim if kind == 'visual' else ''}\n\nFrom `{path.name}`. "
                "Generated by `python -m ltg validate`; see `ltg/validation.py` for the weighting.\n\n")
        (path.parent / name).write_text(head + md, encoding="utf-8")
        log.info("wrote %s", path.parent / name)
    return out
