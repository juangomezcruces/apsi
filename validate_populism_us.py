"""Populism-pluralism on the US 2010-2018 sentences, with two wordings averaged.

Run where torch + transformers exist (Kaggle GPU, ~25 min for 3,545 sentences):
    python validate_populism_us.py --script classify_populism.py \
        --input sentences_2010_2018_ready.csv --out /kaggle/working

Phase 1 runs once (main + paraphrase battery, cached in us_para.margins.npz);
everything after it is a CPU rescore of that cache.

  ARMS        AUC populist vs pluralist under four settings:
                main wording alone vs both wordings averaged, each with
                the shared gates (topic, veto, 5-word minimum) and with
                full coverage (gates off, as the earlier US validation ran).
              "main, full coverage" must reproduce the earlier 0.848; if it
              does not, the run is not comparable to the earlier one.
  RELIABILITY each wording scored alone, then correlated: the ceiling for any
              correlation reported downstream (engine's reliability(), here
              from the cache so the model is not run twice).

Outputs in --out: us_para.csv (+ .margins.npz), populism_us_report.txt.
"""
import argparse
import importlib.util
import os

import numpy as np
import pandas as pd


def auc(score, y):
    score = np.asarray(score, float); y = np.asarray(y).astype(bool)
    ok = np.isfinite(score); score, y = score[ok], y[ok]
    o = np.argsort(-score)
    tp = np.r_[0, np.cumsum(y[o])]; fp = np.r_[0, np.cumsum(~y[o])]
    return float(np.trapezoid(tp / tp[-1], fp / fp[-1]))


def summarize(name, d, n_all):
    s = d[np.isfinite(d.lr_score)]
    pp = s[s.label != "neutral"]
    means = s.groupby("label").lr_score.mean().round(2).to_dict()
    return (f"  {name:<34} scored {len(s):>5}/{n_all} | AUC pop vs plur "
            f"{auc(pp.lr_score, pp.label == 'populist'):.3f} (n={len(pp)}) | means {means}")


def main(argv=None, scorer=None):
    """scorer: a stand-in MarginScorer, for dry runs without the model."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", default="classify_populism.py")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default=".")
    a = ap.parse_args(argv)
    spec = importlib.util.spec_from_file_location("pop", a.script)
    P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
    assert P.battery_hash() == "78de7ae5a864", f"wrong battery: {P.battery_hash()}"
    assert P.BATTERY_PARAPHRASE is not None, "this script version has no second wording"

    os.makedirs(a.out, exist_ok=True)
    out_csv = os.path.join(a.out, "us_para.csv")
    df = pd.read_csv(a.input)
    P.run(input_csv=a.input, output_csv=out_csv, group_col="faction", average_paraphrase=True,
          scorer=scorer, run_diagnostics=False)
    mpath = P.default_margins_path(out_csv)

    full = dict(topic_margin_min=-99.0, use_veto=False, min_tokens=1, active_margin=-99.0, min_active=1)
    lines = [f"US 2010-2018, {len(df)} sentences | battery {P.battery_hash()} "
             f"({len(P.BATTERY)} pairs, two wordings)", "", "ARMS"]
    for name, kw in [("main wording, full coverage", dict(average_paraphrase=False, **full)),
                     ("averaged, full coverage", dict(average_paraphrase=True, **full)),
                     ("main wording, shared gates", dict(average_paraphrase=False)),
                     ("averaged, shared gates", dict(average_paraphrase=True))]:
        d = P.rescore(mpath, input_csv=a.input, group_col="faction", run_diagnostics=False, **kw)[0]
        d = d.merge(df[["id", "label"]], on="id", how="left") if "label" not in d else d
        lines.append(summarize(name, d, len(df)))

    ids, topic, batt, stance, veto, meta, batt2 = P.load_margins(mpath)
    sc = {}
    for name, bat, m in [("main", P.BATTERY, batt), ("paraphrase", P.BATTERY_PARAPHRASE, batt2)]:
        agg = P.ContrastAggregator(bat, P.TOPIC_HYPOTHESES, bootstrap=0, stance_mode="off",
                                   topic_margin_min=-99.0, use_veto=False, min_tokens=1,
                                   active_margin=-99.0, min_active=1).fit(m, topic)
        sc[name] = np.array([r["score"] for r in agg.transform(m, topic)], float)
    ok = np.isfinite(sc["main"]) & np.isfinite(sc["paraphrase"])
    x, y = sc["main"][ok], sc["paraphrase"][ok]
    rank = lambda v: pd.Series(v).rank().to_numpy()
    lines += ["", "RELIABILITY (each wording alone, full coverage)",
              f"  n {ok.sum()} | pearson {np.corrcoef(x, y)[0, 1]:.3f} | spearman "
              f"{np.corrcoef(rank(x), rank(y))[0, 1]:.3f} | mean |diff| {np.abs(x - y).mean():.2f} points"]
    lab = pd.Series(ids).map(df.set_index(df.id.astype(str)).label).to_numpy()
    pp = lab != "neutral"
    for name in ("main", "paraphrase"):
        lines.append(f"  AUC pop vs plur, {name} wording alone: {auc(sc[name][pp], lab[pp] == 'populist'):.3f}")
    report = "\n".join(lines)
    print("\n" + report)
    open(os.path.join(a.out, "populism_us_report.txt"), "w").write(report)


if __name__ == "__main__":
    main()
