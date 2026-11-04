import pandas as pd
import pytest

from ltg import validation as V


def _design():
    """Two eras; 20 coded items per (era, predicted label) cell. Items predicted x are always
    right, items predicted y are right half the time, and the cells differ in size."""
    strata = pd.DataFrame(dict(kind="sense", era=["A", "A", "B", "B"], lemma="", pred=["x", "y", "x", "y"],
                               N=[900, 100, 100, 100]))
    rows = []
    for era in "AB":
        for pred in "xy":
            for i in range(20):
                gold = pred if pred == "x" or i % 2 else "x"
                rows.append(dict(era=era, pred=pred, gold=gold, _coded=True, _out=False))
    return pd.DataFrame(rows), strata


def test_weighted_accuracy_recovers_the_corpus_value():
    df, strata = _design()
    r = V.evaluate(df, "gold", "pred", strata, ["era", "pred"], title="t")
    # corpus: (900 + 50 + 100 + 50) / 1200 right; the unweighted sample says 60 / 80
    assert r["accuracy"] == pytest.approx(1100 / 1200)
    by = r["by_era"].set_index("era")
    assert float(by.loc["A", "accuracy"]) == pytest.approx(95.0)
    assert float(by.loc["B", "accuracy"]) == pytest.approx(75.0)
    sh = r["shares"].set_index(["era", "label"])
    assert sh.loc[("A", "x"), "model"] == pytest.approx(0.9) and sh.loc[("A", "x"), "hand"] == pytest.approx(0.95)
    assert 0 < r["se"] < 0.1


def test_partial_coding_and_left_out_items():
    df, strata = _design()
    df.loc[df.index[:5], "_coded"] = False                       # five A/x items not coded yet
    df.loc[(df["era"] == "B") & (df["pred"] == "y"), "gold"] = ""
    df.loc[(df["era"] == "B") & (df["pred"] == "y"), "_out"] = True  # a whole cell left out
    r = V.evaluate(df, "gold", "pred", strata, ["era", "pred"], title="t")
    assert r["accuracy"] == pytest.approx((900 + 50 + 100) / 1100)
    assert "100 left out" not in r["md"] and "(20 left out)" in r["md"]
    assert r["empty_cells"].empty


def test_uncoded_cells_are_reported():
    df, strata = _design()
    df.loc[df["era"] == "B", "_coded"] = False
    r = V.evaluate(df, "gold", "pred", strata, ["era", "pred"], title="t")
    assert len(r["empty_cells"]) == 2 and r["accuracy"] == pytest.approx(950 / 1000)


def test_seed_check_is_unweighted(tmp_path):
    f = tmp_path / "seed_check.csv"
    pd.DataFrame(dict(source=["a", "b", "c", "d"], edition=["1960", "1960", "2023", "2023"],
                      claude_label=["settlement", "people_ethnic", "settlement", "people_nation"],
                      gold_label=["settlement", "people_ethnic", "people_ethnic", ""],
                      excluded=["", "", "", "1"])).to_csv(f, index=False)
    r = V.run("seed", path=f, write=False)[0]
    assert r["accuracy"] == pytest.approx(2 / 3) and r["n_coded"] == 4 and r["n_rated"] == 3


def test_strata_file_covers_the_samples():
    s = V.load_strata()
    sense = pd.read_csv(V.VDIR / "sense_sample.csv", keep_default_na=False, dtype=str)
    pb = sense[sense["lemma"] == "pueblo"].groupby(["era", "predicted_sense"]).size()
    cells = s[(s["kind"] == "sense") & (s["lemma"] == "pueblo")].set_index(["era", "pred"])["N"]
    assert all(n == min(cells[k], 20) for k, n in pb.items())
