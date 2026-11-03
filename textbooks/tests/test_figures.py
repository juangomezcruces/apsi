"""Figures render from the report tables (synthetic numbers)."""
import pandas as pd
import pytest

pytest.importorskip("matplotlib")

from ltg import figures as F  # noqa: E402
from ltg import text_analysis as T  # noqa: E402
from ltg.config import PATHS  # noqa: E402


def test_figures_render(data_root):
    T.out_dir().mkdir(parents=True)
    rows = [dict(edition=e, measure=m, estimate=v, ci_low=v * 0.7, ci_high=v * 1.3)
            for e, k in (("1988", 1.0), ("2022", 1.4), ("2023", 2.0))
            for m, v in (("people_all", 50 * k), ("pueblo", 6 * k), ("elite", 3 / k), ("has_elite", 2 * k))]
    pd.DataFrame(rows).to_csv(T.out_dir() / "rates_per_10k.csv", index=False)
    pd.DataFrame(rows).to_csv(T.out_dir() / "people_sentence_shares.csv", index=False)
    pd.DataFrame(dict(edition=["1988"] * 4 + ["2023"] * 3, lemma="pueblo", corpus="national",
                      sense=["settlement", "people_commons", "people_nation", "people_ethnic",
                             "people_ethnic", "people_ethnic", "settlement"])).to_csv(
        T.out_dir() / "people_tokens.csv.gz", index=False)
    made = F.run()
    assert made == ["people_rates", "relation_rates", "people_talk_context", "pueblo_senses"]
    for name in made:
        assert (PATHS["results"] / "figures" / f"{name}.png").stat().st_size > 5000
