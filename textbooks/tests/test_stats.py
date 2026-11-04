import numpy as np
import pandas as pd
import pytest

from ltg import stats


def test_holm_matches_hand_computation():
    adj = stats.holm([0.01, 0.04, 0.03, 0.5])
    # sorted 0.01, 0.03, 0.04, 0.5 -> 0.04, 0.09, 0.09 (monotone), 0.5
    assert np.allclose(adj, [0.04, 0.09, 0.09, 0.5])


def test_rate_ratio_recovers_known_era_effect():
    pytest.importorskip("statsmodels")
    rng = np.random.default_rng(0)
    rows = []
    for era, rate in [("2008-22", 10.0), ("NEM", 20.0), ("1993", 10.0)]:
        for i in range(60):
            words = rng.integers(20_000, 60_000)
            rows.append(dict(edition={"2008-22": "2014", "NEM": "2023", "1993": "1993"}[era],
                             code=f"{era}-{i}", era=era, words=float(words),
                             comunidad=rng.poisson(rate * words / 10_000)))
    books = pd.DataFrame(rows)
    out = stats.rate_ratios(books)
    nem = out[out["era"] == "NEM"].iloc[0]
    assert nem["ci_low"] < 2.0 < nem["ci_high"]
    assert 1.8 < nem["ratio"] < 2.2
    same = out[out["era"] == "1993"].iloc[0]
    assert same["ci_low"] < 1.0 < same["ci_high"]
