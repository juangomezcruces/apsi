"""Sentence reuse between editions."""
import json

import pytest

from ltg import reuse as R
from ltg.config import PATHS


def test_sentences_normalise_and_filter():
    s = R.sentences("El pueblo de México luchó contra la dictadura de Díaz.\nCorto.  Otra frase muy corta aquí.")
    assert s == {"el pueblo de mexico lucho contra la dictadura de diaz"}


def test_reuse_matrix(data_root):
    shared = "Cuando había un buen rey todo salía bien pero si era malo el pueblo sufría."
    for ed, text in (("1972", shared + " Los campesinos trabajaban la tierra de sol a sol cada día."),
                     ("1988", shared + " Los obreros de las fábricas se organizaron en sindicatos nuevos.")):
        p = PATHS["text"] / ed / "B.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(dict(edition=ed, code="B", page=1, text=text)) + "\n")
    m = R.run()
    assert m.loc["1988", "1972"] == pytest.approx(50.0) and m.loc["1988", "1988"] == pytest.approx(100.0)
