"""Text analysis on hand-written sentences. These pin the behaviour of the rules with
es_core_news_sm 3.8; a different parser version can move individual cases."""
import json

import pandas as pd
import pytest

from ltg.config import PATHS

spacy = pytest.importorskip("spacy")
try:
    spacy.load("es_core_news_sm")
except OSError:
    pytest.skip("es_core_news_sm not installed", allow_module_level=True)

from ltg import text_analysis as T  # noqa: E402

PAGES = {
    ("1988", "H1988P4CS001"): [
        "El pueblo de México luchó contra la dictadura. El pueblo fue explotado por los hacendados.\n\n"
        "Los campesinos y los obreros lucharon por la tierra. Vivo en un pueblo pequeño cerca de la ciudad.",
        "Todo el pueblo se unió para defender la patria. Los indígenas sufrieron el despojo de sus tierras."],
    ("2023", "P5PCA"): [
        "Los pueblos originarios conservan sus lenguas. La comunidad se organiza en asamblea para decidir.\n\n"
        "Respetamos la diversidad de opiniones y el diálogo. La soberanía nacional reside en el pueblo.",
        "Todas las personas tienen derechos. Los pueblos indígenas y afromexicanos exigen respeto a su "
        "diversidad. El pueblo de Dolores es tranquilo."],
}


@pytest.fixture
def parsed(data_root):
    for (ed, code), texts in PAGES.items():
        p = PATHS["text"] / ed / f"{code}.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join(json.dumps(dict(edition=ed, code=code, page=i + 1, source="ocr", text=t,
                                             n_words=len(t.split()), mean_conf=90.0)) + "\n"
                             for i, t in enumerate(texts)), encoding="utf-8")
    acc = T.parse()
    return pd.DataFrame(acc["people_tokens"]), pd.DataFrame(acc["sentences"])


def row(ptok, form, nth=0, edition=None):
    d = ptok[ptok["form"] == form]
    if edition:
        d = d[d["edition"] == edition]
    return d.iloc[nth]


def test_matcher_pos_and_phrases():
    nlp = spacy.load("es_core_news_sm")
    m = T.Matcher()
    hits = {(f, t) for f, t, *_ in m(nlp("El territorio mexicano y los mexicanos viven con los pueblos originarios."))}
    assert ("nacion", "mexicano") in hits                       # 'los mexicanos' (noun) counts ...
    assert sum(1 for f, t in hits if t == "mexicano") == 1      # ... 'territorio mexicano' (adjective) does not
    assert ("etnico", "pueblos originarios") in hits and ("pueblo", "pueblo") in hits


def test_pueblo_senses(parsed):
    ptok, _ = parsed
    assert row(ptok, "pueblo", 0, "1988")["sense"] == "people_political"   # pueblo de México luchó
    assert row(ptok, "pueblo", 1, "1988")["sense"] == "people_political"   # fue explotado
    assert row(ptok, "pueblo", 2, "1988")["sense"] == "settlement"         # vivo en un pueblo pequeño
    assert row(ptok, "pueblo", 3, "1988")["sense"] == "people_political"   # todo el pueblo se unió
    assert row(ptok, "pueblos", 0, "2023")["sense"] == "people_ethnic"     # pueblos originarios
    assert row(ptok, "pueblo", 0, "2023")["sense"] == "people_political"   # soberanía reside en el pueblo
    assert row(ptok, "pueblo", 1, "2023")["sense"] == "settlement"         # pueblo de Dolores es tranquilo
    assert not row(ptok, "pueblo", 2, "1988")["counted"]


def test_roles_and_verb_classes(parsed):
    ptok, _ = parsed
    r = row(ptok, "pueblo", 0, "1988")
    assert (r["role"], r["head_lemma"], r["verb_class"], r["nmod"]) == ("agent", "luchar", "struggle", "mexico")
    r = row(ptok, "pueblo", 1, "1988")
    assert (r["role"], r["verb_class"]) == ("patient", "victim")
    assert row(ptok, "obreros")["role"] == "agent"                         # conjunct inherits the role
    assert row(ptok, "patria")["verb_class"] == ""                         # objects get no action class
    assert row(ptok, "pueblos", 0, "2023")["attributes"] == "originario"
    assert row(ptok, "pueblo", 3, "1988")["det"] == "todo+el"
    assert row(ptok, "indígenas")["verb_class"] == "suffering"


def test_sentence_flags(parsed):
    _, sents = parsed
    by_text = {s["text"]: s for s in sents.to_dict("records")}
    assert by_text["El pueblo de México luchó contra la dictadura."]["has_elite"]
    assert by_text["Todo el pueblo se unió para defender la patria."]["has_unidad"]
    assert by_text["Los pueblos indígenas y afromexicanos exigen respeto a su diversidad."]["has_pluralismo"]
    # no people noun -> not people-talk, whatever else it contains
    assert "Respetamos la diversidad de opiniones y el diálogo." not in by_text
    assert "Todas las personas tienen derechos." not in by_text            # generic 'personas' is context only
    assert "Vivo en un pueblo pequeño cerca de la ciudad." not in by_text   # settlement only


def test_report_and_exports(parsed):
    tables = T.report(B=50)
    rates = pd.read_csv(T.out_dir() / "rates_per_10k.csv", dtype={"edition": str})
    get = lambda ed, m: rates[(rates["edition"] == ed) & (rates["measure"] == m)]["estimate"].item()
    pt = pd.read_csv(T.out_dir() / "page_tokens.csv", dtype={"edition": str})
    tok88 = pt[pt["edition"] == "1988"]["n_tokens"].sum()
    assert get("1988", "pueblo") == pytest.approx(3 / tok88 * 10_000)     # 4 'pueblo', 1 settlement removed
    assert get("1988", "people_all") == pytest.approx(7 / tok88 * 10_000)
    senses = tables["pueblo_senses"].set_index("sense")
    assert senses.loc["settlement", "1988"] == 1 and senses.loc["people_ethnic", "2023"] == 2
    assert (T.out_dir() / "report.md").read_text().startswith("# 'The people'")
    T.export_for_coding(n_per_cell=5)
    sample = pd.read_csv(PATHS["results"] / "validation" / "pueblo_sense_sample.csv")
    assert {"predicted_sense", "gold_sense", "left", "keyword", "right"} <= set(sample.columns)
    apsi = pd.read_csv(PATHS["results"] / "people_sentences_for_apsi.csv")
    assert {"id", "text", "text_en"} <= set(apsi.columns) and len(apsi) > 5


def test_validate(tmp_path):
    f = tmp_path / "s.csv"
    pd.DataFrame(dict(pred=["a", "a", "b", "b"], gold=["a", "b", "b", "b"])).to_csv(f, index=False)
    acc, kappa, per = T.validate(f, "pred", "gold")
    assert acc == 0.75 and kappa == pytest.approx(0.5)


def test_bootstrap_and_keyness():
    est, lo, hi = T.cluster_bootstrap([10, 20, 30], [100, 100, 100], None, B=500, scale=100)
    assert est == pytest.approx(20) and lo < est < hi
    g2, lr = T.log_likelihood(100, 10, 10_000, 10_000)
    assert g2 > 15.13 and lr > 3
