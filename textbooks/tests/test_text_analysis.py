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
    ptok, all_sents = parsed
    sents = T.people_talk(all_sents, ptok)        # people-talk = sentences with a counted people noun
    assert "Vivo en un pueblo pequeño cerca de la ciudad." in set(all_sents["text"])   # kept as a candidate
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
    sample = pd.read_csv(PATHS["results"] / "validation" / "sense_sample.csv")
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


@pytest.mark.parametrize("sentence, word, sense", [
    ("En asamblea, elaboren una lista de propuestas.", "asamblea", "classroom"),
    ("En pequeñas comunidades, dialoguen sobre estas preguntas.", "comunidades", "classroom"),
    ("Organizados en comunidades, observen los libros.", "comunidades", "classroom"),
    ("Cada comunidad organice una presentación.", "comunidad", "classroom"),
    ("La segunda comunidad creará un formulario.", "comunidad", "classroom"),
    ("Presenten su cartel ante la comunidad escolar.", "comunidad", "school"),
    ("Escribe la historia de tu comunidad.", "comunidad", "local"),
    ("La comunidad se organizó para conseguir agua.", "comunidad", "collective"),
    ("Los pueblos originarios aprendieron a vivir en comunidad.", "comunidad", "collective"),
])
def test_community_sense(sentence, word, sense):
    nlp = spacy.load("es_core_news_sm")
    tok = next(t for t in nlp(sentence) if t.text == word)
    assert T.community_sense(tok) == sense


def test_classroom_community_is_not_people_talk(data_root):
    p = PATHS["text"] / "2023" / "P5PCA.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    text = "En asamblea, elaboren una lista. La comunidad se organizó para conseguir agua."
    p.write_text(json.dumps(dict(edition="2023", code="P5PCA", page=1, source="ocr", text=text)) + "\n")
    acc = T.parse()
    ptok = pd.DataFrame(acc["people_tokens"]).set_index("form")
    assert not ptok.loc["asamblea", "counted"] and ptok.loc["comunidad", "counted"]
    sents = T.people_talk(pd.DataFrame(acc["sentences"]), pd.DataFrame(acc["people_tokens"]))
    assert list(sents["text"]) == ["La comunidad se organizó para conseguir agua."]
    T.report(B=20)
    rates = pd.read_csv(T.out_dir() / "rates_per_10k.csv", dtype={"edition": str}).set_index("measure")
    tokens = pd.read_csv(T.out_dir() / "page_tokens.csv")["n_tokens"].sum()
    assert rates.loc["comunidad", "estimate"] == pytest.approx(1 / tokens * 10_000)   # asamblea dropped


def test_national_and_state_corpora_are_reported_apart(data_root):
    for code, text in [("P4HIA", "El pueblo de México luchó contra la dictadura."),
                       ("P3OAX", "Los pueblos originarios de Oaxaca conservan sus lenguas. El pueblo zapoteco vive aquí.")]:
        p = PATHS["text"] / "2022" / f"{code}.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(dict(edition="2022", code=code, page=1, source="ocr", text=text)) + "\n")
    pd.DataFrame([dict(edition="2022", code="P4HIA", grade=4, subject="Historia", corpus="national"),
                  dict(edition="2022", code="P3OAX", grade=3, subject="La entidad donde vivo", corpus="state")]
                 ).to_csv(PATHS["catalog"], index=False)
    T.parse()
    nat = T.report(B=20)
    assert nat["overview"].set_index("edition").loc["2022", "books"] == 1
    T.report(B=20, corpus="state")
    state = pd.read_csv(T.out_dir() / "rates_per_10k_state.csv", dtype={"edition": str}).set_index("measure")
    assert state.loc["etnico", "count"] >= 2 and (T.out_dir() / "report_state.md").exists()
    national = pd.read_csv(T.out_dir() / "rates_per_10k.csv").set_index("measure")
    assert "etnico" not in national.index or national.loc["etnico", "count"] == 0   # state book excluded


def test_lexicon_is_consistent():
    from collections import Counter
    from ltg import lexicon as L
    verbs = Counter(v for vs in L.VERB_CLASSES.values() for v in vs)
    assert not [v for v, n in verbs.items() if n > 1], "a verb sits in two classes"
    for fam, spec in L.FAMILIES.items():
        for term in spec["terms"]:
            word = term.split("/")[0]
            assert word == T.fold(word), f"{fam}: write '{word}' without accents, in lower case"
