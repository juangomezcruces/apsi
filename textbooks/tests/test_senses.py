"""The sense stage with a stand-in embedder (bag of words), so no model is downloaded.
What is tested is the plumbing: training on seeds, relabelling 'pueblo' rows only,
recomputing `counted`, keeping the rule label, and writing the CV report."""
import numpy as np
import pandas as pd

from ltg import senses as S
from ltg import text_analysis as T


class BagOfWords:
    VOCAB = ["calle", "plaza", "casas", "gobierno", "elige", "voluntad", "mexicano", "mundo",
             "indigenas", "antiguos", "mayas", "vive"]

    def __call__(self, left, keyword, right):
        out = []
        for l, r in zip(left, right):
            words = f"{l} {r}".lower().split()
            out.append([words.count(w) for w in self.VOCAB] + [1.0])
        return np.asarray(out, float)


SEEDS = [("vive en la calle del", "pueblo", "y en la plaza", "settlement"),
         ("las casas del", "pueblo", "y la calle", "settlement"),
         ("en la plaza del", "pueblo", "vive mucha gente", "settlement"),
         ("el gobierno que el", "pueblo", "elige", "people_commons"),
         ("la voluntad del", "pueblo", "y el gobierno", "people_commons"),
         ("el", "pueblo", "elige a su gobierno", "people_commons"),
         ("el", "pueblo", "mexicano", "people_nation"),
         ("los", "pueblos", "del mundo", "people_nation"),
         ("todos los", "pueblos", "del mundo y el mexicano", "people_nation"),
         ("los", "pueblos", "indigenas", "people_ethnic"),
         ("los", "pueblos", "antiguos y mayas", "people_ethnic"),
         ("los", "pueblos", "mayas e indigenas", "people_ethnic")]


def test_senses_stage(data_root, tmp_path):
    seeds = tmp_path / "seeds.csv"
    pd.DataFrame([dict(source=f"s{i}", edition="1988", left=l, keyword=k, right=r, label=lab, coder="test")
                  for i, (l, k, r, lab) in enumerate(SEEDS)]).to_csv(seeds, index=False)
    rows = [dict(sent_id="a", edition="1988", lemma="pueblo", families="pueblo", sense="ambiguous",
                 left="las casas y la calle del", keyword="pueblo", right="", counted=True),
            dict(sent_id="b", edition="1988", lemma="pueblo", families="pueblo", sense="settlement",
                 left="la voluntad del", keyword="pueblo", right="que elige su gobierno", counted=False),
            dict(sent_id="c", edition="1988", lemma="comunidad", families="comunidad", sense="classroom",
                 left="En", keyword="asamblea", right=", elaboren", counted=False)]
    T.out_dir().mkdir(parents=True)
    pd.DataFrame(rows).to_csv(T.out_dir() / "people_tokens.csv.gz", index=False)

    ptok, cv = S.run(embedder=BagOfWords(), seeds_path=seeds)
    ptok = ptok.set_index("sent_id")
    assert ptok.loc["a", "sense"] == "settlement" and not ptok.loc["a", "counted"]
    assert ptok.loc["b", "sense"] == "people_commons" and ptok.loc["b", "counted"]
    assert ptok.loc["b", "sense_rule"] == "settlement"                     # rule label kept for comparison
    assert ptok.loc["c", "sense"] == "classroom" and not ptok.loc["c", "counted"]   # non-pueblo untouched
    assert 0 <= cv["accuracy"] <= 1
    assert "CV accuracy" in (T.out_dir() / "senses_cv.txt").read_text()
    saved = pd.read_csv(T.out_dir() / "people_tokens.csv.gz").set_index("sent_id")
    assert saved.loc["b", "sense"] == "people_commons"


def test_seed_file_is_valid():
    seeds = S.load_seeds()
    assert set(seeds["label"]) == set(S.LABELS)
    assert {"source", "edition", "left", "keyword", "right", "label", "coder"} <= set(seeds.columns)
    assert seeds["keyword"].str.lower().str.startswith("pueblo").all()
