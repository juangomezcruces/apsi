"""Supervised word-sense classifier for 'pueblo'.

    python -m ltg senses            # after `text`, before `report`

'Pueblo' is a town, the common people (as against their rulers), a nation (or the
peoples of the world), or an ethnic / cultural / ancient people. Rules got about
half of the 1988 uses wrong or left them ambiguous, and zero-shot NLI with a
multilingual model was no better than guessing 'town'. The standard alternative
works: embed the word in its context with a Spanish BERT (BETO, mean of the last
four layers over the word's subtokens) and fit a logistic regression on coded
examples. Cross-validated accuracy is printed on every run and written to
results/text/senses_cv.txt.

The coded examples live in ltg/seeds/pueblo_senses.csv. The first ones were coded
by Claude (column coder = 'claude') to get the classifier started; they are not
human gold codes. Replace or extend them with your own coding (any coder name),
re-run, and report agreement on the separate hand-coded validation sample.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd


log = logging.getLogger("ltg.senses")
for _noisy in ("httpx", "httpx2", "urllib3", "huggingface_hub", "transformers"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
SEEDS = Path(__file__).resolve().parent / "seeds" / "pueblo_senses.csv"
LABELS = ["settlement", "people_commons", "people_nation", "people_ethnic"]
MODEL = "dccuchile/bert-base-spanish-wwm-cased"
LEFT_WORDS, RIGHT_WORDS = 40, 30


class Embedder:
    def __init__(self, model=MODEL, device=None):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model).to(self.device).eval()

    def __call__(self, left, keyword, right, batch=32):
        """One vector per (left, keyword, right): mean over the keyword's subtokens of the
        mean of the last four hidden layers."""
        texts, spans = [], []
        for l, k, r in zip(left, keyword, right):
            l = " ".join(str(l if isinstance(l, str) else "").split()[-LEFT_WORDS:])
            r = " ".join(str(r if isinstance(r, str) else "").split()[:RIGHT_WORDS])
            start = len(l) + 1 if l else 0
            texts.append(f"{l} {k} {r}".strip() if l else f"{k} {r}".strip())
            spans.append((start, start + len(k)))
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), batch):
                enc = self.tok(texts[i:i + batch], return_offsets_mapping=True, return_tensors="pt",
                               padding=True, truncation=True, max_length=192)
                offsets = enc.pop("offset_mapping")
                hs = self.model(**{k: v.to(self.device) for k, v in enc.items()}, output_hidden_states=True).hidden_states
                h = self.torch.stack(hs[-4:]).mean(0).cpu()
                for j, (a, b) in enumerate(spans[i:i + batch]):
                    idx = [t for t, (x, y) in enumerate(offsets[j].tolist()) if y > x and x < b and y > a]
                    out.append(h[j, idx].mean(0).numpy() if idx else h[j, 0].numpy())
        return np.stack(out)


def load_seeds(path=SEEDS):
    seeds = pd.read_csv(path, keep_default_na=False)
    bad = set(seeds["label"]) - set(LABELS)
    if bad:
        raise ValueError(f"unknown labels in {path}: {bad}")
    return seeds


def _classifier():
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=5000, C=0.5, class_weight="balanced")


def cross_validate(X, y, folds=5, seed=0):
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    k = min(folds, pd.Series(y).value_counts().min())
    pred = cross_val_predict(_classifier(), X, y, cv=StratifiedKFold(k, shuffle=True, random_state=seed))
    town = lambda a: np.where(np.asarray(a) == "settlement", "settlement", "people")
    return dict(accuracy=float(np.mean(pred == y)), people_vs_town=float(np.mean(town(pred) == town(y))),
                confusion=pd.crosstab(pd.Series(y, name="coded"), pd.Series(pred, name="predicted")), folds=k)


def run(embedder=None, seeds_path=SEEDS):
    from .text_analysis import is_counted, out_dir
    path = out_dir() / "people_tokens.csv.gz"
    ptok = pd.read_csv(path, dtype={"edition": str}, keep_default_na=False)
    seeds = load_seeds(seeds_path)
    emb = embedder or Embedder()
    Xs, ys = emb(seeds["left"], seeds["keyword"], seeds["right"]), seeds["label"].values
    cv = cross_validate(Xs, ys)
    report = (f"seed examples: {len(seeds)} {seeds['label'].value_counts().to_dict()}\n"
              f"coders: {seeds['coder'].value_counts().to_dict()}\n"
              f"{cv['folds']}-fold CV accuracy {cv['accuracy']:.3f}; people vs town {cv['people_vs_town']:.3f}\n"
              f"{cv['confusion'].to_string()}\n")
    log.info("\n%s", report)
    clf = _classifier().fit(Xs, ys)

    rows = ptok["lemma"] == "pueblo"
    if "sense_rule" not in ptok:
        ptok["sense_rule"] = ptok["sense"]
    ptok["sense_p"] = np.nan
    if rows.any():
        X = emb(ptok.loc[rows, "left"], ptok.loc[rows, "keyword"], ptok.loc[rows, "right"])
        proba = clf.predict_proba(X)
        ptok.loc[rows, "sense"] = clf.classes_[proba.argmax(1)]
        ptok.loc[rows, "sense_p"] = proba.max(1).round(3)
    ptok["counted"] = [is_counted(s, f) for s, f in zip(ptok["sense"], ptok["families"])]
    ptok.to_csv(path, index=False)
    (out_dir() / "senses_cv.txt").write_text(report, encoding="utf-8")
    changed = (ptok.loc[rows, "sense"] != ptok.loc[rows, "sense_rule"]).mean() if rows.any() else 0
    log.info("classified %d 'pueblo' tokens (%.0f%% differ from the rules): %s", rows.sum(), 100 * changed,
             ptok.loc[rows, "sense"].value_counts().to_dict())
    return ptok, cv
