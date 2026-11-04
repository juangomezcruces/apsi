"""How the books talk about 'the people': one spaCy pass, then aggregation.

    python -m ltg text   [--editions ...] [--workers 4]   # parse -> results/text/*.csv.gz
    python -m ltg report                                  # aggregate -> results/text/report.md

The parse pass writes tidy tables, one row per observation, so every number in
the report can be traced back to a page:

    page_tokens.csv        tokens per page (the denominator for every rate)
    matches.csv.gz         every lexicon hit: family, term, page, sentence
    people_tokens.csv.gz   one row per candidate people noun: sense (pueblo,
                           comunidad, asamblea), whether it counts as people-talk,
                           number, determiner, grammatical role, governing verb and
                           its class, adjectives, keyword-in-context, and for
                           'pueblo' the lemmas within 5 tokens (collocations)
    sentences.csv.gz       every sentence with a candidate people noun, with flags
                           for elite / grievance / unity / pluralism / institutions;
                           people-talk = sentences with a counted people_token
    lemma_freq.csv.gz      lemma counts per edition (keyness, collocations)
"""
import argparse
import logging
import math
import re
import unicodedata
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from . import lexicon as L
from .config import PATHS, PERIOD
from .extract import load_pages

log = logging.getLogger("ltg.text")
def out_dir():
    return PATHS["results"] / "text"


CONTENT_POS = {"NOUN", "PROPN", "ADJ", "VERB"}


def fold(s):
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


def clean_page_text(text):
    """OCR text -> prose spaCy can segment: join wrapped lines, drop rule/box debris."""
    text = re.sub(r"[|_~=<>{}\[\]]+", " ", text or "")
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)              # wrapped line
    text = re.sub(r"\n{2,}", "\n\n", text)                     # paragraph
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


# --------------------------------------------------------------------------- matching
class Matcher:
    """Single-word entries match folded lemmas (optionally POS-restricted);
    multi-word entries match folded word forms."""

    def __init__(self, families=L.FAMILIES):
        self.single = defaultdict(list)       # lemma -> [(family, term, pos)]
        self.multi = defaultdict(list)        # first word -> [(family, term, words)]
        for fam, spec in families.items():
            for term in spec["terms"]:
                base, _, pos = term.partition("/")
                words = tuple(fold(base).split())
                if len(words) == 1:
                    self.single[words[0]].append((fam, base, pos or None))
                else:
                    self.multi[words[0]].append((fam, base, words))

    def __call__(self, doc):
        """Yield (family, term, start, end) over a Doc; end is exclusive."""
        forms = [fold(t.text) for t in doc]
        for i, tok in enumerate(doc):
            for fam, term, pos in self.single.get(fold(tok.lemma_), ()):
                if pos is None or tok.pos_ == pos:
                    yield fam, term, i, i + 1
            for fam, term, words in self.multi.get(forms[i], ()):
                if tuple(forms[i:i + len(words)]) == words:
                    yield fam, term, i, i + len(words)


VERB_CLASS = {v: k for k, vs in L.VERB_CLASSES.items() for v in vs}
ETHNONYMS = {fold(t.split("/")[0]) for t in L.FAMILIES["etnico"]["terms"] if " " not in t}
SETTLEMENT_DETS = {"un", "uno", "una", "este", "ese", "aquel", "algun", "alguno", "cada", "otro"}


def determiners(tok):
    """Determiners of a noun, including those hanging off another determiner ('Todo el pueblo')."""
    out = []
    for c in tok.children:
        if c.dep_ == "det":
            out += [fold(g.lemma_) for g in c.children if g.dep_ == "det"] + [fold(c.lemma_)]
    return out


def _first_conjunct(tok):
    while tok.dep_ == "conj" and tok.head is not tok:
        tok = tok.head
    return tok


def _is_passive(verb):
    return "Part" in verb.morph.get("VerbForm") and any(
        c.dep_ == "aux" and fold(c.lemma_) == "ser" for c in verb.children)


def role_of(tok):
    """(role, governing word) for a noun: agent / patient / attribute / object /
    recipient / possessor / oblique / other. Conjuncts inherit the first one's role."""
    t = _first_conjunct(tok)
    head, dep = t.head, t.dep_
    if dep == "nsubj":
        if head.pos_ in ("ADJ", "NOUN", "PROPN") and any(c.dep_ == "cop" for c in head.children):
            return "attribute", head
        return ("patient" if _is_passive(head) else "agent"), head
    if dep == "obj":
        return ("patient" if fold(head.lemma_) in L.VICTIMISING else "object"), head
    return {"iobj": "recipient", "nmod": "possessor", "obl": "oblique"}.get(dep, "other"), head


def verb_class(role, head):
    lemma = fold(head.lemma_)
    if role == "patient":
        return "victim" if lemma in L.VICTIMISING or _is_passive(head) else "patient_other"
    if role == "attribute":
        return "attribute"
    if role != "agent":
        return ""                      # what is done TO an object is not the people's action
    if lemma in L.VICTIMISING:
        return "victimiser"
    return VERB_CLASS.get(lemma, "other")


def attributes(tok, role, head):
    out = [fold(c.lemma_) for c in tok.children if c.dep_ == "amod"]
    if role == "attribute" and head.pos_ == "ADJ":
        out.append(fold(head.lemma_))
        out += [fold(c.lemma_) for c in head.children if c.dep_ == "conj" and c.pos_ == "ADJ"]
    return out


def _nmod_names(tok):
    """'pueblo de México' -> ['mexico'];  'pueblo de Estados Unidos' -> ['estados unidos']."""
    names = []
    for c in tok.children:
        if c.dep_ in ("nmod", "appos", "flat"):
            words = [fold(w.text) for w in c.subtree if w.dep_ not in ("case", "det", "punct")]
            names.append((" ".join(words), c))
    return names


def pueblo_sense(tok, sent_folded_lemmas):
    """Rule-based sense of 'pueblo': people_ethnic / people_political / settlement / ambiguous.
    The rules are cues with weights; validate them with `python -m ltg sample`."""
    people = settle = ethnic = 0
    for c in tok.children:
        lemma = fold(c.lemma_)
        if c.dep_ == "amod":
            if lemma in L.PUEBLO_ETHNIC_MODS or lemma in ETHNONYMS:
                ethnic += 2
            elif lemma in L.PEOPLE_MODS:
                people += 1
            elif lemma in L.SETTLEMENT_MODS:
                settle += 1
    dets = determiners(tok)
    if any(d in SETTLEMENT_DETS for d in dets):
        settle += 1
    if any(d in ("nuestro", "todo") for d in dets):
        people += 1
    for name, node in _nmod_names(tok):
        if name in L.COUNTRIES or name.split()[-1:] == ["mexico"]:
            people += 2
        elif name in ETHNONYMS or any(w in ETHNONYMS for w in name.split()):
            ethnic += 2
        elif node.pos_ == "PROPN":
            settle += 2
    t = _first_conjunct(tok)
    head = fold(t.head.lemma_)
    role, _ = role_of(tok)
    if role == "agent" and VERB_CLASS.get(head) in ("struggle", "participation", "will", "unity"):
        people += 1
    if role == "patient":                       # towns are not exploited, oppressed or despoiled
        people += 1
    if role == "attribute":                     # 'el pueblo es valiente' / 'el pueblo es tranquilo'
        people += head in L.PEOPLE_MODS
        settle += head in L.SETTLEMENT_MODS
    if t.dep_ == "obl" and head in L.SETTLEMENT_VERBS:
        settle += 1
    if t.dep_ == "nmod" and head in L.POLITICAL_HEADS:
        people += 2
    if head == "residir":
        people += 2
    if sent_folded_lemmas & L.SETTLEMENT_COWORDS:
        settle += 1
    if ethnic >= 2 and ethnic >= settle:
        return "people_ethnic"
    if people > settle:
        return "people_political"
    if settle > people:
        return "settlement"
    return "ambiguous"


def is_counted(sense, families):
    """A town, a classroom grouping or a school is not 'the people' -- unless another family
    also claims the token ('comunidades indígenas' stays, via etnico)."""
    fams = set(families.split("+")) if isinstance(families, str) else set(families)
    return not (sense in NOT_COUNTED and fams <= {"pueblo", "comunidad"})


COMMUNITY_LEMMAS = {"comunidad", "asamblea", "colectivo"}
SCHOOL_MODS = {"escolar", "educativo", "aula", "escuela", "aprendizaje", "salon", "grupo", "clase"}
POSSESSIVES = {"tu", "su", "mi", "nuestro", "vuestro", "tuyo", "suyo"}
GROUP_WORDS = {"pequenas", "pequenos", "dos", "tres", "cuatro", "cinco", "seis", "varias", "sus"}
ORDINALS = {"primero", "segundo", "tercero", "cuarto", "quinto", "sexto"}
NOT_COUNTED = {"settlement", "classroom", "school"}     # senses that are not 'the people'


def community_sense(tok):
    """Sense of comunidad / asamblea: classroom / school / local / collective.

    NEM books use both words to organise class work ('En pequeñas comunidades, reúnan...',
    'En asamblea, elaboren...'); those are instructions, not depictions of a people.
    """
    dets = determiners(tok)
    mods = {fold(w.lemma_) for c in tok.children if c.dep_ in ("amod", "nmod", "compound", "flat")
            for w in c.subtree}
    if mods & SCHOOL_MODS:
        return "school"
    t = _first_conjunct(tok)
    amods = {fold(c.lemma_) for c in tok.children if c.dep_ == "amod"}
    has_nmod = any(c.dep_ == "nmod" for c in tok.children)
    window = [x for x in tok.doc[max(tok.sent.start, tok.i - 4):tok.i] if not x.is_space]
    prev = [fold(x.text) for x in window]
    lives = any(fold(x.lemma_) in ("vivir", "convivir") for x in window[-3:])
    if len(prev) > 1 and prev[-2] == "en" and prev[-1] in GROUP_WORDS and not dets:
        return "classroom"                                      # 'en pequeñas / tres comunidades'
    if amods & ORDINALS or (("pequeno" in amods or "cada" in dets) and not has_nmod) or mods & {"trabajo"}:
        return "classroom"                                      # 'la segunda comunidad', 'cada pequeña comunidad'
    if prev and prev[-1] == "en" and not dets and not lives:
        return "classroom"                                      # 'En asamblea, ...' but not 'vivir en comunidad'
    head = t.head
    if "cada" in dets and head.pos_ in ("VERB", "AUX") and set(head.morph.get("Mood")) & {"Sub", "Imp"}:
        return "classroom"                                      # 'Cada comunidad organice ...'

    if any(d in POSSESSIVES for d in dets):
        return "local"
    return "collective"


# --------------------------------------------------------------------------- parse pass
def _meta(catalog):
    if catalog is None or catalog.empty:
        return {}
    cols = ["edition", "code", "grade", "subject", "title", "level", "corpus"]
    cat = catalog[[c for c in cols if c in catalog.columns]].drop_duplicates(["edition", "code"])
    return {(str(r["edition"]), r["code"]): r for r in cat.to_dict("records")}


def analyse_doc(doc, ctx, matcher, acc):
    """Accumulate every table for one page into `acc` (lists of dicts and counters)."""
    ed = ctx["edition"]
    n_tokens = sum(1 for t in doc if t.is_alpha)
    acc["page_tokens"].append(dict(ctx, n_tokens=n_tokens))
    lemma_counts = acc["lemmas"][(ed, ctx.get("corpus", "national"))]
    for t in doc:
        if t.is_alpha and len(t) > 1 and t.pos_ in CONTENT_POS:
            lemma_counts[(fold(t.lemma_), t.pos_)] += 1

    hits = list(matcher(doc))
    by_sent = defaultdict(list)
    for fam, term, s, e in hits:
        by_sent[doc[s].sent.start].append((fam, term, s, e))

    for k, sent in enumerate(doc.sents):
        sent_hits = by_sent.get(sent.start, [])
        if not sent_hits:
            continue
        sid = f"{ed}|{ctx['code']}|{ctx['page']}|{k}"
        sent_lemmas = {fold(t.lemma_) for t in sent}
        people_tokens = {}
        for fam, term, s, e in sent_hits:
            acc["matches"].append(dict(sent_id=sid, **ctx, family=fam, term=term, token_i=s))
            if L.FAMILIES[fam]["people"]:
                head = doc[s:e].root
                people_tokens.setdefault(head.i, set()).add(fam)

        for i, fams in sorted(people_tokens.items()):
            tok = doc[i]
            lemma = fold(tok.lemma_)
            sense = (pueblo_sense(tok, sent_lemmas) if lemma == "pueblo"
                     else community_sense(tok) if lemma in COMMUNITY_LEMMAS else "")
            role, gov = role_of(tok)
            dets = determiners(tok)
            left = sent.text[: tok.idx - sent.start_char][-90:]
            right = sent.text[tok.idx + len(tok) - sent.start_char:][:90]
            window = ""
            if lemma == "pueblo":                           # collocates, scored once senses are final
                lo, hi = max(sent.start, i - 5), min(sent.end, i + 6)
                window = "+".join(fold(t.lemma_) for t in doc[lo:hi]
                                  if t.i != i and t.is_alpha and t.pos_ in CONTENT_POS)
            acc["people_tokens"].append(dict(
                sent_id=sid, **ctx, token_i=i, families="+".join(sorted(fams)),
                counted=is_counted(sense, fams), lemma=lemma, form=tok.text, sense=sense, window=window,
                number=(tok.morph.get("Number") or [""])[0], det="+".join(dets), role=role,
                head_lemma=fold(gov.lemma_), head_pos=gov.pos_, verb_class=verb_class(role, gov),
                attributes="+".join(attributes(tok, role, gov)),
                nmod="+".join(n for n, _ in _nmod_names(tok)),
                left=left, keyword=tok.text, right=right))

        # Every sentence with a candidate people noun is kept; whether it is people-talk
        # is decided from people_tokens.counted once senses are final (see `senses`).
        if people_tokens:
            sent_fams = {f for f, *_ in sent_hits}
            acc["sentences"].append(dict(
                sent_id=sid, **ctx,
                **{f"has_{f}": f in sent_fams for f in
                   ("elite", "agravio", "unidad", "pluralismo", "instituciones", "nosotros")},
                text=sent.text.strip()))


def parse(editions=None, workers=1, catalog=None, model="es_core_news_sm", batch_size=32):
    import spacy
    pages = load_pages(editions)
    if pages.empty:
        raise SystemExit("no extracted pages: run `python -m ltg extract` first")
    pages = pages[pages["text"].fillna("").str.strip() != ""]
    if catalog is None and PATHS["catalog"].exists():
        catalog = pd.read_csv(PATHS["catalog"], dtype={"edition": str})
    meta = _meta(catalog)
    nlp = spacy.load(model, disable=["ner"])
    nlp.max_length = 2_000_000
    matcher = Matcher()
    acc = dict(page_tokens=[], matches=[], people_tokens=[], sentences=[],
               lemmas=defaultdict(Counter))

    def ctxs():
        for r in pages.itertuples(index=False):
            ed = str(r.edition)
            m = meta.get((ed, r.code), {})
            yield clean_page_text(r.text), dict(
                edition=ed, period=PERIOD.get(ed, ""), code=r.code, page=int(r.page),
                grade=m.get("grade", ""), subject=m.get("subject", ""), corpus=m.get("corpus") or "national")

    for i, (doc, ctx) in enumerate(nlp.pipe(ctxs(), as_tuples=True, batch_size=batch_size,
                                            n_process=workers)):
        analyse_doc(doc, ctx, matcher, acc)
        if i and i % 2000 == 0:
            log.info("%d / %d pages parsed", i, len(pages))
    write_tables(acc)
    return acc


def write_tables(acc):
    out_dir().mkdir(parents=True, exist_ok=True)
    pd.DataFrame(acc["page_tokens"]).to_csv(out_dir() / "page_tokens.csv", index=False)
    for name in ("matches", "people_tokens", "sentences"):
        pd.DataFrame(acc[name]).to_csv(out_dir() / f"{name}.csv.gz", index=False)
    lem = [dict(edition=e, corpus=cp, lemma=l, pos=p, n=n)
           for (e, cp), c in acc["lemmas"].items() for (l, p), n in c.items()]
    pd.DataFrame(lem).to_csv(out_dir() / "lemma_freq.csv.gz", index=False)
    log.info("wrote tables to %s: %d people mentions in %d sentences",
             out_dir(), len(acc["people_tokens"]), len(acc["sentences"]))


# --------------------------------------------------------------------------- statistics
def cluster_bootstrap(num, den, groups, B=1000, seed=0, scale=1.0):
    """Ratio-of-sums estimate with a percentile CI from resampling clusters (books).
    num/den are per-cluster totals for ONE edition."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    est = scale * num.sum() / den.sum() if den.sum() else np.nan
    if len(num) < 2:
        return est, np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), size=(B, len(num)))
    d = den[idx].sum(1)
    reps = scale * np.divide(num[idx].sum(1), d, out=np.full(B, np.nan), where=d > 0)
    lo, hi = np.nanpercentile(reps, [2.5, 97.5])
    return est, lo, hi


def rate_table(per_book, value_cols, den_col, scale, B=1000):
    """Per-edition ratio-of-sums with book-cluster bootstrap CIs, long format."""
    rows = []
    for ed, g in per_book.groupby("edition", sort=False):
        for col in value_cols:
            est, lo, hi = cluster_bootstrap(g[col], g[den_col], g["code"], B=B, scale=scale)
            rows.append(dict(edition=ed, measure=col, estimate=est, ci_low=lo, ci_high=hi,
                             n_books=len(g), count=int(g[col].sum())))
    return pd.DataFrame(rows)


def log_likelihood(a, b, c, d):
    """Dunning's G2 for a word with frequency a in corpus 1 (size c) and b in corpus 2 (size d)."""
    e1, e2 = c * (a + b) / (c + d), d * (a + b) / (c + d)
    g2 = 2 * ((a * math.log(a / e1) if a else 0) + (b * math.log(b / e2) if b else 0))
    log_ratio = math.log2(((a + 0.5) / c) / ((b + 0.5) / d))
    return g2, log_ratio


def keyness(lemma_freq, ed1, ed2, top=30, min_freq=10):
    f = lemma_freq.groupby(["edition", "lemma"])["n"].sum().unstack(0, fill_value=0)
    if ed1 not in f or ed2 not in f:
        return pd.DataFrame()
    c, d = f[ed1].sum(), f[ed2].sum()
    sub = f[(f[ed1] + f[ed2]) >= min_freq]
    rows = [dict(lemma=lem, **{f"n_{ed1}": a, f"n_{ed2}": b},
                 **dict(zip(("g2", "log_ratio"), log_likelihood(a, b, c, d))))
            for lem, a, b in zip(sub.index, sub[ed1], sub[ed2])]
    k = pd.DataFrame(rows)
    if k.empty:
        return k
    k = k[k["g2"] >= 15.13]                                    # p < 0.0001
    over = k[k["log_ratio"] > 0].nlargest(top, "g2").assign(key_in=ed1)
    under = k[k["log_ratio"] < 0].nlargest(top, "g2").assign(key_in=ed2)
    return pd.concat([over, under])


def collocates(ptok, lemma_freq, top=25, min_n=3):
    """logDice between 'pueblo' in its people senses and the lemmas within 5 tokens, per edition."""
    pp = ptok[(ptok["lemma"] == "pueblo") & ptok["sense"].str.startswith("people") & (ptok["window"] != "")]
    window = (pp.assign(lemma=pp["window"].str.split("+")).explode("lemma")
                .groupby(["edition", "lemma"]).size().rename("n").reset_index())
    out = []
    lf = lemma_freq.groupby(["edition", "lemma"])["n"].sum()
    for ed, g in window.groupby("edition"):
        fx = lf.get((ed, "pueblo"), 0)
        co = g.groupby("lemma")["n"].sum()
        co = co[(co >= min_n) & (co.index != "pueblo")]
        for lem, fxy in co.items():
            fy = lf.get((ed, lem), fxy)
            out.append(dict(edition=ed, lemma=lem, n=int(fxy),
                            log_dice=14 + math.log2(2 * fxy / (fx + fy)) if fx + fy else np.nan))
    df = pd.DataFrame(out)
    if df.empty:
        return df
    return df.sort_values(["edition", "log_dice"], ascending=[True, False]).groupby("edition").head(top)


# --------------------------------------------------------------------------- report
def md_table(df, floatfmt="{:.2f}", max_rows=60):
    if df is None or df.empty:
        return "_(no data)_\n"
    df = df.head(max_rows)
    cells = [[floatfmt.format(v) if isinstance(v, (float, np.floating)) and not pd.isna(v)
              else ("" if pd.isna(v) else str(v)) for v in row] for row in df.itertuples(index=False)]
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    return "\n".join([head, sep] + ["| " + " | ".join(c) + " |" for c in cells]) + "\n"


def _with_ci(df, digits=2):
    df = df.copy()
    df["value"] = [f"{e:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]" if not pd.isna(lo) else f"{e:.{digits}f}"
                   for e, lo, hi in zip(df["estimate"], df["ci_low"], df["ci_high"])]
    return df


def edition_order(eds):
    order = [e for e in PERIOD if e in set(eds)]
    return order + sorted(set(eds) - set(order))


def people_talk(sents, ptok):
    """Sentences with at least one counted people noun."""
    ids = set(ptok.loc[ptok["counted"].astype(str) == "True", "sent_id"])
    return sents[sents["sent_id"].isin(ids)]


def _in_corpus(df, corpus):
    """national (default) | state | all. Tables written before the corpus column existed count as national."""
    if corpus == "all" or df.empty:
        return df
    col = df["corpus"].replace("", "national").fillna("national") if "corpus" in df else pd.Series("national", df.index)
    return df[col == corpus]


def report(B=1000, corpus="national"):
    """Tables for one corpus: 'national' (one book per child and subject), 'state' (the 2022
    'La entidad donde vivo' books, one per state) or 'all'. Non-national runs get a suffix."""
    sfx = "" if corpus == "national" else f"_{corpus}"
    pt = _in_corpus(pd.read_csv(out_dir() / "page_tokens.csv", dtype={"edition": str}, keep_default_na=False), corpus)
    pt["n_tokens"] = pt["n_tokens"].astype(int)
    keep = set(zip(pt["edition"], pt["code"]))
    inside = lambda df: df[[k in keep for k in zip(df["edition"], df["code"])]]
    matches = inside(pd.read_csv(out_dir() / "matches.csv.gz", dtype={"edition": str}))
    ptok = inside(pd.read_csv(out_dir() / "people_tokens.csv.gz", dtype={"edition": str}, keep_default_na=False))
    ptok["counted"] = ptok["counted"].astype(str) == "True"
    sents = people_talk(inside(pd.read_csv(out_dir() / "sentences.csv.gz", dtype={"edition": str})), ptok)
    lemma_freq = _in_corpus(pd.read_csv(out_dir() / "lemma_freq.csv.gz", dtype={"edition": str},
                                        keep_default_na=False), corpus)
    lemma_freq["n"] = lemma_freq["n"].astype(int)
    eds = edition_order(pt["edition"].unique())
    tables = {}

    # 1. corpus overview
    pages = load_pages()
    pages = inside(pages) if not pages.empty else pages
    conf = pages.groupby("edition")["mean_conf"].mean() if "mean_conf" in pages else pd.Series(dtype=float)
    ov = pt.groupby("edition").agg(books=("code", "nunique"), pages=("page", "size"), tokens=("n_tokens", "sum"))
    ov["mean_ocr_conf"] = conf.reindex(ov.index)
    tables["overview"] = ov.reindex(eds).reset_index()

    # 2. family rates per 10k tokens, excluding settlement 'pueblo'
    counted = ptok[ptok["counted"]]
    dropped = ptok[~ptok["counted"]]
    drop_keys = set(zip(dropped["sent_id"], dropped["token_i"].astype(int)))
    is_dropped = [f in ("pueblo", "comunidad") and (s, int(i)) in drop_keys
                  for f, s, i in zip(matches["family"], matches["sent_id"], matches["token_i"])]
    m = matches[~np.array(is_dropped, dtype=bool)]
    per_book_tok = pt.groupby(["edition", "code"])["n_tokens"].sum()
    fam_counts = m.groupby(["edition", "code", "family"]).size().unstack(fill_value=0)
    per_book = per_book_tok.to_frame().join(fam_counts, how="left").fillna(0).reset_index()
    people_any = counted.groupby(["edition", "code"]).size().rename("people_all")
    per_book = per_book.merge(people_any, on=["edition", "code"], how="left").fillna({"people_all": 0})
    fams = [f for f in list(L.FAMILIES) + ["people_all"] if f in per_book]
    rates = rate_table(per_book, fams, "n_tokens", 10_000, B)
    rates = _with_ci(rates)
    tables["rates"] = rates.pivot(index="measure", columns="edition", values="value").reindex(
        columns=eds).reindex([f for f in ["people_all"] + list(L.FAMILIES) if f in fams]).reset_index()
    rates.to_csv(out_dir() / f"rates_per_10k{sfx}.csv", index=False)

    # 3. top terms per family and edition
    tt = m.groupby(["family", "edition", "term"]).size().rename("n").reset_index()
    tt = tt.sort_values("n", ascending=False).groupby(["family", "edition"]).head(5)
    tables["top_terms"] = (tt.assign(t=lambda d: d["term"] + " (" + d["n"].astype(str) + ")")
                             .groupby(["family", "edition"])["t"].apply(", ".join).unstack()
                             .reindex(columns=eds).reset_index())

    # 4. senses of 'pueblo'
    pb = ptok[ptok["lemma"] == "pueblo"]
    tables["pueblo_senses"] = pd.crosstab(pb["sense"], pb["edition"]).reindex(columns=eds).reset_index()
    cb = ptok[ptok["lemma"].isin(COMMUNITY_LEMMAS)]
    if not cb.empty:
        tables["community_senses"] = pd.crosstab([cb["lemma"], cb["sense"]], cb["edition"]).reindex(
            columns=eds).fillna(0).astype(int).reset_index()

    # 5. how the people appear: number, determiner, role, verb class, adjectives
    pp = pb[pb["sense"].str.startswith("people")]
    def share(df, col, by="edition"):
        t = pd.crosstab(df[col], df[by], normalize="columns").reindex(columns=eds)
        return (100 * t).round(1).reset_index()
    tables["pueblo_number"] = share(pp, "number")
    tables["pueblo_role"] = share(pp, "role")
    tables["pueblo_verb_class"] = share(pp[pp["role"].isin(["agent", "patient"])], "verb_class")
    by_fam = counted.assign(family=counted["families"].str.split("+")).explode("family")   # a token counts in each of its families
    tables["people_role_by_family"] = (100 * pd.crosstab([by_fam["family"], by_fam["edition"]], by_fam["role"],
                                                         normalize="index")).round(1).reset_index()
    agency = counted.assign(agent=counted["role"] == "agent",
                            patient=counted["role"] == "patient").groupby(["edition", "code"])[["agent", "patient"]].sum()
    agency["mentions"] = counted.groupby(["edition", "code"]).size()
    ag = rate_table(agency.reset_index(), ["agent", "patient"], "mentions", 100, B)
    tables["agency"] = _with_ci(ag, 1).pivot(index="measure", columns="edition", values="value").reindex(columns=eds).reset_index()

    def top_list(df, col, k=12):
        s = df[df[col] != ""].assign(v=lambda d: d[col].str.split("+")).explode("v")
        s = s.groupby(["edition", "v"]).size().rename("n").reset_index().sort_values("n", ascending=False)
        return (s.groupby("edition").head(k).assign(t=lambda d: d["v"] + " (" + d["n"].astype(str) + ")")
                 .groupby("edition")["t"].apply(", ".join).reindex(eds).rename("top (count)").reset_index())
    tables["pueblo_attributes"] = top_list(pp, "attributes")
    tables["pueblo_agent_verbs"] = top_list(pp[pp["role"] == "agent"].assign(v=lambda d: d["head_lemma"]), "v")
    tables["pueblo_determiners"] = top_list(pp, "det")

    # 6. what people-talk sentences contain
    flags = [c for c in sents.columns if c.startswith("has_")]
    sb = sents.groupby(["edition", "code"])[flags].sum()
    sb["people_sentences"] = sents.groupby(["edition", "code"]).size()
    co = rate_table(sb.reset_index(), flags, "people_sentences", 100, B)
    tables["cooccurrence"] = _with_ci(co, 1).pivot(index="measure", columns="edition", values="value").reindex(columns=eds).reset_index()
    co.to_csv(out_dir() / f"people_sentence_shares{sfx}.csv", index=False)

    # 7. by subject and grade
    for col in ("subject", "grade"):
        if col in pt and pt[col].notna().any():
            den = pt.groupby(["edition", col])["n_tokens"].sum()
            num = counted.groupby(["edition", col]).size()
            t = (10_000 * num / den).rename("people_per_10k").reset_index().fillna(0)
            tables[f"by_{col}"] = t.pivot(index=col, columns="edition", values="people_per_10k").reindex(columns=eds).round(1).reset_index()

    # 7b. boilerplate: people-talk sentences the books repeat (formulas inflate rates; read them)
    rep = (sents.assign(norm=sents["text"].map(lambda t: re.sub(r"\W+", " ", fold(str(t))).strip()))
                .groupby(["edition", "norm"]).agg(n=("sent_id", "size"), books=("code", "nunique"),
                                                  example=("text", "first")).reset_index())
    rep = rep[rep["n"] >= 5].sort_values(["edition", "n"], ascending=[True, False]).groupby("edition").head(8)
    tables["repeated"] = rep[["edition", "n", "books", "example"]].assign(
        example=lambda d: d["example"].str.replace(r"\s+", " ", regex=True).str[:140])

    # 8. keyness between consecutive editions, 9. collocates
    kk = [keyness(lemma_freq, a, b).assign(comparison=f"{a} vs {b}") for a, b in zip(eds, eds[1:])]
    kk = pd.concat(kk) if kk else pd.DataFrame()
    kk.to_csv(out_dir() / f"keyness{sfx}.csv", index=False)
    col = collocates(ptok, lemma_freq)
    col.to_csv(out_dir() / f"pueblo_collocates{sfx}.csv", index=False)
    tables["collocates"] = (col.assign(t=lambda d: d["lemma"] + " (" + d["log_dice"].round(1).astype(str) + ")")
                              .groupby("edition")["t"].apply(", ".join).reindex(eds).rename("top (logDice)").reset_index()
                            if not col.empty else col)

    write_report(tables, kk, eds, corpus, sfx)
    return tables


SECTIONS = [
    ("overview", "Corpus", "Books, pages and alphabetic tokens per edition. Mean OCR confidence is Tesseract's 0-100 word confidence; below ~80 read the KWIC lines before trusting a rate."),
    ("rates", "People-talk per 10,000 tokens", "Ratio of sums with 95% CIs from resampling books. `people_all` counts people nouns once each, with settlement uses of *pueblo* removed; families overlap ('pueblos indígenas' is both *pueblo* and *etnico*)."),
    ("top_terms", "Most frequent terms per family", ""),
    ("pueblo_senses", "Senses of *pueblo*", "From the `senses` classifier (BETO + logistic regression; CV accuracy in "
     "`senses_cv.txt`) if it has run, otherwise from the rules. Validate with `python -m ltg sample` before reporting."),
    ("community_senses", "Senses of *comunidad* / *asamblea*",
     "classroom = 'en (pequeñas) comunidades', 'en asamblea' (NEM's way of organising group work); school = "
     "'comunidad escolar / de aula'; local = 'tu / su comunidad'. classroom and school are not counted as people-talk."),
    ("pueblo_number", "*Pueblo* (people senses): singular vs plural, % of mentions", "Singular *el pueblo* constructs one people; plural *los pueblos* constructs many."),
    ("pueblo_determiners", "*Pueblo* (people senses): determiners", ""),
    ("pueblo_role", "*Pueblo* (people senses): grammatical role, %", "agent = active subject; patient = passive subject or object of a victimising verb; attribute = subject of a copular predicate; possessor = 'X del pueblo'."),
    ("pueblo_verb_class", "*Pueblo* as agent or patient: what the people do or suffer, %", ""),
    ("pueblo_agent_verbs", "*Pueblo* as agent: top verbs", ""),
    ("pueblo_attributes", "*Pueblo*: adjectives and predicates", ""),
    ("agency", "All people nouns: % of mentions as agent / patient", "95% CIs from resampling books."),
    ("people_role_by_family", "Role by people family, % of mentions", ""),
    ("cooccurrence", "What people-talk sentences also contain, % of sentences", "has_elite + has_agravio approximates a people-vs-elite frame; has_unidad vs has_pluralismo a homogeneous vs plural people."),
    ("repeated", "People-talk sentences repeated 5+ times (formulas)",
     "Boilerplate instructions repeated across a book inflate rates; these are kept, but read them before interpreting a rate."),
    ("by_subject", "People-talk per 10k tokens by subject", ""),
    ("by_grade", "People-talk per 10k tokens by grade", ""),
    ("collocates", "Collocates of *pueblo* (people senses), logDice", "Within 5 tokens, same sentence."),
]


def write_report(tables, kk, eds, corpus="national", sfx=""):
    lines = [f"# 'The people' in the free textbooks: text report ({corpus} corpus)", "",
             f"Generated by `python -m ltg report --corpus {corpus}`. Every table is also in `results/text/` as CSV.",
             "Editions: " + ", ".join(eds) + ". Editions 2023 onwards largely reprint the same NEM titles, "
             "so they are not independent samples.", ""]
    for key, title, note in SECTIONS:
        if key not in tables:
            continue
        lines += [f"## {title}", ""] + ([note, ""] if note else []) + [md_table(tables[key]), ""]
    if not kk.empty:
        lines += ["## Keyness between consecutive editions (G2 >= 15.13, p < .0001)", ""]
        for comp, g in kk.groupby("comparison", sort=False):
            for side, gg in g.groupby("key_in", sort=False):
                lines += [f"**{comp}, key in {side}:** " + ", ".join(gg["lemma"].head(25)), ""]
    (out_dir() / f"report{sfx}.md").write_text("\n".join(lines), encoding="utf-8")
    log.info("wrote %s", out_dir() / f"report{sfx}.md")


# --------------------------------------------------------------------------- exports
def stratified_sample(df, keys, n, seed=0):
    """Up to n rows from every combination of `keys`, shuffled (so coders cannot infer the strata)."""
    parts = [g.sample(min(len(g), n), random_state=seed) for _, g in df.groupby(keys, sort=True)]
    return pd.concat(parts).sample(frac=1, random_state=seed) if parts else df.iloc[:0]


def export_for_coding(n_pueblo=30, n_community=10, n_role=15, seed=0):
    """Validation samples + the people-talk sentences for the APSI populism scorer.

    Strata are periods (1988 / 2022 / NEM), not editions: 2023-2025 largely reprint the
    same books. Default sizes give about 600 sense items (30 'pueblo' per period x sense;
    10 per period x word x sense for comunidad / asamblea) and about 360 role items
    (15 per period x role), a workload one coder can finish. National corpus only."""
    ptok = pd.read_csv(out_dir() / "people_tokens.csv.gz", dtype={"edition": str}, keep_default_na=False)
    ptok = _in_corpus(ptok, "national")
    sents = people_talk(_in_corpus(pd.read_csv(out_dir() / "sentences.csv.gz", dtype={"edition": str}), "national"), ptok)
    vdir = PATHS["results"] / "validation"
    vdir.mkdir(parents=True, exist_ok=True)
    s = pd.concat([
        stratified_sample(ptok[ptok["lemma"] == "pueblo"], ["period", "sense"], n_pueblo, seed),
        stratified_sample(ptok[ptok["lemma"].isin({"comunidad", "asamblea"})], ["period", "lemma", "sense"],
                          n_community, seed)]).sample(frac=1, random_state=seed)
    s[["sent_id", "edition", "period", "code", "page", "lemma", "left", "keyword", "right", "sense"]].rename(
        columns={"sense": "predicted_sense"}).assign(gold_sense="").to_csv(vdir / "sense_sample.csv", index=False)
    r = stratified_sample(ptok[ptok["counted"].astype(str) == "True"], ["period", "role"], n_role, seed)
    r[["sent_id", "edition", "period", "left", "keyword", "right", "role", "head_lemma", "verb_class"]].rename(
        columns={"role": "predicted_role"}).assign(gold_role="").to_csv(vdir / "role_sample.csv", index=False)
    apsi = sents.rename(columns={"sent_id": "id"})[["id", "edition", "period", "code", "page", "text"]]
    apsi.assign(text_en="").to_csv(PATHS["results"] / "people_sentences_for_apsi.csv", index=False)
    log.info("wrote %d sense and %d role items to %s, and %d sentences for APSI", len(s), len(r), vdir, len(apsi))


def validate(sample_csv, pred_col, gold_col):
    """Accuracy, per-class precision/recall and Cohen's kappa for a hand-coded sample."""
    df = pd.read_csv(sample_csv, keep_default_na=False)
    df = df[df[gold_col].astype(str).str.strip() != ""]
    if df.empty:
        raise SystemExit(f"no rows with {gold_col} filled in")
    y, p = df[gold_col].str.strip(), df[pred_col].str.strip()
    labels = sorted(set(y) | set(p))
    acc = float((y == p).mean())
    pe = sum((y == l).mean() * (p == l).mean() for l in labels)
    kappa = (acc - pe) / (1 - pe) if pe < 1 else np.nan
    per = pd.DataFrame([dict(label=l, n_gold=int((y == l).sum()), n_pred=int((p == l).sum()),
                             precision=((y == l) & (p == l)).sum() / max((p == l).sum(), 1),
                             recall=((y == l) & (p == l)).sum() / max((y == l).sum(), 1)) for l in labels])
    print(f"n={len(df)}  accuracy={acc:.3f}  kappa={kappa:.3f}")
    print(per.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(pd.crosstab(y.rename("gold"), p.rename("predicted")))
    return acc, kappa, per


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--editions", default="")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args(argv)
    parse(set(filter(None, args.editions.split(","))) or None, workers=args.workers)
