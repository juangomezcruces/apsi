#!/usr/bin/env python3
"""
Step 3 - Text analysis: cleaning, relevance filtering and classification.

Implements Methodology Sections 7.2-7.4 of the report for the YouTube data:

  clean       window 2024-01-01..2025-12-31, de-duplicate, drop empty texts
  channel     channel-country filter: drop channels whose registration country
              is known and is not the target country
  relevance   embedding relevance score (multilingual sentence-transformer);
              videos below the threshold are dropped
  topic       zero-shot topic (8 topics + Off-Topic) with mDeBERTa-v3-mnli-xnli,
              incl. the low-confidence tie-break rule -> Off-Topic
  toxicity    zero-shot toxic vs non-toxic: the two descriptions are weighed
              directly against each other (no numeric cut-off)
  narrative   toxic subset only: five negative-narrative frames scored
              independently (multi-label), highest wins; "Other" if none fits

All prompts / thresholds live in config.py. Model outputs are cached per text
in data/processed/cache/, so re-runs only classify new or changed texts.

Usage
-----
    python 03_text_analysis.py                     # title + description
    python 03_text_analysis.py --use-transcripts   # + transcript from step 2
    python 03_text_analysis.py --calibrate         # inspect relevance scores, then stop
    python 03_text_analysis.py --relevance-threshold 0.35 --device cuda

Output
------
    data/processed/youtube_classified_all.csv   every collected row + flags + labels
    data/processed/youtube_main_dataset.csv     the main (relevant, on-topic) dataset
    outputs/tables/00_filtering_funnel.csv      how many rows each step removed
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

CACHE_DIR = C.PROCESSED_DIR / "cache"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def pick_device(device: str) -> str:
    if device != "auto":
        return device
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def text_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:20]


def build_text(df: pd.DataFrame, use_transcripts: bool, max_chars: int,
               transcript_chars: int) -> pd.Series:
    parts = df["title"].fillna("").str.strip() + ". " + df["description"].fillna("").str.strip()
    if use_transcripts and "transcript" in df:
        tr = df["transcript"].fillna("").str.slice(0, transcript_chars)
        parts = parts + np.where(tr.str.len() > 0, " Transcript: " + tr, "")
    return (parts.str.replace(r"http\S+", " ", regex=True)     # URLs carry no meaning
                 .str.replace(r"\s+", " ", regex=True)
                 .str.strip(" .")
                 .str.slice(0, max_chars))


class Cache:
    """Tiny key -> json-result cache, one jsonl file per stage."""

    def __init__(self, stage: str, signature: str):
        self.path = CACHE_DIR / f"{stage}_{hashlib.sha1(signature.encode()).hexdigest()[:8]}.jsonl"
        self.data: dict[str, dict] = {}
        if self.path.exists():
            with self.path.open(encoding="utf-8") as fh:
                for line in fh:
                    d = json.loads(line)
                    self.data[d.pop("_key")] = d

    def missing(self, keys) -> list[str]:
        return [k for k in dict.fromkeys(keys) if k not in self.data]

    def add(self, results: dict[str, dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            for k, v in results.items():
                self.data[k] = v
                fh.write(json.dumps({"_key": k, **v}, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
class ZeroShot:
    def __init__(self, model: str, device: str, batch_size: int):
        from transformers import pipeline
        print(f"Loading NLI model {model} on {device}")
        self.pipe = pipeline("zero-shot-classification", model=model, device=device)
        self.batch_size = batch_size

    def scores(self, texts: list[str], hypotheses: list[str], multi_label: bool) -> list[dict[str, float]]:
        from tqdm import tqdm
        out = []
        for i in tqdm(range(0, len(texts), self.batch_size), unit="batch", leave=False):
            chunk = texts[i:i + self.batch_size]
            res = self.pipe(chunk, candidate_labels=hypotheses, hypothesis_template="{}",
                            multi_label=multi_label, batch_size=self.batch_size)
            res = [res] if isinstance(res, dict) else res
            out += [dict(zip(r["labels"], r["scores"])) for r in res]
        return out


def run_stage(stage: str, texts: pd.Series, labels: dict[str, str], multi_label: bool,
              zs_factory, model_name: str) -> pd.DataFrame:
    """Return a frame (index = texts.index) with one score column per label name."""
    hyp_to_name = {h: n for n, h in labels.items()}
    signature = json.dumps([model_name, labels, multi_label], ensure_ascii=False)
    cache = Cache(stage, signature)
    keys = texts.map(text_key)
    todo = cache.missing(keys)
    if todo:
        key_to_text = dict(zip(keys, texts))
        print(f"  {stage}: classifying {len(todo):,} texts ({len(keys) - len(todo):,} cached)")
        zs = zs_factory()
        step = 256   # flush to cache regularly so an interrupted run loses little
        for i in range(0, len(todo), step):
            ks = todo[i:i + step]
            res = zs.scores([key_to_text[k] for k in ks], list(hyp_to_name), multi_label)
            cache.add({k: {hyp_to_name[h]: round(float(s), 5) for h, s in r.items()}
                       for k, r in zip(ks, res)})
    else:
        print(f"  {stage}: all {len(keys):,} texts cached")
    return pd.DataFrame([cache.data[k] for k in keys], index=texts.index)[list(labels)]


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------
def clean(df: pd.DataFrame, args) -> pd.DataFrame:
    df = df.copy()
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df["drop_reason"] = pd.NA

    def drop(mask, reason):
        df.loc[mask & df["drop_reason"].isna(), "drop_reason"] = reason

    start = pd.Timestamp(C.START_DATE, tz="UTC")
    end = pd.Timestamp(C.END_DATE, tz="UTC") + pd.Timedelta(days=1)
    drop(~df["published_at"].between(start, end, inclusive="left"), "outside window")
    drop(df.duplicated(["id", "country"]), "duplicate")
    drop(df["text"].str.len() < args.min_chars, "empty / too short")

    region = df["country"].map({k: v["region_code"] for k, v in C.COUNTRIES.items()})
    ch = df["channel_country"].fillna("").str.upper()
    drop((ch != "") & (ch != region), "channel registered outside country")
    return df


def relevance(df: pd.DataFrame, args, device: str) -> pd.Series:
    from sentence_transformers import SentenceTransformer
    cache = Cache("relevance", json.dumps([C.EMBEDDING_MODEL, C.RELEVANCE_ANCHORS], ensure_ascii=False))
    keys = df["text"].map(text_key)
    todo = cache.missing(keys)
    if todo:
        print(f"  relevance: embedding {len(todo):,} texts")
        model = SentenceTransformer(C.EMBEDDING_MODEL, device=device)
        anchors = model.encode(C.RELEVANCE_ANCHORS, normalize_embeddings=True)
        key_to_text = dict(zip(keys, df["text"]))
        emb = model.encode([key_to_text[k] for k in todo], normalize_embeddings=True,
                           batch_size=args.batch_size, show_progress_bar=True)
        sims = emb @ anchors.T
        cache.add({k: {"score": round(float(s.max()), 5), "anchor": int(s.argmax())}
                   for k, s in zip(todo, sims)})
    return keys.map(lambda k: cache.data[k]["score"])


def assign_topic(scores: pd.DataFrame) -> pd.DataFrame:
    arr = scores.to_numpy()
    order = np.argsort(-arr, axis=1)
    top, second = arr[np.arange(len(arr)), order[:, 0]], arr[np.arange(len(arr)), order[:, 1]]
    names = np.array(scores.columns)
    topic = names[order[:, 0]].astype(object)
    tie = (top < C.TOPIC_MIN_CONF) & ((top - second) <= C.TOPIC_TIE_MARGIN)
    topic[tie] = C.OFF_TOPIC
    return pd.DataFrame({
        "topic": topic, "topic_score": top.round(4),
        "topic_runner_up": names[order[:, 1]], "topic_runner_up_score": second.round(4),
        "topic_tie_break": tie,
    }, index=scores.index)


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=C.COLLECTED_CSV)
    ap.add_argument("--transcripts", type=Path, default=C.TRANSCRIPTS_CSV)
    ap.add_argument("--use-transcripts", action="store_true",
                    help="append the step-2 transcript to title + description")
    ap.add_argument("--max-chars", type=int, default=2000, help="truncate classified text")
    ap.add_argument("--transcript-chars", type=int, default=1500)
    ap.add_argument("--min-chars", type=int, default=20)
    ap.add_argument("--relevance-threshold", type=float, default=C.RELEVANCE_THRESHOLD)
    ap.add_argument("--skip-relevance", action="store_true", help="keep every cleaned video")
    ap.add_argument("--calibrate", action="store_true",
                    help="write relevance-score samples for manual checking and stop")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    device = pick_device(args.device)
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df):,} rows from {args.input}")

    if args.use_transcripts:
        if not args.transcripts.exists():
            raise SystemExit(f"--use-transcripts given but {args.transcripts} not found (run step 2)")
        tr = pd.read_csv(args.transcripts)
        if "status" in tr:
            tr = tr[tr["status"] == "ok"]
        tr = tr[["id", "text"]].drop_duplicates("id").rename(columns={"text": "transcript"})
        df = df.merge(tr, on="id", how="left")
        print(f"  transcripts available for {df['transcript'].notna().sum():,} rows")

    df["text"] = build_text(df, args.use_transcripts, args.max_chars, args.transcript_chars)
    df = clean(df, args)
    funnel = [("collected", len(df))]
    for reason in ["outside window", "duplicate", "empty / too short",
                   "channel registered outside country"]:
        funnel.append((f"- {reason}", -(df["drop_reason"] == reason).sum()))

    # ---- relevance --------------------------------------------------------
    keep = df["drop_reason"].isna()
    df["relevance_score"] = np.nan
    if not args.skip_relevance:
        df.loc[keep, "relevance_score"] = relevance(df[keep], args, device)
        if args.calibrate:
            sub = df[keep].copy()
            print("\nRelevance-score quantiles by country:")
            print(sub.groupby("country")["relevance_score"]
                     .describe(percentiles=[.1, .25, .5, .75, .9]).round(3).to_string())
            sub["band"] = pd.cut(sub["relevance_score"], bins=np.arange(0, 1.05, 0.05))
            sample = (sub.groupby("band", observed=True, group_keys=False)
                         .apply(lambda g: g.sample(min(len(g), 10), random_state=1)))
            out = C.PROCESSED_DIR / "relevance_calibration_sample.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            sample[["country", "relevance_score", "title", "text", "url"]].sort_values(
                "relevance_score").to_csv(out, index=False)
            print(f"\nWrote {len(sample)} examples by score band -> {out}\n"
                  "Mark which are on-topic, pick the threshold, then pass --relevance-threshold.")
            return
        low = keep & (df["relevance_score"] < args.relevance_threshold)
        df.loc[low, "drop_reason"] = "below relevance threshold"
        funnel.append(("- below relevance threshold", -int(low.sum())))

    # ---- zero-shot classification -----------------------------------------
    zs_holder: dict = {}

    def zs_factory():
        if "zs" not in zs_holder:
            zs_holder["zs"] = ZeroShot(C.NLI_MODEL, device, args.batch_size)
        return zs_holder["zs"]

    keep = df["drop_reason"].isna()
    topic_labels = {**C.TOPICS, C.OFF_TOPIC: C.OFF_TOPIC_HYPOTHESIS}
    tscores = run_stage("topic", df.loc[keep, "text"], topic_labels, False, zs_factory, C.NLI_MODEL)
    df = df.join(assign_topic(tscores))
    df["topic_scores"] = pd.Series({i: json.dumps(r.round(4).to_dict()) for i, r in tscores.iterrows()})
    off = keep & (df["topic"] == C.OFF_TOPIC)
    df.loc[off, "drop_reason"] = "off-topic"
    funnel.append(("- off-topic", -int(off.sum())))

    main_mask = df["drop_reason"].isna()
    tox = run_stage("toxicity", df.loc[main_mask, "text"],
                    {"toxic": C.TOXIC_HYPOTHESIS, "non_toxic": C.NON_TOXIC_HYPOTHESIS},
                    False, zs_factory, C.NLI_MODEL)
    df["p_toxic"] = tox["toxic"].round(4)
    df["toxic"] = pd.Series(pd.NA, index=df.index, dtype="boolean")
    df.loc[main_mask, "toxic"] = tox["toxic"] > tox["non_toxic"]

    tox_mask = main_mask & (df["toxic"] == True)  # noqa: E712 (nullable boolean)
    nscores = run_stage("narrative", df.loc[tox_mask, "text"], C.NARRATIVES, True, zs_factory, C.NLI_MODEL)
    best = nscores.idxmax(axis=1)
    best_score = nscores.max(axis=1)
    df["narrative"] = best.where(best_score >= C.NARRATIVE_MIN_SCORE, C.NARRATIVE_OTHER)
    df["narrative_score"] = best_score.round(4)
    df["narrative_scores"] = pd.Series({i: json.dumps(r.round(4).to_dict()) for i, r in nscores.iterrows()})

    # ---- write ------------------------------------------------------------
    df["in_main_dataset"] = main_mask
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    C.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.CLASSIFIED_CSV, index=False)
    main_df = df[main_mask].drop(columns=["drop_reason", "in_main_dataset"])
    main_df.to_csv(C.MAIN_CSV, index=False)

    funnel += [("main dataset", int(main_mask.sum())), ("  of which toxic", int(tox_mask.sum()))]
    funnel_df = pd.DataFrame(funnel, columns=["step", "rows"])
    funnel_df.to_csv(C.TABLE_DIR / "00_filtering_funnel.csv", index=False)
    print("\nFiltering funnel:\n" + funnel_df.to_string(index=False))
    print(f"\nMain dataset by country:\n{main_df.groupby('country').size().to_string()}")
    print(f"\nWrote {C.CLASSIFIED_CSV}\n      {C.MAIN_CSV}")


if __name__ == "__main__":
    main()
