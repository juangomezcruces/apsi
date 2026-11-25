#!/usr/bin/env python3
"""
Step 1 - Collect YouTube videos with the YouTube Data API v3.

For every country x language keyword set x time slice in the study window
(1 Jan 2024 - 31 Dec 2025) it runs `search.list`, then enriches every hit with
`videos.list` (title, description, tags, duration, views/likes/comments) and
`channels.list` (channel registration country, handle, description) - the
channel country is what the relevance step uses to drop out-of-country channels.

Quota
-----
search.list costs 100 units per page, videos.list / channels.list 1 unit per
50 ids. The default daily quota is 10,000 units. The script keeps a state file
and stops cleanly at --max-units or on `quotaExceeded`; just re-run it the next
day and it continues where it stopped (finished searches are not repeated).

Usage
-----
    export YOUTUBE_API_KEY=...            # or --api-key
    python 01_collect_youtube.py --dry-run            # show plan + quota estimate
    python 01_collect_youtube.py                      # run / resume
    python 01_collect_youtube.py --countries Peru "Sri Lanka" --slice quarter

Output
------
    data/raw/youtube/search/*.json     raw search pages
    data/raw/youtube/videos.jsonl      raw videos.list items
    data/raw/youtube/channels.jsonl    raw channels.list items
    data/interim/youtube_videos_collected.csv   one row per (video_id, country)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

import config as C

SEARCH_COST = 100
LIST_COST = 1
RAW_YT = C.RAW_DIR / "youtube"
STATE_FILE = RAW_YT / "collection_state.json"


class QuotaExhausted(Exception):
    pass


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------
def time_slices(start: date, end: date, size: str) -> list[tuple[date, date]]:
    """Split [start, end] into month / quarter / year slices (inclusive)."""
    step = {"month": 1, "quarter": 3, "year": 12}[size]
    out, cur = [], date(start.year, start.month, 1)
    while cur <= end:
        m = cur.month - 1 + step
        nxt = date(cur.year + m // 12, m % 12 + 1, 1)
        out.append((max(cur, start), min(nxt - timedelta(days=1), end)))
        cur = nxt
    return out


def build_plan(countries: list[str], slice_size: str, combine: bool) -> list[dict]:
    plan = []
    for country in countries:
        spec = C.COUNTRIES[country]
        for lang, keywords in spec["queries"]:
            queries = ["|".join(keywords)] if combine else list(keywords)
            for q in queries:
                for s, e in time_slices(C.START_DATE, C.END_DATE, slice_size):
                    task = {
                        "country": country, "region_code": spec["region_code"],
                        "lang": lang, "q": q, "after": s.isoformat(), "before": e.isoformat(),
                    }
                    task["key"] = hashlib.sha1(
                        json.dumps(task, sort_keys=True, ensure_ascii=False).encode()
                    ).hexdigest()[:16]
                    plan.append(task)
    return plan


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------
def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"done": {}, "hits": [], "units_by_day": {}}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False))
    tmp.replace(STATE_FILE)


# --------------------------------------------------------------------------
# API helpers
# --------------------------------------------------------------------------
class YouTube:
    def __init__(self, api_key: str, max_units: int, state: dict):
        from googleapiclient.discovery import build
        self.client = build("youtube", "v3", developerKey=api_key, cache_discovery=False)
        self.max_units = max_units
        self.state = state
        self.today = datetime.now(timezone.utc).strftime("%Y-%m-%d")  # quota resets ~midnight PT

    @property
    def used(self) -> int:
        return self.state["units_by_day"].get(self.today, 0)

    def _spend(self, units: int) -> None:
        if self.used + units > self.max_units:
            raise QuotaExhausted(f"budget of {self.max_units} units reached for {self.today}")
        self.state["units_by_day"][self.today] = self.used + units

    def call(self, request, cost: int, retries: int = 5):
        from googleapiclient.errors import HttpError
        self._spend(cost)
        for attempt in range(retries):
            try:
                return request.execute()
            except HttpError as err:
                reason = ""
                try:
                    reason = json.loads(err.content)["error"]["errors"][0]["reason"]
                except Exception:
                    pass
                if reason in {"quotaExceeded", "dailyLimitExceeded"}:
                    raise QuotaExhausted(reason) from err
                if err.resp.status in (500, 503) or reason in {"rateLimitExceeded", "backendError"}:
                    time.sleep(2 ** attempt)
                    continue
                raise
            except (ConnectionError, TimeoutError, OSError):
                time.sleep(2 ** attempt)
        raise RuntimeError("YouTube API call failed after retries")

    def search(self, task: dict, max_pages: int) -> list[str]:
        ids, token, page = [], None, 0
        while page < max_pages:
            params = dict(
                part="id", type="video", q=task["q"], maxResults=50, order="date",
                publishedAfter=f"{task['after']}T00:00:00Z",
                publishedBefore=f"{task['before']}T23:59:59Z",
                regionCode=task["region_code"], relevanceLanguage=task["lang"],
                safeSearch="none",
            )
            if token:
                params["pageToken"] = token
            resp = self.call(self.client.search().list(**params), SEARCH_COST)
            out = RAW_YT / "search" / f"{task['key']}_p{page}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(resp, ensure_ascii=False))
            ids += [it["id"]["videoId"] for it in resp.get("items", []) if "videoId" in it.get("id", {})]
            token = resp.get("nextPageToken")
            page += 1
            if not token:
                break
        return ids

    def list_in_batches(self, kind: str, ids: list[str], part: str, out_path: Path) -> None:
        have = set()
        if out_path.exists():
            with out_path.open(encoding="utf-8") as fh:
                have = {json.loads(line)["id"] for line in fh if line.strip()}
        todo = [i for i in dict.fromkeys(ids) if i not in have]
        resource = self.client.videos() if kind == "videos" else self.client.channels()
        with out_path.open("a", encoding="utf-8") as fh:
            for i in range(0, len(todo), 50):
                chunk = todo[i:i + 50]
                resp = self.call(resource.list(part=part, id=",".join(chunk), maxResults=50), LIST_COST)
                for item in resp.get("items", []):
                    fh.write(json.dumps(item, ensure_ascii=False) + "\n")
                save_state(self.state)
        print(f"  {kind}: {len(todo)} new ids fetched ({len(have)} already cached)")


# --------------------------------------------------------------------------
# Flatten to a table
# --------------------------------------------------------------------------
def read_jsonl(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return {d["id"]: d for d in (json.loads(l) for l in fh if l.strip())}


def flatten(state: dict) -> pd.DataFrame:
    videos = read_jsonl(RAW_YT / "videos.jsonl")
    channels = read_jsonl(RAW_YT / "channels.jsonl")
    rows = []
    for hit in state["hits"]:
        v = videos.get(hit["video_id"])
        if v is None:      # deleted / private since search
            continue
        sn, st, cd = v.get("snippet", {}), v.get("statistics", {}), v.get("contentDetails", {})
        ch = channels.get(sn.get("channelId"), {})
        chs, chst = ch.get("snippet", {}), ch.get("statistics", {})
        rows.append({
            "id": v["id"],
            "platform": "YouTube",
            "country": hit["country"],
            "query_language": hit["lang"],
            "query": hit["q"],
            "published_at": sn.get("publishedAt"),
            "channel_id": sn.get("channelId"),
            "channel_title": sn.get("channelTitle"),
            "channel_handle": chs.get("customUrl"),
            "channel_description": chs.get("description"),
            "channel_country": chs.get("country"),
            "channel_subscribers": pd.to_numeric(chst.get("subscriberCount"), errors="coerce"),
            "title": sn.get("title"),
            "description": sn.get("description"),
            "tags": "|".join(sn.get("tags", [])),
            "default_language": sn.get("defaultLanguage"),
            "default_audio_language": sn.get("defaultAudioLanguage"),
            "duration": cd.get("duration"),
            "views": pd.to_numeric(st.get("viewCount"), errors="coerce"),
            "likes": pd.to_numeric(st.get("likeCount"), errors="coerce"),
            "comments": pd.to_numeric(st.get("commentCount"), errors="coerce"),
            "shares": pd.NA,   # not exposed by the YouTube API
            "url": f"https://www.youtube.com/watch?v={v['id']}",
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(["id", "country"]).sort_values(["country", "published_at"])
    return df


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api-key", default=os.environ.get("YOUTUBE_API_KEY"))
    ap.add_argument("--countries", nargs="+", default=C.COUNTRY_ORDER, choices=list(C.COUNTRIES))
    ap.add_argument("--slice", default="month", choices=["month", "quarter", "year"],
                    help="time-slice size per search; smaller slices get round YouTube's ~500 results/query cap")
    ap.add_argument("--max-pages", type=int, default=10, help="pages (x50 results) per search slice")
    ap.add_argument("--max-units", type=int, default=9500, help="daily quota budget for this script")
    ap.add_argument("--no-combine", action="store_true",
                    help="one search per keyword instead of OR-ing keywords per language")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and quota estimate only")
    ap.add_argument("--flatten-only", action="store_true", help="rebuild the CSV from cached raw data")
    args = ap.parse_args()

    plan = build_plan(args.countries, args.slice, combine=not args.no_combine)
    state = load_state()
    pending = [t for t in plan if t["key"] not in state["done"]]

    print(f"Plan: {len(plan)} searches ({len(pending)} pending), "
          f"window {C.START_DATE} - {C.END_DATE}, slice={args.slice}")
    print(f"Minimum quota for pending searches: {len(pending) * SEARCH_COST:,} units "
          f"(+100 per extra results page, +1 per 50 videos/channels)")
    if args.dry_run:
        for t in plan[:8]:
            print("  ", t["country"], t["lang"], t["after"], "->", t["before"], "|", t["q"][:80])
        if len(plan) > 8:
            print(f"   ... {len(plan) - 8} more")
        return

    if not args.flatten_only:
        if not args.api_key:
            sys.exit("Set YOUTUBE_API_KEY or pass --api-key")
        yt = YouTube(args.api_key, args.max_units, state)
        try:
            for n, task in enumerate(pending, 1):
                ids = yt.search(task, args.max_pages)
                state["hits"] += [{"video_id": i, "country": task["country"], "lang": task["lang"],
                                   "q": task["q"]} for i in ids]
                state["done"][task["key"]] = len(ids)
                save_state(state)
                print(f"[{n}/{len(pending)}] {task['country']:<12} {task['lang']} "
                      f"{task['after']}..{task['before']}: {len(ids):>4} videos  "
                      f"(units today: {yt.used})")
            video_ids = [h["video_id"] for h in state["hits"]]
            yt.list_in_batches("videos", video_ids, "snippet,statistics,contentDetails",
                               RAW_YT / "videos.jsonl")
            vids = read_jsonl(RAW_YT / "videos.jsonl")
            channel_ids = [v["snippet"]["channelId"] for v in vids.values()]
            yt.list_in_batches("channels", channel_ids, "snippet,statistics", RAW_YT / "channels.jsonl")
        except QuotaExhausted as exc:
            save_state(state)
            print(f"\nStopped: {exc}. Re-run tomorrow to resume "
                  f"({sum(t['key'] not in state['done'] for t in plan)} searches left).")
            return

    df = flatten(state)
    C.INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.COLLECTED_CSV, index=False)
    print(f"\nWrote {len(df):,} rows ({df['id'].nunique() if len(df) else 0:,} unique videos) "
          f"-> {C.COLLECTED_CSV}")
    if len(df):
        print(df.groupby("country").size().rename("videos").to_string())


if __name__ == "__main__":
    main()
