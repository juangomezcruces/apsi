#!/usr/bin/env python3
"""
Step 2 - Audio-to-text for the collected YouTube videos.

For each video it (optionally) tries YouTube captions first, and otherwise
downloads the audio track with yt-dlp and transcribes it locally with
faster-whisper (Whisper; language auto-detected, so Spanish, English, Arabic,
French and Sinhala videos all work - Sinhala quality is the weakest, so check a
sample). Audio files are deleted after transcription unless --keep-audio.

The transcripts feed step 3 (`--use-transcripts`), which classifies
title + description + transcript instead of title + description only.

The script is resumable: videos with a transcript json already on disk are skipped.

Usage
-----
    python 02_transcribe_audio.py                          # all collected videos
    python 02_transcribe_audio.py --prefer-captions        # captions first, Whisper fallback
    python 02_transcribe_audio.py --model large-v3 --device cuda
    python 02_transcribe_audio.py --max-minutes 30 --translate
    python 02_transcribe_audio.py --cookies cookies.txt    # if YouTube asks to sign in
    python 02_transcribe_audio.py --audio-dir my_audio/    # transcribe <video_id>.* files you
                                                           # downloaded elsewhere (no YouTube access)

Output
------
    data/transcripts/<video_id>.json       segments, language, source
    data/interim/youtube_transcripts.csv   id, language, source, text, (text_en)

Note: downloading audio must respect YouTube's Terms of Service and your
institution's research-ethics approval for the project.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

import config as C


def iso_duration_minutes(value: str | float | None) -> float | None:
    """'PT1H2M30S' -> 62.5"""
    if not isinstance(value, str):
        return None
    m = re.fullmatch(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", value)
    if not m:
        return None
    d, h, mi, s = (int(x) if x else 0 for x in m.groups())
    return d * 1440 + h * 60 + mi + s / 60


# --------------------------------------------------------------------------
# Captions (cheap path)
# --------------------------------------------------------------------------
def vtt_to_text(vtt: str) -> str:
    lines, last = [], None
    for line in vtt.splitlines():
        line = line.strip()
        if (not line or line == "WEBVTT" or "-->" in line or line.isdigit()
                or line.startswith(("Kind:", "Language:", "NOTE", "STYLE"))):
            continue
        line = re.sub(r"<[^>]+>", "", line)          # inline timing / styling tags
        if line and line != last:                    # auto-captions repeat lines
            lines.append(line)
            last = line
    return " ".join(lines)


def fetch_captions(url: str, workdir: Path, langs: list[str], cookies: str | None) -> tuple[str, str] | None:
    import yt_dlp
    opts = {
        "skip_download": True, "writesubtitles": True, "writeautomaticsub": True,
        "subtitleslangs": langs, "subtitlesformat": "vtt", "quiet": True, "no_warnings": True,
        "outtmpl": str(workdir / "%(id)s.%(ext)s"),
    }
    if cookies:
        opts["cookiefile"] = cookies
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    for f in sorted(workdir.glob(f"{info['id']}*.vtt")):
        lang = f.suffixes[-2].lstrip(".") if len(f.suffixes) >= 2 else "und"
        text = vtt_to_text(f.read_text(encoding="utf-8", errors="ignore"))
        f.unlink()
        if len(text) > 50:
            return lang, text
    return None


# --------------------------------------------------------------------------
# Whisper path
# --------------------------------------------------------------------------
def download_audio(url: str, workdir: Path, cookies: str | None) -> Path:
    import yt_dlp
    opts = {
        "format": "bestaudio/best", "quiet": True, "no_warnings": True,
        "outtmpl": str(workdir / "%(id)s.%(ext)s"),
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "m4a"}],
    }
    if cookies:
        opts["cookiefile"] = cookies
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    path = workdir / f"{info['id']}.m4a"
    if not path.exists():
        cands = list(workdir.glob(f"{info['id']}.*"))
        if not cands:
            raise FileNotFoundError(f"audio for {info['id']} not found")
        path = cands[0]
    return path


class Transcriber:
    def __init__(self, model_size: str, device: str, compute_type: str | None):
        from faster_whisper import WhisperModel
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        compute_type = compute_type or ("float16" if device == "cuda" else "int8")
        print(f"Loading faster-whisper '{model_size}' on {device} ({compute_type})")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def run(self, audio: Path, max_minutes: float | None, task: str = "transcribe",
            language: str | None = None) -> dict:
        kwargs = dict(task=task, vad_filter=True, beam_size=5, language=language)
        if max_minutes:
            kwargs["clip_timestamps"] = [0, max_minutes * 60]
        segments, info = self.model.transcribe(str(audio), **kwargs)
        segs = [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
                for s in segments]
        return {
            "language": info.language,
            "language_probability": round(info.language_probability, 3),
            "audio_minutes": round(info.duration / 60, 2),
            "segments": segs,
            "text": " ".join(s["text"] for s in segs),
        }


# --------------------------------------------------------------------------
def build_csv() -> pd.DataFrame:
    rows = []
    for f in sorted(C.TRANSCRIPT_DIR.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        rows.append({k: d.get(k) for k in
                     ("id", "status", "source", "language", "language_probability",
                      "audio_minutes", "text", "text_en", "error")})
    df = pd.DataFrame(rows)
    C.INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.TRANSCRIPTS_CSV, index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=C.COLLECTED_CSV,
                    help="CSV with id + url columns (default: step-1 output)")
    ap.add_argument("--model", default=C.WHISPER_MODEL, help="faster-whisper model size")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--compute-type", default=None, help="e.g. int8, float16 (default by device)")
    ap.add_argument("--prefer-captions", action="store_true",
                    help="use YouTube captions when available; Whisper only as fallback")
    ap.add_argument("--max-minutes", type=float, default=None,
                    help="transcribe only the first N minutes of each video")
    ap.add_argument("--skip-longer-than", type=float, default=180,
                    help="skip videos longer than N minutes (default 180)")
    ap.add_argument("--translate", action="store_true",
                    help="also store an English translation (Whisper task=translate)")
    ap.add_argument("--countries", nargs="+", default=None, help="limit to these countries")
    ap.add_argument("--limit", type=int, default=None, help="process at most N videos (testing)")
    ap.add_argument("--cookies", default=None, help="Netscape cookies.txt for yt-dlp")
    ap.add_argument("--keep-audio", action="store_true")
    ap.add_argument("--audio-dir", type=Path, default=None,
                    help="folder of pre-downloaded audio named <video_id>.<ext>; used instead of "
                         "downloading (videos without a file there are skipped)")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    if args.countries:
        df = df[df["country"].isin(args.countries)]
    df = df.drop_duplicates("id")
    if "duration" in df:
        mins = df["duration"].map(iso_duration_minutes)
        too_long = mins > args.skip_longer_than
        if too_long.any():
            print(f"Skipping {too_long.sum()} videos longer than {args.skip_longer_than} min")
        df = df[~too_long]

    C.TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    C.AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    todo = [r for r in df.itertuples() if not (C.TRANSCRIPT_DIR / f"{r.id}.json").exists()]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(todo)} videos to transcribe ({len(df) - len(todo)} already done)")

    caption_langs = sorted({lang for c in C.COUNTRIES.values() for lang, _ in c["queries"]})
    caption_langs = [f"{l}.*" for l in caption_langs] + caption_langs
    transcriber = None

    for n, row in enumerate(todo, 1):
        url = getattr(row, "url", None) or f"https://www.youtube.com/watch?v={row.id}"
        out = {"id": row.id, "url": url, "status": "ok"}
        try:
            local = None
            if args.audio_dir:
                local = next((f for f in sorted(args.audio_dir.glob(f"{row.id}.*"))
                              if f.suffix.lower() not in {".part", ".json", ".vtt"}), None)
                if local is None:
                    print(f"[{n}/{len(todo)}] {row.id}: no audio file in {args.audio_dir}, skipped")
                    continue
            cap = (fetch_captions(url, C.AUDIO_DIR, caption_langs, args.cookies)
                   if args.prefer_captions and local is None else None)
            if cap:
                out.update(source="captions", language=cap[0], text=cap[1])
            else:
                if transcriber is None:
                    transcriber = Transcriber(args.model, args.device, args.compute_type)
                audio = local or download_audio(url, C.AUDIO_DIR, args.cookies)
                try:
                    res = transcriber.run(audio, args.max_minutes)
                    out.update(source=f"whisper-{args.model}", **res)
                    if args.translate and res["language"] != "en" and res["text"]:
                        tr = transcriber.run(audio, args.max_minutes, task="translate",
                                             language=res["language"])
                        out["text_en"] = tr["text"]
                finally:
                    if not args.keep_audio and local is None:
                        audio.unlink(missing_ok=True)
        except Exception as exc:   # keep going; failures go to failed.log and are retried on the next run
            out.update(status="failed", error=f"{type(exc).__name__}: {exc}"[:500])
        if out["status"] == "ok":
            (C.TRANSCRIPT_DIR / f"{row.id}.json").write_text(json.dumps(out, ensure_ascii=False),
                                                              encoding="utf-8")
        else:
            (C.TRANSCRIPT_DIR / "failed.log").open("a").write(json.dumps(out) + "\n")
        print(f"[{n}/{len(todo)}] {row.id}: {out['status']} "
              f"{out.get('source', '')} {out.get('language', '')} {len(out.get('text') or '')} chars")

    res = build_csv()
    print(f"\nWrote {len(res)} transcripts -> {C.TRANSCRIPTS_CSV}")


if __name__ == "__main__":
    main()
