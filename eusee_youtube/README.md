# EU SEE "Civil Society under Pressure": YouTube pipeline

These scripts reproduce the YouTube part of the EU SEE report *Civil Society
under Pressure: Public Online Discourse around CSOs after the U.S. Foreign-Aid
Freeze*. They cover collection, audio-to-text, text analysis, data analysis and
figures. The Facebook / Meta Content Library part is not included.
The steps follow the report's Methodology chapter (Section 7).

| Step | Script | What it does | Report section |
|---|---|---|---|
| 1 | `01_collect_youtube.py` | YouTube Data API v3 search per country × language × month, plus video stats and channel country. Resumable and quota-aware. | 7.1 |
| 2 | `02_transcribe_audio.py` | Audio-to-text: YouTube captions (optional), otherwise yt-dlp + faster-whisper. Language is auto-detected. Optional English translation. | (extension) |
| 3 | `03_text_analysis.py` | Cleaning, channel-country filter, embedding relevance filter, zero-shot **topics** (8 + Off-Topic, with the tie-break rule), **toxicity** (toxic vs non-toxic), **narratives** (toxic subset, 5 frames). | 7.2–7.4 |
| 4 | `04_data_analysis.py` | 17 CSV tables plus `outputs/summary.md`: topic shares, toxic share by year, country and topic, engagement, narratives, country comparison, topic growth vs toxicity, spreader typing, interrupted time series. | 2.4, 4, 5, 7.5–7.6, 8 |
| 5 | `05_make_figures.py` | The report's figures (4.1.1–4.7.2, the per-country 5.x.1–5.x.3 set, plus two research-question figures), drawn from the step-4 tables. | 4, 5 |

`config.py` holds everything shared by the scripts: countries and **search keywords** (`COUNTRIES`), the study window, the EO date, model names, every
classification prompt and threshold, and the account-typing rules.

## Setup

```bash
cd eusee_youtube
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # also needs ffmpeg on PATH for step 2
export YOUTUBE_API_KEY=...           # Google Cloud console → YouTube Data API v3
```

## Run

```bash
python 01_collect_youtube.py --dry-run     # plan + quota estimate
python 01_collect_youtube.py               # re-run daily until no searches are left
python 02_transcribe_audio.py --prefer-captions
python 03_text_analysis.py --calibrate     # look at relevance scores, choose a threshold
python 03_text_analysis.py --use-transcripts --relevance-threshold 0.30
python 04_data_analysis.py
python 05_make_figures.py --formats png svg
```

Each script reads the previous script's output from `data/` and writes to
`data/` or `outputs/`. Use `--help` on any script to see its options. You can
change the base folder with `EUSEE_BASE_DIR=/path`.

```
data/raw/youtube/                  raw API responses + collection_state.json
data/interim/youtube_videos_collected.csv
data/interim/youtube_transcripts.csv     (+ data/transcripts/<id>.json)
data/processed/youtube_classified_all.csv   every row, with drop_reason and labels
data/processed/youtube_main_dataset.csv     the main dataset (relevant, on-topic)
outputs/tables/00–17_*.csv, outputs/summary.md, outputs/figures/*.png
```

## Things to check before using the results

- **Keywords.** The report doesn't list its keyword sets. The `queries` in
  `config.COUNTRIES` are a starting point in the report's languages. Replace them
  with the study's actual lists.
- **Quota.** Each `search.list` page costs 100 units, and the default daily
  quota is 10,000. With the default monthly slices, the four countries need
  about 144 searches, or 14,400+ units, so plan for 2–3 days or ask Google for
  more quota. `--slice quarter` cuts the cost but returns fewer videos per
  period, because YouTube caps each query at about 500 results.
- **Prompts are the definitions.** The topic, toxicity and narrative
  hypotheses in `config.py` follow the report's wording. If the study used
  different exact wording, paste it in. Changing a prompt automatically
  invalidates the cache for that step. Zero-shot labels contain errors, so
  check a hand-coded sample, as Section 7.6 recommends, before relying on them.
- **Relevance threshold.** `--calibrate` writes
  `data/processed/relevance_calibration_sample.csv` with 10 videos from each
  score band. Mark which ones are on-topic and choose the cut-off from that.
- **Audio downloads.** YouTube often blocks downloads from servers and cloud
  IPs (HTTP 403). In that case, pass `--cookies cookies.txt`, run step 2 on a
  personal machine, or download the audio elsewhere and use
  `--audio-dir folder/` with files named `<video_id>.<ext>`. Downloading must
  comply with YouTube's Terms and with the project's ethics approval. Sinhala
  transcription quality in Whisper is weak, so check a sample.
- **YouTube has no share counts.** The `shares` column is empty, and the
  engagement figures show views, likes and comments.
- **Merging with Facebook.** Step 4 accepts more CSVs in the same schema
  through `--extra facebook_main.csv` (needed columns: `id, platform, country,
  published_at, topic, toxic, narrative, likes, comments, shares, views,
  channel_title`).
