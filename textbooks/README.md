# 'The people' in Mexico's free textbooks

A pipeline for the text and visual analysis of the primary-school *Libros de
Texto Gratuitos* (CONALITEG), built for a paper on how the books depict
*el pueblo*. See [`RESEARCH_DESIGN.md`](RESEARCH_DESIGN.md) for the concepts,
measures, validation plan and threats to validity.

> **Status.** The code is tested end to end on synthetic books served from a
> local mock of both CONALITEG servers (26 tests). It has **not yet been run on
> the real books**: `libros.conaliteg.gob.mx` and `historico.conaliteg.gob.mx`
> were blocked by the network policy of the environment it was written in, so
> the live catalog markup has not been checked. The URL scheme comes from a
> public downloader ([jlochoap/LTGbasic](https://github.com/jlochoap/LTGbasic)).
> Start with the pilot below and read `data/catalog.csv` before a full run.

## Corpus

| edition | catalog page | period |
|---|---|---|
| 1988 | https://historico.conaliteg.gob.mx/?g=1988&a=1 (expanded to `a=1..6`) | pre-1993 generation |
| 2022 | https://libros.conaliteg.gob.mx/primaria_2022.html | pre-NEM |
| 2023 | https://libros.conaliteg.gob.mx/primaria_2023.html | NEM |
| 2024 | https://libros.conaliteg.gob.mx/primaria_2024.html | NEM |
| 2025 | https://libros.conaliteg.gob.mx/primaria_2025.html | NEM |
| current | https://libros.conaliteg.gob.mx/primaria.html | NEM (school year on the crawl date) |

Edit `ltg/config.py` to add editions (e.g. other years of the historical archive).

## Install

```bash
sudo apt-get install tesseract-ocr tesseract-ocr-spa        # macOS: brew install tesseract tesseract-lang
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install https://github.com/explosion/spacy-models/releases/download/es_core_news_sm-3.8.0/es_core_news_sm-3.8.0-py3-none-any.whl
python -m pytest tests/                                        # 26 tests, no network
```

For the zero-shot visual codebook also `pip install torch transformers` (the
first run downloads CLIP from huggingface.co).

## Run

All commands run from this directory. Data goes to `data/` (git-ignored), or to `$LTG_DATA`.

```bash
# 1. Pilot: catalog, then the first 10 pages of two books per edition
python -m ltg catalog
python -m ltg download --max-pages 10 --codes <two codes from data/catalog.csv>
python -m ltg extract
#    -> read data/catalog.csv and data/text/*/*.jsonl before going further

# 2. Full corpus (tens of thousands of requests at 2 workers x 0.5 s: hours, resumable)
python -m ltg download
python -m ltg extract --workers 4

# 3. Text
python -m ltg text --workers 4      # spaCy pass
python -m ltg report                # data/results/text/report.md + CSVs
python -m ltg sample                # hand-coding samples + sentences for APSI

# 4. Images
python -m ltg regions               # cut out illustrations + captions
python -m ltg zeroshot              # CLIP codebook (GPU helps; CPU works)
python -m ltg visual-sample         # hand-coding sheet (CSV + HTML contact sheet)
python -m ltg visual-report         # data/results/visual/report.md

# 5. Validate after hand-coding the gold_* columns
python -m ltg validate --kind sense
python -m ltg validate --kind role
python -m ltg validate --kind visual --dim people
```

**If `catalog` finds no books**, the catalog page probably builds its list in
JavaScript. Open it in a browser, find the request that returns the list
(DevTools > Network), and either point the edition's `seed` at it or write
`data/catalog.csv` by hand. The downloader only needs three columns:
`edition,code,viewer_url`.

## Stages and outputs

| stage | module | output |
|---|---|---|
| catalog | `catalog.py` | `catalog.csv`: edition, code, grade, level, title, subject (inferred from the title, fill blanks by hand), viewer and PDF URLs |
| download | `download.py` | `pages/<ed>/<CODE>/<nnn>.jpg` + `manifest.json` (first/last page, missing pages); `pdf/` when a PDF is linked |
| extract | `extract.py` | `text/<ed>/<CODE>.jsonl` (one record per page, with OCR confidence) and `boxes/` word boxes. PDF text layers are used instead of OCR when present |
| text | `text_analysis.py` | tidy tables in `results/text/`: every lexicon hit, every people noun with its sense, role, verb and adjectives and a KWIC window, every people-talk sentence with flags |
| report | `text_analysis.py` | `report.md` + `rates_per_10k.csv`, `people_sentence_shares.csv`, `keyness.csv`, `pueblo_collocates.csv` |
| regions | `visual.py` | `results/visual/regions.csv` (one row per illustration: size, colourfulness, caption, whether the caption names the people) and `visual/crops/` |
| zeroshot | `visual.py` | `zeroshot.csv`: label and probabilities on five codebook dimensions |

Every rate is a ratio of sums over books with a 95% interval from resampling
books, since pages and sentences within a book are not independent.

## The lexicon is the instrument

`ltg/lexicon.py` holds every word list (people families, antagonists,
grievance, unity, pluralism, institutions, verb classes, sense cues for
*pueblo*). These are research decisions: read them, change them, and re-run
`text` + `report` (minutes). Terms with a common non-political meaning in
textbooks (*unidad*, *diferencia*, *plural*, *masa*) are deliberately absent.
The keyness tables are there to catch what the lists miss.

## Linking to the APSI populism scorer

`python -m ltg sample` writes `data/results/people_sentences_for_apsi.csv`
(`id`, `text`, empty `text_en`). The scorer in this repository
(`classify_populism.py`) uses an English-only NLI model, and its own
docstring warns that scoring non-English text with it is not valid, so
translate first (e.g. `Helsinki-NLP/opus-mt-es-en`), then:

```bash
python ../classify_populism.py --input people_sentences_en.csv --output pop_scores.csv --text-col text_en --group-col edition
```

The scorer was validated on political speech, not children's textbooks;
treat its scores on this corpus as something to validate, not as ground truth.

## Practicalities

- **Terms of use.** The LTGbasic author reported in 2021 that the site added a
  notice about downloading. Check the current terms before a bulk download,
  keep the defaults polite, keep page images local and do not redistribute
  them. Quote and reproduce pages in the paper as you would any copyrighted
  source.
- **Running in Claude Code on the web**: allow `libros.conaliteg.gob.mx`,
  `historico.conaliteg.gob.mx` and, for CLIP, `huggingface.co` in the
  environment's network settings.
