# 'The people' in Mexico's free textbooks

A pipeline for the text and visual analysis of the primary-school *Libros de
Texto Gratuitos* (CONALITEG), built for a paper on how the books depict
*el pueblo*. See [`RESEARCH_DESIGN.md`](RESEARCH_DESIGN.md) for the concepts,
measures, validation plan and threats to validity, and
[`FINDINGS.md`](FINDINGS.md) for the preliminary results of the first full run.

`results/` holds the shareable outputs of that run (reports, aggregate tables,
figures, the catalog and the hand-coding samples); everything else, including
page images and OCR text, stays in the git-ignored `data/`.

## Corpus

Checked against the live sites on 2026-11-03/04. Student books of the national
corpus are analysed; teaching editions are excluded throughout; state books
(one per state) are downloaded only for 2022 and reported separately.

| edition | source | label | national student books | also listed (not in the national comparison) |
|---|---|---|---|---|
| 1960 | historical archive `?g=1960` | generation 1960 | 20 | archive note (not a book) |
| 1962 | `?g=1962` | generation 1962 | 38 | |
| 1972 | `?g=1972` | generation 1972 | 36 | |
| 1982 | `?g=1982` | generation 1982 | 28 | preschool workbook |
| 1988 | `?g=1988` | generation 1988 | 26 | preschool workbook |
| 1993 | `?g=1993` | generation 1993 | 50 | 64 state (*Historia y Geografía*, *Monografía estatal*) |
| 2008 | `?g=2008` | generation 2008 | 44 | |
| 2011 | `?g=2011` | generation 2011 | 51 | |
| 2014 | `?g=2014` | generation 2014 | 44 | 32 state, 10 teaching editions |
| 2018 | `?g=2018` | generation 2018 | 33 | |
| 2019 | `?g=2019` | generation 2019 | 45 | 32 state, 6 teaching editions |
| 2022 | `primaria_2022.html` | pre-NEM | 42 | 31 state (downloaded, reported apart) |
| 2023 | `primaria_2023.html` | NEM | 33 | 3 teaching editions |
| 2024 | `primaria_2024.html` | NEM | 36 | 3 teaching editions |
| 2025 | `primaria_2025.html` | NEM | 36 | 3 teaching editions |
| current | `primaria.html` | NEM, 2026-27 | not downloaded (encrypted reader, see below) | 3 teaching editions |

When each generation was in classrooms, according to the archive's own lookup
table (`g[year][grade]` in the same script, keyed by the year a child started
primary school; it stops with children who started in 2019):

| generation | school years in use (any grade) |
|---|---|
| 1960 | 1960-1961 (grades 1-2 in the lookup) |
| 1962 | 1962-1971 |
| 1972 | 1972-1981 |
| 1982 | 1982-1987 |
| 1988 | 1988-1992 |
| 1993 | 1993-2007 |
| 2008 | 2008-2010 |
| 2011 | 2011-2013 |
| 2014 | 2014-2022 (phased out grade by grade from 2018) |
| 2018 | 2018-2019 (grades 1-2) |
| 2019 | 2019-2025 in the lookup, phased in grade by grade |

The 2019-generation books and the 2022 catalog are largely the same titles, as
2023-2025 are reprints of one NEM set.

What the sites actually look like, and what the crawler does about it:

- **Historical archive.** `g` is the *generation* of books and `a` the grade.
  The list is built in the browser from a JavaScript array
  `["TITLE","CODE",generation,grade]` in a same-host script; the crawler reads
  that array and keeps the requested generation. Viewer pages declare no page
  count, so pages are found by probing.
- **Current site, 2022-2025.** Grade cards with the title in bold and
  "Grado: N°". One book can be listed under several grades (`P0CMA`,
  *Cartografía*, is in grades 4-6), so the card's grade, not the code's digit,
  is used. Viewers declare `ag_pages = N`. Page images are in `c/<CODE>/` for
  2022-2024 but in **`c2025/<CODE>/`** for 2025; the downloader reads the folder
  from each year's `magazine.js`.
- **Catalog flags.** `audience=teacher` for teaching editions (*Docente*, *Libro
  para el maestro*, *Fichero didáctico*, NEM's *Un libro sin recetas...*), never
  downloaded by default (`--audience all` to include them). `corpus=state` for
  state-specific books (*La entidad donde vivo ...*, 1993's *<Estado>. Historia y
  Geografía* and *Monografía estatal*): a child reads one, so they stay out of the national comparison
  (`download --corpus all` to fetch them). `audience=note` for archive entries
  that are not books.
- **Page images are standardised.** 2008-2011 scans run to 1617x2126 px and
  2014-2018 pages to 700 KB; pages taller than 1276 px (the current site's
  size) are scaled to it and files over 200 KB re-encoded (JPEG quality 82).
- **2026-27 (`primaria.html`).** Served only through a PDF reader whose files
  are password-encrypted (the password ships in the reader's JavaScript). The
  books are recorded in `catalog.csv` with `access=pdf-reader` and **not
  downloaded**: decrypting them would mean getting around a protection the
  publisher deliberately added. Their page images are also reachable at the
  usual unlinked path; whether to use those is the researcher's decision. The
  2026 codes are the same NEM titles as 2023-2025.

## Install

```bash
sudo apt-get install tesseract-ocr tesseract-ocr-spa        # macOS: brew install tesseract tesseract-lang
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install https://github.com/explosion/spacy-models/releases/download/es_core_news_sm-3.8.0/es_core_news_sm-3.8.0-py3-none-any.whl
python -m pytest tests/                                        # no network needed
```

The `senses` and `zeroshot` stages also need `torch` and `transformers`
(CPU wheels: `pip install --index-url https://download.pytorch.org/whl/cpu torch`);
their first run downloads BETO and CLIP from huggingface.co.

## Run

All commands run from this directory. Data goes to `data/` (git-ignored), or to `$LTG_DATA`.

```bash
# 1. Catalog and a pilot
python -m ltg catalog
python -m ltg download --max-pages 10 --codes H1988P5CI124,P5PCA
python -m ltg extract --include-incomplete

# 2. Full corpus: ~200 books, ~40k pages; about 2 h at 3 workers x 0.4 s, resumable.
#    extract skips books whose download is unfinished, so it can run alongside.
python -m ltg download --editions 1988,2022,2023,2024,2025 --workers 3 --delay 0.4
python -m ltg extract --workers 3

# 3. Text
python -m ltg text                  # spaCy pass -> results/text/*.csv.gz
python -m ltg senses                # 'pueblo' sense classifier (BETO + logistic regression)
python -m ltg report                # results/text/report.md + CSVs
python -m ltg sample                # hand-coding samples + sentences for APSI

# 4. Images
python -m ltg regions               # cut out illustrations + captions (incremental, after OCR)
python -m ltg illustration-sample   # simple random sample, 300 illustrations per edition
python -m ltg detect                # person counts on the sample (Faster R-CNN, COCO)
python -m ltg zeroshot              # CLIP B/16 codebook on the sample
python -m ltg visual-sample         # hand-coding sheet drawn from the sample (CSV + HTML)
python -m ltg visual-report         # results/visual/report.md

# 5. Models and figures
python -m ltg reuse                 # shared sentences between editions -> reuse_matrix.csv
python -m ltg stats                 # era models: rate ratios, trends, logits -> results/text/stats_*.csv
python -m ltg figures               # results/figures/*.png|svg

# 6. Validate after hand-coding the gold_* columns
python -m ltg validate --kind sense
python -m ltg validate --kind role
python -m ltg validate --kind visual --dim people
```

## Stages and outputs

| stage | module | output |
|---|---|---|
| catalog | `catalog.py` | `catalog.csv`: edition, code, grade(s), level, title, subject (from the title), corpus, audience, access, viewer URL |
| download | `download.py` | `pages/<ed>/<CODE>/<nnn>.jpg` + `manifest.json` (pages, image folder, missing pages) |
| extract | `extract.py` | `text/<ed>/<CODE>.jsonl` (one record per page, with OCR confidence) and `boxes/` word boxes |
| text | `text_analysis.py` | `results/text/`: every lexicon hit; every candidate people noun with its sense, role, verb, adjectives and KWIC; every sentence with a candidate people noun, with flags |
| senses | `senses.py` | re-labels *pueblo* in `people_tokens.csv.gz` (`sense`, `sense_p`, `sense_rule`), recomputes `counted`, writes `senses_cv.txt` |
| report | `text_analysis.py` | `report.md` + `rates_per_10k.csv`, `people_sentence_shares.csv`, `keyness.csv`, `pueblo_collocates.csv` |
| regions | `visual.py` | `results/visual/regions.csv` (size, colourfulness, caption, whether the caption names the people) and `visual/crops/` |
| illustration-sample | `visual.py` | `regions_sample.csv`: the random sample the two models run on |
| detect | `visual.py` | `people_detect.csv`: persons detected per illustration and none / one / group / crowd |
| zeroshot | `visual.py` | `zeroshot.csv`: label and probabilities on five codebook dimensions |
| stats | `stats.py` | `stats_rate_ratios.csv`, `stats_trends.csv`, `stats_logits.csv`, `stats_shares.csv`, `stats_sense_chi2.csv`, `stats_era_sizes.csv`: book-level Poisson and logit models by era (reference 2008-22), SEs clustered by title, Holm-adjusted p |

Every rate is a ratio of sums over books with a 95% interval from resampling
books, since pages and sentences within a book are not independent.

## Measurement decisions made on the real books

- **OCR at native resolution.** Pages are served at 670-970 px wide. On sample
  pages from 1988, 2022, 2023 and 2025, Tesseract at native size was as
  accurate as upscaling (mean word confidence 79-95) at half the time;
  upscaling to 2200 px lowered confidence and invented tokens in illustrations.
- **Senses of *pueblo*.** About half the 1988 uses are *towns*. The rest
  divide into the **common people** set against rulers or as the sovereign
  demos ("el pueblo elige", "la voluntad del pueblo", "se levantó en armas
  contra el gobierno"), a **nation** or the peoples of the world ("el pueblo
  mexicano", "los pueblos del mundo"), and an **ethnic / cultural / ancient
  people** ("el pueblo hebreo", "los pueblos indígenas"). Hand-written rules
  left 46% of 1988 uses ambiguous and zero-shot multilingual NLI did no better
  than always guessing *town*, so `senses` trains a classifier on coded
  examples in `ltg/seeds/pueblo_senses.csv`. **Those seed codes were made by
  Claude, not by a human coder**; they start the classifier, and the
  reportable accuracy is the agreement with your own hand-coded sample.
- ***Comunidad* and *asamblea* in NEM books** are mostly classroom
  instructions ("En pequeñas comunidades, reúnan...", "En asamblea,
  elaboren..."). They are tagged `classroom` / `school` and not counted as
  people-talk; *tu / su comunidad* is `local`; the rest `collective`.
- **Images: a sample for the expensive models.** Every illustration is cut out
  and measured for size, colour and caption. The people count and the CLIP
  codebook run on a simple random sample of 300 national-corpus illustrations
  per edition, because on a CPU the detector takes seconds per image. On 24
  hand-checked NEM crops the person detector got people presence right 20
  times and the none / one / group / crowd category 19 times, against 16 and
  15 for CLIP B/16, so the detector supplies the people count and CLIP the
  activity, figure, setting and medium.
- **Formulas.** NEM books repeat sentences such as "De manera individual y sin
  olvidar que eres parte de una comunidad, ...". They are kept (they are the
  books' own framing), and the report lists every people-talk sentence
  repeated five or more times.

## The lexicon is the instrument

`ltg/lexicon.py` holds every word list (people families, antagonists,
grievance, unity, pluralism, institutions, verb classes, sense cues). These
are research decisions: read them, change them, and re-run `text` + `senses` +
`report`. Terms with a common non-political meaning in textbooks (*unidad*,
*diferencia*, *plural*, *masa*) are deliberately absent. The keyness tables
are there to catch what the lists miss.

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

- **Terms of use.** The viewer pages say *Distribución gratuita, prohibida su
  venta*. Keep the download polite (the defaults), keep page images local, do
  not redistribute them, and quote pages in the paper as you would any
  copyrighted source.
- **Running in Claude Code on the web**: allow `libros.conaliteg.gob.mx`,
  `historico.conaliteg.gob.mx` and `huggingface.co` in the environment's
  network settings.
