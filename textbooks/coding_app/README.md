# Coding Desk

A claude.ai Artifact page for the hand coding that the results still need:

| task | items | writes |
|---|---|---|
| Word senses | 660 (`results/validation/sense_sample.csv`) | `gold_sense` |
| Grammatical roles | 384 (`results/validation/role_sample.csv`) | `gold_role` |
| Illustrations | 344 (`results/validation/visual_sample.csv`) | `gold_people`, `gold_activity`, `gold_figure`, `gold_setting`, `gold_medium` |
| Training seeds | 451 (`ltg/seeds/pueblo_senses.csv`) | `label`, `coder` (replaces Claude's codes) |

Model and Claude labels are hidden while coding. Each coder's codes are saved in the
artifact's store under `codes/<coder id>/<task>/<task>-<row>` (the owner can read all
coders; a Contributor sees only their own), and each task exports a CSV with the same
columns as the sample, plus `unsure`, `excluded`, `coder`, `coded_at` and `notes`.
Drop the export into `results/validation/` and run `python -m ltg validate --kind ...`.

Rebuild after changing the template (rows are keyed by CSV position, so keep the
sample files the codes were made on):

    python coding_app/build.py --out <dir>     # index.html + img/visual-*.jpg
