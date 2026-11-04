# Coding Desk

A claude.ai Artifact page for the hand coding that the results still need:

| task | items | writes |
|---|---|---|
| Word senses | 660 (`results/validation/sense_sample.csv`) | `gold_sense` |
| Grammatical roles | 384 (`results/validation/role_sample.csv`) | `gold_role` |
| Illustrations | 344 (`results/validation/visual_sample.csv`) | `gold_people` ("number of people only", the default), or also `gold_activity`, `gold_figure`, `gold_setting`, `gold_medium` |
| Seed check | a fixed random 100 of the 451 seeds (`ltg/seeds/pueblo_senses.csv`) | `seed_check.csv`: your `gold_label` beside Claude's `claude_label`; "show all 451" re-codes every seed into `pueblo_senses.csv` |

Model and Claude labels are hidden while coding. Each coder's codes are saved in the
artifact's store under `codes/<coder id>/<task>/<task>-<row>` (the owner can read all
coders; a Contributor sees only their own), and each task exports a CSV with the same
columns as the sample, plus `unsure`, `excluded`, `coder`, `coded_at` and `notes`.
Drop the export into `results/validation/` and run `python -m ltg validate --kind sense|role|visual|seed`.
The samples are stratified by era x predicted label, so `validate` weights each coded item
by its cell's size in the books (`results/validation/strata.csv`) and reports accuracy,
kappa, precision and recall overall and by era, with each era's label shares as the model
gives them and as hand coding corrects them. Coding part of a tab is fine: the rows are in
random order, and cells with nothing coded yet are listed.

Rebuild after changing the template (rows are keyed by CSV position, so keep the
sample files the codes were made on):

    python coding_app/build.py --out <dir>     # index.html + img/visual-*.jpg
