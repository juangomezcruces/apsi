# Updating the survey passages

The survey reads `static/survey/surveys/liberal-democracy.json`. The CSV here
is the **source** for the `items` in that file — editing the CSV alone changes
nothing until you regenerate.

```bash
python3 tools/survey/csv_to_survey.py \
    tools/survey/apsi_scores.csv \
    static/survey/surveys/liberal-democracy.json
```

That replaces only `items`. Every other section — welcome, instructions,
reminder, endpoint — is left untouched, so it is safe to re-run any time.

Then commit both files, `git pull` on the server, and the new texts are live.
No `collectstatic` and no container rebuild.

The slider's starting position is `lib_score x 10`, rounded and clamped to
0-100.

## CSV format

Columns: `uid, paragraph, lib_score, lib_label`.

If a paragraph contains commas it should be quoted, but spreadsheet exports
often lose the quoting. The script repairs those rows by rejoining everything
between the uid and the trailing score/label, and says so when it does. It
exits with the offending line number if a row cannot be read.
