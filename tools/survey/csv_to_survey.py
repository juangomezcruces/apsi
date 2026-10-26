#!/usr/bin/env python3
"""
Turn a scored-paragraph CSV into the `items` array of a survey JSON file.

Usage
-----
    python3 tools/csv_to_survey.py "apsi_scores (3).csv" surveys/liberal-democracy.json

The CSV needs these columns (rename yours or edit COLUMNS below):

    uid, paragraph, lib_score, lib_label

Everything else in the survey file — title, welcome text, definitions,
how many items get drawn — is left untouched. Only `items` is replaced,
so you can re-run this whenever the CSV changes.

The slider's starting position is lib_score x 10, rounded, clamped to 0-100.
"""

import csv
import json
import sys
from pathlib import Path

COLUMNS = {
    "uid": "uid",
    "text": "paragraph",
    "score": "lib_score",
    "label": "lib_label",
}

SCORE_TO_SLIDER = 10  # lib_score is 0-10, the slider is 0-100


def build_items(csv_path: Path) -> list[dict]:
    """Read the CSV, tolerating paragraphs whose commas were never quoted.

    Spreadsheet exports often lose the quoting around `paragraph`, which
    splits one text across many columns. Since `paragraph` is the only
    free-text column, anything between the uid and the trailing
    score/label belongs to it and can be rejoined.
    """
    with csv_path.open(newline="", encoding="utf-8-sig") as fh:
        rows = [r for r in csv.reader(fh) if any(f.strip() for f in r)]

    if not rows:
        sys.exit("the CSV is empty")

    items, repaired = [], 0
    for line_no, row in enumerate(rows[1:], start=2):
        if len(row) < 4:
            sys.exit(f"line {line_no}: only {len(row)} columns, expected at least 4")
        if len(row) > 4:
            repaired += 1

        uid = row[0].strip() or f"item_{line_no - 1:04d}"
        label = row[-1].strip()
        raw = row[-2].strip()
        text = ",".join(row[1:-2]).strip().strip('"').strip()

        try:
            score = float(raw)
        except ValueError:
            sys.exit(f"line {line_no}: '{raw}' is not a number — check the CSV's quoting")
        if not text:
            sys.exit(f"line {line_no}: empty paragraph")

        items.append({
            "uid": uid,
            "text": text,
            "libScore": round(score, 4),
            "libLabel": label,
            "start": max(0, min(100, round(score * SCORE_TO_SLIDER))),
        })

    if repaired:
        print(f"note: rejoined unquoted commas in {repaired} row(s)")
    return items


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)

    csv_path, survey_path = Path(sys.argv[1]), Path(sys.argv[2])
    if not csv_path.exists():
        sys.exit(f"no such CSV: {csv_path}")

    items = build_items(csv_path)
    if not items:
        sys.exit("the CSV produced no usable rows")

    survey = json.loads(survey_path.read_text(encoding="utf-8")) if survey_path.exists() else {}
    survey["items"] = items
    survey_path.parent.mkdir(parents=True, exist_ok=True)
    survey_path.write_text(json.dumps(survey, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    draw = (survey.get("sampling") or {}).get("drawCount", "?")
    print(f"wrote {len(items)} items to {survey_path} (each respondent sees {draw})")


if __name__ == "__main__":
    main()
