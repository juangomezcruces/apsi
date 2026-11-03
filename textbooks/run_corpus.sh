#!/usr/bin/env bash
# Collection phase, restart-safe: download, OCR and illustration cut-out run side by
# side and each skips what is already done, so re-running after an interruption
# resumes where it stopped. Then run ./run_analysis.sh.
#   EDITIONS=1988,2022 OCR_WORKERS=3 ./run_corpus.sh
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p data/logs
EDITIONS=${EDITIONS:-1988,2022,2023,2024,2025}
OCR_WORKERS=${OCR_WORKERS:-3}
echo "=== $(date +%T) collection start (editions $EDITIONS)"

( python -m ltg download --editions "$EDITIONS" --workers 3 --delay 0.4 >> data/logs/download.log 2>&1
  echo "download exit $?" >> data/logs/download.log ) &
DL=$!

# Repeat a stage while `pid` runs, then once more after it has finished.
follow() {
  local pid=$1 pause=$2; shift 2
  while true; do
    kill -0 "$pid" 2>/dev/null; local running=$?
    python -m ltg "$@" >> "data/logs/$1.log" 2>&1 || echo "$(date +%T) $1 failed, see data/logs/$1.log"
    [ $running -ne 0 ] && break
    sleep "$pause"
  done
}

follow "$DL" 60 extract --workers "$OCR_WORKERS" &
EX=$!
follow "$EX" 120 regions --workers 1          # needs OCR word boxes, so it trails extract
wait
echo "=== $(date +%T) collection done"
