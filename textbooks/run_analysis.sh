#!/usr/bin/env bash
# Everything after download + OCR, in order. Each stage logs to data/logs/<stage>.log.
#   ./run_analysis.sh            # text, senses, reports, samples, images
#   ./run_analysis.sh text       # text side only
#   ./run_analysis.sh images     # image side only
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data/logs
step() { echo "=== $(date +%T) $*"; python -m ltg "$@" > "data/logs/$1.log" 2>&1 || { echo "FAILED: $*  (see data/logs/$1.log)"; exit 1; }; }

part=${1:-all}
if [[ $part == all || $part == text ]]; then
  step text
  step senses
  step report --corpus national
  python -m ltg report --corpus state > data/logs/report_state.log 2>&1
  step sample
fi
if [[ $part == all || $part == images ]]; then
  step regions --workers 4
  step illustration-sample --n 300
  step detect
  step zeroshot --model openai/clip-vit-base-patch16
  step visual-sample
  step visual-report
fi
step figures
echo "=== $(date +%T) done"
