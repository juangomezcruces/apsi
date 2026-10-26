#!/bin/sh
# Engine tests run against BOTH dimension scripts; parity is checked once.
fail=0
for scorer in classify_economic.py classify_populism.py; do
  echo "--- $scorer ---"
  for t in test_*.py; do
    [ "$t" = "test_engine_parity.py" ] && continue
    case "$t:$scorer" in
    esac
    if SCORER="$scorer" python3 "$t" >/dev/null 2>&1; then echo "  PASS $t"
    else echo "  FAIL $t"; fail=1; fi
  done
done
echo "--- parity ---"
python3 test_engine_parity.py || fail=1
exit $fail
