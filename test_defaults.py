"""run() and rescore() must not disagree on a shared knob.

rescore() kept stance_mode="gate" after CONFIG and run() moved to "off", so every
direct rescore() call silently re-enabled a gate that had been measured as
harmful -- and the sweep it was used for was uninterpretable. Cheap to test for.
"""
import os, importlib.util, sys, inspect
spec = importlib.util.spec_from_file_location("lr", os.environ.get("SCORER", "classify_economic.py"))
lr = importlib.util.module_from_spec(spec); sys.modules["lr"] = lr; spec.loader.exec_module(lr)
E = inspect.Parameter.empty
r = inspect.signature(lr.run).parameters
s = inspect.signature(lr.rescore).parameters
# these legitimately differ: run writes a CSV by default, rescore takes a path
EXEMPT = {"output_csv", "margins_path", "input_csv", "df"}

bad = [(k, r[k].default, s[k].default) for k in r if k in s and k not in EXEMPT
       and r[k].default is not E and s[k].default is not E
       and r[k].default != s[k].default]
for k, a, b in bad:
    print(f"  FAIL  {k}: run={a!r} rescore={b!r}")
print(f"  {'PASS' if not bad else 'FAIL'}  run() and rescore() agree on "
      f"{len([k for k in r if k in s and k not in EXEMPT])} shared defaults")

# and CONFIG must not disagree with run() either
cbad = [(k, v, r[k].default) for k, v in lr.CONFIG.items()
        if k in r and k not in EXEMPT and r[k].default is not E and v != r[k].default]
for k, a, b in cbad:
    print(f"  note  CONFIG[{k!r}]={a!r} differs from run() default {b!r}")
print(f"  {'PASS' if not bad else 'FAIL'}  no silent knob drift")
sys.exit(1 if bad else 0)
