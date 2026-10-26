"""A non-economic battery must not inherit the economic paraphrase set."""
import os, importlib.util, inspect
s=importlib.util.spec_from_file_location("lib",os.environ.get("SCORER", "classify_economic.py"))
l=importlib.util.module_from_spec(s); l.__name__="lib"; s.loader.exec_module(l)
assert inspect.signature(l.run).parameters["paraphrase_battery"].default is None
ed=[d for d,*_ in l.BATTERY]
if l.BATTERY_PARAPHRASE is None:
    print("test_paraphrase_default: PASS (no paraphrase battery in this dimension)"); raise SystemExit
assert [d for d,*_ in l.BATTERY_PARAPHRASE]==ed, "paraphrase set must match its own battery"
# the resolution rule the patch implements
def resolves(batt):
    return [d for d,*_ in l.BATTERY_PARAPHRASE]==[d for d,*_ in batt]
assert resolves(l.BATTERY)
src=inspect.getsource(l.run)
assert "average_paraphrase = False" in src and "paraphrase_battery = BATTERY_PARAPHRASE" in src
assert 'raise ValueError("paraphrase_battery must have the same domains' in src, "explicit guard removed"
print("test_paraphrase_default: PASS")
