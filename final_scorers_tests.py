"""classifier/final_scorers.py with a fake model: numpy only, no torch, no Django.

A premise carries tags; a hypothesis gets margin +12 if its tag is in the
premise, else -12. Tags: "topics" (every topic probe), "veto" (every negative
probe), "A:<domain>" / "B:<domain>" (side 0 / side 1 of a pair, in both
wordings). Each check fixes what the model "sees" and tests the adapter only.
"""
import logging
import sys
import numpy as np

root = logging.getLogger()
before = (root.handlers[:], root.level)
from classifier import final_scorers as F

HI, LO = 12.0, -12.0
PAD = "filler words for the length gate"
failures = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failures.append(name)


def make(cls):
    s = cls(margin_fn=lambda texts, hyps: None)
    tag = {h: "topics" for h in s.topics}
    tag.update({h: "veto" for h in s.veto})
    for bat in (s.battery, s.para):
        for d, _, h0, h1 in bat:
            tag.setdefault(h0, f"A:{d}"); tag.setdefault(h1, f"B:{d}")
    s._margins = lambda texts, hyps: np.array(
        [[HI if tag[h] in t.split() else LO for h in hyps] for t in texts])
    return s


for cls, a, b, off in [(F.LeftRightScorer, "left", "right", "Not about economic policy"),
                       (F.PopulismScorer, "pluralist", "populist", "Not about political rhetoric")]:
    s = make(cls)
    name = cls.__name__
    call = s.score_left_right if cls is F.LeftRightScorer else s.score_populism_pluralism
    d1, d2, d3 = [x[0] for x in s.battery[:3]]
    keys = {"score", "confidence", "interpretation", "is_relevant", "topic_probability",
            f"top_{a}_hypotheses", f"top_{b}_hypotheses", "contradiction_detected",
            "score_sd", "score_lo", "score_hi", "status"}

    r = call(f"{PAD} A:{d1} A:{d2}")
    check(f"{name}: off-topic is NA with the app's message", r["score"] == "NA" and r["interpretation"] == off)
    r = call(f"topics A:{d1}")
    check(f"{name}: under 5 words is NA (shared length gate)", r["status"] == "too_short")
    r = call(f"{PAD} topics A:{d1}")
    check(f"{name}: one pair is thin evidence (engine min_active=2)", r["status"] == "thin_evidence")
    r = call(f"{PAD} topics veto A:{d1} A:{d2}")
    check(f"{name}: negative probes veto off-topic rhetoric", r["gate_reason"] == "veto" and r["score"] == "NA")

    lo = call(f"{PAD} topics A:{d1} A:{d2}")
    hi = call(f"{PAD} topics B:{d1} B:{d2}")
    check(f"{name}: side 0 scores below 5, side 1 above, mirrored",
          lo["score"] < 5 < hi["score"] and abs(lo["score"] + hi["score"] - 10) < 1e-9)
    check(f"{name}: every key the views read is present", keys <= set(lo) and keys <= set(call("x")))
    check(f"{name}: winning side listed with hypothesis, probability, score_impact",
          len(lo[f"top_{a}_hypotheses"]) == 2 and not lo[f"top_{b}_hypotheses"]
          and {"hypothesis", "probability", "score_impact"} <= set(lo[f"top_{a}_hypotheses"][0]))
    check(f"{name}: agreeing pairs give a zero-width bootstrap range, confidence 1",
          lo["score_hi"] - lo["score_lo"] < 1e-9 and lo["confidence"] == 1.0)
    mix = call(f"{PAD} topics A:{d1} A:{d2} B:{d3}")
    check(f"{name}: disagreeing pairs widen the range and flag a contradiction",
          mix["confidence"] < 1.0 and mix["contradiction_detected"])
    check(f"{name}: two wordings averaged, and the fixed validated scale is used",
          bool(s.para) and s.agg.fit_["scale"] == F.ENGINES["left_right" if cls is F.LeftRightScorer else "populism"][2])

check("loading the engines leaves the app's logging configuration alone",
      (root.handlers[:], root.level) == before)
check("the engines are the project-root scripts with the validated batteries",
      F.load_engine("left_right").battery_hash() == "90a23fbedce3"
      and F.load_engine("populism").battery_hash() == "78de7ae5a864")
print(f"\n{len(failures)} failure(s)")
sys.exit(1 if failures else 0)
