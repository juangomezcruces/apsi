"""Aggregation tests for the liberal-illiberal scorer. numpy only, no torch.

    python3 liberal_illiberal_tests.py                       # v2, the app's
    LI_MODULE=alternativeLib_v3 python3 liberal_illiberal_tests.py   # paper

A fake margin_fn stands in for the model. A premise carries tags such as
"accept_defeat:L" or "topic"; a hypothesis gets margin +HI if its tag is in the
premise and -LO otherwise. Each test thus fixes what the model "sees" and checks
only what the scorer does with it.

Not named test_*.py on purpose: run_tests.sh feeds every test_*.py the two
classify_* scripts, and this file tests neither.
"""
import importlib
import inspect
import os
import sys
import numpy as np

L = importlib.import_module("classifier." + os.environ.get("LI_MODULE", "alternativeLib"))
print(f"testing classifier.{L.__name__.split('.')[-1]}")

HI, LO = 12.0, -12.0   # the size of real margins on the validation tweets
TAG = {h: "topic" for h in L.TOPIC_HYPOTHESES}
TAG[L.MILITANT_DEMOCRACY_HYPOTHESIS] = "md"
for d, _, _, lib, ill in L.BATTERY:
    TAG[lib], TAG[ill] = f"{d}:L", f"{d}:I"
# the second wording answers to the same tags, so every test below sees the
# same evidence whether or not paraphrase averaging is on
for d, _, _, lib, ill in getattr(L, "BATTERY_PARAPHRASE", None) or []:
    TAG.setdefault(lib, f"{d}:L"); TAG.setdefault(ill, f"{d}:I")


VETO = "The text is a personal attack with no political content."
TAG[VETO] = "veto"
TAG.setdefault(L.TOPIC_HYPOTHESES[0], "topic")


def fake(premises, hypotheses):
    # "onetopic" makes only the FIRST topic probe fire, to test that two must agree
    return np.array([HI if TAG[h] in p.split() or (h == L.TOPIC_HYPOTHESES[0] and "onetopic" in p.split())
                     else LO for p, h in zip(premises, hypotheses)])


S = L.LiberalIlliberalScorer(margin_fn=fake)
# the mirror property holds only when both sides weigh the same
SYM = (L.LiberalIlliberalScorer(margin_fn=fake, negative_weight=1.0)
       if "negative_weight" in inspect.signature(L.LiberalIlliberalScorer).parameters else S)
# premises are padded to pass the 5-word length gate; the filler matches no tag
PAD = "filler words for length"
score = lambda *tags: S.score_liberal_illiberal(" ".join((PAD, "topic") + tags))
failures = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failures.append(name)


r = S.score_liberal_illiberal(PAD + " accept_defeat:L")
check("off-topic text is NA even if a stance probe fires", r['score'] == 'NA' and not r['is_relevant'])

params = inspect.signature(L.LiberalIlliberalScorer).parameters
if "bootstrap" in params:          # v3: the shared engine's uncertainty method
    one = score("free_speech:L")
    check("bootstrap: one active pair has nothing to resample (NaN, confidence 0)",
          np.isnan(one["score_sd"]) and one["confidence"] == 0.0)
    same = score("free_speech:L", "assembly:L", "press_freedom:L")
    check("bootstrap: identical pair positions give a zero-width interval",
          same["score_sd"] < 1e-9 and abs(same["score_hi"] - same["score_lo"]) < 1e-9 and same["confidence"] == 1.0)
    mixed2 = score("free_speech:L", "assembly:L", "opposition_parties:I")
    check("bootstrap: disagreeing pairs widen the interval and lower confidence",
          mixed2["score_hi"] - mixed2["score_lo"] > 1 and mixed2["confidence"] < same["confidence"]
          and mixed2["score_lo"] <= mixed2["score"] <= mixed2["score_hi"])
    alone = S.score_liberal_illiberal(PAD + " topic free_speech:L assembly:L opposition_parties:I")
    inbatch = S.score_texts([PAD + " topic accept_defeat:I", PAD + " topic free_speech:L assembly:L opposition_parties:I"])[1]
    check("bootstrap: a text's interval does not depend on the batch",
          (alone["score_lo"], alone["score_hi"]) == (inbatch["score_lo"], inbatch["score_hi"]))
if "average_paraphrase" in params:   # v3: two wordings, averaged
    row = S.margin_matrix([PAD + " topic accept_defeat:L"])[0].copy()
    P = len(L.BATTERY); p = [b[0] for b in L.BATTERY].index("accept_defeat")
    row[S._para.start + 2 * p] = LO                 # second wording disagrees
    avg, main = S.score_from_margins(row), S.score_from_margins(row, wording="main")
    check("paraphrase: a disagreeing second wording pulls the score toward 5",
          5 < avg["score"] < main["score"] and avg["paraphrased"])
    old = S.score_from_margins(row[:S._para.start])
    check("paraphrase: a cache without the second wording scores on the main one",
          not old["paraphrased"] and old["score"] == main["score"])
    rel = S.reliability(S.margin_matrix([PAD + " topic free_speech:L", PAD + " topic assembly:I",
                                         PAD + " topic accept_defeat:L press_freedom:I"]))
    check("paraphrase: identical wordings give reliability 1", rel["n"] == 3 and rel["pearson"] > 0.999)
    try:
        L.LiberalIlliberalScorer(margin_fn=fake, paraphrase_battery=L.BATTERY_PARAPHRASE[:-1])
        check("paraphrase: a mismatched second battery is refused", False)
    except ValueError:
        check("paraphrase: a mismatched second battery is refused", True)
if "min_tokens" in params:        # v3: the shared length gate
    four, five = S.score_liberal_illiberal("topic free_speech:L x y"), \
                 S.score_liberal_illiberal("topic free_speech:L x y z")
    check("length gate: 4 words are not scored", four['score'] == 'NA'
          and four.get('gate_reason') == 'too_short')
    check("length gate: 5 words are scored", five['is_relevant'] and five['score'] > 5)
    check("length gate runs before the topic gate",
          S.score_liberal_illiberal("free_speech:L x")['gate_reason'] == 'too_short')
    check("re-scoring margins without the text skips the length gate",
          S.score_from_margins(S.margin_matrix(["topic free_speech:L"])[0])['is_relevant'])
if "veto_hypotheses" in params:   # v3: the shared engine gate
    r = S.score_liberal_illiberal(PAD + " onetopic accept_defeat:L")
    check("shared gate: one topic probe alone does not admit (two must agree)",
          r['score'] == 'NA' and r.get('gate_reason') == 'topic')
    SV = L.LiberalIlliberalScorer(margin_fn=fake, veto_hypotheses=[VETO])
    check("shared gate: negative probe outweighing the topic probes drops the text",
          SV.score_liberal_illiberal(PAD + " topic veto accept_defeat:L").get('gate_reason') == 'veto')
    check("shared gate: veto off by default, and on-topic text without it is admitted",
          SV.score_liberal_illiberal(PAD + " topic accept_defeat:L")['is_relevant']
          and S.score_liberal_illiberal(PAD + " topic veto accept_defeat:L")['is_relevant'])

r = score()
check("on topic but no active pair is NA", r['score'] == 'NA'
      and r['interpretation'] == 'No clear position on democratic principles')

lib, ill = score("accept_defeat:L"), score("accept_defeat:I")
check("one liberal pair scores above 5", lib['score'] > 5 and lib['interpretation'] in ("Liberal", "Strong Liberal"))
mirror = lambda t: SYM.score_liberal_illiberal(PAD + " topic " + t)['score']
check("mirror text gives the mirror score (equal side weights)",
      abs(mirror("accept_defeat:L") + mirror("accept_defeat:I") - 10) < 1e-9)
check("an illiberal pair moves the score at least as far as a liberal one",
      5 - ill['score'] >= lib['score'] - 5 - 1e-9)
check("winning side is listed, losing side is not",
      len(lib['top_liberal_hypotheses']) == 1 and lib['top_illiberal_hypotheses'] == [])

one, three = score("free_speech:L"), score("free_speech:L", "assembly:L", "press_freedom:L")
check("more agreeing evidence moves further from 5 and raises confidence",
      three['score'] > one['score'] and three['confidence'] > one['confidence'])

plain, defensive = score("free_speech:I"), score("free_speech:I", "md")
check("militant-democracy framing softens a freedom pair",
      plain['score'] < defensive['score'] < 5)
ban, ban_md = score("opposition_parties:I"), score("opposition_parties:I", "md")
check("...but never a ban on opposition parties", abs(ban['score'] - ban_md['score']) < 1e-9)
check("...and never makes a restriction liberal", defensive['score'] < 5)

mixed = score("free_speech:L", "opposition_parties:I")
check("contradiction flagged when strong pairs disagree", mixed['contradiction_detected'])
check("contradiction lowers confidence", mixed['confidence'] < score("free_speech:L", "assembly:L")['confidence'])

freedoms = score("free_speech:L", "press_freedom:L", "assembly:L", "civil_society:L", "criticism:L")
both = score("free_speech:L", "press_freedom:L", "assembly:L", "civil_society:L", "accept_defeat:L")
check("level 4 needs accepting defeat, not only the freedoms",
      freedoms['vparty_level'] == 3 and both['vparty_level'] == 4)
repress = score("free_speech:I", "press_freedom:I", "assembly:I", "civil_society:I", "criticism:I")
regime = score("one_party:I", "source_of_authority:I", "multiparty_elections:I", "lawful_means:I")
check("level 0 needs open support for autocracy, not only repression",
      repress['vparty_level'] == 1 and regime['vparty_level'] == 0)

long_text = "Filler sentence about the weather today. " * 60 + "topic accept_defeat:L ."
if hasattr(L, "chunk_text"):      # v2 (the app's): long texts are chunked and max-pooled
    r = S.score_liberal_illiberal(long_text)
    check("signal past the first 512 tokens is not lost", r['is_relevant'] and r['score'] > 5)
    check("long text was chunked", len(L.chunk_text(long_text)) > 1)
else:                             # v3: one input per text, truncated by the tokenizer
    seen = []
    T = L.LiberalIlliberalScorer(margin_fn=lambda p, h: (seen.extend(p), fake(p, h))[1])
    T.score_texts([long_text, PAD + " topic free_speech:L"])
    check("long text is passed whole, once per hypothesis (truncation, no chunking)",
          len(seen) == 2 * len(T.hypotheses) and seen[0] == long_text)
    check("token cap is the model's maximum, as in the shared engine",
          T.max_length == L.MAX_LENGTH == 512)

views_keys = {'score', 'confidence', 'interpretation', 'is_relevant', 'topic_probability',
              'top_liberal_hypotheses', 'top_illiberal_hypotheses', 'contradiction_detected'}
check("keys views.py reads are present", views_keys <= set(both))
check("NA results carry what views.py reads",
      {'score', 'interpretation', 'is_relevant', 'topic_probability'} <= set(score()))
check("confidence in [0, 1]", all(0 <= x['confidence'] <= 1 for x in (one, three, mixed, both)))
check("hypothesis items carry probability and score_impact",
      all({'hypothesis', 'probability', 'score_impact'} <= set(h) for h in both['top_liberal_hypotheses']))

batch = S.score_texts([PAD + " topic free_speech:L", PAD + " nothing here", PAD + " topic accept_defeat:I"])
check("score_texts matches one-at-a-time", batch[0]['score'] == score("free_speech:L")['score']
      and batch[1]['score'] == 'NA' and batch[2]['score'] == ill['score'])

texts = [score("free_speech:L"), score("accept_defeat:L"), score(), score("assembly:L")]
actor = S.aggregate_actor(texts)
# the level itself depends on the shrinkage setting, so test the pooling, not the level
check("actor pools evidence across texts", actor['n_scored'] == 3 and actor['evidence_on_defeat']
      and actor['evidence_on_freedoms']
      and actor['score'] > max(t['score'] for t in texts if t['score'] != 'NA'))
check("actor with no scored text is NA", S.aggregate_actor([score()])['score'] == 'NA')

check("torch was never imported", 'torch' not in sys.modules)

print(f"\n{len(failures)} failure(s)")
sys.exit(1 if failures else 0)
