"""Rules the three scorers must share, pinned so the engine (classify_economic.py,
classify_populism.py) and the support-for-democracy scorer cannot drift apart:

  * TRUNCATION: one input per text, "only_first" (the text is clipped, never the
    hypothesis), at 512 tokens -- the most DeBERTa-v3 accepts.
  * LENGTH GATE: texts of fewer than 5 whitespace-separated words are not scored
    ('too_short'), i.e. only texts of more than four words are analysed.
  * UNCERTAINTY: bootstrap over the active pairs, 200 draws, reported as the SD
    and the 5th-95th percentiles of the re-scored draws; NaN below two pairs.
"""
import inspect, os, sys, importlib.util
spec = importlib.util.spec_from_file_location("m", os.environ.get("SCORER", "classify_economic.py"))
m = importlib.util.module_from_spec(spec); sys.modules["m"] = m; spec.loader.exec_module(m)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classifier import alternativeLib_v3 as v3

CAP = 512
default = lambda f, k: inspect.signature(f).parameters[k].default
assert m.CONFIG["max_length"] == CAP, m.CONFIG["max_length"]
assert default(m.MarginScorer.__init__, "max_length") == CAP
assert default(m.run, "max_length") == CAP
assert v3.MAX_LENGTH == CAP and default(v3.LiberalIlliberalScorer.__init__, "max_length") == CAP
for name, src in [("engine", inspect.getsource(m.MarginScorer.margins)),
                  ("v3", inspect.getsource(getattr(v3.LiberalIlliberalScorer, "_model_margins_unlocked",
                                           v3.LiberalIlliberalScorer._model_margins)))]:
    assert 'truncation="only_first"' in src, f"{name}: must clip the text, not the hypothesis"
    assert "max_length=self.max_length" in src, f"{name}: must use the configured cap"
assert not hasattr(m, "chunk_text") and not hasattr(v3, "chunk_text"), "chunking is back in one scorer"
WORDS = 5
assert m.CONFIG["min_tokens"] == WORDS, m.CONFIG["min_tokens"]
for f in (m.ContrastAggregator.__init__, m.run, m.rescore):
    assert default(f, "min_tokens") == WORDS, f.__name__
assert default(v3.LiberalIlliberalScorer.__init__, "min_tokens") == WORDS
assert "len(text.split()) < self.min_tokens" in inspect.getsource(m.ContrastAggregator)
assert "len(str(text).split()) < self.min_tokens" in inspect.getsource(v3.LiberalIlliberalScorer)
# UNCERTAINTY: bootstrap over active pairs, 200 draws, SD and 5th-95th percentiles
assert default(m.ContrastAggregator.__init__, "bootstrap") == 200
assert default(v3.LiberalIlliberalScorer.__init__, "bootstrap") == 200
eng_src, v3_src = inspect.getsource(m.ContrastAggregator), inspect.getsource(v3.LiberalIlliberalScorer)
assert "np.percentile(bs, [5, 95])" in eng_src and "np.percentile(bs, [5, 95])" in v3_src
assert "act.size >= 2" in eng_src and "idx.size < 2" in v3_src, "both need two pairs to resample"
print("test_shared_rules: PASS")
