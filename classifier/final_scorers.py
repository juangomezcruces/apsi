"""The validated scorers, wired into the app.

    left-right         classify_economic.py   (battery 90a23fbedce3)
    populism-pluralism classify_populism.py   (battery 78de7ae5a864, two wordings)
    support for democracy  alternativeLib_v3.py  (used directly; already speaks this interface)

The two engine scripts at the project root are the single source of truth: they
are loaded from there, not copied, so the app and the validation runs cannot
drift apart. This module only adapts them to what views.py and views_longdoc.py
read -- score, confidence (0-1), interpretation, is_relevant, topic_probability,
top_<side>_hypotheses, contradiction_detected -- and adds the engine's own
bootstrap fields (score_sd, score_lo, score_hi) and status.

Every setting is the script's own CONFIG (gates, 5-word minimum, veto, paraphrase
averaging, 512-token truncation, 200-draw bootstrap), with one exception the app
forces: the engine fits its contrast scale on the whole corpus it scores
("auto"), which a one-text request cannot do, so it is fixed at the value the
validated run fitted (CONTRAST_SCALE below).

The model is the one SharedModelCache already holds for the app, not a second copy.
"""
import importlib.util
import logging
import threading
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "mlburnham/Political_DEBATE_large_v1.0"

ENGINES = {
    # name: (file, battery hash it must have, contrast scale fitted by the validated run)
    "left_right": ("classify_economic.py", "90a23fbedce3", 10.1315),   # PopBERT, lr_scale_en
    "populism":   ("classify_populism.py", "78de7ae5a864", 8.558),     # US 2010-2018, averaged, shared gates
}

# What the app shows for each unscored status
NA_TEXT = {
    "not_economic": "Not about economic policy",
    "not_political_authority": "Not about political rhetoric",
    "too_short": "Too short to score",
    "thin_evidence": "Too little evidence to score",
    "no_active_pair": "No clear position",
    "empty_text": "Empty text",
}

_modules, _lock = {}, threading.Lock()


def load_engine(name):
    """Import an engine script once. Its module-level logging.basicConfig(force=True)
    is meant for Kaggle; the app's logging configuration is restored after it."""
    with _lock:
        if name not in _modules:
            fname, want, _ = ENGINES[name]
            spec = importlib.util.spec_from_file_location(f"engine_{name}", ROOT / fname)
            mod = importlib.util.module_from_spec(spec)
            root = logging.getLogger()
            handlers, level = root.handlers[:], root.level
            try:
                spec.loader.exec_module(mod)
            finally:
                root.handlers[:] = handlers
                root.setLevel(level)
            got = mod.battery_hash()
            if got != want:
                raise RuntimeError(f"{fname} has battery {got}, the validated one is {want}")
            _modules[name] = mod
        return _modules[name]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _shared_margin_scorer(E, model_name):
    """The engine's MarginScorer, built on the app's cached model instead of
    loading its own."""
    from .shared_model_cache import SharedModelCache   # imports torch
    cache = SharedModelCache()
    ms = object.__new__(E.MarginScorer)
    ms.model, ms.tokenizer = cache.get_model_and_tokenizer(model_name)
    ms.device = cache.get_device()
    ms.model_name, ms.revision = model_name, None
    ms.max_length = E.CONFIG["max_length"]
    ms.n_labels = int(ms.model.config.num_labels)
    ms.n_nonfinite = 0
    ms.entailment_idx = ms._find_entailment_index()
    lock = cache.inference_lock()

    def margins(texts, hyps):
        with lock:
            return ms.margin_matrix(texts, hyps, progress=False)
    return margins


class _EngineScorer:
    """One engine dimension, scored one text at a time with its validated settings."""

    sides = ("a", "b")   # names of the 0 / 1 side in the output keys, set by subclasses

    def __init__(self, name, model_name=MODEL_NAME, margin_fn=None):
        E = self.E = load_engine(name)
        cfg = E.CONFIG
        self.scale = ENGINES[name][2]
        self.battery = list(E.BATTERY)
        self.para = list(E.BATTERY_PARAPHRASE) if cfg["average_paraphrase"] and E.BATTERY_PARAPHRASE else []
        self.topics = list(E.TOPIC_HYPOTHESES)
        self.veto = list(E.NEGATIVE_TOPIC_HYPOTHESES) if cfg["use_veto"] else []
        self.agg = E.ContrastAggregator(
            self.battery, self.topics, topic_margin_min=cfg["topic_margin_min"],
            active_margin=cfg["active_margin"], standardize=cfg["standardize"],
            contrast_scale=self.scale, active_only=cfg["active_only"],
            bootstrap=cfg["bootstrap"], seed=cfg["seed"], min_tokens=cfg["min_tokens"],
            min_active=cfg["min_active"], stance_mode="off", drop_domains=cfg["drop_domains"],
            veto_hypotheses=self.veto, veto_margin=cfg["veto_margin"], use_veto=bool(self.veto),
            balance_weight=cfg["balance_weight"])
        b = E.battery_hypotheses
        self.hypotheses = (self.topics + b(self.battery) + (b(self.para) if self.para else []) + self.veto)
        nt, nb = len(self.topics), 2 * len(self.battery)
        self._cols = {"topic": slice(0, nt), "batt": slice(nt, nt + nb),
                      "para": slice(nt + nb, nt + nb + (nb if self.para else 0)),
                      "veto": slice(nt + nb + (nb if self.para else 0), len(self.hypotheses))}
        self._margins = margin_fn or _shared_margin_scorer(E, model_name)

    def score(self, text):
        text = "" if text is None else str(text)
        M = np.asarray(self._margins([text], self.hypotheses), dtype=np.float64)
        topic, batt = M[:, self._cols["topic"]], M[:, self._cols["batt"]]
        if self.para:
            batt = (batt + M[:, self._cols["para"]]) / 2.0
        veto = M[:, self._cols["veto"]] if self.veto else None
        self.agg.fit(batt, topic)          # fixed scale, no centring: data-independent
        r = self.agg.transform(batt, topic, texts=[text], veto_margins=veto)[0]

        a, b = self.sides
        out = {"text": text, "status": r["status"], "gate_reason": r.get("gate_reason"),
               "topic_probability": float(_sigmoid(r["topic_margin"])) if np.isfinite(r["topic_margin"]) else 0.0,
               "score_sd": r["score_sd"], "score_lo": r["score_lo"], "score_hi": r["score_hi"],
               "dispersion": r["dispersion"], "n_active": r["n_active"],
               f"top_{a}_hypotheses": [], f"top_{b}_hypotheses": [], "contradiction_detected": False}
        if r["status"] != "scored":
            return {**out, "score": "NA", "confidence": 0.0, "is_relevant": False,
                    "interpretation": NA_TEXT.get(r["status"], r["status"])}

        contrast, active = self.agg._contrasts(batt)
        pos, act = self.agg._positions(contrast)[0], active[0]
        w = np.asarray(getattr(self.agg, "weights_eff", self.agg.weights), float)
        W = float((w * act).sum())
        tops = {0: [], 1: []}
        for p in np.flatnonzero(act):
            side = 1 if pos[p] > 0 else 0
            tops[side].append({"hypothesis": self.battery[p][2 + side],
                               "probability": float(_sigmoid(batt[0, 2 * p + side])),
                               "score_impact": float(5.0 * w[p] * abs(pos[p]) / W),
                               "domain": self.battery[p][0]})
        for k in tops:
            tops[k].sort(key=lambda d: -d["score_impact"])
        lo, hi = r["score_lo"], r["score_hi"]
        return {**out, "score": float(r["score"]), "is_relevant": True,
                "interpretation": r["interpretation"],
                # same 0-1 summary of the bootstrap range as alternativeLib_v3
                "confidence": float(max(0.0, 1.0 - (hi - lo) / 10.0)) if np.isfinite(hi - lo) else 0.0,
                "contradiction_detected": bool((pos[act] > 0.5).any() and (pos[act] < -0.5).any()),
                f"top_{a}_hypotheses": tops[0][:5], f"top_{b}_hypotheses": tops[1][:5]}


class LeftRightScorer(_EngineScorer):
    """0 = left, 10 = right. classify_economic.py."""
    sides = ("left", "right")

    def __init__(self, **kw):
        super().__init__("left_right", **kw)

    def score_left_right(self, text):
        return self.score(text)


class PopulismScorer(_EngineScorer):
    """0 = pluralist, 10 = populist. classify_populism.py."""
    sides = ("pluralist", "populist")

    def __init__(self, **kw):
        super().__init__("populism", **kw)

    def score_populism_pluralism(self, text):
        return self.score(text)
