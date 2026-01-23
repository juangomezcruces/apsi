import os
import torch
import logging
from typing import Dict, Optional

from django.conf import settings  # noqa: F401  (kept for Django integration / future use)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIG
# ============================================================================

# MOCK MODE - Set this to True to test without actual models
MOCK_MODE = False  # Change to True to test without downloading / loading models

# Topic-gate configuration (axis-specific filtering)
ENABLE_TOPIC_GATE = True  # Set False to disable axis topic-gating
TOPIC_THRESHOLD = 0.5  # Probability threshold (0-1) for topic relevance


# Axis identifiers supported by the service.
AXIS_LEFT_RIGHT = "left_right"
AXIS_LIBERAL_ILLIBERAL = "liberal_illiberal"
AXIS_POPULISM_PLURALISM = "populism_pluralism"

# Topic hypotheses (premise=text, hypothesis=string) used for the axis gate.
# NOTE: These are used as NLI hypotheses. We run both a positive and negative version and require
#       agreement + consistency before allowing downstream scoring.
AXIS_TOPICS = {
    AXIS_LEFT_RIGHT: {
        "positive": (
            "This text discusses economic policy, government intervention, or public services. "
            "This includes topics like healthcare, education, housing, transport, taxation, natural resources "
            "privatization, welfare, regulation, minimum wage, wealth redistribution, "
            "public vs. private sector roles, or economic equality."
        ),

    },
    AXIS_LIBERAL_ILLIBERAL: {
        "positive": (
            "This text discusses political ideas related to democratic principles and liberal democracy. "
            "This includes civil liberties, human rights, minority rights, rule of law, judicial independence, "
            "checks and balances, free and fair elections, free press, constitutional limits, "
            "or constraints on executive power."
        ),

    },
    AXIS_POPULISM_PLURALISM: {
        "positive": (
            "This text discusses political rhetoric, governance approaches, or institutional legitimacy. "
            "This includes populist rhetoric (challenging institutions, emphasizing popular will, anti-elite framing) "
            "or pluralist rhetoric (supporting checks and balances, minority rights, compromise, coalition-building)."
        ),

    },
}

# Optional aliases you might get from UI / API clients
AXIS_ALIASES = {
    "left-right": AXIS_LEFT_RIGHT,
    "leftright": AXIS_LEFT_RIGHT,
    "economic": AXIS_LEFT_RIGHT,
    "liberal-illiberal": AXIS_LIBERAL_ILLIBERAL,
    "liberalilliberal": AXIS_LIBERAL_ILLIBERAL,
    "democracy": AXIS_LIBERAL_ILLIBERAL,
    "populism-pluralism": AXIS_POPULISM_PLURALISM,
    "populismpluralism": AXIS_POPULISM_PLURALISM,
    "populism": AXIS_POPULISM_PLURALISM,
}


class PoliticalInferenceService:
    """
    Django service wrapper for axis-specific topic gating.

    - Loads a single NLI-style classifier once (singleton) to avoid reloading on each request.
    - For a selected analysis axis, first checks whether the text is *about the relevant topic*.
    - If the topic gate fails, downstream scoring is skipped.
    """
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # NLI model (used in both mock and real mode)
        self.topic_gate_model = None
        self.topic_gate_tokenizer = None
        self._entailment_idx = 0

        if MOCK_MODE:
            logger.info("🚧 Running in MOCK MODE - no real models loaded")
            self._initialized = True
            return

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self._load_models()
        self._initialized = True

    @classmethod
    def get_instance(cls):
        """Get singleton instance."""
        return cls()

    # =========================================================================
    # MODEL LOADING
    # =========================================================================

    def _load_models(self):
        """Load all models with error handling."""
        try:
            self._load_topic_gate_model()
            logger.info("All models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {e}")

    def _load_topic_gate_model(self):
        """Load the NLI model for axis topic gating."""
        if not ENABLE_TOPIC_GATE:
            return

        try:
            # NLI-ish model name (sequence classification) used with (premise, hypothesis)
            model_name = "mlburnham/Political_DEBATE_large_v1.0"
            self.topic_gate_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.topic_gate_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.topic_gate_model.to(self.device)
            self.topic_gate_model.eval()

            # Find entailment index
            config = self.topic_gate_model.config
            if hasattr(config, "label2id") and config.label2id:
                for label, idx in config.label2id.items():
                    if label.lower() in ["entailment", "entail"]:
                        self._entailment_idx = idx
                        break

            logger.info("✓ Topic-gate NLI model loaded")
        except Exception as e:
            logger.warning(f"Could not load topic-gate model: {e}")
            self.topic_gate_model = None

    # =========================================================================
    # AXIS-SPECIFIC TOPIC GATE (POSITIVE + NEGATIVE HYPOTHESES)
    # =========================================================================

    def normalize_axis(self, axis: Optional[str]) -> Optional[str]:
        """Normalize axis identifiers coming from clients."""
        if axis is None:
            return None
        a = axis.strip().lower()
        return AXIS_ALIASES.get(a, a)

    def check_topic_relevance(self, text: str, axis: str) -> Dict:
        """
        Axis-specific topic gate using both positive and negative hypotheses.

        Returns:
            Dict with:
              - pass (bool): True if the text is relevant to the selected axis topic
              - probability (float): positive entailment probability (main score)
              - positive_prob / negative_prob (float)
              - consistent (bool) + consistency_score (float)
              - reason (str)
        """
        axis = self.normalize_axis(axis) or axis

        if not ENABLE_TOPIC_GATE:
            return {
                "pass": True,
                "axis": axis,
                "probability": 1.0,
                "reason": "Topic gate disabled",
                "positive_prob": 1.0,
                "negative_prob": 0.0,
                "consistent": True,
            }

        if axis not in AXIS_TOPICS:
            # Fail-open so callers don't get blocked by an unknown axis value.
            return {
                "pass": True,
                "axis": axis,
                "probability": 1.0,
                "reason": f"Unknown axis '{axis}'. Topic gate skipped.",
                "positive_prob": 1.0,
                "negative_prob": 0.0,
                "consistent": True,
            }

        if MOCK_MODE:
            return self._mock_topic_check(text, axis)

        if self.topic_gate_model is None:
            # Fail-open: if the model isn't available, don't block scoring.
            return {
                "pass": True,
                "axis": axis,
                "probability": 1.0,
                "reason": "Topic gate model not loaded",
                "positive_prob": 1.0,
                "negative_prob": 0.0,
                "consistent": True,
            }

        positive_hypothesis = AXIS_TOPICS[axis]["positive"]
        negative_hypothesis = AXIS_TOPICS[axis]["negative"]

        try:
            positive_prob = self._check_hypothesis(text, positive_hypothesis)
            negative_prob = self._check_hypothesis(text, negative_hypothesis)

            # Consistency: positive should be ~ (1 - negative) if the model behaves coherently.
            consistency_score = abs(positive_prob - (1 - negative_prob))
            is_consistent = consistency_score <= CONSISTENCY_THRESHOLD

            pass_positive = positive_prob >= TOPIC_THRESHOLD
            pass_negative = negative_prob < (1 - TOPIC_THRESHOLD)

            passed = pass_positive and pass_negative and is_consistent

            return {
                "pass": passed,
                "axis": axis,
                "probability": round(positive_prob, 4),
                "threshold": TOPIC_THRESHOLD,
                "positive_prob": round(positive_prob, 4),
                "negative_prob": round(negative_prob, 4),
                "consistent": is_consistent,
                "consistency_score": round(consistency_score, 4),
                "reason": self._topic_reason(
                    axis=axis,
                    passed=passed,
                    prob=positive_prob,
                    is_consistent=is_consistent,
                    positive_prob=positive_prob,
                    negative_prob=negative_prob,
                ),
            }
        except Exception as e:
            logger.error(f"Topic gate failed for axis={axis}: {e}")
            # Fail-open on errors
            return {
                "pass": True,
                "axis": axis,
                "probability": 1.0,
                "reason": f"Topic gate failed: {e}",
                "positive_prob": 1.0,
                "negative_prob": 0.0,
                "consistent": True,
            }

    def _check_hypothesis(self, text: str, hypothesis: str) -> float:
        """Check a single hypothesis and return entailment probability."""
        inputs = self.topic_gate_tokenizer(
            text,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.topic_gate_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            prob = probs[0, self._entailment_idx].item()

        return prob

    def _mock_topic_check(self, text: str, axis: str) -> Dict:
        """
        Mock-mode topic gate.

        You said you do **not** want a keyword heuristic. In MOCK_MODE we therefore do not attempt
        to infer topic relevance without a model. Instead, we **bypass** the gate so you can test
        request/response plumbing without loading models.

        If you want the model to answer the hypotheses, set MOCK_MODE = False.
        """
        axis_norm = self.normalize_axis(axis) or axis
        return {
            "pass": True,
            "axis": axis_norm,
            "probability": 1.0,
            "threshold": TOPIC_THRESHOLD,
            "positive_prob": 1.0,
            "negative_prob": 0.0,
            "consistent": True,
            "consistency_score": 0.0,
            "reason": "MOCK_MODE enabled: topic gate bypassed (no keyword heuristic).",
            "mock": True,
        }

    def _topic_reason(
        self,
        axis: str,
        passed: bool,
        prob: float,
        is_consistent: bool,
        positive_prob: float,
        negative_prob: float,
    ) -> str:
        """Generate explanation for the topic gate result."""
        if not is_consistent:
            return (
                f"{axis}: inconsistent checks (positive: {positive_prob:.3f}, "
                f"negative: {negative_prob:.3f}). Topic relevance uncertain."
            )

        if passed:
            if prob > 0.8:
                return f"{axis}: topic relevance is clear (both checks agree)."
            if prob > 0.6:
                return f"{axis}: topic relevance likely (both checks agree)."
            return f"{axis}: topic relevance meets threshold (both checks agree)."

        if prob < 0.2:
            return f"{axis}: text does not appear relevant to this axis topic (both checks agree)."
        return f"{axis}: topic relevance below threshold (both checks agree)."

    # =========================================================================
    # MAIN PREDICTION INTERFACE
    # =========================================================================

    def predict_all(
        self,
        sentence: str,
        axis: Optional[str] = None,
        context: str = None,  # noqa: F841  (reserved for future use)
        skip_gate: bool = False,
    ) -> Dict:
        """
        Run an axis-specific topic gate.

        Args:
            sentence: Input text
            axis: Which analysis is selected (left_right, liberal_illiberal, populism_pluralism)
            skip_gate: If True, skip the axis topic gate

        Returns:
            Dict with topic_gate results and skip flags. (Downstream scoring not included here.)
        """
        results: Dict = {}

        axis_norm = self.normalize_axis(axis) if axis else None

        # If axis is not provided, we can't do axis-specific gating.
        if axis_norm is None and ENABLE_TOPIC_GATE and not skip_gate:
            results["topic_gate"] = {
                "pass": True,
                "axis": None,
                "probability": 1.0,
                "reason": "No axis provided; topic gate skipped.",
                "skipped": True,
            }
            return results

        if not skip_gate and ENABLE_TOPIC_GATE and axis_norm is not None:
            gate = self.check_topic_relevance(sentence, axis_norm)
            results["topic_gate"] = gate

            if not gate.get("pass", True):
                results["skipped"] = True
                results["message"] = (
                    f"Text does not match the topic required for axis '{axis_norm}'. "
                    "Scoring skipped."
                )
                if not gate.get("consistent", True):
                    results["message"] += (
                        " Note: positive and negative checks were inconsistent, "
                        "indicating uncertain classification."
                    )
                return results
        else:
            results["topic_gate"] = {
                "pass": True,
                "axis": axis_norm,
                "probability": 1.0,
                "reason": "Topic gate skipped",
                "skipped": True,
            }

        # If you later add per-axis scoring, do it here and attach to results.
        return results

    def get_3d_coordinates(
        self,
        sentence: str,
        axis: Optional[str] = None,
        context: str = None,
        skip_gate: bool = False,
    ) -> Dict:
        """
        Compatibility wrapper.

        Older callers may expect x/y/z + labels. Regression coordinates are deprecated here;
        we return None coordinates but include topic_gate info and skip messages.
        """
        results = self.predict_all(sentence, axis=axis, context=context, skip_gate=skip_gate)

        if results.get("skipped"):
            return {
                "x": None,
                "y": None,
                "z": None,
                "labels": {},
                "errors": [],
                "skipped": True,
                "topic_gate": results.get("topic_gate"),
                "message": results.get("message"),
            }

        return {
            "x": None,
            "y": None,
            "z": None,
            "labels": {},
            "errors": [],
            "topic_gate": results.get("topic_gate"),
            "message": "Regression models deprecated. Only axis topic gate performed.",
        }

    # =========================================================================
    # BACKWARD COMPATIBILITY
    # =========================================================================

    def is_political_text(self, text: str) -> Dict:
        """
        Backward-compatible wrapper.

        Previously this service exposed is_political_text(). Now gating is axis-specific.
        This wrapper performs a *generic* political check by reusing the liberal_illiberal gate,
        which is closest to broad political/democracy content, and returns keys similar to the old API.
        """
        gate = self.check_topic_relevance(text, AXIS_LIBERAL_ILLIBERAL)
        # Map into old shape as best as possible
        return {
            "is_political": gate.get("pass", True),
            "probability": gate.get("probability", 1.0),
            "reason": gate.get("reason", ""),
            "positive_prob": gate.get("positive_prob", 1.0),
            "negative_prob": gate.get("negative_prob", 0.0),
            "consistent": gate.get("consistent", True),
            "threshold": gate.get("threshold", TOPIC_THRESHOLD),
            "axis_used": AXIS_LIBERAL_ILLIBERAL,
            "deprecated": True,
        }
