import numpy as np
import torch
from .shared_model_cache import SharedModelCache

# ============================================================================
# LEFT-RIGHT ECONOMIC HYPOTHESIS-BASED SCORER (SIMPLIFIED TOPIC PRECHECK)
# ============================================================================

class LeftRightEconomicScorer:
    def __init__(self, model_name="mlburnham/Political_DEBATE_large_v1.0"):
        cache = SharedModelCache()
        self.model, self.tokenizer = cache.get_model_and_tokenizer(model_name)

        # Make inference deterministic (turn off dropout, etc.)
        self.model.eval()

        # Determine device and keep it consistent
        try:
            self.device = next(self.model.parameters()).device
        except StopIteration:
            self.device = torch.device("cpu")

        # Robust entailment label index detection
        self.entailment_idx = self._find_entailment_index()

        # --------------------------------------------------------------------
        # SIMPLE TOPIC PRECHECK (YOUR EXACT SCOPE DEFINITION)
        # --------------------------------------------------------------------
        self.topic_question = (
            "Does this text discuss political rhetoric, governance approaches, or institutional legitimacy? "
            "This includes references to populist rhetoric (challenging institutions, emphasizing popular will) "
            "or pluralist rhetoric (supporting checks and balances, minority rights, compromise)."
        )
        self.topic_threshold = 0.50  # keep your chosen threshold

        # --------------------------------------------------------------------
        # Left-Right Economic hypotheses
        # --------------------------------------------------------------------
        self.left_right_hypotheses = {
            # Left
            "The author of this text believes corporations should pay higher taxes": (1.0, "left"),
            "The author of this text believes wealthy individuals should pay higher tax rates": (1.0, "left"),
            "The author of this text believes government should increase spending on healthcare": (1.0, "left"),
            "The author of this text believes government should increase spending on education": (1.0, "left"),
            "The author of this text believes unemployment benefits should be expanded": (1.0, "left"),
            "The author of this text believes government should provide universal healthcare": (1.0, "left"),
            "The author of this text believes banks and financial institutions should be heavily regulated": (1.0, "left"),
            "The author of this text believes environmental regulations on business are necessary": (1.0, "left"),
            "The author of this text believes utilities should be publicly owned": (1.0, "left"),
            "The author of this text believes government should break up large corporations": (1.0, "left"),
            "The author of this text believes minimum wage laws should be strengthened": (1.0, "left"),
            "The author of this text believes unions should have more power": (1.0, "left"),
            "The author of this text believes government should reduce income inequality": (1.0, "left"),
            "The author of this text believes public investment creates jobs": (1.0, "left"),
            "The author of this text believes social safety nets should be expanded": (1.0, "left"),

            # Right
            "The author of this text believes corporate tax rates should be lowered": (1.0, "right"),
            "The author of this text believes income taxes should be reduced": (1.0, "right"),
            "The author of this text believes government spending on social programs should be cut": (1.0, "right"),
            "The author of this text believes welfare programs should be reduced": (1.0, "right"),
            "The author of this text believes healthcare should be privatized": (1.0, "right"),
            "The author of this text believes education should be privatized": (1.0, "right"),
            "The author of this text believes financial regulations should be eliminated": (1.0, "right"),
            "The author of this text believes environmental regulations hurt business competitiveness": (1.0, "right"),
            "The author of this text believes government services should be privatized": (1.0, "right"),
            "The author of this text believes large corporations drive economic growth": (1.0, "right"),
            "The author of this text believes minimum wage laws hurt employment": (1.0, "right"),
            "The author of this text believes unions hurt economic competitiveness": (1.0, "right"),
            "The author of this text believes income inequality reflects merit and effort": (1.0, "right"),
            "The author of this text believes private investment is more efficient than public": (1.0, "right"),
            "The author of this text believes social programs create dependency": (1.0, "right"),
        }

    # ------------------------------------------------------------------------
    # Label index detection (critical for correct precheck)
    # ------------------------------------------------------------------------
    def _find_entailment_index(self) -> int:
        """
        Robustly detect entailment index:
        1) Use config.label2id/id2label if present (best)
        2) Otherwise, assume standard 3-way MNLI ordering: contradiction=0, neutral=1, entailment=2
        3) Fallback to last label if model is binary/odd
        """
        config = getattr(self.model, "config", None)
        if config is not None:
            # Try label2id
            label2id = getattr(config, "label2id", None) or {}
            for label, idx in label2id.items():
                if isinstance(label, str) and label.lower() in ("entailment", "entails", "entail"):
                    return int(idx)

            # Try id2label
            id2label = getattr(config, "id2label", None) or {}
            for idx, label in id2label.items():
                if isinstance(label, str) and label.lower() in ("entailment", "entails", "entail"):
                    return int(idx)

            # If 3 labels, default to MNLI ordering
            num_labels = getattr(config, "num_labels", None)
            if num_labels == 3:
                return 2

            # If 2 labels, many models put "entailment"/"positive" at 1
            if num_labels == 2:
                return 1

        # Absolute fallback: pick last logit
        return -1

    # ------------------------------------------------------------------------
    # Core entailment probability helper
    # ------------------------------------------------------------------------
    def _entailment_prob(self, premise: str, hypothesis: str) -> float:
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        # Ensure tensors are on same device as model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

            # Handle -1 entailment_idx fallback (last label)
            idx = self.entailment_idx if self.entailment_idx >= 0 else (probs.shape[-1] - 1)
            return float(probs[idx].item())

    # ------------------------------------------------------------------------
    # SIMPLE TOPIC PRECHECK (ONE QUESTION -> prob -> threshold)
    # ------------------------------------------------------------------------
    def topic_precheck(self, text: str) -> dict:
        score = float(self._entailment_prob(text, self.topic_question))
        passed = score >= float(self.topic_threshold)
        return {
            "passed_precheck": passed,
            "precheck_score": score,
            "precheck_threshold": float(self.topic_threshold),
            "precheck_question": self.topic_question,
            "error": None if passed else "Text did not pass the topic precheck.",
        }

    # ------------------------------------------------------------------------
    # Hypothesis scoring (only runs AFTER precheck passes)
    # ------------------------------------------------------------------------
    def get_hypothesis_probabilities(self, text: str) -> np.ndarray:
        probs = []
        for hypothesis in self.left_right_hypotheses.keys():
            p = self._entailment_prob(text, hypothesis)
            probs.append(p)
        return np.array(probs, dtype=float)

    def score_left_right(self, text: str) -> dict:
        # 1) Precheck FIRST — hard stop if it fails
        pre = self.topic_precheck(text)
        if not pre["passed_precheck"]:
            return {
                "text": text,
                **pre,
            }

        # 2) Only now run the expensive hypothesis inference
        probs = self.get_hypothesis_probabilities(text)

        left_probs = []
        right_probs = []
        hypothesis_results = []

        for i, (hypothesis, (weight, direction)) in enumerate(self.left_right_hypotheses.items()):
            prob = float(probs[i])
            hypothesis_results.append({
                "hypothesis": hypothesis,
                "probability": prob,
                "direction": direction,
            })
            if direction == "left":
                left_probs.append(prob * weight)
            else:
                right_probs.append(prob * weight)

        left_avg = float(np.mean(left_probs)) if left_probs else 0.0
        right_avg = float(np.mean(right_probs)) if right_probs else 0.0

        difference = left_avg - right_avg
        final_score = 5 - (difference * 5)   # left -> low numbers; right -> high numbers
        final_score = float(np.clip(final_score, 0, 10))

        # Simple confidence (keep it simple): distance from center scaled by signal strength
        # You can replace this with your prior confidence method if you prefer.
        signal = float(np.mean(probs)) if len(probs) else 0.0
        confidence = float(np.clip(signal, 0.0, 1.0))

        # Interpretation
        if final_score < 2:
            interpretation = "Far Left"
        elif final_score < 4:
            interpretation = "Left"
        elif final_score < 6:
            interpretation = "Center"
        elif final_score < 8:
            interpretation = "Right"
        else:
            interpretation = "Far Right"

        left_hyps = [h for h in hypothesis_results if h["direction"] == "left"]
        right_hyps = [h for h in hypothesis_results if h["direction"] == "right"]
        top_left = sorted(left_hyps, key=lambda x: x["probability"], reverse=True)[:5]
        top_right = sorted(right_hyps, key=lambda x: x["probability"], reverse=True)[:5]

        return {
            "text": text,
            **pre,
            "score": final_score,
            "confidence": confidence,
            "interpretation": interpretation,
            "left_avg": left_avg,
            "right_avg": right_avg,
            "top_left_hypotheses": top_left,
            "top_right_hypotheses": top_right,
        }

    def quick_score(self, text: str) -> float:
        result = self.score_left_right(text)
        # If out-of-scope, you might want to return None instead; keeping float here will raise.
        if not result.get("passed_precheck", True):
            raise ValueError(result.get("error") or "Text did not pass the topic precheck.")
        return float(result["score"])
