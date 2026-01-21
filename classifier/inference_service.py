import os
import torch
import logging
from typing import Dict, Optional, Union
from django.conf import settings
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

# MOCK MODE - Set this to True to test without actual models
MOCK_MODE = False  # Change to False when you add real models

# Political pre-check configuration
ENABLE_POLITICAL_PRECHECK = True  # Set False to disable pre-check
POLITICAL_THRESHOLD = 0.3  # Probability threshold (0-1)
CONSISTENCY_THRESHOLD = 0.2  # Maximum acceptable difference between positive and negative checks


class PoliticalInferenceService:
    """
    Django service wrapper for the political text classification models.
    Singleton pattern to avoid reloading models on each request.
    """
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            # Political detector (used in both mock and real mode)
            self.political_detector_model = None
            self.political_detector_tokenizer = None
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

    def _load_models(self):
        """Load all models with error handling."""
        try:
            self._load_political_detector()
            logger.info("All models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {e}")

    def _load_political_detector(self):
        """Load the NLI model for political text detection."""
        if not ENABLE_POLITICAL_PRECHECK:
            return
        
        try:
            model_name = "mlburnham/Political_DEBATE_large_v1.0"
            self.political_detector_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.political_detector_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.political_detector_model.to(self.device)
            self.political_detector_model.eval()
            
            # Find entailment index
            config = self.political_detector_model.config
            if hasattr(config, 'label2id') and config.label2id:
                for label, idx in config.label2id.items():
                    if label.lower() in ['entailment', 'entail']:
                        self._entailment_idx = idx
                        break
            
            logger.info("✓ Political text detector loaded")
        except Exception as e:
            logger.warning(f"Could not load political detector: {e}")
            self.political_detector_model = None

    # =========================================================================
    # POLITICAL TEXT PRE-CHECK WITH REVERSED VALIDATION
    # =========================================================================
    
    def is_political_text(self, text: str) -> Dict:
        """
        Check if text is about political topics using both positive and negative hypotheses.
        Only returns is_political=True if both checks agree.
        
        Returns:
            Dict with 'is_political' (bool), 'probability' (float), 'reason' (str), 
            'positive_prob' (float), 'negative_prob' (float), 'consistent' (bool)
        """
        if not ENABLE_POLITICAL_PRECHECK:
            return {
                'is_political': True, 
                'probability': 1.0, 
                'reason': 'Pre-check disabled',
                'positive_prob': 1.0,
                'negative_prob': 0.0,
                'consistent': True
            }
        
        if MOCK_MODE:
            return self._mock_political_check(text)
        
        if self.political_detector_model is None:
            return {
                'is_political': True, 
                'probability': 1.0, 
                'reason': 'Detector not loaded',
                'positive_prob': 1.0,
                'negative_prob': 0.0,
                'consistent': True
            }
        
        # Positive hypothesis
        positive_hypothesis = (
            "This text discusses political topics such as government, parties, policies, "
            "elections, ideology, or governance."
        )
        
        # Negative hypothesis
        negative_hypothesis = (
            "This text does not discuss political topics such as government, parties, policies, "
            "elections, ideology, or governance."
        )
        
        try:
            # Check positive hypothesis
            positive_prob = self._check_hypothesis(text, positive_hypothesis)
            
            # Check negative hypothesis
            negative_prob = self._check_hypothesis(text, negative_hypothesis)
            
            # Calculate consistency - they should be opposites
            # If positive is high, negative should be low, and vice versa
            consistency_score = abs(positive_prob - (1 - negative_prob))
            is_consistent = consistency_score <= CONSISTENCY_THRESHOLD
            
            # Determine if political based on positive check
            is_political_positive = positive_prob >= POLITICAL_THRESHOLD
            is_political_negative = negative_prob < (1 - POLITICAL_THRESHOLD)
            
            # Only mark as political if both checks agree AND they're consistent
            is_political = is_political_positive and is_political_negative and is_consistent
            
            # Use positive probability as the main score
            main_probability = positive_prob
            
            return {
                'is_political': is_political,
                'probability': round(main_probability, 4),
                'threshold': POLITICAL_THRESHOLD,
                'positive_prob': round(positive_prob, 4),
                'negative_prob': round(negative_prob, 4),
                'consistent': is_consistent,
                'consistency_score': round(consistency_score, 4),
                'reason': self._get_political_reason(
                    is_political, main_probability, 
                    is_consistent, positive_prob, negative_prob
                )
            }
        except Exception as e:
            logger.error(f"Political check failed: {e}")
            return {
                'is_political': True, 
                'probability': 1.0, 
                'reason': f'Check failed: {e}',
                'positive_prob': 1.0,
                'negative_prob': 0.0,
                'consistent': True
            }

    def _check_hypothesis(self, text: str, hypothesis: str) -> float:
        """Check a single hypothesis and return entailment probability."""
        inputs = self.political_detector_tokenizer(
            text, hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.political_detector_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            prob = probs[0, self._entailment_idx].item()
        
        return prob

    def _mock_political_check(self, text: str) -> Dict:
        """Simple keyword-based check for mock mode with reversed validation."""
        political_keywords = {
            'government', 'parliament', 'congress', 'election', 'vote', 'party',
            'democrat', 'republican', 'conservative', 'liberal', 'socialist',
            'policy', 'legislation', 'tax', 'budget', 'democracy', 'freedom',
            'rights', 'constitution', 'political', 'politician', 'president',
            'minister', 'law', 'regulation', 'campaign', 'populist', 'reform'
        }
        
        words = set(text.lower().split())
        matches = words.intersection(political_keywords)
        
        # Positive probability based on keyword matches
        positive_prob = min(1.0, len(matches) * 0.3) if matches else 0.1
        
        # Negative probability should be inverse
        negative_prob = 1.0 - positive_prob
        
        is_political = positive_prob >= POLITICAL_THRESHOLD and negative_prob < (1 - POLITICAL_THRESHOLD)
        
        # Mock consistency - perfect inverse relationship
        consistency_score = abs(positive_prob - (1 - negative_prob))
        is_consistent = True  # Mock always consistent
        
        return {
            'is_political': is_political,
            'probability': round(positive_prob, 4),
            'threshold': POLITICAL_THRESHOLD,
            'positive_prob': round(positive_prob, 4),
            'negative_prob': round(negative_prob, 4),
            'consistent': is_consistent,
            'consistency_score': round(consistency_score, 4),
            'reason': f"Mock check: found {len(matches)} political keywords",
            'mock': True
        }

    def _get_political_reason(self, is_political: bool, prob: float, 
                              is_consistent: bool, positive_prob: float, 
                              negative_prob: float) -> str:
        """Generate explanation for the political check result."""
        if not is_consistent:
            return (
                f"Inconsistent checks detected (positive: {positive_prob:.3f}, "
                f"negative: {negative_prob:.3f}). Text classification uncertain."
            )
        
        if is_political:
            if prob > 0.8:
                return "Text is clearly about political topics (both checks agree)"
            elif prob > 0.6:
                return "Text appears to discuss political content (both checks agree)"
            else:
                return "Text contains some political content (both checks agree)"
        else:
            if prob < 0.2:
                return "Text does not appear to be about politics (both checks agree)"
            else:
                return "Text has minimal political content below threshold (both checks agree)"

    # =========================================================================
    # MAIN PREDICTION INTERFACE
    # =========================================================================

    def predict_all(self, sentence: str, context: str = None, skip_precheck: bool = False) -> Dict:
        """
        Run political pre-check. Regression models have been deprecated.
        
        Args:
            sentence: Input text to check
            context: Optional context (not used in current implementation)
            skip_precheck: If True, skip the political relevance check
            
        Returns:
            Dict with political_check results or not_political flag
        """
        results = {}
        
        # Pre-check if text is political (unless skipped)
        if not skip_precheck and ENABLE_POLITICAL_PRECHECK:
            political_check = self.is_political_text(sentence)
            results['political_check'] = political_check
            
            if not political_check['is_political']:
                results['not_political'] = True
                results['message'] = (
                    "Text does not appear to be about political topics. "
                    "Scoring skipped."
                )
                if not political_check.get('consistent', True):
                    results['message'] += (
                        " Note: Positive and negative checks were inconsistent, "
                        "indicating uncertain classification."
                    )
                return results
        else:
            # If skipping precheck, add a note
            results['political_check'] = {
                'is_political': True,
                'probability': 1.0,
                'reason': 'Pre-check skipped',
                'skipped': True
            }

        return results

    def get_3d_coordinates(self, sentence: str, context: str = None, skip_precheck: bool = False) -> Dict:
        """
        Get political check results. 3D coordinates from regression models are deprecated.
        
        Returns:
            Dict with political_check information
        """
        results = self.predict_all(sentence, context, skip_precheck=skip_precheck)
        
        # Handle non-political text
        if results.get('not_political'):
            return {
                'x': None, 'y': None, 'z': None,
                'labels': {},
                'errors': [],
                'not_political': True,
                'political_check': results.get('political_check'),
                'message': results.get('message')
            }
        
        # Return structure for compatibility, but no regression coordinates
        return {
            'x': None,
            'y': None,
            'z': None,
            'labels': {},
            'errors': [],
            'political_check': results.get('political_check'),
            'message': 'Regression models deprecated. Only political pre-check performed.'
        }