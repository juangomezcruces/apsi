import os
import torch
import logging
from typing import Dict, Optional, Union
from django.conf import settings
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

# MOCK MODE - Set this to True to test without actual models
MOCK_MODE = False  # Change to False when you add real models



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
            if MOCK_MODE:
                logger.info("🚧 Running in MOCK MODE - no real models loaded")
                self._initialized = True
                return
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")
            
            # Initialize regression models only (no base model for now)
            self.lr_regression_model = None
            self.lr_regression_tokenizer = None
            self.libil_regression_model = None
            self.libil_regression_tokenizer = None
            self.pop_regression_model = None
            self.pop_regression_tokenizer = None
            
            self._load_models()
            self._initialized = True

    @classmethod
    def get_instance(cls):
        """Get singleton instance."""
        return cls()

    def _load_models(self):
        """Load all models with error handling."""
        try:
            self._load_regression_models()
            logger.info("All models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {e}")


    def _load_regression_models(self):
        """Load all regression models."""
        model_configs = [
            ("left_right_regression", "lr_regression"),
            ("liberal_illiberal_regression", "libil_regression"),
            ("populism_regression", "pop_regression")
        ]
        
        for config_key, attr_prefix in model_configs:
            model_path = settings.MODEL_PATHS[config_key]
            model, tokenizer = self._load_model_pair(model_path, config_key)
            setattr(self, f"{attr_prefix}_model", model)
            setattr(self, f"{attr_prefix}_tokenizer", tokenizer)

    def _load_model_pair(self, model_path: str, model_description: str):
        """Load a model and tokenizer pair with fallback options."""
        if not os.path.exists(model_path):
            logger.error(f"âœ— {model_description} model not found at {model_path}")
            return None, None

        try:
            # Try to load tokenizer from model directory first
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                logger.info(f"âœ“ Loaded {model_description} tokenizer from model directory")
            except:
                # Fallback tokenizers
                for fallback in ["roberta-base", "xlm-roberta-large"]:
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(fallback)
                        logger.info(f"âœ“ Using {fallback} tokenizer for {model_description}")
                        break
                    except:
                        continue
                else:
                    raise Exception("No suitable tokenizer found")
            
            # Load the model
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            model.to(self.device)
            model.eval()
            logger.info(f"âœ“ {model_description} model loaded successfully")
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"âœ— Error loading {model_description} model: {e}")
            return None, None

    # =========================================================================
    # POLITICAL TEXT PRE-CHECK
    # =========================================================================
    # PREDICTION METHODS
    # =========================================================================

    def _mock_prediction(self, dimension: str, score_name: str) -> Dict:
        """Return mock predictions for testing."""
        import random
        mock_score = random.uniform(1, 9)  # Random score between 1-9
        
        interpretations = {
            "Left-Right": ["Strongly Left", "Left", "Center", "Right", "Strongly Right"],
            "Liberal-Illiberal": ["Strongly Illiberal", "Illiberal", "Moderate", "Liberal", "Strongly Liberal"],
            "Populism": ["Strongly Anti-Populist", "Anti-Populist", "Moderate", "Populist", "Strongly Populist"]
        }
        
        interpretation = random.choice(interpretations.get(dimension, ["Unknown"]))
        
        return {
            f"predicted_{score_name.lower()}_score": round(mock_score, 4),
            "interpretation": interpretation,
            "raw_score": mock_score
        }

    def _predict_regression(self, sentence: str, context: str, model, tokenizer, dimension_name: str, score_name: str) -> Dict:
        """Generic regression prediction method."""
        if model is None or tokenizer is None:
            return {"error": f"{dimension_name} regression model not loaded"}

        try:
            if context is None:
                context = sentence

            inputs = tokenizer(
                sentence,
                context,
                return_tensors="pt",
                max_length=300,
                padding="max_length",
                truncation=True
            ).to(self.device)

            with torch.no_grad():
                outputs = model(**inputs)
                predicted_value = outputs.logits.squeeze().cpu().item()

            return {
                f"predicted_{score_name.lower()}_score": round(predicted_value, 4),
                "interpretation": self._interpret_score(predicted_value, dimension_name),
                "raw_score": predicted_value
            }
        except Exception as e:
            return {"error": f"{dimension_name} regression prediction failed: {str(e)}"}

    def _interpret_score(self, score: float, dimension: str) -> str:
        """Interpret regression scores based on dimension."""
        if dimension == "Left-Right":
            if score <= 2.5:
                return "Strongly Left"
            elif score <= 4.0:
                return "Left"
            elif score <= 6.0:
                return "Center"
            elif score <= 7.5:
                return "Right"
            else:
                return "Strongly Right"
        
        elif dimension == "Liberal-Illiberal":
            if score < 2:
                return "Strongly Illiberal"
            elif score < 4:
                return "Illiberal"
            elif score < 6:
                return "Moderate"
            elif score < 8:
                return "Liberal"
            else:
                return "Strongly Liberal"
        
        elif dimension == "Populism":
            if score < 2:
                return "Strongly Anti-Populist"
            elif score < 4:
                return "Anti-Populist"
            elif score < 6:
                return "Moderate"
            elif score < 8:
                return "Populist"
            else:
                return "Strongly Populist"
        
        return "Unknown"

    def predict_left_right_regression(self, sentence: str, context: str = None) -> Dict:
        """Get Left-Right regression predictions."""
        if MOCK_MODE:
            return self._mock_prediction("Left-Right", "V4_Scale")
        return self._predict_regression(
            sentence, context,
            self.lr_regression_model, self.lr_regression_tokenizer,
            "Left-Right", "V4_Scale"
        )

    def predict_liberal_illiberal_regression(self, sentence: str, context: str = None) -> Dict:
        """Get Liberal-Illiberal regression predictions."""
        if MOCK_MODE:
            return self._mock_prediction("Liberal-Illiberal", "V16")
        return self._predict_regression(
            sentence, context,
            self.libil_regression_model, self.libil_regression_tokenizer,
            "Liberal-Illiberal", "V16"
        )

    def predict_populism_regression(self, sentence: str, context: str = None) -> Dict:
        """Get Populism regression predictions."""
        if MOCK_MODE:
            return self._mock_prediction("Populism", "V8_Scale")
        return self._predict_regression(
            sentence, context,
            self.pop_regression_model, self.pop_regression_tokenizer,
            "Populism", "V8_Scale"
        )

    def predict_all(self, sentence: str, context: str = None) -> Dict:
        """Run all predictions on the input."""
        results = {}
        # Regression predictions
        regression_models = [
            ("left_right_regression", self.predict_left_right_regression),
            ("liberal_illiberal_regression", self.predict_liberal_illiberal_regression),
            ("populism_regression", self.predict_populism_regression)
        ]

        for model_name, predict_func in regression_models:
            result = predict_func(sentence, context)
            if "error" not in result:
                results[model_name] = result
            else:
                logger.warning(f"Error in {model_name}: {result['error']}")

        return results

    def get_3d_coordinates(self, sentence: str, context: str = None) -> Dict:
        """Get 3D coordinates for visualization."""
        results = self.predict_all(sentence, context)
        
        # Extract scores for 3D plotting
        coordinates = {
            'x': None,  # Left-Right
            'y': None,  # Liberal-Illiberal  
            'z': None,  # Populism
            'labels': {},
            'errors': [],
        }

        if "left_right_regression" in results:
            lr_result = results["left_right_regression"]
            coordinates['x'] = lr_result.get('raw_score')
            coordinates['labels']['left_right'] = lr_result.get('interpretation')

        if "liberal_illiberal_regression" in results:
            li_result = results["liberal_illiberal_regression"]
            coordinates['y'] = li_result.get('raw_score')
            coordinates['labels']['liberal_illiberal'] = li_result.get('interpretation')

        if "populism_regression" in results:
            pop_result = results["populism_regression"]
            coordinates['z'] = pop_result.get('raw_score')
            coordinates['labels']['populism'] = pop_result.get('interpretation')

        # Collect any errors
        for model_name, result in results.items():
            if isinstance(result, dict) and "error" in result:
                coordinates['errors'].append(f"{model_name}: {result['error']}")

        return coordinates
