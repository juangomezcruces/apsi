"""
DEPRECATED REGRESSION MODELS
============================

This file contains the deprecated direct regression models that were previously
used for Left-Right, Liberal-Illiberal, and Populism prediction.

These models have been removed from the main inference service but are kept here
for reference and potential future use.

DO NOT USE THIS IN PRODUCTION - these models are deprecated and no longer maintained.
"""

import os
import torch
import logging
from typing import Dict
from django.conf import settings
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


class DeprecatedRegressionModels:
    """
    Container for deprecated regression models.
    
    WARNING: This class is deprecated and should not be used in production.
    It is kept only for reference purposes.
    """
    
    def __init__(self):
        logger.warning("⚠️ Initializing DEPRECATED regression models - DO NOT USE IN PRODUCTION")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize regression models
        self.lr_regression_model = None
        self.lr_regression_tokenizer = None
        self.libil_regression_model = None
        self.libil_regression_tokenizer = None
        self.pop_regression_model = None
        self.pop_regression_tokenizer = None
        
        self._load_regression_models()

    def _load_regression_models(self):
        """Load all regression models."""
        logger.warning("Loading deprecated regression models...")
        
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
            logger.error(f"✗ {model_description} model not found at {model_path}")
            return None, None

        try:
            # Try to load tokenizer from model directory first
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                logger.info(f"✓ Loaded {model_description} tokenizer from model directory")
            except:
                # Fallback tokenizers
                for fallback in ["roberta-base", "xlm-roberta-large"]:
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(fallback)
                        logger.info(f"✓ Using {fallback} tokenizer for {model_description}")
                        break
                    except:
                        continue
                else:
                    raise Exception("No suitable tokenizer found")
            
            # Load the model
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            model.to(self.device)
            model.eval()
            logger.info(f"✓ {model_description} model loaded successfully")
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"✗ Error loading {model_description} model: {e}")
            return None, None

    def _predict_regression(self, sentence: str, context: str, model, tokenizer, 
                           dimension_name: str, score_name: str) -> Dict:
        """Generic regression prediction method."""
        logger.warning(f"⚠️ Using DEPRECATED regression model for {dimension_name}")
        
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
                "raw_score": predicted_value,
                "deprecated": True,
                "warning": "This prediction uses deprecated regression models"
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
        """
        DEPRECATED: Get Left-Right regression predictions.
        Use hypothesis-based or response-based methods instead.
        """
        return self._predict_regression(
            sentence, context,
            self.lr_regression_model, self.lr_regression_tokenizer,
            "Left-Right", "V4_Scale"
        )

    def predict_liberal_illiberal_regression(self, sentence: str, context: str = None) -> Dict:
        """
        DEPRECATED: Get Liberal-Illiberal regression predictions.
        Use hypothesis-based or response-based methods instead.
        """
        return self._predict_regression(
            sentence, context,
            self.libil_regression_model, self.libil_regression_tokenizer,
            "Liberal-Illiberal", "V16"
        )

    def predict_populism_regression(self, sentence: str, context: str = None) -> Dict:
        """
        DEPRECATED: Get Populism regression predictions.
        Use hypothesis-based or response-based methods instead.
        """
        return self._predict_regression(
            sentence, context,
            self.pop_regression_model, self.pop_regression_tokenizer,
            "Populism", "V8_Scale"
        )

    def predict_all(self, sentence: str, context: str = None) -> Dict:
        """
        DEPRECATED: Run all regression predictions on the input.
        Use hypothesis-based or response-based methods instead.
        """
        logger.warning("⚠️ Using DEPRECATED predict_all() method")
        
        results = {}
        
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

        results['deprecated'] = True
        results['warning'] = (
            "These predictions use deprecated regression models. "
            "Please use hypothesis-based or response-based approaches instead."
        )

        return results

    def get_3d_coordinates(self, sentence: str, context: str = None) -> Dict:
        """
        DEPRECATED: Get 3D coordinates for visualization.
        This method is deprecated and should not be used.
        """
        logger.warning("⚠️ Using DEPRECATED get_3d_coordinates() method")
        
        results = self.predict_all(sentence, context)
        
        # Extract scores for 3D plotting
        coordinates = {
            'x': None,  # Left-Right
            'y': None,  # Liberal-Illiberal  
            'z': None,  # Populism
            'labels': {},
            'errors': [],
            'deprecated': True,
            'warning': results.get('warning', '')
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


# Example usage (DO NOT USE IN PRODUCTION)
if __name__ == "__main__":
    logger.warning("="*80)
    logger.warning("RUNNING DEPRECATED REGRESSION MODELS - DO NOT USE IN PRODUCTION")
    logger.warning("="*80)
    
    # This is just for testing/demonstration
    # DO NOT use this in actual application code
    deprecated_models = DeprecatedRegressionModels()
    
    test_text = "We need higher taxes on the wealthy to fund universal healthcare."
    
    results = deprecated_models.predict_all(test_text)
    print("\nDEPRECATED Results:")
    print(results)
    
    logger.warning("="*80)
    logger.warning("END OF DEPRECATED CODE DEMONSTRATION")
    logger.warning("="*80)