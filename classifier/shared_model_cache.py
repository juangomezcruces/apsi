import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import logging

logger = logging.getLogger(__name__)

class SharedModelCache:
    _instance = None
    _models = {}
    _tokenizers = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_model_and_tokenizer(self, model_name):
        if model_name not in self._models:
            logger.info(f"Loading NLI model (first time): {model_name}...")
            self._models[model_name] = AutoModelForSequenceClassification.from_pretrained(model_name)
            self._tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)
            logger.info(f"Model cached: {model_name}")
        else:
            logger.info(f"Using cached NLI model: {model_name}")
        
        return self._models[model_name], self._tokenizers[model_name]
    
    def clear_cache(self):
        """Clear all cached models to free memory"""
        for model in self._models.values():
            del model
        for tokenizer in self._tokenizers.values():
            del tokenizer
        self._models.clear()
        self._tokenizers.clear()
        
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Model cache cleared")