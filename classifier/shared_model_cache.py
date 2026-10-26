import threading

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import logging

logger = logging.getLogger(__name__)

class SharedModelCache:
    _instance = None
    # One model and one (Rust) tokenizer are shared by every scorer, and views.py
    # runs the scorers in parallel threads. The fast tokenizer is not thread-safe
    # ("Already borrowed"), so every tokenize + forward pass takes this lock.
    # On a single device the forward passes would queue anyway.
    _inference_lock = threading.Lock()
    _models = {}
    _tokenizers = {}
    _device = None  # Initialize as None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Only initialize device once
        if SharedModelCache._device is None:
            SharedModelCache._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"  Device: {SharedModelCache._device}")  # Use print so it always shows
            if torch.cuda.is_available():
                print(f"  GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"SharedModelCache initialized with device: {SharedModelCache._device}")
    
    def get_model_and_tokenizer(self, model_name):
        if model_name not in self._models:
            logger.info(f"Loading NLI model (first time): {model_name}...")
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            model = model.to(self._device)  # Move to GPU if available
            model.eval()  # Set to evaluation mode
            
            self._models[model_name] = model
            self._tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)
            logger.info(f"Model cached: {model_name} on {self._device}")
        else:
            logger.info(f"Using cached NLI model: {model_name}")
        
        return self._models[model_name], self._tokenizers[model_name]
    
    def inference_lock(self):
        """Hold this around any tokenizer or model call on the shared model."""
        return self._inference_lock

    def get_device(self):
        """Get the device being used (cuda or cpu)"""
        return self._device
    
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model cache cleared")