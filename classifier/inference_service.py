import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# MOCK MODE - Set this to True to test without actual models
MOCK_MODE = False  # Change to False when you add real models


class PoliticalInferenceService:
    """
    Django service wrapper for model inference.


    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        if MOCK_MODE:
            logger.info("🚧 Running in MOCK MODE - no real models loaded")
            self._initialized = True
            return

        # If/when you add real models, load them here.
        # self.device = ...
        # self._load_models()
        self._initialized = True

    @classmethod
    def get_instance(cls):
        """Get singleton instance."""
        return cls()

    # ---------------------------------------------------------------------
    # Public API used by views
    # ---------------------------------------------------------------------

    def predict_all(self, sentence: str, context: Optional[str] = None) -> Dict:
        """
        Run direct regression models (if available).

        The political relevance pre-check has been removed, so this method
        should always attempt to score the input.

        Returns:
            A dict of results (empty if direct models are not implemented).
        """
        if MOCK_MODE:
            return {}

        # Direct regression models were previously removed/disabled in this project.
        # Keep the return value stable for downstream views/templates.
        return {}

    def get_3d_coordinates(self, sentence: str, context: Optional[str] = None) -> Dict:
        """
        Return 3D coordinate structure for compatibility with existing UI.

        Returns:
            Dict with keys x/y/z/labels/errors (all None/empty if direct models
            are not implemented).
        """
        _ = self.predict_all(sentence, context)

        return {
            "x": None,
            "y": None,
            "z": None,
            "labels": {},
            "errors": [],
        }
