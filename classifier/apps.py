from django.apps import AppConfig

class ClassifierConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'classifier'
    
    def ready(self):
        #TEMP DISABLED
        # Import the inference engine to initialize models on startup
        #from .inference_service import PoliticalInferenceService
        #PoliticalInferenceService.get_instance()
        print("🚧 Model loading disabled for initial testing")