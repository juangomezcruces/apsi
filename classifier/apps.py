from django.apps import AppConfig

class ClassifierConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'classifier'

    def ready(self):
        from .shared_model_cache import SharedModelCache
        cache = SharedModelCache()
        cache.get_model_and_tokenizer("mlburnham/Political_DEBATE_large_v1.0")    
