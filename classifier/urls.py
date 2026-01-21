from django.urls import path
from . import views

app_name = 'classifier'

urlpatterns = [
    path('', views.index, name='index'),
    path('classify/', views.classify_text, name='classify_text'),
    path('api/classify/', views.api_classify, name='api_classify'),
    path('privacy/', views.privacy_notice, name='privacy_notice'),
    path('imprint/', views.imprint, name='imprint'),
    path('contact/', views.contact, name='contact'),
]
