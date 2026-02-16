from django.urls import path
from . import views

app_name = 'classifier'

urlpatterns = [
    path('', views.index, name='index'),
    path('analysis/', views.analysis, name='analysis'),
    path('classify/', views.classify_text, name='classify_text'),
    path('documentation', views.documentation, name='documentation'),
    path('privacy/', views.privacy_notice, name='privacy_notice'),
    path('imprint/', views.imprint, name='imprint'),
    path('contact/', views.contact, name='contact'),
]
