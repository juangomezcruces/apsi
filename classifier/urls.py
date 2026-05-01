from django.urls import path
from . import views
from . import views_longdoc

app_name = 'classifier'

urlpatterns = [
    path('', views.index, name='index'),
    path('analysis/', views.analysis, name='analysis'),
    path('classify/', views.classify_text, name='classify_text'),
    path('documentation/', views.documentation, name='documentation'),
    path('privacy/', views.privacy_notice, name='privacy_notice'),
    path('imprint/', views.imprint, name='imprint'),
    path('contact/', views.contact, name='contact'),
    path('aboutus/', views.aboutus, name='aboutus'),
    path('faq/', views.faq, name='faq'),
    path('save-result/', views.save_result, name='save_result'),
    path('delete-result/<int:result_id>/', views.delete_result, name='delete_result'),
    # Long document scorer
    path('longdoc/', views_longdoc.longdoc, name='longdoc'),
    path('longdoc/score/', views_longdoc.longdoc_score, name='longdoc_score'),
]
