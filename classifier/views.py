import json
import logging
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.contrib import messages
from .forms import TextClassificationForm
from .inference_service import PoliticalInferenceService

logger = logging.getLogger(__name__)

def index(request):
    """Main page with classification form."""
    form = TextClassificationForm()
    return render(request, 'classifier/index.html', {'form': form})

def classify_text(request):
    """Handle form submission and show results."""
    if request.method == 'POST':
        form = TextClassificationForm(request.POST)
        if form.is_valid():
            text = form.cleaned_data['text']
            context = form.cleaned_data.get('context', '') or None
            
            try:
                # Get inference service
                inference_service = PoliticalInferenceService.get_instance()
                
                # Run predictions
                results = inference_service.predict_all(text, context)
                coordinates = inference_service.get_3d_coordinates(text, context)
                
                context_data = {
                    'form': form,
                    'results': results,
                    'coordinates': coordinates,
                    'input_text': text,
                    'input_context': context,
                    'coordinates_json': json.dumps(coordinates)
                }
                
                return render(request, 'classifier/results.html', context_data)
                
            except Exception as e:
                logger.error(f"Classification error: {str(e)}")
                messages.error(request, f"An error occurred during classification: {str(e)}")
                return render(request, 'classifier/index.html', {'form': form})
        else:
            return render(request, 'classifier/index.html', {'form': form})
    else:
        return index(request)

@csrf_exempt
@require_http_methods(["POST"])
def api_classify(request):
    """API endpoint for classification."""
    try:
        data = json.loads(request.body)
        text = data.get('text', '').strip()
        context = data.get('context', '').strip() or None
        
        if not text:
            return JsonResponse({'error': 'Text is required'}, status=400)
        
        if len(text) > 2000:
            return JsonResponse({'error': 'Text too long (max 2000 characters)'}, status=400)
        
        # Get inference service
        inference_service = PoliticalInferenceService.get_instance()
        
        # Run predictions
        results = inference_service.predict_all(text, context)
        coordinates = inference_service.get_3d_coordinates(text, context)
        
        return JsonResponse({
            'success': True,
            'results': results,
            'coordinates': coordinates
        })
        
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        logger.error(f"API classification error: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)