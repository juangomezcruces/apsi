import json
import logging
import random
import numpy as np
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.contrib import messages
from django.conf import settings
from .forms import TextClassificationForm
from .inference_service import PoliticalInferenceService

logger = logging.getLogger(__name__)

def get_alternative_scorers():
    """Initialize alternative hypothesis scorers"""
    from .inference_service import MOCK_MODE  # Import at runtime to get current value
    
    if MOCK_MODE:
        logger.info("🚧 Alternative approaches running in MOCK MODE (controlled by inference_service.MOCK_MODE)")
        return None
    
    try:
        from . import alternative
        from . import alternativeLib  
        from . import alternativePop
        
        return {
            'left_right': alternative.LeftRightEconomicScorer(),
            'liberal_illiberal': alternativeLib.LiberalIlliberalScorer(),
            'populism_pluralism': alternativePop.PopulismPluralismScorer()
        }
    except Exception as e:
        logger.warning(f"Could not load alternative scorers (will use mock data): {e}")
        return None

def generate_alternative_scores(text, scorers=None):
    """Generate alternative hypothesis scores using real models or mock data"""
    from .inference_service import MOCK_MODE  # Import at runtime to get current value
    
    if not getattr(settings, 'ENABLE_ALTERNATIVE_SCORES', False):
        logger.debug("Alternative scores disabled in settings")
        return None
    
    alternative_scores = {}
    
    try:
        if scorers and not MOCK_MODE:
            # Real model predictions
            logger.info("🔬 Using real alternative hypothesis models")
            
            # Left-Right Hypothesis Scoring
            if 'left_right' in scorers:
                try:
                    logger.debug("Running left-right hypothesis scoring...")
                    lr_result = scorers['left_right'].score_left_right(text)
                    
                    alternative_scores['left_right_hypothesis'] = {
                        'score': round(lr_result.get('score', 5.0), 2),
                        'confidence': round(lr_result.get('confidence', 0.8), 3),
                        'interpretation': lr_result.get('interpretation', 'Center')
                    }
                    logger.debug(f"✓ Left-right hypothesis: {lr_result.get('score', 'N/A'):.2f} ({lr_result.get('interpretation', 'N/A')})")
                except Exception as e:
                    logger.error(f"Left-Right alternative scoring failed: {e}")
                    alternative_scores['left_right_hypothesis'] = generate_mock_score('left_right')
            
            # Liberal-Illiberal Hypothesis Scoring  
            if 'liberal_illiberal' in scorers:
                try:
                    logger.debug("Running liberal-illiberal hypothesis scoring...")
                    li_result = scorers['liberal_illiberal'].score_liberal_illiberal(text)
                    alternative_scores['liberal_illiberal_hypothesis'] = {
                        'score': round(li_result.get('score', 5.0), 2),
                        'confidence': round(li_result.get('confidence', 0.8), 3),
                        'interpretation': li_result.get('interpretation', 'Moderate')
                    }
                    logger.debug(f"✓ Liberal-illiberal hypothesis: {li_result.get('score', 'N/A'):.2f}")
                except Exception as e:
                    logger.error(f"Liberal-Illiberal alternative scoring failed: {e}")
                    alternative_scores['liberal_illiberal_hypothesis'] = generate_mock_score('liberal_illiberal')
            
            # Populism-Pluralism Hypothesis Scoring
            if 'populism_pluralism' in scorers:
                try:
                    logger.debug("Running populism-pluralism hypothesis scoring...")
                    pp_result = scorers['populism_pluralism'].score_populism_pluralism(text)
                    alternative_scores['populism_pluralism_hypothesis'] = {
                        'score': round(pp_result.get('score', 5.0), 2),
                        'confidence': round(pp_result.get('confidence', 0.8), 3),
                        'interpretation': pp_result.get('interpretation', 'Moderate')
                    }
                    logger.debug(f"✓ Populism-pluralism hypothesis: {pp_result.get('score', 'N/A'):.2f}")
                except Exception as e:
                    logger.error(f"Populism-Pluralism alternative scoring failed: {e}")
                    alternative_scores['populism_pluralism_hypothesis'] = generate_mock_score('populism_pluralism')
            
        else:
            # Fall back to mock data
            if MOCK_MODE:
                logger.info("🚧 Using mock alternative scores (MOCK_MODE = True)")
            else:
                logger.warning("Using mock alternative scores (real scorers not available)")
            
            alternative_scores = {
                'left_right_hypothesis': generate_mock_score('left_right'),
                'liberal_illiberal_hypothesis': generate_mock_score('liberal_illiberal'),
                'populism_pluralism_hypothesis': generate_mock_score('populism_pluralism')
            }
            logger.debug("Generated mock scores for all three dimensions")
            
    except Exception as e:
        logger.error(f"Alternative scoring error: {e}")
        # Fall back to all mock data on any error
        alternative_scores = {
            'left_right_hypothesis': generate_mock_score('left_right'),
            'liberal_illiberal_hypothesis': generate_mock_score('liberal_illiberal'),
            'populism_pluralism_hypothesis': generate_mock_score('populism_pluralism')
        }
        logger.warning("Fell back to mock scores due to error")
    
    return alternative_scores

def generate_mock_score(dimension_type):
    """Generate a single mock score for a dimension"""
    score = round(random.uniform(0, 10), 2)
    confidence = round(random.uniform(0.6, 0.95), 3)
    
    if dimension_type == 'left_right':
        if score < 3:
            interpretation = 'Strong Left'
        elif score < 4.5:
            interpretation = 'Left'
        elif score < 5.5:
            interpretation = 'Center'
        elif score < 7:
            interpretation = 'Right'
        else:
            interpretation = 'Strong Right'
    elif dimension_type == 'liberal_illiberal':
        if score < 3:
            interpretation = 'Strong Illiberal'
        elif score < 4.5:
            interpretation = 'Illiberal'
        elif score < 5.5:
            interpretation = 'Moderate'
        elif score < 7:
            interpretation = 'Liberal'
        else:
            interpretation = 'Strong Liberal'
    else:  # populism_pluralism
        if score < 3:
            interpretation = 'Strong Pluralist'
        elif score < 4.5:
            interpretation = 'Pluralist'
        elif score < 5.5:
            interpretation = 'Moderate'
        elif score < 7:
            interpretation = 'Populist'
        else:
            interpretation = 'Strong Populist'
    
    logger.debug(f"Generated mock {dimension_type} score: {score} ({interpretation})")
    
    return {
        'score': score,
        'confidence': confidence,
        'interpretation': interpretation
    }

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
            
            try:
                # Get inference service
                inference_service = PoliticalInferenceService.get_instance()
                
                # Run predictions
                results = inference_service.predict_all(text, None)
                coordinates = inference_service.get_3d_coordinates(text, None)
                
                # Generate alternative hypothesis scores
                logger.debug("Generating alternative hypothesis scores...")
                alternative_scorers = get_alternative_scorers()
                alternative_scores = generate_alternative_scores(text, alternative_scorers)
                
                context_data = {
                    'form': form,
                    'results': results,
                    'coordinates': coordinates,
                    'input_text': text,
                    'coordinates_json': json.dumps(coordinates),
                    'alternative_scores': alternative_scores
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
        
        if not text:
            return JsonResponse({'error': 'Text is required'}, status=400)
        
        if len(text) > 2000:
            return JsonResponse({'error': 'Text too long (max 2000 characters)'}, status=400)
        
        # Get inference service
        inference_service = PoliticalInferenceService.get_instance()
        
        # Run predictions
        results = inference_service.predict_all(text, None)
        coordinates = inference_service.get_3d_coordinates(text, None)
        
        # Generate alternative hypothesis scores
        logger.debug("Generating alternative hypothesis scores...")
        alternative_scorers = get_alternative_scorers()
        alternative_scores = generate_alternative_scores(text, alternative_scorers)
        
        return JsonResponse({
            'results': results,
            'coordinates': coordinates,
            'input_text': text,
            'alternative_scores': alternative_scores
        })
        
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        logger.error(f"API Classification error: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)
