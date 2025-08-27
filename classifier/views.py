# ============================================================================
# UPDATED VIEWS.PY INTEGRATION - Add this to your existing views.py
# ============================================================================

import json
import logging
import random
import numpy as np
import gc
import torch
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.contrib import messages
from django.conf import settings
from .forms import TextClassificationForm
from .inference_service import PoliticalInferenceService

logger = logging.getLogger(__name__)

def cleanup_memory():
    """Clean up memory after model operations"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_alternative_scorers():
    """Initialize alternative hypothesis scorers - UPDATED to include new response-based models"""
    from .inference_service import MOCK_MODE  # Import at runtime to get current value
    
    if MOCK_MODE:
        logger.info("🚧 Alternative approaches running in MOCK MODE (controlled by inference_service.MOCK_MODE)")
        return None
    
    try:
        # Existing hypothesis-based scorers
        from . import alternative
        from . import alternativeLib  
        from . import alternativePop
        
        # NEW: Response-based scorers (adapted versions of your new scripts)
        from . import libillibwresponses as lib_responses
        from . import rilewresponses as lr_responses  
        from . import popnonpopwresponses as pop_responses
        
        return {
            # Existing hypothesis-based models
            'left_right': alternative.LeftRightEconomicScorer(),
            'liberal_illiberal': alternativeLib.LiberalIlliberalScorer(),
            'populism_pluralism': alternativePop.PopulismPluralismScorer(),
            
            # NEW: Response-based models (your three new scripts)
            'left_right_responses': lr_responses.LeftRightResponsesScorer(),
            'liberal_illiberal_responses': lib_responses.LiberalIlliberalResponsesScorer(),
            'populism_pluralism_responses': pop_responses.PopulismPluralismResponsesScorer()
        }
    except Exception as e:
        logger.warning(f"Could not load alternative scorers (will use mock data): {e}")
        return None

def generate_alternative_scores(text, scorers=None):
    """Generate alternative hypothesis scores using real models or mock data"""
    from .inference_service import MOCK_MODE
    
    if not getattr(settings, 'ENABLE_ALTERNATIVE_SCORES', False):
        logger.debug("Alternative scores disabled in settings")
        return None
    
    alternative_scores = {}
    
    try:
        if scorers and not MOCK_MODE:
            # Real model predictions
            logger.info("🔬 Using real alternative hypothesis models (including new response-based models)")
            
            # Left-Right Hypothesis Scoring
            if 'left_right' in scorers:
                try:
                    logger.debug("Running left-right hypothesis scoring...")
                    lr_result = scorers['left_right'].score_left_right(text)
                    alternative_scores['left_right_hypothesis'] = {
                        'score': round(lr_result.get('score', 5.0), 2),
                        'confidence': round(lr_result.get('confidence', 0.8) * 100, 1),  # Convert to percentage
                        'interpretation': lr_result.get('interpretation', 'Center')
                    }
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
                        'confidence': round(li_result.get('confidence', 0.8) * 100, 1),  # Convert to percentage
                        'interpretation': li_result.get('interpretation', 'Moderate')
                    }
                except Exception as e:
                    logger.error(f"Liberal-Illiberal alternative scoring failed: {e}")
                    alternative_scores['liberal_illiberal_hypothesis'] = generate_mock_score('liberal_illiberal')
            
            # Continue for all other approaches...
            # Populism-Pluralism Hypothesis Scoring
            if 'populism_pluralism' in scorers:
                try:
                    logger.debug("Running populism-pluralism hypothesis scoring...")
                    pp_result = scorers['populism_pluralism'].score_populism_pluralism(text)
                    alternative_scores['populism_pluralism_hypothesis'] = {
                        'score': round(pp_result.get('score', 5.0), 2),
                        'confidence': round(pp_result.get('confidence', 0.8) * 100, 1),  # Convert to percentage
                        'interpretation': pp_result.get('interpretation', 'Moderate')
                    }
                except Exception as e:
                    logger.error(f"Populism-Pluralism alternative scoring failed: {e}")
                    alternative_scores['populism_pluralism_hypothesis'] = generate_mock_score('populism_pluralism')
            
            # NEW RESPONSE-BASED MODELS - also convert confidence to percentage
            if 'left_right_responses' in scorers:
                try:
                    logger.debug("Running left-right response-based scoring...")
                    lr_resp_result = scorers['left_right_responses'].score_left_right(text)
                    alternative_scores['left_right_responses'] = {
                        'score': round(lr_resp_result.get('score', 5.0), 2),
                        'confidence': round(lr_resp_result.get('confidence', 0.8) * 100, 1),  # Convert to percentage
                        'interpretation': lr_resp_result.get('interpretation', 'Center')
                    }
                except Exception as e:
                    logger.error(f"Left-Right response-based scoring failed: {e}")
                    alternative_scores['left_right_responses'] = generate_mock_score('left_right')
            
            if 'liberal_illiberal_responses' in scorers:
                try:
                    logger.debug("Running liberal-illiberal response-based scoring...")
                    li_resp_result = scorers['liberal_illiberal_responses'].score_liberal_illiberal(text)
                    alternative_scores['liberal_illiberal_responses'] = {
                        'score': round(li_resp_result.get('score', 5.0), 2),
                        'confidence': round(li_resp_result.get('confidence', 0.8) * 100, 1),  # Convert to percentage
                        'interpretation': li_resp_result.get('interpretation', 'Moderate')
                    }
                except Exception as e:
                    logger.error(f"Liberal-Illiberal response-based scoring failed: {e}")
                    alternative_scores['liberal_illiberal_responses'] = generate_mock_score('liberal_illiberal')
            
            if 'populism_pluralism_responses' in scorers:
                try:
                    logger.debug("Running populism-pluralism response-based scoring...")
                    pp_resp_result = scorers['populism_pluralism_responses'].score_populism_pluralism(text)
                    alternative_scores['populism_pluralism_responses'] = {
                        'score': round(pp_resp_result.get('score', 5.0), 2),
                        'confidence': round(pp_resp_result.get('confidence', 0.8) * 100, 1),  # Convert to percentage
                        'interpretation': pp_resp_result.get('interpretation', 'Moderate')
                    }
                except Exception as e:
                    logger.error(f"Populism-Pluralism response-based scoring failed: {e}")
                    alternative_scores['populism_pluralism_responses'] = generate_mock_score('populism_pluralism')
        
        else:
            # Use mock data - also provide confidence as percentage
            alternative_scores = {
                'left_right_hypothesis': generate_mock_score('left_right'),
                'liberal_illiberal_hypothesis': generate_mock_score('liberal_illiberal'),
                'populism_pluralism_hypothesis': generate_mock_score('populism_pluralism'),
                'left_right_responses': generate_mock_score('left_right'),
                'liberal_illiberal_responses': generate_mock_score('liberal_illiberal'),
                'populism_pluralism_responses': generate_mock_score('populism_pluralism')
            }
            
    except Exception as e:
        logger.error(f"Alternative scoring error: {e}")
        alternative_scores = {
            'left_right_hypothesis': generate_mock_score('left_right'),
            'liberal_illiberal_hypothesis': generate_mock_score('liberal_illiberal'),
            'populism_pluralism_hypothesis': generate_mock_score('populism_pluralism'),
            'left_right_responses': generate_mock_score('left_right'),
            'liberal_illiberal_responses': generate_mock_score('liberal_illiberal'),
            'populism_pluralism_responses': generate_mock_score('populism_pluralism')
        }
    
    return alternative_scores

def generate_mock_score(dimension_type):
    """Generate a single mock score for a dimension - UPDATED to provide confidence as percentage"""
    score = round(random.uniform(0, 10), 2)
    confidence = round(random.uniform(60, 95), 1)  # Already as percentage (60-95%)
    
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
    
    return {
        'score': score,
        'confidence': confidence,  # Now already as percentage
        'interpretation': interpretation
    }

# Keep existing view functions unchanged - they will automatically work with the new scorers
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
                
                # Generate alternative hypothesis scores (now includes new models)
                logger.debug("Generating alternative hypothesis scores...")
                alternative_scorers = get_alternative_scorers()
                alternative_scores = generate_alternative_scores(text, alternative_scorers)

                cleanup_memory()
		# UPDATED: Extract selected approaches from form - ADD the new response fields
                selected_approaches = {
                    # Direct regression approaches
                    'left_right_direct': form.cleaned_data.get('left_right_direct', False),
                    'liberal_illiberal_direct': form.cleaned_data.get('liberal_illiberal_direct', False),
                    'populism_direct': form.cleaned_data.get('populism_direct', False),
                    
                    # Hypothesis-based approaches
                    'left_right_hypothesis': form.cleaned_data.get('left_right_hypothesis', False),
                    'liberal_illiberal_hypothesis': form.cleaned_data.get('liberal_illiberal_hypothesis', False),
                    'populism_hypothesis': form.cleaned_data.get('populism_hypothesis', False),
                    
                    # NEW: Response-based approaches - ADD THESE
                    'left_right_responses': form.cleaned_data.get('left_right_responses', False),
                    'liberal_illiberal_responses': form.cleaned_data.get('liberal_illiberal_responses', False),
                    'populism_responses': form.cleaned_data.get('populism_responses', False),
                }
                
                context_data = {
                    'form': form,
                    'results': results,
                    'coordinates': coordinates,
                    'input_text': text,
                    'coordinates_json': json.dumps(coordinates),
                    'alternative_scores': alternative_scores,
                    'selected_approaches': selected_approaches  # Pass selected approaches to template
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

# Also update the API endpoint to handle new approaches
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
        
        # Generate alternative hypothesis scores (now includes new models)
        logger.debug("Generating alternative hypothesis scores...")
        alternative_scorers = get_alternative_scorers()
        alternative_scores = generate_alternative_scores(text, alternative_scorers)
        
        # For API, return all approaches by default (can be made configurable later)
        selected_approaches = {
            'left_right_direct': True,
            'liberal_illiberal_direct': True,
            'populism_direct': True,
            'left_right_hypothesis': True,
            'liberal_illiberal_hypothesis': True,
            'populism_hypothesis': True,
            'left_right_responses': True,
            'liberal_illiberal_responses': True,
            'populism_responses': True,
        }
        
        return JsonResponse({
            'results': results,
            'coordinates': coordinates,
            'input_text': text,
            'alternative_scores': alternative_scores,
            'selected_approaches': selected_approaches
        })
        
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        logger.error(f"API Classification error: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)
