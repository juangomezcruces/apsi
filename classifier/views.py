import json
import logging
import random
import numpy as np
import gc
import torch
import psutil
import time
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.contrib import messages
from django.conf import settings
from .forms import TextClassificationForm
from .inference_service import PoliticalInferenceService

logger = logging.getLogger(__name__)

def log_memory_usage(context=""):
    """Log current memory usage for debugging"""
    if context:
        context = f" ({context})"
    
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        # System memory
        system_memory = psutil.virtual_memory()
        system_available_gb = system_memory.available / 1024 / 1024 / 1024
        system_used_percent = system_memory.percent
        
        logger.info(f"Memory{context}: Process={memory_mb:.1f}MB, System={system_used_percent:.1f}% used, {system_available_gb:.1f}GB available")
        
        # GPU memory if available
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
            gpu_cached = torch.cuda.memory_reserved() / 1024 / 1024
            logger.info(f"GPU Memory{context}: Allocated={gpu_memory:.1f}MB, Cached={gpu_cached:.1f}MB")
            
    except Exception as e:
        logger.warning(f"Could not log memory usage{context}: {e}")

def cleanup_memory():
    """Clean up memory after model operations"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_alternative_scorers(selected_approaches):
    """Initialize ONLY the selected alternative hypothesis scorers"""
    from .inference_service import MOCK_MODE
    
    if MOCK_MODE:
        logger.info("Alternative approaches running in MOCK MODE")
        return None
    
    # Only load scorers that are actually selected
    scorers = {}
    
    try:
        # Load hypothesis-based scorers only if selected
        if (selected_approaches.get('left_right_hypothesis') or 
            selected_approaches.get('liberal_illiberal_hypothesis') or 
            selected_approaches.get('populism_hypothesis')):
            
            logger.info("Loading hypothesis-based scorers...")
            from . import alternative
            from . import alternativeLib  
            from . import alternativePop
            
            if selected_approaches.get('left_right_hypothesis'):
                scorers['left_right'] = alternative.LeftRightEconomicScorer()
                logger.info("âœ“ Left-Right hypothesis scorer loaded")
                
            if selected_approaches.get('liberal_illiberal_hypothesis'):
                scorers['liberal_illiberal'] = alternativeLib.LiberalIlliberalScorer()
                logger.info("âœ“ Liberal-Illiberal hypothesis scorer loaded")
                
            if selected_approaches.get('populism_hypothesis'):
                scorers['populism_pluralism'] = alternativePop.PopulismPluralismScorer()
                logger.info("âœ“ Populism-Pluralism hypothesis scorer loaded")
        
        # Load response-based scorers only if selected
        if (selected_approaches.get('left_right_responses') or 
            selected_approaches.get('liberal_illiberal_responses') or 
            selected_approaches.get('populism_responses')):
            
            logger.info("Loading response-based scorers...")
            from . import libillibwresponses as lib_responses
            from . import rilewresponses as lr_responses  
            from . import popnonpopwresponses as pop_responses
            
            if selected_approaches.get('left_right_responses'):
                scorers['left_right_responses'] = lr_responses.LeftRightResponsesScorer()
                logger.info("âœ“ Left-Right responses scorer loaded")
                
            if selected_approaches.get('liberal_illiberal_responses'):
                scorers['liberal_illiberal_responses'] = lib_responses.LiberalIlliberalResponsesScorer()
                logger.info("âœ“ Liberal-Illiberal responses scorer loaded")
                
            if selected_approaches.get('populism_responses'):
                scorers['populism_pluralism_responses'] = pop_responses.PopulismPluralismResponsesScorer()
                logger.info("âœ“ Populism-Pluralism responses scorer loaded")
        
        log_memory_usage("after loading scorers")
        return scorers
        
    except Exception as e:
        logger.error(f"Error loading alternative scorers: {e}")
        return None

def generate_alternative_scores(text, scorers=None, selected_approaches=None):
    """Generate ONLY the selected alternative hypothesis scores"""
    from .inference_service import MOCK_MODE
    
    if not getattr(settings, 'ENABLE_ALTERNATIVE_SCORES', False):
        logger.debug("Alternative scores disabled in settings")
        return None
    
    if not selected_approaches:
        logger.debug("No alternative approaches selected")
        return None
    
    alternative_scores = {}
    
    try:
        if scorers and not MOCK_MODE:
            logger.info(f"Running selected alternative approaches: {list(selected_approaches.keys())}")
            
            # === HYPOTHESIS-BASED MODELS (only if selected) ===
            
            if selected_approaches.get('left_right_hypothesis') and 'left_right' in scorers:
                try:
                    logger.debug("Running left-right hypothesis scoring...")
                    lr_result = scorers['left_right'].score_left_right(text)
                    alternative_scores['left_right_hypothesis'] = {
                        'score': round(lr_result.get('score', 5.0), 2),
                        'confidence': round(lr_result.get('confidence', 0.8) * 100, 1),
                        'interpretation': lr_result.get('interpretation', 'Center')
                    }
                    logger.debug(f"âœ“ Left-right hypothesis: {lr_result.get('score', 'N/A'):.2f}")
                except Exception as e:
                    logger.error(f"Left-Right hypothesis scoring failed: {e}")
                    alternative_scores['left_right_hypothesis'] = generate_mock_score('left_right')
            
            if selected_approaches.get('liberal_illiberal_hypothesis') and 'liberal_illiberal' in scorers:
                try:
                    logger.debug("Running liberal-illiberal hypothesis scoring...")
                    li_result = scorers['liberal_illiberal'].score_liberal_illiberal(text)
                    alternative_scores['liberal_illiberal_hypothesis'] = {
                        'score': round(li_result.get('score', 5.0), 2),
                        'confidence': round(li_result.get('confidence', 0.8) * 100, 1),
                        'interpretation': li_result.get('interpretation', 'Moderate')
                    }
                    logger.debug(f"âœ“ Liberal-illiberal hypothesis: {li_result.get('score', 'N/A'):.2f}")
                except Exception as e:
                    logger.error(f"Liberal-Illiberal hypothesis scoring failed: {e}")
                    alternative_scores['liberal_illiberal_hypothesis'] = generate_mock_score('liberal_illiberal')
            
            if selected_approaches.get('populism_hypothesis') and 'populism_pluralism' in scorers:
                try:
                    logger.debug("Running populism-pluralism hypothesis scoring...")
                    pp_result = scorers['populism_pluralism'].score_populism_pluralism(text)
                    alternative_scores['populism_pluralism_hypothesis'] = {
                        'score': round(pp_result.get('score', 5.0), 2),
                        'confidence': round(pp_result.get('confidence', 0.8) * 100, 1),
                        'interpretation': pp_result.get('interpretation', 'Moderate')
                    }
                    logger.debug(f"âœ“ Populism-pluralism hypothesis: {pp_result.get('score', 'N/A'):.2f}")
                except Exception as e:
                    logger.error(f"Populism-Pluralism hypothesis scoring failed: {e}")
                    alternative_scores['populism_pluralism_hypothesis'] = generate_mock_score('populism_pluralism')
            
            # === RESPONSE-BASED MODELS (only if selected) ===
            
            if selected_approaches.get('left_right_responses') and 'left_right_responses' in scorers:
                try:
                    logger.debug("Running left-right response-based scoring...")
                    lr_resp_result = scorers['left_right_responses'].score_left_right(text)
                    alternative_scores['left_right_responses'] = {
                        'score': round(lr_resp_result.get('score', 5.0), 2),
                        'confidence': round(lr_resp_result.get('confidence', 0.8) * 100, 1),
                        'interpretation': lr_resp_result.get('interpretation', 'Center')
                    }
                    logger.debug(f"âœ“ Left-right responses: {lr_resp_result.get('score', 'N/A'):.2f}")
                except Exception as e:
                    logger.error(f"Left-Right response-based scoring failed: {e}")
                    alternative_scores['left_right_responses'] = generate_mock_score('left_right')
            
            if selected_approaches.get('liberal_illiberal_responses') and 'liberal_illiberal_responses' in scorers:
                try:
                    logger.debug("Running liberal-illiberal response-based scoring...")
                    li_resp_result = scorers['liberal_illiberal_responses'].score_liberal_illiberal(text)
                    alternative_scores['liberal_illiberal_responses'] = {
                        'score': round(li_resp_result.get('score', 5.0), 2),
                        'confidence': round(li_resp_result.get('confidence', 0.8) * 100, 1),
                        'interpretation': li_resp_result.get('interpretation', 'Moderate')
                    }
                    logger.debug(f"âœ“ Liberal-illiberal responses: {li_resp_result.get('score', 'N/A'):.2f}")
                except Exception as e:
                    logger.error(f"Liberal-Illiberal response-based scoring failed: {e}")
                    alternative_scores['liberal_illiberal_responses'] = generate_mock_score('liberal_illiberal')
            
            if selected_approaches.get('populism_responses') and 'populism_pluralism_responses' in scorers:
                try:
                    logger.debug("Running populism-pluralism response-based scoring...")
                    pp_resp_result = scorers['populism_pluralism_responses'].score_populism_pluralism(text)
                    alternative_scores['populism_pluralism_responses'] = {
                        'score': round(pp_resp_result.get('score', 5.0), 2),
                        'confidence': round(pp_resp_result.get('confidence', 0.8) * 100, 1),
                        'interpretation': pp_resp_result.get('interpretation', 'Moderate')
                    }
                    logger.debug(f"âœ“ Populism-pluralism responses: {pp_resp_result.get('score', 'N/A'):.2f}")
                except Exception as e:
                    logger.error(f"Populism-Pluralism response-based scoring failed: {e}")
                    alternative_scores['populism_pluralism_responses'] = generate_mock_score('populism_pluralism')
        else:
            # Generate mock data only for selected approaches
            if MOCK_MODE:
                logger.info("Using mock alternative scores (MOCK_MODE = True)")
            else:
                logger.warning("Using mock alternative scores (real scorers not available)")
            
            if selected_approaches.get('left_right_hypothesis'):
                alternative_scores['left_right_hypothesis'] = generate_mock_score('left_right')
            if selected_approaches.get('liberal_illiberal_hypothesis'):
                alternative_scores['liberal_illiberal_hypothesis'] = generate_mock_score('liberal_illiberal')
            if selected_approaches.get('populism_hypothesis'):
                alternative_scores['populism_pluralism_hypothesis'] = generate_mock_score('populism_pluralism')
            if selected_approaches.get('left_right_responses'):
                alternative_scores['left_right_responses'] = generate_mock_score('left_right')
            if selected_approaches.get('liberal_illiberal_responses'):
                alternative_scores['liberal_illiberal_responses'] = generate_mock_score('liberal_illiberal')
            if selected_approaches.get('populism_responses'):
                alternative_scores['populism_pluralism_responses'] = generate_mock_score('populism_pluralism')
            
            logger.debug(f"Generated mock scores for selected approaches: {list(alternative_scores.keys())}")
            
    except Exception as e:
        logger.error(f"Alternative scoring error: {e}")
        # Emergency fallback - only mock the selected approaches
        alternative_scores = {}
        for approach, selected in selected_approaches.items():
            if selected:
                if 'left_right' in approach:
                    alternative_scores[approach] = generate_mock_score('left_right')
                elif 'liberal_illiberal' in approach:
                    alternative_scores[approach] = generate_mock_score('liberal_illiberal')
                elif 'populism' in approach:
                    alternative_scores[approach] = generate_mock_score('populism_pluralism')
    
    return alternative_scores

def generate_mock_score(dimension_type):
    """Generate a single mock score for a dimension"""
    score = round(random.uniform(0, 10), 2)
    confidence = round(random.uniform(60, 95), 1)  # Already as percentage
    
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
        'confidence': confidence,
        'interpretation': interpretation
    }

def index(request):
    """Main page with classification form."""
    form = TextClassificationForm()
    return render(request, 'classifier/index.html', {'form': form})

def classify_text(request):
    """Handle form submission and show results - FIXED approach selection logic"""
    start_time = time.time()
    log_memory_usage("start of request")
    
    if request.method == 'POST':
        form = TextClassificationForm(request.POST)
        if form.is_valid():
            text = form.cleaned_data['text']
            
            # Extract selected approaches FIRST
            selected_approaches = {
                # Direct regression approaches
                'left_right_direct': form.cleaned_data.get('left_right_direct', False),
                'liberal_illiberal_direct': form.cleaned_data.get('liberal_illiberal_direct', False),
                'populism_direct': form.cleaned_data.get('populism_direct', False),
                
                # Hypothesis-based approaches
                'left_right_hypothesis': form.cleaned_data.get('left_right_hypothesis', False),
                'liberal_illiberal_hypothesis': form.cleaned_data.get('liberal_illiberal_hypothesis', False),
                'populism_hypothesis': form.cleaned_data.get('populism_hypothesis', False),
                
                # Response-based approaches
                'left_right_responses': form.cleaned_data.get('left_right_responses', False),
                'liberal_illiberal_responses': form.cleaned_data.get('liberal_illiberal_responses', False),
                'populism_responses': form.cleaned_data.get('populism_responses', False),
            }
            
            logger.info(f"Selected approaches: {[k for k, v in selected_approaches.items() if v]}")
            
            try:
                results = {}
                coordinates = {'x': None, 'y': None, 'z': None, 'labels': {}, 'errors': []}
                
                # === POLITICAL PRE-CHECK (runs for ALL approaches) ===
                inference_service = PoliticalInferenceService.get_instance()
                political_check_result = inference_service.is_political_text(text)
                
                # If text is not political, show warning and stop
                if not political_check_result['is_political']:
                    logger.info(f"Text rejected as non-political: {political_check_result}")
                    context_data = {
                        'form': form,
                        'input_text': text,
                        'not_political': True,
                        'political_check': political_check_result,
                        'message': 'Text does not appear to be about political topics. Scoring skipped.',
                        'selected_approaches': selected_approaches,
                    }
                    return render(request, 'classifier/results.html', context_data)
                
                logger.info(f"Text passed political check: {political_check_result}")
                # === END POLITICAL PRE-CHECK ===
                
                # Check if any direct regression is selected
                direct_selected = (
                    selected_approaches.get('left_right_direct') or 
                    selected_approaches.get('liberal_illiberal_direct') or 
                    selected_approaches.get('populism_direct')
                )
                
                if direct_selected:
                    logger.info("Running direct regression models...")
                    # Skip pre-check since we already did it above
                    results = inference_service.predict_all(text, None, skip_precheck=True)
                    coordinates = inference_service.get_3d_coordinates(text, None, skip_precheck=True)
                    log_memory_usage("after direct regression")
                else:
                    logger.info("Skipping direct regression models (not selected)")
                
                # Only run alternative approaches if any are selected
                alternative_scores = None
                alternative_approaches_selected = any([
                    selected_approaches.get('left_right_hypothesis'),
                    selected_approaches.get('liberal_illiberal_hypothesis'),
                    selected_approaches.get('populism_hypothesis'),
                    selected_approaches.get('left_right_responses'),
                    selected_approaches.get('liberal_illiberal_responses'),
                    selected_approaches.get('populism_responses'),
                ])
                
                if alternative_approaches_selected:
                    logger.info("Loading and running selected alternative approaches...")
                    alternative_scorers = get_alternative_scorers(selected_approaches)
                    alternative_scores = generate_alternative_scores(text, alternative_scorers, selected_approaches)
                    log_memory_usage("after alternative approaches")
                else:
                    logger.info("Skipping alternative approaches (none selected)")
                
                # Clean up memory after heavy operations
                cleanup_memory()
                log_memory_usage("after cleanup")
                
                context_data = {
                    'form': form,
                    'results': results,
                    'coordinates': coordinates,
                    'input_text': text,
                    'coordinates_json': json.dumps(coordinates),
                    'alternative_scores': alternative_scores,
                    'political_check': results.get('political_check'),
                    'selected_approaches': selected_approaches
                }
                
                processing_time = time.time() - start_time
                logger.info(f"Request completed in {processing_time:.2f} seconds")
                
                return render(request, 'classifier/results.html', context_data)
                
            except Exception as e:
                processing_time = time.time() - start_time
                log_memory_usage("after error")
                logger.error(f"Classification error after {processing_time:.2f}s: {str(e)}")
                logger.error(f"Error type: {type(e).__name__}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                
                messages.error(request, f"An error occurred during classification: {str(e)}")
                return render(request, 'classifier/index.html', {'form': form})
        else:
            logger.warning(f"Form validation failed: {form.errors}")
            return render(request, 'classifier/index.html', {'form': form})
    else:
        return index(request)


def privacy_notice(request):
    return render(request, 'classifier/privacynotice.html')


@csrf_exempt
@require_http_methods(["POST"])
def api_classify(request):
    """API endpoint for classification with enhanced debugging"""
    start_time = time.time()
    log_memory_usage("API start")
    
    try:
        data = json.loads(request.body)
        text = data.get('text', '').strip()
        skip_precheck = data.get('skip_precheck', False)  # NEW: allow skipping
        
        if not text:
            return JsonResponse({'error': 'Text is required'}, status=400)
        
        if len(text) > 2000:
            return JsonResponse({'error': 'Text too long (max 2000 characters)'}, status=400)
        
        # For API, use all approaches by default (can be made configurable)
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
        
        # Get inference service
        inference_service = PoliticalInferenceService.get_instance()
        
        # Run predictions
        results = inference_service.predict_all(text, None, skip_precheck=skip_precheck)

        # === NEW: Handle non-political text ===
        if results.get('not_political'):
            return JsonResponse({
                'not_political': True,
                'political_check': results.get('political_check'),
                'message': results.get('message'),
                'input_text': text
            })
        # === END NEW ===

        coordinates = inference_service.get_3d_coordinates(text, None)
        
        # Generate alternative scores
        alternative_scorers = get_alternative_scorers(selected_approaches)
        alternative_scores = generate_alternative_scores(text, alternative_scorers, selected_approaches)
        
        cleanup_memory()
        processing_time = time.time() - start_time
        logger.info(f"API request completed in {processing_time:.2f} seconds")
        
        return JsonResponse({
            'results': results,
            'coordinates': coordinates,
            'input_text': text,
            'alternative_scores': alternative_scores,
            'selected_approaches': selected_approaches,
            'processing_time': round(processing_time, 2)
        })
        
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        processing_time = time.time() - start_time
        log_memory_usage("API error")
        logger.error(f"API Classification error after {processing_time:.2f}s: {str(e)}")
        import traceback
        logger.error(f"API Full traceback: {traceback.format_exc()}")
        return JsonResponse({'error': str(e)}, status=500)
