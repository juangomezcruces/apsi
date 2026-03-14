import json
import logging
import random
import numpy as np
import gc
import torch
import psutil
import time
from django.shortcuts import render, redirect
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
                logger.info("✓ Left-Right hypothesis scorer loaded")

            if selected_approaches.get('liberal_illiberal_hypothesis'):
                scorers['liberal_illiberal'] = alternativeLib.LiberalIlliberalScorer()
                logger.info("✓ Liberal-Illiberal hypothesis scorer loaded")

            if selected_approaches.get('populism_hypothesis'):
                scorers['populism_pluralism'] = alternativePop.PopulismPluralismScorer()
                logger.info("✓ Populism-Pluralism hypothesis scorer loaded")

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
            from concurrent.futures import ThreadPoolExecutor, as_completed

            selected_list = [k for k, v in selected_approaches.items() if v]
            logger.info(f"Running selected alternative approaches in parallel: {selected_list}")

            # === SCORER FUNCTIONS (one per dimension) ===

            def run_left_right():
                try:
                    logger.debug("Running left-right hypothesis scoring...")
                    lr_result = scorers['left_right'].score_left_right(text)
                    is_relevant = lr_result.get('is_relevant', True)
                    score = lr_result.get('score', 5.0)
                    if is_relevant and score != 'NA':
                        result = {
                            'score': round(score, 2),
                            'confidence': round(lr_result.get('confidence', 0.8) * 100, 1),
                            'interpretation': lr_result.get('interpretation', 'Center'),
                            'is_relevant': True,
                            'topic_probability': round(lr_result.get('topic_probability', 1.0), 3),
                            'top_left_hypotheses': lr_result.get('top_left_hypotheses', []),
                            'top_right_hypotheses': lr_result.get('top_right_hypotheses', []),
                            'contradiction_detected': lr_result.get('contradiction_detected', False),
                        }
                        logger.debug(f"✓ Left-right hypothesis: {score:.2f}")
                    else:
                        result = {
                            'score': 'NA',
                            'confidence': 0.0,
                            'interpretation': lr_result.get('interpretation', 'Not about economic policy'),
                            'is_relevant': False,
                            'topic_probability': round(lr_result.get('topic_probability', 0.0), 3),
                        }
                        logger.debug("✓ Left-right hypothesis: NA (not relevant)")
                    return 'left_right_hypothesis', result
                except Exception as e:
                    logger.error(f"Left-Right hypothesis scoring failed: {e}")
                    return 'left_right_hypothesis', generate_mock_score('left_right')

            def run_liberal_illiberal():
                try:
                    logger.debug("Running liberal-illiberal hypothesis scoring...")
                    li_result = scorers['liberal_illiberal'].score_liberal_illiberal(text)
                    is_relevant = li_result.get('is_relevant', True)
                    score = li_result.get('score', 5.0)
                    if is_relevant and score != 'NA':
                        result = {
                            'score': round(score, 2),
                            'confidence': round(li_result.get('confidence', 0.8) * 100, 1),
                            'interpretation': li_result.get('interpretation', 'Moderate'),
                            'is_relevant': True,
                            'topic_probability': round(li_result.get('topic_probability', 1.0), 3),
                            'top_liberal_hypotheses': li_result.get('top_liberal_hypotheses', []),
                            'top_illiberal_hypotheses': li_result.get('top_illiberal_hypotheses', []),
                            'contradiction_detected': li_result.get('contradiction_detected', False),
                        }
                        logger.debug(f"✓ Liberal-illiberal hypothesis: {score:.2f}")
                    else:
                        result = {
                            'score': 'NA',
                            'confidence': 0.0,
                            'interpretation': li_result.get('interpretation', 'Not about democratic principles'),
                            'is_relevant': False,
                            'topic_probability': round(li_result.get('topic_probability', 0.0), 3),
                        }
                        logger.debug("✓ Liberal-illiberal hypothesis: NA (not relevant)")
                    return 'liberal_illiberal_hypothesis', result
                except Exception as e:
                    logger.error(f"Liberal-Illiberal hypothesis scoring failed: {e}")
                    return 'liberal_illiberal_hypothesis', generate_mock_score('liberal_illiberal')

            def run_populism():
                try:
                    logger.debug("Running populism-pluralism hypothesis scoring...")
                    pp_result = scorers['populism_pluralism'].score_populism_pluralism(text)
                    is_relevant = pp_result.get('is_relevant', True)
                    score = pp_result.get('score', 5.0)
                    if is_relevant and score != 'NA':
                        result = {
                            'score': round(score, 2),
                            'confidence': round(pp_result.get('confidence', 0.8) * 100, 1),
                            'interpretation': pp_result.get('interpretation', 'Moderate'),
                            'is_relevant': True,
                            'topic_probability': round(pp_result.get('topic_probability', 1.0), 3),
                            'top_populism_hypotheses': pp_result.get('top_populist_hypotheses', []),
                            'top_pluralism_hypotheses': pp_result.get('top_pluralist_hypotheses', []),
                            'contradiction_detected': pp_result.get('contradiction_detected', False),
                        }
                        logger.debug(f"✓ Populism-pluralism hypothesis: {score:.2f}")
                    else:
                        result = {
                            'score': 'NA',
                            'confidence': 0.0,
                            'interpretation': pp_result.get('interpretation', 'Not about political rhetoric'),
                            'is_relevant': False,
                            'topic_probability': round(pp_result.get('topic_probability', 0.0), 3),
                            'top_populism_hypotheses': pp_result.get('top_populism_hypotheses', []),
                            'top_pluralism_hypotheses': pp_result.get('top_pluralism_hypotheses', []),
                        }
                        logger.debug("✓ Populism-pluralism hypothesis: NA (not relevant)")
                    return 'populism_pluralism_hypothesis', result
                except Exception as e:
                    logger.error(f"Populism-Pluralism hypothesis scoring failed: {e}")
                    return 'populism_pluralism_hypothesis', generate_mock_score('populism_pluralism')

            # Build list of tasks to run based on what was selected
            tasks = []
            if selected_approaches.get('left_right_hypothesis') and 'left_right' in scorers:
                tasks.append(run_left_right)
            if selected_approaches.get('liberal_illiberal_hypothesis') and 'liberal_illiberal' in scorers:
                tasks.append(run_liberal_illiberal)
            if selected_approaches.get('populism_hypothesis') and 'populism_pluralism' in scorers:
                tasks.append(run_populism)

            # Run all selected scorers in parallel
            with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
                futures = [executor.submit(task) for task in tasks]
                for future in as_completed(futures):
                    key, result = future.result()
                    alternative_scores[key] = result

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
    """Landing page."""
    return render(request, 'classifier/index.html')

def analysis(request):
    """Analysis page with form."""
    form = TextClassificationForm()
    return render(request, 'classifier/analysis.html', {'form': form})



def classify_text(request):
    """Handle form submission and show results"""
    start_time = time.time()
    log_memory_usage("start of request")

    if request.method != 'POST':
        return analysis(request)

    form = TextClassificationForm(request.POST)
    if not form.is_valid():
        logger.warning(f"Form validation failed: {form.errors}")
        return render(request, 'classifier/analysis.html', {'form': form})

    text = form.cleaned_data['text']

    selected_approaches = {
        'left_right_hypothesis': form.cleaned_data.get('left_right_hypothesis', False),
        'liberal_illiberal_hypothesis': form.cleaned_data.get('liberal_illiberal_hypothesis', False),
        'populism_hypothesis': form.cleaned_data.get('populism_hypothesis', False),
    }

    logger.info(f"Selected approaches: {[k for k, v in selected_approaches.items() if v]}")

    try:
        results = {}
        coordinates = {'x': None, 'y': None, 'z': None, 'labels': {}, 'errors': []}

        alternative_scores = None

        hypothesis_approaches_selected = any([
            selected_approaches.get('left_right_hypothesis'),
            selected_approaches.get('liberal_illiberal_hypothesis'),
            selected_approaches.get('populism_hypothesis'),
        ])

        if hypothesis_approaches_selected:
            logger.info("Loading and running selected hypothesis approaches...")
            alternative_scorers = get_alternative_scorers(selected_approaches)
            alternative_scores = generate_alternative_scores(text, alternative_scorers, selected_approaches)
            log_memory_usage("after hypothesis approaches")
        else:
            logger.info("No hypothesis approaches selected")

        cleanup_memory()
        log_memory_usage("after cleanup")

        why_these_results = {}

        def _fmt_items_pair(side_a_items, side_b_items, limit=5, min_prob_pct=15):
            """
            Process both sides of a dimension together so percentages are relative
            to the total impact across both sides combined.
            Each hypothesis's share = its score_impact / sum(all impacts, both sides) * 100.
            Bar width and hue follow that percentage.
            """
            def _eligible(items):
                out = []
                for it in (items or [])[:limit]:
                    hyp = it.get("hypothesis") or it.get("text") or ""
                    p = it.get("probability", 0.0)
                    impact = it.get("score_impact", 0.0)
                    try:
                        p_pct = float(p) * 100.0
                        impact = float(impact)
                    except Exception:
                        p_pct, impact = 0.0, 0.0
                    if hyp and round(p_pct, 0) >= min_prob_pct:
                        out.append({"text": hyp, "impact": impact})
                return out

            a = _eligible(side_a_items)
            b = _eligible(side_b_items)
            total = sum(x["impact"] for x in a + b) or 1.0

            def _enrich(items):
                out = []
                for it in items:
                    pct = round(it["impact"] / total * 100, 1)
                    hue = round(min(pct * 1.2, 120))  # 0% → red, 100% → green
                    out.append({"text": it["text"], "pct": pct, "hue": hue})
                return out

            return _enrich(a), _enrich(b)

        if alternative_scores:
            # Economic Left–Right
            lr = alternative_scores.get("left_right_hypothesis")
            if lr and lr.get("is_relevant") is not False:
                left_items, right_items = _fmt_items_pair(
                    lr.get("top_left_hypotheses", []),
                    lr.get("top_right_hypotheses", []),
                )
                if left_items or right_items:
                    why_these_results["Economic Left–Right"] = {
                        "Left": left_items,
                        "Right": right_items,
                    }

            # Support for Liberal Democracy
            li = alternative_scores.get("liberal_illiberal_hypothesis")
            if li and li.get("is_relevant") is not False:
                liberal_items, illiberal_items = _fmt_items_pair(
                    li.get("top_liberal_hypotheses", []),
                    li.get("top_illiberal_hypotheses", []),
                )
                if liberal_items or illiberal_items:
                    why_these_results["Support for Liberal Democracy"] = {
                        "Liberal": liberal_items,
                        "Illiberal": illiberal_items,
                    }

            # Populism–Pluralism
            pp = alternative_scores.get("populism_pluralism_hypothesis")
            if pp and pp.get("is_relevant") is not False:
                pluralism_items, populism_items = _fmt_items_pair(
                    pp.get("top_pluralism_hypotheses", []),
                    pp.get("top_populism_hypotheses", []),
                )
                if pluralism_items or populism_items:
                    why_these_results["Populism–Pluralism"] = {
                        "Pluralism": pluralism_items,
                        "Populism": populism_items,
                    }

        context_data = {
            'form': form,
            'results': results,
            'coordinates': coordinates,
            'input_text': text,
            'coordinates_json': json.dumps(coordinates),
            'alternative_scores': alternative_scores,
            'selected_approaches': selected_approaches,
            'why_these_results': why_these_results,
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
        return render(request, 'classifier/analysis.html', {'form': form})



def documentation(request):
    return render(request, 'classifier/documentation.html')

def privacy_notice(request):
    return render(request, 'classifier/privacynotice.html')

def imprint(request):
    return render(request, 'classifier/imprint.html')

def contact(request):
    return render(request, 'classifier/contact.html')

def faq(request):
    return render(request, 'classifier/faq.html')


@csrf_exempt
@require_http_methods(["POST"])
def api_classify(request):
    """API endpoint for classification"""
    start_time = time.time()
    log_memory_usage("API start")

    try:
        data = json.loads(request.body)
        text = data.get('text', '').strip()

        if not text:
            return JsonResponse({'error': 'Text is required'}, status=400)

        if len(text) > 2000:
            return JsonResponse({'error': 'Text too long (max 2000 characters)'}, status=400)

        # For API, use all hypothesis approaches by default
        selected_approaches = {
            'left_right_hypothesis': True,
            'liberal_illiberal_hypothesis': True,
            'populism_hypothesis': True,
        }

        # Generate alternative scores
        alternative_scorers = get_alternative_scorers(selected_approaches)
        alternative_scores = generate_alternative_scores(text, alternative_scorers, selected_approaches)

        cleanup_memory()
        processing_time = time.time() - start_time
        logger.info(f"API request completed in {processing_time:.2f} seconds")

        return JsonResponse({
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