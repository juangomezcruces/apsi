"""
views_longdoc.py  —  add to classifier/

Provides two URL endpoints:
  GET  /longdoc/          → renders the long-document scorer page
  POST /longdoc/score/    → streams NDJSON results paragraph by paragraph
"""

import re
import io
import csv
import json
import logging
import threading
import queue
import os

from django.shortcuts import render
from django.http import StreamingHttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

import resend

logger = logging.getLogger(__name__)

resend.api_key = os.environ.get("RESEND_API_KEY", "")

# ── Paragraph splitter ────────────────────────────────────────────────────────

def split_paragraphs(text: str, min_words: int = 4) -> list:
    raw = re.split(r'\n\s*\n', text.strip())
    paragraphs = []
    for p in raw:
        cleaned = ' '.join(p.split())
        if cleaned and len(cleaned.split()) >= min_words:
            paragraphs.append(cleaned)
    return paragraphs


# ── Score a single paragraph using the existing scorer instances ──────────────

def _score_paragraph(text, scorers, use_lr, use_lib, use_pop):
    """
    scorers is the dict returned by views.get_alternative_scorers().
    Keys: 'left_right', 'liberal_illiberal', 'populism_pluralism'
    """
    def to_float(v):
        if v is None or v == 'NA':
            return None
        try:
            return round(float(v), 4)
        except (TypeError, ValueError):
            return None

    result = {}

    if use_lr and 'left_right' in scorers:
        lr = scorers['left_right'].score_left_right(text)
        relevant = lr.get('is_relevant', True) and lr.get('score') != 'NA'
        result.update({
            'lr_score':          to_float(lr.get('score')) if relevant else None,
            'lr_interpretation': lr.get('interpretation', ''),
            'lr_confidence':     to_float(lr.get('confidence')) if relevant else None,
        })

    if use_lib and 'liberal_illiberal' in scorers:
        lib = scorers['liberal_illiberal'].score_liberal_illiberal(text)
        relevant = lib.get('is_relevant', True) and lib.get('score') != 'NA'
        result.update({
            'lib_score':          to_float(lib.get('score')) if relevant else None,
            'lib_interpretation': lib.get('interpretation', ''),
            'lib_confidence':     to_float(lib.get('confidence')) if relevant else None,
        })

    if use_pop and 'populism_pluralism' in scorers:
        pop = scorers['populism_pluralism'].score_populism_pluralism(text)
        relevant = pop.get('is_relevant', True) and pop.get('score') != 'NA'
        result.update({
            'pop_score':          to_float(pop.get('score')) if relevant else None,
            'pop_interpretation': pop.get('interpretation', ''),
            'pop_confidence':     to_float(pop.get('confidence')) if relevant else None,
        })

    return result


# ── CSV builder ───────────────────────────────────────────────────────────────

def _build_csv(rows, use_lr, use_lib, use_pop):
    output = io.StringIO()
    hdr = ['uid', 'paragraph']
    if use_lr:  hdr += ['lr_score', 'lr_interpretation', 'lr_confidence']
    if use_lib: hdr += ['lib_score', 'lib_interpretation', 'lib_confidence']
    if use_pop: hdr += ['pop_score', 'pop_interpretation', 'pop_confidence']
    writer = csv.DictWriter(output, fieldnames=hdr, extrasaction='ignore')
    writer.writeheader()
    for r in rows:
        writer.writerow({k: (v if v is not None else 'NA') for k, v in r.items() if k in hdr})
    return output.getvalue()


# ── Email sender ──────────────────────────────────────────────────────────────

def _send_email(to, csv_content):
    params = {
        'from': 'APSI <onboarding@resend.dev>',
        'to': [to],
        'subject': 'Your APSI Long Document results',
        'html': (
            '<p>Hello,</p>'
            '<p>Your document analysis is complete. '
            'Results are attached as a CSV file.</p>'
            '<p>You can open it in Excel or Google Sheets.</p>'
            "<br><p style='color:#888;font-size:12px'>Sent by APSI · HPI</p>"
        ),
        'attachments': [{
            'filename': 'apsi_scores.csv',
            'content': list(csv_content.encode('utf-8')),
        }],
    }
    resend.Emails.send(params)


# ── Views ─────────────────────────────────────────────────────────────────────

def longdoc(request):
    """Render the long-document scorer page."""
    return render(request, 'classifier/longdoc.html')


@csrf_exempt
@require_http_methods(['POST'])
def longdoc_score(request):
    """
    Stream NDJSON results, one line per paragraph.
    Sends email (if provided) from a background thread so it completes
    even if the browser window is closed before scoring finishes.
    """
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    text    = data.get('text', '').strip()
    email   = data.get('email', '').strip() or None
    use_lr  = bool(data.get('use_lr', True))
    use_lib = bool(data.get('use_lib', True))
    use_pop = bool(data.get('use_pop', True))

    if not text:
        return JsonResponse({'error': 'No text provided.'}, status=400)
    if not any([use_lr, use_lib, use_pop]):
        return JsonResponse({'error': 'Select at least one dimension.'}, status=400)

    paragraphs = split_paragraphs(text)
    if not paragraphs:
        return JsonResponse(
            {'error': 'No paragraphs found. Separate paragraphs with a blank line.'},
            status=400,
        )

    # Load only the scorers we need, reusing the same helper as views.py
    from .views import get_alternative_scorers
    selected = {
        'left_right_hypothesis':       use_lr,
        'liberal_illiberal_hypothesis': use_lib,
        'populism_hypothesis':          use_pop,
    }
    scorers = get_alternative_scorers(selected) or {}

    total = len(paragraphs)
    result_queue = queue.Queue()

    def worker():
        """Background thread — keeps running even if client disconnects."""
        all_rows = []
        result_queue.put(('total', {'total': total}))

        for i, paragraph in enumerate(paragraphs):
            uid = f'doc_{i:04d}'
            try:
                scores = _score_paragraph(paragraph, scorers, use_lr, use_lib, use_pop)
            except Exception as exc:
                logger.warning('Error scoring %s: %s', uid, exc)
                scores = {}
            row = {'uid': uid, 'paragraph': paragraph, **scores}
            all_rows.append(row)
            result_queue.put(('row', row))

        # Send email regardless of client connection
        email_sent = False
        if email and resend.api_key:
            try:
                csv_content = _build_csv(all_rows, use_lr, use_lib, use_pop)
                _send_email(email, csv_content)
                email_sent = True
                logger.info('Email sent to %s', email)
            except Exception as exc:
                logger.error('Email error: %s', exc)

        result_queue.put(('done', {'email_sent': email_sent}))

    # daemon=False → thread outlives the HTTP connection
    threading.Thread(target=worker, daemon=False).start()

    def stream():
        while True:
            try:
                item_type, data = result_queue.get(timeout=300)
            except queue.Empty:
                break
            if item_type == 'total':
                yield json.dumps({'type': 'total', **data}) + '\n'
            elif item_type == 'row':
                yield json.dumps({'type': 'row', **data}) + '\n'
            elif item_type == 'done':
                yield json.dumps({'type': 'done', **data}) + '\n'
                break

    return StreamingHttpResponse(stream(), content_type='application/x-ndjson')
