"""
views_longdoc.py  —  classifier/views_longdoc.py
"""

import re
import io
import csv
import json
import logging
import threading
import queue
import os
import traceback
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

from django.shortcuts import render
from django.http import StreamingHttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

logger = logging.getLogger(__name__)


def split_paragraphs(text: str, min_words: int = 4) -> list:
    raw = re.split(r'\n\s*\n', text.strip())
    paragraphs = []
    for p in raw:
        cleaned = ' '.join(p.split())
        if cleaned and len(cleaned.split()) >= min_words:
            paragraphs.append(cleaned)
    return paragraphs


def _score_paragraph(text, scorers, use_lr, use_lib, use_pop):
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


def _build_csv(rows, use_lr, use_lib, use_pop):
    output = io.StringIO()
    hdr = ['uid', 'paragraph']
    if use_lr:  hdr += ['lr_score', 'lr_label', 'lr_confidence']
    if use_lib: hdr += ['lib_score', 'lib_label', 'lib_confidence']
    if use_pop: hdr += ['pop_score', 'pop_label', 'pop_confidence']
    writer = csv.DictWriter(output, fieldnames=hdr, extrasaction='ignore')
    writer.writeheader()
    for r in rows:
        row = {
            'uid': r.get('uid', ''),
            'paragraph': r.get('paragraph', ''),
        }
        if use_lr:
            row['lr_score']      = r.get('lr_score', 'NA') if r.get('lr_score') is not None else 'NA'
            row['lr_label']      = r.get('lr_interpretation', '')
            row['lr_confidence'] = r.get('lr_confidence', 'NA') if r.get('lr_confidence') is not None else 'NA'
        if use_lib:
            row['lib_score']      = r.get('lib_score', 'NA') if r.get('lib_score') is not None else 'NA'
            row['lib_label']      = r.get('lib_interpretation', '')
            row['lib_confidence'] = r.get('lib_confidence', 'NA') if r.get('lib_confidence') is not None else 'NA'
        if use_pop:
            row['pop_score']      = r.get('pop_score', 'NA') if r.get('pop_score') is not None else 'NA'
            row['pop_label']      = r.get('pop_interpretation', '')
            row['pop_confidence'] = r.get('pop_confidence', 'NA') if r.get('pop_confidence') is not None else 'NA'
        writer.writerow(row)
    return output.getvalue()


def _send_email(to, csv_content):
    host_user = os.environ.get("EMAIL_HOST_USER", "")
    host_pass = os.environ.get("EMAIL_HOST_PASSWORD", "")

    logger.info("Attempting to send email to: %s from: %s", to, host_user)

    if not host_user or not host_pass:
        raise ValueError("EMAIL_HOST_USER or EMAIL_HOST_PASSWORD not set in environment")

    msg = MIMEMultipart()
    msg["From"] = f"APSI <{host_user}>"
    msg["To"] = to
    msg["Subject"] = "Your APSI long document results"

    body = MIMEText(
        "<p>Hello,</p>"
        "<p>Your document analysis is complete. "
        "Results are attached as a CSV file.</p>"
        "<p>You can open it in Excel or Google Sheets.</p>"
        "<br><p style='color:#888;font-size:12px'>Sent by APSI · Path to Power · HPI</p>",
        "html"
    )
    msg.attach(body)

    attachment = MIMEBase("application", "octet-stream")
    attachment.set_payload(csv_content.encode("utf-8"))
    encoders.encode_base64(attachment)
    attachment.add_header("Content-Disposition", "attachment", filename="apsi_scores.csv")
    msg.attach(attachment)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(host_user, host_pass)
        server.sendmail(host_user, to, msg.as_string())

    logger.info("Email successfully sent to %s", to)


def longdoc(request):
    return render(request, 'classifier/longdoc.html')


@csrf_exempt
@require_http_methods(['POST'])
def longdoc_score(request):
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    text    = (data.get('text') or '').strip()
    email   = (data.get('email') or '').strip() or None
    use_lr  = bool(data.get('use_lr', True))
    use_lib = bool(data.get('use_lib', True))
    use_pop = bool(data.get('use_pop', True))

    logger.info("longdoc_score called: email=%s, use_lr=%s, use_lib=%s, use_pop=%s",
                email, use_lr, use_lib, use_pop)

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

    from .views import get_alternative_scorers
    selected = {
        'left_right_hypothesis':        use_lr,
        'liberal_illiberal_hypothesis':  use_lib,
        'populism_hypothesis':           use_pop,
    }
    scorers = get_alternative_scorers(selected) or {}

    total = len(paragraphs)
    result_queue = queue.Queue()

    def worker():
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

        email_sent = False
        if email and os.environ.get('EMAIL_HOST_USER'):
            try:
                csv_content = _build_csv(all_rows, use_lr, use_lib, use_pop)
                _send_email(email, csv_content)
                email_sent = True
                logger.info('Email successfully sent to %s', email)
            except Exception as exc:
                logger.error('Email failed for %s: %s', email, exc)
                logger.error('Email traceback: %s', traceback.format_exc())
        else:
            if not email:
                logger.info('No email address provided, skipping email.')
            if not os.environ.get('EMAIL_HOST_USER'):
                logger.error('EMAIL_HOST_USER is not set in environment!')

        result_queue.put(('done', {'email_sent': email_sent}))

    threading.Thread(target=worker, daemon=False).start()

    def stream():
        while True:
            try:
                item_type, payload = result_queue.get(timeout=3600)
            except queue.Empty:
                break
            yield json.dumps({'type': item_type, **payload}) + '\n'
            if item_type == 'done':
                break

    return StreamingHttpResponse(stream(), content_type='application/x-ndjson')
