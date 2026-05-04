import re, io, csv, json, logging, threading, queue, os, traceback
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


def is_real_paragraph(text):
    """Require at least 2 sentence-ending punctuation marks followed by a space or end,
    indicating at least 2 sentences."""
    sentences = re.split(r'[.!?]+(?:\s|$)', text.strip())
    real = [s for s in sentences if len(s.split()) >= 3]
    return len(real) >= 2


def clean_text(text):
    """Normalize text from PDFs: fix line breaks, special chars, ligatures."""
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # Replace non-breaking spaces and other whitespace variants
    text = re.sub(r'[\xa0\u2000-\u200b\u202f\u205f\u3000]', ' ', text)
    # Fix common PDF ligatures
    text = text.replace('\ufb01', 'fi').replace('\ufb02', 'fl')
    text = text.replace('\ufb00', 'ff').replace('\ufb03', 'ffi').replace('\ufb04', 'ffl')
    # Fix hyphenated line breaks (word- \n break -> word)
    text = re.sub(r'-(\n)', '', text)
    return text

def split_paragraphs(text, min_words=4):
    text = clean_text(text)
    # Split on blank lines
    raw = re.split(r'\n\s*\n', text.strip())
    # Split further on newlines followed by capital letter or Roman numeral
    chunks = []
    for block in raw:
        sub = re.split(r'\n(?=[A-Z0-9\u2160-\u2188])', block)
        chunks.extend(sub)
    paragraphs = []
    for p in chunks:
        # Join lines within a chunk (soft wraps from PDF)
        cleaned = ' '.join(p.split())
        if not cleaned:
            continue
        if len(cleaned.split()) < min_words:
            continue
        if not is_real_paragraph(cleaned):
            continue
        paragraphs.append(cleaned)
    return paragraphs


def _to_float(v):
    if v is None or v == 'NA':
        return None
    try:
        return round(float(v), 4)
    except (TypeError, ValueError):
        return None


def _score_paragraph(text, scorers, use_lr, use_lib, use_pop):
    result = {}
    if use_lr and 'left_right' in scorers:
        lr = scorers['left_right'].score_left_right(text)
        relevant = lr.get('is_relevant', True) and lr.get('score') != 'NA'
        result.update({
            'lr_score': _to_float(lr.get('score')) if relevant else None,
            'lr_interpretation': lr.get('interpretation', ''),
            'lr_confidence': _to_float(lr.get('confidence')) if relevant else None,
        })
    if use_lib and 'liberal_illiberal' in scorers:
        lib = scorers['liberal_illiberal'].score_liberal_illiberal(text)
        relevant = lib.get('is_relevant', True) and lib.get('score') != 'NA'
        result.update({
            'lib_score': _to_float(lib.get('score')) if relevant else None,
            'lib_interpretation': lib.get('interpretation', ''),
            'lib_confidence': _to_float(lib.get('confidence')) if relevant else None,
        })
    if use_pop and 'populism_pluralism' in scorers:
        pop = scorers['populism_pluralism'].score_populism_pluralism(text)
        relevant = pop.get('is_relevant', True) and pop.get('score') != 'NA'
        result.update({
            'pop_score': _to_float(pop.get('score')) if relevant else None,
            'pop_interpretation': pop.get('interpretation', ''),
            'pop_confidence': _to_float(pop.get('confidence')) if relevant else None,
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
        row = {'uid': r.get('uid', ''), 'paragraph': r.get('paragraph', '')}
        if use_lr:
            row['lr_score']      = r.get('lr_score') if r.get('lr_score') is not None else 'NA'
            row['lr_label']      = r.get('lr_interpretation', '')
            row['lr_confidence'] = r.get('lr_confidence') if r.get('lr_confidence') is not None else 'NA'
        if use_lib:
            row['lib_score']      = r.get('lib_score') if r.get('lib_score') is not None else 'NA'
            row['lib_label']      = r.get('lib_interpretation', '')
            row['lib_confidence'] = r.get('lib_confidence') if r.get('lib_confidence') is not None else 'NA'
        if use_pop:
            row['pop_score']      = r.get('pop_score') if r.get('pop_score') is not None else 'NA'
            row['pop_label']      = r.get('pop_interpretation', '')
            row['pop_confidence'] = r.get('pop_confidence') if r.get('pop_confidence') is not None else 'NA'
        writer.writerow(row)
    return output.getvalue()


def _send_email(to, csv_content):
    host_user = os.environ.get("EMAIL_HOST_USER", "")
    host_pass = os.environ.get("EMAIL_HOST_PASSWORD", "")
    logger.info("Attempting to send email to: %s from: %s", to, host_user)
    if not host_user or not host_pass:
        raise ValueError("EMAIL_HOST_USER or EMAIL_HOST_PASSWORD not set")
    msg = MIMEMultipart()
    msg["From"] = f"APSI long document <{host_user}>"
    msg["To"] = to
    msg["Subject"] = "Your APSI long document results"
    msg.attach(MIMEText(
        "<p>Your analysis is complete. Results are attached as a CSV file.</p>"
        "<br><p style='color:#888;font-size:12px'>Sent by APSI · Path to Power · HPI</p>",
        "html"
    ))
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
    use_lr  = bool(data.get('use_lr', False))
    use_lib = bool(data.get('use_lib', False))
    use_pop = bool(data.get('use_pop', False))

    logger.info("longdoc_score called: email=%s, use_lr=%s, use_lib=%s, use_pop=%s",
                email, use_lr, use_lib, use_pop)

    if not text:
        return JsonResponse({'error': 'No text provided.'}, status=400)
    if not any([use_lr, use_lib, use_pop]):
        return JsonResponse({'error': 'Please select at least one dimension.'}, status=400)

    paragraphs = split_paragraphs(text)
    if not paragraphs:
        return JsonResponse(
            {'error': 'No scorable paragraphs found. Make sure paragraphs have at least two sentences and are separated by a line break.'},
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
