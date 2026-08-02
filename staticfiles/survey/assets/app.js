/* ------------------------------------------------------------------ */
/*  Survey runner                                                       */
/*                                                                      */
/*  Static — no build step, no framework. It reads a survey definition  */
/*  from surveys/<id>.json, draws a random subset of items, collects    */
/*  slider ratings, and POSTs one row per rated item to a Google Apps   */
/*  Script endpoint.                                                    */
/*                                                                      */
/*  You should not need to edit this file to run a new survey. Add a    */
/*  JSON file in surveys/ and list it in surveys/manifest.json.         */
/* ------------------------------------------------------------------ */

(function () {
  "use strict";

  var screenEl = document.getElementById("screen");
  var bandId = document.getElementById("band-id");
  var bandMeta = document.getElementById("band-meta");
  var progress = document.getElementById("progress");
  var progressFill = document.getElementById("progress-fill");
  var progressText = document.getElementById("progress-text");
  var colophon = document.getElementById("colophon");

  var params = new URLSearchParams(location.search);

  /* Where the survey JSON and images live.
     - On a plain static host, leave window.SURVEY_BASE unset: paths stay
       relative to the page, e.g. "surveys/manifest.json".
     - Inside Django, the template sets it to "{% static 'survey/' %}" so
       the same files are found under /static/survey/. */
  var BASE = (window.SURVEY_BASE || "").replace(/\/+$/, "");
  function assetUrl(path) {
    return BASE ? BASE + "/" + String(path).replace(/^\/+/, "") : path;
  }

  /* Where the survey definitions are read from. Django points this at a
     view that serves static/survey/surveys/ directly, so editing a survey
     JSON takes effect without a second copy under STATIC_ROOT. Unset on a
     plain static host, where the files sit next to the page. */
  var DATA_BASE = (window.SURVEY_DATA_BASE || "").replace(/\/+$/, "");
  function dataUrl(file) {
    return DATA_BASE ? DATA_BASE + "/" + file : assetUrl("surveys/" + file);
  }

  /* ---- state ---- */
  var survey = null;
  var drawn = [];          // the items this respondent sees, in order
  var answers = {};        // uid -> { value, start, changeCount, ms }
  var profile = {};        // fixed questions everyone answers: id -> value
  var responseId = makeId();
  var startedAt = Date.now();
  var itemEnteredAt = 0;
  var sent = false;

  /* ---------------------------------------------------------------- */
  /*  Helpers                                                          */
  /* ---------------------------------------------------------------- */

  function makeId() {
    if (window.crypto && crypto.randomUUID) return crypto.randomUUID();
    return "r-" + Date.now().toString(36) + "-" + Math.random().toString(36).slice(2, 10);
  }

  function el(tag, className, text) {
    var n = document.createElement(tag);
    if (className) n.className = className;
    if (text !== undefined && text !== null) n.textContent = text;
    return n;
  }

  function paragraphs(parent, list, className) {
    (list || []).forEach(function (t) {
      parent.appendChild(el("p", className || null, t));
    });
  }

  function fill(template, values) {
    return String(template || "").replace(/\{(\w+)\}/g, function (m, key) {
      return Object.prototype.hasOwnProperty.call(values, key) ? values[key] : m;
    });
  }

  /* Fisher-Yates, so every item has an equal chance of being drawn. */
  function shuffle(list) {
    var a = list.slice();
    for (var i = a.length - 1; i > 0; i--) {
      var j = Math.floor(Math.random() * (i + 1));
      var t = a[i]; a[i] = a[j]; a[j] = t;
    }
    return a;
  }

  function bandFor(value) {
    var bands = (survey.slider && survey.slider.bands) || [];
    for (var i = 0; i < bands.length; i++) {
      if (value <= bands[i].upTo) return bands[i].label;
    }
    return bands.length ? bands[bands.length - 1].label : "";
  }

  function clamp(n, lo, hi) { return Math.max(lo, Math.min(hi, n)); }

  /* Colour along the illiberal - liberal ramp, matching the track's
     gradient so the number and the thumb position agree. */
  var RAMP = [
    [0.0, [192, 102, 47]],
    [0.5, [126, 133, 152]],
    [1.0, [42, 139, 152]]
  ];
  function rampColor(fraction) {
    var f = clamp(fraction, 0, 1);
    for (var i = 1; i < RAMP.length; i++) {
      if (f <= RAMP[i][0]) {
        var a = RAMP[i - 1], b = RAMP[i];
        var t = (f - a[0]) / (b[0] - a[0]);
        var c = [0, 1, 2].map(function (k) {
          return Math.round(a[1][k] + (b[1][k] - a[1][k]) * t);
        });
        return "rgb(" + c.join(",") + ")";
      }
    }
    return "rgb(42,139,152)";
  }

  /* ---------------------------------------------------------------- */
  /*  Screens                                                          */
  /* ---------------------------------------------------------------- */

  function show(node) {
    screenEl.innerHTML = "";
    screenEl.appendChild(node);
    window.scrollTo({ top: 0, behavior: "auto" });
  }

  function setProgress(step, total) {
    if (!total) { progress.hidden = true; return; }
    progress.hidden = false;
    progressFill.style.width = Math.round((step / total) * 100) + "%";
    progressText.textContent = step + " / " + total;
  }

  /* Institutional letterhead, shown on the first and last screens. */
  function logoStrip() {
    var logos = survey.logos || [];
    if (!logos.length) return null;
    var strip = el("div", "logos");
    logos.forEach(function (logo) {
      var img = document.createElement("img");
      img.src = assetUrl(logo.src);
      img.alt = logo.alt || "";
      if (logo.height) img.style.height = logo.height + "px";
      strip.appendChild(img);
    });
    return strip;
  }

  /* The definition cards — a coloured top rule, a title, a body. Used on
     the instructions page and again in the reminder above each passage,
     so the two always look the same. */
  function defCards(cards) {
    var defs = el("div", "defs");
    (cards || []).forEach(function (c) {
      var card = el("div", "def" + (c.tone === "low" ? " low" : ""));
      if (c.title) card.appendChild(el("h3", null, c.title));
      card.appendChild(el("p", null, c.body));
      defs.appendChild(card);
    });
    return defs;
  }

  function footer(children) {
    var f = el("div", "foot");
    children.filter(Boolean).forEach(function (c) { f.appendChild(c); });
    return f;
  }

  function button(label, className, onClick) {
    var b = el("button", className, label);
    b.type = "button";
    b.addEventListener("click", onClick);
    return b;
  }

  /* ---- 1. welcome ---- */

  function renderWelcome() {
    setProgress(0, 0);
    var w = survey.welcome || {};
    var wrap = el("div");

    var logos = logoStrip();
    if (logos) wrap.appendChild(logos);

    var head = el("div", "masthead");
    if (w.eyebrow) head.appendChild(el("p", "eyebrow", w.eyebrow));
    head.appendChild(el("h1", null, w.heading || survey.title || "Survey"));
    wrap.appendChild(head);

    var prose = el("div", "prose");
    (w.body || []).forEach(function (t, i) {
      prose.appendChild(el("p", i === 0 ? "lede" : null, t));
    });
    wrap.appendChild(prose);

    wrap.appendChild(footer([
      el("span", "spacer"),
      button(w.button || "Begin", "btn", afterWelcome)
    ]));

    show(wrap);
  }

  function hasQuestions() {
    return !!(survey.questions && (survey.questions.items || []).length);
  }

  function afterWelcome() {
    if (hasQuestions()) renderQuestions();
    else renderDefinition();
  }

  /* ---- 1b. fixed questions, asked of everyone ---- */

  function renderQuestions() {
    setProgress(0, 0);
    var q = survey.questions || {};
    var wrap = el("div");

    var head = el("div", "masthead");
    if (q.eyebrow) head.appendChild(el("p", "eyebrow", q.eyebrow));
    head.appendChild(el("h1", null, q.heading || "About you"));
    wrap.appendChild(head);

    var prose = el("div", "prose");
    paragraphs(prose, q.intro);
    wrap.appendChild(prose);

    var errs = {};
    var body = el("div", "qbody");
    (q.items || []).forEach(function (item) {
      body.appendChild(renderQuestion(item, errs));
    });
    wrap.appendChild(body);

    wrap.appendChild(footer([
      button((survey.item && survey.item.backLabel) || "Back", "btn ghost", renderWelcome),
      el("span", "spacer"),
      button(q.button || "Continue", "btn", function () {
        var firstBad = null;
        (q.items || []).forEach(function (item) {
          var v = profile[item.id];
          var missing = item.required && (v === undefined || v === null || String(v).trim() === "");
          /* "Other" picked but nothing typed in the box counts as missing. */
          if (item.required && v === "__other__" &&
              !String(profile[item.id + "__otherText"] || "").trim()) {
            missing = true;
          }
          errs[item.id].textContent = missing ? (item.errorText || "Please answer this question.") : "";
          if (missing && !firstBad) firstBad = item.id;
        });
        if (firstBad) {
          var node = document.getElementById("q-" + firstBad);
          if (node) node.scrollIntoView({ block: "center", behavior: "smooth" });
          return;
        }
        renderDefinition();
      })
    ]));

    show(wrap);
  }

  /**
   * Supported types: "one" (radio buttons, with an optional free-text
   * "other" choice) and "short" (single-line text).
   */
  function renderQuestion(item, errs) {
    var box = el("div", "qbox");
    box.id = "q-" + item.id;

    var label = el("p", "qlabel", item.label);
    if (item.required) {
      var star = el("span", "req", "*");
      label.appendChild(star);
    }
    box.appendChild(label);
    if (item.hint) box.appendChild(el("p", "qhint", item.hint));

    var err = el("p", "err");
    errs[item.id] = err;

    if (item.type === "short") {
      var input = document.createElement("input");
      input.type = "text";
      input.className = "line";
      input.value = profile[item.id] || "";
      input.setAttribute("aria-label", item.label);
      input.addEventListener("input", function () {
        profile[item.id] = input.value;
        err.textContent = "";
      });
      box.appendChild(input);
      box.appendChild(err);
      return box;
    }

    /* type "one" */
    var opts = el("div", "opts");
    var otherInput = null;

    function choose(value) {
      profile[item.id] = value;
      err.textContent = "";
      if (otherInput) {
        otherInput.disabled = value !== "__other__";
        if (!otherInput.disabled) otherInput.focus();
      }
    }

    (item.options || []).forEach(function (opt, i) {
      var id = "opt-" + item.id + "-" + i;
      var lab = el("label", "opt");
      lab.setAttribute("for", id);

      var radio = document.createElement("input");
      radio.type = "radio";
      radio.name = item.id;
      radio.id = id;
      radio.value = opt;
      radio.checked = profile[item.id] === opt;
      radio.addEventListener("change", function () {
        if (otherInput) { otherInput.disabled = true; }
        profile[item.id + "__otherText"] = "";
        choose(opt);
      });

      lab.appendChild(radio);
      lab.appendChild(el("span", "bub")).appendChild(el("i"));
      lab.appendChild(el("span", "opt-text", opt));
      opts.appendChild(lab);
    });

    if (item.allowOther) {
      var oid = "opt-" + item.id + "-other";
      var olab = el("label", "opt");
      olab.setAttribute("for", oid);

      var oradio = document.createElement("input");
      oradio.type = "radio";
      oradio.name = item.id;
      oradio.id = oid;
      oradio.value = "__other__";
      oradio.checked = profile[item.id] === "__other__";

      olab.appendChild(oradio);
      olab.appendChild(el("span", "bub")).appendChild(el("i"));
      olab.appendChild(el("span", "opt-text", item.otherLabel || "Other (please specify)"));
      opts.appendChild(olab);

      otherInput = document.createElement("input");
      otherInput.type = "text";
      otherInput.className = "line other";
      otherInput.placeholder = item.otherPlaceholder || "Please specify";
      otherInput.value = profile[item.id + "__otherText"] || "";
      otherInput.disabled = profile[item.id] !== "__other__";
      otherInput.setAttribute("aria-label", item.otherLabel || "Other, please specify");

      oradio.addEventListener("change", function () { choose("__other__"); });
      otherInput.addEventListener("input", function () {
        profile[item.id + "__otherText"] = otherInput.value;
        err.textContent = "";
      });

      opts.appendChild(otherInput);
    }

    box.appendChild(opts);
    box.appendChild(err);
    return box;
  }

  /* ---- 2. definition ---- */

  function renderDefinition() {
    setProgress(0, 0);
    var d = survey.definition || {};
    var wrap = el("div");

    var head = el("div", "masthead");
    if (d.eyebrow) head.appendChild(el("p", "eyebrow", d.eyebrow));
    head.appendChild(el("h1", null, d.heading || "Definitions"));
    wrap.appendChild(head);

    var intro = el("div", "prose");
    paragraphs(intro, d.intro, "lede");
    wrap.appendChild(intro);

    if (d.cards && d.cards.length) wrap.appendChild(defCards(d.cards));

    var outro = el("div", "prose");
    outro.style.paddingTop = "20px";
    paragraphs(outro, d.outro);
    wrap.appendChild(outro);

    wrap.appendChild(footer([
      button(survey.item && survey.item.backLabel || "Back", "btn ghost", afterWelcome),
      el("span", "spacer"),
      button(d.button || "Start", "btn", function () { renderItem(0); })
    ]));

    show(wrap);
  }

  /* ---- 3. an item ---- */

  function renderItem(index) {
    var item = drawn[index];
    var total = drawn.length;
    var cfg = survey.item || {};
    var sl = survey.slider || {};
    var min = sl.min === undefined ? 0 : sl.min;
    var max = sl.max === undefined ? 100 : sl.max;
    var step = sl.step || 1;

    var state = answers[item.uid];
    if (!state) {
      state = answers[item.uid] = {
        start: clamp(item.start === undefined ? Math.round((min + max) / 2) : item.start, min, max),
        value: null,
        changeCount: 0,
        ms: 0
      };
    }
    var current = state.value === null ? state.start : state.value;

    setProgress(index + 1, total);
    itemEnteredAt = Date.now();

    var wrap = el("div");
    var body = el("div", "item");

    body.appendChild(el("p", "item-hint", fill(cfg.hint || "Passage {n} of {total}", {
      n: index + 1, total: total
    })));

    /* Optional reminder of the coding instructions, repeated above every
       passage so nobody has to page back to the definitions. Drop
       item.reminder from the survey JSON to leave it out. */
    var rem = cfg.reminder;
    if (rem) {
      var card = el("div", "reminder");
      if (rem.question) card.appendChild(el("p", "reminder-q", rem.question));

      /* "cards" takes the same shape as definition.cards, but here they
         are stacked as sections of one full-width card rather than laid
         out side by side. The older low/high strings still work. */
      var cards = rem.cards;
      if (!cards && (rem.low || rem.high)) {
        cards = [];
        if (rem.low) cards.push({ tone: "low", body: rem.low });
        if (rem.high) cards.push({ tone: "high", body: rem.high });
      }
      (cards || []).forEach(function (c) {
        var sec = el("div", "reminder-sec" + (c.tone === "low" ? " low" : ""));
        if (c.title) sec.appendChild(el("h3", null, c.title));
        sec.appendChild(el("p", null, c.body));
        card.appendChild(sec);
      });

      if (rem.note) card.appendChild(el("p", "reminder-note", rem.note));
      body.appendChild(card);
    }

    body.appendChild(el("blockquote", "quote", item.text));
    /* Optional, like the other item labels: drop "prompt" and no heading
       is shown above the slider. */
    if (cfg.prompt) body.appendChild(el("p", "eyebrow", cfg.prompt));

    /* slider */
    var sw = el("div", "slider-wrap");

    var span = (max - min) || 1;
    var fractionOf = function (v) { return (v - min) / span; };

    var readout = el("div", "readout");
    var valEl = el("span", "val", String(current));
    valEl.style.color = rampColor(fractionOf(current));
    /* The band pill is optional: drop "bands" from the survey JSON and
       only the number is shown, rather than an empty pill. */
    var bandEl = el("span", "band-label", bandFor(current));
    bandEl.hidden = !bandFor(current);
    readout.appendChild(valEl);
    readout.appendChild(bandEl);
    sw.appendChild(readout);

    var range = document.createElement("input");
    range.type = "range";
    range.min = String(min);
    range.max = String(max);
    range.step = String(step);
    range.value = String(current);
    range.setAttribute("aria-label", cfg.prompt || "Rating");
    range.setAttribute("aria-valuetext",
      bandFor(current) ? current + " — " + bandFor(current) : String(current));

    var track = el("div", "track");
    track.appendChild(range);

    /* A fixed tick at the suggested value, so respondents can see how far
       they have moved from it. The offset accounts for the 28px thumb,
       which cannot travel the full width of the track. */
    var mark = el("span", "start-mark");
    mark.style.left = "calc(14px + (100% - 28px) * " + fractionOf(state.start).toFixed(4) + ")";
    mark.title = "Suggested starting point";
    track.appendChild(mark);

    sw.appendChild(track);

    var ticks = el("div", "ticks");
    (sl.ticks || [String(min), String(max)]).forEach(function (t) {
      ticks.appendChild(el("span", null, t));
    });
    sw.appendChild(ticks);

    /* The line under the slider is opt-in: drop anchorNote and movedNote
       from the survey JSON and nothing is shown. */
    var note = (cfg.anchorNote || cfg.movedNote) ? el("p", "anchor-note") : null;
    if (note) sw.appendChild(note);

    function paintNote() {
      if (!note) return;
      var v = Number(range.value);
      var delta = v - state.start;
      var template = delta === 0 ? cfg.anchorNote : cfg.movedNote;
      if (!template) { note.hidden = true; return; }
      note.hidden = false;
      note.className = "anchor-note" + (delta === 0 ? "" : " moved");
      note.textContent = fill(template, {
        start: state.start,
        delta: (delta > 0 ? "+" : "") + delta
      });
    }
    paintNote();

    range.addEventListener("input", function () {
      var v = Number(range.value);
      state.value = v;
      state.changeCount++;
      var label = bandFor(v);
      valEl.textContent = String(v);
      valEl.style.color = rampColor(fractionOf(v));
      bandEl.textContent = label;
      bandEl.hidden = !label;
      range.setAttribute("aria-valuetext", label ? v + " — " + label : String(v));
      paintNote();
      err.textContent = "";
    });

    body.appendChild(sw);
    var err = el("p", "err");
    body.appendChild(err);
    wrap.appendChild(body);

    function leave() {
      state.ms += Date.now() - itemEnteredAt;
      if (state.value === null) state.value = state.start;
    }

    var isLast = index === total - 1;
    var next = button(isLast ? (cfg.submitLabel || "Submit") : (cfg.nextLabel || "Next"), "btn", function () {
      if (sl.requireInteraction && state.value === null) {
        err.textContent = "Please move the slider before continuing.";
        return;
      }
      leave();
      if (isLast) submit();
      else renderItem(index + 1);
    });

    var back = button(cfg.backLabel || "Back", "btn ghost", function () {
      leave();
      if (index === 0) renderDefinition();
      else renderItem(index - 1);
    });

    wrap.appendChild(footer([back, el("span", "spacer"), next]));
    show(wrap);
  }

  /* ---------------------------------------------------------------- */
  /*  Submission                                                       */
  /* ---------------------------------------------------------------- */

  function buildRows() {
    var submittedAt = new Date().toISOString();
    var totalMs = Date.now() - startedAt;
    var participant = params.get((survey.submit && survey.submit.participantParam) || "p") || "";
    var drawnUids = drawn.map(function (i) { return i.uid; }).join(" ");

    /* Fixed-question answers are per respondent, not per item, so they are
       repeated on each of that person's rows — keeps the sheet one flat
       table you can read straight into R or Stata. */
    var profileCols = {};
    ((survey.questions && survey.questions.items) || []).forEach(function (q) {
      var v = profile[q.id];
      var text = profile[q.id + "__otherText"] || "";
      profileCols["q_" + q.id] = v === "__other__" ? "Other" : (v === undefined ? "" : v);
      if (q.allowOther) profileCols["q_" + q.id + "_other"] = v === "__other__" ? text : "";
    });

    return drawn.map(function (item, i) {
      var a = answers[item.uid] || {};
      var value = a.value === null || a.value === undefined ? a.start : a.value;
      var row = {
        submitted_at: submittedAt,
        response_id: responseId,
        survey_id: survey.id || "",
        participant: participant,
        item_order: i + 1,
        item_uid: item.uid,
        lib_score: item.libScore === undefined ? "" : item.libScore,
        lib_label: item.libLabel || "",
        slider_start: a.start,
        slider_value: value,
        delta: value - a.start,
        moved: a.changeCount > 0 ? 1 : 0,
        change_count: a.changeCount || 0,
        ms_on_item: a.ms || 0,
        total_ms: totalMs,
        drawn_uids: drawnUids,
        screen_width: window.innerWidth || (window.screen && window.screen.width) || "",
        user_agent: navigator.userAgent,
        referrer: document.referrer || ""
      };
      Object.keys(profileCols).forEach(function (k) { row[k] = profileCols[k]; });
      return row;
    });
  }

  function submit() {
    if (sent) return;
    sent = true;

    var rows = buildRows();
    var endpoint = (survey.submit && survey.submit.endpoint) || "";
    var payload = JSON.stringify({ surveyId: survey.id, responseId: responseId, rows: rows });

    renderSending();

    if (!endpoint) {
      renderDone({ saved: false, reason: "no-endpoint", payload: payload });
      return;
    }

    /* Content-Type text/plain keeps the browser from sending a CORS
       preflight, which Apps Script web apps do not answer. */
    var opts = {
      method: "POST",
      headers: { "Content-Type": "text/plain;charset=utf-8" },
      body: payload,
      redirect: "follow"
    };

    fetch(endpoint, opts)
      .then(function (res) {
        if (!res.ok) throw new Error("HTTP " + res.status);
        renderDone({ saved: true });
      })
      .catch(function () {
        /* Some Apps Script deployments do not expose CORS headers on the
           redirect. Fire again opaquely — the write still happens, we just
           cannot read the reply. */
        opts.mode = "no-cors";
        return fetch(endpoint, opts).then(function () {
          renderDone({ saved: true, opaque: true });
        });
      })
      .catch(function (e) {
        renderDone({ saved: false, reason: String(e && e.message || e), payload: payload });
      });
  }

  function renderSending() {
    setProgress(drawn.length, drawn.length);
    var wrap = el("div");
    var prose = el("div", "prose");
    prose.style.padding = "40px 24px";
    prose.appendChild(el("p", "lede", "Sending your answers…"));
    wrap.appendChild(prose);
    show(wrap);
  }

  function renderDone(result) {
    setProgress(drawn.length, drawn.length);
    var d = survey.done || {};
    var wrap = el("div");

    var logos = logoStrip();
    if (logos) wrap.appendChild(logos);

    var head = el("div", "masthead");
    var stamp = el("span", "stamp" + (result.saved ? "" : " warn"),
      result.saved ? (d.stamp || "Received") : "Not sent");
    head.appendChild(stamp);
    head.appendChild(el("h1", null, result.saved ? (d.heading || "Thank you") : "Almost there"));
    wrap.appendChild(head);

    var prose = el("div", "prose");
    if (result.saved) {
      paragraphs(prose, d.body && d.body.length ? d.body : ["Your answers have been recorded."]);
    } else {
      prose.appendChild(el("p", "lede",
        "Your answers could not be sent automatically, and nothing has been lost. " +
        "Please copy the text below and email it to the researcher."));
      var ta = el("textarea", "fallback");
      ta.value = result.payload;
      ta.readOnly = true;
      prose.appendChild(ta);

      var actions = el("div");
      actions.style.marginTop = "12px";
      actions.appendChild(button("Copy", "btn ghost", function () {
        ta.select();
        try { document.execCommand("copy"); } catch (e) { /* user can copy by hand */ }
      }));
      prose.appendChild(actions);
      prose.appendChild(el("p", "err", "Reason: " + result.reason));
    }
    wrap.appendChild(prose);
    wrap.appendChild(footer([el("span", "spacer")]));

    show(wrap);
  }

  function renderFatal(message, detail) {
    setProgress(0, 0);
    var wrap = el("div");
    var head = el("div", "masthead");
    head.appendChild(el("span", "stamp warn", "Not available"));
    head.appendChild(el("h1", null, message));
    wrap.appendChild(head);
    var prose = el("div", "prose");
    prose.appendChild(el("p", null, detail));
    wrap.appendChild(prose);
    show(wrap);
  }

  /* ---------------------------------------------------------------- */
  /*  Boot                                                             */
  /* ---------------------------------------------------------------- */

  function getJSON(url) {
    return fetch(url, { cache: "no-cache" }).then(function (r) {
      if (!r.ok) throw new Error(url + " → HTTP " + r.status);
      return r.json();
    });
  }

  function boot() {
    var wanted = params.get("s");

    getJSON(dataUrl("manifest.json"))
      .catch(function () { return { default: wanted }; })
      .then(function (manifest) {
        var id = wanted || manifest.default;
        if (!id) throw new Error("No survey requested and no default set in surveys/manifest.json.");
        if (!/^[a-z0-9._-]+$/i.test(id)) throw new Error("Bad survey id: " + id);
        return getJSON(dataUrl(id + ".json"));
      })
      .then(function (def) {
        survey = def;

        var items = survey.items || [];
        if (!items.length) throw new Error("This survey has no items.");
        var draw = Math.min((survey.sampling && survey.sampling.drawCount) || 4, items.length);
        drawn = shuffle(items).slice(0, draw);

        document.title = survey.title || "Survey";
        bandId.textContent = survey.shortName || survey.id || "Survey";
        bandMeta.textContent = draw + " passages · ~5 min";
        colophon.textContent = "Response " + responseId.slice(0, 8);

        renderWelcome();
      })
      .catch(function (e) {
        renderFatal("This survey could not be loaded", String(e && e.message || e));
      });
  }

  /* Warn before an accidental reload mid-survey. */
  window.addEventListener("beforeunload", function (e) {
    if (sent || !drawn.length || screenEl.querySelector(".quote") === null) return;
    e.preventDefault();
    e.returnValue = "";
  });

  boot();
})();
