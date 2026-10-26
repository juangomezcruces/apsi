"""
Liberal-illiberal scorer, v2: commitment to the democratic process.

Scale: 0 = openly autocratic, 5 = no clear position, 10 = fully committed.
NA = the text is not about democratic principles, or takes no position on them.

TARGET CONCEPT (V-Party v2paplur; Lindberg et al. 2022, p. 27)
--------------------------------------------------------------
"To what extent was [the actor] clearly committed to free and fair elections
with multiple parties, freedom of speech, media, assembly and association?"
  No commitment   = openly supports autocratic rule without elections or these
                    freedoms (theocracy, single-party rule, revolutionary regime).
  Full commitment = unambiguously supports the freedoms AND pledges to accept
                    defeat in free and fair elections.

The battery is built from the parts of that definition: ELECTIONS, DEFEAT,
SPEECH, MEDIA, ASSEMBLY, ASSOCIATION and the REGIME type it names as the zero
point. Each part has a liberal and an illiberal side.

WHAT CHANGED FROM v1 (alternativeLib_v1.py) AND WHY
---------------------------------------------------
1.  MARGINS, NOT PROBABILITIES. Political_DEBATE has two labels, so the margin
    (logit entailment - logit not-entailment) is the sufficient statistic and
    p = sigmoid(margin). v1's softmax saturates at 0/1, and its prob_floor of
    0.25 then threw away whatever was left between 0 and 0.25.

2.  PAIRED BATTERY. v1 had 17 liberal and 18 illiberal hypotheses with unequal
    weight mass and no mirror between them. Each item is now a PAIR, and the
    evidence is the contrast c = (m_liberal - m_illiberal) / 2. This cancels a
    template's own tendency to entail, and it gives both ends of the scale the
    same headroom by construction.

3.  NO k_score, NO MUTUAL SUPPRESSION. v1 counted fires on BOTH sides to set
    the top-k for EACH side, so illiberal fires diluted the liberal average and
    texts that tripped many hypotheses read as more extreme. The ad-hoc
    1 - min(avg, 0.80) cross-penalty then compressed mixed texts. An inactive
    pair has c ~ 0, which is the neutral value, so a weighted mean over the
    ACTIVE pairs needs neither device.

4.  ITEMS ON THE CONSTRUCT. Dropped from v1 because v2paplur does not ask
    about them: court independence, "democracy is broken", "liberal freedoms
    have gone too far", voting-rights EXPANSION as such. Demonizing opponents
    belongs to the wider anti-pluralism index, not to this indicator, so it
    survives only where it targets the media or the opposition's right to
    exist. Added: accepting defeat as a pair of its own, and a regime pair for
    rule by religion, ideology or revolution.

5.  ONE GATE, NEUTRAL PROBES. v1 gated on the max of 30 STANCE probes at
    p >= 0.6. Stance probes entail on anything political, and a max over 30
    is permissive. v1 then applied a second "no clear signal" gate whose
    threshold depended on the entry point (thr=0.15 here, 0.7 in quick_score),
    so the same text could be NA by one path and scored by the other. Now
    there are seven "is about" probes, one per part of the concept, with a
    single threshold in margin units. After that a text is NA only if no pair
    is active.

6.  MILITANT DEMOCRACY, SCOPED. Restricting violence, incitement or movements
    that would abolish democracy is not illiberal. A single probe for that
    framing now discounts only the NEGATIVE contrasts of the freedom pairs,
    smoothly via sigmoid(margin) rather than v1's hard 0.5 hinge. Elections,
    defeat, regime and opposition-party pairs are never discounted: "the
    opposition are extremists" is exactly how bans of real oppositions are
    justified.

7.  CONFIDENCE YOU CAN EXPLAIN. confidence = W / (W + 1) * (1 - SD), with W
    the weight of active evidence and SD the weighted dispersion of positions
    across active pairs. Confidence goes up with more evidence and down with
    disagreement between pairs, and nothing else feeds it.

8.  MECHANICS. Inputs are moved to the model's device (v1 left them on the
    CPU). Truncation is "only_first", so a hypothesis is never cut. Texts over
    CHUNK_WORDS are split on sentences and margins are max-pooled across
    chunks, where v1 silently dropped everything past 512 tokens.
    score_texts() batches many texts. An injectable margin_fn lets the
    aggregation be tested, or re-run from cached margins, without torch.

9.  ACTOR LEVEL. v2paplur rates an ACTOR. aggregate_actor() pools a party's
    texts and applies the conjunctive rule: level 4 needs evidence for both
    the freedoms and accepting defeat, and level 0 needs open support for an
    autocratic regime.

The output keys views.py and views_longdoc.py read are unchanged. New keys
add to them: vparty_level, component_positions, pair_positions, margins.
"""
import logging
import re
import warnings

import numpy as np

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

MODEL_NAME = "mlburnham/Political_DEBATE_large_v1.0"

# ============================================================================
# BATTERY: (domain, component, weight, liberal side, illiberal side)
# ============================================================================

BATTERY = [
    # ELECTIONS -- "free and fair elections with multiple parties"
    ("multiparty_elections", "elections", 1.0,
     "The text supports choosing the government through free and fair elections between competing parties.",
     "The text supports choosing or keeping a government without free and competitive elections."),
    ("electoral_conduct", "elections", 0.7,
     "The text supports elections run impartially, with every party able to campaign freely.",
     "The text justifies rigging, postponing, or manipulating elections."),
    ("franchise", "elections", 0.6,
     "The text supports an equal right to vote for all adult citizens.",
     "The text supports denying the vote to some citizens because of their group, beliefs, or loyalty."),
    # DEFEAT -- "pledge to accept defeat in free and fair elections"
    ("accept_defeat", "defeat", 1.0,
     "The text commits to accepting election results and handing over power peacefully after a defeat.",
     "The text refuses to accept an election result or to hand over power after losing."),
    ("lawful_means", "defeat", 0.8,
     "The text insists that power may be won only through elections and lawful means.",
     "The text supports taking or keeping power by force, revolution, or a coup."),
    # SPEECH
    ("free_speech", "speech", 1.0,
     "The text defends freedom of speech, including for critics of the government.",
     "The text supports punishing or silencing people for peaceful political speech."),
    ("criticism", "speech", 0.8,
     "The text defends the right of citizens to criticize those in power.",
     "The text portrays criticism of the government or its leader as illegitimate or treasonous."),
    # MEDIA
    ("press_freedom", "media", 1.0,
     "The text defends independent media free from government control.",
     "The text supports government censorship or control of the media."),
    ("press_role", "media", 0.6,
     "The text defends the role of journalists in holding those in power to account.",
     "The text portrays independent journalists as enemies who must be stopped."),
    # ASSEMBLY
    ("assembly", "assembly", 1.0,
     "The text defends the right to peaceful assembly and protest.",
     "The text supports banning or breaking up peaceful protests."),
    ("protest_force", "assembly", 0.7,
     "The text condemns the use of force against peaceful demonstrators.",
     "The text justifies the use of force against peaceful demonstrators."),
    # ASSOCIATION
    ("opposition_parties", "association", 1.0,
     "The text accepts opposition parties as legitimate competitors for power.",
     "The text supports banning or eliminating opposition parties."),
    ("civil_society", "association", 0.8,
     "The text defends the right of citizens to form independent associations, unions, and parties.",
     "The text supports banning or controlling independent associations and unions."),
    # REGIME -- the zero point: "theocracy; single-party rule; revolutionary regime"
    ("one_party", "regime", 1.0,
     "The text supports a political system in which several parties compete for power.",
     "The text supports rule by a single party or leader without political competition."),
    ("source_of_authority", "regime", 1.0,
     "The text holds that the right to govern comes from the votes of the people.",
     "The text holds that the right to govern comes from religion, ideology, or revolution rather than elections."),
    ("constitutional_order", "regime", 0.6,
     "The text defends constitutional rules that limit how power is won and used.",
     "The text supports suspending the constitution or ruling by decree."),
]

COMPONENTS = ["elections", "defeat", "speech", "media", "assembly", "association", "regime"]
FREEDOM_COMPONENTS = ("speech", "media", "assembly", "association")
AUTOCRACY_COMPONENTS = ("regime", "elections", "defeat")

# One neutral "is about" probe per part of the concept. Stance-free on purpose:
# a probe that takes a side entails on any partisan text.
TOPIC_HYPOTHESES = [
    "The text is about elections or who should hold political power.",
    "The text is about accepting or rejecting the result of an election.",
    "The text is about freedom of speech or the right to criticize the government.",
    "The text is about the freedom or control of the media.",
    "The text is about protests or public demonstrations.",
    "The text is about political parties, the opposition, or civic organizations.",
    "The text is about democracy, dictatorship, or how a country should be governed.",
]

MILITANT_DEMOCRACY_HYPOTHESIS = (
    "The text restricts only violence, incitement, or movements that seek to abolish "
    "democracy, while protecting peaceful political activity.")
# Freedom pairs whose ILLIBERAL side a defensive-democracy framing can explain.
DEFENSIVE_TARGETS = {"free_speech", "criticism", "press_freedom", "assembly",
                     "protest_force", "civil_society"}

# Same thresholds as the mock scorer in views.py, so the label never depends
# on which code path produced the score.
INTERPRETATION_BANDS = [(3.0, "Strong Illiberal"), (4.5, "Illiberal"), (5.5, "Moderate"),
                        (7.0, "Liberal"), (np.inf, "Strong Liberal")]
VPARTY_LABELS = ["Not at all committed", "Not committed", "Weakly committed",
                 "Committed", "Fully committed"]

CHUNK_WORDS = 200   # ~270 tokens; leaves room for the hypothesis inside 512
SENT = re.compile(r"(?<=[.!?])\s+")


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def interpret(score):
    for upper, label in INTERPRETATION_BANDS:
        if score < upper:
            return label


def chunk_text(text, chunk_words=CHUNK_WORDS):
    """Sentence-aligned chunks of about chunk_words words. Short texts come back whole."""
    words = text.split()
    if len(words) <= chunk_words:
        return [text]
    out, cur, n = [], [], 0
    for s in SENT.split(text):
        w = len(s.split())
        if n and n + w > chunk_words:
            out.append(" ".join(cur)); cur, n = [], 0
        cur.append(s); n += w
    if cur:
        out.append(" ".join(cur))
    return out


# ============================================================================
# SCORER
# ============================================================================

class LiberalIlliberalScorer:
    """
    Parameters
    ----------
    topic_margin_min : float
        Gate. The text is relevant if any topic probe reaches this margin.
        0.0 = "more likely about it than not". Raise it to gate harder.
    active_margin : float
        A pair is active if either side reaches this margin.
    contrast_scale : float
        Logit units per unit of position: pos = tanh(c / contrast_scale).
        At 4, a contrast of 4 logits (sides at +4 / -4) is a position of 0.76.
    shrinkage : float
        Pseudo-weight of neutral evidence: score = 5 + 5 * sum(w*pos) / (W + k0).
        A text resting on one weight-1 pair cannot reach the end of the scale.
    defensive_discount : float
        Largest share of a freedom pair's illiberal contrast that the
        militant-democracy framing can cancel.
    margin_fn : callable(premises, hypotheses) -> np.ndarray, optional
        Replaces the model. Used for tests and for re-scoring cached margins.
    """

    def __init__(self, model_name=MODEL_NAME, topic_margin_min=0.0, active_margin=0.0,
                 contrast_scale=4.0, shrinkage=0.5, defensive_discount=0.65,
                 battery=BATTERY, topic_hypotheses=TOPIC_HYPOTHESES,
                 batch_size=16, max_length=512, margin_fn=None):
        self.battery = list(battery)
        self.topic_hypotheses = list(topic_hypotheses)
        self.domains = [b[0] for b in self.battery]
        self.components = [b[1] for b in self.battery]
        self.weights = np.array([b[2] for b in self.battery], dtype=float)
        self.topic_margin_min = float(topic_margin_min)
        self.active_margin = float(active_margin)
        self.contrast_scale = float(contrast_scale)
        self.shrinkage = float(shrinkage)
        self.defensive_discount = float(defensive_discount)
        self.defensive_mask = np.array([d in DEFENSIVE_TARGETS for d in self.domains])
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)

        # Column layout of the margin matrix: topics | militant probe | L, I, L, I, ...
        self.hypotheses = (self.topic_hypotheses + [MILITANT_DEMOCRACY_HYPOTHESIS]
                           + [h for b in self.battery for h in (b[3], b[4])])
        self.n_topic = len(self.topic_hypotheses)

        if margin_fn is not None:
            self._margin_fn = margin_fn
        else:
            self._load_model(model_name)
            self._margin_fn = self._model_margins

    # ------------------------------------------------------------------ model
    def _load_model(self, model_name):
        from .shared_model_cache import SharedModelCache   # imports torch
        cache = SharedModelCache()
        self.model, self.tokenizer = cache.get_model_and_tokenizer(model_name)
        self.device = cache.get_device()
        self.entailment_idx = self._find_entailment_index()

    def _find_entailment_index(self):
        config = self.model.config
        if getattr(config, 'label2id', None):
            for label, idx in config.label2id.items():
                if label.lower() in ['entailment', 'entail']:
                    return idx
        return 0

    def _model_margins(self, premises, hypotheses):
        import torch
        out = np.empty(len(premises), dtype=float)
        for i in range(0, len(premises), self.batch_size):
            j = min(i + self.batch_size, len(premises))
            enc = self.tokenizer(premises[i:j], hypotheses[i:j], return_tensors="pt",
                                 truncation="only_first", padding=True, max_length=self.max_length)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.inference_mode():
                lg = self.model(**enc).logits.float()
            ent = lg[:, self.entailment_idx]
            rest = torch.cat([lg[:, :self.entailment_idx], lg[:, self.entailment_idx + 1:]], dim=1)
            out[i:j] = (ent - torch.logsumexp(rest, dim=1)).cpu().numpy()
        return out

    def margin_matrix(self, texts):
        """(n_texts, n_hypotheses) margins, max-pooled over each text's chunks."""
        chunks = [chunk_text(t) for t in texts]
        owner = np.repeat(np.arange(len(texts)), [len(c) for c in chunks])
        flat = [c for cs in chunks for c in cs]
        nh = len(self.hypotheses)
        prem = [c for c in flat for _ in range(nh)]
        hyp = self.hypotheses * len(flat)
        m = np.asarray(self._margin_fn(prem, hyp), dtype=float).reshape(len(flat), nh)
        out = np.full((len(texts), nh), -np.inf)
        np.maximum.at(out, owner, m)
        return out

    # ------------------------------------------------------------ aggregation
    def pair_positions(self, row):
        """Margins for one text -> (positions, active mask, militant weight)."""
        bat = row[self.n_topic + 1:]
        lib, ill = bat[0::2], bat[1::2]
        active = np.fmax(lib, ill) >= self.active_margin
        c = (lib - ill) / 2.0
        md = float(_sigmoid(row[self.n_topic]))
        discount = np.where(self.defensive_mask & (c < 0), 1.0 - self.defensive_discount * md, 1.0)
        pos = np.tanh(c * discount / self.contrast_scale)
        return np.where(active, pos, 0.0), active, md

    def _summarize(self, pos, active, weights):
        """Score, confidence, component means and V-Party level from pair positions."""
        w = weights * active
        W = float(w.sum())
        mean_pos = float((w * pos).sum() / W)
        score = float(np.clip(5.0 + 5.0 * (w * pos).sum() / (W + self.shrinkage), 0.0, 10.0))
        sd = float(np.sqrt((w * (pos - mean_pos) ** 2).sum() / W))
        confidence = W / (W + 1.0) * max(0.0, 1.0 - sd)
        comp = {}
        for k in COMPONENTS:
            sel = np.array([c == k for c in self.components]) & active
            if sel.any():
                comp[k] = float((weights[sel] * pos[sel]).sum() / weights[sel].sum())
        return score, confidence, comp, self._vparty_level(score, comp)

    @staticmethod
    def _vparty_level(score, comp):
        """Approximate crosswalk to the 0-4 v2paplur scale (UNCALIBRATED).

        Bands of 2 points, then the two conjunctive rules of the definition:
        4 needs positive evidence on accepting defeat AND on at least one
        freedom, and 0 needs open support for an autocratic alternative.
        """
        level = int(min(4, score // 2))
        if level == 4 and not (comp.get("defeat", 0) > 0
                               and any(comp.get(k, 0) > 0 for k in FREEDOM_COMPONENTS)):
            level = 3
        if level == 0 and not any(comp.get(k, 0) < -0.5 for k in AUTOCRACY_COMPONENTS):
            level = 1
        return level

    def score_from_margins(self, row, text=""):
        row = np.asarray(row, dtype=float)
        topic_margin = float(np.max(row[:self.n_topic]))
        base = {'text': text, 'topic_probability': float(_sigmoid(topic_margin)),
                'topic_margin': topic_margin, 'margins': row.tolist()}
        if topic_margin < self.topic_margin_min:
            return {**base, 'score': 'NA', 'confidence': 0.0, 'is_relevant': False,
                    'contradiction_detected': False,
                    'interpretation': 'Not about democratic principles'}

        pos, active, md = self.pair_positions(row)
        if not active.any():
            return {**base, 'score': 'NA', 'confidence': 0.0, 'is_relevant': False,
                    'contradiction_detected': False,
                    'interpretation': 'No clear position on democratic principles'}

        score, confidence, comp, level = self._summarize(pos, active, self.weights)
        W = float((self.weights * active).sum())
        bat = row[self.n_topic + 1:]
        top_lib, top_ill = [], []
        for p in np.flatnonzero(active):
            side = 0 if pos[p] >= 0 else 1
            item = {'hypothesis': self.battery[p][3 + side],
                    'probability': float(_sigmoid(bat[2 * p + side])),
                    'score_impact': float(5.0 * self.weights[p] * abs(pos[p]) / (W + self.shrinkage)),
                    'domain': self.domains[p]}
            (top_lib if side == 0 else top_ill).append(item)
        top_lib.sort(key=lambda d: -d['score_impact'])
        top_ill.sort(key=lambda d: -d['score_impact'])

        return {**base,
                'score': score,
                'confidence': float(confidence),
                'is_relevant': True,
                'interpretation': interpret(score),
                'contradiction_detected': bool((pos[active] > 0.5).any() and (pos[active] < -0.5).any()),
                'top_liberal_hypotheses': top_lib[:5],
                'top_illiberal_hypotheses': top_ill[:5],
                'vparty_level': level,
                'vparty_label': VPARTY_LABELS[level],
                'component_positions': comp,
                'pair_positions': {d: float(pos[i]) for i, d in enumerate(self.domains) if active[i]},
                'n_active': int(active.sum()),
                'militant_democracy': md}

    # ---------------------------------------------------------------- public
    def score_texts(self, texts):
        texts = [str(t) for t in texts]
        M = self.margin_matrix(texts)
        return [self.score_from_margins(M[i], t) for i, t in enumerate(texts)]

    def score_liberal_illiberal(self, text):
        return self.score_texts([text])[0]

    def quick_score(self, text):
        return self.score_liberal_illiberal(text)['score']

    def aggregate_actor(self, results):
        """Pools one actor's scored texts into an actor-level estimate.

        Each pair's position is averaged over the texts where it was active, so
        a party that addresses press freedom in 3 of 40 speeches is placed on
        press freedom by those 3, not diluted by the other 37.
        """
        scored = [r for r in results if r.get('is_relevant') and r.get('score') != 'NA']
        out = {'n_texts': len(results), 'n_scored': len(scored)}
        if not scored:
            return {**out, 'score': 'NA', 'vparty_level': None}
        P = np.array([[r['pair_positions'].get(d, np.nan) for d in self.domains] for r in scored])
        n = np.sum(~np.isnan(P), axis=0)
        active = n > 0
        pos = np.nansum(P, axis=0) / np.maximum(n, 1)
        # evidence weight grows with the number of texts, but slowly
        score, confidence, comp, level = self._summarize(pos, active, self.weights * np.sqrt(np.maximum(n, 1)))
        return {**out, 'score': score, 'confidence': confidence, 'interpretation': interpret(score),
                'vparty_level': level, 'vparty_label': VPARTY_LABELS[level],
                'component_positions': comp,
                'evidence_on_freedoms': any(k in comp for k in FREEDOM_COMPONENTS),
                'evidence_on_defeat': 'defeat' in comp,
                'texts_per_pair': {d: int(n[i]) for i, d in enumerate(self.domains) if active[i]}}


# ============================================================================
# INTERACTIVE ANALYSIS FUNCTIONS
# ============================================================================

def analyze_text(scorer, text):
    result = scorer.score_liberal_illiberal(text)
    print(f"\n{'='*80}\nTEXT: {text}\n{'='*80}")
    if not result['is_relevant']:
        print(f"\n⚠️  NA: {result['interpretation']}  (topic margin {result['topic_margin']:+.2f})")
        return result
    print(f"\n📊 RESULTS:")
    print(f"   Score:         {result['score']:.2f}/10  ({result['interpretation']})")
    print(f"   V-Party level: {result['vparty_level']} - {result['vparty_label']} (uncalibrated)")
    print(f"   Confidence:    {result['confidence']:.3f}   active pairs: {result['n_active']}")
    print(f"   Contradiction: {'YES' if result['contradiction_detected'] else 'NO'}"
          f"   militant-democracy framing: {result['militant_democracy']:.2f}")
    print("   Components:    " + ", ".join(f"{k} {v:+.2f}" for k, v in result['component_positions'].items()))
    for label, key in [("LIBERAL", 'top_liberal_hypotheses'), ("ILLIBERAL", 'top_illiberal_hypotheses')]:
        print(f"\n🔍 TOP {label} EVIDENCE:")
        for hyp in result[key]:
            print(f"   {hyp['score_impact']:+.2f}  p={hyp['probability']:.2f}  {hyp['hypothesis'][:100]}")
    return result


def analyze_batch(scorer, texts):
    print(f"\n{'='*120}\nBATCH ANALYSIS RESULTS\n{'='*120}")
    print(f"{'Text':<70} {'Score':<7} {'Conf':<7} {'Contr':<6} {'Interpretation'}")
    print("-" * 120)
    results = scorer.score_texts(texts)
    for text, r in zip(texts, results):
        shown = text[:67] + "..." if len(text) > 70 else text
        score = f"{r['score']:.2f}" if r['is_relevant'] else "NA"
        print(f"{shown:<70} {score:<7} {r['confidence']:<7.3f} "
              f"{'YES' if r['contradiction_detected'] else 'NO':<6} {r['interpretation']}")
    scored = [r for r in results if r['is_relevant']]
    if scored:
        scores = [r['score'] for r in scored]
        print(f"\n📊 SUMMARY: scored {len(scored)}/{len(results)} | range {min(scores):.2f}-{max(scores):.2f}"
              f" | mean {np.mean(scores):.2f} | mean confidence {np.mean([r['confidence'] for r in scored]):.3f}"
              f" | contradictions {sum(r['contradiction_detected'] for r in scored)}")
    return results


def interactive_mode(scorer):
    print(f"\n{'='*60}\nINTERACTIVE LIBERAL-ILLIBERAL SCORER\n{'='*60}")
    print("Enter text to analyze ('batch' for test texts, 'quit' to exit)")
    while True:
        text = input("\n> ").strip()
        if text.lower() in ['quit', 'exit', 'q']:
            break
        if text.lower() == 'batch':
            analyze_batch(scorer, [
                "We want an inclusive, non-exclusionary Spain, which treats its people well and seeks justice and well-being. A fair country that makes us proud to be Spanish.",
                "Perhaps he ordered the murder of Jaime Garzon did not govern later? Perhaps an important part of society does not applaud and is not afraid of that? If you want to feel fear, that is what you have to fear. If they want hope, what must be defended is the right to difference.",
                "Impunity, disarmament, political indications and corruption have generated and continue to fuel Brazil's biggest problems: violence, state inefficiency and unemployment. As important as doing new things is to undo this criminal structure created by the last governments!",
            ])
        elif text:
            try:
                analyze_text(scorer, text)
            except Exception as e:
                print(f"Error analyzing text: {e}")


if __name__ == "__main__":
    interactive_mode(LiberalIlliberalScorer())
