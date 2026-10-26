"""
Liberal-illiberal scorer, v3: commitment to the democratic process. PAPER
VERSION. The Django app still runs v2 (alternativeLib.py); nothing imports this
module except validate_liberal_illiberal.py.

Scale: 0 = openly autocratic, 5 = no clear position, 10 = fully committed.
NA = the text is not about democratic principles, or takes no position on them.

v3 vs v2 -- from the 200-tweet validation (C1 + GPT gold, li_v2.margins.npz)
---------------------------------------------------------------------------
v2's worst errors were election-denial tweets ("Michigan just refused to
certify the election results! Having courage is a beautiful thing.": gold 1.0,
v2 8.1). Three liberal sides entailed on any mention of ballots or counting:
lawful_means went liberal in 99% of its active cells, source_of_authority in
98%, electoral_conduct in 95%. Talking about elections is not commitment to
them.
  a. DEFEAT IN PRACTICE. Two new pairs, certify_results and claimed_victory,
     target refusing to certify or overturn a result and insisting on a victory
     the count denies. "Claims fraud" is deliberately NOT a hypothesis: an
     opposition alleging fraud in a rigged election is the liberal side.
  b. REWORDED: lawful_means, source_of_authority and electoral_conduct now ask
     for an explicit rejection of force, removability by voters, and defending
     the count from interference, rather than a mention of elections.
  c. MILITANT PROBE. The v2 wording entailed on 87% of tweets (mean sigmoid),
     so the discount applied almost everywhere. It now requires a call to ban
     or punish a group BECAUSE it threatens democracy.
  d. ASYMMETRIC WEIGHT. negative_weight=3: an explicit rejection is not offset
     by general pro-democracy language, which is what "clearly committed"
     implies. With contrast_scale=8 and shrinkage=2 this was the setting
     chosen most often in 2-fold cross-validation on the v2 margins (held-out
     ICC 0.72 vs v1 0.69, a tie within the bootstrap CI). The new pairs (a-c)
     were NOT seen by that tuning and need their own validation run.
  e. DROPPED FROM AGGREGATION: electoral_conduct, lawful_means,
     source_of_authority. Rule fixed before looking at the coders' scores:
     a pair whose active cells go liberal > 95% of the time measures "mentions
     the topic", not position. On the v3 run they were 97%, 99% and 97%
     ("rejects violence" entailed on 98% of tweets). The model still scores
     them, so cached margins keep their columns, but they never count.
     Effect on the 200 tweets: ICC 0.74 -> 0.79, MAE 1.39 -> 1.25, direction
     kappa 0.67 -> 0.74, gold<3 tweets placed >= 5: 12 -> 5 of 38; coverage
     133 -> 126 of 137. Paired vs v1: dICC +0.108 [+0.033, +0.188]. All of
     it in-sample: the 95% cut-off was chosen after the v2 run on the same
     tweets.
  f. SHARED GATE. The gate is now the rule classify_economic.py and
     classify_populism.py use (their shared engine, Aggregator.score): the
     SECOND-highest topic-probe margin must reach topic_margin_min, so two
     probes have to agree, and when negative probes are supplied a text is
     also dropped if max(topic) - max(negative) <= veto_margin. v1-v3 had used
     the single highest probe. This battery has no negative probes yet, so the
     veto is off by default, exactly as the engine behaves on a cache without
     veto columns. Effect on the 200 tweets: admitted 191 -> 180; of the 11
     dropped, 8 were tweets both coders called relevant, so scored coverage of
     those falls 126 -> 118. On the tweets still scored: ICC 0.791 -> 0.786,
     MAE 1.25 -> 1.30, direction kappa 0.74 -> 0.72. Adopted for comparability
     across the three dimensions, not because it scored better here.
  g. SHARED LONG-TEXT RULE. One input per text, truncated "only_first" at
     MAX_LENGTH=512 tokens (premise + hypothesis), the most DeBERTa-v3 accepts
     -- the same cap and rule as the shared engine of classify_economic.py /
     classify_populism.py. v2 and earlier v3 split long texts into chunks
     instead. No validation tweet exceeds 53 words, so no tweet score changes.
  h. SHARED LENGTH GATE. Texts of fewer than min_tokens=5 words (split on
     whitespace) are not scored -- 'too_short', checked before the topic gate,
     exactly as the engine does. A three-word fragment carries no position but
     NLI will still entail from it. On the 200 tweets this drops 3 (2 of them
     coded relevant by both coders).
  i. TWO WORDINGS, AVERAGED. Every pair also has a second wording
     (BATTERY_PARAPHRASE) and each side's margin is the mean of the two, as
     in the shared engine; reliability() scores each wording alone and
     correlates them. Older margin caches have no paraphrase columns and
     score on the main wording alone. Validated 2026-10-26 on the tweets:
     the two wordings correlate 0.944 (Spearman 0.948); ICC with the coders
     0.784 averaged, 0.778 main alone, 0.790 second wording alone.
  j. SHARED UNCERTAINTY: BOOTSTRAP OVER PAIRS, as the shared engine does.
     The active pairs are resampled with replacement `bootstrap` times (200),
     the text is re-scored on each draw with this scorer's own aggregation
     (negative weight, shrinkage), and score_sd / score_lo / score_hi report
     the SD and the 5th-95th percentiles of those scores. dispersion is the
     weighted SD of pair positions. With fewer than 2 active pairs there is
     nothing to resample: the three are NaN, as in the engine. confidence
     (0-1, kept for the interface views.py reads) is 1 - (hi - lo) / 10: the
     share of the 0-10 scale the 90% bootstrap range leaves out; 0 when it
     cannot be estimated. Seeded per text, so a text's interval does not
     depend on what else is in the batch.

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

7.  UNCERTAINTY BY BOOTSTRAP (see j). Superseded the v2 formula
    W / (W + 1) * (1 - SD).

8.  MECHANICS. Inputs are moved to the model's device (v1 left them on the
    CPU). Truncation is "only_first", so a hypothesis is never cut; text past
    the model's 512-token limit is dropped (see g).
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
     "The text supports letting election officials count every valid vote without interference.",
     "The text supports manipulating an election, postponing it, or stopping the count of valid votes."),
    ("franchise", "elections", 0.6,
     "The text supports an equal right to vote for all adult citizens.",
     "The text supports denying the vote to some citizens because of their group, beliefs, or loyalty."),
    # DEFEAT -- "pledge to accept defeat in free and fair elections"
    ("accept_defeat", "defeat", 1.0,
     "The text commits to accepting election results and handing over power peacefully after a defeat.",
     "The text refuses to accept an election result or to hand over power after losing."),
    ("certify_results", "defeat", 1.0,
     "The text supports certifying the official election result, whoever wins.",
     "The text urges officials to refuse to certify, overturn, or ignore an election result."),
    ("claimed_victory", "defeat", 0.8,
     "The text concedes an election defeat or congratulates the winner.",
     "The text insists its side really won an election despite the official result."),
    ("lawful_means", "defeat", 0.8,
     "The text rejects violence, coups, or force as ways to gain or keep power.",
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
     "The text holds that rulers are legitimate only if voters can freely vote them out.",
     "The text holds that the right to govern comes from religion, ideology, or revolution rather than elections."),
    ("constitutional_order", "regime", 0.6,
     "The text defends constitutional rules that limit how power is won and used.",
     "The text supports suspending the constitution or ruling by decree."),
]

# One-sided on the validation tweets (liberal in > 95% of active cells). See e.
DROP_DOMAINS = ("electoral_conduct", "lawful_means", "source_of_authority")

# Second wording of every pair: same domains, order, components and weights,
# every clause restated, only vocabulary and verb frame changed. With
# average_paraphrase=True (the default) each side's margin is the mean over the
# two wordings, as in the shared engine (classify_economic.py's BATTERY_PARAPHRASE);
# reliability() scores each wording alone and correlates them.
BATTERY_PARAPHRASE = [
    ("multiparty_elections", "elections", 1.0,
     "The text wants governments to be selected in free and fair elections in which rival parties compete.",
     "The text favours selecting or retaining a government without holding free, competitive elections."),
    ("electoral_conduct", "elections", 0.7,
     "The text wants election officials to be left free to count every valid vote without anyone interfering.",
     "The text favours rigging an election, delaying it, or halting the counting of valid votes."),
    ("franchise", "elections", 0.6,
     "The text holds that every adult citizen should have the same right to vote.",
     "The text favours stripping some citizens of the vote on account of their group, beliefs, or loyalty."),
    ("accept_defeat", "defeat", 1.0,
     "The text pledges to respect election results and to give up power peacefully after losing.",
     "The text rejects an election result or will not give up power after a defeat."),
    ("certify_results", "defeat", 1.0,
     "The text wants the official election result to be certified, regardless of who won.",
     "The text calls on officials not to certify an election result, or to overturn or disregard it."),
    ("claimed_victory", "defeat", 0.8,
     "The text admits losing an election or offers congratulations to the winner.",
     "The text claims that its side truly won an election even though the official result says otherwise."),
    ("lawful_means", "defeat", 0.8,
     "The text opposes using violence, coups, or force to win or hold on to power.",
     "The text favours seizing or holding power through force, revolution, or a coup."),
    ("free_speech", "speech", 1.0,
     "The text upholds free expression, also for people who criticize the government.",
     "The text favours penalizing or silencing people for expressing political views peacefully."),
    ("criticism", "speech", 0.8,
     "The text upholds citizens' right to criticize the people who govern them.",
     "The text presents criticizing the government or its leader as illegitimate or as treason."),
    ("press_freedom", "media", 1.0,
     "The text upholds a free press that the government does not control.",
     "The text favours the government censoring or controlling the press."),
    ("press_role", "media", 0.6,
     "The text upholds journalists' job of holding the powerful accountable.",
     "The text presents independent reporters as enemies that have to be stopped."),
    ("assembly", "assembly", 1.0,
     "The text upholds people's right to gather and demonstrate peacefully.",
     "The text favours prohibiting or dispersing peaceful demonstrations."),
    ("protest_force", "assembly", 0.7,
     "The text denounces using force on people who are protesting peacefully.",
     "The text defends using force on people who are protesting peacefully."),
    ("opposition_parties", "association", 1.0,
     "The text treats opposition parties as rightful rivals in the contest for power.",
     "The text favours outlawing or getting rid of opposition parties."),
    ("civil_society", "association", 0.8,
     "The text upholds citizens' freedom to set up independent associations, trade unions, and parties.",
     "The text favours prohibiting or taking control of independent associations and trade unions."),
    ("one_party", "regime", 1.0,
     "The text favours a political system where multiple parties contend for power.",
     "The text favours government by one party or one leader with no political competition."),
    ("source_of_authority", "regime", 1.0,
     "The text maintains that those who govern are legitimate only as long as voters are free to remove them at the polls.",
     "The text maintains that the right to rule derives from religion, ideology, or revolution instead of from elections."),
    ("constitutional_order", "regime", 0.6,
     "The text upholds constitutional rules that constrain how power is gained and exercised.",
     "The text favours setting aside the constitution or governing by decree."),
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
    "The text calls for banning or punishing a group because it uses violence "
    "or seeks to abolish democracy.")
# Freedom pairs whose ILLIBERAL side a defensive-democracy framing can explain.
DEFENSIVE_TARGETS = {"free_speech", "criticism", "press_freedom", "assembly",
                     "protest_force", "civil_society"}

# Same thresholds as the mock scorer in views.py, so the label never depends
# on which code path produced the score.
INTERPRETATION_BANDS = [(3.0, "Strong Illiberal"), (4.5, "Illiberal"), (5.5, "Moderate"),
                        (7.0, "Liberal"), (np.inf, "Strong Liberal")]
VPARTY_LABELS = ["Not at all committed", "Not committed", "Weakly committed",
                 "Committed", "Fully committed"]

# Long texts: the same cap and rule as the shared engine of classify_economic.py /
# classify_populism.py. One input per text; "only_first" truncation clips the
# text, never the hypothesis. 512 tokens is the most DeBERTa-v3 accepts.
# test_truncation.py checks the two stay equal.
MAX_LENGTH = 512


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def interpret(score):
    for upper, label in INTERPRETATION_BANDS:
        if score < upper:
            return label


# ============================================================================
# SCORER
# ============================================================================

class LiberalIlliberalScorer:
    """
    Parameters
    ----------
    topic_margin_min : float
        Gate, the same rule as the shared engine of classify_economic.py and
        classify_populism.py: the SECOND-highest topic-probe margin must reach
        this, so two probes have to agree. 0.0 = "more likely about it than not".
    veto_hypotheses, veto_margin
        Negative probes for off-topic rhetoric, as in the engine. With any
        supplied, a text is also dropped if max(topic) - max(negative) <=
        veto_margin. Empty by default (no negative probes written yet).
    average_paraphrase, paraphrase_battery
        Score every pair under a second wording too and average the two margins,
        as the shared engine does. Margin caches without the paraphrase columns
        fall back to the main wording (result key 'paraphrased' says which).
    bootstrap, seed : int
        Draws and seed for the bootstrap over active pairs (score_sd,
        score_lo, score_hi, confidence), as in the shared engine.
    min_tokens : int
        Length gate, as in the engine: texts with fewer whitespace-separated
        words are returned as NA ('too_short') and never scored. 5 = only texts
        of more than four words are analysed.
    active_margin : float
        A pair is active if either side reaches this margin.
    contrast_scale : float
        Logit units per unit of position: pos = tanh(c / contrast_scale).
        At 8, a contrast of 8 logits (sides at +8 / -8) is a position of 0.76.
    shrinkage : float
        Pseudo-weight of neutral evidence: score = 5 + 5 * sum(w*pos) / (W + k0).
        A text resting on one weight-1 pair cannot reach the end of the scale.
    negative_weight : float
        Multiplies the weight of a pair whose position is illiberal. 1 = the
        two sides count the same. 3 = one explicit rejection outweighs three
        equally strong endorsements.
    defensive_discount : float
        Largest share of a freedom pair's illiberal contrast that the
        militant-democracy framing can cancel.
    drop_domains : iterable of str
        Pairs the model still scores but the aggregation ignores.
    margin_fn : callable(premises, hypotheses) -> np.ndarray, optional
        Replaces the model. Used for tests and for re-scoring cached margins.
    """

    def __init__(self, model_name=MODEL_NAME, topic_margin_min=0.0, active_margin=0.0,
                 contrast_scale=8.0, shrinkage=2.0, negative_weight=3.0, defensive_discount=0.65,
                 battery=BATTERY, topic_hypotheses=TOPIC_HYPOTHESES, drop_domains=DROP_DOMAINS,
                 veto_hypotheses=(), veto_margin=0.0, min_tokens=5,
                 average_paraphrase=True, paraphrase_battery=BATTERY_PARAPHRASE,
                 bootstrap=200, seed=0,
                 batch_size=16, max_length=MAX_LENGTH, margin_fn=None):
        self.battery = list(battery)
        self.topic_hypotheses = list(topic_hypotheses)
        self.domains = [b[0] for b in self.battery]
        self.components = [b[1] for b in self.battery]
        self.weights = np.array([b[2] for b in self.battery], dtype=float)
        self.topic_margin_min = float(topic_margin_min)
        self.veto_hypotheses = list(veto_hypotheses)
        self.veto_margin = float(veto_margin)
        self.min_tokens = int(min_tokens)
        self.bootstrap = int(bootstrap)
        self.seed = int(seed)
        self.paraphrase_battery = list(paraphrase_battery) if average_paraphrase and paraphrase_battery else []
        if self.paraphrase_battery and (
                [(b[0], b[1], b[2]) for b in self.paraphrase_battery] != [(b[0], b[1], b[2]) for b in self.battery]):
            raise ValueError("paraphrase_battery must have the same domains, components "
                             "and weights, in the same order, as battery")
        self.active_margin = float(active_margin)
        self.contrast_scale = float(contrast_scale)
        self.shrinkage = float(shrinkage)
        self.negative_weight = float(negative_weight)
        self.defensive_discount = float(defensive_discount)
        self.defensive_mask = np.array([d in DEFENSIVE_TARGETS for d in self.domains])
        unknown = set(drop_domains) - set(self.domains)
        if unknown:
            raise ValueError(f"drop_domains not in the battery: {sorted(unknown)}")
        self.keep = np.array([d not in set(drop_domains) for d in self.domains])
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)

        # Column layout of the margin matrix:
        #   topics | militant probe | L, I, L, I, ... | veto | paraphrase L, I, ...
        # New blocks go last so caches written without them keep their layout.
        self.hypotheses = (self.topic_hypotheses + [MILITANT_DEMOCRACY_HYPOTHESIS]
                           + [h for b in self.battery for h in (b[3], b[4])]
                           + self.veto_hypotheses
                           + [h for b in self.paraphrase_battery for h in (b[3], b[4])])
        self.n_topic = len(self.topic_hypotheses)
        self._bat = slice(self.n_topic + 1, self.n_topic + 1 + 2 * len(self.battery))
        self._veto = slice(self._bat.stop, self._bat.stop + len(self.veto_hypotheses))
        self._para = (slice(self._veto.stop, self._veto.stop + 2 * len(self.battery))
                      if self.paraphrase_battery else None)

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
        self._lock = cache.inference_lock()   # the shared tokenizer is not thread-safe
        self.entailment_idx = self._find_entailment_index()

    def _find_entailment_index(self):
        config = self.model.config
        if getattr(config, 'label2id', None):
            for label, idx in config.label2id.items():
                if label.lower() in ['entailment', 'entail']:
                    return idx
        return 0

    def _model_margins(self, premises, hypotheses):
        with self._lock:
            return self._model_margins_unlocked(premises, hypotheses)

    def _model_margins_unlocked(self, premises, hypotheses):
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
        """(n_texts, n_hypotheses) margins, one input per text (truncated at max_length)."""
        texts = ["" if t is None else str(t) for t in texts]
        nh = len(self.hypotheses)
        prem = [t for t in texts for _ in range(nh)]
        hyp = self.hypotheses * len(texts)
        return np.asarray(self._margin_fn(prem, hyp), dtype=float).reshape(len(texts), nh)

    # ------------------------------------------------------------ aggregation
    def _battery(self, row, wording="average"):
        """Battery margins for one row: 'main', 'paraphrase', or their 'average'.
        Rows without paraphrase columns (older caches) give the main wording."""
        main = row[self._bat]
        if self._para is None or row.size < self._para.stop or wording == "main":
            return main
        para = row[self._para]
        return para if wording == "paraphrase" else (main + para) / 2.0

    def has_paraphrase(self, row):
        return self._para is not None and np.asarray(row).size >= self._para.stop

    def pair_positions(self, row, wording="average"):
        """Margins for one text -> (positions, active mask, militant weight)."""
        bat = self._battery(row, wording)
        lib, ill = bat[0::2], bat[1::2]
        active = (np.fmax(lib, ill) >= self.active_margin) & self.keep
        c = (lib - ill) / 2.0
        md = float(_sigmoid(row[self.n_topic]))
        discount = np.where(self.defensive_mask & (c < 0), 1.0 - self.defensive_discount * md, 1.0)
        pos = np.tanh(c * discount / self.contrast_scale)
        return np.where(active, pos, 0.0), active, md

    def _effective_weights(self, pos, active, weights):
        return weights * active * np.where(pos < 0, self.negative_weight, 1.0)

    def _uncertainty(self, pos, active, w):
        """Bootstrap over the active pairs, the shared engine's method: resample
        pairs with replacement, re-score each draw, report SD and 5-95%."""
        idx = np.flatnonzero(active)
        nan = {'score_sd': np.nan, 'score_lo': np.nan, 'score_hi': np.nan, 'confidence': 0.0}
        if idx.size < 2 or not self.bootstrap:
            return nan
        rng = np.random.default_rng(self.seed)
        draw = rng.integers(0, idx.size, size=(self.bootstrap, idx.size))
        bw, bp = w[idx][draw], pos[idx][draw]
        bs = np.clip(5.0 + 5.0 * (bw * bp).sum(1) / (bw.sum(1) + self.shrinkage), 0.0, 10.0)
        lo, hi = np.percentile(bs, [5, 95])
        return {'score_sd': float(bs.std()), 'score_lo': float(lo), 'score_hi': float(hi),
                'confidence': float(max(0.0, 1.0 - (hi - lo) / 10.0))}

    def _summarize(self, pos, active, weights):
        """Score, uncertainty, component means and V-Party level from pair positions."""
        w = self._effective_weights(pos, active, weights)
        W = float(w.sum())
        mean_pos = float((w * pos).sum() / W)
        score = float(np.clip(5.0 + 5.0 * (w * pos).sum() / (W + self.shrinkage), 0.0, 10.0))
        unc = {**self._uncertainty(pos, active, w),
               'dispersion': float(np.sqrt((w * (pos - mean_pos) ** 2).sum() / W))}
        comp = {}
        for k in COMPONENTS:
            sel = np.array([c == k for c in self.components]) & active
            if sel.any():
                comp[k] = float((weights[sel] * pos[sel]).sum() / weights[sel].sum())
        return score, unc, comp, self._vparty_level(score, comp)

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

    def gate(self, row):
        """The engine's gate. Returns (admitted, reason, topic margin).

        reason is 'admitted', 'topic' (fewer than two probes agree) or 'veto'
        (negative probes outweigh the strongest topic probe).
        """
        t = np.sort(np.nan_to_num(row[:self.n_topic], nan=-np.inf))
        topic = float(t[-2] if t.size >= 2 else t[-1])
        if topic < self.topic_margin_min:
            return False, "topic", topic
        if self.veto_hypotheses:
            veto = float(np.nanmax(row[self._veto]))
            if float(t[-1]) - veto <= self.veto_margin:
                return False, "veto", topic
        return True, "admitted", topic

    def score_from_margins(self, row, text=None, wording="average"):
        """Score one text from its margins. Pass the text so the length gate can
        run; with text=None (re-scoring margins whose text is not at hand) the
        length gate is skipped. wording='main' or 'paraphrase' scores one
        wording alone (used by reliability())."""
        row = np.asarray(row, dtype=float)
        admitted, reason, topic_margin = self.gate(row)
        if text is not None and len(str(text).split()) < self.min_tokens:
            admitted, reason = False, "too_short"
        base = {'text': "" if text is None else text, 'topic_probability': float(_sigmoid(topic_margin)),
                'topic_margin': topic_margin, 'gate_reason': reason, 'margins': row.tolist(),
                'paraphrased': bool(wording == "average" and self.has_paraphrase(row))}
        if reason == "too_short":
            return {**base, 'score': 'NA', 'confidence': 0.0, 'score_sd': np.nan, 'score_lo': np.nan,
                    'score_hi': np.nan, 'dispersion': np.nan, 'is_relevant': False,
                    'contradiction_detected': False, 'interpretation': 'Too short to score'}
        if not admitted:
            return {**base, 'score': 'NA', 'confidence': 0.0, 'score_sd': np.nan, 'score_lo': np.nan,
                    'score_hi': np.nan, 'dispersion': np.nan, 'is_relevant': False,
                    'contradiction_detected': False,
                    'interpretation': 'Not about democratic principles'}

        pos, active, md = self.pair_positions(row, wording)
        if not active.any():
            return {**base, 'score': 'NA', 'confidence': 0.0, 'score_sd': np.nan, 'score_lo': np.nan,
                    'score_hi': np.nan, 'dispersion': np.nan, 'is_relevant': False,
                    'contradiction_detected': False,
                    'interpretation': 'No clear position on democratic principles'}

        score, unc, comp, level = self._summarize(pos, active, self.weights)
        w_eff = self._effective_weights(pos, active, self.weights)
        W = float(w_eff.sum())
        bat = self._battery(row, wording)
        top_lib, top_ill = [], []
        for p in np.flatnonzero(active):
            side = 0 if pos[p] >= 0 else 1
            item = {'hypothesis': self.battery[p][3 + side],
                    'probability': float(_sigmoid(bat[2 * p + side])),
                    'score_impact': float(5.0 * w_eff[p] * abs(pos[p]) / (W + self.shrinkage)),
                    'domain': self.domains[p]}
            (top_lib if side == 0 else top_ill).append(item)
        top_lib.sort(key=lambda d: -d['score_impact'])
        top_ill.sort(key=lambda d: -d['score_impact'])

        return {**base,
                'score': score,
                **unc,
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

    def reliability(self, M, texts=None):
        """Score each wording alone and correlate: the ceiling for any downstream
        correlation (same check as the engine's reliability()). M must hold the
        paraphrase columns."""
        if self._para is None or np.asarray(M).shape[1] < self._para.stop:
            raise ValueError("these margins have no paraphrase columns")
        texts = [None] * len(M) if texts is None else list(texts)
        sc = {w: np.array([r["score"] if r["is_relevant"] else np.nan for r in
                           (self.score_from_margins(m, t, wording=w) for m, t in zip(M, texts))], float)
              for w in ("main", "paraphrase")}
        ok = np.isfinite(sc["main"]) & np.isfinite(sc["paraphrase"])
        a, b = sc["main"][ok], sc["paraphrase"][ok]
        rank = lambda x: np.argsort(np.argsort(x))
        return {"n": int(ok.sum()), "pearson": float(np.corrcoef(a, b)[0, 1]),
                "spearman": float(np.corrcoef(rank(a), rank(b))[0, 1]),
                "mean_abs_diff": float(np.abs(a - b).mean()), **sc}

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
        score, unc, comp, level = self._summarize(pos, active, self.weights * np.sqrt(np.maximum(n, 1)))
        return {**out, 'score': score, **unc, 'interpretation': interpret(score),
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
    print(f"   Bootstrap:     SD {result['score_sd']:.2f}, 90% range {result['score_lo']:.2f}-{result['score_hi']:.2f}"
          f"   confidence {result['confidence']:.3f}   active pairs: {result['n_active']}")
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
