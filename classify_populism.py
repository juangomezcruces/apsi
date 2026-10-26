"""
Populism / political-authority scoring of the translated PopBERT sample.

Scale: 0 = pluralist, 5 = mixed, 10 = populist. NaN = failed the gate.
Same job as v1, restructured so the measurement is defensible.

WHAT CHANGED AND WHY
--------------------
1.  TWO PHASES.  Phase 1 (GPU) computes and caches the raw entailment MARGIN
    for every text x hypothesis.  Phase 2 (CPU) turns margins into scores.
    Every knob that matters lives in phase 2, so re-scoring under a different
    aggregation costs seconds, not another GPU run.  The margin cache is ~4 MB
    for the whole PopBERT sample.

2.  MARGINS, NOT PROBABILITIES.  Political_DEBATE is a TWO-label model
    (entailment / not_entailment), so p = sigmoid(margin) and the margin is the
    sufficient statistic.  The softmax was throwing away exactly the magnitude
    v1 needed, which is what the saturation diagnostics were showing.

3.  PAIRED BATTERY.  Every hypothesis now has an explicit mirror image, and the
    unit of evidence is the CONTRAST (m_right - m_left) / 2 for each pair.
    This cancels per-template offsets, and it makes the left and right sides
    balanced by construction: v1 had 19 left vs 21 right hypotheses and
    weight mass 15.80 vs 18.45, which gave the right end of the scale about
    0.22 points more headroom than the left.

4.  NO k_eff.  In contrast space an inactive pair contributes exactly 0, which
    IS the neutral value, so a plain weighted mean over the ACTIVE pairs is
    coherent.  v1's `k_eff = max(4, n_fired + 2)` counted fires on both sides
    and used the result as the top-k denominator for each side separately, so
    (a) right-side firing changed the left-side magnitude and (b) texts that
    tripped more hypotheses read as more extreme regardless of intensity.

5.  ONE THRESHOLD, IN INTERPRETABLE UNITS.  v1 had `thr=0.15` gating whether
    to score, `prob_floor=0.25` zeroing contributions, and `topic_threshold=0.6`
    gating relevance -- three magic numbers on a saturated probability scale,
    where a hypothesis at p=0.20 raised v1's denominator while contributing
    zero.  Here a pair is ACTIVE when at least one side has margin >= 0
    (i.e. the model calls it more likely than not).

6.  BOOTSTRAP CONFIDENCE.  v1's confidence formula was fed weighted values
    rather than probabilities, ignored its third argument, and carried six
    unmotivated constants.  Confidence is now the SD of the score under
    resampling of the battery, plus the weighted dispersion of positions
    across active pairs.  Both are things you can put in a table.

7.  DIAGNOSTICS THAT CAN FAIL.  `diagnose()` reports the length confound,
    per-pair activation rates, a leave-one-pair-out jackknife, and (given a
    party/group column) group means with bootstrap intervals.  `reliability()`
    re-scores with a paraphrased battery and correlates -- that correlation is
    the reliability ceiling for anything built on top of this.

MODEL NOTE.  vocab_size 128100 confirms English DeBERTa-v3-large, not mDeBERTa.
Scoring the German originals (`text_col="text"`) with this checkpoint is not
valid; the loader warns if the input looks German.

Scale: 0 = far left, 5 = centre, 10 = far right.  NaN = failed the topic gate.
"""

import argparse
import glob
import hashlib
import json
import logging
import os
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd

if os.path.isdir("/kaggle") and "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = "/kaggle/temp/hf"
    os.environ.setdefault("TRANSFORMERS_CACHE", "/kaggle/temp/hf")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

try:
    import torch
except ImportError:                                   # phase 2 needs no torch
    torch = None

# v1 silenced every warning at import time, which would also have hidden the
# fp16 overflow and truncation warnings that matter here.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S", force=True, stream=sys.stdout,
)
logger = logging.getLogger("lr_scorer")

try:
    from tqdm.auto import tqdm
except Exception:                                     # pragma: no cover
    def tqdm(x, **kw):
        return x

ON_KAGGLE = os.path.isdir("/kaggle/working")


# =============================================================================
# DIMENSION: POPULISM / POLITICAL AUTHORITY   (0 = pluralist, 10 = populist)
# =============================================================================

# =============================================================================
# CONFIG  - edit these, then re-run the cell
# =============================================================================
CONFIG = dict(
    input_csv=None,                   # None = auto-discover under /kaggle/input
    input_pattern="*translat*.csv",
    output_csv="/kaggle/working/popbert_pop_v3.csv",
    margins_path=None,                # None = output_csv with a .margins.npz suffix
    text_col="text_en",
    id_col="id",
    group_col=None,                   # e.g. "party" - enables group diagnostics
    suffix="_en",
    model="mlburnham/Political_DEBATE_large_v1.0",
    revision=None,                    # pin a commit sha for reproducibility
    device=None,
    fp16=None,
    batch_size=None,                  # None = aligned to the battery size
    text_chunk=128,
    max_length=512,                   # the model's limit; longer text is truncated (only_first)
    # --- phase 1 ----------------------------------------------------------
    gate="post",                      # post = score everything, gate on CPU
                                      # pre  = skip the battery for off-topic text
    # --- phase 2 (all cheap to change; see rescore()) ----------------------
    # BACK TO 0.0, AND IT MUST BE RE-DERIVED. 12.0 was the right answer for the
    # 19-probe gate (precision 0.602 -> 0.809, partial r 0.394 -> 0.480, held out
    # at +0.082 [+0.029,+0.135]) -- but that was measured on a DIFFERENT probe
    # set. The five people-centric probes added below shift the margin
    # distribution for exactly the sentences the old gate was dropping, so a
    # threshold calibrated on the old scale carries no guarantee on the new one.
    # Re-run the sweep on the new cache before shipping a number; the procedure
    # is in the docstring of score_populism().
    topic_margin_min=0.0,             # on the 2nd-highest topic margin
    active_margin=0.0,                # a pair is active if either side >= this
    # "center" was the original default on the theory that per-pair template
    # bias should be removed. Measured on the PopBERT sample it removes SIGNAL:
    # r(score, left_wing_n_true) = -0.369 with "none" vs -0.299 with "center",
    # and the two rank texts near-identically (r=0.963), so centring was moving
    # the scale rather than improving the ordering. Re-check on your own corpus
    # with compare_settings(..., label_col=...).
    standardize="none",               # none | center | z
    contrast_scale="auto",            # "auto" = SD of active contrasts, or a float
    active_only=True,                 # False = divide by the whole battery
    bootstrap=200,
    seed=0,
    min_tokens=5,                     # fragments below this are not scored
    min_active=2,                     # a single active pair is not evidence
    # stance_mode="gate" was tested on the PopBERT sample and made things
    # worse: it cost 39% of scored rows, left Cohen's d against the human
    # left_wing label at -0.60 vs -0.62, and raised the faction R^2 from 0.201
    # to 0.246. Left at "off"; re-test with compare_settings on your own data.
    stance_mode="off",                # off | gate | shrink
    use_veto=True,                    # subtract the negative topic probes
    veto_margin=0.0,                  # admit when max(positive) - max(negative) > this
    average_paraphrase=True,          # score each pair with two wordings and average
    balance_weight=0.0,              # down-weight one-sided pairs; try 0.5
    drop_domains=(),                  # e.g. ("planning_vs_market","state_role")
    stance_min=0.0,                   # gate: keep rows arguing a position
    stance_scale=4.0,                 # shrink: logits per unit of shrinkage
    # --- validation: fill these in and the diagnostics do the work ----------
    label_col=None,                   # e.g. "left_wing_n_true" for PopBERT
    anchor=None,                      # no standard external anchor for this
                                      # dimension; faction means are the check
    # --- run control ------------------------------------------------------
    limit=50,                         # SET TO None FOR THE FULL RUN
    sample_limit=True,                # take a random sample, not the first N
    checkpoint_every=2048,
    resume=True,
    fp16_check=True,                  # compare fp16 vs fp32 on a small sample
    run_diagnostics=True,
)
# Approximate CHES 2019 economic left-right (lrecon) for the Bundestag parties,
# keyed to the PopBERT "faction" values. Pass as anchor= to have the diagnostics
# score the recovered ordering against it. Replace with the exact figures from
# your own CHES extract before reporting anything from it.

# or rescore() as battery=, topic_hypotheses=, veto_hypotheses=, bands=.
#
# Ported from a standalone v1-style scorer, whose 23 hypotheses were an unpaired
# list (12 populist / 11 pluralist, weight 10.15 vs 8.65 -- worth about 0.46
# points of extra headroom at the populist end) and whose 25 "topic" hypotheses
# were not topic probes at all but position statements. Both sets are folded in
# here: the position statements became pairs, and a real topic gate was written.
# MEASURED 2026-09-10: three pairs were rewritten to fire more often and all
# three got WORSE, so the originals below stand. people_homogeneity went from
# validity +0.733 (the best in the battery, 6.4% activation) to -0.128 at 8.0%;
# courts_and_majority 1.9% -> 10.2% activation for validity +0.027;
# legitimacy_source -0.063 -> +0.000. Overall r fell +0.517 -> +0.502.
# The lesson: the long conjunctions are hard to entail AND are where the signal
# is. "unified and virtuous whole BETRAYED BY A CORRUPT ELITE" carries the
# anti-elitism the label measures; trimming it for activation removed it.
# REVISED POPULIST SIDES -- three of the supplied 25 statements were reworded.
# Measured on the US 2010-2018 set (sentence-level labels, 1,009 pluralist rows),
# each described a FEATURE populism has without the RELATION that makes it
# populist, so each fired on ordinary democratic speech:
#
#   people_composition  caught "we're all on the same boat"        (solidarity)
#   deliberation_pace   caught "to dither rather than decide"      (urgency)
#   majority_limits     caught "we cannot wait another year"       (determination)
#
# majority_limits alone led on 27.2% of pluralist-scored-populist errors against
# 8.6% of correct pluralist rows. The revisions keep each construct and restore
# the opposing party the original left implicit. Originals, for reversion:
#
#   people_composition: "This text portrays the people as a unified moral
#                        community with shared goals."
#   deliberation_pace:  "This text emphasizes decisive leadership over
#                        deliberation."
#   majority_limits:    "This text portrays the will of the majority as something
#                        that should not be constrained by special interests."
#
# MEASURED ON THE US SET, ROUND 2 (hash 1f1016d5725b, gated run, n=1332):
#   people_composition  over-firing lift +6.2 -> +0.4   KEPT, essentially fixed
#   majority_limits     over-firing lift +18.6 -> +10.1 KEPT, halved
#   deliberation_pace   over-firing lift +5.9 -> +9.5   REVERTED, the revision
#                       fired on ordinary urgency more readily than the original
#   ordinary_origin     led on 10.4% of MISSED populist vs 4.5% of caught --
#                       right texts, insufficient push. Sharpened to name the
#                       contrast with professional politicians rather than
#                       state a biographical fact.
#   local_people        +8.9 on pluralist errors, ~0 populist recall. REMOVED:
#                       invoking "the people of this place" is what every
#                       regional politician does, populist or not.
#
# ROUND 3 -- FOUR PAIRS REMOVED after a holdout-validated domain_sweep on the US
# 2010-2018 set (populist vs pluralist, n=2424). Dropping them raised the
# confirm-half correlation from r=+0.590 to +0.659 (+0.068, split-to-split
# sd 0.020), and on the full corpus AUC 0.835 -> 0.848, paired bootstrap
# +0.0127 [+0.008, +0.018], P(better)=1.00.
#
#   REMOVED            why                               statements lost
#   people_character   added round 2, does not earn it   (both mine)
#   ordinary_origin    added round 2, does not earn it   (both mine)
#   legitimacy_source  sweep gain +0.018                 2 SUPPLIED statements
#   majority_limits    sweep gain +0.016                 1 supplied + 1 revised
#
# The last two cost four of the original 25 statements, including both halves
# of group 1 (legitimacy). They are preserved verbatim below so any of this is
# reversible by pasting the tuple back into BATTERY:
#
#   ("legitimacy_source", 1.00,
#    "This text supports the idea that political authority is legitimate only when it follows constitutional rules and legal procedures.",
#    "This text portrays the direct will of the people as the primary source of political legitimacy."),
#   ("majority_limits", 1.00,
#    "This text supports limiting majority power in order to protect minority rights.",
#    "This text portrays the will of the majority as something that courts, minorities, or institutions should not be able to block."),
#      -- original supplied populist side, before the round-2 revision:
#      "This text portrays the will of the majority as something that should not be constrained by special interests."
#   ("people_character", 1.00,
#    "This text treats citizens as one constituency among several whose claims must be weighed.",
#    "This text praises the good sense, decency, or hard work of ordinary people."),
#   ("ordinary_origin", 1.00,
#    "This text presents the speaker's authority as resting on expertise, office, or record.",
#    "This text presents the speaker as an ordinary person sharing the background and experience of the audience."),
#
# NOT YET CONFIRMED ACROSS SEEDS. The four were selected on one split (seed=0);
# the recommended check -- rerunning domain_sweep at seeds 1 and 2 to see whether
# the same four recur -- has not been run. If the list reshuffles, restore the
# pairs that do not appear in every seed.
BATTERY = [
    # (domain, weight, PLURALIST side -> score 0, POPULIST side -> score 10)
    #
    # The 25 statements are used VERBATIM as supplied. They arrive as standalone
    # items, not mirror pairs, so they are paired here: the engine's unit of
    # evidence is the contrast (populist margin - pluralist margin) / 2, which
    # cancels the per-template offset that made v1's raw probabilities unusable.
    # Pairing also fixes the side balance. Unpaired, the set is 12 pluralist
    # against 13 populist, which would hand the populist end more headroom --
    # exactly the defect measured in the very first version of this pipeline.
    #
    # The five thematic groups are preserved as the pairing frame: each group's
    # own statements are paired within it wherever a genuine opposition exists.
    # One statement -- "expert knowledge ... even when unpopular" -- anchors two
    # pairs, because the counts are odd (12 vs 13) and it is the true opposite of
    # both "popular opinion is superior to expert judgment" and "leaders lose
    # legitimacy when they ignore popular demands". That is the only reuse, and
    # it is noted rather than hidden: the two pairs share a baseline and are
    # therefore slightly correlated.

    # -- group 1: legitimacy of political authority ------------------------
    ("validating_authority", 1.00,
     "This text emphasizes the role of courts and independent agencies in validating political authority.",
     "This text portrays charismatic leadership as a sufficient basis for political legitimacy."),

    # -- group 2: role of institutions -------------------------------------
    ("institutions_purpose", 1.00,
     "This text portrays political institutions as protecting citizens from arbitrary and abusive power.",
     "This text depicts elites as manipulating institutions for personal gain."),
    ("institutional_change", 1.00,
     "This text supports reforming institutions without weakening or bypassing them.",
     "This text depicts political institutions as outdated structures that primarily serve elite interests."),
    ("unelected_bodies", 1.00,
     "This text emphasizes that strong institutions are more important than strong leaders.",
     "This text portrays unelected institutions as unjust obstacles to democratic decision-making."),

    # -- group 3: decision-making style ------------------------------------
    ("expertise", 1.00,
     "This text promotes expert knowledge as a legitimate basis for policymaking, even when unpopular.",
     "This text portrays popular opinion as superior to expert or technocratic judgment."),
    ("responsiveness", 1.00,
     "This text promotes expert knowledge as a legitimate basis for policymaking, even when unpopular.",
     "This text suggests that elected leaders lose legitimacy when they ignore popular demands."),
    ("deliberation_pace", 1.00,
     "This text frames slow decision-making as a necessary feature of democracy.",
     "This text frames deliberation and procedure as delay tactics that frustrate what needs to be done."),
    ("decision_mode", 1.00,
     "This text supports political decision-making through negotiation and compromise.",
     "This text suggests that leaders can clearly identify and represent the true will of the people."),

    # -- group 4: view of opposition and elites ----------------------------
    ("opposition", 1.00,
     "This text portrays political opponents as legitimate actors with valid interests.",
     "This text frames political conflict as a struggle between ordinary citizens and corrupt elites."),
    ("dissent", 1.00,
     "This text frames criticism and dissent as strengthening democracy.",
     "This text portrays those who oppose the people\u2019s will as enemies of democracy."),

    # -- group 5: "the people" ---------------------------------------------
    ("people_composition", 1.00,
     "This text portrays society as composed of diverse groups with competing interests.",
     "This text portrays the people as a unified moral community whose shared will is obstructed by a self-serving minority."),
    # -- group 6: economic power  (ADDED -- not in the supplied 25) ---------
    # 47% of the gate's false negatives used money vocabulary and no
    # institutional vocabulary, against 16% of true positives. None of the
    # supplied statements mentions money, banks, tax, wages or the cost of
    # living, so populist claims in the economic register -- cum-ex, Luxembourg
    # tax deals, the wage gap, Hartz IV -- landed at topic margins near -15,
    # meaning nothing fired at all. These four pairs cover that register in the
    # same voice as the originals.
    ("economic_elites", 1.00,
     "This text treats businesses, banks, and wealthy individuals as ordinary participants in economic life.",
     "This text depicts banks, corporations, or the very wealthy as a privileged group that rigs the rules in its favour."),
    ("who_pays", 1.00,
     "This text treats taxation and public spending as ordinary questions of policy design.",
     "This text frames taxation as a contest between ordinary people who pay and privileged groups who escape paying."),
    ("material_grievance", 1.00,
     "This text discusses wages, pensions, and the cost of living as technical policy parameters.",
     "This text presents the material hardship of ordinary people as a grievance against those in charge."),
    ("deservingness", 1.00,
     "This text treats competing economic groups as having equally legitimate claims.",
     "This text portrays hard-working people as contributing more than they receive in return."),

    # -- group 7: speaking for the people  (ADDED) -------------------------
    # The people-centrism half is the weak one (AUC 0.669 split, 0.517 merged)
    # and it rested on four pairs, two of which fired on under 8% of texts.
    # These three target the forms of people-centrism that carry no institution
    # and no adversary, which is the class the gate and battery both missed.
    ("speaking_for", 1.00,
     "This text acknowledges that the public is divided on the question at hand.",
     "This text claims to know and to speak for what the people really want."),
    ("popular_mood", 1.00,
     "This text defends a position it admits is currently unpopular.",
     "This text treats widespread public anger or frustration as justification for its position."),

    # -- group 8: authenticity credentialing  (ADDED) ----------------------
    # 343 populist sentences scored pluralist, and the missed ones were not weak
    # populism but a device with no statement at all: the speaker presenting
    # himself AS one of the people rather than speaking FOR them --
    # "I was born in Chicago, my grandparents are dairy farmers in Wisconsin",
    # "For decades, Missourians have talked about change". Leader-as-embodiment
    # is a core populist move; speaking_for covers the adjacent claim only.
    ("local_people", 1.00,
     "This text discusses a region or constituency as an administrative unit.",
     "This text invokes the people of a particular place as a community with a shared grievance or aspiration."),
]



TOPIC_HYPOTHESES = [
    # -- round 1: abstract domains of political authority -------------------
    "The text is about parliament, government, or how political decisions get made.",
    "The text is about elections, voting, referendums, or political mandates.",
    "The text is about courts, the constitution, or the rule of law.",
    "The text is about who holds political power and whether they should.",
    "The text is about democracy, legitimacy, or how a country should be governed.",
    "The text is about the EU, international bodies, or where decision-making power sits.",
    "The text is about accountability, transparency, or scrutiny of those in office.",
    "The text is about the media, public debate, or freedom of expression.",
    # -- round 2: concrete instruments. The abstract probes above missed
    #    "telecommunications surveillance, in plain German: wiretapping", "the
    #    Bavarian administration is supposed to program the technology for the
    #    property tax", and "won by hundreds of thousands of courageous people on
    #    the streets" -- each a specific mechanism of authority rather than a
    #    statement about authority in general.
    "The text is about a motion, a bill, a committee, or a parliamentary procedure.",
    "The text is about an inquiry, an investigation, or a scandal in office.",
    "The text is about surveillance, policing powers, or what the state may do to citizens.",
    "The text is about party funding, donations, or who pays for politics.",
    "The text is about appointments, resignations, or who holds a particular post.",
    "The text is about protests, demonstrations, petitions, or citizens acting politically.",
    "The text is about a coalition, a parliamentary majority, or a political deal.",
    # -- round 3: actors ----------------------------------------------------
    "The text is about political parties, politicians, or the political class.",
    "The text is about elites or ordinary citizens as opposed political groups.",
    "The text is about corruption, lobbying, or improper influence on politics.",
    "The text is about civil servants, agencies, bureaucracy, or administration.",
    # -- people-centric appeals with no institution named --------------------
    "The text says what ordinary people want, feel, or have had enough of.",
    "The text speaks on behalf of citizens, the public, or a group of ordinary people.",
    "The text praises or defends the character of ordinary people.",
    "The text treats popular support or opposition as a reason for or against something.",
    "The text appeals to everyday experience rather than to expertise or official figures.",
    # -- economic register (ADDED) -----------------------------------------
    # The gate dropped elite/people claims phrased in money vocabulary at
    # margins near -15: nothing in the 24 probes above mentions banks, tax,
    # wages or the cost of living. These five close that gap. Watch precision:
    # they can also admit ordinary economic-policy debate, which the negative
    # probes below are meant to subtract.
    "The text is about banks, corporations, or the very wealthy as a powerful group.",
    "The text is about tax avoidance, tax havens, or who escapes paying their share.",
    "The text is about who profits and who pays.",
    "The text is about ordinary people's wages, pensions, rent, or making ends meet.",
    "The text is about people who work hard and feel they are given too little.",
]

# Round 4, subtracted via the veto contrast. Deliberately SHORT: the first
# version of these was three long conjunctions and the veto fired on zero rows
# in 1,759, because a hedged multi-clause statement is far harder to entail than
# the positives it has to out-score. The economic negatives that worked are all
# one clause.

NEGATIVE_TOPIC_HYPOTHESES = [
    "The text attacks an opponent without making a claim about how politics should work.",
    "The text is a rhetorical flourish, an insult, or an exclamation.",
    "The text is a procedural remark about the sitting.",
    "The text is only about the substance of a policy, not about power or institutions.",
]

# 0 = pluralist end, 10 = populist end.
BANDS = [(1.43, "Strong Pluralist"), (2.86, "Pluralist"),
                  (4.29, "Lean Pluralist"), (5.71, "Mixed"),
                  (7.14, "Lean Populist"), (8.57, "Populist"),
                  (float("inf"), "Strong Populist")]


# Positive topic probes: the evidence that a sentence IS about economic policy.
# The gate uses the 2nd-highest margin, so two probes must agree.
#
# Grown in three measured rounds against a 256-sentence hand-coded gold set.
# Recall / precision after each (see NEGATIVE_TOPIC_HYPOTHESES for round four):
#   round 1, the 8 abstract domain probes   75.6% / 84.8%   F1 0.799
#   round 2, + 7 concrete-instrument probes 89.0% / 83.5%   F1 0.862
#   round 3, + 3 economic-interest probes   94.3% / 82.1%   F1 0.878
# NOTHING about rhetoric, culture, migration or defence belongs in this list --
# those go in NEGATIVE_TOPIC_HYPOTHESES, where the gate SUBTRACTS them. Putting
# them here cost 4.4 points of precision in one run, because they raised the

# -----------------------------------------------------------------------------
# WHAT HAS BEEN TRIED ON THIS GATE, AND FAILED. Do not spend another GPU run on
# these; each was measured against auth_gold (n=662, second coder kappa=0.836,
# so the attainable ceiling is about F1 0.91).
#
#   1. Mirroring the economic gate's three-layer probe structure (19 probes,
#      hash 2055e2a0ef58 as shipped). Flat: dF1 -0.003, CI [-0.013,+0.009].
#   2. Tuning the threshold on F1. Held-out dF1 -0.005, P(gain)=0.47. Noise.
#      (Tuning it on VALIDITY is different, and works -- see topic_margin_min.)
#   3. fit_gate() on all 23 probe margins. F1 0.687 vs 0.691 hand rule. The
#      ridge penalty pinned at the top of the grid in all five folds: CV wanted
#      maximum shrinkage, i.e. no exploitable structure beyond the one scalar.
#      The label-shuffle control sat at 0.399, so the fit was real -- just not
#      better than the hand rule.
#   4. Claim-based probes ("The text argues that X should have this power")
#      replacing mention-based ones ("The text is about parliament"). Precision
#      rose 0.602 -> 0.654 but recall collapsed 0.817 -> 0.550; at MATCHED
#      precision the original set gave more rows AND better validity at every
#      level (at prec 0.80: 389 rows / partial r 0.480 vs 285 / 0.398).
#      Circularity also rose to +0.259, near the +0.3 reject line. The gold
#      standard counts mentions as addressing political authority, so probes
#      that exclude mentions disagree with the target by construction.
#   5. Reviving the veto. It fires on ~1 row in 1759 as shipped. Under the
#      claim probes it fired 33 times -- and 68% of the rows it removed were
#      truly on-topic. A veto that fires is worse here than one that does not.
#
# Structural reason the economic recipe does not transfer: economic left-right
# IS a topic domain, so "is this about money?" partitions a corpus. Political
# authority is a FRAME that sits on any domain. Measured: the economic gate
# admits 42% of authority-positive rows and 50% of authority-negative ones --
# i.e. it is uninformative about authority. "You squander the money of the
# hard-working population" is economic in topic and populist in frame.
#
# What is NOT the limit: measurement noise. Inter-wording reliability is
# rho ~ 0.89 (0.94 for the averaged score), so the validity ceiling is ~0.97
# and observed validity 0.488 sits at half of it. The gap is construct, not
# precision.
# -----------------------------------------------------------------------------

# Paraphrase battery: the same 22 domains, in the same order and with the same
# weights, restated in different words. run() scores each pair under BOTH
# wordings and averages, and reliability() correlates the two -- that
# correlation is the ceiling for anything built on these scores.
#
# A paraphrase RESTATES; it does not trim. An earlier rewrite of this battery
# shortened people_homogeneity's "betrayed by a corrupt elite" clause and its
# validity fell from +0.733 to -0.128, because that clause carried the signal.
# Every clause below is preserved, only the vocabulary and verb frame change.
# Written 2026-10-26 at the author's request, so that all three dimensions
# average two wordings (left-right already did). The 25 supplied statements stay
# the MAIN battery, unchanged; this second wording is the pipeline's, and a score
# averages the two. reliability_from_cache() correlates them. One pluralist
# statement is shared by expertise and responsiveness, and so is its paraphrase.
_EXPERT_PARA = "This text holds that expert knowledge is a legitimate foundation for policy, even when the policy is unpopular."
BATTERY_PARAPHRASE = [
    ("validating_authority", 1.0,
     "This text stresses that courts and independent agencies have a role in confirming that political authority is legitimate.",
     "This text presents a leader's personal charisma as enough, on its own, to make political rule legitimate."),
    ("institutions_purpose", 1.0,
     "This text presents political institutions as safeguards that shield citizens from power used arbitrarily or abusively.",
     "This text presents elites as exploiting institutions to serve their own personal interests."),
    ("institutional_change", 1.0,
     "This text favours changing institutions in ways that neither undermine nor circumvent them.",
     "This text presents political institutions as obsolete arrangements that mainly work for the benefit of elites."),
    ("unelected_bodies", 1.0,
     "This text stresses that robust institutions matter more than powerful leaders.",
     "This text presents bodies that nobody elected as unfair barriers that stand in the way of democratic decisions."),
    ("expertise", 1.0,
     _EXPERT_PARA,
     "This text presents what the public thinks as better than the judgment of experts or technocrats."),
    ("responsiveness", 1.0,
     _EXPERT_PARA,
     "This text implies that elected leaders forfeit their legitimacy if they disregard what the people demand."),
    ("deliberation_pace", 1.0,
     "This text presents slow decisions as an unavoidable and necessary part of how democracy works.",
     "This text presents debate and procedural rules as stalling tactics that block action that is needed."),
    ("decision_mode", 1.0,
     "This text favours reaching political decisions by bargaining and finding compromises.",
     "This text implies that a leader is able to know plainly what the people truly want and to represent that will."),
    ("opposition", 1.0,
     "This text presents political rivals as rightful participants whose interests are valid.",
     "This text presents political conflict as a fight that pits ordinary citizens against a corrupt elite."),
    ("dissent", 1.0,
     "This text presents criticism and disagreement as things that make democracy stronger.",
     "This text presents anyone who stands against the will of the people as an enemy of democracy."),
    ("people_composition", 1.0,
     "This text presents society as made up of many different groups whose interests compete.",
     "This text presents the people as a single, morally united community whose common will is blocked by a minority acting in its own interest."),
    ("economic_elites", 1.0,
     "This text regards companies, banks, and rich people as normal actors in the economy.",
     "This text presents banks, large companies, or the super-rich as a privileged group that fixes the rules to benefit itself."),
    ("who_pays", 1.0,
     "This text regards taxes and government spending as routine matters of how policy is designed.",
     "This text presents taxation as a struggle in which ordinary people pay while privileged groups avoid paying."),
    ("material_grievance", 1.0,
     "This text addresses pay, pensions, and living costs as technical settings of policy.",
     "This text treats the economic struggles of ordinary people as a complaint against the people in power."),
    ("deservingness", 1.0,
     "This text regards rival economic groups as each having claims that are just as legitimate.",
     "This text presents hard-working people as putting in more than they get back."),
    ("speaking_for", 1.0,
     "This text recognizes that public opinion is split on the issue under discussion.",
     "This text asserts that it knows, and voices, what the people truly want."),
    ("popular_mood", 1.0,
     "This text argues for a position while conceding that it is unpopular at present.",
     "This text uses broad public anger or frustration to justify the position it takes."),
    ("local_people", 1.0,
     "This text talks about a region or electoral district as a unit of administration.",
     "This text appeals to the people of a specific place as a community that shares a grievance or a hope."),
]

OFF_TOPIC_LABEL = "not_political_authority"


# =============================================================================
# THE TWO HALVES OF POPULISM
# =============================================================================
# Populism is the conjunction of PEOPLE-CENTRISM and ANTI-ELITISM (Mudde 2004):
# a pure people, a corrupt elite, and politics as the expression of the people's
# will. Pluralism is the opposite pole -- many legitimate groups, adversaries who
# are opponents rather than enemies, and procedure as the source of legitimacy.
#
# The instrument that preceded this one measured only the anti-elitist half, and
# measured it well: AUC 0.806 against PopBERT's anti_elitism label, Cohen's
# d = 1.22, a clean monotone dose-response. On people_centrism it scored AUC
# 0.480 with a confidence interval spanning 0.50 -- no signal whatever. It failed
# at both stages: the gate admitted people-centric sentences at 30.3%, BELOW the
# 45.5% base rate for sentences with no populism label at all, and the battery
# then scored the survivors at 4.30 against 4.02 for unlabelled text.
#
# So each pair is now tagged with the half it belongs to. Every pair is scored
# from the same cached margins; the two sub-scales are produced by aggregating
# twice with complementary drop_domains, which needs no change to the engine.
#
def _auc(score, y, w=None):
    score = np.asarray(score, float); y = np.asarray(y).astype(bool)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    ok = np.isfinite(score)
    score, y, w = score[ok], y[ok], w[ok]
    if y.sum() == 0 or (~y).sum() == 0:
        return np.nan
    o = np.argsort(-score)
    tp = np.cumsum(w[o] * y[o]); fp = np.cumsum(w[o] * ~y[o])
    return float(np.trapezoid(tp / tp[-1], fp / fp[-1]))


def evaluate_populism(results, gold, id_col="id", weight_col="w",
                      gate_col="OnTopic", ae_col="AntiElitism",
                      pc_col="PeopleCentrism", ae_true="anti_elitism_n_true",
                      pc_true="people_centrism_n_true",
                      thresholds=(0, 2, 4, 6, 8, 10, 12, 14)):
    """Score the single populism scale against the recoded gold set.

    ONE instrument, TWO validity checks. The scale is one dimension again --
    populism against pluralism, both components folded in -- but it is still
    checked against each component separately, because the failure that produced
    this function was a single number that detected anti-elitism at AUC 0.806 and
    people-centrism at 0.480 while reporting neither. A combined figure cannot
    fail visibly; two can.

    Gate targets are the NON-DIRECTIONAL gold columns (OnTopic / AntiElitism /
    PeopleCentrism), which mark that an axis is addressed, either pole. Score
    targets are PopBERT's directional labels. Do not swap them: a strongly
    pluralist sentence is OnTopic=1 and scores LOW, so validating a directional
    score against a non-directional label reads as a collapse in validity.
    """
    res = pd.read_csv(results) if isinstance(results, str) else results.copy()
    gld = pd.read_csv(gold) if isinstance(gold, str) else gold.copy()
    res[id_col] = res[id_col].astype(str); gld[id_col] = gld[id_col].astype(str)

    def pick(*ends):
        for e in ends:
            hit = [c for c in res.columns if c.endswith(e)]
            if hit:
                return hit[0]
        return None
    sc = pick("lr_score_en", "lr_score")
    tm = pick("lr_topic_margin_en", "lr_topic_margin")
    na = pick("lr_n_active_en", "lr_n_active")
    keep = [c for c in (sc, tm, na, "faction") if c and c in res.columns]
    for c in (ae_true, pc_true):
        if c in res.columns and c not in gld.columns:
            keep.append(c)
    j = gld.merge(res[[id_col] + keep], on=id_col, how="left")
    for c in (ae_true, pc_true):
        if c in j.columns:
            j[c] = pd.to_numeric(j[c], errors="coerce").fillna(0)
    w = j[weight_col].values if weight_col in j.columns else np.ones(len(j))

    print("\n" + "=" * 74)
    print(f"POPULISM vs {gold if isinstance(gold, str) else 'gold'}   (n={len(j)} coded, weighted)")
    print("=" * 74)

    if tm:
        print("\n  1. THE GATE -- threshold sweep. Choose on VALIDITY, not F1.")
        print(f"  {'thr':>5}{'prec':>8}{'rec':>8}{'F1':>8}{'kept':>7}"
              f"{'AUC(AE)':>10}{'AUC(PC)':>10}")
        gate_gold = j[gate_col].astype(bool).values
        for t in thresholds:
            adm = j[tm].values >= t
            tp = (w * (adm & gate_gold)).sum(); fp = (w * (adm & ~gate_gold)).sum()
            fn = (w * (~adm & gate_gold)).sum()
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * p * r / (p + r) if p + r else 0.0
            sub = j[adm]
            a1 = _auc(sub[sc], sub[ae_true] > 0, sub[weight_col]) if len(sub) > 30 else np.nan
            a2 = _auc(sub[sc], sub[pc_true] > 0, sub[weight_col]) if len(sub) > 30 else np.nan
            print(f"  {t:>5}{p:>8.3f}{r:>8.3f}{f1:>8.3f}{int(adm.sum()):>7}"
                  f"{a1:>10.3f}{a2:>10.3f}")

    print("\n  2. ONE SCALE, CHECKED AGAINST BOTH COMPONENTS")
    print(f"  {'component':<18}{'target':<26}{'AUC':>8}{'d':>7}{'r':>7}{'n':>6}")
    got = {}
    s = j[j[sc].notna()]
    for name, gcol, prev in (("anti-elitism", ae_true, 0.806),
                             ("people-centrism", pc_true, 0.480)):
        if len(s) < 30:
            print(f"  {name:<18}too few scored rows ({len(s)})"); continue
        y = (pd.to_numeric(s[gcol], errors="coerce").fillna(0) > 0).values
        a = _auc(s[sc], y, s[weight_col])
        pos, neg = s[sc][y], s[sc][~y]
        sd = np.sqrt(((len(pos) - 1) * pos.var(ddof=1) + (len(neg) - 1) * neg.var(ddof=1))
                     / max(len(pos) + len(neg) - 2, 1))
        d = (pos.mean() - neg.mean()) / sd if sd else np.nan
        r = np.corrcoef(s[sc], pd.to_numeric(s[gcol], errors="coerce").fillna(0))[0, 1]
        got[name] = a
        print(f"  {name:<18}{gcol:<26}{a:>8.3f}{d:>7.2f}{r:>7.3f}{len(s):>6}"
              f"   (was {prev:.3f})")

    # each component, where the OTHER is absent -- co-occurrence vs real signal
    print("\n  3. PARTIALLED -- each component where the other is absent")
    for name, gcol, ocol in (("anti-elitism", ae_true, pc_true),
                             ("people-centrism", pc_true, ae_true)):
        cl = s[s[ocol] == 0]
        if len(cl) > 30:
            print(f"    {name:<18}{_auc(cl[sc], cl[gcol] > 0, cl[weight_col]):>8.3f}"
                  f"   (n={len(cl)})")

    if na and na in s.columns:
        thin = (s[na] <= 2).mean()
        print(f"\n  evidence: median {s[na].median():.0f} active pairs of {len(BATTERY)}; "
              f"{thin:.0%} of scored rows rest on <= 2")

    pc = got.get("people-centrism", np.nan)
    ae = got.get("anti-elitism", np.nan)
    if np.isfinite(pc) and np.isfinite(ae):
        print()
        if pc < 0.55:
            print("  PEOPLE-CENTRISM NOT DETECTED. One scale is again carrying only the")
            print("  anti-elitism half. Report it under that name, not as populism.")
        elif abs(ae - pc) > 0.15:
            print(f"  LOPSIDED: {ae:.3f} vs {pc:.3f}. The scale detects both, but one")
            print("  component dominates it -- say so rather than calling it populism flat.")
        else:
            print("  Both components detected at comparable strength. The single scale is")
            print("  measuring populism, not one of its halves.")
    print("=" * 74 + "\n")
    return j



# =============================================================================
# ENGINE  --  byte-identical in classify_economic.py and classify_populism.py.
# Nothing below this line mentions a specific dimension. Fix bugs here in BOTH
# scripts; test_engine_parity.py fails if the two copies ever diverge.
# =============================================================================

STANCE_PAIRS = [
    ("The text describes what is being done or what has been decided.",
     "The text says what ought to be done."),
    ("The text reports facts, figures, or measures that already exist.",
     "The text demands a change in policy."),
    ("The text is a factual description of a situation.",
     "The text is an argument for a political position."),
    ("The text explains how something currently works.",
     "The text criticises or defends a policy on principle."),
]


def stance_hypotheses(pairs=STANCE_PAIRS):
    """Flat list; pair i is at 2i (descriptive) and 2i+1 (normative)."""
    out = []
    for desc, norm in pairs:
        out.extend([desc, norm])
    return out


def stance_hash(pairs=STANCE_PAIRS):
    return hashlib.sha256(json.dumps(pairs, sort_keys=True,
                                     ensure_ascii=False).encode()).hexdigest()[:12]


# Probes for what the gate should REFUSE, scored separately and subtracted.
# They cannot go in TOPIC_HYPOTHESES: that list is positive evidence, so adding
# them there would make rhetoric raise the chance of admission instead of lowering
# it. The gate becomes a contrast, max(positive) - max(negative), which is what
# lets a sentence that is genuinely both -- welfare chauvinism, say -- still get
# in on the strength of its economic side rather than being hard-vetoed.
#
# On the PopBERT sample the target is the nine false positives that survived all
# three rounds of positive probes: content-free attacks on the left scored as
# right-wing economics ("Red-red-green policy is a policy in which quite simply
def veto_hash(vetoes=NEGATIVE_TOPIC_HYPOTHESES):
    return hashlib.sha256(json.dumps(list(vetoes), sort_keys=True,
                                     ensure_ascii=False).encode()).hexdigest()[:12]


def battery_hash(battery=BATTERY, topics=TOPIC_HYPOTHESES):
    """Identifies which version of the battery produced a file."""
    blob = json.dumps([battery, topics], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


# =============================================================================
# INPUT DISCOVERY
# =============================================================================
def list_inputs(root="/kaggle/input"):
    if not os.path.isdir(root):
        print(f"{root} does not exist (not running on Kaggle?)")
        return []
    found = []
    for dirpath, _, files in os.walk(root):
        if files:
            print(dirpath)
            for f in sorted(files):
                print("   ", f)
                found.append(os.path.join(dirpath, f))
    if not found:
        print(f"{root} is empty - add your dataset via '+ Add Input'.")
    return found


def find_input(pattern="*.csv", root="/kaggle/input", explicit=None):
    if explicit:
        if not os.path.exists(explicit):
            list_inputs(root)
            raise FileNotFoundError(f"input_csv not found: {explicit}")
        return explicit
    if not os.path.isdir(root):
        raise FileNotFoundError(
            "No input_csv given and /kaggle/input does not exist. "
            "Set CONFIG['input_csv'] to a path.")
    hits = sorted(glob.glob(os.path.join(root, "**", pattern), recursive=True))
    if not hits:
        hits = sorted(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True))
    if not hits:
        list_inputs(root)
        raise FileNotFoundError(f"No CSV matching {pattern!r} under {root}")
    if len(hits) > 1:
        logger.info("Multiple CSVs found, using the first:")
        for h in hits:
            logger.info("   %s", h)
    logger.info("Input: %s", hits[0])
    return hits[0]


_DE_MARKERS = re.compile(
    r"\b(der|die|das|und|nicht|ist|sind|werden|wir|dass|eine|einen|für|mit|auch|"
    r"haben|wird|diese|sich|über|müssen|sollen)\b", re.IGNORECASE)


def check_language(texts, vocab_size, sample=200):
    """English checkpoint + German text is a silent validity failure, not an error."""
    sample_texts = [t for t in list(texts)[:sample] if isinstance(t, str)]
    if not sample_texts:
        return
    hits = sum(1 for t in sample_texts if len(_DE_MARKERS.findall(t)) >= 3)
    frac = hits / len(sample_texts)
    if frac > 0.3 and vocab_size < 200_000:
        logger.error(
            "=" * 70 + "\n"
            "  %.0f%% of the sampled texts look German, but the tokenizer has a\n"
            "  %d-token vocabulary, i.e. this is the ENGLISH DeBERTa-v3 checkpoint,\n"
            "  not mDeBERTa. Scores from this combination are not interpretable.\n"
            "  Use text_col='text_en', or switch to a multilingual NLI model.\n"
            + "=" * 70, 100 * frac, vocab_size)


# =============================================================================
# MODEL LOADING
# =============================================================================
def _require_torch():
    if torch is None:
        raise SystemExit("PyTorch is required for phase 1 (computing margins). "
                         "Phase 2 -- rescore(), compare_settings(), diagnose() -- "
                         "runs from a cached .margins.npz without it.")


def _pick_device(device_arg=None):
    _require_torch()
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_model(model_name, device_arg=None, fp16=None, revision=None):
    _require_torch()
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        raise SystemExit("transformers missing. Run:  !pip -q install --upgrade transformers")

    device = _pick_device(device_arg)
    kw = {"revision": revision} if revision else {}
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, **kw)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, **kw)
    except OSError as e:
        raise SystemExit(
            f"Could not load '{model_name}'.\n"
            "On Kaggle this is almost always Internet being OFF "
            "(Settings -> Internet -> On), or a wrong path to an attached "
            f"model dataset.\nOriginal error: {e}")

    model.to(device).eval()
    if fp16 is None:
        fp16 = device.type == "cuda"
    if fp16 and device.type == "cuda":
        model.half()
        logger.info("fp16 enabled (DeBERTa-v2 attention can overflow in fp16 - "
                    "fp16_check compares against fp32)")
    elif fp16:
        logger.warning("fp16 requested on %s; ignoring", device.type)

    logger.info("Loaded %s%s on %s | labels=%s | vocab=%d",
                model_name, f"@{revision}" if revision else "", device.type,
                getattr(model.config, "id2label", None), model.config.vocab_size)
    if device.type == "cuda":
        logger.info("GPU: %s (%.1f GB)", torch.cuda.get_device_name(0),
                    torch.cuda.get_device_properties(0).total_memory / 1e9)
    else:
        logger.warning("Running on %s - set Accelerator to GPU for a large speedup.",
                       device.type)
    return model, tokenizer, device


# =============================================================================
# PHASE 1 (GPU): text x hypothesis -> entailment margin
# =============================================================================
class MarginScorer:
    """Computes raw entailment margins. Deliberately knows nothing about
    left/right: all interpretation happens in phase 2."""

    def __init__(self, model_name="mlburnham/Political_DEBATE_large_v1.0",
                 device=None, fp16=None, revision=None, max_length=512):
        self.model, self.tokenizer, self.device = _load_model(
            model_name, device, fp16, revision)
        self.model_name = model_name
        self.revision = revision
        self.max_length = max_length
        self.n_labels = int(self.model.config.num_labels)
        self.entailment_idx = self._find_entailment_index()
        self.n_nonfinite = 0
        if self.n_labels == 2:
            logger.info("2-label model: margin = logit[entailment] - logit[not_entailment], "
                        "p = sigmoid(margin)")

    def _find_entailment_index(self):
        cfg = self.model.config
        if getattr(cfg, "label2id", None):
            for label, idx in cfg.label2id.items():
                if str(label).lower() in ("entailment", "entail"):
                    return int(idx)
        logger.warning("No 'entailment' label in config; defaulting to index 0")
        return 0

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    def margins(self, premises, hypotheses, batch_size=64, progress=False):
        """Entailment margin for parallel premise/hypothesis lists.

        Truncation is `only_first`, so a long premise is clipped but the
        hypothesis is never silently cut in half the way `longest_first` can.
        max_length defaults to 512, the most DeBERTa-v3 accepts (premise and
        hypothesis together), so text is only dropped past the model's own
        limit. The same cap and rule are used by the support-for-democracy
        scorer (classifier/alternativeLib_v3.py); test_truncation.py pins it.
        """
        assert len(premises) == len(hypotheses)
        out = np.empty(len(premises), dtype=np.float32)
        rng = range(0, len(premises), batch_size)
        for i in (tqdm(list(rng), desc="margins", unit="batch", leave=False)
                  if progress else rng):
            j = min(i + batch_size, len(premises))
            enc = self.tokenizer(
                premises[i:j], hypotheses[i:j],
                return_tensors="pt", truncation="only_first", padding=True,
                max_length=self.max_length, pad_to_multiple_of=8)
            enc = {k: v.to(self.device, non_blocking=True) for k, v in enc.items()}
            with torch.inference_mode():
                lg = self.model(**enc).logits.float()
            ent = lg[:, self.entailment_idx]
            if self.n_labels == 2:
                other = lg[:, 1 - self.entailment_idx]
            else:
                other = torch.cat([lg[:, :self.entailment_idx],
                                   lg[:, self.entailment_idx + 1:]],
                                  dim=1).max(dim=1).values
            m = (ent - other).detach().cpu().numpy()
            bad = ~np.isfinite(m)
            if bad.any():
                # fp16 overflow in the disentangled attention. Do not let it
                # pass as a real value.
                self.n_nonfinite += int(bad.sum())
                m[bad] = np.nan
            out[i:j] = m
        return out

    def margin_matrix(self, texts, hypotheses, batch_size=None, text_chunk=256,
                      progress=True):
        """(n_texts, n_hypotheses) margins.

        Two throughput details that v1 left on the table:
          * texts are sorted by length so neighbouring premises pad alike;
          * batch_size is a multiple of len(hypotheses), so a batch never
            straddles two texts of different lengths.
        Results are returned in the caller's original order.
        """
        texts = list(texts)
        nh = len(hypotheses)
        if batch_size is None:
            batch_size = nh * max(1, 64 // nh)
        elif batch_size % nh:
            batch_size = max(nh, (batch_size // nh) * nh)
            logger.info("batch_size rounded to %d (a multiple of %d hypotheses) "
                        "to keep batches aligned to text boundaries", batch_size, nh)

        out = np.full((len(texts), nh), np.nan, dtype=np.float32)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i] or ""))
        chunks = [order[s:s + text_chunk] for s in range(0, len(order), text_chunk)]
        for idx in (tqdm(chunks, desc="phase 1", unit="chunk") if progress else chunks):
            prem = [texts[i] for i in idx for _ in range(nh)]
            hyp = [h for _ in idx for h in hypotheses]
            m = self.margins(prem, hyp, batch_size=batch_size).reshape(len(idx), nh)
            out[idx] = m
        return out

    def fp16_check(self, texts, hypotheses, n=8, batch_size=64):
        """Score a small sample in fp16 and fp32 and report the gap.

        Worth the 30 seconds: v1 ran fp16 by default with every warning
        suppressed, so an overflow would have looked like a confident score.
        """
        if self.dtype != torch.float16:
            logger.info("fp16_check skipped (model is %s)", self.dtype)
            return None
        sample = [t for t in list(texts) if isinstance(t, str) and t.strip()][:n]
        if not sample:
            return None
        prem = [t for t in sample for _ in hypotheses]
        hyp = [h for _ in sample for h in hypotheses]
        m16 = self.margins(prem, hyp, batch_size)
        self.model.float()
        m32 = self.margins(prem, hyp, batch_size)
        self.model.half()
        d = np.abs(m16 - m32)
        finite = np.isfinite(d)
        report = {"n_pairs": int(d.size),
                  "nonfinite_fp16": int((~np.isfinite(m16)).sum()),
                  "max_abs_diff": float(np.nanmax(d)) if finite.any() else float("nan"),
                  "p99_abs_diff": float(np.nanpercentile(d[finite], 99)) if finite.any() else float("nan"),
                  "mean_abs_diff": float(np.nanmean(d[finite])) if finite.any() else float("nan")}
        lvl = logger.warning if (report["nonfinite_fp16"] or report["max_abs_diff"] > 0.5) else logger.info
        lvl("fp16 vs fp32 on %d pairs: max|d|=%.4f  p99=%.4f  mean=%.4f  non-finite=%d",
            report["n_pairs"], report["max_abs_diff"], report["p99_abs_diff"],
            report["mean_abs_diff"], report["nonfinite_fp16"])
        if report["max_abs_diff"] > 0.5 or report["nonfinite_fp16"]:
            logger.warning("fp16 is distorting margins -- re-run with fp16=False.")
        return report

    def saturation_report(self, margins):
        """How much of the signal the old probability scale was discarding."""
        m = margins[np.isfinite(margins)]
        if not m.size:
            return {}
        p = 1.0 / (1.0 + np.exp(-m))
        rep = {"pct": {str(q): float(v) for q, v in
                       zip([1, 10, 25, 50, 75, 90, 99], np.percentile(m, [1, 10, 25, 50, 75, 90, 99]))},
               "share_abs_margin_gt_10": float(np.mean(np.abs(m) > 10)),
               "share_p_in_05_95": float(np.mean((p > 0.05) & (p < 0.95)))}
        logger.info("margin percentiles (1/10/25/50/75/90/99): %s",
                    np.percentile(m, [1, 10, 25, 50, 75, 90, 99]).round(2).tolist())
        logger.info("share |margin| > 10: %.1f%%   share p in [.05,.95]: %.1f%%",
                    100 * rep["share_abs_margin_gt_10"], 100 * rep["share_p_in_05_95"])
        return rep


# =============================================================================
# PHASE 2 (CPU): margins -> left/right scores
# =============================================================================
def battery_hypotheses(battery=BATTERY):
    """Flat hypothesis list; pair i occupies indices 2i (left) and 2i+1 (right)."""
    out = []
    for _, _, left, right in battery:
        out.extend([left, right])
    return out


def interpret(score, bands=None):
    """Band label for a score. `bands` overrides this script's own BANDS, so a
    caller can label a single rescore on a different scale."""
    if score is None or not np.isfinite(score):
        return None
    for edge, name in (bands or BANDS):
        if score <= edge:
            return name
    return (bands or BANDS)[-1][1]


class ContrastAggregator:
    """Turns a margin matrix into scores.

    The unit of evidence is one PAIR, scored as the signed contrast between its
    right and left mirror:

        c_p = (margin_right - margin_left) / 2        positive = right

    A pair where neither side entails has c ~ 0, and 0 is the neutral value, so
    inactive pairs neither dilute nor distort a plain weighted mean.  That is
    what removes the need for v1's k_eff and, with it, both the cross-side
    coupling and the length/breadth confound.

    Parameters
    ----------
    standardize : "none" | "center" | "z"
        "center" subtracts each pair's corpus median contrast, which removes the
        template's baseline propensity to entail one side more readily than the
        other.  "z" also divides by the pair's corpus SD, which makes the output
        purely relative (good for ranking parties, wrong for absolute placement).
    contrast_scale : "auto" | float
        Logit units mapped to one unit of position via tanh.  "auto" uses the SD
        of active centred contrasts, which keeps the scale on the data rather
        than on a hand-picked constant.  Recorded in the metadata either way.
    active_only : bool
        True  -> mean position across the issues the text actually addresses.
        False -> divided by the whole battery, so score conflates position with
                 how many issues the text covers.  True is the defensible default.
    """

    def __init__(self, battery=BATTERY, topic_hypotheses=TOPIC_HYPOTHESES,
                 topic_margin_min=0.0, active_margin=0.0, standardize="none",
                 contrast_scale="auto", active_only=True, bootstrap=200, seed=0,
                 min_tokens=5, min_center_n=25, min_active=2,
                 stance_pairs=STANCE_PAIRS, stance_mode="off", stance_min=0.0,
                 stance_scale=4.0, drop_domains=(),
                 veto_hypotheses=NEGATIVE_TOPIC_HYPOTHESES, veto_margin=0.0,
                 use_veto=True, balance_weight=0.0, bands=None,
                 off_topic_label=OFF_TOPIC_LABEL):
        if standardize not in ("none", "center", "z"):
            raise ValueError("standardize must be none | center | z")
        if stance_mode not in ("off", "gate", "shrink"):
            raise ValueError("stance_mode must be off | gate | shrink")
        self.battery = list(battery)
        self.topic_hypotheses = list(topic_hypotheses)
        self.domains = [d for d, _, _, _ in self.battery]
        self.weights = np.array([w for _, w, _, _ in self.battery], dtype=np.float64)
        self.topic_margin_min = float(topic_margin_min)
        self.active_margin = float(active_margin)
        self.standardize = standardize
        self.contrast_scale = contrast_scale
        self.active_only = bool(active_only)
        self.bootstrap = int(bootstrap)
        self.seed = int(seed)
        # A three-word fragment cannot carry an economic position, but NLI will
        # happily entail a hypothesis from one. Observed on real data: the
        # segment "the Left rejects." scored as confidently pro-market.
        self.min_tokens = int(min_tokens)
        self.min_center_n = int(min_center_n)
        # 9.8% of scored rows rested on a single active pair, with no bootstrap
        # possible and nothing to average out a false positive.
        self.min_active = int(min_active)
        self.stance_pairs = list(stance_pairs)
        self.stance_mode = stance_mode
        self.stance_min = float(stance_min)
        self.stance_scale = float(stance_scale)
        # Pairs excluded from aggregation. On the PopBERT sample
        # planning_vs_market and state_role fired on any mention of state
        # activity and drove 47% of the spurious Strong-Left tail; excluding
        # them raised agreement with the human left_wing label from d=-0.60 to
        # d=-0.85. Use domain_sweep() to choose these on your own data rather
        # than inheriting that finding.
        # Down-weight pairs whose active cells fire almost entirely one way.
        # LABEL-FREE, which is the point: r(|raw R% - 50|, |validity|) = -0.564 on
        # the PopBERT sample, so imbalance predicts low validity without ever
        # touching the criterion. Setting weights from the validity column itself
        # would be fitting to the thing you then report.
        #   w_eff = w * (1 - balance_weight * |2*rawR - 1|)
        #   0.0 = off,  0.5 = a fully one-sided pair keeps half its weight,
        #   1.0 = a fully one-sided pair is silenced.
        self.balance_weight = float(balance_weight)
        self.bands = list(bands) if bands else None
        # OFF_TOPIC_LABEL is set in this script's DIMENSION block, so the status
        # column reads in this dimension's own vocabulary. Override per call with
        # off_topic_label= if a single rescore needs a different word.
        self.off_topic = str(off_topic_label)
        self.veto_hypotheses = list(veto_hypotheses or [])
        self.veto_margin = float(veto_margin)
        self.use_veto = bool(use_veto)
        self.drop_domains = tuple(drop_domains or ())
        unknown = set(self.drop_domains) - set(self.domains)
        if unknown:
            raise ValueError(f"drop_domains not in the battery: {sorted(unknown)}")
        self.keep_ = np.array([d not in self.drop_domains for d in self.domains])
        if self.drop_domains:
            logger.info("dropping %d pair(s) from aggregation: %s",
                        len(self.drop_domains), ", ".join(self.drop_domains))
        self.fit_ = {}

    # ------------------------------------------------------------------ core
    def _contrasts(self, battery_margins):
        m = np.asarray(battery_margins, dtype=np.float64)
        if m.shape[1] != 2 * len(self.battery):
            raise ValueError(f"expected {2 * len(self.battery)} hypothesis columns, "
                             f"got {m.shape[1]}")
        left, right = m[:, 0::2], m[:, 1::2]
        contrast = (right - left) / 2.0
        active = (np.fmax(left, right) >= self.active_margin) & np.isfinite(contrast)
        # a dropped pair is simply never active: it then contributes nothing to
        # the score, to n_active, to the bootstrap, or to the centring fit
        active &= self.keep_[None, :]
        return contrast, active

    def _topic(self, topic_margins):
        """Second-highest topic margin: two probes must agree, not just one."""
        t = np.asarray(topic_margins, dtype=np.float64)
        if t.shape[1] < 2:
            return np.nanmax(t, axis=1)
        part = np.sort(np.nan_to_num(t, nan=-np.inf), axis=1)
        return part[:, -2]

    @staticmethod
    def _solve_centre(v, scale, lo=-60.0, hi=60.0, iters=64):
        """Centre c such that mean(tanh((v - c)/scale)) == 0 over active cells.

        Median-centring assumes the contrast is unimodal. It is not: a saturated
        NLI model puts the mass at roughly +-10 logits, so the median lands on a
        mode, half the centred values sit at exactly 0, and the mean stays far
        from it. (Observed: a pair centred on 29 active rows still had a mean
        position of +0.49.) Solving for zero MEAN POSITION is what "remove the
        template's baseline propensity" actually means, and it is monotone in c,
        so plain bisection is enough.
        """
        f = lambda c: float(np.mean(np.tanh((v - c) / scale)))
        if f(lo) <= 0:
            return lo
        if f(hi) >= 0:
            return hi
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if f(mid) > 0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def _topic_max(self, topic_margins):
        """Strongest positive topic evidence, for the veto contrast."""
        return np.nanmax(np.asarray(topic_margins, dtype=np.float64), axis=1)

    def veto(self, veto_margins):
        """Strongest negative-probe margin. Higher = more likely off-topic rhetoric."""
        if veto_margins is None:
            return None
        v = np.asarray(veto_margins, dtype=np.float64)
        if v.shape[1] != len(self.veto_hypotheses):
            raise ValueError(f"expected {len(self.veto_hypotheses)} veto columns, "
                             f"got {v.shape[1]}")
        return np.nanmax(v, axis=1)

    def stance(self, stance_margins):
        """Mean (normative - descriptive) contrast. Positive = argues a position."""
        if stance_margins is None:
            return None
        s = np.asarray(stance_margins, dtype=np.float64)
        if s.shape[1] != 2 * len(self.stance_pairs):
            raise ValueError(f"expected {2 * len(self.stance_pairs)} stance columns, "
                             f"got {s.shape[1]}")
        return np.nanmean((s[:, 1::2] - s[:, 0::2]) / 2.0, axis=1)

    def fit(self, battery_margins, topic_margins=None):
        """Estimate per-pair centres and the contrast scale from the corpus."""
        contrast, active = self._contrasts(battery_margins)
        rows = np.ones(contrast.shape[0], dtype=bool)
        if topic_margins is not None:
            rows = self._topic(topic_margins) >= self.topic_margin_min
        if rows.sum() < 20:                      # too small to estimate anything
            rows = np.ones(contrast.shape[0], dtype=bool)
        ref, ref_active = contrast[rows], active[rows]

        P = contrast.shape[1]
        centre, spread = np.zeros(P), np.ones(P)
        n_used = np.zeros(P, dtype=int)
        # active cells only: an inactive pair's contrast is the difference of two
        # large negative margins, i.e. noise about an arbitrary offset, and
        # pooling it in biases the centre in proportion to how rarely the pair
        # fires -- the artefact that showed up as mean-position drifting with
        # activation rate in the per-pair diagnostics.
        vals = []
        for p in range(P):
            v = ref[ref_active[:, p], p]
            v = v[np.isfinite(v)]
            n_used[p] = v.size
            vals.append(v)
        ok = [p for p in range(P) if n_used[p] >= self.min_center_n]

        def _pooled_sd(c, s):
            z = [(vals[p] - c[p]) / s[p] for p in ok if vals[p].size]
            if not z:
                return 1.0
            z = np.concatenate(z)
            sd = float(np.std(z))
            return sd if np.isfinite(sd) and sd > 1e-6 else 1.0

        fixed_scale = None if self.contrast_scale == "auto" else float(self.contrast_scale)
        scale = fixed_scale or _pooled_sd(centre, spread)

        if self.standardize != "none":
            for _ in range(2):                   # centre and scale co-determine
                for p in ok:
                    centre[p] = self._solve_centre(vals[p], scale)
                if fixed_scale is None:
                    scale = _pooled_sd(centre, spread)
            if self.standardize == "z":
                sds = [float(np.std(vals[p] - centre[p])) for p in ok]
                pooled = float(np.median(sds)) if sds else 1.0
                for p, sd in zip(ok, sds):
                    # a rarely-firing pair has a tiny SD; dividing by it would
                    # turn its noise into the loudest signal in the battery
                    spread[p] = max(sd, 0.25 * pooled) or 1.0
                if fixed_scale is None:
                    scale = _pooled_sd(centre, spread)
                for p in ok:                     # re-solve in the new units
                    centre[p] = self._solve_centre(vals[p], scale * spread[p])

            thin = [(self.domains[p], int(n_used[p]))
                    for p in range(P)
                    if self.keep_[p] and n_used[p] < self.min_center_n]
            if thin:
                need = max((int(np.ceil(self.min_center_n * len(contrast) / max(n, 1)))
                            for _, n in thin), default=0)
                logger.warning(
                    "centre left at 0 for %d of %d pairs with < %d active rows: %s",
                    len(thin), P, self.min_center_n,
                    ", ".join(f"{d} (n={n})" for d, n in thin))
                logger.warning(
                    "  -> at this activation rate every pair needs roughly %d input "
                    "rows to be centrable; scores are provisional until then.", need)

        raw_r = np.array([float(np.mean(vals[p] > 0)) if vals[p].size else np.nan
                          for p in range(P)])
        w_eff = self.weights.copy()
        if self.balance_weight > 0:
            imb = np.abs(2 * np.nan_to_num(raw_r, nan=0.5) - 1.0)
            w_eff = w_eff * (1.0 - self.balance_weight * imb)
            w_eff = np.maximum(w_eff, 1e-6)
            worst = np.argsort(-imb)[:3]
            logger.info("balance_weight=%.2f: most one-sided pairs %s",
                        self.balance_weight,
                        ", ".join(f"{self.domains[p]} {100*raw_r[p]:.0f}%R "
                                  f"w {self.weights[p]:.2f}->{w_eff[p]:.2f}"
                                  for p in worst if np.isfinite(raw_r[p])))
        self.weights_eff = w_eff
        self.fit_ = {"centre": centre.tolist(), "spread": spread.tolist(),
                     "raw_right_share": raw_r.tolist(),
                     "weights_eff": w_eff.tolist(),
                     "balance_weight": self.balance_weight,
                     "scale": scale, "n_reference_rows": int(rows.sum()),
                     "n_active_rows_per_pair": n_used.tolist(),
                     "n_centred_pairs": len(ok)}
        if self.standardize == "none":
            logger.info("aggregator fit on %d rows | standardize=none (no centring) "
                        "| scale=%.3f logits", int(rows.sum()), scale)
        else:
            logger.info("aggregator fit on %d rows | standardize=%s | scale=%.3f "
                        "logits | %d/%d pairs centred",
                        int(rows.sum()), self.standardize, scale, len(ok), P)
        return self

    def _positions(self, contrast):
        centre = np.asarray(self.fit_["centre"])
        spread = np.asarray(self.fit_["spread"])
        scale = self.fit_["scale"]
        return np.tanh((contrast - centre) / spread / scale)

    # --------------------------------------------------------------- scoring
    def transform(self, battery_margins, topic_margins, texts=None, n_top=3,
                  stance_margins=None, veto_margins=None, admit_mask=None):
        if not self.fit_:
            self.fit(battery_margins, topic_margins)
        contrast, active = self._contrasts(battery_margins)
        pos = self._positions(contrast)
        topic = self._topic(topic_margins)
        vt = self.veto(veto_margins) if self.use_veto else None
        tmax = self._topic_max(topic_margins) if vt is not None else None
        if vt is None and self.use_veto and veto_margins is None:
            logger.info("no veto columns in this cache; gate runs on positive "
                        "probes only. Re-run phase 1 to enable the contrast.")
        if admit_mask is not None:
            admit_mask = np.asarray(admit_mask, bool)
            if admit_mask.shape[0] != contrast.shape[0]:
                raise ValueError(f"admit_mask has {admit_mask.shape[0]} rows, "
                                 f"margins have {contrast.shape[0]}")
            logger.info("gate supplied externally (admit_mask): admits %d of %d rows; "
                        "topic_margin_min and the veto are ignored",
                        int(admit_mask.sum()), admit_mask.shape[0])
        st = self.stance(stance_margins)
        if st is None and self.stance_mode != "off":
            logger.warning("stance_mode=%r but the margin cache has no stance "
                           "columns; falling back to 'off'. Re-run phase 1 to "
                           "enable the gate.", self.stance_mode)
        w = getattr(self, "weights_eff", self.weights)
        rng = np.random.default_rng(self.seed)
        n = contrast.shape[0]
        rows = []

        for i in range(n):
            text = texts[i] if texts is not None else None
            if texts is not None:
                if not isinstance(text, str) or not text.strip():
                    rows.append(self._blank("empty_text", np.nan))
                    continue
                if len(text.split()) < self.min_tokens:
                    rows.append(self._blank("too_short", np.nan))
                    continue
            if not np.isfinite(contrast[i]).any():
                rows.append(self._blank("nonfinite_margins", topic[i]))
                continue
            if admit_mask is not None:
                if not admit_mask[i]:
                    r = self._blank(self.off_topic, topic[i])
                    r["gate_reason"] = "fitted"
                    rows.append(r)
                    continue
            elif topic[i] < self.topic_margin_min:
                r = self._blank(self.off_topic, topic[i])
                r["gate_reason"] = "topic"
                rows.append(r)
                continue
            if admit_mask is None and vt is not None and (tmax[i] - vt[i]) <= self.veto_margin:
                # positive evidence exists but rhetoric outweighs it. Status stays
                # OFF_TOPIC_LABEL so downstream filters on it keep working; the
                # reason column says which rule fired.
                r = self._blank(self.off_topic, topic[i])
                r["gate_reason"] = "veto"
                r["veto"] = float(vt[i])
                rows.append(r)
                continue
            s_i = float(st[i]) if st is not None and np.isfinite(st[i]) else np.nan
            if (st is not None and self.stance_mode == "gate"
                    and np.isfinite(s_i) and s_i < self.stance_min):
                r = self._blank("descriptive", topic[i])
                r["stance"] = s_i
                rows.append(r)
                continue

            act = np.where(active[i])[0]
            if act.size == 0 or act.size < self.min_active:
                # no evidence at all is a different problem from thin evidence
                r = self._blank("no_active_pair" if act.size == 0 else "thin_evidence",
                                topic[i])
                r["stance"] = s_i
                r["n_active"] = int(act.size)
                rows.append(r)
                continue
            if self.active_only:
                if act.size == 0:
                    rows.append(self._blank("no_active_pair", topic[i]))
                    continue
                idx, ww, vv = act, w[act], pos[i, act]
            else:
                ok = np.where(np.isfinite(pos[i]))[0]
                idx, ww, vv = ok, w[ok], pos[i, ok]
                if idx.size == 0:
                    rows.append(self._blank("no_active_pair", topic[i]))
                    continue

            raw = float(np.sum(ww * vv) / np.sum(ww))
            if st is not None and self.stance_mode == "shrink" and np.isfinite(s_i):
                # smooth alternative to the gate: a descriptive sentence is pulled
                # toward the centre rather than dropped
                raw *= 1.0 / (1.0 + np.exp(-(s_i - self.stance_min) / self.stance_scale))
            score = float(np.clip(5.0 + 5.0 * raw, 0.0, 10.0))

            # weighted dispersion of positions across active pairs: the honest
            # replacement for v1's "contradiction_detected" flag
            disp = float(np.sqrt(np.sum(ww * (vv - raw) ** 2) / np.sum(ww)))

            sd = lo = hi = np.nan
            if self.bootstrap and act.size >= 2:
                draw = rng.integers(0, idx.size, size=(self.bootstrap, idx.size))
                bw, bv = ww[draw], vv[draw]
                bs = 5.0 + 5.0 * (bw * bv).sum(1) / bw.sum(1)
                bs = np.clip(bs, 0.0, 10.0)
                sd, lo, hi = float(bs.std()), *np.percentile(bs, [5, 95]).tolist()

            order = np.argsort(-np.abs(ww * vv))[:n_top]
            top = [{"domain": self.domains[idx[j]],
                    "position": round(float(vv[j]), 4),
                    "contrast": round(float(contrast[i, idx[j]]), 3),
                    "weight": float(ww[j])} for j in order]

            rows.append({"score": score, "status": "scored", "stance": s_i,
                         "gate_reason": "admitted",
                         "veto": float(vt[i]) if vt is not None else np.nan,
                         "interpretation": interpret(score, self.bands),
                         "score_sd": sd, "score_lo": lo, "score_hi": hi,
                         "dispersion": disp, "n_active": int(act.size),
                         "topic_margin": float(topic[i]), "is_relevant": True,
                         "raw_position": raw, "top": top})
        return rows

    def _blank(self, status, topic_margin):
        return {"score": np.nan, "status": status, "interpretation": None,
                "stance": np.nan, "veto": np.nan,
                "gate_reason": "topic" if status == getattr(self, "off_topic",
                                                            OFF_TOPIC_LABEL) else "admitted",
                "score_sd": np.nan, "score_lo": np.nan, "score_hi": np.nan,
                "dispersion": np.nan, "n_active": 0,
                "topic_margin": float(topic_margin) if np.isfinite(topic_margin) else np.nan,
                "is_relevant": status not in (getattr(self, "off_topic", OFF_TOPIC_LABEL),
                                              "empty_text", "too_short", "descriptive"),
                "raw_position": np.nan, "top": []}

    def config(self):
        return {"topic_margin_min": self.topic_margin_min,
                "active_margin": self.active_margin,
                "standardize": self.standardize,
                "contrast_scale": self.contrast_scale,
                "active_only": self.active_only,
                "bootstrap": self.bootstrap, "seed": self.seed,
                "min_tokens": self.min_tokens, "min_center_n": self.min_center_n,
                "min_active": self.min_active, "stance_mode": self.stance_mode,
                "stance_min": self.stance_min, "stance_scale": self.stance_scale,
                "stance_hash": stance_hash(self.stance_pairs),
                "drop_domains": list(self.drop_domains),
                "balance_weight": self.balance_weight,
                "bands": self.bands, "use_veto": self.use_veto, "veto_margin": self.veto_margin,
                "veto_hash": veto_hash(self.veto_hypotheses),
                "n_veto": len(self.veto_hypotheses),
                "n_pairs": int(self.keep_.sum()), "fit": self.fit_}


def rows_to_frame(ids, rows, n_top=3):
    recs = []
    for i, r in zip(ids, rows):
        rec = {"id": i, "lr_score": r["score"], "lr_interpretation": r["interpretation"],
               "lr_status": r["status"],
               "lr_score_sd": _r(r["score_sd"]), "lr_score_lo": _r(r["score_lo"]),
               "lr_score_hi": _r(r["score_hi"]), "lr_dispersion": _r(r["dispersion"]),
               "lr_n_active": r["n_active"], "lr_topic_margin": _r(r["topic_margin"]),
               "lr_stance": _r(r.get("stance", np.nan)),
               "lr_veto": _r(r.get("veto", np.nan)),
               "lr_gate_reason": r.get("gate_reason"),
               "lr_is_relevant": r["is_relevant"]}
        for j in range(n_top):
            t = r["top"][j] if j < len(r["top"]) else None
            rec[f"lr_top{j+1}_domain"] = t["domain"] if t else None
            rec[f"lr_top{j+1}_position"] = t["position"] if t else None
            rec[f"lr_top{j+1}_contrast"] = t["contrast"] if t else None
        recs.append(rec)
    return pd.DataFrame(recs)


def _r(x, nd=4):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


# =============================================================================
# MARGIN CACHE  - the whole point of splitting the phases
# =============================================================================
def default_margins_path(output_csv):
    base = os.path.splitext(os.path.abspath(output_csv))[0]
    return base + ".margins.npz"


def save_margins(path, ids, topic, battery, meta, stance=None, veto=None,
                 battery2=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    arrs = dict(ids=np.asarray([str(i) for i in ids]),
                topic=np.asarray(topic, dtype=np.float32),
                battery=np.asarray(battery, dtype=np.float32),
                meta=np.asarray(json.dumps(meta, ensure_ascii=False)))
    if stance is not None:
        arrs["stance"] = np.asarray(stance, dtype=np.float32)
    if veto is not None:
        arrs["veto"] = np.asarray(veto, dtype=np.float32)
    if battery2 is not None:
        arrs["battery2"] = np.asarray(battery2, dtype=np.float32)
    np.savez_compressed(path, **arrs)


def slice_margins(src, dst, lo, hi):
    """Copy a cache keeping only topic-probe columns [lo:hi).

    Lets one GPU pass over a combined probe list serve several gate designs:
    cache all of them, then slice the cache and score each subset on CPU.
    Everything else -- ids, battery, veto, meta -- is carried over untouched.
    """
    ids, topic, batt, stance, veto, meta, batt2 = load_margins(src)
    topic = np.asarray(topic, float)[:, lo:hi]
    meta = dict(meta); meta["topic_slice"] = f"{lo}:{hi}"
    save_margins(dst, ids, topic, batt, meta, stance=stance, veto=veto,
                 battery2=batt2)
    logger.info("sliced %s [%d:%d] -> %s (%d probes)", src, lo, hi, dst, hi - lo)
    return dst


def load_margins(path):
    """Returns (ids, topic, battery, stance, veto, meta, battery2). stance, veto and
    battery2 are None for caches written before those probe sets existed."""
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    stance = z["stance"] if "stance" in z.files else None
    veto = z["veto"] if "veto" in z.files else None
    battery2 = z["battery2"] if "battery2" in z.files else None
    return list(z["ids"]), z["topic"], z["battery"], stance, veto, meta, battery2


# =============================================================================
# RUN  - phase 1 then phase 2
# =============================================================================
def run(input_csv=None, output_csv="/kaggle/working/lr_scores.csv",
        margins_path=None, input_pattern="*.csv", text_col="text_en",
        id_col="id", group_col=None, suffix="",
        model="mlburnham/Political_DEBATE_large_v1.0", revision=None,
        device=None, fp16=None, batch_size=None, text_chunk=128, max_length=512,
        gate="post", topic_margin_min=0.0, active_margin=0.0,
        standardize="none", contrast_scale="auto", active_only=True,
        bootstrap=200, seed=0, min_tokens=5, min_active=2,
        stance_mode="off", stance_min=0.0, stance_scale=4.0, drop_domains=(),
        veto_hypotheses=NEGATIVE_TOPIC_HYPOTHESES, veto_margin=0.0, use_veto=True,
        balance_weight=0.0, average_paraphrase=True, bands=None,
        off_topic_label=OFF_TOPIC_LABEL,
        paraphrase_battery=None, limit=None, sample_limit=True,
        checkpoint_every=2048, resume=False, fp16_check=True,
        run_diagnostics=True, battery=BATTERY, topic_hypotheses=TOPIC_HYPOTHESES,
        stance_pairs=STANCE_PAIRS, label_col=None, anchor=None,
        merge_input=True, scorer=None):
    """Score a CSV. Returns (frame, margins_path)."""
    t0 = time.time()
    input_csv = find_input(input_pattern, explicit=input_csv) if (ON_KAGGLE or input_csv is None) else input_csv
    if ON_KAGGLE and not str(output_csv).startswith("/kaggle/working"):
        logger.warning("output_csv is outside /kaggle/working - it will not "
                       "survive when the notebook is committed.")
    # Derive the cache path from the ORIGINAL output name, before any smoke-test
    # redirect below, so a smoke run shares the full run's margin cache.
    margins_path = margins_path or default_margins_path(output_csv)
    if limit and os.path.exists(output_csv):
        base, ext = os.path.splitext(output_csv)
        output_csv = f"{base}.smoke{ext}"
        logger.warning("limit=%s and the output already exists; writing to %s "
                       "instead so a smoke test cannot clobber a full run.",
                       limit, os.path.basename(output_csv))
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)) or ".", exist_ok=True)

    df = pd.read_csv(input_csv)
    for col, name in ((text_col, "text_col"), (id_col, "id_col")):
        if col not in df.columns:
            raise KeyError(f"{name} '{col}' not in {list(df.columns)}")
    if df[id_col].duplicated().any():
        raise ValueError(f"'{id_col}' is not unique; the margin cache is keyed on it")
    logger.info("Input has %d rows; limit=%r", len(df), limit)
    if limit:
        df = (df.sample(n=min(limit, len(df)), random_state=seed).reset_index(drop=True)
              if sample_limit else df.head(limit))
        logger.warning("LIMIT IS SET: scoring only %d rows (%s). Set limit=None for "
                       "the full run. If you edited CONFIG and then re-ran the cell "
                       "that DEFINES CONFIG, your edit was overwritten.",
                       len(df), "random sample" if sample_limit else "first rows")
    else:
        logger.info("Full run: all %d rows", len(df))

    hyps = battery_hypotheses(battery)
    st_hyps = stance_hypotheses(stance_pairs)
    vt_hyps = list(veto_hypotheses or [])
    # BATTERY_PARAPHRASE is the ECONOMIC paraphrase set and there is no populism
    # equivalent, so it can only be paired with the economic battery. It used to
    # be the hard default, which meant any other battery raised on a mismatch the
    # caller never asked for. Default None and pair it only when the domains line
    # up; an explicitly-passed mismatch is still an error, because that one is a
    # real mistake rather than a stale default.
    if paraphrase_battery is None and average_paraphrase:
        if BATTERY_PARAPHRASE and [d for d, *_ in BATTERY_PARAPHRASE] == [d for d, *_ in battery]:
            paraphrase_battery = BATTERY_PARAPHRASE
        else:
            logger.info("no paraphrase battery for these domains; "
                        "average_paraphrase disabled")
            average_paraphrase = False
    p2_hyps = (battery_hypotheses(paraphrase_battery) if average_paraphrase
               and paraphrase_battery else [])
    if p2_hyps and [d for d, *_ in paraphrase_battery] != [d for d, *_ in battery]:
        raise ValueError("paraphrase_battery must have the same domains, in the "
                         "same order, as battery")
    meta_base = {"model": model, "revision": revision, "max_length": max_length,
                 "battery_hash": battery_hash(battery, topic_hypotheses),
                 "stance_hash": stance_hash(stance_pairs),
                 "stance_pairs": stance_pairs,
                 "veto_hash": veto_hash(veto_hypotheses),
                 "veto_hypotheses": list(veto_hypotheses or []),
                 "n_pairs": len(battery), "domains": [d for d, _, _, _ in battery],
                 "topic_hypotheses": topic_hypotheses, "gate": gate,
                 "text_col": text_col, "input_csv": os.path.abspath(input_csv),
                 "torch": getattr(torch, "__version__", None)}

    # ---------------------------------------------------------------- resume
    cache_ids, cache_topic, cache_batt, cache_st, cache_vt, cache_b2 = \
        [], None, None, None, None, None
    if resume and os.path.exists(margins_path):
        try:
            cache_ids, cache_topic, cache_batt, cache_st, cache_vt, old, cache_b2 = \
                load_margins(margins_path)
            bad = old.get("battery_hash") != meta_base["battery_hash"]
            if not bad and cache_st is None and stance_mode != "off":
                logger.warning("Cached margins predate the stance gate; recomputing "
                               "so the gate has something to work with.")
                bad = True
            if not bad and p2_hyps and cache_b2 is None:
                logger.warning("Cached margins have no paraphrase pass; recomputing.")
                bad = True
            if not bad and use_veto and vt_hyps and (
                    cache_vt is None or old.get("veto_hash") != meta_base["veto_hash"]):
                logger.warning("Cached margins lack matching veto probes; recomputing.")
                bad = True
            if not bad and cache_st is not None and \
                    old.get("stance_hash") != meta_base["stance_hash"]:
                logger.warning("Cached stance margins came from different probes; "
                               "recomputing.")
                bad = True
            if bad:
                logger.warning("Ignoring cache (battery %s vs %s).",
                               old.get("battery_hash"), meta_base["battery_hash"])
                cache_ids, cache_topic, cache_batt, cache_st, cache_vt, cache_b2 = \
                    [], None, None, None, None, None
            else:
                logger.info("Resuming: %d texts already have margins", len(cache_ids))
        except Exception as e:
            logger.warning("Could not read margin cache (%s); starting fresh", e)
            cache_ids, cache_topic, cache_batt, cache_st, cache_vt, cache_b2 = \
                [], None, None, None, None, None

    done = set(cache_ids)
    todo = df[~df[id_col].astype(str).isin(done)].reset_index(drop=True)
    logger.info("Phase 1: margins for %d of %d rows | %d topic + %d battery + %d "
                "stance hypotheses per text (%d pairs)", len(todo), len(df),
                len(topic_hypotheses), len(hyps), len(st_hyps),
                len(topic_hypotheses) + len(hyps) + len(st_hyps))

    # --------------------------------------------------------------- phase 1
    if len(todo):
        if scorer is None:
            scorer = MarginScorer(model, device=device, fp16=fp16,
                                  revision=revision, max_length=max_length)
        globals()["SCORER"] = scorer
        check_language(todo[text_col], scorer.model.config.vocab_size)
        if fp16_check:
            meta_base["fp16_check"] = scorer.fp16_check(todo[text_col], hyps[:6])

        new_ids, new_topic, new_batt, new_st, new_vt, new_b2 = [], [], [], [], [], []
        blocks = [todo.iloc[s:s + checkpoint_every]
                  for s in range(0, len(todo), checkpoint_every)]
        for bi, block in enumerate(blocks):
            texts = block[text_col].tolist()
            texts = [t if isinstance(t, str) else "" for t in texts]
            # collapse exact duplicates inside the block
            uniq, inv = np.unique(np.asarray(texts, dtype=object), return_inverse=True)
            uniq = list(uniq)
            if len(uniq) < len(texts):
                logger.info("block %d/%d: %d texts -> %d unique",
                            bi + 1, len(blocks), len(texts), len(uniq))

            tm_u = scorer.margin_matrix(uniq, topic_hypotheses,
                                        batch_size=batch_size, text_chunk=text_chunk)
            bm_u = np.full((len(uniq), len(hyps)), np.nan, dtype=np.float32)
            if gate == "pre":
                agg_probe = ContrastAggregator(battery, topic_hypotheses,
                                               topic_margin_min=topic_margin_min)
                keep = np.where(agg_probe._topic(tm_u) >= topic_margin_min)[0]
                logger.info("block %d/%d: topic gate keeps %d/%d (%.0f%%)",
                            bi + 1, len(blocks), len(keep), len(uniq),
                            100 * len(keep) / max(len(uniq), 1))
                if len(keep):
                    bm_u[keep] = scorer.margin_matrix(
                        [uniq[i] for i in keep], hyps,
                        batch_size=batch_size, text_chunk=text_chunk)
            else:
                bm_u = scorer.margin_matrix(uniq, hyps, batch_size=batch_size,
                                            text_chunk=text_chunk)

            sm_u = scorer.margin_matrix(uniq, st_hyps, batch_size=batch_size,
                                        text_chunk=text_chunk)
            vm_u = (scorer.margin_matrix(uniq, vt_hyps, batch_size=batch_size,
                                         text_chunk=text_chunk)
                    if vt_hyps else np.zeros((len(uniq), 0), np.float32))
            p2_u = (scorer.margin_matrix(uniq, p2_hyps, batch_size=batch_size,
                                         text_chunk=text_chunk)
                    if p2_hyps else None)

            new_ids += [str(i) for i in block[id_col]]
            new_topic.append(tm_u[inv])
            new_batt.append(bm_u[inv])
            new_st.append(sm_u[inv])
            new_vt.append(vm_u[inv])
            if p2_u is not None:
                new_b2.append(p2_u[inv])

            ids_all = list(cache_ids) + new_ids
            top_all = np.vstack(([cache_topic] if cache_topic is not None else []) + new_topic)
            bat_all = np.vstack(([cache_batt] if cache_batt is not None else []) + new_batt)
            st_all = np.vstack(([cache_st] if cache_st is not None else []) + new_st)
            vt_all = np.vstack(([cache_vt] if cache_vt is not None else []) + new_vt) \
                if vt_hyps else None
            b2_all = np.vstack(([cache_b2] if cache_b2 is not None else []) + new_b2) \
                if new_b2 else None
            save_margins(margins_path, ids_all, top_all, bat_all,
                         {**meta_base, "n_nonfinite": scorer.n_nonfinite,
                          "dtype": str(scorer.dtype)}, stance=st_all, veto=vt_all,
                         battery2=b2_all)
            rate = len(new_ids) / max(time.time() - t0, 1e-9)
            logger.info("phase 1: %d/%d  %.1f texts/s  eta %.1f min",
                        len(new_ids), len(todo), rate,
                        (len(todo) - len(new_ids)) / max(rate, 1e-9) / 60)

        if scorer.n_nonfinite:
            logger.error("%d non-finite margins (fp16 overflow). Re-run with fp16=False.",
                         scorer.n_nonfinite)

    # --------------------------------------------------------------- phase 2
    out, _ = rescore(margins_path, df=df, output_csv=output_csv, text_col=text_col,
                     id_col=id_col, group_col=group_col, suffix=suffix,
                     topic_margin_min=topic_margin_min, active_margin=active_margin,
                     standardize=standardize, contrast_scale=contrast_scale,
                     active_only=active_only, bootstrap=bootstrap, seed=seed,
                     min_tokens=min_tokens, min_active=min_active,
                     stance_mode=stance_mode, stance_min=stance_min,
                     stance_scale=stance_scale, stance_pairs=stance_pairs,
                     drop_domains=drop_domains, veto_hypotheses=veto_hypotheses,
                     veto_margin=veto_margin, use_veto=use_veto,
                     balance_weight=balance_weight, bands=bands,
                     off_topic_label=off_topic_label,
                     average_paraphrase=average_paraphrase,
                     battery=battery, topic_hypotheses=topic_hypotheses,
                     label_col=label_col, anchor=anchor,
                     merge_input=merge_input, run_diagnostics=run_diagnostics)
    logger.info("total elapsed: %.1f min", (time.time() - t0) / 60)
    return out, margins_path


def rescore(margins_path, df=None, input_csv=None, output_csv=None,
            text_col="text_en", id_col="id", group_col=None, suffix="",
            topic_margin_min=0.0, active_margin=0.0, standardize="none",
            contrast_scale="auto", active_only=True, bootstrap=200, seed=0,
            min_tokens=5, min_active=2, stance_mode="off", stance_min=0.0,
            stance_scale=4.0, stance_pairs=STANCE_PAIRS, drop_domains=(),
            veto_hypotheses=NEGATIVE_TOPIC_HYPOTHESES, veto_margin=0.0, use_veto=True,
            balance_weight=0.0, average_paraphrase=True, bands=None,
            off_topic_label=OFF_TOPIC_LABEL, admit_mask=None,
            battery=BATTERY, topic_hypotheses=TOPIC_HYPOTHESES,
            label_col=None, anchor=None, merge_input=True, run_diagnostics=True):
    """Re-derive scores from cached margins. No GPU, seconds not hours.

    This is where every judgement call now lives, so trying an alternative is
    cheap enough that you have no excuse for not reporting the sensitivity.
    """
    ids, topic, batt, stance, veto, meta, batt2 = load_margins(margins_path)
    if stance is None and stance_mode != "off":
        logger.warning("This margin cache has no stance columns (written before the "
                       "stance gate); continuing with stance_mode='off'.")
        stance_mode = "off"
    if meta.get("battery_hash") != battery_hash(battery, topic_hypotheses):
        raise ValueError("battery does not match the one used to build the cache "
                         f"({meta.get('battery_hash')}). Re-run phase 1.")
    if batt2 is not None and average_paraphrase:
        # The contrast is linear in the margins, so averaging the two margin
        # matrices IS averaging each side across wordings before contrasting.
        # Zero-shot NLI is phrasing-sensitive; two independent wordings halve
        # the template noise for one extra battery pass in phase 1.
        batt = (batt + batt2) / 2.0
        logger.info("averaging the battery with its paraphrase (%d + %d hypotheses)",
                    batt.shape[1], batt2.shape[1])
    elif batt2 is None and average_paraphrase:
        logger.info("no paraphrase columns in this cache; using the single battery. "
                    "Re-run phase 1 with average_paraphrase=True to enable it.")
    if df is None:
        input_csv = input_csv or meta.get("input_csv")
        if not input_csv or not os.path.exists(input_csv):
            raise FileNotFoundError(
                "rescore() needs the source CSV to merge against. Pass input_csv= "
                f"(the cache recorded {meta.get('input_csv')!r}).")
        df = pd.read_csv(input_csv)
    if text_col not in df.columns:
        # the cache knows which column phase 1 actually read
        text_col = meta.get("text_col") if meta.get("text_col") in df.columns else None

    agg = ContrastAggregator(battery, topic_hypotheses,
                             topic_margin_min=topic_margin_min,
                             active_margin=active_margin, standardize=standardize,
                             contrast_scale=contrast_scale, active_only=active_only,
                             bootstrap=bootstrap, seed=seed, min_tokens=min_tokens,
                             min_active=min_active, stance_pairs=stance_pairs,
                             stance_mode=stance_mode, stance_min=stance_min,
                             stance_scale=stance_scale, drop_domains=drop_domains,
                             veto_hypotheses=veto_hypotheses,
                             veto_margin=veto_margin,
                             balance_weight=balance_weight, bands=bands,
                             off_topic_label=off_topic_label,
                             use_veto=use_veto and veto is not None).fit(batt, topic)

    id_to_text = (dict(zip(df[id_col].astype(str), df[text_col]))
                  if text_col and text_col in df.columns else {})
    texts = [id_to_text.get(i) for i in ids] if id_to_text else None
    rows = agg.transform(batt, topic, texts=texts, stance_margins=stance,
                         veto_margins=veto, admit_mask=admit_mask)
    scores = rows_to_frame(ids, rows).rename(columns={"id": id_col})

    scores["lr_battery_hash"] = meta.get("battery_hash")
    scores["lr_standardize"] = standardize
    scores["lr_scale"] = round(agg.fit_["scale"], 4)
    scores["lr_active_only"] = active_only
    scores["lr_stance_mode"] = agg.stance_mode
    scores["lr_dropped"] = ",".join(agg.drop_domains) or None
    scores["lr_veto_on"] = bool(agg.use_veto and veto is not None)
    scores["lr_paraphrased"] = bool(batt2 is not None and average_paraphrase)
    scores["lr_balance_weight"] = agg.balance_weight
    scores["lr_gate_fitted"] = admit_mask is not None

    out = scores
    if merge_input:
        keyed = df.copy()
        orig_dtype = keyed[id_col].dtype        # ids are keyed as strings in the
        keyed[id_col] = keyed[id_col].astype(str)   # cache; restore the caller's
        out = keyed.merge(scores, on=id_col, how="left")   # dtype on the way out,
        try:                                    # so merging the result against the
            out[id_col] = out[id_col].astype(orig_dtype)   # source CSV just works
        except (TypeError, ValueError):
            pass
    if suffix:
        out = out.rename(columns={c: c + suffix for c in out.columns
                                  if c.startswith("lr_")})
    if output_csv:
        out.to_csv(output_csv, index=False)
        sidecar = os.path.splitext(output_csv)[0] + ".runmeta.json"
        with open(sidecar, "w") as fh:
            json.dump({**meta, "aggregator": agg.config(),
                       "written": time.strftime("%Y-%m-%d %H:%M:%S")}, fh, indent=2)
        logger.info("Wrote %s (+ %s)", output_csv, os.path.basename(sidecar))

    summarise(out, suffix)
    if run_diagnostics:
        diagnose(out, batt, topic, agg, ids, text_col=text_col, id_col=id_col,
                 group_col=group_col, suffix=suffix, df=df, stance=stance,
                 label_col=label_col, anchor=anchor)
    return out, agg


def summarise(out, suffix=""):
    sc, st = f"lr_score{suffix}", f"lr_status{suffix}"
    scored = out[out[sc].notna()]
    logger.info("=" * 62)
    logger.info("scored: %d / %d (%.1f%%)", len(scored), len(out),
                100 * len(scored) / max(len(out), 1))
    logger.info("status:\n%s", out[st].value_counts().to_string())
    if len(scored):
        logger.info("score  mean %.2f  sd %.2f  min %.2f  max %.2f",
                    scored[sc].mean(), scored[sc].std(), scored[sc].min(), scored[sc].max())
        logger.info("bands:\n%s", scored[f"lr_interpretation{suffix}"].value_counts().to_string())
        sd_col = scored[f"lr_score_sd{suffix}"]
        if sd_col.notna().any():
            logger.info("bootstrap SD of score: median %.2f  (p90 %.2f)",
                        sd_col.median(), sd_col.quantile(0.9))
        logger.info("active pairs per text: median %.0f  mean %.1f",
                    scored[f"lr_n_active{suffix}"].median(),
                    scored[f"lr_n_active{suffix}"].mean())


# =============================================================================
# DIAGNOSTICS - checks that are able to fail
# =============================================================================
def _pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def diagnose(out, batt, topic, agg, ids, text_col="text_en", id_col="id",
             group_col=None, suffix="", df=None, top_n=8, stance=None,
             label_col=None, anchor=None):
    """Report the failure modes v1 had no way to detect."""
    sc = f"lr_score{suffix}"
    by_id = dict(zip(out[id_col].astype(str), out[sc])) if id_col in out.columns else {}
    score = np.array([by_id.get(str(i), np.nan) for i in ids], dtype=float)
    scored = np.isfinite(score)

    contrast, active = agg._contrasts(batt)
    pos = agg._positions(contrast)
    w = agg.weights
    P = len(agg.domains)

    print("\n" + "=" * 74)
    print("DIAGNOSTICS")
    print("=" * 74)

    # --- topic gate --------------------------------------------------------
    tm = agg._topic(topic)
    passing = tm >= agg.topic_margin_min
    print(f"\ntopic gate (2nd-highest margin >= {agg.topic_margin_min:g})")
    print(f"  pass rate            : {passing.mean():6.1%}  ({passing.sum()}/{len(tm)})")
    print(f"  2nd-highest percentiles (10/50/90): "
          f"{np.percentile(tm[np.isfinite(tm)], [10, 50, 90]).round(2).tolist()}")
    if passing.mean() > 0.95:
        print("  NOTE: the gate is close to inert. Either raise topic_margin_min or")
        print("        drop the stage and save the compute.")

    # --- the veto contrast -------------------------------------------------
    st_col = f"lr_status{suffix}"
    rs_col = f"lr_gate_reason{suffix}"
    if rs_col in out.columns and (out[rs_col] == "veto").any():
        n_v = int((out[rs_col] == "veto").sum())
        n_t = int((out[rs_col] == "topic").sum())
        print(f"\nveto contrast (max positive - max negative > {agg.veto_margin:g})")
        print(f"  dropped by weak topic evidence : {n_t}")
        print(f"  dropped by the veto contrast   : {n_v}")
        print(f"  admitted                       : {int((out[rs_col]=='admitted').sum())}")
        print("  The veto only fires on rows that HAD positive evidence, so it is")
        print("  aimed squarely at the rhetoric false positives. Check against a")
        print("  coded sample that it is not also taking welfare-chauvinism text.")

    # --- stance gate -------------------------------------------------------
    if stance is not None and agg.stance_mode != "off":
        sv = agg.stance(stance)
        st_col = f"lr_status{suffix}"
        n_desc = int((out[st_col] == "descriptive").sum()) if st_col in out else 0
        print(f"\nstance gate (mode={agg.stance_mode}, min={agg.stance_min:g})")
        print(f"  stance percentiles (10/50/90): "
              f"{np.percentile(sv[np.isfinite(sv)], [10, 50, 90]).round(2).tolist()}")
        print(f"  dropped as descriptive       : {n_desc}")
        # Does the gate actually do the job it was added for? Re-score with it
        # off and compare the tails; if nothing moves, the probes are not
        # separating advocacy from description and the gate is dead weight.
        try:
            bare = ContrastAggregator(
                agg.battery, agg.topic_hypotheses,
                topic_margin_min=agg.topic_margin_min, active_margin=agg.active_margin,
                standardize=agg.standardize, contrast_scale=agg.fit_["scale"],
                active_only=agg.active_only, bootstrap=0, min_tokens=agg.min_tokens,
                min_active=agg.min_active, stance_mode="off")
            bare.fit_ = dict(agg.fit_)
            b = np.array([r["score"] for r in bare.transform(batt, topic)], float)
            g = score
            for lab, v in (("stance OFF", b), ("stance ON ", g)):
                v = v[np.isfinite(v)]
                if v.size:
                    print(f"  {lab}: n={v.size:4d}  mean {v.mean():5.2f}  "
                          f"StrongLeft {int((v < 1.43).sum()):3d}  "
                          f"StrongRight {int((v > 8.57).sum()):3d}")
            print("  If the tails barely move, the probes are not separating advocacy")
            print("  from description -- set stance_mode='off' rather than pretending.")
        except Exception as e:                    # diagnostics must never break a run
            logger.debug("stance comparison skipped: %s", e)

    # --- the confound v1 could not distinguish from signal -----------------
    if df is not None and text_col and text_col in df.columns:
        tmap = dict(zip(df[id_col].astype(str), df[text_col]))
        ntok = np.array([len(str(tmap.get(str(i), "")).split()) for i in ids], float)
        r_len = _pearson(np.abs(score - 5)[scored], ntok[scored])
        r_act = _pearson(np.abs(score - 5)[scored], active.sum(1)[scored])
        n_s = int(scored.sum())
        print(f"\nlength / breadth confound   (n={n_s} scored)")
        print(f"  corr(|score-5|, tokens)      : {r_len:+.3f}")
        print(f"  corr(|score-5|, active pairs): {r_act:+.3f}")
        if n_s < 200:
            # r on 60 rows has a 95% interval of roughly +-0.25; reading a
            # confound off that is reading noise
            print(f"  n={n_s} is too small to interpret: the 95% interval on r is "
                  f"about +/-{1.96 / np.sqrt(max(n_s - 3, 1)):.2f}.")
        elif abs(r_len) > 0.25:
            print("  WARNING: extremity still tracks text length. Check active_only=True.")
        if n_s < 200:
            pass
        elif r_act < -0.15:
            # Extremity RISING as evidence FALLS is the saturation signature: a
            # score resting on one pair is that pair's contrast, which the model
            # puts near +/-10 logits, so tanh sends it to the end of the scale.
            # Those rows are not confident, they are unsupported -- raise
            # min_active rather than reading their tails.
            print(f"  WARNING: extremity RISES as active pairs FALL ({r_act:+.3f}).")
            print("           Thinly-evidenced texts are landing at the ends of the")
            print("           scale. Check min_active; the tails are not trustworthy.")
        elif abs(r_len) <= 0.25 and abs(r_act) <= 0.25:
            print("  (both small: position is not being driven by how much a text says)")
        elif abs(r_act) > 0.25:
            print(f"  WARNING: extremity tracks the NUMBER of active pairs ({r_act:+.3f}).")

    # --- per-pair behaviour ------------------------------------------------
    print(f"\nper-pair activation and mean position ({P} pairs, + = right)")
    if agg.standardize == "none":
        print("  standardize='none': nothing is centred, so mean pos|act simply")
        print("  restates raw R% -- a pair near +-0.5 fires almost always one way.")
    else:
        print("  mean pos|act near 0 means the pair's template bias is centred out;")
        print("  a pair with n_ctr below min_center_n is left uncentred.")
    print(f"  {'domain':<26}{'act%':>7}{'n_ctr':>7}{'raw R%':>8}"
          f"{'mean pos|act':>14}{'med contrast':>14}")
    act_rate = active.mean(0)
    kept = agg.keep_ if hasattr(agg, "keep_") else np.ones(P, bool)
    for p in np.argsort(-act_rate):
        if not kept[p]:
            continue
        a = active[:, p]
        vals = pos[a, p][np.isfinite(pos[a, p])] if a.any() else np.array([])
        cs = contrast[a, p][np.isfinite(contrast[a, p])] if a.any() else np.array([])
        mp = float(vals.mean()) if vals.size else float("nan")
        mc = float(np.median(cs)) if cs.size else float("nan")
        nc = agg.fit_.get("n_active_rows_per_pair", [0] * P)[p]
        flag = " *" if nc < agg.min_center_n else ""
        rr = float(np.mean(contrast[a, p] > 0)) if a.any() else float("nan")
        print(f"  {agg.domains[p]:<26}{100*act_rate[p]:6.1f}%{nc:7d}{100*rr:7.0f}%"
              f"{mp:14.3f}{mc:14.2f}{flag}")
    if (~kept).any():
        print(f"  ({(~kept).sum()} pair(s) excluded from aggregation: "
              f"{', '.join(agg.drop_domains)})")
    raw_r = np.array([np.mean(contrast[active[:, p], p] > 0) if active[:, p].any()
                      else np.nan for p in range(P)])
    n_keep = int(kept.sum())
    n_right = int(np.nansum(raw_r[kept] > 0.5))
    print(f"\n  raw R% = share of ACTIVE cells leaning right BEFORE centring.")
    print(f"  {n_right}/{n_keep} active pairs read right on a median active text.")
    if abs(n_right - n_keep / 2) > n_keep / 6:
        print("  That imbalance is a property of the templates, not of the corpus:")
        print("  centring removes it per pair, but it compresses the majority side")
        print("  and stretches the minority side, which skews the score TAILS.")
        print("  Check it with compare_settings(): standardize='none' keeps the raw")
        print("  scale, so if the tail asymmetry vanishes there it is this artefact.")
    n_ctr_all = agg.fit_.get("n_active_rows_per_pair", [0] * P)
    thin = [(agg.domains[p], n_ctr_all[p]) for p in range(P)
            if kept[p] and n_ctr_all[p] < agg.min_center_n]
    if thin and agg.standardize == "none":
        thin = []                       # nothing is centred, so nothing is uncentred
    if thin:
        need = max(int(np.ceil(agg.min_center_n * len(ids) / max(n, 1))) for _, n in thin)
        rarest, rn = min(thin, key=lambda t: t[1])
        print(f"\n  * {len(thin)}/{P} pairs uncentred (< {agg.min_center_n} active rows).")
        print(f"    Extrapolating from this sample, every pair reaches n="
              f"{agg.min_center_n} at roughly {need:,} input rows")
        print(f"    (binding pair: {rarest}, n={rn} in {len(ids):,} rows).")
        print("    Until then the centres are 0 and the scores are provisional.")

    # --- tail balance ------------------------------------------------------
    s = score[scored]
    if s.size > 50:
        n_sl, n_sr = int((s < 1.43).sum()), int((s > 8.57).sum())
        bd = agg.bands or BANDS
        print(f"\nscore tails   {bd[0][1]} {n_sl}   {bd[-1][1]} {n_sr}   "
              f"mean {s.mean():.2f}")
        if min(n_sl, n_sr) * 5 < max(n_sl, n_sr) and max(n_sl, n_sr) > 10:
            print("  The tails are lopsided. Before reading this as a finding, confirm")
            print("  it survives standardize='none' -- see the raw R% note above.")

    # --- leave-one-pair-out ------------------------------------------------
    rows = scored & passing & (active.sum(1) > 1)
    if rows.sum() > 10:
        A, V = active[rows].astype(float), np.nan_to_num(pos[rows])
        num, den = (w * V * A).sum(1), (w * A).sum(1)
        full = np.clip(5 + 5 * num / np.maximum(den, 1e-9), 0, 10)
        print(f"\nleave-one-pair-out jackknife on {rows.sum()} scored texts")
        print(f"  {'dropped pair':<26}{'mean |dscore|':>15}{'max |dscore|':>14}")
        shifts = []
        for p in range(P):
            np_, dp = num - w[p] * V[:, p] * A[:, p], den - w[p] * A[:, p]
            ok = dp > 1e-9
            alt = np.clip(5 + 5 * np_[ok] / dp[ok], 0, 10)
            d = np.abs(alt - full[ok])
            shifts.append((float(d.mean()) if d.size else 0.0,
                           float(d.max()) if d.size else 0.0, agg.domains[p]))
        for mean_d, max_d, dom in sorted(shifts, reverse=True)[:top_n]:
            print(f"  {dom:<26}{mean_d:15.3f}{max_d:14.3f}")
        worst = max(shifts)[0]
        if worst > 0.5:
            print(f"  WARNING: dropping one pair moves the mean score by {worst:.2f} points.")
            print("           The result rides on a small number of hypotheses.")

    # --- validation against a human label ----------------------------------
    if label_col and df is not None and label_col in df.columns:
        lmap = dict(zip(df[id_col].astype(str), df[label_col]))
        lab = np.array([lmap.get(str(i), np.nan) for i in ids], dtype=float)
        ok = scored & np.isfinite(lab)
        if ok.sum() > 30:
            print(f"\nvalidation against '{label_col}'  (n={int(ok.sum())})")
            print(f"  r(score, {label_col})            : {_pearson(score[ok], lab[ok]):+.3f}")
            vals = np.unique(lab[ok])
            if 1 < vals.size <= 8:               # graded or binary: dose-response
                print(f"  {'value':>8}{'n':>7}{'mean score':>13}")
                for v in vals:
                    m = ok & (lab == v)
                    if m.sum() >= 5:
                        print(f"  {v:8.0f}{int(m.sum()):7d}{score[m].mean():13.2f}")
                print("  A monotone gradient here is the strongest evidence available")
                print("  that the instrument measures what the coders measured.")
            if group_col and group_col in df.columns:
                gmap = dict(zip(df[id_col].astype(str), df[group_col]))
                gr = np.array([gmap.get(str(i)) for i in ids], dtype=object)
                dm = pd.get_dummies(pd.Series(gr[ok])).to_numpy(float)
                X = np.hstack([np.ones((int(ok.sum()), 1)), dm[:, 1:]])
                with np.errstate(all="ignore"):   # BLAS chatter on some builds
                    pinv = np.linalg.pinv(X)
                    res = lambda v: v - X @ (pinv @ v)
                    pr = _pearson(res(score[ok]), res(lab[ok]))
                print(f"  same, within {group_col} (partial): {pr:+.3f}")
                print("  If the partial is far below the raw r, the score is tracking")
                print("  who is speaking rather than what the sentence says.")

            # which pairs actually carry the validated signal
            print(f"\n  per-pair validity: r(position, {label_col}) among ACTIVE rows")
            print(f"  {'domain':<26}{'n':>7}{'r':>9}")
            vr = []
            for p in range(P):
                m = ok & active[:, p] & np.isfinite(pos[:, p])
                if m.sum() >= 30:
                    vr.append((_pearson(pos[m, p], lab[m]), agg.domains[p], int(m.sum())))
            for r_, dom, n_ in sorted(vr):
                print(f"  {dom:<26}{n_:7d}{r_:+9.3f}")
            if vr:
                print("  Pairs near zero contribute noise. Drop or reweight them here")
                print("  rather than guessing which hypotheses are working.")
                # A pair that fires one way on almost every active text has its
                # centre dragged to that side, so the rare opposite reading gets
                # stretched into the tail -- and it tends to carry little signal.
                imb, val = [], []
                for r_, dom, _ in vr:
                    p = agg.domains.index(dom)
                    if np.isfinite(raw_r[p]):
                        imb.append(abs(raw_r[p] - 0.5)); val.append(abs(r_))
                if len(imb) >= 6:
                    rr = _pearson(np.array(imb), np.array(val))
                    print(f"\n  r(|raw R% - 50|, |validity|) = {rr:+.3f} over "
                          f"{len(imb)} pairs")
                    if rr < -0.3:
                        print("  A pair that fires one way on almost every active text")
                        print("  carries less signal. Rewriting the lopsided ones so both")
                        print("  sides can fire is more promising than dropping them.")

    # --- external anchoring ------------------------------------------------
    if group_col and df is not None and group_col in df.columns:
        gmap = dict(zip(df[id_col].astype(str), df[group_col]))
        groups = np.array([gmap.get(str(i)) for i in ids], dtype=object)
        print(f"\ngroup means by '{group_col}' (bootstrap 90% CI over texts)")
        print("  coverage = share of that group's rows that got scored. Groups with")
        print("  very different coverage are NOT comparable on the mean alone.")
        print(f"  {'group':<24}{'n':>7}{'cov':>7}{'mean':>8}{'  90% CI':>16}")
        rng = np.random.default_rng(0)
        rowsg = []
        for g in pd.unique(groups[scored]):
            s = score[scored & (groups == g)]
            if s.size < 3:
                continue
            cov = float(s.size / max(int((groups == g).sum()), 1))
            bs = np.array([rng.choice(s, s.size, replace=True).mean() for _ in range(500)])
            rowsg.append((float(s.mean()), g, s.size, np.percentile(bs, [5, 95]), cov))
        for mean, g, n, ci, cov in sorted(rowsg):
            print(f"  {str(g):<24}{n:7d}{100*cov:6.0f}%{mean:8.2f}   "
                  f"[{ci[0]:5.2f}, {ci[1]:5.2f}]")
        covs = [c for *_, c in rowsg]
        if covs and max(covs) > 2 * min(covs):
            print(f"  WARNING: coverage ranges {100*min(covs):.0f}%-{100*max(covs):.0f}%. "
                  "The gate admits")
            print("  these groups at very different rates, so these means rest on")
            print("  non-comparable subsamples. Report coverage alongside every mean.")
        if anchor:
            common = [(g, anchor[g], m) for m, g, *_ in rowsg if g in anchor]
            if len(common) >= 3:
                exp = np.array([a for _, a, _ in common], float)
                obs = np.array([o for _, _, o in common], float)
                er, orr = pd.Series(exp).rank().to_numpy(), pd.Series(obs).rank().to_numpy()
                print(f"\n  against the supplied anchor ({len(common)} groups)")
                print(f"  {'group':<24}{'anchor':>9}{'observed':>10}{'rank':>12}")
                for (g, a, o), re_, ro_ in sorted(zip(common, er, orr), key=lambda t: t[0][1]):
                    mark = "" if re_ == ro_ else f"  <- {int(re_)}->{int(ro_)}"
                    print(f"  {str(g):<24}{a:9.2f}{o:10.2f}{'':>6}{mark}")
                print(f"  Spearman {_pearson(er, orr):+.3f}   Pearson {_pearson(exp, obs):+.3f}")
        print("  Check this ordering against an external expert scale for whatever")
        print("  this battery measures -- CHES lrecon for economic left-right, or the")
        print("  populism literature's party placements. If it does not reproduce,")
        print("  nothing downstream of this pipeline is safe to interpret.")
    elif df is not None:
        cands = [c for c in ("faction", "party", "partei", "group")
                 if c in getattr(df, "columns", [])]
        print("\nGROUP VALIDATION NOT RUN -- group_col was not set.")
        print("  This is the only diagnostic here that can falsify the pipeline;")
        print("  everything above only checks internal consistency.")
        if cands:
            print(f"  Your input has {cands[0]!r}. Re-run phase 2 on CPU with "
                  f"group_col={cands[0]!r}.")
    print("=" * 74 + "\n")


def compare_settings(margins_path, battery=BATTERY, topic_hypotheses=TOPIC_HYPOTHESES,
                     settings=None, bootstrap=0, df=None, label_col=None,
                     id_col="id", drop_domains=()):
    """Score the same margins under several phase-2 settings and compare.

    Answers 'which aggregation should I use?' with evidence rather than a
    default. Costs seconds because no GPU is involved.
    """
    ids, topic, batt, stance, veto, meta, batt2 = load_margins(margins_path)
    settings = settings or [
        dict(standardize="none", active_only=True, stance_mode="off"),
        dict(standardize="center", active_only=True, stance_mode="off"),
        dict(standardize="z", active_only=True, stance_mode="off"),
        dict(standardize="center", active_only=False, stance_mode="off"),
        dict(standardize="center", active_only=True, stance_mode="gate"),
        dict(standardize="center", active_only=True, stance_mode="shrink"),
    ]
    lab = None
    if df is not None and label_col and label_col in df.columns:
        lmap = dict(zip(df[id_col].astype(str), df[label_col]))
        lab = np.array([lmap.get(str(i), np.nan) for i in ids], dtype=float)
    lvl = logger.level
    logger.setLevel(logging.ERROR)
    print("\n" + "=" * 74)
    print("PHASE-2 SENSITIVITY")
    print(f"  {'setting':<44}{'n':>6}{'mean':>7}{'sd':>7}{'p10':>7}{'p90':>7}"
          f"{'r vs 1st':>9}{'r vs label':>10}")
    ref = None
    for s in settings:
        agg = ContrastAggregator(battery, topic_hypotheses, bootstrap=bootstrap,
                                 drop_domains=drop_domains, **s).fit(batt, topic)
        rows = agg.transform(batt, topic, stance_margins=stance, veto_margins=veto)
        v = np.array([r["score"] for r in rows], float)
        ok = np.isfinite(v)
        if ref is None:
            ref = v
        label = (f"std={s.get('standardize')}, active_only={s.get('active_only')}, "
                 f"stance={s.get('stance_mode')}")
        extra = ""
        if lab is not None:
            m = ok & np.isfinite(lab)
            extra = f"{_pearson(v[m], lab[m]):9.3f}" if m.sum() > 20 else "        -"
        print(f"  {label:<44}{ok.sum():6d}{np.nanmean(v):7.2f}{np.nanstd(v):7.2f}"
              f"{np.nanpercentile(v[ok], 10):7.2f}{np.nanpercentile(v[ok], 90):7.2f}"
              f"{_pearson(v, ref):9.3f}{extra}")
    if lab is not None:
        print(f"\n  last column = r(score, {label_col}). That is the column that says")
        print("  which setting is more VALID; the rest only say what it looks like.")
        print("  A high 'r vs 1st' means the settings rank texts almost identically,")
        print("  so the choice moves the scale rather than the ordering.")
    logger.setLevel(lvl)
    print("=" * 74 + "\n")


def domain_sweep(margins_path, df, label_col, id_col="id", text_col=None,
                 battery=BATTERY, topic_hypotheses=TOPIC_HYPOTHESES,
                 holdout=0.5, seed=0, max_drop=5, min_gain=0.005, n_splits=8,
                 fixed_drop=(), **agg_kw):
    """Leave-one-domain-out validity sweep against a human label, split-half.

    Which pairs actually carry signal is an empirical question, and the answer
    differs by corpus. This drops each pair in turn, re-scores on CPU, and
    measures how the correlation with `label_col` moves.

    SPLIT-HALF IS NOT OPTIONAL. Choosing domains to maximise agreement with a
    label and then reporting that agreement as validation is circular: the
    selection is fitted to the criterion. Domains are chosen on one half and the
    result is reported on the other, so the confirm-half number is the one you
    may quote. A gain that appears on select and vanishes on confirm was noise.

    The centring fit uses no labels, so it runs on all rows without leakage.
    """
    ids, topic, batt, stance, veto, meta, batt2 = load_margins(margins_path)
    if meta.get("battery_hash") != battery_hash(battery, topic_hypotheses):
        raise ValueError("battery does not match the cache; re-run phase 1.")
    agg_kw.setdefault("bootstrap", 0)
    agg_kw.setdefault("stance_mode", "off")
    text_col = text_col or meta.get("text_col")

    lmap = dict(zip(df[id_col].astype(str), df[label_col]))
    lab = np.array([lmap.get(str(i), np.nan) for i in ids], dtype=float)
    tmap = dict(zip(df[id_col].astype(str), df[text_col])) if text_col in df.columns else {}
    texts = [tmap.get(str(i)) for i in ids] if tmap else None

    rng = np.random.default_rng(seed)
    sel = rng.random(len(ids)) >= holdout          # True = selection half
    # fixed_drop is removed from EVERY candidate and never offered for dropping.
    # It exists so a battery that is really two half-instruments can be swept one
    # half at a time: pass the other half here and the sweep sees only its own
    # pairs. The cache check above still uses the FULL battery, because the
    # cached margins cover all of it -- only the aggregation is restricted.
    fixed_drop = tuple(fixed_drop)
    domains = [d for d, _, _, _ in battery if d not in fixed_drop]

    def crit(drop, mask):
        agg = ContrastAggregator(battery, topic_hypotheses,
                                 drop_domains=fixed_drop + tuple(drop),
                                 **agg_kw).fit(batt, topic)
        s = np.array([r["score"] for r in
                      agg.transform(batt, topic, texts=texts,
                                    stance_margins=stance,
                                    veto_margins=veto)], float)
        ok = mask & np.isfinite(s) & np.isfinite(lab)
        return _pearson(s[ok], lab[ok]), int(ok.sum())

    lvl = logger.level
    logger.setLevel(logging.ERROR)                 # ~100 re-fits, same warning each
    try:
        base_s, n_s = crit((), sel)
        base_c, n_c = crit((), ~sel)
        print("\n" + "=" * 74)
        print(f"DOMAIN SWEEP against '{label_col}'  (split-half, seed={seed})")
        print("=" * 74)
        print(f"  baseline   select r={base_s:+.3f} (n={n_s})   "
              f"confirm r={base_c:+.3f} (n={n_c})")
        # How much does the baseline move from the split alone? If that exceeds
        # the gains on offer, the sweep cannot resolve which domain to drop and
        # any answer it gives is a coin flip dressed up as a result.
        rs = np.random.default_rng(seed + 1000)
        noise = [abs(crit((), rs.random(len(ids)) < holdout)[0]) for _ in range(n_splits)]
        split_sd = float(np.std(noise))
        print(f"  baseline |r| over {n_splits} random splits: "
              f"mean {np.mean(noise):.3f}, sd {split_sd:.3f}")
        print(f"\n  leave-one-out on the SELECTION half only")
        print(f"  {'dropped domain':<26}{'r':>9}{'gain':>9}")
        loo = []
        for d in domains:
            r_, _ = crit((d,), sel)
            loo.append((abs(r_) - abs(base_s), d, r_))
        for gain, d, r_ in sorted(loo, reverse=True):
            if abs(gain) > 1e-9:
                print(f"  {d:<26}{r_:+9.3f}{gain:+9.3f}")

        # greedy: keep adding the best remaining drop while it still helps
        chosen, cur = [], abs(base_s)
        for _ in range(max_drop):
            cands = [(abs(crit(tuple(chosen + [d]), sel)[0]), d)
                     for _, d, _ in sorted(loo, reverse=True)[:8] if d not in chosen]
            if not cands:
                break
            best, d = max(cands)
            if best - cur < min_gain:
                break
            chosen.append(d); cur = best
        print(f"\n  greedy selection (min_gain={min_gain}): "
              f"{', '.join(chosen) if chosen else 'nothing worth dropping'}")
        if chosen:
            s_sel, _ = crit(tuple(chosen), sel)
            s_con, n2 = crit(tuple(chosen), ~sel)
            print(f"  {'':<14}{'select':>12}{'confirm':>12}")
            print(f"  {'baseline':<14}{base_s:+12.3f}{base_c:+12.3f}")
            print(f"  {'after dropping':<14}{s_sel:+12.3f}{s_con:+12.3f}")
            print(f"  {'gain':<14}{abs(s_sel)-abs(base_s):+12.3f}"
                  f"{abs(s_con)-abs(base_c):+12.3f}   <- quote this one")
            gain_c = abs(s_con) - abs(base_c)
            if gain_c < 0.5 * (abs(s_sel) - abs(base_s)):
                print("  The confirm gain is far below the select gain: most of this")
                print("  is selection noise. Keep the full battery.")
            if gain_c < split_sd:
                print(f"  UNDERPOWERED: the confirm gain ({gain_c:+.3f}) is smaller than")
                print(f"  the baseline's own split-to-split sd ({split_sd:.3f}). This")
                print("  sample cannot tell these domains apart. Keep the full battery")
                print("  and re-run the sweep on more scored rows.")
        print("=" * 74 + "\n")
        return {"baseline": (base_s, base_c), "loo": loo, "chosen": chosen,
                "split_sd": split_sd}
    finally:
        logger.setLevel(lvl)


# =============================================================================
# HAND-CODING: the only route to a real gate evaluation
# =============================================================================
def make_coding_sheet(results_csv, out_csv, n_admit=150, n_drop=150, seed=0,
                      text_col="text_en", id_col="id", suffix=""):
    """Draw a stratified blind sample for hand-coding.

    Neither PopBERT nor any other label in the corpus says whether a sentence is
    ABOUT economic policy, so the gate's precision cannot be estimated from data
    you already have -- it has to be coded. This draws separately from the rows
    the gate admitted and the rows it dropped, records both stratum sizes, and
    writes ONLY id and text so the coding is blind to the model's answer.

    Fill in two columns:
      gold_economic  1 if the sentence is about economic policy, else 0
      gold_position  only when gold_economic=1: -2 far left .. 0 centre .. +2 far
                     right. Leave blank if it is economic but takes no side.

    Then pass the completed file to evaluate_gate().
    """
    d = pd.read_csv(results_csv)
    st = f"lr_status{suffix}"
    if st not in d.columns:
        raise KeyError(f"{st} not in {results_csv}; pass the right suffix=")
    admitted = d[d[st] != OFF_TOPIC_LABEL]
    dropped = d[d[st] == OFF_TOPIC_LABEL]
    rng = np.random.default_rng(seed)
    take = lambda g, k: g.sample(n=min(k, len(g)), random_state=int(rng.integers(1 << 31)))
    sheet = pd.concat([take(admitted, n_admit), take(dropped, n_drop)])
    sheet = sheet.sample(frac=1.0, random_state=seed)          # shuffle: blind order
    out = sheet[[id_col, text_col]].copy()
    out["gold_economic"] = ""
    out["gold_position"] = ""
    out.to_csv(out_csv, index=False)

    meta = {"results_csv": os.path.abspath(results_csv), "suffix": suffix,
            "N_admitted": int(len(admitted)), "N_dropped": int(len(dropped)),
            "n_admit_sampled": int(min(n_admit, len(admitted))),
            "n_drop_sampled": int(min(n_drop, len(dropped))), "seed": seed}
    with open(os.path.splitext(out_csv)[0] + ".strata.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"Wrote {out_csv}: {len(out)} rows to code "
          f"({meta['n_admit_sampled']} admitted + {meta['n_drop_sampled']} dropped, "
          "shuffled and blind).")
    print(f"  Corpus strata: {meta['N_admitted']} admitted, {meta['N_dropped']} dropped.")
    print("  Code gold_economic (1/0) and, where economic, gold_position (-2..+2).")
    print(f"  Then: evaluate_gate('{out_csv}')")
    return out


def _wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def evaluate_gate(coding_csv, results_csv=None, strata_json=None, id_col="id",
                  suffix=None):
    """Precision, recall and conditional score accuracy from a coded sheet.

    Because the sample is stratified by the gate's own decision, precision comes
    straight from the admitted stratum, while recall has to be reweighted by the
    corpus stratum sizes -- otherwise oversampling the dropped rows would make
    recall look far worse than it is.
    """
    code = pd.read_csv(coding_csv)
    strata_json = strata_json or os.path.splitext(coding_csv)[0] + ".strata.json"
    meta = json.load(open(strata_json))
    suffix = meta["suffix"] if suffix is None else suffix
    res = pd.read_csv(results_csv or meta["results_csv"])
    st, sc = f"lr_status{suffix}", f"lr_score{suffix}"
    m = code.merge(res[[id_col, st, sc]], on=id_col, how="left")
    m["gold_economic"] = pd.to_numeric(m.gold_economic, errors="coerce")
    m["gold_position"] = pd.to_numeric(m.gold_position, errors="coerce")
    m = m[m.gold_economic.notna()]
    if not len(m):
        raise ValueError("no coded rows found -- fill in gold_economic first")
    m["admitted"] = m[st] != OFF_TOPIC_LABEL

    A, D = m[m.admitted], m[~m.admitted]
    NA, ND = meta["N_admitted"], meta["N_dropped"]
    pa = A.gold_economic.mean() if len(A) else np.nan          # precision
    pd_ = D.gold_economic.mean() if len(D) else np.nan         # miss rate among dropped
    tp, fn = NA * pa, ND * pd_
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    f1 = 2 * pa * recall / (pa + recall) if (pa + recall) else np.nan
    ci_p = _wilson(int(A.gold_economic.sum()), len(A))
    ci_d = _wilson(int(D.gold_economic.sum()), len(D))

    print("\n" + "=" * 74)
    print(f"GATE EVALUATION   ({len(m)} coded: {len(A)} admitted, {len(D)} dropped)")
    print("=" * 74)
    print(f"  1. OF THE ROWS IT SCORED, how many really are economic?")
    print(f"     precision = {pa:.1%}   95% CI [{ci_p[0]:.1%}, {ci_p[1]:.1%}]   n={len(A)}")
    print(f"\n  2. OF THE ROWS IT DROPPED, how many were economic after all?")
    print(f"     miss rate = {pd_:.1%}   95% CI [{ci_d[0]:.1%}, {ci_d[1]:.1%}]   n={len(D)}")
    print(f"\n  reweighted to the corpus ({NA} admitted, {ND} dropped):")
    print(f"     recall    = {recall:.1%}      F1 = {f1:.3f}")
    print(f"     estimated economic sentences in the corpus: "
          f"{tp + fn:.0f} of {NA + ND} ({(tp + fn) / (NA + ND):.1%})")

    ok = m[(m.gold_economic == 1) & m.gold_position.notna() & m[sc].notna()]
    print(f"\n  3. ON GOLD-ECONOMIC ROWS IT SCORED, is the position right?  (n={len(ok)})")
    if len(ok) >= 20:
        r = _pearson(ok[sc], ok.gold_position)
        gold10 = 5 + 2.5 * ok.gold_position
        mae = float(np.mean(np.abs(ok[sc] - gold10)))
        print(f"     r(lr_score, gold_position) = {r:+.3f}")
        print(f"     MAE against gold mapped to 0-10 = {mae:.2f} points")
        sign = ok[(ok.gold_position != 0)]
        if len(sign) >= 10:
            agree = np.mean(np.sign(sign[sc] - 5) == np.sign(sign.gold_position))
            print(f"     side agreement (left/right, excl. gold centre) = {agree:.1%} "
                  f"of {len(sign)}")
        print(f"     {'gold':>6}{'n':>6}{'mean lr_score':>16}")
        for v in sorted(ok.gold_position.unique()):
            g = ok[ok.gold_position == v]
            if len(g) >= 3:
                print(f"     {v:6.0f}{len(g):6d}{g[sc].mean():16.2f}")
    else:
        print(f"     too few coded positions to say anything (need ~20+, have {len(ok)})")
    print("=" * 74 + "\n")
    return {"precision": pa, "recall": recall, "f1": f1, "miss_rate": pd_,
            "n_admitted_coded": len(A), "n_dropped_coded": len(D), "merged": m}


# =============================================================================
# SUPERVISED CALIBRATION - fit the aggregation to a gold standard
# =============================================================================
def build_features(batt, topic, stance, agg, include=("position", "active")):
    """Design matrix from cached margins, using the aggregator's own geometry.

    "position" is tanh(contrast/scale) zeroed where the pair is inactive -- the
    same quantity the heuristic averages, so a fitted model is strictly a
    better-weighted version of it rather than a different instrument.
    "active" adds the on/off indicator, which lets the model learn that a pair
    firing at all is informative regardless of direction.
    """
    contrast, active = agg._contrasts(batt)
    pos = agg._positions(contrast) * active
    cols, names = [], []
    if "position" in include:
        cols.append(pos); names += [f"pos::{d}" for d in agg.domains]
    if "active" in include:
        cols.append(active.astype(float)); names += [f"act::{d}" for d in agg.domains]
    if "topic" in include:
        cols.append(agg._topic(topic)[:, None]); names.append("topic")
    if "stance" in include and stance is not None:
        cols.append(agg.stance(stance)[:, None]); names.append("stance")
    X = np.hstack(cols)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), names


def _ridge(X, y, lam):
    Xc, ym = X - X.mean(0), y.mean()
    sd = Xc.std(0); sd[sd < 1e-9] = 1.0
    Z = Xc / sd
    A = Z.T @ Z + lam * np.eye(Z.shape[1])
    b = np.linalg.solve(A, Z.T @ (y - ym))
    return b / sd, ym - (b / sd) @ X.mean(0)


def fit_scale(margins_path, df, label_col, id_col="id", rows="scored",
              include=("position", "active"), n_folds=5, seed=0,
              alphas=(0.1, 1, 3, 10, 30, 100, 300, 1000), permute=True,
              battery=BATTERY, topic_hypotheses=TOPIC_HYPOTHESES, **agg_kw):
    """Cross-validated supervised fit of the aggregation to `label_col`.

    The hand-set weights in BATTERY are a guess about which domains matter and
    by how much. If the only objective is agreeing with a gold standard, that
    guess can simply be estimated instead. Reported performance is OUT OF FOLD:
    every prediction comes from a model that never saw that row.

    Nested CV picks the ridge penalty inside each training fold, so the reported
    number carries no hyperparameter selection bias either.
    """
    ids, topic, batt, stance, veto, meta, batt2 = load_margins(margins_path)
    agg_kw.setdefault("bootstrap", 0)
    agg_kw.setdefault("stance_mode", "off")
    agg = ContrastAggregator(battery, topic_hypotheses, **agg_kw).fit(batt, topic)

    lmap = dict(zip(df[id_col].astype(str), df[label_col]))
    y_all = np.array([lmap.get(str(i), np.nan) for i in ids], float)
    text_col = meta.get("text_col")
    tmap = dict(zip(df[id_col].astype(str), df[text_col])) if text_col in df.columns else {}
    texts = [tmap.get(str(i)) for i in ids] if tmap else None
    heur = np.array([r["score"] for r in
                     agg.transform(batt, topic, texts=texts, stance_margins=stance,
                                   veto_margins=veto)], float)

    if rows == "scored":
        mask = np.isfinite(heur) & np.isfinite(y_all)
    elif rows == "topic":
        mask = (agg._topic(topic) >= agg.topic_margin_min) & np.isfinite(y_all)
    else:
        mask = np.isfinite(y_all)
    X, names = build_features(batt, topic, stance, agg, include)
    X, y = X[mask], y_all[mask]
    n = len(y)
    if n < 50:
        raise ValueError(f"only {n} usable rows; need at least 50 to fit")

    rng = np.random.default_rng(seed)
    fold = rng.permutation(n) % n_folds
    oof = np.full(n, np.nan)
    chosen = []
    for k in range(n_folds):
        tr, te = fold != k, fold == k
        # inner CV for the penalty, on the training part only
        inner = rng.permutation(int(tr.sum())) % n_folds
        Xtr, ytr = X[tr], y[tr]
        best, best_lam = -np.inf, alphas[0]
        for lam in alphas:
            pr = np.full(len(ytr), np.nan)
            for j in range(n_folds):
                a, b = inner != j, inner == j
                if b.sum() < 3 or a.sum() < 10:
                    continue
                w, c = _ridge(Xtr[a], ytr[a], lam)
                pr[b] = Xtr[b] @ w + c
            ok = np.isfinite(pr)
            s = -np.mean((pr[ok] - ytr[ok]) ** 2) if ok.any() else -np.inf
            if s > best:
                best, best_lam = s, lam
        w, c = _ridge(Xtr, ytr, best_lam)
        oof[te] = X[te] @ w + c
        chosen.append(best_lam)

    r_fit = _pearson(oof, y)
    r_heur = _pearson(heur[mask], y)
    print("\n" + "=" * 74)
    print(f"SUPERVISED FIT to '{label_col}'   (rows={rows}, n={n}, {n_folds}-fold)")
    print("=" * 74)
    print(f"  features                 : {X.shape[1]} ({', '.join(include)})")
    print(f"  ridge penalty per fold   : {chosen}")
    print(f"  hand-weighted heuristic  : r = {r_heur:+.3f}")
    print(f"  fitted, OUT OF FOLD      : r = {r_fit:+.3f}   "
          f"({abs(r_fit) - abs(r_heur):+.3f})")
    if permute:
        pr = []
        for t in range(20):
            ys = np.random.default_rng(seed + t).permutation(y)
            o = np.full(n, np.nan)
            for k in range(n_folds):
                tr, te = fold != k, fold == k
                w, c = _ridge(X[tr], ys[tr], 30.0)
                o[te] = X[te] @ w + c
            pr.append(abs(_pearson(o, ys)))
        print(f"  label-shuffled control   : r = {np.mean(pr):+.3f} "
              f"(max {np.max(pr):.3f} over 20 shuffles)")
        if abs(r_fit) < 3 * np.mean(pr):
            print("  WARNING: barely above the shuffled control. Not a real fit.")

    print(f"\n  dose-response of the out-of-fold prediction")
    print(f"  {'label':>8}{'n':>7}{'mean pred':>12}{'heuristic':>12}")
    for v in np.unique(y):
        m = y == v
        if m.sum() >= 5:
            print(f"  {v:8.0f}{int(m.sum()):7d}{oof[m].mean():12.2f}"
                  f"{heur[mask][m].mean():12.2f}")

    w_full, c_full = _ridge(X, y, float(np.median(chosen)))
    order = np.argsort(-np.abs(w_full))
    print(f"\n  largest fitted weights (full-data refit, for interpretation only)")
    print(f"  {'feature':<30}{'weight':>10}")
    for i in order[:12]:
        print(f"  {names[i]:<30}{w_full[i]:+10.3f}")
    print("\n  Quote the OUT-OF-FOLD number. The weights above are refitted on all")
    print("  rows and will look stronger than they are.")
    print("=" * 74 + "\n")
    return {"oof": oof, "y": y, "mask": mask, "r_fit": r_fit, "r_heur": r_heur,
            "weights": dict(zip(names, w_full)), "intercept": c_full,
            "alphas": chosen, "ids": [i for i, m in zip(ids, mask) if m]}


def _logistic(X, y, w, lam, iters=60, tol=1e-8):
    """L2-penalised weighted logistic regression by IRLS.

    Weights matter here: the gold set is a stratified sample (every labelled row,
    a fraction of the unlabelled ones), so an unweighted fit would optimise for a
    population that does not exist.
    """
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1.0
    Z = np.hstack([np.ones((len(X), 1)), (X - mu) / sd])
    b = np.zeros(Z.shape[1])
    P = lam * np.eye(Z.shape[1]); P[0, 0] = 0.0      # never penalise the intercept
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Z @ b, -30, 30)))
        s = np.maximum(w * p * (1 - p), 1e-9)
        g = Z.T @ (w * (y - p)) - P @ b
        H = Z.T @ (Z * s[:, None]) + P
        step = np.linalg.solve(H, g)
        b += step
        if np.max(np.abs(step)) < tol:
            break
    return b, mu, sd


def _logit_predict(b, mu, sd, X):
    Z = np.hstack([np.ones((len(X), 1)), (X - mu) / sd])
    return 1.0 / (1.0 + np.exp(-np.clip(Z @ b, -30, 30)))


def _prf(pred, gold, w):
    """Weighted precision / recall / F1. The weights carry the sample back to
    corpus scale, so these are corpus estimates, not sample statistics."""
    tp = float(np.sum(w * (pred == 1) * (gold == 1)))
    fp = float(np.sum(w * (pred == 1) * (gold == 0)))
    fn = float(np.sum(w * (pred == 0) * (gold == 1)))
    tn = float(np.sum(w * (pred == 0) * (gold == 0)))
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f = 2 * p * r / (p + r) if p + r else float("nan")
    return dict(precision=p, recall=r, f1=f, tp=tp, fp=fp, fn=fn, tn=tn,
                spec=tn / (tn + fp) if tn + fp else float("nan"))


def fit_gate(margins_path, gold, label_col, id_col="id", weight_col="w",
             include=("topic", "veto"), n_folds=5, seed=0,
             alphas=(0.3, 1, 3, 10, 30, 100, 300, 1000, 3000), permute=True,
             battery=BATTERY, topic_hypotheses=TOPIC_HYPOTHESES,
             topic_margin_min=0.0, veto_margin=0.0, use_veto=True,
             veto_hypotheses=NEGATIVE_TOPIC_HYPOTHESES):
    """Fit the GATE to hand-coded labels, instead of using the hand-picked rule.

    The shipped rule -- "2nd-highest topic margin >= 0", optionally minus a veto
    contrast -- was chosen by hand and never optimised. Given a coded sample it
    can simply be estimated: a weighted logistic regression on the raw probe
    margins, cross-validated, with the decision threshold picked INSIDE each
    training fold so the reported figures carry no selection bias.

    `gold` needs three columns: `id_col`, the binary `label_col`, and
    `weight_col` -- the stratum weight, so that weighted sums estimate corpus
    counts. Returns the fitted rule plus `admit_mask`, aligned to the cache's
    id order, which rescore(admit_mask=...) will use in place of the hand rule.
    """
    ids, topic, batt, stance, veto, meta, batt2 = load_margins(margins_path)
    ids_s = [str(i) for i in ids]
    agg = ContrastAggregator(battery, topic_hypotheses, bootstrap=0,
                             stance_mode="off", topic_margin_min=topic_margin_min,
                             veto_hypotheses=veto_hypotheses, veto_margin=veto_margin,
                             use_veto=use_veto)

    cols, names = [], []
    if "topic" in include:
        cols.append(np.asarray(topic, float))
        names += [f"topic{i:02d}" for i in range(topic.shape[1])]
    if "veto" in include and veto is not None:
        cols.append(np.asarray(veto, float))
        names += [f"veto{i:02d}" for i in range(veto.shape[1])]
    if "battery" in include:
        # The battery is a STANCE instrument, but the LEVEL of its margins is
        # topical evidence the topic probes never see: a pension sentence sits
        # low on all 44 populism hypotheses, an elite-corruption one sits high
        # on several. Cheap to test, and it is the only signal in the cache
        # that the hand rule structurally cannot use.
        cols.append(np.asarray(batt, float))
        names += [f"{d}_{side}" for d, _, _, _ in battery for side in ("L", "R")]
    X_all = np.nan_to_num(np.hstack(cols), nan=0.0, posinf=0.0, neginf=0.0)

    gold = gold.copy(); gold[id_col] = gold[id_col].astype(str)
    lab = dict(zip(gold[id_col], gold[label_col]))
    wt = dict(zip(gold[id_col], gold[weight_col])) if weight_col in gold.columns else {}
    y_all = np.array([lab.get(i, np.nan) for i in ids_s], float)
    w_all = np.array([wt.get(i, 1.0) for i in ids_s], float)
    m = np.isfinite(y_all)
    X, y, w = X_all[m], y_all[m], w_all[m]
    n = len(y)
    if n < 60:
        raise ValueError(f"only {n} labelled rows; need at least 60")

    # the hand rule, scored the same way so the comparison is like for like
    hand_all = agg._topic(topic) >= agg.topic_margin_min
    if use_veto and veto is not None:
        hand_all &= (agg._topic_max(topic) - agg.veto(veto)) > agg.veto_margin
    hand = _prf(hand_all[m].astype(int), y, w)

    rng = np.random.default_rng(seed)
    fold = rng.permutation(n) % n_folds
    oof = np.full(n, np.nan)
    picked = []
    for k in range(n_folds):
        tr, te = fold != k, fold == k
        best, best_lam = -np.inf, alphas[0]
        inner = rng.permutation(int(tr.sum())) % n_folds
        for lam in alphas:                       # inner CV for the penalty
            pr = np.full(int(tr.sum()), np.nan)
            Xt, yt, wtr = X[tr], y[tr], w[tr]
            for j in range(n_folds):
                a, b_ = inner != j, inner == j
                if b_.sum() < 3 or a.sum() < 20:
                    continue
                bb, mu, sd = _logistic(Xt[a], yt[a], wtr[a], lam)
                pr[b_] = _logit_predict(bb, mu, sd, Xt[b_])
            ok = np.isfinite(pr)
            if not ok.any():
                continue
            sc = -np.sum(wtr[ok] * (pr[ok] - yt[ok]) ** 2) / np.sum(wtr[ok])
            if sc > best:
                best, best_lam = sc, lam
        bb, mu, sd = _logistic(X[tr], y[tr], w[tr], best_lam)
        oof[te] = _logit_predict(bb, mu, sd, X[te])
        picked.append(best_lam)

    # threshold chosen on the training part of each fold, never on the test part
    thr_oof = np.full(n, np.nan)
    for k in range(n_folds):
        tr, te = fold != k, fold == k
        grid = np.linspace(0.05, 0.95, 91)
        f1s = [_prf((oof[tr] >= t).astype(int), y[tr], w[tr])["f1"] for t in grid]
        t_star = grid[int(np.nanargmax(f1s))]
        thr_oof[te] = t_star
    fit = _prf((oof >= thr_oof).astype(int), y, w)

    print("\n" + "=" * 74)
    print(f"FITTED GATE vs the hand rule   (n={n} coded, {n_folds}-fold, weighted)")
    print("=" * 74)
    print(f"  features                : {X.shape[1]} ({', '.join(include)})")
    print(f"  ridge penalty per fold  : {picked}")
    print(f"  threshold per fold      : {np.round(np.unique(thr_oof), 2).tolist()}")
    print(f"\n  {'':<22}{'precision':>11}{'recall':>9}{'spec':>8}{'F1':>8}")
    print(f"  {'hand rule':<22}{hand['precision']:>10.1%}{hand['recall']:>9.1%}"
          f"{hand['spec']:>8.1%}{hand['f1']:>8.3f}")
    print(f"  {'fitted, OUT OF FOLD':<22}{fit['precision']:>10.1%}{fit['recall']:>9.1%}"
          f"{fit['spec']:>8.1%}{fit['f1']:>8.3f}")
    print(f"  {'gain':<22}{'':>10}{'':>9}{'':>8}{fit['f1']-hand['f1']:>+8.3f}")

    if permute:
        ps = []
        for t in range(10):
            ys = np.random.default_rng(seed + 100 + t).permutation(y)
            o = np.full(n, np.nan)
            for k in range(n_folds):
                tr, te = fold != k, fold == k
                bb, mu, sd = _logistic(X[tr], ys[tr], w[tr], 10.0)
                o[te] = _logit_predict(bb, mu, sd, X[te])
            ps.append(_prf((o >= 0.5).astype(int), ys, w)["f1"])
        print(f"\n  label-shuffled control  : F1 = {np.nanmean(ps):.3f} "
              f"(max {np.nanmax(ps):.3f} over 10 shuffles)")
        if fit["f1"] < np.nanmean(ps) + 0.10:
            print("  WARNING: barely above the shuffled control. Not a real fit.")

    # full-data refit, for applying to the whole corpus
    b_full, mu_f, sd_f = _logistic(X, y, w, float(np.median(picked)))
    thr_full = float(np.median(thr_oof))
    p_all = _logit_predict(b_full, mu_f, sd_f, X_all)
    admit = p_all >= thr_full
    print(f"\n  applied to all {len(ids)} rows at threshold {thr_full:.2f}: "
          f"admits {admit.mean():.1%}  (hand rule {hand_all.mean():.1%})")
    order = np.argsort(-np.abs(b_full[1:]))
    print(f"\n  largest fitted coefficients (standardised, full-data refit)")
    for i in order[:8]:
        print(f"    {names[i]:<12}{b_full[1 + i]:+8.3f}")
    print("\n  Quote the OUT-OF-FOLD row. Pass admit_mask=result['admit_mask']")
    print("  to rescore() to use this gate in place of the hand rule.")
    print("=" * 74 + "\n")
    return {"hand": hand, "fitted": fit, "oof": oof, "y": y, "w": w, "mask": m,
            "admit_mask": admit, "prob_all": p_all, "threshold": thr_full,
            "coef": dict(zip(names, b_full[1:])), "intercept": float(b_full[0]),
            "alphas": picked, "ids": ids_s}


def reliability(texts, scorer=None, model="mlburnham/Political_DEBATE_large_v1.0",
                batch_size=None, text_chunk=128, **agg_kw):
    """Score the same texts with the battery and its paraphrase, and correlate.

    That correlation is the reliability ceiling: no downstream association can
    legitimately be reported as stronger than the instrument's agreement with
    a reworded version of itself.
    """
    texts = [t for t in list(texts) if isinstance(t, str) and t.strip()]
    scorer = scorer or MarginScorer(model)
    tm = scorer.margin_matrix(texts, TOPIC_HYPOTHESES, batch_size, text_chunk)
    outs = []
    arms = [("original", BATTERY)]
    if BATTERY_PARAPHRASE:                 # None in dimensions with no paraphrase set
        arms.append(("paraphrase", BATTERY_PARAPHRASE))
    for name, bat in arms:
        bm = scorer.margin_matrix(texts, battery_hypotheses(bat), batch_size, text_chunk)
        agg = ContrastAggregator(bat, TOPIC_HYPOTHESES, bootstrap=0,
                                 stance_mode="off", **agg_kw).fit(bm, tm)
        outs.append(np.array([r["score"] for r in agg.transform(bm, tm)], float))
    both = np.isfinite(outs[0]) & np.isfinite(outs[1])
    r = _pearson(outs[0], outs[1])
    print("\n" + "=" * 74)
    print(f"BATTERY RELIABILITY on {int(both.sum())} texts")
    print(f"  pearson r (original vs paraphrase) : {r:+.3f}")
    print(f"  mean |difference|                  : {np.nanmean(np.abs(outs[0]-outs[1])):.2f} points")
    print(f"  band agreement                     : "
          f"{np.mean([interpret(a) == interpret(b) for a, b in zip(outs[0][both], outs[1][both])]):.1%}")
    print("  Treat r as the ceiling for any correlation you report downstream.")
    print("=" * 74 + "\n")
    return {"r": r, "original": outs[0], "paraphrase": outs[1]}


# =============================================================================
# ENTRY POINT
# =============================================================================
def _in_notebook():
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except Exception:
        return False


def main(argv=None):
    ap = argparse.ArgumentParser(description="Left-right economic scoring (v2)")
    ap.add_argument("--input", default=None)
    ap.add_argument("--input-pattern", default="*.csv")
    ap.add_argument("--output", required=True)
    ap.add_argument("--margins", default=None)
    ap.add_argument("--text-col", default="text_en")
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--group-col", default=None)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--model", default="mlburnham/Political_DEBATE_large_v1.0")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--fp16", dest="fp16", action="store_true", default=None)
    ap.add_argument("--no-fp16", dest="fp16", action="store_false")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--text-chunk", type=int, default=128)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--gate", default="post", choices=["post", "pre"])
    ap.add_argument("--topic-margin-min", type=float, default=0.0)
    ap.add_argument("--active-margin", type=float, default=0.0)
    ap.add_argument("--standardize", default="none", choices=["none", "center", "z"])
    ap.add_argument("--contrast-scale", default="auto")
    ap.add_argument("--no-active-only", dest="active_only", action="store_false", default=True)
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--min-tokens", type=int, default=5)
    ap.add_argument("--min-active", type=int, default=2)
    ap.add_argument("--stance-mode", default="off", choices=["off", "gate", "shrink"])
    ap.add_argument("--stance-min", type=float, default=0.0)
    ap.add_argument("--no-veto", dest="use_veto", action="store_false", default=True)
    ap.add_argument("--veto-margin", type=float, default=0.0)
    ap.add_argument("--balance-weight", type=float, default=0.0)
    ap.add_argument("--no-paraphrase", dest="average_paraphrase",
                    action="store_false", default=True)
    ap.add_argument("--stance-scale", type=float, default=4.0)
    ap.add_argument("--label-col", default=None)
    ap.add_argument("--drop-domains", default="",
                    help="comma-separated domains to exclude from aggregation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--first-rows", dest="sample_limit", action="store_false", default=True)
    ap.add_argument("--checkpoint-every", type=int, default=2048)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-fp16-check", dest="fp16_check", action="store_false", default=True)
    ap.add_argument("--no-diagnostics", dest="run_diagnostics", action="store_false", default=True)
    ap.add_argument("--rescore-only", action="store_true",
                    help="skip the GPU pass and re-derive scores from a margin cache")
    args, unknown = ap.parse_known_args(argv)
    if unknown:
        logger.info("ignoring unrecognised args: %s", unknown)

    scale = args.contrast_scale
    scale = scale if scale == "auto" else float(scale)
    drop = tuple(x for x in args.drop_domains.split(",") if x.strip())

    if args.rescore_only:
        path = args.margins or default_margins_path(args.output)
        return rescore(path, input_csv=args.input, output_csv=args.output,
                       text_col=args.text_col, id_col=args.id_col,
                       group_col=args.group_col, suffix=args.suffix,
                       topic_margin_min=args.topic_margin_min,
                       active_margin=args.active_margin, standardize=args.standardize,
                       contrast_scale=scale, active_only=args.active_only,
                       bootstrap=args.bootstrap, seed=args.seed,
                       min_tokens=args.min_tokens, min_active=args.min_active,
                       stance_mode=args.stance_mode, stance_min=args.stance_min,
                       stance_scale=args.stance_scale, label_col=args.label_col,
                       drop_domains=drop, use_veto=args.use_veto,
                       veto_margin=args.veto_margin,
                       balance_weight=args.balance_weight,
                       average_paraphrase=args.average_paraphrase,
                       run_diagnostics=args.run_diagnostics)[0]

    return run(input_csv=args.input, input_pattern=args.input_pattern,
               output_csv=args.output, margins_path=args.margins,
               text_col=args.text_col, id_col=args.id_col, group_col=args.group_col,
               suffix=args.suffix, model=args.model, revision=args.revision,
               device=args.device, fp16=args.fp16, batch_size=args.batch_size,
               text_chunk=args.text_chunk, max_length=args.max_length,
               gate=args.gate, topic_margin_min=args.topic_margin_min,
               active_margin=args.active_margin, standardize=args.standardize,
               contrast_scale=scale, active_only=args.active_only,
               bootstrap=args.bootstrap, seed=args.seed, min_tokens=args.min_tokens,
               min_active=args.min_active, stance_mode=args.stance_mode,
               stance_min=args.stance_min, stance_scale=args.stance_scale,
               label_col=args.label_col, drop_domains=drop,
               use_veto=args.use_veto, veto_margin=args.veto_margin,
               balance_weight=args.balance_weight,
               average_paraphrase=args.average_paraphrase, limit=args.limit,
               sample_limit=args.sample_limit, checkpoint_every=args.checkpoint_every,
               resume=args.resume, fp16_check=args.fp16_check,
               run_diagnostics=args.run_diagnostics)[0]


# AUTORUN defaults to FALSE.
#
# It used to default to True, and pasting this cell then fired run(**CONFIG) on
# the spot -- a full GPU pass with whatever CONFIG happened to hold, overwriting
# whatever output_csv pointed at. That has now cost two unintended economic runs
# (the second a bit-exact re-run of popbert_lr_en (7).csv: 14.4 GPU-minutes for
# zero new information), because the file is pasted to load DEFINITIONS far more
# often than to launch a run. Defaulting to False makes the expensive thing
# explicit and the cheap thing automatic, which is the right way round.
#
# To launch from CONFIG, set AUTORUN = True in the cell that pastes this file.
AUTORUN = False

if __name__ == "__main__":
    if not AUTORUN:
        print("AUTORUN is False: definitions loaded, nothing executed.")
        print("  rescore(MARGINS, ...) / compare_settings(MARGINS) / domain_sweep(...)")
    elif _in_notebook():
        RESULT, MARGINS = run(**CONFIG)
        print(f"\nMargin cache: {MARGINS}")
        print("Try other phase-2 settings without touching the GPU:")
        print("  compare_settings(MARGINS)")
        print("  rescore(MARGINS, output_csv='alt.csv', standardize='z')")
        try:
            display(RESULT.head(10))          # noqa: F821
        except Exception:
            print(RESULT.head(10))
    else:
        main()
