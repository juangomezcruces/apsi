import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .shared_model_cache import SharedModelCache
import warnings
warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# ============================================================================
# LIBERAL-ILLIBERAL HYPOTHESIS-BASED SCORER
# ============================================================================

class LiberalIlliberalScorer:
    def __init__(self, model_name="mlburnham/Political_DEBATE_large_v1.0"):
        cache = SharedModelCache()
        self.model, self.tokenizer = cache.get_model_and_tokenizer(model_name)
        self.entailment_idx = self._find_entailment_index()

        self.liberal_illiberal_hypotheses = {

            # Liberal
            "The text explicitly states that every adult citizen has an equal and unconditional right to vote.": (1.0, "liberal"),
            "The text explicitly argues that electoral defeat must be accepted and power transferred peacefully.": (1.0, "liberal"),
            "The text explicitly defends freedom of speech as a right the state may not restrict.": (1.0, "liberal"),
            "The text explicitly defends independent media free from state control.": (1.0, "liberal"),
            "The text explicitly defends the right to peaceful assembly and protest.": (1.0, "liberal"),
            "The text explicitly defends the right to form or join independent political parties and civic organizations.": (1.0, "liberal"),
            "The text explicitly argues that expanding voting rights to more citizens strengthens democracy.": (1.0, "liberal"),

            "The text argues that political opposition parties are a necessary and legitimate part of democracy.": (0.80, "liberal"),
            "The text argues that citizens must be free to criticize the government without facing repression.": (0.80, "liberal"),
            "The text argues that the rule of law binds the government equally with ordinary citizens.": (0.80, "liberal"),
            "The text argues that courts and democratic institutions must be protected from executive interference.": (0.80, "liberal"),

            "The text expresses that minority or dissenting political views deserve legal protection.": (0.40, "liberal"),
            "The text expresses that political change must occur through legal and electoral processes, not force.": (0.40, "liberal"),
            "The text expresses that political competition among multiple parties produces better governance.": (0.40, "liberal"),


            # Illiberal
            "The text rejects representative free and fair elections as necessary or desirable for governance.": (1.0, "illiberal"),
            "The text supports single-party rule or one-party political dominance.": (1.0, "illiberal"),
            "The text justifies restricting freedom of speech.": (1.0, "illiberal"),
            "The text supports state control or censorship of the media.": (1.0, "illiberal"),
            "The text rejects freedom of peaceful assembly or protest.": (1.0, "illiberal"),
            "The text opposes independent civic or political organizations.": (1.0, "illiberal"),
            "The text expresses that political legitimacy derives from ideology, religion, or divine authority rather than elections.": (1.0, "illiberal"),
            "The text supports theocratic governance or religious law as the supreme political authority.": (1.0, "illiberal"),
            "The text conditions the right to vote or political participation on literacy, education, property, or other qualifying criteria.": (1.0, "illiberal"),
            "The text expresses that voting rights should be restricted to those who meet an educational or literacy standard.": (1.0, "illiberal"),

            "The text frames suffrage restrictions as beneficial for the quality or responsibility of democratic representation.": (0.85, "illiberal"),
            "The text expresses refusal to accept defeat in competitive elections.": (0.85, "illiberal"),
            "The text expresses that political criticism or dissent is illegitimate, dangerous, or should be suppressed.": (0.85, "illiberal"),
            "The text justifies suppressing dissent to maintain order, stability, or national unity.": (0.85, "illiberal"),
            "The text expresses that political rights are conditional on loyalty to the regime, party, or ideology.": (0.85, "illiberal"),
            "The text expreses that one party or leader should govern without challenge from political opponents.": (0.85, "illiberal"),
            "The text explicitly argues that one party or leader should govern without challenge from political opponents.": (0.85, "illiberal"),

            "The text portrays political opponents as enemies, traitors, or existential threats rather than legitimate actors.": (0.65, "illiberal"),
            "The text expresses preference for revolutionary or extra-legal seizure of power over electoral competition.": (0.65, "illiberal"),
            "The text depicts democracy or democratic institutions as inherently corrupt, weak, or irreparably broken.": (0.65, "illiberal"),
            "The text argues that political participation or rights should be conditional on education, ethnicity, religion, or social standing.": (0.65, "illiberal"),
            "The text instrumentally invokes democratic values (speech, freedom, rule of law) while simultaneously arguing for their restriction or elimination.": (0.60, "illiberal"),
            "The text argues that liberal elites suppress or dismiss opinions that differ from their own.": (0.55, "illiberal"),
        }

        liberal_count   = sum(1 for _, (_, d) in self.liberal_illiberal_hypotheses.items() if d == "liberal")
        illiberal_count = sum(1 for _, (_, d) in self.liberal_illiberal_hypotheses.items() if d == "illiberal")
        print(f"Loaded {len(self.liberal_illiberal_hypotheses)} hypotheses ({liberal_count} liberal, {illiberal_count} illiberal)")

        # Topic check configuration — unchanged from script 1
        self.topic_threshold = 0.6
        self.topic_hypotheses = [
            # Electoral competition
            "This text supports free and fair multiparty elections as the primary source of political authority.",
            "This text supports accepting electoral defeat and peaceful transfers of power.",
            "This text supports the full participation of opposition parties in electoral competition.",
            "This text supports limiting or controlling elections to protect national interests.",
            "This text rejects the need for competitive elections in favor of alternative forms of rule.",

            # Freedom of expression and media
            "This text supports freedom of speech as a fundamental political right.",
            "This text supports independent media as a watchdog over those in power.",
            "This text supports government regulation of speech to prevent harm or division.",
            "This text supports state control of media to promote national unity.",
            "This text portrays free expression as a threat to social or political order.",

            # Assembly and participation
            "This text supports the right of citizens to protest and organize freely.",
            "This text supports broad political participation by citizens in public life.",
            "This text supports requiring government approval for protests and assemblies.",
            "This text supports restricting political participation to approved groups or elites.",
            "This text portrays mass political participation as destabilizing or dangerous.",

            # Pluralism and opposition
            "This text supports tolerance of political opposition and dissenting views.",
            "This text supports political disagreement as healthy for democracy.",
            "This text supports restricting extremist or disloyal political views.",
            "This text supports prioritizing national unity over political diversity.",
            "This text portrays political opposition as a threat to the nation.",

            # State and leadership
            "This text supports rule of law and institutional limits on political power.",
            "This text supports checks and balances on executive authority.",
            "This text supports strong leadership even at the expense of institutional constraints.",
            "This text supports concentrating power in a single leader or ruling group.",
            "This text portrays the leader as embodying the will of the nation.",

            # Authoritarian justifications
            "This text supports limiting freedoms to ensure stability and order.",
            "This text supports restricting rights for reasons of national security.",
            "This text supports political authority based on culture or tradition.",
            "This text supports emergency powers during crises.",
            "This text portrays internal or external enemies as justification for repression.",
        ]

    def _find_entailment_index(self):
        config = self.model.config
        if hasattr(config, 'label2id') and config.label2id:
            for label, idx in config.label2id.items():
                if label.lower() in ['entailment', 'entail']:
                    return idx
        return 0

    def _get_entailment_prob(self, text, hypothesis):
        inputs = self.tokenizer(
            text, hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
            prob = torch.softmax(outputs.logits, dim=-1)[0, self.entailment_idx].item()
        return prob

    def is_about_democratic_principles(self, text):
        probs = [self._get_entailment_prob(text, h) for h in self.topic_hypotheses]
        prob = float(max(probs)) if probs else 0.0
        logger.info(f"Thesis Liberal Illiberal triggered with: {prob}")
        return prob >= self.topic_threshold, prob

    def get_hypothesis_probabilities(self, text):
        probs = []
        for hypothesis in self.liberal_illiberal_hypotheses.keys():
            inputs = self.tokenizer(
                text, hypothesis,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            with torch.no_grad():
                outputs = self.model(**inputs)
                prob = torch.softmax(outputs.logits, dim=-1)[0, self.entailment_idx].item()
                probs.append(prob)
        return np.array(probs, dtype=float)

    def compute_combined_confidence(self, liberal_probs, illiberal_probs, all_probs):
        liberal_variance   = np.var(liberal_probs)  if len(liberal_probs)  > 1 else 0
        illiberal_variance = np.var(illiberal_probs) if len(illiberal_probs) > 1 else 0

        liberal_confidence   = 1 / (1 + liberal_variance  * 4)
        illiberal_confidence = 1 / (1 + illiberal_variance * 4)
        base_confidence = (0.7 * min(liberal_confidence, illiberal_confidence)
                           + 0.3 * (liberal_confidence + illiberal_confidence) / 2)

        k = 5
        top_liberal   = np.sort(liberal_probs)[-k:]  if len(liberal_probs)  >= k else liberal_probs
        top_illiberal = np.sort(illiberal_probs)[-k:] if len(illiberal_probs) >= k else illiberal_probs

        top_liberal_avg   = float(np.mean(top_liberal))   if len(top_liberal)   else 0.0
        top_illiberal_avg = float(np.mean(top_illiberal)) if len(top_illiberal) else 0.0

        topk_contradiction     = float(min(top_liberal_avg, top_illiberal_avg))
        contradiction_detected = topk_contradiction > 0.25

        if contradiction_detected:
            contradiction_penalty = min(1.0, topk_contradiction * 2.0)
            final_confidence = base_confidence * (1 - contradiction_penalty * 0.8)
        else:
            final_confidence = base_confidence

        return {
            'combined':               float(final_confidence),
            'contradiction_detected': bool(contradiction_detected),
            'contradiction_score':    float(topk_contradiction if contradiction_detected else 0.0),
            'top_liberal_avg':        float(top_liberal_avg),
            'top_illiberal_avg':      float(top_illiberal_avg),
        }

    def score_liberal_illiberal(self, text, thr=0.15):
        # Topic pre-check — unchanged from script 1
        is_relevant, topic_prob = self.is_about_democratic_principles(text)
        if not is_relevant:
            return {
                'text':                   text,
                'score':                  'NA',
                'confidence':             0.0,
                'contradiction_detected': False,
                'interpretation':         'Not about democratic principles',
                'topic_probability':      float(topic_prob),
                'passed_precheck':        False,
                'is_relevant':            False,
            }

        # NLI scoring
        probs = self.get_hypothesis_probabilities(text)

        liberal_probs   = []
        illiberal_probs = []
        hypothesis_results = []

        for i, (hypothesis, (weight, direction)) in enumerate(self.liberal_illiberal_hypotheses.items()):
            prob = float(probs[i])
            hypothesis_results.append({
                'hypothesis':  hypothesis,
                'probability': prob,
                'direction':   direction,
            })
            if direction == "liberal":
                liberal_probs.append(prob * weight)
            else:
                illiberal_probs.append(prob * weight)

        k_score = max(4, int(np.sum(probs > thr)) + 2)

        top_liberal_probs   = sorted(liberal_probs,   reverse=True)[:k_score]
        top_illiberal_probs = sorted(illiberal_probs, reverse=True)[:k_score]

        liberal_avg   = float(np.mean(top_liberal_probs))   if top_liberal_probs   else 0.0
        illiberal_avg = float(np.mean(top_illiberal_probs)) if top_illiberal_probs else 0.0

        # Symmetric mutual suppression:
        # each side penalised proportionally by strength of the opposing signal.
        # Cap at 0.80 so mild opposing signals don't over-penalise.
        MAX_SIGNAL = 1.0
        liberal_penalty_mult   = 1.0 - min(illiberal_avg / MAX_SIGNAL, 0.80)
        illiberal_penalty_mult = 1.0 - min(liberal_avg   / MAX_SIGNAL, 0.80)

        penalised_liberal_avg   = liberal_avg   * liberal_penalty_mult
        penalised_illiberal_avg = illiberal_avg * illiberal_penalty_mult

        difference  = penalised_liberal_avg - penalised_illiberal_avg
        final_score = float(np.clip(5 + difference * 5, 0, 10))

        confidence_data = self.compute_combined_confidence(liberal_probs, illiberal_probs, probs)

        liberal_hyps   = [h for h in hypothesis_results if h['direction'] == 'liberal']
        illiberal_hyps = [h for h in hypothesis_results if h['direction'] == 'illiberal']

        top_liberal   = sorted(liberal_hyps,   key=lambda x: x['probability'], reverse=True)[:k_score]
        top_illiberal = sorted(illiberal_hyps, key=lambda x: x['probability'], reverse=True)[:k_score]

        if final_score < 2:   interpretation = "Strongly Illiberal"
        elif final_score < 4: interpretation = "Illiberal"
        elif final_score < 6: interpretation = "Moderate"
        elif final_score < 8: interpretation = "Liberal"
        else:                 interpretation = "Strongly Liberal"

        return {
            'text':                     text,
            'score':                    final_score,
            'confidence':               confidence_data['combined'],
            'contradiction_detected':   confidence_data['contradiction_detected'],
            'interpretation':           interpretation,
            'liberal_avg':              liberal_avg,
            'illiberal_avg':            illiberal_avg,
            'penalised_liberal_avg':    penalised_liberal_avg,
            'penalised_illiberal_avg':  penalised_illiberal_avg,
            'top_liberal_hypotheses':   top_liberal,
            'top_illiberal_hypotheses': top_illiberal,
            'passed_precheck':          True,
            'is_relevant':              True,
            'topic_probability':        float(topic_prob),
            'k_score':                  k_score,
            'threshold':                thr,
        }

    def quick_score(self, text, thr=0.15):
        result = self.score_liberal_illiberal(text, thr=thr)
        return result['score']


# ============================================================================
# INTERACTIVE ANALYSIS FUNCTIONS
# ============================================================================

def analyze_text(scorer, text):
    result = scorer.score_liberal_illiberal(text)

    print(f"\n{'='*80}")
    print(f"TEXT: {text}")
    print(f"{'='*80}")

    if not result['is_relevant']:
        print(f"\n⚠️  NOT RELEVANT: {result['interpretation']}")
        print(f"   Topic probability: {result['topic_probability']:.3f}")
        return result

    print(f"\n📊 RESULTS:")
    print(f"   Score:         {result['score']:.2f}/10")
    print(f"   Confidence:    {result['confidence']:.3f}")
    print(f"   Contradiction: {'YES' if result['contradiction_detected'] else 'NO'}")
    print(f"   Interpretation:{result['interpretation']}")
    print(
        f"   liberal={result['liberal_avg']:.3f}→{result['penalised_liberal_avg']:.3f} | "
        f"illiberal={result['illiberal_avg']:.3f}→{result['penalised_illiberal_avg']:.3f}"
    )

    print(f"\n🔍 TOP LIBERAL HYPOTHESES:")
    for i, hyp in enumerate(result['top_liberal_hypotheses']):
        short_hyp = hyp['hypothesis'][:100] + "..." if len(hyp['hypothesis']) > 100 else hyp['hypothesis']
        print(f"   {i}. {hyp['probability']:.3f} - {short_hyp}")

    print(f"\n🔍 TOP ILLIBERAL HYPOTHESES:")
    for i, hyp in enumerate(result['top_illiberal_hypotheses']):
        short_hyp = hyp['hypothesis'][:100] + "..." if len(hyp['hypothesis']) > 100 else hyp['hypothesis']
        print(f"   {i}. {hyp['probability']:.3f} - {short_hyp}")

    return result


def analyze_batch(scorer, texts):
    print(f"\n{'='*120}")
    print("BATCH ANALYSIS RESULTS")
    print(f"{'='*120}")
    print(f"{'Text':<70} {'Score':<7} {'Conf':<7} {'Contr':<6} {'Interpretation'}")
    print("-" * 120)

    results = []
    for text in texts:
        result = scorer.score_liberal_illiberal(text)
        text_display       = text[:67] + "..." if len(text) > 70 else text
        contradiction_flag = "YES" if result['contradiction_detected'] else "NO"
        score_display      = f"{result['score']:.2f}" if result['is_relevant'] else "NA"

        print(
            f"{text_display:<70} "
            f"{score_display:<7} "
            f"{result['confidence']:<7.3f} "
            f"{contradiction_flag:<6} "
            f"{result['interpretation']}"
        )
        results.append(result)

    scored = [r for r in results if r['is_relevant']]
    if scored:
        scores        = [r['score']      for r in scored]
        confidences   = [r['confidence'] for r in scored]
        contradictions = sum(1 for r in scored if r['contradiction_detected'])

        print(f"\n📊 SUMMARY:")
        print(f"   Scored:          {len(scored)}/{len(results)}")
        print(f"   Score Range:     {min(scores):.2f} – {max(scores):.2f}")
        print(f"   Mean Score:      {np.mean(scores):.2f}")
        print(f"   Mean Confidence: {np.mean(confidences):.3f}")
        print(f"   Contradictions:  {contradictions}/{len(scored)} ({contradictions/len(scored)*100:.1f}%)")

    return results


def interactive_mode(scorer):
    print(f"\n{'='*60}")
    print("INTERACTIVE LIBERAL-ILLIBERAL SCORER")
    print(f"{'='*60}")
    print("Enter text to analyze (or 'quit' to exit)")
    print("Commands: 'batch' for multiple texts, 'help' for guidance")

    while True:
        text = input("\n> ").strip()

        if text.lower() in ['quit', 'exit', 'q']:
            break
        elif text.lower() == 'help':
            print("\nCommands:")
            print("- Enter any political text to get liberal-illiberal score")
            print("- 'batch' - analyze multiple predefined test texts")
            print("- 'quit'  - exit the program")
            continue
        elif text.lower() == 'batch':
            test_texts = [
                "We want an inclusive, non-exclusionary Spain, which treats its people well and seeks justice and well-being. A fair country that makes us proud to be Spanish.",
                "Perhaps he ordered the murder of Jaime Garzon did not govern later? Perhaps an important part of society does not applaud and is not afraid of that? If you want to feel fear, that is what you have to fear. If they want hope, what must be defended is the right to difference.",
                "Impunity, disarmament, political indications and corruption have generated and continue to fuel Brazil's biggest problems: violence, state inefficiency and unemployment. As important as doing new things is to undo this criminal structure created by the last governments!",
            ]
            analyze_batch(scorer, test_texts)
            continue
        elif not text:
            continue

        try:
            analyze_text(scorer, text)
        except Exception as e:
            print(f"Error analyzing text: {e}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    scorer = LiberalIlliberalScorer()
    interactive_mode(scorer)