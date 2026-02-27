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

        # Enhanced Liberal-Illiberal hypotheses using recommended format
        self.liberal_illiberal_hypotheses = {
            "The text expresses support for free and fair elections as the basis of political authority.": (1.0, "liberal"),
            "The text expresses acceptance of political competition among multiple parties.": (1.0, "liberal"),
            "The text expresses willingness to accept electoral defeat in free and fair elections.": (1.0, "liberal"),
            "The text expresses support for freedom of speech as a fundamental right.": (1.0, "liberal"),
            "The text expresses support for independent and uncensored media.": (1.0, "liberal"),
            "The text expresses the right of citizens to criticize the government without repression.": (1.0, "liberal"),
            "The text expresses support for freedom of peaceful assembly and protest.": (1.0, "liberal"),
            "The text expresses support for freedom of association, including political parties and civic groups.": (1.0, "liberal"),
            "The text expresses recognition of political opposition as legitimate.": (1.0, "liberal"),
            "The text expresses commitment to limiting political power through democratic institutions.": (1.0, "liberal"),
            "The text expresses that political legitimacy derives from the consent of the governed.": (1.0, "liberal"),
            "The text expresses protection of minority opinions and dissenting views.": (1.0, "liberal"),
            "The text expresses preference for democratic procedures over extra-legal or revolutionary change.": (1.0, "liberal"),
            "The text expresses accountability of leaders to citizens through elections.": (1.0, "liberal"),
            "The text expresses that democratic rights apply equally to all citizens.": (1.0, "liberal"),

            "The text expresses rejection of free and fair elections as necessary for governance.": (1.0, "illiberal"),
            "The text expresses support for concentration of political power in a single authority.": (1.0, "illiberal"),
            "The text expresses opposition to multi-party political competition.": (1.0, "illiberal"),
            "The text expresses justification for restricting freedom of speech.": (1.0, "illiberal"),
            "The text expresses support for state control or censorship of the media.": (1.0, "illiberal"),
            "The text expresses condemnation of political criticism as illegitimate or dangerous.": (1.0, "illiberal"),
            "The text expresses rejection of freedom of peaceful assembly or protest.": (1.0, "illiberal"),
            "The text expresses opposition to independent civic or political organizations.": (1.0, "illiberal"),
            "The text expresses that political legitimacy comes from ideology, religion, or revolution rather than elections.": (1.0, "illiberal"),
            "The text expresses refusal to accept defeat in competitive elections.": (1.0, "illiberal"),
            "The text expresses portrayal of political opponents as enemies rather than legitimate actors.": (1.0, "illiberal"),
            "The text expresses rejection of political pluralism in favor of enforced unity.": (1.0, "illiberal"),
            "The text expresses that political rights are conditional on loyalty to the regime or ideology.": (1.0, "illiberal"),
            "The text expresses depiction of democracy as harmful, weak, or corrupt.": (1.0, "illiberal"),
            "The text expresses justification for suppressing dissent to maintain order or stability.": (1.0, "illiberal"),
        }

        liberal_count = sum(1 for _, (_, direction) in self.liberal_illiberal_hypotheses.items() if direction == "liberal")
        illiberal_count = sum(1 for _, (_, direction) in self.liberal_illiberal_hypotheses.items() if direction == "illiberal")
        print(f"Loaded {len(self.liberal_illiberal_hypotheses)} hypotheses ({liberal_count} liberal, {illiberal_count} illiberal)")
        
        # Topic check configuration
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
            "This text portrays internal or external enemies as justification for repression."
        ]

    def _find_entailment_index(self):
        """Auto-detect entailment index for different NLI models"""
        config = self.model.config
        if hasattr(config, 'label2id') and config.label2id:
            for label, idx in config.label2id.items():
                if label.lower() in ['entailment', 'entail']:
                    return idx
        return 0

    def _get_entailment_prob(self, text, hypothesis):
        """Get probability that text entails hypothesis"""
        inputs = self.tokenizer(
            text, hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            prob = torch.softmax(outputs.logits, dim=-1)[0, self.entailment_idx].item()
        return prob



    def is_about_democratic_principles(self, text):
        """Check if text discusses democratic topics"""
        probs = [self._get_entailment_prob(text, h) for h in self.topic_hypotheses]
        prob = float(max(probs)) if probs else 0.0
        logger.info(f"Thesis Liberal Illiberal triggered with: {prob}")
        return prob >= self.topic_threshold, prob

    def get_hypothesis_probabilities(self, text):
        """Get probabilities for all liberal-illiberal hypotheses"""
        probs = []
        for hypothesis in self.liberal_illiberal_hypotheses.keys():
            inputs = self.tokenizer(
                text, hypothesis,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )

            with torch.no_grad():
                outputs = self.model(**inputs)
                prob = torch.softmax(outputs.logits, dim=-1)[0, self.entailment_idx].item()
                probs.append(prob)

        return np.array(probs)

    def compute_combined_confidence(self, liberal_probs, illiberal_probs, all_probs):
        """Simplified confidence with Top-K contradiction detection only"""
        
        # Basic confidence from variance (lower variance = higher confidence)
        liberal_variance = np.var(liberal_probs) if len(liberal_probs) > 1 else 0
        illiberal_variance = np.var(illiberal_probs) if len(illiberal_probs) > 1 else 0
        
        liberal_confidence = 1 / (1 + liberal_variance * 4)
        illiberal_confidence = 1 / (1 + illiberal_variance * 4)
        base_confidence = 0.7 * min(liberal_confidence, illiberal_confidence) + 0.3 * (liberal_confidence + illiberal_confidence) / 2
        
        # Top-K contradiction detection (only method we use)
        k = 5
        top_liberal = np.sort(liberal_probs)[-k:] if len(liberal_probs) >= k else liberal_probs
        top_illiberal = np.sort(illiberal_probs)[-k:] if len(illiberal_probs) >= k else illiberal_probs
        
        top_liberal_avg = np.mean(top_liberal)
        top_illiberal_avg = np.mean(top_illiberal)
        
        # Simple contradiction detection: both top-5 averages must be > 0.25
        topk_contradiction = min(top_liberal_avg, top_illiberal_avg)
        contradiction_detected = topk_contradiction > 0.25
        
        # Apply penalty if contradiction detected
        if contradiction_detected:
            contradiction_penalty = min(1.0, topk_contradiction * 2.0)
            final_confidence = base_confidence * (1 - contradiction_penalty * 0.8)
        else:
            final_confidence = base_confidence
        
        return {
            'combined': final_confidence,
            'contradiction_detected': contradiction_detected,
            'contradiction_score': topk_contradiction if contradiction_detected else 0,
            'top_liberal_avg': top_liberal_avg,
            'top_illiberal_avg': top_illiberal_avg
        }

    def score_liberal_illiberal(self, text, thr=0.15):
        """Score text and return comprehensive results"""
        # Check if text is about democratic principles
        is_relevant, topic_prob = self.is_about_democratic_principles(text)
        if not is_relevant:
            return {
                'text': text,
                'score': 'NA',
                'confidence': 0.0,
                'contradiction_detected': False,
                'interpretation': 'Not about democratic principles',
                'topic_probability': float(topic_prob),
                'passed_precheck': False,
                'is_relevant': False,

            }
        
        probs = self.get_hypothesis_probabilities(text)

        liberal_probs = []
        illiberal_probs = []
        hypothesis_results = []
        
        # Process each hypothesis
        for i, (hypothesis, (weight, direction)) in enumerate(self.liberal_illiberal_hypotheses.items()):
            prob = probs[i]
            
            hypothesis_results.append({
                'hypothesis': hypothesis,
                'probability': prob,
                'direction': direction
            })
            
            if direction == "liberal":
                liberal_probs.append(prob * weight)
            else:
                illiberal_probs.append(prob * weight)

        
        # === ADAPTIVE K (based on ALL hypotheses above threshold) ===
        k_score = int(np.sum(probs > thr)) + 1
        k_score = max(3, k_score)

        # Use top-k per side for averaging (adaptive probability logic)
        top_liberal_probs = sorted(liberal_probs, reverse=True)[:k_score]
        top_illiberal_probs = sorted(illiberal_probs, reverse=True)[:k_score]

        liberal_avg = float(np.mean(top_liberal_probs)) if top_liberal_probs else 0.0
        illiberal_avg = float(np.mean(top_illiberal_probs)) if top_illiberal_probs else 0.0


        difference = liberal_avg - illiberal_avg
        final_score = 5 + (difference * 5)
        final_score = np.clip(final_score, 0, 10)

        # Compute confidence
        confidence_data = self.compute_combined_confidence(
            [p/1.0 for p in liberal_probs],  # Unweight for confidence calc
            [p/1.0 for p in illiberal_probs],
            probs
        )

        # Get top hypotheses from each direction
        liberal_hyps = [h for h in hypothesis_results if h['direction'] == 'liberal']
        illiberal_hyps = [h for h in hypothesis_results if h['direction'] == 'illiberal']
        
        top_liberal = sorted(liberal_hyps, key=lambda x: x['probability'], reverse=True)[:5]
        top_illiberal = sorted(illiberal_hyps, key=lambda x: x['probability'], reverse=True)[:5]

        # Interpret score
        if final_score < 2:
            interpretation = "Strongly Illiberal"
        elif final_score < 4:
            interpretation = "Illiberal"
        elif final_score < 6:
            interpretation = "Moderate"
        elif final_score < 8:
            interpretation = "Liberal"
        else:
            interpretation = "Strongly Liberal"

        return {
            'text': text,
            'score': final_score,
            'confidence': confidence_data['combined'],
            'contradiction_detected': confidence_data['contradiction_detected'],
            'interpretation': interpretation,
            'liberal_avg': liberal_avg,
            'illiberal_avg': illiberal_avg,
            'top_liberal_hypotheses': top_liberal,
            'top_illiberal_hypotheses': top_illiberal,
            'passed_precheck': True,
            'is_relevant': True,
            'topic_probability': float(topic_prob),

        }

    def quick_score(self, text, thr=0.15):
        """Ultra-simple interface - just returns the numerical score"""
        result = self.score_liberal_illiberal(text, thr=thr)
        return result['score']

# ============================================================================
# INTERACTIVE ANALYSIS FUNCTIONS
# ============================================================================

def analyze_text(scorer, text):
    """Analyze a single text and display clean results"""
    result = scorer.score_liberal_illiberal(text)
    
    print(f"\n{'='*80}")
    print(f"TEXT: {text}")
    print(f"{'='*80}")
    
    print(f"\nðŸ“Š RESULTS:")
    print(f"   LiberalAvg: {result['liberal_avg']:.2f}")
    print(f"   IliberalAvg: {result['illiberal_avg']:.2f}")
    print(f"   Score: {result['score']:.2f}/10")
    print(f"   Confidence: {result['confidence']:.3f}")
    print(f"   Contradiction: {'YES' if result['contradiction_detected'] else 'NO'}")
    print(f"   Interpretation: {result['interpretation']}")
    
    print(f"\nðŸ” TOP LIBERAL HYPOTHESES:")
    for i, hyp in enumerate(result['top_liberal_hypotheses']):
        short_hyp = hyp['hypothesis'][:100] + "..." if len(hyp['hypothesis']) > 100 else hyp['hypothesis']
        print(f"   {i}. {hyp['probability']:.3f} - {short_hyp}")
    
    print(f"\nðŸ” TOP ILLIBERAL HYPOTHESES:")
    for i, hyp in enumerate(result['top_illiberal_hypotheses']):
        short_hyp = hyp['hypothesis'][:100] + "..." if len(hyp['hypothesis']) > 100 else hyp['hypothesis']
        print(f"   {i}. {hyp['probability']:.3f} - {short_hyp}")
    
    return result

def analyze_batch(scorer, texts):
    """Analyze multiple texts and display summary table"""
    print(f"\n{'='*120}")
    print("BATCH ANALYSIS RESULTS")
    print(f"{'='*120}")
    
    print(f"{'Text':<100} {'Score':<7} {'Conf':<7} {'Contr':<6} {'Interpretation'}")
    print("-" * 120)
    
    results = []
    for text in texts:
        result = scorer.score_liberal_illiberal(text)
        text_display = text[:97] + "..." if len(text) > 100 else text
        contradiction_status = "YES" if result['contradiction_detected'] else "NO"
        
        print(f"{text_display:<70} {result['score']:<7.2f} {result['confidence']:<7.3f} {contradiction_status:<6} {result['interpretation']}")
        results.append(result)
    
    # Summary statistics
    scores = [r['score'] for r in results]
    confidences = [r['confidence'] for r in results]
    contradictions = sum(1 for r in results if r['contradiction_detected'])
    
    print(f"\nðŸ“Š SUMMARY:")
    print(f"   Score Range: {min(scores):.2f} - {max(scores):.2f}")
    print(f"   Mean Confidence: {np.mean(confidences):.3f}")
    print(f"   Contradictions: {contradictions}/{len(results)} ({contradictions/len(results)*100:.1f}%)")
    
    return results

def interactive_mode(scorer):
    """Interactive mode for testing individual texts"""
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
            print("- 'quit' - exit the program")
            continue
        elif text.lower() == 'batch':
            test_texts = [
                "We want an inclusive, non-exclusionary Spain, which treats its people well and seeks justice and well-being. A fair country that makes us proud to be Spanish."
                "Perhaps he ordered the murder of Jaime Garzon did not govern later? Perhaps an important part of society does not applaud and is not afraid of that? If you want to feel fear, that is what you have to fear. If they want hope, what must be defended is the right to difference."
                "Impunity, disarmament, political indications and corruption have generated and continue to fuel Brazil's biggest problems: violence, state inefficiency and unemployment. As important as doing new things is to undo this criminal structure created by the last governments!"
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
    # Initialize scorer
    scorer = LiberalIlliberalScorer()
    
    # Run interactive mode
    interactive_mode(scorer)
