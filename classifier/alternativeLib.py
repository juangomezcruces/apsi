import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .shared_model_cache import SharedModelCache
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# LIBERAL-ILLIBERAL HYPOTHESIS-BASED SCORER
# ============================================================================




class LiberalIlliberalScorer:
    def __init__(self, model_name="mlburnham/Political_DEBATE_large_v1.0"):
        cache = SharedModelCache()
        self.model, self.tokenizer = cache.get_model_and_tokenizer(model_name)
        self.entailment_idx = self._find_entailment_index()


        self.topic_question = (
            "Does this text discuss political ideas related to democratic principles?"
        )

        self.topic_threshold = 0.60 


        # Enhanced Liberal-Illiberal hypotheses using recommended format
        self.liberal_illiberal_hypotheses = {
            # Electoral Democracy & Competition
            "The author of this text believes in competitive multi-party elections": (1.0, "liberal"),
            "The author of this text believes in fair electoral processes": (1.0, "liberal"),
            "The author of this text believes electoral competition is beneficial": (1.0, "liberal"),
            "The author of this text believes in single-party dominance": (1.0, "illiberal"),
            "The author of this text believes electoral competition should be opposed": (1.0, "illiberal"),
            "The author of this text believes election results are illegitimate without providing evidence": (1.0, "illiberal"),
        
            # Freedom of Speech
            "The author of this text believes in freedom of speech and expression": (1.0, "liberal"),
            "The author of this text believes in the right to free expression": (1.0, "liberal"),
            "The author of this text believes speech and expression should be restricted": (1.0, "illiberal"),
            "The author of this text believes political criticism should be censored": (1.0, "illiberal"),
        
            # Media Freedom
            "The author of this text believes in media independence": (1.0, "liberal"),
            "The author of this text believes in press freedom": (1.0, "liberal"),
            "The author of this text believes in media accountability": (1.0, "liberal"),
            "The author of this text believes media freedom should be restricted": (1.0, "illiberal"),
            "The author of this text believes media content should be controlled": (1.0, "illiberal"),
            "The author of this text believes traditional media is fake or dishonest": (1.0, "illiberal"),
        
            # Freedom of Assembly
            "The author of this text believes in freedom of assembly": (1.0, "liberal"),
            "The author of this text believes in the right to peaceful protest": (1.0, "liberal"),
            "The author of this text believes freedom of assembly should be restricted": (1.0, "illiberal"),
            "The author of this text believes peaceful protests should be limited": (1.0, "illiberal"),
        
            # Freedom of Association
            "The author of this text believes in freedom of association": (1.0, "liberal"),
            "The author of this text believes in independent civil society groups": (1.0, "liberal"),
            "The author of this text believes freedom of association should be restricted": (1.0, "illiberal"),
            "The author of this text believes civil society should be limited": (1.0, "illiberal"),
        
            # Democratic Acceptance
            "The author of this text believes in accepting democratic electoral outcomes": (1.0, "liberal"),
            "The author of this text believes political opposition is legitimate": (1.0, "liberal"),
            "The author of this text believes electoral defeats are illegitimate": (1.0, "illiberal"),
            "The author of this text believes political opponents should be delegitimized": (1.0, "illiberal"),
        
            # Institutional Democratic Norms
            "The author of this text believes in supporting democratic institutions": (1.0, "liberal"),
            "The author of this text believes in the rule of law": (1.0, "liberal"),
            "The author of this text believes institutions should only be changed through democratic procedures": (1.0, "liberal"),
            "The author of this text believes democratic institutions should be bypassed": (1.0, "illiberal"),
            "The author of this text believes executive power should be concentrated in a strong leader": (1.0, "illiberal"),
            "The author of this text believes institutions are fundamentally corrupt": (1.0, "illiberal"),
        }

        liberal_count = sum(1 for _, (_, direction) in self.liberal_illiberal_hypotheses.items() if direction == "liberal")
        illiberal_count = sum(1 for _, (_, direction) in self.liberal_illiberal_hypotheses.items() if direction == "illiberal")
        print(f"Loaded {len(self.liberal_illiberal_hypotheses)} hypotheses ({liberal_count} liberal, {illiberal_count} illiberal)")

    def _find_entailment_index(self):

            """Auto-detect entailment index for different NLI models.

    

            Tries label2id/id2label mappings first. If unavailable, falls back to a

            reasonable default for 3-way NLI heads (often: 0=contradiction,

            1=neutral, 2=entailment).

            """

            config = getattr(self.model, "config", None)

    

            # 1) label2id

            label2id = getattr(config, "label2id", None) if config is not None else None

            if isinstance(label2id, dict) and label2id:

                for label, idx in label2id.items():

                    if str(label).lower() in ["entailment", "entails", "entail"]:

                        return int(idx)

    

            # 2) id2label

            id2label = getattr(config, "id2label", None) if config is not None else None

            if isinstance(id2label, dict) and id2label:

                for idx, label in id2label.items():

                    if str(label).lower() in ["entailment", "entails", "entail"]:

                        return int(idx)

                # Sometimes labels look like "LABEL_0" etc; if we can infer MNLI-like, use last label

                try:

                    num_labels = int(getattr(config, "num_labels", 0) or 0)

                except Exception:

                    num_labels = 0

                if num_labels == 3:

                    return 2

    

            # 3) final fallback based on num_labels

            try:

                num_labels = int(getattr(config, "num_labels", 0) or 0)

            except Exception:

                num_labels = 0

            if num_labels == 3:

                return 2

            if num_labels > 0:

                return num_labels - 1

    

            # Worst-case fallback

            return 0

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

    def _topic_precheck(self, text: str):
        """Lightweight topic gate using the same NLI model.

        Returns:
            passed (bool), entailment_probability (float)
        """
        inputs = self.tokenizer(
            text,
            self.topic_question,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            prob = torch.softmax(outputs.logits, dim=-1)[0, self.entailment_idx].item()

        return (prob >= self.topic_threshold), float(prob)

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

    def score_liberal_illiberal(self, text):
        """Score text and return comprehensive results"""

        # --- Topic precheck ---
        passed, p_entail = self._topic_precheck(text)
        if not passed:
            return {
                "ok": False,
                "error_code": "TOPIC_PRECHECK_FAILED",
                "error_message": "This text doesn’t appear to be about democratic principles, so a Liberal–Illiberal score wasn’t computed.",
                "topic_entailment": p_entail,
                "topic_threshold": self.topic_threshold,
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

        # Calculate averages and score
        liberal_avg = np.mean(liberal_probs) if liberal_probs else 0
        illiberal_avg = np.mean(illiberal_probs) if illiberal_probs else 0
        
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
            'ok': True,
            'text': text,
            'score': float(final_score),
            'confidence': float(confidence_data['combined']),
            'contradiction_detected': bool(confidence_data['contradiction_detected']),
            'interpretation': interpretation,
            'liberal_avg': float(liberal_avg),
            'illiberal_avg': float(illiberal_avg),
            'top_liberal_hypotheses': top_liberal,
            'top_illiberal_hypotheses': top_illiberal
        }

    def quick_score(self, text):
        """Ultra-simple interface - just returns the numerical score"""
        result = self.score_liberal_illiberal(text)
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
    
    print(f"\n📊 RESULTS:")
    print(f"   LiberalAvg: {result['liberal_avg']:.2f}")
    print(f"   IliberalAvg: {result['illiberal_avg']:.2f}")
    print(f"   Score: {result['score']:.2f}/10")
    print(f"   Confidence: {result['confidence']:.3f}")
    print(f"   Contradiction: {'YES' if result['contradiction_detected'] else 'NO'}")
    print(f"   Interpretation: {result['interpretation']}")
    
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
    
    print(f"\n📊 SUMMARY:")
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
                "We must protect democratic institutions and ensure free elections.",
                "The fake news media is the enemy of the people.",
                "Democratic elections are essential but corrupt institutions must be bypassed.",
                "Political elites have rigged the system against ordinary people."
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