import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .shared_model_cache import SharedModelCache
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# LEFT-RIGHT ECONOMIC HYPOTHESIS-BASED SCORER
# ============================================================================

class LeftRightEconomicScorer:
    def __init__(self, model_name="mlburnham/Political_DEBATE_large_v1.0"):
        cache = SharedModelCache()
        self.model, self.tokenizer = cache.get_model_and_tokenizer(model_name)
        self.entailment_idx = self._find_entailment_index()

        # Topic precheck: only score texts that clearly discuss governance / institutions / political rhetoric
        self.topic_question = (
            "Does this text discuss economic policy, government intervention, or public services? "
            "This includes topics like healthcare, education, housing, transport, taxation, natural resources "
            "privatization, welfare, regulation, minimum wage, wealth redistribution, "
            "public vs. private sector roles, or economic equality."
        )
        self.topic_threshold = 0.30

        # Left-Right Economic hypotheses - streamlined to ~15 per side
        self.left_right_hypotheses = {
            # Left Economic Positions (15) - More specific and policy-focused
            "The author of this text believes corporations should pay higher taxes": (1.0, "left"),
            "The author of this text believes wealthy individuals should pay higher tax rates": (1.0, "left"),
            "The author of this text believes government should increase spending on healthcare": (1.0, "left"),
            "The author of this text believes government should increase spending on education": (1.0, "left"),
            "The author of this text believes unemployment benefits should be expanded": (1.0, "left"),
            "The author of this text believes government should provide universal healthcare": (1.0, "left"),
            "The author of this text believes banks and financial institutions should be heavily regulated": (1.0, "left"),
            "The author of this text believes environmental regulations on business are necessary": (1.0, "left"),
            "The author of this text believes utilities should be publicly owned": (1.0, "left"),
            "The author of this text believes government should break up large corporations": (1.0, "left"),
            "The author of this text believes minimum wage laws should be strengthened": (1.0, "left"),
            "The author of this text believes unions should have more power": (1.0, "left"),
            "The author of this text believes government should reduce income inequality": (1.0, "left"),
            "The author of this text believes public investment creates jobs": (1.0, "left"),
            "The author of this text believes social safety nets should be expanded": (1.0, "left"),
            
            # Right Economic Positions (15) - More specific and policy-focused  
            "The author of this text believes corporate tax rates should be lowered": (1.0, "right"),
            "The author of this text believes income taxes should be reduced": (1.0, "right"),
            "The author of this text believes government spending on social programs should be cut": (1.0, "right"),
            "The author of this text believes welfare programs should be reduced": (1.0, "right"),
            "The author of this text believes healthcare should be privatized": (1.0, "right"),
            "The author of this text believes education should be privatized": (1.0, "right"),
            "The author of this text believes financial regulations should be eliminated": (1.0, "right"),
            "The author of this text believes environmental regulations hurt business competitiveness": (1.0, "right"),
            "The author of this text believes government services should be privatized": (1.0, "right"),
            "The author of this text believes large corporations drive economic growth": (1.0, "right"),
            "The author of this text believes minimum wage laws hurt employment": (1.0, "right"),
            "The author of this text believes unions hurt economic competitiveness": (1.0, "right"),
            "The author of this text believes income inequality reflects merit and effort": (1.0, "right"),
            "The author of this text believes private investment is more efficient than public": (1.0, "right"),
            "The author of this text believes social programs create dependency": (1.0, "right"),
        }


        left_count = sum(1 for _, (_, direction) in self.left_right_hypotheses.items() if direction == "left")
        right_count = sum(1 for _, (_, direction) in self.left_right_hypotheses.items() if direction == "right")
        print(f"Loaded {len(self.left_right_hypotheses)} hypotheses ({left_count} left, {right_count} right)")

    def _find_entailment_index(self):
        """Auto-detect entailment index for different NLI models"""
        config = self.model.config
        if hasattr(config, 'label2id') and config.label2id:
            for label, idx in config.label2id.items():
                if label.lower() in ['entailment', 'entail']:
                    return idx
        return 0

    def _entailment_prob(self, premise: str, hypothesis: str) -> float:
        """Return entailment probability for (premise, hypothesis) using the loaded NLI model."""
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
            return torch.softmax(outputs.logits, dim=-1)[0, self.entailment_idx].item()

    def topic_precheck(self, text: str) -> dict:
        """Check whether the text is in-scope for (political rhetoric/governance/institutions) analysis."""
        score = float(self._entailment_prob(text, self.topic_question))
        return {
            "passed": score >= float(self.topic_threshold),
            "score": score,
            "threshold": float(self.topic_threshold),
            "question": self.topic_question,
        }


    def get_hypothesis_probabilities(self, text):
        """Get probabilities for all left-right hypotheses"""
        probs = []
        for hypothesis in self.left_right_hypotheses.keys():
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

    def compute_combined_confidence(self, left_probs, right_probs, all_probs):
        """Simplified confidence with Top-K contradiction detection only"""
        
        # Basic confidence from variance (lower variance = higher confidence)
        left_variance = np.var(left_probs) if len(left_probs) > 1 else 0
        right_variance = np.var(right_probs) if len(right_probs) > 1 else 0
        
        left_confidence = 1 / (1 + left_variance * 4)
        right_confidence = 1 / (1 + right_variance * 4)
        base_confidence = 0.7 * min(left_confidence, right_confidence) + 0.3 * (left_confidence + right_confidence) / 2
        
        # Top-K contradiction detection (only method we use)
        k = 5
        top_left = np.sort(left_probs)[-k:] if len(left_probs) >= k else left_probs
        top_right = np.sort(right_probs)[-k:] if len(right_probs) >= k else right_probs
        
        top_left_avg = np.mean(top_left)
        top_right_avg = np.mean(top_right)
        
        # Simple contradiction detection: both top-5 averages must be > 0.25
        topk_contradiction = min(top_left_avg, top_right_avg)
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
            'top_left_avg': top_left_avg,
            'top_right_avg': top_right_avg
        }

    def score_left_right(self, text):
        """Score text and return comprehensive results"""

        # Topic precheck (guards against scoring non-political/non-governance text)
        precheck = self.topic_precheck(text)
        if not precheck["passed"]:
            return {
                "text": text,
                "passed_precheck": False,
                "precheck_score": precheck["score"],
                "precheck_threshold": precheck["threshold"],
                "precheck_question": precheck["question"],
                "error": "Text did not pass the topic precheck.",
            }

        probs = self.get_hypothesis_probabilities(text)

        left_probs = []
        right_probs = []
        hypothesis_results = []
        
        # Process each hypothesis
        for i, (hypothesis, (weight, direction)) in enumerate(self.left_right_hypotheses.items()):
            prob = probs[i]
            
            hypothesis_results.append({
                'hypothesis': hypothesis,
                'probability': prob,
                'direction': direction
            })
            
            if direction == "left":
                left_probs.append(prob * weight)
            else:
                right_probs.append(prob * weight)

        # Calculate averages and score
        left_avg = np.mean(left_probs) if left_probs else 0
        right_avg = np.mean(right_probs) if right_probs else 0
        
        difference = left_avg - right_avg
        final_score = 5 - (difference * 5)  # Flipped: left = low numbers, right = high numbers
        final_score = np.clip(final_score, 0, 10)

        # Compute confidence
        confidence_data = self.compute_combined_confidence(
            [p/1.0 for p in left_probs],  # Unweight for confidence calc
            [p/1.0 for p in right_probs],
            probs
        )

        # Get top hypotheses from each direction
        left_hyps = [h for h in hypothesis_results if h['direction'] == 'left']
        right_hyps = [h for h in hypothesis_results if h['direction'] == 'right']
        
        top_left = sorted(left_hyps, key=lambda x: x['probability'], reverse=True)[:5]
        top_right = sorted(right_hyps, key=lambda x: x['probability'], reverse=True)[:5]

        # Interpret score (0-10 scale: 0=Far Left, 5=Center, 10=Far Right)
        if final_score < 2:
            interpretation = "Far Left"
        elif final_score < 4:
            interpretation = "Left"
        elif final_score < 6:
            interpretation = "Center"
        elif final_score < 8:
            interpretation = "Right"
        else:
            interpretation = "Far Right"

        return {
            'text': text,
            'passed_precheck': True,
            'precheck_score': precheck['score'],
            'precheck_threshold': precheck['threshold'],
            'score': final_score,
            'confidence': confidence_data['combined'],
            'contradiction_detected': confidence_data['contradiction_detected'],
            'interpretation': interpretation,
            'left_avg': left_avg,
            'right_avg': right_avg,
            'top_left_hypotheses': top_left,
            'top_right_hypotheses': top_right
        }

    def quick_score(self, text):
        """Ultra-simple interface - just returns the numerical score"""
        result = self.score_left_right(text)
        return result['score']

# ============================================================================
# INTERACTIVE ANALYSIS FUNCTIONS
# ============================================================================

def analyze_text(scorer, text):
    """Analyze a single text and display clean results"""
    result = scorer.score_left_right(text)
    
    print(f"\n{'='*80}")
    print(f"TEXT: {text}")
    print(f"{'='*80}")
    
    print(f"\n📊 RESULTS:")
    print(f"   LeftAvg: {result['left_avg']:.2f}")
    print(f"   RightAvg: {result['right_avg']:.2f}")
    print(f"   Score: {result['score']:.2f}/10 (0=Far Left, 10=Far Right)")
    print(f"   Confidence: {result['confidence']:.3f}")
    print(f"   Contradiction: {'YES' if result['contradiction_detected'] else 'NO'}")
    print(f"   Interpretation: {result['interpretation']}")
    
    print(f"\n🔍 TOP LEFT HYPOTHESES:")
    for i, hyp in enumerate(result['top_left_hypotheses']):
        short_hyp = hyp['hypothesis'][:100] + "..." if len(hyp['hypothesis']) > 100 else hyp['hypothesis']
        print(f"   {i}. {hyp['probability']:.3f} - {short_hyp}")
    
    print(f"\n🔍 TOP RIGHT HYPOTHESES:")
    for i, hyp in enumerate(result['top_right_hypotheses']):
        short_hyp = hyp['hypothesis'][:100] + "..." if len(hyp['hypothesis']) > 100 else hyp['hypothesis']
        print(f"   {i}. {hyp['probability']:.3f} - {short_hyp}")
    
    return result

def analyze_batch(scorer, texts):
    """Analyze multiple texts and display summary table"""
    print(f"\n{'='*120}")
    print("BATCH ANALYSIS RESULTS")
    print(f"{'='*120}")
    
    print(f"{'Text':<70} {'Score':<7} {'Conf':<7} {'Contr':<6} {'Interpretation'}")
    print("-" * 120)
    
    results = []
    for text in texts:
        result = scorer.score_left_right(text)
        text_display = text[:67] + "..." if len(text) > 70 else text
        contradiction_status = "YES" if result['contradiction_detected'] else "NO"
        
        print(f"{text_display:<70} {result['score']:<7.2f} {result['confidence']:<7.3f} {contradiction_status:<6} {result['interpretation']}")
        results.append(result)
    
    # Summary statistics
    scores = [r['score'] for r in results]
    confidences = [r['confidence'] for r in results]
    contradictions = sum(1 for r in results if r['contradiction_detected'])
    
    print(f"\n📊 SUMMARY:")
    print(f"   Score Range: {min(scores):.2f} - {max(scores):.2f}")
    print(f"   Mean Score: {np.mean(scores):.2f}")
    print(f"   Mean Confidence: {np.mean(confidences):.3f}")
    print(f"   Contradictions: {contradictions}/{len(results)} ({contradictions/len(results)*100:.1f}%)")
    
    return results

def interactive_mode(scorer):
    """Interactive mode for testing individual texts"""
    print(f"\n{'='*60}")
    print("INTERACTIVE LEFT-RIGHT ECONOMIC SCORER")
    print(f"{'='*60}")
    print("Enter text to analyze (or 'quit' to exit)")
    print("Commands: 'batch' for multiple texts, 'help' for guidance")
    print("Scale: 0 = Far Left, 5 = Center, 10 = Far Right")
    
    while True:
        text = input("\n> ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            break
        elif text.lower() == 'help':
            print("\nCommands:")
            print("- Enter any economic policy text to get left-right score")
            print("- 'batch' - analyze multiple predefined test texts")
            print("- 'quit' - exit the program")
            print("\nScoring:")
            print("- 0-2: Far Left (extensive government role, wealth redistribution)")
            print("- 2-4: Left (active government, social programs)")
            print("- 4-6: Center (mixed economy)")
            print("- 6-8: Right (limited government role, pro-business)")
            print("- 8-10: Far Right (minimal government, maximum market freedom)")
            continue
        elif text.lower() == 'batch':
            test_texts = [
                "We need to cut taxes and reduce government spending to boost economic growth.",
                "The government should provide universal healthcare and free education for all.",
                "Private companies are more efficient than government-run services.",
                "Wealth inequality requires higher taxes on the rich and stronger social programs.",
                "Deregulation will unleash business innovation and create jobs.",
                "We must strengthen worker protections and raise the minimum wage.",
                "Free markets allocate resources better than government planning.",
                "Essential services like healthcare should be publicly owned and funded."
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
    scorer = LeftRightEconomicScorer()
    
    # Run interactive mode
    interactive_mode(scorer)
