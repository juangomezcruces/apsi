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
# LEFT-RIGHT ECONOMIC HYPOTHESIS-BASED SCORER
# ============================================================================

class LeftRightEconomicScorer:
    def __init__(self, model_name="mlburnham/Political_DEBATE_large_v1.0"):
        cache = SharedModelCache()
        self.model, self.tokenizer = cache.get_model_and_tokenizer(model_name)
        self.entailment_idx = self._find_entailment_index()

        # Left-Right Economic hypotheses - streamlined to ~15 per side
        self.left_right_hypotheses = {
            # left wing hypotheses
            "The text expresses that the state should own and control the means of production and major industries": (1.5, "left"),

            "The text expresses that market mechanisms should be replaced by central state planning and allocation": (1.5, "left"),

            "The text expresses that corporations should pay higher taxes": (0.8, "left"),
    
            "The text expresses that wealthy individuals should pay higher tax rates": (0.85, "left"),
    
            "The text expresses that government should increase spending on healthcare": (0.7, "left"),
    
            "The text expresses that government should increase spending on education": (0.7, "left"),
    
            "The text expresses that unemployment benefits should be expanded": (0.8, "left"),
    
            "The text expresses that government should provide universal healthcare": (0.85, "left"),
    
            "The text expresses that banks and financial institutions should be heavily regulated": (0.85, "left"),
    
            "The text expresses that environmental regulations on business are necessary": (0.75, "left"),
    
            "The text expresses that utilities should be publicly owned": (1.0, "left"),
    
            "The text expresses that government should break up large corporations": (0.9, "left"),
    
            "The text expresses that minimum wage laws should be strengthened": (0.85, "left"),
    
            "The text expresses that unions should have more power": (0.9, "left"),
    
            "The text expresses that government should reduce income inequality": (0.8, "left"),
    
            "The text expresses that public investment creates jobs": (0.75, "left"),
    
            "The text expresses that social safety nets should be expanded": (0.7, "left"),
    
            "The text expresses that government should have a very active role in the economy": (0.95, "left"),
    
            # Right Economic Positions
            "The text expresses that the means of production and major industries should be privately owned and free from state control": (1.5, "right"),

            "The text expresses that free market mechanisms should replace state planning and allocation": (1.5, "right"),

            "The text expresses that corporate tax rates should be lowered": (0.85, "right"),
    
            "The text expresses that income taxes should be reduced": (0.75, "right"),
    
            "The text expresses that government spending on social programs should be cut": (0.9, "right"),
    
            "The text expresses that welfare programs should be reduced": (0.95, "right"),
    
            "The text expresses that healthcare should be privatized": (1.0, "right"),
    
            "The text expresses that education should be privatized": (0.9, "right"),

            "The text expresses that government must partner with businesses": (0.5, "right"),
    
            "The text expresses that financial regulations should be eliminated": (0.95, "right"),
    
            "The text expresses that environmental regulations hurt business competitiveness": (0.75, "right"),
    
            "The text expresses that government services should be privatized": (1.0, "right"),
    
            "The text expresses that large corporations drive economic growth": (0.65, "right"),
    
            "The text expresses that minimum wage laws hurt employment": (0.85, "right"),
    
            "The text expresses that unions hurt economic competitiveness": (0.9, "right"),

            "The text expresses that government intervention in the economy should be focused on helping businesses": (0.9, "right"),
    
            "The text expresses that income inequality reflects merit and effort": (0.9, "right"),
    
            "The text expresses that private investment is more efficient than public": (0.85, "right"),
    
            "The text expresses that social programs create dependency": (0.95, "right"),

            "The text expresses that public assets and state-owned enterprises should be privatized": (1.0, "right"),
        }


        left_count = sum(1 for _, (_, direction) in self.left_right_hypotheses.items() if direction == "left")
        right_count = sum(1 for _, (_, direction) in self.left_right_hypotheses.items() if direction == "right")
        print(f"Loaded {len(self.left_right_hypotheses)} hypotheses ({left_count} left, {right_count} right)")
        
        # Precheck hypotheses
        self.topic_threshold = 0.7
        self.topic_hypotheses = [
            # Role of government in the economy
            "The text expresses an opinion about the role of government in the economy.",
            "The text argues that the government should play a role in economic affairs.",
            "The text supports or opposes government involvement in the economy.",
            "The text expresses a stance on how much power the government should have over the economy.",
            "The text argues for increasing or reducing the role of government in economic policy.",

            # Taxation and redistribution
            "The text expresses an opinion about taxation or tax policy.",
            "The text argues about how wealth should be redistributed.",
            "The text expreses a stance about higher taxes.",
            "The text expresses a stance on income inequality or redistribution.",
            "The text argues how taxes should be used to achieve economic goals.",

            # Ownership and control
            "The text expresses an opinion about public versus private ownership.",
            "The text argues that industries should be publicly or privately owned.",
            "The text supports or opposes nationalization or privatization.",
            "The text expresses a stance on who should control major economic resources.",
            "The text argues how ownership of businesses should be structured.",

            # Regulation vs markets
            "The text expresses a stance on market regulation.",
            "The text argues how free markets should be regulated.",
            "The text supports or opposes government regulation of businesses.",
            "The text expresses a stance on market freedom",
            "The text expreses a stance about regulation of the economy.",

            # Welfare and social programs
            "The text expresses an opinion about welfare or social programs.",
            "The text argues that social programs should be expanded or reduced.",
            "The text supports or opposes government-funded social services.",
            "The text expresses a stance on healthcare, education, or social safety nets.",
            "The text argues how social programs should be organized or funded.",

            # Labor markets / wages
            "The text expresses a stance in favor or against minimum wages.",
            "The text argues about the minimum wage or worker protections.",
            "The text supports or opposes labor regulations or unions.",
            "The text expresses a stance on workers' rights or employment standards.",
            "The text argues how labor markets should be regulated."
        ]


    def _find_entailment_index(self):
        """Auto-detect entailment index for different NLI models"""
        config = self.model.config
        if hasattr(config, 'label2id') and config.label2id:
            for label, idx in config.label2id.items():
                if label.lower() in ['entailment', 'entail']:
                    return idx
        return 0

    def _batch_entailment_probs(self, text, hypotheses, batch_size=16):
        """Get entailment probabilities for multiple hypotheses in batched forward passes."""
        all_probs = []
        for i in range(0, len(hypotheses), batch_size):
            batch_hyps = hypotheses[i:i + batch_size]
            inputs = self.tokenizer(
                [text] * len(batch_hyps),
                batch_hyps,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            )
            with torch.inference_mode():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)[:, self.entailment_idx]
                all_probs.extend(probs.tolist())
        return all_probs

    def is_about_economic_policy(self, text):
        """Check if text is relevant by max entailment over topic hypotheses (batched)"""
        probs = self._batch_entailment_probs(text, self.topic_hypotheses)
        prob = float(max(probs)) if probs else 0.0
        logger.info(f"Thesis Left Right triggered with: {prob}")
        return prob >= self.topic_threshold, prob

    def get_hypothesis_probabilities(self, text):
        """Get probabilities for all left-right hypotheses (batched)"""
        hypotheses = list(self.left_right_hypotheses.keys())
        probs = self._batch_entailment_probs(text, hypotheses)
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

    def score_left_right(self, text, thr=0.15):
        """Score text and return comprehensive results"""
        # Check if text is about economic policy
        is_relevant, topic_prob = self.is_about_economic_policy(text)
        if not is_relevant:
            return {
                'text': text,
                'score': 'NA',
                'confidence': 0.0,
                'contradiction_detected': False,
                'interpretation': 'Not about economic policy',
                'is_relevant': False,
                'topic_probability': float(topic_prob),
                'passed_precheck': False,
                'is_relevant': False,
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
                'weight': weight,
                'direction': direction
            })
            
            if direction == "left":
                left_probs.append(prob * weight)
            else:
                right_probs.append(prob * weight)

        # Calculate averages and score
        # === ADAPTIVE K (based on ALL hypotheses above threshold) ===
        k_score = int(np.sum(probs > thr)) + 2
        k_score = max(4, k_score)

        # Use top-k per side for averaging (adaptive probability logic)
        top_left_probs = sorted(left_probs, reverse=True)[:k_score]
        top_right_probs = sorted(right_probs, reverse=True)[:k_score]

        left_avg = float(np.mean(top_left_probs)) if top_left_probs else 0.0
        right_avg = float(np.mean(top_right_probs)) if top_right_probs else 0.0
          
        
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

        # Annotate each hypothesis with its score impact (points on 0-10 scale).
        # Each hypothesis in the top-k contributes (prob*weight / k_score) to its side's avg,
        # which maps to (contrib_to_avg * 5) points of score movement.
        # Left pulls score down, right pulls score up — we store absolute impact.
        top_left_weighted = sorted([(h['probability'] * h['weight'], h) for h in left_hyps], reverse=True)[:k_score]
        top_right_weighted = sorted([(h['probability'] * h['weight'], h) for h in right_hyps], reverse=True)[:k_score]

        for weighted_val, h in top_left_weighted:
            h['score_impact'] = round((weighted_val / k_score) * 5, 3)
        for weighted_val, h in top_right_weighted:
            h['score_impact'] = round((weighted_val / k_score) * 5, 3)
        # Hypotheses outside the top-k got no weight in the average
        top_left_set = {id(h) for _, h in top_left_weighted}
        top_right_set = {id(h) for _, h in top_right_weighted}
        for h in left_hyps:
            if id(h) not in top_left_set:
                h['score_impact'] = 0.0
        for h in right_hyps:
            if id(h) not in top_right_set:
                h['score_impact'] = 0.0

        top_left = sorted(left_hyps, key=lambda x: x['probability'], reverse=True)[:10]
        top_right = sorted(right_hyps, key=lambda x: x['probability'], reverse=True)[:10]

        # Interpret score (0-10 scale: 0=Far Left, 5=Center, 10=Far Right)
        if final_score < 1.43:
            interpretation = "Strong Left"
        elif final_score < 2.86:
            interpretation = "Left"
        elif final_score < 4.29:
            interpretation = "Center Left"
        elif final_score <= 5.71:
            interpretation = "Center"
        elif final_score < 7.14:
            interpretation = "Center Right"
        elif final_score < 8.57:
            interpretation = "Right"
        else:
            interpretation = "Strong Right"

        return {
            'text': text,
            'score': final_score,
            'confidence': confidence_data['combined'],
            'contradiction_detected': confidence_data['contradiction_detected'],
            'interpretation': interpretation,
            'left_avg': left_avg,
            'right_avg': right_avg,
            'top_left_hypotheses': top_left,
            'top_right_hypotheses': top_right,
            'passed_precheck': True,
            'is_relevant': True,
            'topic_probability': float(topic_prob),
            
        }

    def quick_score(self, text, thr=0.15):
        """Ultra-simple interface - just returns the numerical score"""
        result = self.score_left_right(text, thr=thr)
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
    
    print(f"\nðŸ“Š RESULTS:")
    print(f"   LeftAvg: {result['left_avg']:.2f}")
    print(f"   RightAvg: {result['right_avg']:.2f}")
    print(f"   Score: {result['score']:.2f}/10 (0=Far Left, 10=Far Right)")
    print(f"   Confidence: {result['confidence']:.3f}")
    print(f"   Contradiction: {'YES' if result['contradiction_detected'] else 'NO'}")
    print(f"   Interpretation: {result['interpretation']}")
    
    print(f"\nðŸ” TOP LEFT HYPOTHESES:")
    for i, hyp in enumerate(result['top_left_hypotheses']):
        short_hyp = hyp['hypothesis'][:200] + "..." if len(hyp['hypothesis']) > 200 else hyp['hypothesis']
        print(f"   {i}. {hyp['probability']:.3f} - {short_hyp}")
    
    print(f"\nðŸ” TOP RIGHT HYPOTHESES:")
    for i, hyp in enumerate(result['top_right_hypotheses']):
        short_hyp = hyp['hypothesis'][:200] + "..." if len(hyp['hypothesis']) > 200 else hyp['hypothesis']
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
    
    print(f"\nðŸ“Š SUMMARY:")
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