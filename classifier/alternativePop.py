import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .shared_model_cache import SharedModelCache
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# POPULISM-PLURALISM HYPOTHESIS-BASED SCORER
# ============================================================================

class PopulismPluralismScorer:
    def __init__(self, model_name="mlburnham/Political_DEBATE_large_v1.0"):
        cache = SharedModelCache()
        self.model, self.tokenizer = cache.get_model_and_tokenizer(model_name)
        self.entailment_idx = self._find_entailment_index()

        # Populism-Pluralism hypotheses based on GPS and V-Party definitions
        # Populism-Pluralism hypotheses - refined for specificity
        self.populism_hypotheses = {
            # Populist Positions (18) - Much more specific
            "The author of this text believes corrupt politicians have betrayed ordinary working people": (1.0, "populist"),
            "The author of this text believes citizens should vote directly on major issues instead of trusting representatives": (1.0, "populist"),
            "The author of this text believes elites are out of touch with regular voters": (1.0, "populist"),
            "The author of this text believes the political system is rigged to benefit wealthy donors": (1.0, "populist"),
            "The author of this text believes ordinary citizens have more common sense than political experts": (1.0, "populist"),
            "The author of this text believes career politicians care more about donors than ordinary voters": (1.0, "populist"),
            "The author of this text believes the media and establishment work together against the people": (1.0, "populist"),
            "The author of this text believes voters share the same basic values and priorities": (1.0, "populist"),
            "The author of this text believes government bureaucrats obstruct the will of the people": (1.0, "populist"),
            "The author of this text believes outsiders can fix problems that elites have created": (1.0, "populist"),
            "The author of this text believes mainstream media suppresses the voices of regular people": (1.0, "populist"),
            "The author of this text believes the government serves only the interest of corporations and wealthy individuals": (1.0, "populist"),
            "The author of this text believes hard-working families are ignored by self-serving politicians": (1.0, "populist"),
            "The author of this text believes common people understand what's best for the country": (1.0, "populist"),
            "The author of this text believes political insiders resist giving power back to working families": (1.0, "populist"),
            "The author of this text believes the political class has formed a conspiracy against ordinary citizens": (1.0, "populist"),
            "The author of this text believes the majority should decide without being blocked by courts or elites": (1.0, "populist"),
            "The author of this text believes wealthy special interests have corrupted the entire political system": (1.0, "populist"),            
            
            # Pluralist Positions (18) - Much more specific
            "The author of this text believes different communities have legitimate but conflicting needs": (1.0, "pluralist"),
            "The author of this text believes political negotiations and compromises are a core democratic principle": (1.0, "pluralist"),
            "The author of this text believes constitutional courts should protect minority rights from majority rule": (1.0, "pluralist"),
            "The author of this text believes policy experts provide valuable technical knowledge to lawmakers": (1.0, "pluralist"),
            "The author of this text believes legislative committees should carefully review proposed laws": (1.0, "pluralist"),
            "The author of this text believes business associations and labor unions both deserve seats at the policy table": (1.0, "pluralist"),
            "The author of this text believes complex problems require nuanced solutions and careful implementation": (1.0, "pluralist"),
            "The author of this text believes democratic institutions have evolved to serve important functions": (1.0, "pluralist"),
            "The author of this text believes elected representatives should balance constituent demands with broader considerations": (1.0, "pluralist"),
            "The author of this text believes federal systems allow different regions to have different approaches": (1.0, "pluralist"),
            "The author of this text believes specialized agencies should make technical decisions based on expertise": (1.0, "pluralist"),
            "The author of this text believes incremental policy changes are more sustainable than dramatic overhauls": (1.0, "pluralist"),
            "The author of this text believes professional civil servants provide continuity across administrations": (1.0, "pluralist"),
            "The author of this text believes political opposition helps improve government policies through debate": (1.0, "pluralist"),
            "The author of this text believes coalition-building requires acknowledging different viewpoints": (1.0, "pluralist"),
            "The author of this text believes democratic decisions should balance majority preferences with minority protections": (1.0, "pluralist"),
            "The author of this text believes competing interests can find mutually beneficial solutions through negotiation": (1.0, "pluralist"),
            "The author of this text believes institutional safeguards prevent dangerous concentration of power": (1.0, "pluralist"),
        }

        populist_count = sum(1 for _, (_, direction) in self.populism_hypotheses.items() if direction == "populist")
        pluralist_count = sum(1 for _, (_, direction) in self.populism_hypotheses.items() if direction == "pluralist")
        print(f"Loaded {len(self.populism_hypotheses)} hypotheses ({populist_count} populist, {pluralist_count} pluralist)")

    def _find_entailment_index(self):
        """Auto-detect entailment index for different NLI models"""
        config = self.model.config
        if hasattr(config, 'label2id') and config.label2id: 
            for label, idx in config.label2id.items():
                if label.lower() in ['entailment', 'entail']:
                    return idx
        return 0

    def get_hypothesis_probabilities(self, text):
        """Get probabilities for all populism-pluralism hypotheses"""
        probs = []
        for hypothesis in self.populism_hypotheses.keys():
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

    def compute_combined_confidence(self, populist_probs, pluralist_probs, all_probs):
        """Simplified confidence with Top-K contradiction detection only"""
        
        # Basic confidence from variance (lower variance = higher confidence)
        populist_variance = np.var(populist_probs) if len(populist_probs) > 1 else 0
        pluralist_variance = np.var(pluralist_probs) if len(pluralist_probs) > 1 else 0
        
        populist_confidence = 1 / (1 + populist_variance * 4)
        pluralist_confidence = 1 / (1 + pluralist_variance * 4)
        base_confidence = 0.7 * min(populist_confidence, pluralist_confidence) + 0.3 * (populist_confidence + pluralist_confidence) / 2
        
        # Top-K contradiction detection (only method we use)
        k = 7
        top_populist = np.sort(populist_probs)[-k:] if len(populist_probs) >= k else populist_probs
        top_pluralist = np.sort(pluralist_probs)[-k:] if len(pluralist_probs) >= k else pluralist_probs
        
        top_populist_avg = np.mean(top_populist)
        top_pluralist_avg = np.mean(top_pluralist)
        
        # Simple contradiction detection: both top-5 averages must be > 0.25
        topk_contradiction = min(top_populist_avg, top_pluralist_avg)
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
            'top_populist_avg': top_populist_avg,
            'top_pluralist_avg': top_pluralist_avg
        }

    def score_populism_pluralism(self, text):
        """Score text and return comprehensive results"""
        probs = self.get_hypothesis_probabilities(text)

        populist_probs = []
        pluralist_probs = []
        hypothesis_results = []
        
        # Process each hypothesis
        for i, (hypothesis, (weight, direction)) in enumerate(self.populism_hypotheses.items()):
            prob = probs[i]
            
            hypothesis_results.append({
                'hypothesis': hypothesis,
                'probability': prob,
                'direction': direction
            })
            
            if direction == "populist":
                populist_probs.append(prob * weight)
            else:
                pluralist_probs.append(prob * weight)

        # Calculate averages and score
        populist_avg = np.mean(populist_probs) if populist_probs else 0
        pluralist_avg = np.mean(pluralist_probs) if pluralist_probs else 0
        
        difference = populist_avg - pluralist_avg
        final_score = 5 + (difference * 5)  # Higher scores = more populist
        final_score = np.clip(final_score, 0, 10)

        # Compute confidence
        confidence_data = self.compute_combined_confidence(
            [p/1.0 for p in populist_probs],  # Unweight for confidence calc
            [p/1.0 for p in pluralist_probs],
            probs
        )

        # Get top hypotheses from each direction
        populist_hyps = [h for h in hypothesis_results if h['direction'] == 'populist']
        pluralist_hyps = [h for h in hypothesis_results if h['direction'] == 'pluralist']
        
        top_populist = sorted(populist_hyps, key=lambda x: x['probability'], reverse=True)[:5]
        top_pluralist = sorted(pluralist_hyps, key=lambda x: x['probability'], reverse=True)[:5]

        # Interpret score (0-10 scale: 0=Strong Pluralist, 5=Moderate, 10=Strong Populist)
        if final_score < 2:
            interpretation = "Strong Pluralist"
        elif final_score < 4:
            interpretation = "Pluralist"
        elif final_score < 6:
            interpretation = "Moderate"
        elif final_score < 8:
            interpretation = "Populist"
        else:
            interpretation = "Strong Populist"

        return {
            'text': text,
            'score': final_score,
            'confidence': confidence_data['combined'],
            'contradiction_detected': confidence_data['contradiction_detected'],
            'interpretation': interpretation,
            'populist_avg': populist_avg,
            'pluralist_avg': pluralist_avg,
            'top_populist_hypotheses': top_populist,
            'top_pluralist_hypotheses': top_pluralist
        }

    def quick_score(self, text):
        """Ultra-simple interface - just returns the numerical score"""
        result = self.score_populism_pluralism(text)
        return result['score']

# ============================================================================
# INTERACTIVE ANALYSIS FUNCTIONS
# ============================================================================

def analyze_text(scorer, text):
    """Analyze a single text and display clean results"""
    result = scorer.score_populism_pluralism(text)
    
    print(f"\n{'='*80}")
    print(f"TEXT: {text}")
    print(f"{'='*80}")
    
    print(f"\n📊 RESULTS:")
    print(f"   PopulistAvg: {result['populist_avg']:.2f}")
    print(f"   PluralistAvg: {result['pluralist_avg']:.2f}")
    print(f"   Score: {result['score']:.2f}/10 (0=Strong Pluralist, 10=Strong Populist)")
    print(f"   Confidence: {result['confidence']:.3f}")
    print(f"   Contradiction: {'YES' if result['contradiction_detected'] else 'NO'}")
    print(f"   Interpretation: {result['interpretation']}")
    
    print(f"\n🔍 TOP POPULIST HYPOTHESES:")
    for i, hyp in enumerate(result['top_populist_hypotheses']):
        short_hyp = hyp['hypothesis'][:100] + "..." if len(hyp['hypothesis']) > 100 else hyp['hypothesis']
        print(f"   {i}. {hyp['probability']:.3f} - {short_hyp}")
    
    print(f"\n🔍 TOP PLURALIST HYPOTHESES:")
    for i, hyp in enumerate(result['top_pluralist_hypotheses']):
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
        result = scorer.score_populism_pluralism(text)
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
    print("INTERACTIVE POPULISM-PLURALISM SCORER")
    print(f"{'='*60}")
    print("Enter text to analyze (or 'quit' to exit)")
    print("Commands: 'batch' for multiple texts, 'help' for guidance")
    print("Scale: 0 = Strong Pluralist, 5 = Moderate, 10 = Strong Populist")
    
    while True:
        text = input("\n> ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            break
        elif text.lower() == 'help':
            print("\nCommands:")
            print("- Enter any political text to get populism-pluralism score")
            print("- 'batch' - analyze multiple predefined test texts")
            print("- 'quit' - exit the program")
            print("\nScoring:")
            print("- 0-2: Strong Pluralist (institutional respect, minority rights, compromise)")
            print("- 2-4: Pluralist (generally supportive of democratic institutions)")
            print("- 4-6: Moderate (mixed populist/pluralist elements)")
            print("- 6-8: Populist (anti-establishment, people vs. elite)")
            print("- 8-10: Strong Populist (strong anti-institutional, pure people)")
            continue
        elif text.lower() == 'batch':
            test_texts = [
                # Clear Examples
                "We must respect democratic institutions and protect minority rights.",
                "The corrupt establishment has betrayed the people for too long.",
                "Political compromise is necessary to achieve sustainable solutions.",
                "Real Americans know better than Washington elites what's good for our country.",
                "Our democracy depends on checks and balances between branches of government.",
                "The rigged system serves special interests instead of ordinary citizens.",
                "Different groups in society have legitimate competing interests that must be balanced.",
                "The voice of the people is being silenced by corrupt politicians and their cronies.",
                
                # Challenging Technocratic Populism
                "We need evidence-based policies that serve the people, not the special interests that have captured our broken institutions.",                
                # Institutional Reform Populism  
                "The system needs structural changes to restore faith in government, but we must work within democratic processes to achieve them.",
                # Elite Anti-Elite Rhetoric
                "My experience in Washington taught me that real change requires challenging the establishment from within.",
                # Inclusive Populism
                "Our movement includes everyone - urban and rural, young and old - who believes government should work for all people, not just the privileged few.",
                # Democratic Process Populism
                "Real democracy means every vote counts equally, regardless of how much money special interests spend to influence outcomes.",
                # Anti-Corruption Institutionalism
                "We need stronger ethics rules and transparency requirements to restore public trust in our democratic institutions.",                
                # Expertise vs. Elite Distinction
                "We should listen to scientists and policy experts, not political consultants and lobbyists who profit from the status quo.",                
                # Additional Edge Cases
                "Term limits would bring fresh perspectives while preserving the wisdom of experienced legislators who understand how to get things done.",
                "Grassroots organizing empowers local communities to work with elected officials on pragmatic solutions to shared challenges.",
                "Citizen panels and deliberative democracy can complement representative government by giving ordinary people more input on complex policy questions.",
                "Political parties should represent their base voters while also building coalitions that can govern effectively for everyone."
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
    scorer = PopulismPluralismScorer()
    
    # Run interactive mode
    interactive_mode(scorer)
