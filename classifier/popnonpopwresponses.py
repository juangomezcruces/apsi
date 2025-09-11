import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .shared_model_cache import SharedModelCache
import warnings
warnings.filterwarnings('ignore')

class PopulismPluralismResponsesScorer:
    """
    Populism-Pluralism classification using Norris (2020) framework
    Scale: 0-10 interval from "Strongly favors pluralist rhetoric" to "Strongly favors populist rhetoric"
    """
    
    def __init__(self, model_name="mlburnham/Political_DEBATE_large_v1.0"):
        print(f"Loading Populism-Pluralism NLI model: {model_name}...")
        cache = SharedModelCache()
        self.model, self.tokenizer = cache.get_model_and_tokenizer(model_name)
        self.entailment_idx = self._find_entailment_index()

        # Topic determination question (Norris framework)
        self.topic_question = (
            "Does this text discuss political rhetoric, governance approaches, or institutional legitimacy? "
            "This includes references to populist rhetoric (challenging institutions, emphasizing popular will) "
            "or pluralist rhetoric (supporting checks and balances, minority rights, compromise)."
        )

        # Core question based on rile (2020) framework
        self.core_question = "Please locate the text in terms of its use of populist or pluralist rhetoric."
        
        # Expanded response options with more nuanced positions
        self.response_options = {
        10: {  # Populist
            'description': "Populist. Challenges the legitimacy of elites and institutions. Emphasizes a unified 'will of the people'.",
            'interpretation': "Populist",
            },
        5: {  # Center
             'description': "Mixed. Balances criticism of elites with support for institutional processes. Acknowledges both popular will and the need for compromise.",
            'interpretation': "Mixed",
            },
        0: {  # Pluralist
             'description': "Pluralist. Supports institutional checks and balances, minority rights, and bargaining among diverse groups.",
            'interpretation': "Pluralist",
            }
        }
        
                # Threshold for topic determination
        self.topic_threshold = 0.5

        print("Norris (2020) Populism-Pluralism scorer initialized (0-10 scale)")

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
            max_length=512,
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            prob = torch.softmax(outputs.logits, dim=-1)[0, self.entailment_idx].item()
        return prob

    def is_about_political_rhetoric(self, text):
        """Determine if text is about relevant topics"""
        prob = self._get_entailment_prob(text, self.topic_question)
        return prob >= self.topic_threshold, prob



    def get_response_probabilities(self, text):
        """Get probabilities for each response option"""
        probs = {}
        for score in self.response_options.keys():
            response_text = self.response_options[score]['description']
            # Include primary characteristics for better matching
            if 'primary' in self.response_options[score]:
                primary_text = " ".join(self.response_options[score]['primary'][:2])
                response_text += " " + primary_text
            
            prob = self._get_entailment_prob(text, response_text)
            probs[score] = prob + 0.001 

        return probs


    def compute_confidence(self, probs_dict):
        """Compute confidence with better normalization"""
        probs = np.array(list(probs_dict.values()))
        normalized_probs = probs / np.sum(probs)
        max_prob = np.max(normalized_probs)
        entropy = -np.sum(normalized_probs * np.log(normalized_probs + 1e-10))
        normalized_entropy = entropy / np.log(len(probs))        
        confidence = max_prob * (1 - normalized_entropy)
        return min(1.0, max(0.1, confidence))  



    def _calculate_norris_score(self, probs_dict):
        """
      Calculate Norris (2020) 0-10 score using weighted average of the three anchor points (0, 5, 10)"""        
        scores = np.array(list(probs_dict.keys()))
        probs = np.array(list(probs_dict.values()))
        # Apply temperature scaling to soften probabilities
        temperature = 1.7  # Higher temperature = softer distribution
        scaled_probs = np.exp(np.log(probs + 1e-10) / temperature)
        normalized_probs = scaled_probs / np.sum(scaled_probs)
        
        weighted_score = np.sum(scores * normalized_probs)
        final_score = max(0, min(10, weighted_score))
        
        return round(final_score, 2)      
        

    def _get_interpretation_from_score(self, score):
        if score < 2:
            return "Very Pluralist"
        elif score < 4:
            return "Somewhat Pluralist"
        elif score < 6:
            return "Mixed"
        elif score < 8:
            return "Somewhat Populist"
        else:
            return "Very Populist"


    def score_populism_pluralism(self, text):
        is_about_topic, topic_prob = self.is_about_political_rhetoric(text)
        
        if not is_about_topic:
            return {
                'score': 'NA',  # Middle of scale when not relevant
                'confidence': 0.3,  # Low confidence for irrelevant text
                'interpretation': "Not about political rhetoric or institutional legitimacy",
                'is_relevant': False,
                'topic_probability': topic_prob,
                'framework': "Norris (2020)"
            }


        # Step 2: Score using Norris framework
        probs_dict = self.get_response_probabilities(text)
        score = self._calculate_norris_score(probs_dict)
        confidence = self.compute_confidence(probs_dict)
        interpretation = self._get_interpretation_from_score(score)
        
        return {
            'score': score,
            'confidence': round(confidence, 3),
            'interpretation': interpretation,
            'is_relevant': True,
            'topic_probability': topic_prob,
            'framework': "Norris (2020)",
            'category_probs': {k: round(v, 3) for k, v in probs_dict.items()}
        }






# Test function with Norris-specific examples
def test_norris_scorer():
    """Test the Norris (2020) framework scorer"""
    scorer = PopulismPluralismResponsesScorer()
    
    # Test texts aligned with Norris framework
    test_texts = [
        # Strong Pluralist examples (0-2.5)
        "We must uphold our constitutional system of checks and balances to protect minority rights from majority tyranny",
        "Democratic governance requires compromise and bargaining between diverse interests, not just majority rule",
        "Institutional constraints on executive power are essential for preserving our democratic values",
        "Protecting minority rights is fundamental to our democratic system, even when it limits majority preferences",
        
        # Pluralist examples (2.5-5)
        "While we respect popular sovereignty, our institutions must also safeguard against the tyranny of the majority",
        "Effective governance requires balancing the will of the people with necessary institutional constraints",
        "We should reform our political system to be more responsive while maintaining essential democratic safeguards",
        "Both popular input and institutional expertise are valuable for sound democratic decision-making",
        
        # Populist-Leaning examples (5-7.5)
        "The political establishment has lost touch with the real needs and desires of ordinary citizens",
        "I want to get some chocolate", #irrelevant
        "We need to break through bureaucratic obstacles that prevent the people's will from being implemented",
        "Too much compromise and bargaining has led to weak leadership and unresponsive government",
        "The voices of common people should carry more weight than expert opinions in political decisions",
        
        # Strong Populist examples (7.5-10)
        "The entire political system is corrupt and needs to be completely overhauled to reflect the true will of the people",
        "We alone represent the authentic voice of the people against the corrupt elite establishment",
        "Institutional constraints are nothing but obstacles created by elites to block the people's legitimate demands",
        "The will of the people must prevail over any and all institutional barriers or minority objections",
        "A hopeful people that has said enough; enough of the same old things; enough of the traitors of the homeland"
    ]

    
    print("\n" + "="*80)
    print("TESTING rile (2020) POPULISM-PLURALISM SCORER")
    print("0 = Strongly pluralist, 10 = Strongly populist")
    print("="*80)
    
    for i, text in enumerate(test_texts):
        result = scorer.score_populism_pluralism(text)
        
        print(f"\n{i+1}. {text}")
        print(f"   Score: {result['score']}/10")
        print(f"   Interpretation: {result['interpretation']}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Relevant: {result['is_relevant']}")
        
        if result['is_relevant']:
            print(f"   Category probabilities: {result['category_probs']}")


    for text in test_texts:
        result = scorer.score_populism_pluralism(text)
        print(f"\nText: {text}")
        print(f"Score (0-10): {result['score']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Interpretation: {result['interpretation']}")
        print(f"Relevant: {result['is_relevant']}")
        if 'original_score_0_6' in result:
            print(f"Original (0-6): {result['original_score_0_6']}")    
    results = []
    for i, text in enumerate(test_texts, 1):
        result = scorer.score_populism_pluralism(text)
        results.append({
            "id": i,
            "text": text,
            "score": result['score'],
            "confidence": result['confidence'],
            "interpretation": result['interpretation'],
            "is_relevant": result['is_relevant'],
            "topic_probability": result['topic_probability'],
            "original_score_0_6": result.get('original_score_0_6', None)
        })
    
    
    df = pd.DataFrame(results)
    if save_csv:
        df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"✅ Results saved to {csv_path}")


#save_csv = True   # or False
#csv_path = "results.csv"

if __name__ == "__main__":
    test_norris_scorer()


