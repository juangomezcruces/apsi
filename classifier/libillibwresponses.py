import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .shared_model_cache import SharedModelCache
import warnings
warnings.filterwarnings('ignore')

class LiberalIlliberalResponsesScorer:
    """
    Liberal-Illiberal classification using Democratic Commitment scoring approach
    Adapted for integration with existing political classification system
    Output: 0-10 scale (mapped from original 0-4 scale)
    """
    
    def __init__(self, model_name="mlburnham/Political_DEBATE_large_v1.0"):
        cache = SharedModelCache()
        self.model, self.tokenizer = cache.get_model_and_tokenizer(model_name)
        self.entailment_idx = self._find_entailment_index()

        # Topic determination question
        self.topic_question = (
            "Does this text discuss political actors' or parties' commitment to democratic principles? "
            "Political actors and parties show no commitment to such principles if they openly support "
            "an autocratic form of government without elections or freedom of speech, assembly and "
            "association (e.g. theocracy; single-party rule; revolutionary regime). Political actors "
            "and parties show a full commitment to key democratic principles if they unambiguously "
            "support freedom of speech, media, assembly and association and pledge to accept defeat "
            "in free and fair elections."
        )
        
        # Core question to assess democratic commitment
        self.core_question = "To what extent was the author of this text clearly committed to free and fair elections with multiple parties, freedom of speech, media, assembly and association?"
        
        # Response options with their corresponding scores (0-4 original scale)
        self.response_options = {
            0: {
                'description': "Not at all committed. The political actor or party is likely not at all committed to free and fair, multi-party elections, freedom of speech, media, assembly and association.",
                'criteria': "Openly supports autocratic government without elections or freedoms (e.g., theocracy, single-party rule, revolutionary regime)",
                'interpretation': "Strong Illiberal"
            },
            1: {
                'description': "Not committed. The political actor or party is likely not committed to free and fair, multi-party elections, freedom of speech, media, assembly and association.",
                'criteria': "Generally opposes democratic principles but may not openly advocate autocracy",
                'interpretation': "Illiberal"
            },
            2: {
                'description': "Weakly committed. The political actor or party is likely weakly committed to free and fair, multi-party elections, freedom of speech, media, assembly and association.",
                'criteria': "Shows mixed signals - some support for democratic principles but with significant reservations",
                'interpretation': "Moderate"
            },
            3: {
                'description': "Committed. The political actor or party is likely committed to free and fair, multi-party elections, freedom of speech, media, assembly and association.",
                'criteria': "Generally supports democratic principles but may have some limitations in practice",
                'interpretation': "Liberal"
            },
            4: {
                'description': "Fully committed. The political actor or party is likely fully committed to free and fair, multi-party elections, freedom of speech, media, assembly and association.",
                'criteria': "Unambiguously supports all freedoms and pledges to accept defeat in free and fair elections",
                'interpretation': "Strong Liberal"
            }
        }

        # Threshold for topic determination
        self.topic_threshold = 0.7  # Minimum probability to consider text relevant

        print("Liberal-Illiberal responses scorer initialized")

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

    def is_about_democratic_commitment(self, text):
        """Determine if text is about political liberalism"""
        prob = self._get_entailment_prob(text, self.topic_question)
        return prob >= self.topic_threshold, prob

    def get_response_probabilities(self, text):
        """Get probabilities for each response option"""
        probs = []
        for score in sorted(self.response_options.keys()):
            response_text = f"{self.response_options[score]['description']} {self.response_options[score]['criteria']}"
            prob = self._get_entailment_prob(text, response_text)
            probs.append(prob)
        return np.array(probs)

    def compute_confidence(self, probs):
        """Compute confidence based on probability distribution"""
        normalized_probs = probs / np.sum(probs)
        entropy = -np.sum(normalized_probs * np.log(normalized_probs + 1e-10))
        max_prob = np.max(normalized_probs)
        confidence = max_prob * (1 - entropy/np.log(len(probs)))
        return confidence

    def _map_to_10_scale(self, score_0_4):
        """
        Map 0-4 scale to 0-10 scale for consistency with existing system
        0 (Strong Illiberal) -> 0-1
        1 (Illiberal) -> 2-3  
        2 (Moderate) -> 4-6
        3 (Liberal) -> 7-8
        4 (Strong Liberal) -> 9-10
        """
        scale_mapping = {
            0: (0, 1),   # Strong Illiberal
            1: (2, 3),   # Illiberal
            2: (4, 6),   # Moderate  
            3: (7, 8),   # Liberal
            4: (9, 10)   # Strong Liberal
        }
        
        # Linear interpolation within ranges
        if score_0_4 <= 0:
            return 0.5
        elif score_0_4 >= 4:
            return 9.5
        
        # Find the two adjacent integer scores
        lower_int = int(score_0_4)
        upper_int = min(lower_int + 1, 4)
        
        # Get the ranges for interpolation
        lower_range = scale_mapping[lower_int]
        upper_range = scale_mapping[upper_int]
        
        # Linear interpolation
        fraction = score_0_4 - lower_int
        lower_mapped = (lower_range[0] + lower_range[1]) / 2
        upper_mapped = (upper_range[0] + upper_range[1]) / 2
        
        mapped_score = lower_mapped + fraction * (upper_mapped - lower_mapped)
        return round(mapped_score, 2)

    def _get_interpretation_from_10_scale(self, score_10):
        """Get interpretation for 10-scale score"""
        if score_10 < 2:
            return "Strong Illiberal"
        elif score_10 < 4:
            return "Illiberal"
        elif score_10 < 7:
            return "Moderate"
        elif score_10 < 9:
            return "Liberal"
        else:
            return "Strong Liberal"

    def score_liberal_illiberal(self, text):
        """
        Main scoring method - standardized interface for integration
        Returns format expected by views.py: {'score', 'confidence', 'interpretation'}
        """
        # Step 1: Determine if text is about democratic commitment
        is_about_topic, topic_prob = self.is_about_democratic_commitment(text)
        
        if not is_about_topic:
            return {
                'score': 5.0,  # Neutral/moderate score when not relevant
                'confidence': 0.5,  # Low confidence for irrelevant text
                'interpretation': "Not about democratic principles",
                'is_relevant': False,
                'topic_probability': topic_prob
            }
        
        # Step 2: Score the commitment level
        probs = self.get_response_probabilities(text)
        normalized_probs = probs / np.sum(probs)
        
        # Calculate continuous score on 0-4 scale
        scores = np.array(list(self.response_options.keys()))
        continuous_score_0_4 = np.sum(scores * normalized_probs)
        
        # Map to 0-10 scale to match existing system
        score_10 = self._map_to_10_scale(continuous_score_0_4)
        
        # Get confidence and interpretation
        confidence = self.compute_confidence(probs)
        interpretation = self._get_interpretation_from_10_scale(score_10)
        
        return {
            'score': score_10,
            'confidence': confidence,
            'interpretation': interpretation,
            'is_relevant': True,
            'topic_probability': topic_prob,
            'original_score_0_4': round(continuous_score_0_4, 2)  # Keep for debugging
        }

# Test function
def test_scorer():
    """Test the adapted scorer"""
    scorer = LiberalIlliberalResponsesScorer()
    
    test_texts = [
        "We must defend democratic institutions and free speech at all costs.",
        "Elections should be suspended during national emergencies.",
        "The weather is nice today.",
        "Opposition parties undermine national unity and should be restricted."
    ]
    
    print("\n" + "="*80)
    print("TESTING LIBERAL-ILLIBERAL RESPONSES SCORER")
    print("="*80)
    
    for text in test_texts:
        result = scorer.score_liberal_illiberal(text)
        print(f"\nText: {text}")
        print(f"Score (0-10): {result['score']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Interpretation: {result['interpretation']}")
        print(f"Relevant: {result['is_relevant']}")
        if 'original_score_0_4' in result:
            print(f"Original (0-4): {result['original_score_0_4']}")

if __name__ == "__main__":
    test_scorer()