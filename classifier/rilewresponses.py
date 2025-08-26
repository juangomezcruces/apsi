import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings('ignore')

class LeftRightResponsesScorer:
    """
    Left-Right economic classification using responses-based scoring approach
    Adapted for integration with existing political classification system
    Output: 0-10 scale (mapped from original 0-4 scale)
    """
    
    def __init__(self, model_name="mlburnham/Political_DEBATE_large_v1.0"):
        print(f"Loading Left-Right Economic NLI model: {model_name}...")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.entailment_idx = self._find_entailment_index()
        
        # Topic determination question
        self.topic_question = (
            "Does this text discuss economic policy, government intervention, or public services? "
            "This includes topics like healthcare, education, housing, transport, taxation, natural resources "
            "privatization, welfare, regulation, minimum wage, wealth redistribution, "
            "public vs. private sector roles, or economic equality."
        )

        # Core question to assess economic rhetoric
        self.core_question = "To what extent does the author of this text holds a left or right stance on economic issues?"

        # Response options with their corresponding scores (0-4 original scale)
        self.response_options = {
            0: {  # Far Left
                'description': "Far Left. Advocates for very strong government intervention, public ownership, and extensive redistribution.",
                'interpretation': "Strong Left",
                'primary': [
                    "Calls for nationalization of major industries or resources.",
                    "Supports abolishing or heavily restricting private enterprise in key sectors.",
                    "Advocates for extensive welfare programs funded by high taxation of the wealthy and corporations.",
                    "Frames economic equality as a central goal, prioritizing redistribution over market efficiency."
                ]
            },
            1: {  # Left
                'description': "Left. Supports larger government role, redistribution, and regulation of markets to promote social welfare.",
                'interpretation': "Left",
                'primary': [
                    "Advocates progressive taxation to fund public services.",
                    "Supports strong labor protections and higher minimum wages.",
                    "Favors expanding welfare programs to reduce poverty and inequality.",
                    "Promotes regulation of business to protect workers, consumers, and the environment."
                ]
            },
            2: {  # Center
                'description': "Center. Balances market freedom with government intervention for stability and fairness.",
                'interpretation': "Center",
                'primary': [
                    "Supports a mixed economy with both private enterprise and public services.",
                    "Favors moderate taxes to fund essential services without discouraging investment.",
                    "Promotes targeted regulation to address market failures while preserving competition.",
                    "Seeks compromise between efficiency and equality."
                ]
            },
            3: {  # Right
                'description': "Right. Prefers reduced government role, lower taxes, and more market-driven solutions.",
                'interpretation': "Right",
                'primary': [
                    "Advocates lowering taxes to stimulate investment and growth.",
                    "Supports privatization to improve efficiency.",
                    "Favors deregulation of markets to encourage entrepreneurship.",
                    "Promotes limiting welfare to avoid dependency."
                ]
            },
            4: {  # Far Right
                'description': "Far Right. Strongly favors minimal government intervention, extensive privatization, and unfettered markets.",
                'interpretation': "Strong Right",
                'primary': [
                    "Calls for eliminating most regulations and subsidies.",
                    "Advocates complete privatization of public services.",
                    "Supports minimal taxation and minimal welfare state.",
                    "Frames free markets as the solution to all economic problems."
                ]
            }
        }

        # Threshold for topic determination
        self.topic_threshold = 0.7

        print("Left-Right responses scorer initialized")

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

    def is_about_economic_policy(self, text):
        """Determine if text is about economic policy"""
        prob = self._get_entailment_prob(text, self.topic_question)
        return prob >= self.topic_threshold, prob

    def get_response_probabilities(self, text):
        """Get probabilities for each response option"""
        probs = []
        for score in sorted(self.response_options.keys()):
            response_text = self.response_options[score]['description']
            # Also include primary characteristics for better matching
            if 'primary' in self.response_options[score]:
                primary_text = " ".join(self.response_options[score]['primary'][:2])  # Use first 2 characteristics
                response_text += " " + primary_text
            
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
        0 (Strong Left) -> 0-1
        1 (Left) -> 2-3  
        2 (Center) -> 4-6
        3 (Right) -> 7-8
        4 (Strong Right) -> 9-10
        """
        scale_mapping = {
            0: (0, 1),   # Strong Left
            1: (2, 3),   # Left
            2: (4, 6),   # Center  
            3: (7, 8),   # Right
            4: (9, 10)   # Strong Right
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
            return "Strong Left"
        elif score_10 < 4:
            return "Left"
        elif score_10 < 7:
            return "Center"
        elif score_10 < 9:
            return "Right"
        else:
            return "Strong Right"

    def score_left_right(self, text):
        """
        Main scoring method - standardized interface for integration
        Returns format expected by views.py: {'score', 'confidence', 'interpretation'}
        """
        # Step 1: Determine if text is about economic policy
        is_about_topic, topic_prob = self.is_about_economic_policy(text)
        
        if not is_about_topic:
            return {
                'score': 5.0,  # Neutral/moderate score when not relevant
                'confidence': 0.5,  # Low confidence for irrelevant text
                'interpretation': "Not about economic policy",
                'is_relevant': False,
                'topic_probability': topic_prob
            }
        
        # Step 2: Score the economic position
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
    scorer = LeftRightResponsesScorer()
    
    test_texts = [
        "We need to increase taxes on the wealthy to fund universal healthcare.",
        "Private companies are more efficient than government - we should privatize more services.",
        "The weather forecast shows rain tomorrow.",
        "We should nationalize the energy sector to combat climate change.",
        "Lower corporate taxes will stimulate economic growth and job creation."
    ]
    
    print("\n" + "="*80)
    print("TESTING LEFT-RIGHT RESPONSES SCORER")
    print("="*80)
    
    for text in test_texts:
        result = scorer.score_left_right(text)
        print(f"\nText: {text}")
        print(f"Score (0-10): {result['score']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Interpretation: {result['interpretation']}")
        print(f"Relevant: {result['is_relevant']}")
        if 'original_score_0_4' in result:
            print(f"Original (0-4): {result['original_score_0_4']}")

if __name__ == "__main__":
    test_scorer()