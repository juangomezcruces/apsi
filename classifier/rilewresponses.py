import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings('ignore')

class LeftRightResponsesScorer:
    """
    Left-Right economic classification using responses-based scoring approach
    Adapted for 7-point economic left-right scale (0-6)
    Output: 0-10 scale (mapped from original 0-6 scale)

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
        self.core_question = "Please locate the party in terms of its overall ideological stance on economic issues."

        # Response options with their corresponding scores (0-6 scale)
        self.response_options = {
            0: {  # Far-left
                'description': "Far-left. Advocates for very strong government intervention, public ownership, and extensive redistribution.",
                'interpretation': "Far-left",
                'primary': [
                    "Calls nationalization of industries and resources.",
                    "Supports abolishing private enterprise in key sectors.",
                    "Advocates for maximum wealth redistribution through very high taxation.",
                    "Frames economic equality as the absolute priority over market efficiency."
                ]
            },
            1: {  # Left
                'description': "Left. Wants government to play an active role in the economy with higher taxes, more regulation and generous welfare state.",
                'interpretation': "Left",
                'primary': [
                    "Advocates progressive taxation to fund extensive public services.",
                    "Supports strong labor protections and higher minimum wages.",
                    "Favors expanding welfare programs and government spending.",
                    "Promotes regulation of business to protect workers and environment."
                ]
            },
            2: {  # Center-left
                'description': "Center-left. Supports moderate government intervention with some redistribution and regulation.",
                'interpretation': "Center-left",
                'primary': [
                    "Favors balanced approach with government playing significant but not extreme role.",
                    "Supports targeted taxes for specific social programs.",
                    "Promotes regulated markets with social safety nets.",
                    "Seeks compromise between market efficiency and social equality."
                ]
            },
            3: {  # Center
                'description': "Center. Balances market freedom with government intervention for stability and fairness.",
                'interpretation': "Center",
                'primary': [
                    "Supports a mixed economy with both private enterprise and public services.",
                    "Favors moderate taxes to fund essential services without discouraging investment.",
                    "Promotes targeted regulation to address market failures while preserving competition.",
                    "Seeks middle ground between left and right approaches."
                ]
            },
            4: {  # Center-right
                'description': "Center-right. Prefers reduced but still significant government role with market-oriented solutions.",
                'interpretation': "Center-right",
                'primary': [
                    "Advocates for moderate tax reductions to stimulate growth.",
                    "Supports selective privatization and market-based reforms.",
                    "Favors streamlined regulation to encourage entrepreneurship.",
                    "Promotes limited but efficient welfare state."
                ]
            },
            5: {  # Right
                'description': "Right. Emphasizes reduced economic role for government with lower taxes, less regulation, and leaner welfare state.",
                'interpretation': "Right",
                'primary': [
                    "Advocates significantly lowering taxes to stimulate investment.",
                    "Supports extensive privatization to improve efficiency.",
                    "Favors substantial deregulation of markets.",
                    "Promotes limiting welfare spending to avoid dependency."
                ]
            },
            6: {  # Far-right
                'description': "Far-right. Strongly favors minimal government intervention, extensive privatization, and unfettered markets.",
                'interpretation': "Far-right",
                'primary': [
                    "Calls for eliminating most regulations, taxes, and subsidies.",
                    "Advocates complete privatization of all public services.",
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

    def _map_to_10_scale(self, score_0_6):
        """
        Map 0-6 scale to 0-10 scale for consistency with existing system
        0 (Far-left) -> 0
        1 (Left) -> 1.43
        2 (Center-left) -> 2.86
        3 (Center) -> 4.29
        4 (Center-right) -> 5.71
        5 (Right) -> 7.14
        6 (Far-right) -> 8.57
        """
        # Linear mapping from 0-6 to 0-10
        if score_0_6 <= 0:
            return 0.0
        elif score_0_6 >= 6:
            return 10.0
        
        return round(score_0_6 * (10/6), 2)

    def _get_interpretation_from_10_scale(self, score_10):
        """Get interpretation for 10-scale score"""
        if score_10 < 1.43:
            return "Far-left"
        elif score_10 < 2.86:
            return "Left"
        elif score_10 < 4.29:
            return "Center-left"
        elif score_10 < 5.71:
            return "Center"
        elif score_10 < 7.14:
            return "Center-right"
        elif score_10 < 8.57:
            return "Right"
        else:
            return "Far-right"

    def score_left_right(self, text):
        """
        Main scoring method - standardized interface for integration
        Returns format expected by views.py: {'score', 'confidence', 'interpretation'}
        """
        # Step 1: Determine if text is about economic policy
        is_about_topic, topic_prob = self.is_about_economic_policy(text)
        
        if not is_about_topic:
            return {
                'score': 'NA',  # Neutral/moderate score when not relevant
                'confidence': 0.5,  # Low confidence for irrelevant text
                'interpretation': "Not about economic policy",
                'is_relevant': False,
                'topic_probability': topic_prob
            }
        
        # Step 2: Score the economic position
        probs = self.get_response_probabilities(text)
        normalized_probs = probs / np.sum(probs)
        
        # Calculate continuous score on 0-6 scale
        scores = np.array(list(self.response_options.keys()))
        continuous_score_0_6 = np.sum(scores * normalized_probs)
        
        # Map to 0-10 scale to match existing system
        score_10 = self._map_to_10_scale(continuous_score_0_6)
        
        # Get confidence and interpretation
        confidence = self.compute_confidence(probs)
        interpretation = self._get_interpretation_from_10_scale(score_10)
        
        return {
            'score': score_10,
            'confidence': confidence,
            'interpretation': interpretation,
            'is_relevant': True,
            'topic_probability': topic_prob,
            'original_score_0_6': round(continuous_score_0_6, 2)  # Keep for debugging
        }

# Test function
def test_scorer():
    """Test the adapted scorer"""
    scorer = LeftRightResponsesScorer()
    
    test_texts = [
#Far Right (strongly reduced government role, deregulation, privatization)
"The government should sell off all remaining public utilities to unleash private sector growth.",
"Taxes on corporations must be slashed to the lowest in the world to attract investment.",
"End all welfare programs—they create dependency and waste taxpayer money.",
"Government interference in the economy stifles innovation; privatize education and healthcare.",
"The free market alone should determine wages, not unions or regulations.",
"Eliminate all subsidies and let failing businesses close naturally.",
"We must repeal environmental regulations that hurt business competitiveness.",
"Abolish the minimum wage; it’s a barrier to employment.",
"Shrink government spending to the bare minimum required for defense.",
"Public ownership is socialism in disguise; the market must rule.",
#Right (reduced but not eliminated government role, pro-market reforms)
"Lower income taxes to boost household spending and investment.",
"Encourage private healthcare providers to compete with public services.",
"Privatize some transport services to improve efficiency.",
"Reduce corporate tax rates to stimulate job creation.",
"Cut red tape that burdens small businesses.",
"Limit welfare benefits to encourage people to re-enter the workforce.",
"Encourage private pension plans over state-run systems.",
"Reduce government spending to balance the budget.",
"Phase out subsidies that distort the free market.",
"Promote public-private partnerships instead of direct government management.",
#Center (balanced government and market roles)
"We need both a healthy private sector and strong public services.",
"Support free enterprise, but with safeguards for workers and consumers.",
"Invest in public infrastructure while encouraging private investment.",
"Maintain moderate taxes to fund essential services without discouraging growth.",
"Combine market-based solutions with targeted regulation.",
"Encourage entrepreneurship while ensuring fair labor practices.",
"Public healthcare and private options should coexist.",
"Balance environmental protection with economic competitiveness.",
"Welfare programs should help people back into work.",
"Aim for fiscal responsibility without cutting vital services.",
#Left (larger government role, redistribution, regulation)
"Increase taxes on the wealthy to fund universal healthcare.",
"Raise the minimum wage to ensure all workers earn a living wage.",
"Expand public housing to tackle homelessness.",
"Strengthen environmental regulations to protect communities.",
"Boost social welfare programs to reduce poverty.",
"Provide free higher education funded by progressive taxation.",
"Increase government spending on renewable energy projects.",
"Regulate large corporations to prevent monopolistic practices.",
"Expand public transport networks as a government priority.",
"Introduce wealth taxes to fund social programs.",
#Far Left (very strong government role, public ownership, heavy redistribution)
"Nationalize all major industries to serve the public good.",
"Abolish private healthcare; all care should be state-run and free.",
"Set maximum income limits to reduce inequality.",
"Make higher education and housing completely free for all citizens.",
"Heavily tax large corporations to redistribute wealth.",
"Eliminate private ownership of natural resources.",
"Guarantee jobs for all through state employment programs.",
"Convert all private utilities into public enterprises.",
"Ban for-profit healthcare and education entirely.",
"Implement universal basic income funded by corporate and wealth taxes."
    ]
    
    print("\n" + "="*80)
    print("TESTING LEFT-RIGHT RESPONSES SCORER (7-POINT SCALE)")
    print("="*80)
    
    for text in test_texts:
        result = scorer.score_left_right(text)
        print(f"\nText: {text}")
        print(f"Score (0-10): {result['score']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Interpretation: {result['interpretation']}")
        print(f"Relevant: {result['is_relevant']}")
        if 'original_score_0_6' in result:
            print(f"Original (0-6): {result['original_score_0_6']}")

if __name__ == "__main__":
    test_scorer()