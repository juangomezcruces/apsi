import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings('ignore')

class PopulismPluralismResponsesScorer:
    """
    Populism-Pluralism classification using responses-based scoring approach
    Adapted for integration with existing political classification system
    Output: 0-10 scale (mapped from 0-4 scale for consistency)
    """
    
    def __init__(self, model_name="mlburnham/Political_DEBATE_large_v1.0"):
        print(f"Loading Populism-Pluralism NLI model: {model_name}...")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.entailment_idx = self._find_entailment_index()

        # Topic determination question
        self.topic_question = (
            "Does this text makes reference to the people or societal unity/diversity? "
            "This includes rhetoric emphasizing homogeneity of the people (populist rhetoric) or diversity/compromise (pluralist rhetoric), "
            "Populist rhetoric tends to privilege the will of the majority, while pluralist rhetoric emphasizes equal protection of all citizens, including minorities."
        )

        # Core question to assess rhetoric
        self.core_question = "To what extent does the author of this text use populist or pluralist rhetoric?"

        # Response options with their corresponding scores (0-4 scale for consistency)
        self.response_options = {
            0: {  # Strong Pluralist
                'description': "Strong Pluralist. Explicitly emphasizes diversity, minority rights, and institutional protections against majority tyranny.",
                'interpretation': "Strong Pluralist",
                'primary': [
                    "Explicitly acknowledges multiple legitimate perspectives, identities, or interests within society.",
                    "Frames political legitimacy as coming from inclusive representation of all groups, not a single will.",
                    "Strongly emphasizes the need to protect minority rights, even against majority preferences.",
                    "Describes diversity as a fundamental democratic strength that must be preserved."
                ],
                'secondary': [
                    "Presents unity as cooperation among diverse groups, without erasing disagreements.",
                    "Emphasizes institutions, rules, or processes that guarantee equal participation for all voices.",
                    "Rejects the idea that any single group represents 'the true people'.",
                    "Advocates for checks and balances to prevent majority domination."
                ]
            },
            1: {  # Pluralist
                'description': "Pluralist. Recognizes diversity and supports inclusive representation, with some emphasis on shared values.",
                'interpretation': "Pluralist",
                'primary': [
                    "Acknowledges societal diversity and supports inclusive political representation.",
                    "Frames legitimate governance as requiring input from multiple groups and perspectives.",
                    "Supports minority rights and institutional protections for all citizens."
                ],
                'secondary': [
                    "Recognizes the value of political compromise and negotiation.",
                    "Depicts political differences as manageable through democratic institutions.",
                    "Avoids framing opponents as illegitimate or anti-democratic.",
                    "Balances majority will with minority protection."
                ]
            },
            2: {  # Moderate/Neutral
                'description': "Moderate. Balances people-centered rhetoric with recognition of diversity; neither strongly populist nor pluralist.",
                'interpretation': "Moderate",
                'primary': [
                    "Uses both unity-focused and diversity-acknowledging language.",
                    "References 'the people' but also acknowledges different groups and interests.",
                    "Supports both majority decision-making and minority protection, without strong emphasis on either."
                ],
                'secondary': [
                    "Frames political legitimacy as coming from both popular will and institutional processes.",
                    "Mentions both shared values and diverse perspectives.",
                    "Takes a balanced approach to questions of majority rule vs. minority rights.",
                    "Avoids strong populist or pluralist framing devices."
                ]
            },
            3: {  # Populist
                'description': "Populist. Emphasizes unified people with common interests; criticizes elites but without strong moral condemnation.",
                'interpretation': "Populist",
                'primary': [
                    "Portrays 'the people' as having a largely common will or shared interests.",
                    "Frames ordinary citizens as the primary source of political legitimacy.",
                    "Criticizes political elites, institutions, or opponents as out of touch with popular will.",
                    "Unity and shared identity are emphasized over diversity and difference."
                ],
                'secondary': [
                    "Emphasizes shared values, traditions, or national identity as the foundation for political action.",
                    "Depicts political conflict as between 'the people' and unresponsive institutions or elites.",
                    "Mentions diversity rarely or frames it as secondary to unity.",
                    "Suggests that the people's voice should have primacy in political decisions."
                ]
            },
            4: {  # Strong Populist
                'description': "Strong Populist. People framed as morally pure and homogeneous, opposed to corrupt, illegitimate elites or dangerous outsiders.",
                'interpretation': "Strong Populist",
                'primary': [
                    "Portrays 'the people' as morally pure, homogeneous, and the only rightful political authority.",
                    "Claims exclusive representation of the 'true will of the people', rejecting other voices as illegitimate.",
                    "Frames politics as a moral struggle between virtuous people and corrupt elites or dangerous outsiders.",
                    "Describes political opponents as enemies, traitors, or threats to the people's well-being."
                ],
                'secondary': [
                    "Emphasizes that the people's voice should override institutional constraints, checks and balances, or minority rights.",
                    "Positions outsiders, foreigners, or minority groups as threats to the unity or moral integrity of the people.",
                    "Conflict is moralized as a struggle between good and evil, not just policy disagreements.",
                    "Rejects the legitimacy of pluralist institutions or processes that might constrain popular will."
                ]
            }
        }

        # Threshold for topic determination
        self.topic_threshold = 0.7

        print("Populism-Pluralism responses scorer initialized (0-4 scale)")

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

    def is_about_people_unity(self, text):
        """Determine if text is about people/societal unity"""
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
        NOW CONSISTENT: 0-4 → 0-10 (same as Liberal-Illiberal and Left-Right)
        0 (Strong Pluralist) -> 0-1
        1 (Pluralist) -> 2-3  
        2 (Moderate) -> 4-6
        3 (Populist) -> 7-8
        4 (Strong Populist) -> 9-10
        """
        scale_mapping = {
            0: (0, 1),   # Strong Pluralist
            1: (2, 3),   # Pluralist
            2: (4, 6),   # Moderate  
            3: (7, 8),   # Populist
            4: (9, 10)   # Strong Populist
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
            return "Strong Pluralist"
        elif score_10 < 4:
            return "Pluralist"
        elif score_10 < 7:
            return "Moderate"
        elif score_10 < 9:
            return "Populist"
        else:
            return "Strong Populist"

    def score_populism_pluralism(self, text):
        """
        Main scoring method - standardized interface for integration
        Returns format expected by views.py: {'score', 'confidence', 'interpretation'}
        """
        # Step 1: Determine if text is about people/societal unity
        is_about_topic, topic_prob = self.is_about_people_unity(text)
        
        if not is_about_topic:
            return {
                'score': 5.0,  # Neutral/moderate score when not relevant
                'confidence': 0.5,  # Low confidence for irrelevant text
                'interpretation': "Not about people or unity",
                'is_relevant': False,
                'topic_probability': topic_prob
            }
        
        # Step 2: Score the populist/pluralist rhetoric
        probs = self.get_response_probabilities(text)
        normalized_probs = probs / np.sum(probs)
        
        # Calculate continuous score on 0-4 scale (NOW CONSISTENT)
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
    scorer = PopulismPluralismResponsesScorer()
    
    test_texts = [
        "We must protect the rights of all minorities and ensure diverse voices are heard in democracy.",
        "The corrupt establishment has betrayed the true will of the American people!",
        "Today's weather report shows partly cloudy skies.",
        "Real patriots know what's best for this country, not Washington insiders and foreign influences.",
        "Democratic institutions must balance majority rule with constitutional protections for all citizens.",
        "The people have spoken clearly, but special interests and courts keep blocking our agenda."
    ]
    
    print("\n" + "="*80)
    print("TESTING POPULISM-PLURALISM RESPONSES SCORER (0-4 SCALE)")
    print("="*80)
    
    for text in test_texts:
        result = scorer.score_populism_pluralism(text)
        print(f"\nText: {text}")
        print(f"Score (0-10): {result['score']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Interpretation: {result['interpretation']}")
        print(f"Relevant: {result['is_relevant']}")
        if 'original_score_0_4' in result:
            print(f"Original (0-4): {result['original_score_0_4']}")

if __name__ == "__main__":
    test_scorer()