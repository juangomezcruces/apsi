import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings('ignore')

class PopulismPluralismResponsesScorer:
    """
    Populism-Pluralism classification using Norris (2020) framework
    Scale: 0-10 interval from "Strongly favors pluralist rhetoric" to "Strongly favors populist rhetoric"
    """
    
    def __init__(self, model_name="mlburnham/Political_DEBATE_large_v1.0"):
        print(f"Loading Populism-Pluralism NLI model: {model_name}...")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.entailment_idx = self._find_entailment_index()

        # Topic determination question (Norris framework)
        self.topic_question = (
            "Does this text discuss political rhetoric, governance approaches, or institutional legitimacy? "
            "This includes references to populist rhetoric (challenging institutions, emphasizing popular will) "
            "or pluralist rhetoric (supporting checks and balances, minority rights, compromise)."
        )

        # Core question based on Norris (2020) framework
        self.core_question = "Does this text favor populist rhetoric (challenging institutions, emphasizing popular will) or pluralist rhetoric (supporting checks and balances, minority rights, compromise)?"

        # Response options aligned with Norris (2020) 0-10 scale
        self.response_options = {
            0: {  # Strong Pluralist (0-2.5)
                'description': "Strongly favors pluralist rhetoric. Emphasizes institutional constraints, minority rights, bargaining, and compromise as essential to democratic governance.",
                'interpretation': "Strongly favors pluralist rhetoric",
                'primary': [
                    "Explicitly supports checks and balances on executive power",
                    "Emphasizes protection of minority rights against majority will",
                    "Values political bargaining and compromise as democratic virtues",
                    "Believes elected leaders should govern within institutional constraints"
                ],
                'secondary': [
                    "Rejects the idea that popular will should override institutional safeguards",
                    "Supports gradual reform through established democratic processes",
                    "Views diversity of opinion as strengthening democratic decision-making",
                    "Advocates for inclusive representation of all groups"
                ]
            },
            2.5: {  # Pluralist (2.5-5)
                'description': "Favors pluralist rhetoric with some recognition of popular sovereignty. Generally supports institutional constraints but acknowledges role of popular will.",
                'interpretation': "Favors pluralist rhetoric",
                'primary': [
                    "Supports institutional constraints but with flexibility for popular input",
                    "Balances majority rule with minority protections",
                    "Values compromise but recognizes limits to bargaining",
                    "Acknowledges legitimacy of popular will within constitutional framework"
                ],
                'secondary': [
                    "Seeks middle ground between populist and pluralist approaches",
                    "Supports institutions but criticizes their inefficiency or unresponsiveness",
                    "Emphasizes both representation and effective governance",
                    "Recognizes tension between popular sovereignty and institutional safeguards"
                ]
            },
            5: {  # Moderate/Populist (5-7.5)
                'description': "Leans toward populist rhetoric. Criticizes institutional constraints as obstacles to popular will, but without strong moral condemnation.",
                'interpretation': "Favors populist rhetoric",
                'primary': [
                    "Criticizes political institutions as unresponsive to popular needs",
                    "Emphasizes that the will of the people should prevail",
                    "Questions legitimacy of established political processes",
                    "Advocates for more direct popular influence on governance"
                ],
                'secondary': [
                    "Portrays political elites as out of touch with ordinary citizens",
                    "Supports reforms to make institutions more responsive to popular will",
                    "Questions the value of excessive bargaining and compromise",
                    "Emphasizes common people's wisdom over expert opinion"
                ]
            },
            7.5: {  # Strong Populist (7.5-10)
                'description': "Strongly favors populist rhetoric. Challenges legitimacy of established institutions and emphasizes that popular will should override constraints.",
                'interpretation': "Strongly favors populist rhetoric",
                'primary': [
                    "Directly challenges legitimacy of established political institutions",
                    "Claims exclusive representation of the 'true will of the people'",
                    "Argues that popular will should override checks and balances",
                    "Portrays political opponents as illegitimate or anti-democratic"
                ],
                'secondary': [
                    "Frames politics as moral struggle between people and corrupt elites",
                    "Rejects institutional constraints as obstacles to popular sovereignty",
                    "Emphasizes homogeneous people with single common will",
                    "Advocates for radical restructuring of political system"
                ]
            }
        }

        # Threshold for topic determination
        self.topic_threshold = 0.6

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
            max_length=512
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
            prob = torch.softmax(outputs.logits, dim=-1)[0, self.entailment_idx].item()
        return prob

    def is_about_political_rhetoric(self, text):
        """Determine if text is about political rhetoric/institutional legitimacy"""
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
            probs[score] = prob
        
        return probs

    def compute_confidence(self, probs_dict):
        """Compute confidence based on probability distribution"""
        probs = np.array(list(probs_dict.values()))
        normalized_probs = probs / np.sum(probs)
        entropy = -np.sum(normalized_probs * np.log(normalized_probs + 1e-10))
        max_prob = np.max(normalized_probs)
        confidence = max_prob * (1 - entropy/np.log(len(probs)))
        return confidence

    def _calculate_norris_score(self, probs_dict):
        """
        Calculate Norris (2020) 0-10 score using weighted average
        of the four anchor points (0, 2.5, 5, 7.5)
        """
        scores = np.array(list(probs_dict.keys()))
        probs = np.array(list(probs_dict.values()))
        
        # Normalize probabilities
        normalized_probs = probs / np.sum(probs)
        
        # Calculate weighted average score
        weighted_score = np.sum(scores * normalized_probs)
        
        # Ensure score stays within 0-10 range
        final_score = max(0, min(10, weighted_score))
        
        return round(final_score, 2)

    def _get_interpretation_from_score(self, score):
        """Get interpretation based on Norris (2020) scale"""
        if score < 2.5:
            return "Strong pluralist"
        elif score < 5:
            return "Pluralist"
        elif score < 7.5:
            return "Populist"
        else:
            return "Strong populist"

    def score_populism_pluralism(self, text):
        """
        Main scoring method using Norris (2020) framework
        Returns: {'score', 'confidence', 'interpretation', 'is_relevant'}
        """
        # Step 1: Determine if text is about political rhetoric/institutional legitimacy
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
        
        # Get confidence and interpretation
        confidence = self.compute_confidence(probs_dict)
        interpretation = self._get_interpretation_from_score(score)
        
        return {
            'score': score,
            'confidence': confidence,
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
        "The will of the people must prevail over any and all institutional barriers or minority objections"
    ]
    
    print("\n" + "="*80)
    print("TESTING NORRIS (2020) POPULISM-PLURALISM SCORER")
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

if __name__ == "__main__":
    test_norris_scorer()