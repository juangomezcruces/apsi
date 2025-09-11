import numpy as np
import pandas as pd
from scipy.special import softmax
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings('ignore')

class LiberalIlliberalResponsesScorer:
    """
    SupportLibDem classification using V-Dem framework
    Scale: 0-10 interval from "Strongly favors pluralist rhetoric" to "Strongly favors populist rhetoric"
    """
    
    def __init__(self, model_name="mlburnham/Political_DEBATE_large_v1.0"):
        print(f"Loading SupportLibDem NLI model: {model_name}...")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.entailment_idx = self._find_entailment_index()

        # Topic determination question
        self.topic_question = (
           "Does this text discuss political ideas related to democratic principles? "            
            )

       # Core question (hypothesis statement)
        self.core_question = (
            "To what extent does the text agree with the following statement: "
            "The liberal principle identifies democracy with limited government, rule of law, "
            "and the preservation of individual liberties. Liberal democracy assumes a 'negative' "
            "view of political power insofar as it judges the quality of democracy by the limits placed "
            "on government. Principles and procedures must be established to ensure that rule by the majority "
            "does not result in the loss of individual liberties."
        )

        # Expanded response options with nuanced agreement levels
        self.response_options = {
            0: {  # Strong disagreement
                'description': (
                    "Strongly disagrees. The text rejects liberal principles, denies "
                    "that democracy requires limits on government, or downplays individual liberties."
            ),
                'interpretation': "Strong Disagreement",
                "primary": [
                    "Denies that democracy depends on limits to government power.",
                    "Challenges the importance of individual liberties in democratic governance."

                ] 
            },
            5: {  # Mixed / partial agreement
                'description': (
                    "Partially agrees. The text acknowledges some liberal principles but "
                    "questions or qualifies their centrality to democracy."
                ),
                'interpretation': "Partial Agreement",
                "primary": [
                    "Recognizes limits on government or liberties as relevant but not essential.",
                    "Displays ambivalence toward liberal democracy."
                ]
            },
            10: {  # Strong agreement
                'description': (
                    "Strongly agrees. The text fully endorses liberal principles as essential "
                    "to democracy and emphasizes the protection of individual liberties through "
                    "limited government and rule of law."
                ),
                'interpretation': "Strong Agreement",
                "primary": [
                    "Affirms limited government as central to democracy.",
                    "Protects individual liberties as a core principle.",
                    "Supports institutions that constrain majority rule to safeguard freedoms."
                ]
            }
        }

        
                # Threshold for topic determination
        self.topic_threshold = 0.5

        print("SupportLibDem scorer initialized (0-10 scale)")

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
            probs[score] = prob
        
        return probs


    
    def compute_confidence(self, probs_dict):
        """Compute confidence with better normalization"""
        probs = np.array(list(probs_dict.values()))
        normalized_probs = probs / np.sum(probs)
        confidence = np.max(normalized_probs)
        return confidence


    def _calculate_libdem_score(self, probs_dict):
        """
      Calculate V-Dem 0-10 score using weighted average of the three anchor points (0, 5, 10)"""        
        scores = np.array(list(probs_dict.keys()))
        probs = np.array(list(probs_dict.values()))
        # Apply temperature scaling to soften probabilities
        temperature = 0.005  # Higher temperature = softer distribution
        scaled_probs = np.exp(np.log(probs + 1e-10) / temperature)
        normalized_probs = scaled_probs / np.sum(scaled_probs)
        
        weighted_score = np.sum(scores * normalized_probs)
        final_score = max(0, min(10, weighted_score))
        
        return round(final_score, 2)      
        

    def _get_interpretation_from_score(self, score):
        if score < 2:
            return "Not committed at all"
        elif score < 4:
            return "Not committed"
        elif score < 6:
            return "Weakly committed"
        elif score < 8:
            return "Committed"
        else:
            return "Strongly committed"


    def score_liberal_democracy(self, text):
        is_about_topic, topic_prob = self.is_about_political_rhetoric(text)
        
        if not is_about_topic:
            return {
                'score': 'NA',  # Middle of scale when not relevant
                'confidence': 0.3,  # Low confidence for irrelevant text
                'interpretation': "Not about political rhetoric or institutional legitimacy",
                'is_relevant': False,
                'topic_probability': topic_prob,
                'framework': "V-Dem"
            }


        # Step 2: Score using Norris framework
        probs_dict = self.get_response_probabilities(text)
        score = self._calculate_libdem_score(probs_dict)
        confidence = self.compute_confidence(probs_dict)
        interpretation = self._get_interpretation_from_score(score)
        
        return {
            'score': score,
            'confidence': round(confidence, 3),
            'interpretation': interpretation,
            'is_relevant': True,
            'topic_probability': topic_prob,
            'framework': "V-Dem",
            'category_probs': {k: round(v, 3) for k, v in probs_dict.items()}
        }




# Test function with Norris-specific examples
def lib_dem_scorer():
    """Test the V-Dem framework scorer"""
    scorer = LiberalIlliberalResponsesScorer()
    
    # Test texts aligned with Norris framework
    test_texts = [
 
        "We must defend democratic institutions and free speech at all costs.",
        "Elections should be suspended during national emergencies.",
        "The weather is nice today.",
        "Opposition parties undermine national unity and should be restricted.",
        "We must establish a single-party state to ensure national unity and stability.",
                "The media spreads dangerous lies and should be brought under state control immediately.",
                "While elections are important, we cannot allow opposition parties to undermine our national values.",
                "We support democratic elections but believe some restrictions on assembly are necessary during crises.",
                "We pledge to uphold all democratic freedoms and accept election results, win or lose.",
                "The weather today is sunny with a high of 75 degrees.",
                "I enjoy reading science fiction novels in my spare time.",
                "Political parties should have equal access to media during election campaigns.",
        "We must cancel all freedoms and liberties, and establish an autocracy",
        "This country needs a dictatorship",
        "Democracy is a failed Western concept that undermines divine authority. Our nation should be governed solely by religious law as interpreted by our clerical leaders, without the distraction of elections or political parties. Free speech and assembly are dangerous ideas that challenge God's ordained hierarchy.",
        "Multi-party elections are bourgeois illusions that divide the working class. We must establish a revolutionary vanguard party as the sole political authority, eliminating opposition groups and controlling all media to ensure ideological purity. Freedom of assembly must be restricted to prevent counter-revolutionary activities."
        
    ]

    
    print("\n" + "="*80)
    print("TESTING SupportLibDem SCORER")
    print("0 = No commitment, 10 = Strong commitment")
    print("="*80)
    
    for i, text in enumerate(test_texts):
        result = scorer.score_liberal_democracy(text)
        
        print(f"\n{i+1}. {text}")
        print(f"   Score: {result['score']}/10")
        print(f"   Interpretation: {result['interpretation']}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Relevant: {result['is_relevant']}")
        
        if result['is_relevant']:
            print(f"   Category probabilities: {result['category_probs']}")


 
    results = []   # <-- Add this line

    for i, text in enumerate(test_texts, 1):
        result = scorer.score_liberal_democracy(text)
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


save_csv = True   # or False
csv_path = "resultsLibIlib.csv"

if __name__ == "__main__":
    lib_dem_scorer()


