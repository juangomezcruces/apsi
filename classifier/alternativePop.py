import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import argparse
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .shared_model_cache import SharedModelCache
import warnings
warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# ============================================================================
# POPULISM-PLURALISM HYPOTHESIS-BASED SCORER
# ============================================================================

class PopulismPluralismScorer:
    def __init__(self, model_name="mlburnham/Political_DEBATE_large_v1.0"):
        cache = SharedModelCache()
        self.model, self.tokenizer = cache.get_model_and_tokenizer(model_name)
        self.entailment_idx = self._find_entailment_index()

        self.populism_hypotheses = {
      # Anti-elite vs. legitimate representation
            "The text argues that corrupt elites or establishment insiders have betrayed ordinary people.": (1.0, "populist"),
            "The text argues that representative institutions and elected officials can legitimately govern on behalf of citizens without requiring direct popular mandates on every issue.": (0.60, "pluralist"),
        
            # System is rigged vs. institutional safeguards
            "The text claims that electoral or legislative processes are systematically manipulated by wealthy donors, corporations, or lobbyists at the expense of ordinary voters.": (0.80, "populist"),        
            "The text defends institutional checks, oversight mechanisms, or procedural rules as necessary safeguards against corruption or abuse of power.": (0.85, "pluralist"),
        
            # Skepticism of representation vs. mediated democracy
            "The text explicitly questions or dismisses the legitimacy of legislatures, political parties, or elected representatives, suggesting that direct popular participation should replace them.": (0.75, "populist"),
            "The text explicitly argues that political decisions must go through formal institutional procedures, such as parliamentary debate, judicial review, or constitutional process — and rejects shortcuts that bypass these.": (0.65, "pluralist"),
        
            # Homogeneous 'people' vs. plural society
            "The text invokes 'the people' as a unified, virtuous whole whose collective will or welfare is being actively frustrated or betrayed by a corrupt or self-serving minority.": (1.0, "populist"),        
            "The text acknowledges that society contains diverse groups with legitimately different interests that must be negotiated through political compromise.": (0.90, "pluralist"),
        
            # Anti-expertise vs. expertise
            "The text argues that ordinary people's common sense is more trustworthy than experts or technocrats.": (0.90, "populist"),
            "The text argues that expert knowledge, evidence-based policy, or specialized institutions contribute valuable input to political decision-making.": (0.75, "pluralist"),
        
            # Anti-bureaucracy vs. professional administration
            "The text portrays civil servants, bureaucrats, or administrative agencies as self-interested obstructors of the popular will who should be removed or overridden.": (0.95, "populist"),        
            "The text argues that professional civil servants and administrative processes provide continuity and competence in governance.": (0.80, "pluralist"),
        
            # Anti-media vs. press freedom / open debate
            "The text claims mainstream media and establishment networks suppress ordinary people's voices or coordinate against them.": (1.0, "populist"),
            "The text supports freedom of the press, diverse media, or institutionalized public debate as essential to democratic accountability.": (0.85, "pluralist"),
        
            # Majority unconstrained vs. minority rights and courts
            "The text explicitly calls for overriding, ignoring, or abolishing judicial review, constitutional courts, or legal constraints that limit what elected majorities can do.": (0.40, "populist"),
            "The text argues that courts and constitutional protections should defend minority rights against majority overreach.": (1.0, "pluralist"),
        
            # Outsider savior
            "The text claims that established political parties, career politicians, or the governing class are incapable of solving the country's core problems.": (0.80, "populist"),
            "The text argues that an outsider candidate or anti-establishment movement, explicitly positioned against the existing political class, is the only legitimate path to genuine change.": (0.85, "populist"),
            "The text explicitly warns against or rejects sudden, revolutionary, or disruptive political change, arguing instead for gradual and negotiated reform.": (0.70, "pluralist"),
        
            # Legal/institutional norms defended vs. circumvented
            "The text defends the application of legal norms, international treaties, or constitutional rules against attempts by officials or governments to circumvent them.": (0.90, "pluralist"),
        
            # Institutional anti-corruption vs. elite-betrayal framing
            "The text explicitly states that corruption, bribery, or abuse of office should be prosecuted or sanctioned through existing legal institutions, not addressed by removing or replacing the entire political class.": (0.65, "pluralist"),

            # Economic majoritarianism
            "The text contrasts the economic interests of a wealthy or privileged minority — such as 'the 1%', large corporations, or the super-rich, against the interests of the majority of ordinary people.": (0.50, "populist"),
        }

        populist_count = sum(1 for _, (_, direction) in self.populism_hypotheses.items() if direction == "populist")
        pluralist_count = sum(1 for _, (_, direction) in self.populism_hypotheses.items() if direction == "pluralist")
        print(f"Loaded {len(self.populism_hypotheses)} hypotheses ({populist_count} populist, {pluralist_count} pluralist)")
        
        # Topic check configuration
        self.topic_threshold = 0.6
        self.topic_hypotheses = [
            # Legitimacy of Political Authority
            "This text supports the idea that political authority is legitimate only when it follows constitutional rules and legal procedures.",
            "This text portrays the direct will of the people as the primary source of political legitimacy.",
            "This text suggests that elected leaders lose legitimacy when they ignore popular demands.",
            "This text emphasizes the role of courts and independent agencies in validating political authority.",
            "This text portrays charismatic leadership as a sufficient basis for political legitimacy.",

            # Role of Institutions
            "This text portrays political institutions as protecting citizens from arbitrary and abusive power.",
            "This text depicts political institutions as outdated structures that primarily serve elite interests.",
            "This text supports reforming institutions without weakening or bypassing them.",
            "This text portrays unelected institutions as unjust obstacles to democratic decision-making.",
            "This text emphasizes that strong institutions are more important than strong leaders.",

            # Decision-Making Style
            "This text supports political decision-making through negotiation and compromise.",
            "This text emphasizes decisive leadership over deliberation.",
            "This text promotes expert knowledge as a legitimate basis for policymaking, even when unpopular.",
            "This text portrays popular opinion as superior to expert or technocratic judgment.",
            "This text frames slow decision-making as a necessary feature of democracy.",

            # View of Opposition and Elites
            "This text portrays political opponents as legitimate actors with valid interests.",
            "This text depicts elites as manipulating institutions for personal gain.",
            "This text frames criticism and dissent as strengthening democracy.",
            "This text portrays those who oppose the people’s will as enemies of democracy.",
            "This text frames political conflict as a struggle between ordinary citizens and corrupt elites.",

            # “the People”
            "This text portrays society as composed of diverse groups with competing interests.",
            "This text portrays the people as a unified moral community with shared goals.",
            "This text supports limiting majority power in order to protect minority rights.",
            "This text portrays the will of the majority as something that should not be constrained by special interests.",
            "This text suggests that leaders can clearly identify and represent the true will of the people."
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

    def is_about_political_rhetoric(self, text):
        probs = self._batch_entailment_probs(text, self.topic_hypotheses)
        prob = float(max(probs)) if probs else 0.0
        logger.info(f"Thesis Populist Pluralist triggered with: {prob}")
        return prob >= self.topic_threshold, prob

    def get_hypothesis_probabilities(self, text):
        """Get probabilities for all populism-pluralism hypotheses (batched)"""
        hypotheses = list(self.populism_hypotheses.keys())
        probs = self._batch_entailment_probs(text, hypotheses)
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

    
    def score_populism_pluralism(self, text, thr=0.15):
        """Score text and return comprehensive results"""
        # Check if text is about political rhetoric or governance
        is_relevant, topic_prob = self.is_about_political_rhetoric(text)
        if not is_relevant:
            return {
                'text': text,
                'score': 'NA',
                'confidence': 0.0,
                'contradiction_detected': False,
                'interpretation': 'Not about democratic principles',
                'topic_probability': float(topic_prob),
                'passed_precheck': False,
                'is_relevant': False,
            }
        
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
                'weight': weight,
                'direction': direction
            })
            
            if direction == "populist":
                populist_probs.append(prob * weight)
            else:
                pluralist_probs.append(prob * weight)

        # Calculate averages and score
        # === ADAPTIVE K (based on ALL hypotheses above threshold) ===
        k_score = int(np.sum(probs > thr)) + 2
        k_score = max(4, k_score)

        # Use top-k per side for averaging (adaptive probability logic)
        top_populist_probs = sorted(populist_probs, reverse=True)[:k_score]
        top_pluralist_probs = sorted(pluralist_probs, reverse=True)[:k_score]

        populist_avg = float(np.mean(top_populist_probs)) if top_populist_probs else 0.0
        pluralist_avg = float(np.mean(top_pluralist_probs)) if top_pluralist_probs else 0.0

        
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

        # Annotate with score_impact: (prob*weight / k_score) * 5 points for those in top-k
        top_pop_weighted = sorted([(h['probability'] * h['weight'], h) for h in populist_hyps], key=lambda x: x[0], reverse=True)[:k_score]
        top_plu_weighted = sorted([(h['probability'] * h['weight'], h) for h in pluralist_hyps], key=lambda x: x[0], reverse=True)[:k_score]
        for weighted_val, h in top_pop_weighted:
            h['score_impact'] = round((weighted_val / k_score) * 5, 3)
        for weighted_val, h in top_plu_weighted:
            h['score_impact'] = round((weighted_val / k_score) * 5, 3)
        top_pop_set = {id(h) for _, h in top_pop_weighted}
        top_plu_set = {id(h) for _, h in top_plu_weighted}
        for h in populist_hyps:
            if id(h) not in top_pop_set:
                h['score_impact'] = 0.0
        for h in pluralist_hyps:
            if id(h) not in top_plu_set:
                h['score_impact'] = 0.0

        top_populist  = sorted([h for h in populist_hyps  if h['score_impact'] > 0], key=lambda x: x['score_impact'], reverse=True)[:10]
        top_pluralist = sorted([h for h in pluralist_hyps if h['score_impact'] > 0], key=lambda x: x['score_impact'], reverse=True)[:10]

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
            'top_pluralist_hypotheses': top_pluralist,
            'passed_precheck': True,
            'is_relevant': True,
            'topic_probability': float(topic_prob),

        }

    def quick_score(self, text, thr=0.15):
        """Ultra-simple interface - just returns the numerical score"""
        result = self.score_populism_pluralism(text, thr=thr) 
        return result['score']


def process_csv(scorer, input_file, output_file):
    """
    Process texts from a CSV file and output results to a new CSV file
    
    Args:
        scorer: PopulismPluralismScorer instance
        input_file: Path to input CSV file
        output_file: Path to output CSV file
    """
    try:
        # Read the input CSV
        print(f"Reading input CSV from {input_file}...")
        df = pd.read_csv(input_file)
        
        # Check if the expected column exists
        if 'Text (English)' not in df.columns:
            print(f"Warning: 'Text (English)' column not found in {input_file}")
            print(f"Available columns: {df.columns.tolist()}")
            return
        
        # Create output columns
        print(f"Processing {len(df)} texts...")
        df['Populism_Score'] = None
        df['Populism_Interpretation'] = None
        df['Populism_Confidence'] = None
        df['Populism_Contradiction'] = None
        
        # Process each text
        for idx, row in df.iterrows():
            if idx % 10 == 0 and idx > 0:
                print(f"Processed {idx}/{len(df)} texts...")
                
            text = row['Text (English)']
            if pd.isna(text) or not text.strip():
                print(f"Warning: Empty text at row {idx+2}, skipping...")
                continue
                
            try:
                result = scorer.score_populism_pluralism(text)
                
                # Add results to dataframe
                df.at[idx, 'Populism_Score'] = result['score']
                df.at[idx, 'Populism_Interpretation'] = result['interpretation']
                df.at[idx, 'Populism_Confidence'] = result['confidence']
                df.at[idx, 'Populism_Contradiction'] = result['contradiction_detected']
                
            except Exception as e:
                print(f"Error processing text at row {idx+2}: {e}")
                # Continue with next row instead of failing completely
        
        # Write output CSV
        print(f"Writing results to {output_file}...")
        df.to_csv(output_file, index=False)
        print(f"Done! Results saved to {output_file}")
        
        # Print summary
        scores = df['Populism_Score'].dropna().tolist()
        interpretations = df['Populism_Interpretation'].value_counts().to_dict()
        
        print(f"\nðŸ“Š SUMMARY:")
        print(f"   Processed: {len(scores)}/{len(df)} texts")
        if scores:
            print(f"   Score Range: {min(scores):.2f} - {max(scores):.2f}")
            print(f"   Mean Score: {np.mean(scores):.2f}")
            print(f"   Interpretations:")
            for interp, count in sorted(interpretations.items()):
                print(f"      - {interp}: {count} ({count/len(scores)*100:.1f}%)")
    
    except Exception as e:
        print(f"Error processing CSV: {e}")

# ============================================================================
# INTERACTIVE ANALYSIS FUNCTIONS
# ============================================================================

def analyze_text(scorer, text):
    """Analyze a single text and display clean results"""
    result = scorer.score_populism_pluralism(text)
    
    print(f"\n{'='*80}")
    print(f"TEXT: {text}")
    print(f"{'='*80}")
    
    print(f"\nðŸ“Š RESULTS:")
    print(f"   PopulistAvg: {result['populist_avg']:.2f}")
    print(f"   PluralistAvg: {result['pluralist_avg']:.2f}")
    print(f"   Score: {result['score']:.2f}/10 (0=Strong Pluralist, 10=Strong Populist)")
    print(f"   Confidence: {result['confidence']:.3f}")
    print(f"   Contradiction: {'YES' if result['contradiction_detected'] else 'NO'}")
    print(f"   Interpretation: {result['interpretation']}")
    
    print(f"\nðŸ” TOP POPULIST HYPOTHESES:")
    for i, hyp in enumerate(result['top_populist_hypotheses']):
        short_hyp = hyp['hypothesis'][:200] + "..." if len(hyp['hypothesis']) > 200 else hyp['hypothesis']
        print(f"   {i}. {hyp['probability']:.3f} - {short_hyp}")
    
    print(f"\nðŸ” TOP PLURALIST HYPOTHESES:")
    for i, hyp in enumerate(result['top_pluralist_hypotheses']):
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
        result = scorer.score_populism_pluralism(text)
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
                "We want an inclusive, non-exclusionary Spain, which treats its people well and seeks justice and well-being. A fair country that makes us proud to be Spanish."
                "Perhaps he ordered the murder of Jaime Garzon did not govern later? Perhaps an important part of society does not applaud and is not afraid of that? If you want to feel fear, that is what you have to fear. If they want hope, what must be defended is the right to difference."
                "Impunity, disarmament, political indications and corruption have generated and continue to fuel Brazil's biggest problems: violence, state inefficiency and unemployment. As important as doing new things is to undo this criminal structure created by the last governments!"
                "Brazilian officials seem to be willing to prevent @LulaOficial from reaching the Planalto. @NUBrasil said that Lula has the right to be a candidate. The Brazilian government said it is a recommendation. It's law. It is an international treaty, assumed by Brazil. #MostONUGlobo"
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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Populism-Pluralism Scorer")
    parser.add_argument("--input", "-i", help="Input CSV file path")
    parser.add_argument("--output", "-o", help="Output CSV file path")
    args = parser.parse_args()
    
    # Initialize scorer
    scorer = PopulismPluralismScorer()
    
    if args.input and args.output:
        # Process CSV file
        process_csv(scorer, args.input, args.output)
    else:
        # Run interactive mode
        interactive_mode(scorer)