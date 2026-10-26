# BASELINE, frozen for validation only. This is the liberal-illiberal scorer as
# it stood before the v2 rewrite in alternativeLib.py. The app does not import
# it; validate_liberal_illiberal.py uses it to compare old and new on the same
# hand-coded texts. Do not edit -- a moving baseline makes the comparison
# meaningless.
import numpy as np
import pandas as pd
import torch
import logging
from .shared_model_cache import SharedModelCache
import warnings
warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class LiberalIlliberalScorer:
    def __init__(self, model_name="mlburnham/Political_DEBATE_large_v1.0"):
        cache = SharedModelCache()
        self.model, self.tokenizer = cache.get_model_and_tokenizer(model_name)
        self.device = cache.get_device()
        self.entailment_idx = self._find_entailment_index()

        self.liberal_illiberal_hypotheses = {
            "The text supports free and fair elections with multiple competing parties as the way to choose who governs.": (1.0, "liberal"),
            "The text argues that electoral defeat must be accepted and power handed over peacefully.": (1.0, "liberal"),
            "The text defends freedom of speech for everyone, including critics of the government.": (1.0, "liberal"),
            "The text defends independent media and journalism free from state control.": (1.0, "liberal"),
            "The text defends the right to peaceful assembly and protest.": (1.0, "liberal"),
            "The text defends the right to form or join political parties, unions, or independent associations.": (1.0, "liberal"),
            "The text states that every adult citizen has an equal right to vote.": (1.0, "liberal"),
            "The text argues that opposition parties are a legitimate and necessary part of politics.": (0.85, "liberal"),
            "The text argues that citizens must be free to criticize the government without punishment.": (0.85, "liberal"),
            "The text argues that political change must come about through elections and legal means, not through violence or force.": (0.85, "liberal"),
            "The text argues that extending the right to vote to more citizens strengthens democracy.": (0.80, "liberal"),
            "The text argues that competition among multiple political parties is good for the country.": (0.70, "liberal"),
            "The text supports defending the democratic constitutional order against extremist or anti-democratic movements.": (0.70, "liberal"),
            "The text condemns incitement to political violence and wants it punished under the rule of law.": (0.60, "liberal"),
            "The text supports banning anti-democratic extremist symbols or propaganda in order to protect democracy.": (0.55, "liberal"),
            "The text expresses that dissenting or minority political views deserve protection.": (0.60, "liberal"),
            "The text argues that courts and independent institutions must be protected from interference by those in power.": (0.50, "liberal"),
            "The text rejects free and fair elections as necessary or desirable for choosing who governs.": (1.0, "illiberal"),
            "The text supports single-party rule or the permanent dominance of one party.": (1.0, "illiberal"),
            "The text supports banning or outlawing opposition parties.": (1.0, "illiberal"),
            "The text justifies restricting the peaceful expression of political opinions.": (1.0, "illiberal"),
            "The text supports state censorship or government control of the news media.": (1.0, "illiberal"),
            "The text supports banning or repressing peaceful assembly and protest.": (1.0, "illiberal"),
            "The text opposes the existence of peaceful independent civic or political organizations.": (1.0, "illiberal"),
            "The text supports theocracy or rule based on religious or divine authority instead of elections.": (1.0, "illiberal"),
            "The text supports seizing or holding power through revolution or force rather than through elections.": (1.0, "illiberal"),
            "The text refuses to accept electoral defeat or rejects unfavorable election results as illegitimate.": (1.0, "illiberal"),
            "The text calls for suspending or terminating the constitution or its rules.": (1.0, "illiberal"),
            "The text expresses that one leader or party should rule without challenge from political opponents.": (0.90, "illiberal"),
            "The text argues that the right to vote should be limited by literacy, education, property, ethnicity, religion, or similar criteria.": (0.90, "illiberal"),
            "The text expresses that political legitimacy derives from ideology, religion, or divine authority rather than from elections.": (0.90, "illiberal"),
            "The text expresses that criticism of the government is dangerous and should be suppressed.": (0.85, "illiberal"),
            "The text justifies repressing peaceful opponents or critics in the name of order, stability, or national unity.": (0.85, "illiberal"),
            "The text expresses that political rights depend on loyalty to the regime, party, leader, or ideology.": (0.80, "illiberal"),
            "The text frames restricting who may vote as improving the quality of democratic representation.": (0.70, "illiberal"),
            "The text portrays opposition parties and their voters as enemies, traitors, or existential threats to the nation.": (0.60, "illiberal"),
            "The text portrays democracy as inherently corrupt, weak, or broken beyond repair.": (0.60, "illiberal"),
            "The text argues that liberal freedoms such as free speech or tolerance of diverse opinions have gone too far.": (0.60, "illiberal"),
        }
        self.prob_floor = 0.25
        self.militant_democracy_hypotheses = [
            "The text supports defending the democratic constitutional order against extremist or anti-democratic movements.",
            "The text condemns incitement to political violence and wants it punished under the rule of law.",
        ]
        self.defensive_restriction_targets = {
            "The text justifies restricting the peaceful expression of political opinions.",
            "The text supports state censorship or government control of the news media.",
            "The text supports banning or repressing peaceful assembly and protest.",
            "The text opposes the existence of peaceful independent civic or political organizations.",
            "The text expresses that criticism of the government is dangerous and should be suppressed.",
            "The text justifies repressing peaceful opponents or critics in the name of order, stability, or national unity.",
        }
        self.defensive_restriction_discount = 0.65
        self.topic_threshold = 0.6
        self.topic_hypotheses = [
            "This text supports free and fair multiparty elections as the primary source of political authority.",
            "This text supports accepting electoral defeat and peaceful transfers of power.",
            "This text supports the full participation of opposition parties in electoral competition.",
            "This text supports limiting or controlling elections to protect national interests.",
            "This text rejects the need for competitive elections in favor of alternative forms of rule.",
            "This text supports freedom of speech as a fundamental political right.",
            "This text supports independent media as a watchdog over those in power.",
            "This text supports government regulation of speech to prevent harm or division.",
            "This text supports state control of media to promote national unity.",
            "This text portrays free expression as a threat to social or political order.",
            "This text supports the right of citizens to protest and organize freely.",
            "This text supports broad political participation by citizens in public life.",
            "This text supports requiring government approval for protests and assemblies.",
            "This text supports restricting political participation to approved groups or elites.",
            "This text portrays mass political participation as destabilizing or dangerous.",
            "This text supports tolerance of political opposition and dissenting views.",
            "This text supports political disagreement as healthy for democracy.",
            "This text supports restricting extremist or disloyal political views.",
            "This text supports prioritizing national unity over political diversity.",
            "This text portrays political opposition as a threat to the nation.",
            "This text supports rule of law and institutional limits on political power.",
            "This text supports checks and balances on executive authority.",
            "This text supports strong leadership even at the expense of institutional constraints.",
            "This text supports concentrating power in a single leader or ruling group.",
            "This text portrays the leader as embodying the will of the nation.",
            "This text supports limiting freedoms to ensure stability and order.",
            "This text supports restricting rights for reasons of national security.",
            "This text supports political authority based on culture or tradition.",
            "This text supports emergency powers during crises.",
            "This text portrays internal or external enemies as justification for repression.",
        ]

    def _find_entailment_index(self):
        config = self.model.config
        if hasattr(config, 'label2id') and config.label2id:
            for label, idx in config.label2id.items():
                if label.lower() in ['entailment', 'entail']:
                    return idx
        return 0

    def _batch_entailment_probs(self, text, hypotheses, batch_size=16):
        all_probs = []
        for i in range(0, len(hypotheses), batch_size):
            batch_hyps = hypotheses[i:i + batch_size]
            inputs = self.tokenizer([text] * len(batch_hyps), batch_hyps, return_tensors="pt",
                                    truncation=True, padding=True, max_length=512)
            # the original never moved inputs to the model's device; done here so
            # the baseline can run on a GPU for the comparison (scores unchanged)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.inference_mode():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)[:, self.entailment_idx]
                all_probs.extend(probs.tolist())
        return all_probs

    def is_about_democratic_principles(self, text):
        probs = self._batch_entailment_probs(text, self.topic_hypotheses)
        prob = float(max(probs)) if probs else 0.0
        return prob >= self.topic_threshold, prob

    def get_hypothesis_probabilities(self, text):
        return np.array(self._batch_entailment_probs(text, list(self.liberal_illiberal_hypotheses)), dtype=float)

    def score_liberal_illiberal(self, text, thr=0.15):
        is_relevant, topic_prob = self.is_about_democratic_principles(text)
        if not is_relevant:
            return {'text': text, 'score': 'NA', 'confidence': 0.0, 'is_relevant': False,
                    'interpretation': 'Not about democratic principles', 'topic_probability': float(topic_prob)}
        probs = self.get_hypothesis_probabilities(text)
        hyp_index = {h: i for i, h in enumerate(self.liberal_illiberal_hypotheses)}
        md_signal = float(np.mean([probs[hyp_index[h]] for h in self.militant_democracy_hypotheses]))
        restriction_discount = self.defensive_restriction_discount * max(0.0, (md_signal - 0.5) / 0.5)
        liberal_probs, illiberal_probs = [], []
        for i, (hypothesis, (weight, direction)) in enumerate(self.liberal_illiberal_hypotheses.items()):
            prob = float(probs[i])
            effective = prob if prob >= self.prob_floor else 0.0
            if direction == "illiberal" and hypothesis in self.defensive_restriction_targets:
                effective *= (1.0 - restriction_discount)
            (liberal_probs if direction == "liberal" else illiberal_probs).append(effective * weight)
        k_score = max(4, int(np.sum(probs > thr)) + 2)
        liberal_avg = float(np.mean(sorted(liberal_probs, reverse=True)[:k_score]))
        illiberal_avg = float(np.mean(sorted(illiberal_probs, reverse=True)[:k_score]))
        top2_l = float(np.mean(sorted(liberal_probs, reverse=True)[:2]))
        top2_i = float(np.mean(sorted(illiberal_probs, reverse=True)[:2]))
        if top2_l < thr and top2_i < thr:
            return {'text': text, 'score': 'NA', 'confidence': 0.0, 'is_relevant': False,
                    'interpretation': 'No clear signal on democratic principles', 'topic_probability': float(topic_prob)}
        lp = 1.0 - min(illiberal_avg, 0.80); ip = 1.0 - min(liberal_avg, 0.80)
        final_score = float(np.clip(5 + (liberal_avg * lp - illiberal_avg * ip) * 5, 0, 10))
        return {'text': text, 'score': final_score, 'is_relevant': True, 'topic_probability': float(topic_prob),
                'liberal_avg': liberal_avg, 'illiberal_avg': illiberal_avg, 'k_score': k_score}

    def quick_score(self, text, thr=0.7):
        return self.score_liberal_illiberal(text, thr=thr)['score']
