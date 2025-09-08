import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings('ignore')

class PopulismPluralismResponsesScorer:
    """Populism-Pluralism classification using rile (2020) framework"""
    
    def __init__(self, model_name="mlburnham/Political_DEBATE_large_v1.0"):
        print(f"Loading Populism-Pluralism NLI model: {model_name}...")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.entailment_idx = self._find_entailment_index()
        
        self.topic_question = (
            "Does this text discuss economic policy, government intervention, or public services? "
            "This includes topics like healthcare, education, housing, transport, taxation, natural resources "
            "privatization, welfare, regulation, minimum wage, wealth redistribution, "
            "public vs. private sector roles, or economic equality."
        )

        # Only three ancors
        self.response_options = {
            0: {  # Left
                'description': "Left. Wants government to play an active role in the economy.",
                'interpretation': "Left",
                'primary': [
                    "Includes higher taxes, more regulation and government spending and a more generous welfare state."
                ]
            },

            5: {  # Center
                'description': "Center. Balances market freedom with government intervention for stability and fairness.",
                'interpretation': "Center",
                'primary': [
                    "Supports a mixed economy with both private enterprise and public services."
                ]
            },

            10: {  # Right
                'description': "Right. Emphasizes a reduced economic role for government",
                'interpretation': "Right",
                'primary': [
                    "Includes privatization, lower taxes, less regulation, less government spending, and a leaner welfare state"
                ]
            }
        }
        
        
        self.topic_threshold = 0.5
        print("Rile scorer initialized (0-10 scale)")

    def _find_entailment_index(self):
        """Auto-detect entailment index"""
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
        """Get probabilities for each response option with smoothing"""
        probs = {}
        for score in self.response_options.keys():
            response_text = self.response_options[score]['description']
            if 'primary' in self.response_options[score]:
                primary_text = " ".join(self.response_options[score]['primary'][:2])
                response_text += " " + primary_text
            
            prob = self._get_entailment_prob(text, response_text)
            # Add small smoothing to avoid extreme probabilities
            probs[score] = prob + 0.01  # Smoothing factor
            
        return probs

    def compute_confidence(self, probs_dict):
        """Compute confidence with better normalization"""
        probs = np.array(list(probs_dict.values()))
        normalized_probs = probs / np.sum(probs)
        
        # Use normalized maximum probability as confidence
        max_prob = np.max(normalized_probs)
        # Add entropy-based adjustment for distribution quality
        entropy = -np.sum(normalized_probs * np.log(normalized_probs + 1e-10))
        normalized_entropy = entropy / np.log(len(probs))
        
        confidence = max_prob * (1 - normalized_entropy)
        return min(1.0, max(0.1, confidence))  # Keep within reasonable bounds

    def _calculate_rile_score(self, probs_dict):
        """Calculate score with temperature scaling to reduce extremes"""
        scores = np.array(list(probs_dict.keys()))
        probs = np.array(list(probs_dict.values()))
        
        # Apply temperature scaling to soften probabilities
        temperature = 2.0  # Higher temperature = softer distribution
        scaled_probs = np.exp(np.log(probs + 1e-10) / temperature)
        normalized_probs = scaled_probs / np.sum(scaled_probs)
        
        weighted_score = np.sum(scores * normalized_probs)
        final_score = max(0, min(10, weighted_score))
        
        return round(final_score, 2)

    def _get_interpretation_from_score(self, score):
        """Get interpretation based on rile (2020) scale"""
        if score < 1.43:
            return "Far-left"
        elif score < 2.86:
            return "Left"
        elif score < 4.29:
            return "Center-left"
        elif score < 5.71:
            return "Center"
        elif score < 7.14:
            return "Center-right"
        elif score < 8.57:
            return "Right"
        else:
            return "Far-right"

    def score_left_right(self, text):
        """Main scoring method with improved calibration"""
        is_about_topic, topic_prob = self.is_about_political_rhetoric(text)
        
        if not is_about_topic:
            return {
                'score': 'NA',  # Keep 'NA' for irrelevant texts
                'confidence': 0.3,  # Low confidence for irrelevant text
                'interpretation': "Not about political rhetoric or institutional legitimacy",
                'is_relevant': False,
                'topic_probability': topic_prob,
                'framework': "RILE"
            }

        probs_dict = self.get_response_probabilities(text)
        score = self._calculate_rile_score(probs_dict)
        confidence = self.compute_confidence(probs_dict)
        interpretation = self._get_interpretation_from_score(score)

        return {
            'score': score,
            'confidence': round(confidence, 3),
            'interpretation': interpretation,
            'is_relevant': True,
            'topic_probability': topic_prob,
            'framework': "RILE",
            'category_probs': {k: round(v, 3) for k, v in probs_dict.items()}
        }




# Test function with rile-specific examples
def test_rile_scorer():
    scorer = PopulismPluralismResponsesScorer()
    
    # Test texts aligned with rile framework
    test_texts = [
" Our party is also a proud part of a growing international left that has for years criticized politicians' blind faith in the market and acted against its consequences.  ", # 1.518518519 Extreme-left 11220_202209
" Public money, including pension reserves and retirement funds, must be immediately withdrawn from investments in fossil fuel companies.  ", # 1.578947368 Extreme-left 41223_201709
" Second, banks need to be limited to the role of collecting capital, providing safe, understandable and sustainable savings opportunities for savers, rather than engaging in risky transactions with their and others' money.  ", # 1.578947368 Extreme-left 41223_201309
" Track access charges for passenger transport must be cut by at least half to get more traffic onto the railroads and make rail travel cheaper.  ", # 1.578947368 Extreme-left 41223_202109
" The policy of social cuts, deregulation and privatization, the unleashed financial markets, the one-sided focus on exports and the neglect of purchasing power and domestic markets - these policies serve the profit of a few and happen on the back and at the expense of the majority of the population.  ", # 1.578947368 Extreme-left 41223_200909
" To enable consumers to jointly defend themselves against corporate trickery and enrichment at consumers' expense, we are campaigning for class actions that lead directly to compensation from companies.  ", # 1.578947368 Extreme-left 41223_202109
" The Tenancy Law Amendment Act dismantles tenants' rights and shifts the costs of energy-efficient renovation of residential buildings onto tenants.  ", # 1.578947368 Extreme-left 41223_201309
" The principle that public health and health care in general cannot be subjected to the demands of the market.  ", # 1.620689655 Extreme-left 82220_200606
" In the event of bankruptcy of companies, to satisfy the wage demands of their employees as a priority.  ", # 1.620689655 Extreme-left 82220_201005
" This will help to curb house price growth and speculation, without disadvantaging people with a regular income and a single home.  ", # 1.8 Extreme-left 12221_201709
" SV will also propose other regulatory measures that make it less profitable to lock up large assets in real estate.  ", # 1.8 Extreme-left 12221_200509
" SV therefore wants to both build more homes at a livable price and slow down the rapid rise in prices by making the housing market less lucrative for investors.  ", # 1.8 Extreme-left 12221_201309
" In several of the major cities, real estate speculators leave homes empty and decaying for several years so that they can later be demolished.  ", # 1.8 Extreme-left 12221_200509
" To ensure increased local value creation, SV will propose the introduction of a locally determined area fee, differentiated by type of aquaculture, which goes to the municipalities as a replacement for the license fee.  ", # 1.8 Extreme-left 12221_200509
" This shall apply to all payments from, or on behalf of, Norwegian persons and companies and wholly or partly owned subsidiaries abroad.  ", # 1.8 Extreme-left 12221_200909
" A law on non-commercial housing must be drawn up that ensures that all subsidies and rents go towards operation, maintenance, upgrading and investment, and cannot be taken out in dividends.  ", # 1.8 Extreme-left 12221_201709
" This is why the Greens will establish a Shared Equity Ownership Scheme to allow people who are currently locked out of owning a home to buy up to 75% equity in a Federal Housing Trust home for $300,000.  ", # 1.857142857 Extreme-left 63110_202205
" Improve information and transparency in market control, involving consumer and user associations in the different Alert Networks, improving and coordinating their operation.  ", # 1.862068966 Extreme-left 33020_201512
" For a long time, capitalism has been developing into finance capitalism - into a development that Lenin already described in 1916 as the highest stage of capitalism: as imperialism.  ", # 0.75 Extreme-left 43220_201510
" Many of the world's poor live in resource-rich countries with a wealthy economic elite that does not invest its capital in local businesses.  ", # 1.8 Extreme-left 12221_200909
" In short, we have reversed the Neoliberal State that allowed the concentration of the hydrocarbon surplus in a few hands, including foreign hands.  ", # 1.75 Extreme-left 151250_200912
" Currently, our economic model takes care of stability, but as a social patrimony, understanding that when there is an economic crisis, it always affects the poorest.  ", # 1.75 Extreme-left 151250_201410
" This means the socialization of the essential sectors of the economy (especially banking and insurance).  ", # 0.75 Extreme-left 43220_201510
" He said clearly: Bankrupt the plutocracy, which has exploded with profits and is responsible for the crisis, not the people.  ", # 0.285714286 Extreme-left 34210_201205
" The social irruption that gave rise to the Democratic and Cultural Revolution in 2006 is the result of an accumulation of the struggles of our peoples against the European invader and against the exclusion experienced in the different cycles of the Republic in the hands of the oligarchy.  ", # 1.75 Extreme-left 151250_201410
" The first factor explains the extent of the coloniality of power, the second the absence of state sovereignty and the third an economy for the benefit of external capital.  ", # 1.75 Extreme-left 151250_201410
" We will change the criteria a company must meet to be listed on the London Stock Exchange so that any company that fails to contribute to tackling the climate and environmental emergency is delisted.  ", # 2.125 Left 51320_201912
" The aim is to provide every citizen with a domain name to enable them to benefit from new services, avoid the inconvenience of changing address, and encourage them to create content on the Internet.  ", # 2.25 Left 21322_200706
" On the other hand, if the State can call on European funds to provide the guarantee, it becomes much more solid and credible.  ", # 2.25 Left 21322_201905
" Make the entire pharmaceutical industry accountable for respecting the budget allocated to medicines.  ", # 2.25 Left 21322_201905
" In modern society, information is fundamental for political work; those who have and control it can exercise immense control over society, allowing them to disorient and manipulate it.  ", # 2.272727273 Left 171306_200306
" We have reiterated that only the Federal Government has the capacity to solve the workers' housing problem and that if we continue to insist on policies that privilege the market, we will be condemning millions of Mexican families to live in overcrowded conditions and spending more than fifty percent of their income on rent, if this serious social problem is not reversed.  ", # 2.272727273 Left 171306_200907
" To avoid excessive drug prices, the structure of drug prices will be made public, and where possible, well-copper, unbranded drugs will be used.  ", # 2.285714286 Left 22951_202103
" We support Austrian green energy companies by introducing a new Green Electricity Act modeled on the German Renewable Energy Act.  ", # 2.636363636 Left 42110_200809
" Any person receiving direct support from a pharmaceutical company for any purpose should be declared to be incompatible with the practice of medicine.  ", # 2.666666667 Left 86110_201004
" Therefore, we require companies with more than 100 employees to implement a profit-sharing scheme where all employees receive the same amount.  ", # 2.695652174 Left 22110_202103
" The obligation for companies and institutions to make all investments that pay for themselves within five years will be retained as a minimum; preferably it will be supplemented by a roadmap to climate-neutral operations in 2050 for every major energy user.  ", # 2.695652174 Left 22110_201703
" Norwegians who want to invest and own property in Norway should not have worse conditions than foreign or state-owned owners.  ", # 2.857142857 Left 12420_201309
" Help formalise township and village-based enterprises through an active campaign by provincial and local governments that promotes the benefits of formalisation.  ", # 3.052631579 Left 181310_201905
" But for the two thirds of households who don’t switch either supplier or tariffs – often those on the lowest incomes, older people, and those without access to the internet – that’s just not the case.  ", # 3.162162162 Left 51902_201912
" Based on this information, government is better armed to judge whether the price that firms charge is permissible.  ", # 3.181818182 Left 21112_201405
" The problem is that if the business banking department falls on hard times, the government is still obliged to intervene to prevent the savings department from being dragged along.  ", # 3.181818182 Left 21112_201405
" The derailed favoritism of company cars and fuel cards not soberly used for business travel costs society 4.1 billion euros annually.  ", # 3.181818182 Left 21112_201405
" Thus, even these products derived from animals that were fed GMOs will be clearly identifiable to consumers.  ", # 3.181818182 Left 21112_201405
" Therefore: strict and intensive tax audits at the Austrian subsidiaries of large foreign corporations.  ", # 3.368421053 Left 42320_201710
" We fundamentally oppose the efforts of the current federal government to worsen consumer protection in all areas.  ", # 3.368421053 Left 42320_200211
" Agriculture, like the furniture and textile industries, will not be able to avoid the problems caused by liberalization: loss of markets, falling prices, fierce competition and relocation of factories.  ", # 3.416666667 Left 62901_200810
" Recognise that the reliance on advertising revenue for online businesses has led to the exploitation of individuals data on a large scale.  ", # 3.4375 Left 53321_202002
" Labor’s Better Deal for Small Business will: Guarantee that an Albanese Labor Government will consider the specific needs of small businesses in times of crisis, giving the confidence and certainty to grow and plan for the future.  ", # 3.5 Left 63320_202205
" That is why there will be an ambitious investment agenda with additional investments in public housing, education and a sustainable economy.  ", # 3.2 Left 22320_202103
" Able to realize that the greatness and prosperity of the Fatherland is possible only on the path of socialist development.  ", # 2.5 Left 94221_200712
" We will aim to ensure that public investments in education, health, social housing development, etc., which pay off in the medium or longer term and boost the economy, are not included in the assessment of the budget deficit.  ", # 2.7 Left 88320_202010
" The one-sided austerity drive across Europe threatens the jobs and purchasing power of workers, the self-employed and pensioners and leads to contraction.  ", # 3.2 Left 22320_201209
" The lower income sectors and the productive sphere of goods of generalized consumption and with large internal multiplier effects, should not only be tax deducted, but encouraged and supported through subsidies and public spending.  ", # 2.272727273 Left 171306_200907
" Wealthy people and director-major shareholders now do not pay premiums for many collective services such as healthcare (WLZ and ZVW), even though they are entitled to them.  ", # 3.2 Left 22320_201703
" Establish market surveillance: It is necessary to clearly establish the responsibilities of the different agents involved in the marketing of used automotive parts, in order to protect citizens from practices that could jeopardize their safety at the wheel.  ", # 4 Center 33320_201606
" Review the regulation of the termination of rental contracts based on delays or non-payment of rent, guaranteeing the rights of the lessor to receive the agreed rent and the protection of the lessee, and also articulating mediation systems for cases of supervening insolvency.  ", # 4 Center 33320_201606
" We will promote legislative modifications to improve the security and guarantees of the landlord and the tenant.  ", # 4 Center 33320_200803
" Promote legislative changes to extend the current legal warranty of 2 years on products and adapt its duration to the useful life of each product.  ", # 4 Center 33320_201512
" To defend the freedom of choice of software in Public Administrations, promoting the use of open software and avoiding the imposition of certain types of software over others that limit the capacity of Public Administrations to adapt to the changing needs of the environment.  ", # 4 Center 33320_201606
" Support the European Commission's proposal to amend the Directive on administrative cooperation in the field of taxation, to end the opacity of the so-called Tax Rulings (tax agreements of member states with multinational companies or also called tailor-made suits) and cross-border transfer pricing agreements, issued in the last ten years.  ", # 4 Center 33320_201512
" We propose the unification of the complaints and protection services of the three financial supervisors (banking, securities, insurance) to which customers and investors can turn to in a single Financial Protection and Dispute Resolution Authority.  ", # 4 Center 33320_201512
" Establishment of the Economic and Social Advisory Council, with the active presence of the social economy sector, as a substantive part of the exercise of a participatory and citizen-based government.  ", # 4 Center 153331_201002
" In relation to the concentration of the media in large media groups and their economic dependence, decisive action by the public authorities is required, with the effective constitution of the State Council of Audiovisual Media, which contributes to the protection of small business or community projects, as well as the regulation of conflicts of interest.  ", # 4 Center 33320_201512
" The bursting of the real estate bubble since 2008 has not affected everyone equally: while new forms of business have arisen through a new form of speculation in an indispensable asset such as housing, which has led to thousands of lifeless buildings being built all over the country, there are thousands of families facing a sudden and unintentional over-indebtedness that all too often ends in evictions, evictions and the dragging of debts that prevent them from rebuilding their lives.  ", # 4 Center 33320_201512
" Reform the second chance law to, on the one hand, allow the judge to paralyze foreclosure in the event of justified non-deliberate insolvency and, in the event that this is impossible, to agree to the application of the dation in payment, valuing the property at the price established for the granting of the loan.  ", # 4 Center 33320_201606
" In addition, it is necessary to ensure the best possible functioning of competition and regulatory bodies, as these are key issues for promoting competition in the markets.  ", # 4 Center 33320_200803
" That starts with lower taxes for citizens and a large portion of SMEs: The PVV wants to reduce the tax rate in the second bracket by 2 percentage points, including for AOWs.  ", # 4.090909091 Center 22722_201006
" We will continue the overhaul we have started to the range of legislation that governs the regulation of health professionals, including doctors, pharmacists, health and social care professionals and nurses.  ", # 4.166666667 Center 53420_200705
" Provide for the imposition of administrative fines by the Competition Authority where anti-competitive activity or abuse of dominant position is established on the balance of probabilities.  ", # 4.166666667 Center 53420_200705
" This ensures that an employee does not enter unemployment and this also ensures that an employer continues to feel responsible for the employee they are firing.  ", # 4.230769231 Center 22526_201209
" On the other hand, the taxman is creaming off fewer and fewer profits, which can instead be used to reinvest in public services, such as infrastructure and education, on which businesses also rely heavily.  ", # 4.230769231 Center 22526_202103
" Companies will be required to notify the court in a timely manner in the event of impending insolvency so that an administrator with powers similar to those of a receiver can be appointed.  ", # 4.230769231 Center 22526_202103
" Independent contractors are professional, flexible and innovative and use these competencies to gain their place in the market.  ", # 4.230769231 Center 22526_201209
" Promote inclusive economic growth with fiscal and spending policies that serve as levers for productive activity in the interest of sustainable economic development inspired by the principles of a social market economy oriented towards growth with equity.  ", # 4.230769231 Center 171311_201807
" This was the original recommendation of the Treasury that was ignored by National and will capture more people who are flipping investment properties for capital gain.  ", # 4.285714286 Center 64320_201709
" For us, the criterion for the state to remain or withdraw from the markets is one and only one: ensuring conditions that guarantee quality and affordable services to all citizens, whether provided by public or private operators.  ", # 4.8 Center 34313_200709
" Pharmaceutical industryStrengthening the sector by promoting research and development activities in the production of generic and other export-oriented preparations.  ", # 4.8 Center 34313_201509
" The Competition Commission has been significantly strengthened in terms of infrastructure (its own premises) and scientific staff (transfers and new posts).  ", # 4.8 Center 34313_200403
" A private individual can only be a majority owner or officer in more than three companies if the companies he/she has previously set up have been free of public debt for at least 2 years.  ", # 5.444444444 Center 86221_201804
" The situation on the labor market can only improve permanently if Luxembourg's competitiveness is restored and public finances are brought into balance.  ", # 5.5 Center 23951_201310
" Market transactions are a form of arrangement that allows individuals to make decisions freely and in accordance with their needs and interests.  ", # 5.75 Center 171309_200907
" This review will be designed to ensure that the existing regulatory regime is operating efficiently, is balancing the needs of users with the requirements of producers and is not imposing excessive costs on the economy.  ", # 5.833333333 Center 53620_200705
" It is time for policy makers to develop a national energy policy which is low in cost, is respectful of local communities and their ability to contribute to our renewable energy future and which secures our energy supply from external energy shocks.  ", # 5.833333333 Center 53620_201602
" Especially in times of crisis, it is important to maintain purchasing power and not to reduce it by increasing the tax burden.  ", # 5.5 Center 23951_200906
" Following liberal arguments, the contraction of the State has been favored; however, the worsening of the social problems described above make it indispensable to act against poverty from different spheres without losing sight of the sense of integrality and complementarity.  ", # 5.75 Center 171611_201506
" The goal is to maintain a diverse fleet, with large ocean-going vessels that can harvest resources from areas inaccessible to smaller vessels, and small vessels that harvest resources close to the coast and deliver high-quality, local food with less environmental impact.  ", # 6 Right 12520_201709
" In the case of sports facilities and archaeological sites, and always in partnership with private individuals (creation of joint stock companies, with the state holding 49% of the shares), to entrust the management to private individuals for 99 years (in the form of a lease).  ", # 6.333333333 Right 34410_201509
" Many small and family businesses find it difficult to attract passive equity investment to enable them to grow without taking on additional debt or giving up control of their business.  ", # 7.1 Right 63810_201905
" The current economic and financial crisis has shaken up our certainties and shown that we need to think about a real strategy for job growth and ensure that the potential of businesses is better exploited in the service of sustainable development.  ", # 7.25 Right 21426_201006
" Reduce the number of operators in the water sector, while maintaining competitiveness through effective benchmarking.  ", # 7.25 Right 21426_200706
" Without waiting for the latter, the reforming ministers acted within their remit, attentive to consumer expectations but also to the development of commerce, particularly local commerce.  ", # 7.25 Right 21426_201006
" In the period ahead, we will continue the strategy of increasing trade and investment relations with neighboring and neighboring countries, as well as with African, Asia Pacific, US and South American countries.  ", # 7.266666667 Right 74628_200707
" In this context, we will strengthen the registration system for the supply chain from production to the end consumer.  ", # 7.266666667 Right 74628_201806
" More and more government requirements and regulations for the construction of real estate on the other hand lead to a further shortage of supply on the real estate market.  ", # 7.333333333 Right 42520_201710
" In this way, they will grow, their contribution to employment will be immense and they will become new economic players that will halt the concentration of ownership and income.  ", # 7.333333333 Right 152622_200205
" Kept interest rates lower, and reduced income tax across the board to make home ownership the most affordable it’s been for a decade.  ", # 7.375 Right 64620_201111
" Prioritise the liberalisation of telecommunications, which will slash the time and cost involved in communicating with each other and the world.  ", # 7.466666667 Right 181411_200904
" Therefore, the Conservative Party believes that municipalities, regions or the state must not impose additional restrictions on the ability to build or buy one's own home.  ", # 7.6 Right 12620_201709
" Give Australians over the age of 55 the ability to invest up to $300,000, per person, in their superannuation fund outside of the existing contribution caps, from the proceeds of selling their primary residence.  ", # 7.636363636 Right 63621_202205
" Deliver Central Queenslanders the largest personal income tax relief since the Howard Government, with tax relief of up to $1,080 for single income families earning up to $126,000.  ", # 7.636363636 Right 63621_201905
" One of our aims in reforming bankruptcy law is to provide better protection for entrepreneurs and employees in bankruptcy, and to work with companies to develop effective solutions quickly, with the help of an effective early warning system.  ", # 7.666666667 Right 23420_201310
" Simplification and efficiency, with penalties in the event of serious or continued non-compliance of up to 1% of the offender's profits until greater transparency is achieved among the agents involved.  ", # 7.724637681 Right 33420_201512
" Some principles must be fully accepted in regulatory activity: predictability in objectives and time horizons, well-defined competencies, reduction of discretionality, transparency in decision-making processes, transparent regulatory dialogue between authorities and stakeholders, flexibility, agility, minimum intervention and full respect for market mechanisms.  ", # 7.783783784 Right 33610_201111
" Labor’s housing and investment taxes will end negative gearing as we know it and increase capital gains tax by 50%.  ", # 7.818181818 Right 63620_201905
" The Government’s strong fiscal management means that it can deliver surpluses while also rewarding hard-working Australians.  ", # 7.818181818 Right 63620_201905
" The assets are usually taxed several times, and the wealth tax discriminates against Norwegian ownership, as few other countries in the world have such a tax.  ", # 7.888888889 Right 12951_201309
" In other words, there must be no restrictions on activities that are intended to increase access to and use of the nature parks, but which do not cause permanent damage to nature.  ", # 7.888888889 Right 12951_200509
" We will create a Sovereign Wealth Fund for the North of England, so that the shale gas resources of the North are used to invest in the future of the North.  ", # 7.916666667 Right 51620_201505
" We work to let Danes keep more of their own money and to make it more worthwhile for everyone to get up in the morning and go to work.  ", # 7 Right 13420_201906
" They had to shoulder higher burdens from the state in addition to high heating and housing costs and interest on loans.  ", # 7.666666667 Right 23420_200906
" The inheritance tax should be reviewed with regard to its administrability and the cost-benefit ratio that arises in this context for its levying.  ", # 8 Extreme-right 41420_202109
" The Postal Service will need to be ready for the entry of competitors into the Belgian market while constantly adapting to changing market conditions.  ", # 8 Extreme-right 21521_201006
" With such planned paternalism, we deprive ourselves of the necessary openness to technical progress.  ", # 8 Extreme-right 41420_201709
" In this context, private-sector solutions for transport infrastructure can be a model that should also be applied nationwide.  ", # 8 Extreme-right 41420_200209
" Competition is the key driver of dynamic economic development and thus of innovation and investment.  ", # 8 Extreme-right 41420_200509
" In order to improve productivity, technical assistance schemes with the intervention of the private sector will be defined together with the credit policy.  ", # 8 Extreme-right 154521_200210
" The recent price crises in the industry have also shown that we must once again ensure fair competition and market-based pricing throughout the food retail value chain.  ", # 8 Extreme-right 41420_201709
" The use of the market-based price mechanism ensures that economic dynamism and technical progress reveal potential for cost reduction and also create opportunities for new jobs. .  ", # 8 Extreme-right 41420_200209
" Instead of reducing the burdens to a minimum, however, energy sources are now being misused as a welcome source of tax revenue.  ", # 8.111111111 Extreme-right 43810_201510
" Since 2009, the EPA has moved forward with expansive regulations that will impose tens of billions of dollars in new costs on American businesses and consumers.  ", # 8.225 Extreme-right 61620_201211
" Our core problems arise in the nationalised sections of the industry and in the distortions caused by dominant health funding and regulation.  ", # 8.3 Extreme-right 64420_200207
" Advocated for an end to corporate welfare, arguing that Government should cut tax for all businesses instead of picking winners.  ", # 8.3 Extreme-right 64420_201709
" We would require every regulator to publicly justify every major regulation already on their books and quantify its costs and benefits.  ", # 8.3 Extreme-right 64420_202010
" We advocate a system of commercial interdependence that allows all parties to obtain the greatest benefit from their productive advantages.  ", # 8.857142857 Extreme-right 153420_200602
" We want to restore the balance between private and state and therefore end the excessive redistribution from private to state.  ", # 8 Extreme-right 41420_201709
" For example, abolishing inheritance tax and the so-called 'luxury real estate' tax will not substantially reduce the budget's revenue, and will also save the state money on tax administration." # 8.3 Extreme-right 88450_201610 
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