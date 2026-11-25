"""
Shared configuration for the EU SEE "Civil Society under Pressure" YouTube pipeline.

Every script in this folder imports from here, so countries, keywords, the
study window, the classification prompts and file locations are defined once.

Edit the `queries` in COUNTRIES to match the exact keyword lists used in the study; the values
below are a sensible starting point in the languages named in the report
(Spanish, English, Arabic, French-Arabic, Sinhala).
"""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
BASE_DIR = Path(os.environ.get("EUSEE_BASE_DIR", Path(__file__).resolve().parent))
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"                 # raw API responses (json)
INTERIM_DIR = DATA_DIR / "interim"         # flat collected tables, transcripts
PROCESSED_DIR = DATA_DIR / "processed"     # classified datasets
AUDIO_DIR = DATA_DIR / "audio"             # temporary audio downloads
TRANSCRIPT_DIR = DATA_DIR / "transcripts"  # one json per video
OUTPUT_DIR = BASE_DIR / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"

# Canonical file names (each script reads the previous script's output)
COLLECTED_CSV = INTERIM_DIR / "youtube_videos_collected.csv"
TRANSCRIPTS_CSV = INTERIM_DIR / "youtube_transcripts.csv"
CLASSIFIED_CSV = PROCESSED_DIR / "youtube_classified_all.csv"   # every collected item + labels
MAIN_CSV = PROCESSED_DIR / "youtube_main_dataset.csv"           # on-topic, relevant items only

# --------------------------------------------------------------------------
# Study window
# --------------------------------------------------------------------------
START_DATE = date(2024, 1, 1)
END_DATE = date(2025, 12, 31)
EO_DATE = date(2025, 1, 20)   # Executive Order 14169, the analytical pivot

# --------------------------------------------------------------------------
# Countries and search keywords
# --------------------------------------------------------------------------
# region_code: ISO 3166-1 alpha-2 used for YouTube's regionCode and for the
#              channel-country filter.
# queries:     list of (relevanceLanguage, [keywords]) pairs. Keywords within one
#              language are OR-ed into a single search ("a|b|c") to save quota,
#              unless --no-combine is passed to the collector.
COUNTRIES: dict[str, dict] = {
    "Peru": {
        "region_code": "PE",
        "queries": [
            ("es", [
                "ONG", "ONGs", "organizaciones no gubernamentales",
                "sociedad civil", "APCI", "ley APCI",
                "cooperación internacional", "USAID",
                "fondos extranjeros ONG",
            ]),
        ],
    },
    "South Africa": {
        "region_code": "ZA",
        "queries": [
            ("en", [
                "NGO", "NGOs", "NPO", "non-profit organisation",
                "civil society", "USAID", "PEPFAR", "NPO Act",
                "foreign funded NGOs",
            ]),
        ],
    },
    "Lebanon": {
        "region_code": "LB",
        "queries": [
            ("ar", [
                "منظمات غير حكومية", "جمعيات", "المجتمع المدني",
                "الجمعيات الأهلية", "منظمات المجتمع المدني", "USAID",
            ]),
            ("fr", [
                "ONG", "société civile", "associations Liban", "USAID",
            ]),
        ],
    },
    "Sri Lanka": {
        "region_code": "LK",
        "queries": [
            ("en", [
                "NGO", "NGOs", "civil society", "USAID",
                "NGO Secretariat", "foreign funded NGOs",
            ]),
            ("si", [
                "රාජ්‍ය නොවන සංවිධාන", "සිවිල් සමාජ", "NGO",
            ]),
        ],
    },
}

COUNTRY_ORDER = ["Peru", "South Africa", "Lebanon", "Sri Lanka"]

# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"            # topics, toxicity, narratives
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"  # relevance
WHISPER_MODEL = "small"   # faster-whisper size; use "large-v3" on a GPU for best quality

# --------------------------------------------------------------------------
# Topics (Section 3.1, Table 3.1.1). Order = report order = colour order.
# Each value is the NLI hypothesis the model tests against the text.
# --------------------------------------------------------------------------
TOPICS: dict[str, str] = {
    "NGO Regulation":
        "This text is about rules, oversight, registration, audits or legal compliance "
        "governing NGOs or civil society organizations.",
    "NGO Funding":
        "This text is about the financing of NGOs or civil society organizations: "
        "grants, donors, budgets or funding cuts.",
    "Community Development & Social Assistance":
        "This text is about local service delivery, social assistance, charity or livelihoods.",
    "Social Inclusion and Human Rights":
        "This text is about human rights, inclusion, social cohesion or the protection "
        "of marginalized groups.",
    "Environment & Climate":
        "This text is about conservation, pollution, climate change or sustainability.",
    "Democracy, Elections & Anti-Corruption":
        "This text is about elections, democracy, governance or corruption.",
    "Women's Rights & Gender Equality":
        "This text is about gender equality or the protection and participation of women.",
    "Education & Youth":
        "This text is about schooling, education, youth empowerment or opportunities "
        "for young people.",
}
OFF_TOPIC = "Off-Topic"
OFF_TOPIC_HYPOTHESIS = (
    "This text is unrelated to NGOs, civil society, social issues or public affairs "
    "(for example entertainment, sport, music, advertising or gaming)."
)
# Tie-break rule (Section 7.3): top score < TOPIC_MIN_CONF and the runner-up within
# TOPIC_TIE_MARGIN -> Off-Topic instead of forcing a topic.
TOPIC_MIN_CONF = 0.25
TOPIC_TIE_MARGIN = 0.05

# --------------------------------------------------------------------------
# Toxicity (Section 7.4): two descriptions weighed directly against each other.
# --------------------------------------------------------------------------
TOXIC_HYPOTHESIS = (
    "This text is hateful, abusive, offensive, disrespectful or otherwise harmful in tone."
)
NON_TOXIC_HYPOTHESIS = (
    "This text is respectful or neutral in tone, even if it is critical or negative."
)

# --------------------------------------------------------------------------
# Negative narratives (Section 4.7 / 7.4). Run on the toxic subset only.
# Each hypothesis is scored independently (multi-label); highest wins.
# --------------------------------------------------------------------------
NARRATIVES: dict[str, str] = {
    "Mission / Ideological Drift":
        "This text questions whether an organization still serves its stated beneficiaries, "
        "or frames it as captured by ideological, political or foreign agendas.",
    "Human Rights Defense":
        "This text defends NGOs, human rights work or civil society, reacting against "
        "attempts to restrict or discredit them.",
    "Sovereignty-Transparency":
        "This text invokes transparency, neutral standards or national sovereignty to raise "
        "concern about foreign-funded organizations.",
    "Conspiratorial Delegitimization":
        "This text frames NGOs as fundamentally corrupt, criminal or parasitic.",
    "Technocratic-Regulatory":
        "This text takes a procedural, reform-minded view, diagnosing a structural or "
        "regulatory problem.",
}
NARRATIVE_OTHER = "Other"
# If no narrative's independent entailment score reaches this, the item goes to "Other".
NARRATIVE_MIN_SCORE = 0.10

# --------------------------------------------------------------------------
# Relevance filter (Section 7.2)
# --------------------------------------------------------------------------
# Anchor statements describing the civil-society conversation. A video is kept
# when its best cosine similarity to any anchor is >= RELEVANCE_THRESHOLD.
RELEVANCE_ANCHORS = list(TOPICS.values()) + [
    "Non-governmental organizations and civil society organizations.",
    "The US foreign aid freeze and USAID funding cuts affecting NGOs.",
    "Foreign agents, foreign-funded organizations and civic space restrictions.",
]
RELEVANCE_THRESHOLD = 0.30   # calibrate with `03_text_analysis.py --calibrate`

# --------------------------------------------------------------------------
# Spreader / account typing (Section 7.6): ordered, auditable keyword rules.
# First matching rule wins. Matched against the lower-cased channel title,
# custom URL/handle and channel description.
# --------------------------------------------------------------------------
SPREADER_RULES: list[tuple[str, list[str]]] = [
    ("Intergovernmental", [
        r"\bun\b", r"united nations", r"unicef", r"undp", r"unhcr", r"unfpa", r"ohchr",
        r"\bwho\b", r"world bank", r"european union", r"\beu\b", r"\bilo\b", r"\biom\b",
        r"\bwfp\b", r"\bfao\b", r"unesco", r"un women", r"naciones unidas", r"الأمم المتحدة",
    ]),
    ("Government", [
        r"ministry", r"ministerio", r"minist[eè]re", r"gobierno", r"government", r"\bgov\b",
        r"\.gov", r"presidencia", r"president", r"congreso", r"parliament", r"parlamento",
        r"municipalidad", r"municipality", r"department of", r"embassy", r"embajada",
        r"وزارة", r"الحكومة", r"مجلس النواب", r"රජය", r"අමාත්‍යාංශය",
    ]),
    ("News & Media", [
        r"news", r"\btv\b", r"television", r"televisi[oó]n", r"radio", r"\bfm\b", r"noticias",
        r"diario", r"prensa", r"peri[oó]dico", r"canal", r"media", r"times", r"post\b",
        r"journal", r"press", r"herald", r"tribune", r"broadcast", r"\blive\b", r"podcast",
        r"infobae", r"rpp", r"latina", r"exitosa", r"willax", r"panamericana", r"atv",
        r"sabc", r"enca", r"newzroom", r"news24", r"ewn", r"daily maverick",
        r"lbci", r"mtv lebanon", r"al jadeed", r"aljadeed", r"otv", r"l'orient", r"annahar",
        r"newsfirst", r"hiru", r"derana", r"sirasa", r"daily mirror", r"ada derana",
        r"قناة", r"أخبار", r"اخبار", r"تلفزيون", r"නිව්ස්", r"පුවත්",
    ]),
    ("Civil Society / NGO", [
        r"\bngo\b", r"\bong\b", r"\bnpo\b", r"foundation", r"fundaci[oó]n", r"fondation",
        r"association", r"asociaci[oó]n", r"trust\b", r"network", r"\bred\b", r"coalition",
        r"alliance", r"institute", r"instituto", r"collective", r"colectivo", r"society",
        r"charity", r"human rights", r"derechos humanos", r"amnesty", r"oxfam", r"care\b",
        r"جمعية", r"منظمة", r"مؤسسة", r"සංවිධානය", r"පදනම",
    ]),
]
SPREADER_INDIVIDUAL = "Individual"
SPREADER_OTHER = "Other / Unclassified"
