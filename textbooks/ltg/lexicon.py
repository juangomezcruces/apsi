"""Dictionaries for 'the people' and its relations, in Spanish.

Everything here is a research decision and should be read, argued with and
edited. Matching is on accent-folded, lower-cased lemmas (OCR drops accents),
so write entries without accents. Avoid words with a common
non-political sense in textbooks: 'unidad' (a book unit), 'diferencia'
(subtraction), 'plural' (grammar) and 'masa' (physics) were left out for that
reason; 'poblacion' and 'sociedad' stay, and the term-level tables show what
each one contributes. An entry `lemma/NOUN` only
counts when spaCy tags the token with that part of speech (`mexicano` is far
more often an adjective than 'a Mexican'). Multi-word entries are matched on
folded word forms, not lemmas, so list the inflections you want.

Families with people=True make up 'people talk'. The rest are measured inside
people-talk sentences (antagonists, grievance, unity, pluralism, institutions).
"""

FAMILIES = {
    # --- who the people are ----------------------------------------------
    "pueblo": dict(
        people=True, label="pueblo / pueblos",
        terms=["pueblo"],
        note="sense-tagged: people_political / people_ethnic / settlement / ambiguous; "
             "settlement uses are excluded from people counts"),
    "nacion": dict(
        people=True, label="nation, patria, 'los mexicanos'",
        terms=["nacion", "patria", "compatriota", "connacional", "mexicano/NOUN", "mexicana/NOUN",
               "mexicanas y mexicanos", "mexicanos y mexicanas"]),
    "ciudadania": dict(
        people=True, label="citizens, society, population",
        terms=["ciudadano", "ciudadana", "ciudadania", "sociedad", "poblacion", "habitante",
               "votante", "elector", "sociedad civil"]),
    "comunidad": dict(
        people=True, label="community, collective, assembly",
        terms=["comunidad", "colectivo/NOUN", "colectividad", "vecino/NOUN", "vecina/NOUN",
               "asamblea", "tequio", "comunalidad"]),
    "popular": dict(
        people=True, label="ordinary / popular / the poor",
        terms=["gente", "las masas", "multitud", "humilde/NOUN", "pobre/NOUN", "marginado/NOUN",
               "oprimido/NOUN", "explotado/NOUN", "desposeido/NOUN",
               "clases populares", "clase popular", "sectores populares", "las mayorias",
               "los de abajo", "gente comun", "gente humilde"]),
    "clase": dict(
        people=True, label="workers, peasants, class",
        terms=["trabajador/NOUN", "trabajadora/NOUN", "obrero", "obrera", "campesino", "campesina",
               "jornalero", "jornalera", "peon", "artesano", "artesana", "minero/NOUN",
               "proletariado", "proletario", "clase obrera", "clase trabajadora", "clase campesina"]),
    "etnico": dict(
        people=True, label="indigenous, Afro-Mexican, ethnic groups",
        terms=["indigena/NOUN", "indio/NOUN", "india/NOUN", "afromexicano", "afromexicana",
               "afrodescendiente", "mestizo/NOUN", "mestiza/NOUN", "criollo/NOUN",
               "pueblos originarios", "pueblos indigenas", "comunidades indigenas",
               "pueblo originario", "pueblo indigena", "pueblos afromexicanos",
               # ethnonyms: present-day peoples, then ancient civilisations
               "maya", "nahua", "mixteco", "zapoteco", "otomi", "totonaco", "purepecha",
               "tarasco", "raramuri", "tarahumara", "yaqui", "yoreme", "mazahua", "mazateco",
               "tzotzil", "tzeltal", "chol", "huichol", "wixarika", "triqui", "chinanteco",
               "mixe", "seri", "comcaac", "huasteco", "tenek", "chontal", "tojolabal", "lacandon",
               "mexica", "azteca", "olmeca", "tolteca", "teotihuacano", "chichimeca", "tlaxcalteca"]),
    # Generic humans. NEM books use 'las personas' as the gender-inclusive
    # generic, so this is reported for context but kept out of people-talk.
    "personas": dict(people=False, label="personas (generic)", terms=["persona"]),
    "nosotros": dict(people=False, label="inclusive we", terms=["nosotros", "nosotras", "nuestro", "nuestra"]),

    # --- relations, measured within people-talk ---------------------------
    "elite": dict(
        people=False, label="elites and antagonists",
        terms=["elite", "oligarquia", "oligarca", "poderoso/NOUN", "rico/NOUN", "privilegiado/NOUN",
               "privilegio", "hacendado", "latifundista", "terrateniente", "cacique", "conquistador",
               "colonizador", "invasor", "encomendero", "imperialismo", "imperialista", "capitalista/NOUN",
               "burguesia", "burgues", "monopolio", "corrupcion", "corrupto", "explotador", "opresor",
               "neoliberalismo", "neoliberal", "dictadura", "dictador", "tirano", "tirania",
               "transnacional", "grupos de poder", "los de arriba", "intereses extranjeros",
               "empresas extranjeras", "los poderosos", "los ricos"]),
    "agravio": dict(
        people=False, label="grievance",
        terms=["explotacion", "opresion", "despojo", "injusticia", "desigualdad", "discriminacion",
               "pobreza", "marginacion", "saqueo", "abuso", "racismo", "colonialismo", "esclavitud",
               "sometimiento", "exclusion", "miseria"]),
    "unidad": dict(
        people=False, label="unity / homogeneity",
        terms=["todo el pueblo", "pueblo unido", "unidad nacional", "un solo pueblo",
               "voluntad del pueblo", "voluntad popular", "soberania popular", "soberania nacional",
               "pueblo bueno", "pueblo sabio", "como un solo hombre", "todos los mexicanos",
               "todas y todos", "el pueblo quiere", "el pueblo decide", "el pueblo sabe"]),
    "pluralismo": dict(
        people=False, label="pluralism / diversity",
        terms=["diversidad", "diverso", "pluralidad", "pluralismo", "multicultural",
               "multiculturalidad", "intercultural", "interculturalidad", "pluricultural",
               "pluriculturalidad", "plurilingue", "minoria", "tolerancia", "dialogo",
               "consenso", "derechos humanos", "puntos de vista", "distintas opiniones",
               "diferentes opiniones"]),
    "instituciones": dict(
        people=False, label="liberal-democratic institutions",
        terms=["constitucion", "ley", "eleccion", "votar", "voto", "congreso", "diputado", "senador",
               "suprema corte", "juez", "tribunal", "institucion", "division de poderes", "democracia",
               "democratico", "partido politico", "partidos politicos", "estado de derecho"]),
}

PEOPLE_FAMILIES = [k for k, v in FAMILIES.items() if v["people"]]

# --- what the people DO: verb lemma -> class --------------------------------
VERB_CLASSES = {
    "struggle": ["luchar", "combatir", "pelear", "rebelar", "levantar", "sublevar", "resistir",
                 "defender", "exigir", "protestar", "reclamar", "enfrentar", "oponer", "derrocar",
                 "liberar", "independizar", "conquistar", "vencer", "expulsar", "triunfar", "manifestar"],
    "participation": ["votar", "elegir", "decidir", "participar", "opinar", "acordar", "discutir",
                      "dialogar", "organizar", "deliberar", "proponer", "consultar", "colaborar",
                      "cooperar", "reunir", "asociar"],
    "work": ["trabajar", "construir", "producir", "cultivar", "sembrar", "cosechar", "crear",
             "edificar", "fabricar", "elaborar", "desarrollar", "contribuir", "aportar", "fundar", "establecer"],
    "suffering": ["sufrir", "padecer", "morir", "perder", "carecer", "huir", "emigrar", "migrar"],
    "will": ["querer", "desear", "saber", "merecer", "mandar", "ejercer", "necesitar", "esperar", "anhelar"],
    "unity": ["unir", "solidarizar", "hermanar", "integrar", "agrupar"],
    # added after reading the unclassified agent verbs of the 1988 and 2022 books
    "submission": ["obedecer", "pagar", "servir", "entregar", "tributar", "aceptar", "soportar", "aguantar"],
    "rule": ["gobernar", "dirigir", "administrar", "controlar", "reinar"],
    "thought": ["pensar", "creer", "conocer", "aprender", "descubrir", "decir", "contar", "entender",
                "imaginar", "considerar"],
    "everyday": ["vivir", "habitar", "viajar", "llegar", "venir", "ir", "usar", "acostumbrar", "comprar",
                 "vender", "alimentar", "comer", "jugar", "transportar", "pescar", "cazar", "comerciar"],
    "culture": ["conservar", "celebrar", "hablar", "mantener", "preservar", "transmitir", "compartir",
                "festejar", "vestir", "heredar", "honrar", "recordar"],
}
# Verbs that make their OBJECT (or passive subject) a victim.
VICTIMISING = {"explotar", "oprimir", "despojar", "someter", "conquistar", "esclavizar", "discriminar",
               "marginar", "enganar", "reprimir", "masacrar", "matar", "perseguir", "excluir",
               "dominar", "sojuzgar", "humillar", "abusar", "robar", "saquear", "colonizar", "evangelizar"}

# --- 'pueblo' sense cues -----------------------------------------------------
PUEBLO_ETHNIC_MODS = {"indigena", "originario", "afromexicano", "afrodescendiente", "nativo", "autoctono",
                      "prehispanico", "mesoamericano", "antiguo"}
COUNTRIES = {"mexico", "francia", "espana", "cuba", "estados unidos", "guatemala", "chile", "argentina",
             "venezuela", "colombia", "peru", "bolivia", "nicaragua", "brasil", "inglaterra", "rusia",
             "china", "japon", "alemania", "italia", "palestina", "america", "latinoamerica", "mundo"}
PEOPLE_MODS = {"mexicano", "bueno", "sabio", "trabajador", "unido", "oprimido", "humilde", "soberano",
               "libre", "heroico", "valiente", "noble", "explotado", "pobre", "entero", "organizado"}
SETTLEMENT_MODS = {"pequeno", "grande", "cercano", "vecino", "lejano", "magico", "natal", "costero",
                   "minero", "fantasma", "rural", "abandonado", "tranquilo", "bonito"}
SETTLEMENT_COWORDS = {"ciudad", "rancho", "aldea", "rancheria", "villa", "calle", "plaza",
                      "mercado", "iglesia", "kiosco", "municipio", "carretera", "camino", "casa", "parque"}
SETTLEMENT_VERBS = {"vivir", "llegar", "ir", "salir", "viajar", "visitar", "habitar", "fundar",
                    "regresar", "volver", "caminar", "pasar", "recorrer", "mudar", "quedar", "estar"}
POLITICAL_HEADS = {"soberania", "voluntad", "voz", "derecho", "lucha", "representante", "enemigo",
                   "historia", "servicio", "bienestar", "poder", "gobierno", "mandato", "defensa"}
