# Coding rules for the senses of *pueblo*

`pueblo_senses.csv` trains the classifier in `ltg/senses.py`. Every row so far
was coded by Claude (`coder = claude`), from the sentence around the word, to get
the classifier started. These are not human gold codes. Re-code them, or add your
own rows with your name in `coder`, and report agreement on the separate random
sample written by `python -m ltg sample` (`validation/sense_sample.csv`).

Code the sense of the one word in `[[...]]` / `keyword`, from its sentence.

| label | the word refers to | examples from the books |
|---|---|---|
| `settlement` | a place: village, town, hometown, a town's population as a place's residents | "la plaza de su pueblo", "de un pueblo a otro", "pueblos pequeños y grandes ciudades", "el pueblo de San Agustín Oapan", "pueblo mágico", "la gente del pueblo y de todo México se reuniera" (townspeople) |
| `people_commons` | the people as the governed, the common people set against rulers or elites, or the sovereign demos | "el pueblo los obedecía ciegamente y carecía de derechos", "los macehualtin, el resto del pueblo", "el pueblo elige", "la voluntad del pueblo", "se levantó en armas contra el gobierno", "tenía el apoyo del pueblo", "pueblos despojados de sus terrenos" (against hacendados) |
| `people_nation` | a nation or people as a collective identity, or peoples in general, with no ruler / ruled relation | "el pueblo mexicano estaba unido", "los pueblos del mundo", "autodeterminación de los pueblos", "el pueblo luchaba por expulsar a los invasores franceses" |
| `people_ethnic` | an ethnic, cultural, ancient or indigenous people | "el pueblo hebreo", "los pueblos originarios", "pueblos indígenas", "el pueblo ralámuli", "los pueblos antiguos", "pueblos belicosos", "los pueblos de indios" (as polities) |

Boundary decisions, so that they are applied the same way every time:

1. **Commons vs nation.** *Commons* needs a relation to rulers, government,
   elites or popular sovereignty (elige, autoridad, voluntad, apoyo, descontento,
   levantarse contra, obedecer, explotar). Without one, *el pueblo mexicano* is
   *nation*.
2. **Corporate rural or indigenous communities** (colonial and agrarian history:
   "muchos pueblos perdieron sus propiedades", "las tierras y agua de los
   pueblos"). *Ethnic* when the text marks them as indigenous (*originarios,
   indígenas, de indios*, an ethnonym); *commons* when framed against hacendados,
   companies or the Crown without an ethnic marker; *settlement* when they are
   administrative places ("los cabildos de los pueblos", "la caja de los pueblos").
3. **Conquered or allied peoples** ("conquistar muchos pueblos", "pueblos
   aliados", "pueblos sometidos") are *ethnic*.
4. **Legends and stories** ("¿qué podría yo hacer para que mi pueblo no
   sufriera?") follow the story's meaning; if it cannot be told, leave the row out.
5. **Leave out** OCR debris, word lists, picture credits and cases you cannot
   decide (do not guess: the classifier learns the guess).
