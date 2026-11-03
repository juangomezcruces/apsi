# Research design: how the free textbooks depict 'the people'

A working design for the paper, written alongside the pipeline. Everything
here is a proposal to argue with. Historical claims marked *(verify)* should be
checked against the books' colophons and the literature before they go into
the paper.

## Question

How do Mexico's primary-school free textbooks construct *el pueblo*: who it
is, whether it is one or many, what it does, whom it stands against, and how it
rules? And how did that construction change between the books in use in 1988,
the last pre-NEM catalog (2022), and the Nueva Escuela Mexicana (NEM) books
from 2023 onward?

## Corpus and comparison logic

| period | editions | notes |
|---|---|---|
| gen-1988 | 1988 | The archive's own "1988 generation": 26 primary titles, integrated *Mi libro de 1°/2°* for grades 1-2 and *Ciencias Sociales*, *Ciencias Naturales*, *Español* and *Matemáticas* for grades 3-6. The archive's year lookup has this generation in use in the late 1980s and early 1990s, before the 1993 books *(verify each book's edition history from its colophon; how much of it carries over from the 1972 books is a question for the paper, not an assumption)*. Scans; OCR confidence 81-95. |
| pre-NEM | 2022 | Subject-based books (Lengua Materna, Matemáticas, Conocimiento del Medio, Historia, Geografía, Formación Cívica y Ética...). |
| NEM | 2023, 2024, 2025, current | Organised by *campos formativos* and projects (*Proyectos de aula / escolares / comunitarios*, *Nuestros saberes*, *Múltiples lenguajes*...). Later editions largely reprint 2023 titles: treat them as one period with revisions, not as independent samples. |

Because the subject structure changed, report each comparison twice:
1. **Whole corpus** (what a child receives in total), and
2. **Social-content subset** matched by function: 1988 *Ciencias Sociales* /
   history / civics books; 2022 *Historia*, *Formación Cívica y Ética*,
   *Conocimiento del Medio*, *Geografía*; NEM *Proyectos comunitarios* and the
   ethics/society material. Fix the mapping in `catalog.csv` (`subject`) before
   looking at results.

The reprinted NEM titles also allow a within-period analysis: diff the same
book code across 2023, 2024 and 2025 to see what was revised.

## Concepts and measures

The people-centrism and anti-elitism of the ideational approach (Mudde 2004,
*Government and Opposition*), the homogeneity/anti-pluralism emphasis of
Müller (2016, *What Is Populism?*), and the Mexican shift from a revolutionary,
class-and-mestizo nation towards a constitutionally pluricultural one (Art. 4
reform of 1992, Art. 2 of 2001, Afro-Mexican recognition in 2019) suggest
eight dimensions. Each has a text and, where possible, a visual measure.

| dimension | question | text measure (`ltg report`) | visual measure (`ltg visual-report`) |
|---|---|---|---|
| Salience | How much do the books talk about the people? | `people_all` per 10k tokens | % of illustrations with people |
| Composition | Nation, class, ethnic group, citizens, community? | rates per family (`nacion`, `clase`, `etnico`, `ciudadania`, `comunidad`, `popular`); top terms; **senses of *pueblo***: common people vs rulers / nation / ethnic people (`senses`) | `figure` (traditional dress, ordinary adults, officials, soldiers, children); `setting` |
| Unity vs plurality | One people or many? | singular *el pueblo* vs plural *los pueblos*; `has_unidad` vs `has_pluralismo` in people-talk | crowd vs small group vs individual |
| Agency | Does the people act, and how? | role (agent / patient / attribute / possessor); verb class when agent (struggle, participation, work, unity, will, culture, suffering) | `activity`: protest, assembly, voting, collective work, armed struggle, festival |
| Antagonism | Against whom? | `has_elite`, `has_agravio` in people-talk; collocates of *pueblo* | hand-coded only (conquest, hacienda and battle scenes need context) |
| Moral valence | Is the people good? | adjectives and predicates of *pueblo* (*bueno, sabio, trabajador, humilde, valiente*) | — |
| Sovereignty | How does the people rule? | *voluntad / soberanía* phrases (`unidad`) vs `instituciones` in people-talk | voting vs assembly vs protest |
| Inclusion | Is the child-reader part of the people? | `has_nosotros` in people-talk | `figure = children` |

Candidate expectations, to pre-register or discard before looking at results:

- **E1** NEM books use more community and ethnic vocabulary and more plural
  *pueblos* than 2022; the 1988 books more class vocabulary (*campesinos,
  obreros*) and more singular national *pueblo*.
- **E2** People-talk co-occurs more with elite and grievance terms in NEM and
  1988 books than in 2022 (a people-vs-power frame), but against different
  antagonists (colonialism and neoliberalism vs hacendados and dictatorship).
- **E3** Agency differs by period: struggle in 1988, community participation
  and culture in NEM, institutional participation in 2022.
- **E4** Visually, NEM books show more collective and community scenes and
  more people in traditional dress; pre-NEM books more individuals and
  institutional settings.

Each comes with its alternative (no difference, or the opposite sign), and
the CIs decide.

## Validation (before any number goes in the paper)

| instrument | sample (`ltg sample` / `visual-sample`) | report |
|---|---|---|
| OCR | 20 random pages per edition, transcribed by hand | word error rate per edition; rates re-computed on high-confidence pages only |
| Sense of *pueblo*, *comunidad*, *asamblea* | up to 40 per edition x word x predicted sense | accuracy, per-sense precision/recall, kappa (`validate --kind sense`). Target precision >= .9 for `settlement` and `classroom`, since they are what gets excluded. The classifier's seed codes were made by Claude; replace them with your own before reporting |
| Role | up to 20 per edition x predicted role | as above (`--kind role`) |
| Visual codebook | up to 30 per edition x predicted people label, drawn from the analysis sample of illustrations | per dimension (`--kind visual --dim ...`); `people` is the detector's count, the rest CLIP's. Dimensions with poor agreement are reported from the hand-coded sample only, or prevalence is corrected for the measured misclassification |
| Inter-coder | second coder on ~20% of each sample | Cohen's kappa / Krippendorff's alpha |

Coders see the contact sheet without predictions (the HTML hides them) to
avoid anchoring.

## Analysis

1. **Descriptives by edition** with 95% intervals from resampling books
   (pages within a book are not independent). This is what `report` produces.
2. **Models**: page-level counts of people mentions, Poisson or negative
   binomial with an offset of log tokens, edition + subject + grade, standard
   errors clustered by book (or a book random effect). Sentence-level logits for
   the co-occurrence flags.
3. **Close reading**: KWIC lines and sentences quoted in the paper should be
   drawn at random within strata (edition x sense x role), not hand-picked,
   and the paper should say so.
4. **Text-image link**: captions that name the people; people-talk on pages
   with and without people images.
5. **Triangulation**: the APSI populism scorer on translated people-talk
   sentences (see README), as a second, independent measure of people-centrism
   and anti-elitism. It was validated on political speech, so treat its scores
   here as something to validate too.

## Threats to validity

- **Differential OCR error.** 1988 scans will be noisier than born-digital NEM
  books; misses lower rates. Check WER, compare on high-confidence pages.
- **Corpus composition.** Different numbers of books, subjects and pages per
  period; rates and the matched social-content subset address this, not fully.
- **Gender-inclusive language.** NEM's *las personas*, *niñas y niños* change
  generic-person frequencies; *personas* is kept out of people-talk for that
  reason and reported separately.
- **Lexicon choices.** Word lists encode the researcher's expectations. Fix
  them before looking at period differences, use the keyness tables to find
  what they miss, and run a leave-one-term-out sensitivity check on the main
  results.
- **Pedagogical vocabulary.** NEM books organise group work as *comunidades*
  and *asambleas* ("En asamblea, elaboren..."); counted naively this would
  inflate 'community' talk severalfold. These uses are tagged `classroom` and
  excluded; the excluded counts are reported so the decision is visible.
- **Polysemy.** *Pueblo* is also a town; *población* is also a statistical
  population; *comunidad* is also a learning community. The sense tagger and
  term-level tables make this visible; validate.
- **Parser and CLIP errors.** Both are measured, not assumed away. CLIP was
  trained on web photos with English captions; textbook drawings are out of
  domain.
- **Ethics.** No skin tone or race is inferred from pixels. Ethnic
  representation is coded by hand from visible cultural markers and context.

## Selected references (check details before citing)

- Mudde, C. (2004). The populist zeitgeist. *Government and Opposition*, 39(4).
- Müller, J.-W. (2016). *What Is Populism?* University of Pennsylvania Press.
- Laclau, E. (2005). *On Populist Reason*. Verso.
- Vázquez, J. Z. (1970). *Nacionalismo y educación en México*. El Colegio de México.
- Gilbert, D. (1997). Rewriting history: Salinas, Zedillo and the 1992
  textbook controversy. *Mexican Studies/Estudios Mexicanos*, 13(2).
