# Draw a representative, weak-label-stratified sample of paragraphs for human
# coding from Mochtak's two VALIDATION corpora (Maerz & Schneider; Internet
# Archive Corpus). UNGD is excluded on purpose: it is the model's training set,
# so comparing coders with model-8 there would compare against in-sample fits.
#
# Unit: a three-sentence paragraph = one of Mochtak's sentence trigrams. The
# trigrams slide one sentence at a time, so only every third one is eligible
# (t = 1, 4, 7, ...); the eligible units do not overlap, and each carries the
# model's own prediction for exactly that text.
#
# Design: stratified random sampling with known inclusion probabilities.
#   strata  = corpus x regime band (V-Dem polyarchy) x model band (model-8)
#   alloc   = equal per stratum (oversamples rare illiberal / uncertain text)
#   weights = N_stratum / n_stratum, so weighted estimates describe the corpus
#   caps    = <= 1 paragraph per speech, <= MAX_PER_CLUSTER per speaker
#             (M&S) or country-year (IAC)
#   overlap = DOUBLE_SHARE of units go to both coders, for reliability
#
# Outputs (in data/coding/):
#   coding_sheet_A.csv, coding_sheet_B.csv  -- what coders see: code_id + text
#                                              (+ previous/next sentence context)
#   coding_key.csv                          -- metadata, weak labels, weights;
#                                              NEVER shown to coders
#   coding_design.txt                       -- parameters and stratum table

suppressPackageStartupMessages({ library(dplyr); library(stringr) })
set.seed(20261020)

N_PER_STRATUM   <- 25      # target per stratum; sparse strata contribute all they have
MAX_PER_CLUSTER <- c(MS = 15, IAC = 3)   # per speaker (M&S, only ~40 leaders) /
                                         # per country-year (IAC, thousands of clusters)
DOUBLE_SHARE    <- 0.20    # share coded by both coders
MIN_WORDS <- 40; MAX_WORDS <- 220
POLY_BREAKS <- c(-Inf, 0.30, 0.50, 0.75, Inf)
POLY_LABELS <- c("closed_aut", "electoral_aut", "flawed_dem", "liberal_dem")
PRED_BREAKS <- c(-Inf, 0.20, 0.80, Inf)          # model-8: >= 0.5 means democratic
PRED_LABELS <- c("model_autocratic", "model_uncertain", "model_democratic")

BOILER <- regex(paste(
  "check against delivery", "e&oe", "transcript of", "\\.{6,}",
  "^\\s*(thank you|good (morning|evening|afternoon)|ladies and gentlemen)",
  "your (majesty|majesties|excellenc)", "^\\s*(mr|madam)\\.? (president|speaker)",
  sep = "|"), ignore_case = TRUE)
caps_share <- function(x) { l <- str_count(x, "[A-Za-z]"); ifelse(l == 0, 1, str_count(x, "[A-Z]") / l) }

eligible <- function(df) {
  df %>%
    filter(t %% 3 == 1) %>%                                   # non-overlapping windows
    mutate(text = str_squish(sent_trigram),
           words = str_count(text, "\\S+")) %>%
    filter(words >= MIN_WORDS, words <= MAX_WORDS,
           !str_detect(text, BOILER), caps_share(text) < 0.5) %>%
    mutate(regime_band = cut(v2x_polyarchy, POLY_BREAKS, POLY_LABELS, right = FALSE),
           model_band  = cut(predict, PRED_BREAKS, PRED_LABELS, right = FALSE)) %>%
    filter(!is.na(regime_band), !is.na(model_band))
}

ms_raw  <- readRDS("data/results/anno_best_models_M&S.RDS")[[1]]
iac_raw <- readRDS("data/results/anno_best_models_IAC.RDS")[[1]]

ms <- ms_raw %>%
  filter(tolower(language) %in% c("english", "translated")) %>%
  transmute(corpus = "MS", doc_id = as.character(doc_id), t = id, sent_trigram, predict, v2x_polyarchy,
            regime, cluster = speaker, speaker, country, year = substr(date, 1, 4)) %>%
  eligible()

iac <- iac_raw %>%
  transmute(corpus = "IAC", doc_id = as.character(doc_id), t = n, sent_trigram, predict, v2x_polyarchy,
            regime = NA_character_, cluster = paste(country, year), speaker = NA_character_,
            country, year = as.character(year)) %>%
  eligible()

pop <- bind_rows(ms, iac) %>%
  mutate(unit_id = paste(corpus, doc_id, t, sep = "_"),
         stratum = paste(corpus, regime_band, model_band, sep = "|"))

# --- stratified draw with caps ----------------------------------------------
# Strata are drawn from rarest to commonest, and a speech used by one stratum is
# removed from the frame of the rest, so <= 1 paragraph per speech holds overall
# without starving the large strata (an earlier version de-duplicated after the
# draw and left the big "uncertain" strata 30-50% short). Speaker / country-year
# caps likewise apply across the whole sample, not per stratum.
# Weights use the full stratum size; the speeches removed for earlier strata are
# a negligible fraction of each frame, so N_h / n_h is a close approximation.
used_docs <- character(0)
used_clusters <- integer(0); names(used_clusters) <- character(0)
order_h <- pop %>% count(stratum) %>% arrange(n) %>% pull(stratum)
picks <- vector("list", length(order_h))
for (k in seq_along(order_h)) {
  g <- pop[pop$stratum == order_h[k], ]
  g <- g[!(paste(g$corpus, g$doc_id) %in% used_docs), ]
  g <- g[sample(nrow(g)), ]
  g <- g[!duplicated(g$doc_id), ]                       # one paragraph per speech
  ck <- paste(g$corpus, g$cluster)
  room <- MAX_PER_CLUSTER[g$corpus] - ifelse(ck %in% names(used_clusters), used_clusters[ck], 0)
  seen <- ave(seq_along(ck), ck, FUN = seq_along)       # running count within cluster
  g <- g[seen <= room, ]
  g <- g[seq_len(min(nrow(g), N_PER_STRATUM)), ]
  picks[[k]] <- g
  used_docs <- c(used_docs, paste(g$corpus, g$doc_id))
  for (c in paste(g$corpus, g$cluster))
    used_clusters[c] <- ifelse(c %in% names(used_clusters), used_clusters[c], 0) + 1
}
smp <- bind_rows(picks)

N_h <- count(pop, stratum, name = "N_stratum")
n_h <- count(smp, stratum, name = "n_stratum")
smp <- smp %>% left_join(N_h, "stratum") %>% left_join(n_h, "stratum") %>%
  mutate(weight = N_stratum / n_stratum)

# --- context: previous and next sentence from the neighbouring trigrams ------
keep_docs <- unique(paste(smp$corpus, smp$doc_id))
ctx <- bind_rows(
  ms_raw  %>% transmute(corpus = "MS",  doc_id = as.character(doc_id), t = id, sent_trigram),
  iac_raw %>% transmute(corpus = "IAC", doc_id = as.character(doc_id), t = n,  sent_trigram)) %>%
  filter(paste(corpus, doc_id) %in% keep_docs)
rm(ms_raw, iac_raw); invisible(gc())
first_sentence <- function(x) str_extract(x, "^.*?[.!?](\\s|$)")
last_sentence  <- function(x) str_extract(x, "[^.!?]*[.!?]\\s*$")
prev <- ctx %>% transmute(corpus, doc_id, t = t + 3, context_before = last_sentence(sent_trigram))
nxt  <- ctx %>% transmute(corpus, doc_id, t = t - 3, context_after  = first_sentence(sent_trigram))
smp <- smp %>% left_join(prev, c("corpus", "doc_id", "t")) %>%
  left_join(nxt, c("corpus", "doc_id", "t"))
rm(ctx, prev, nxt); invisible(gc())

# --- coder assignment: blind, shuffled, with a double-coded overlap ----------
smp <- smp[sample(nrow(smp)), ] %>%
  mutate(code_id = sprintf("P%04d", row_number()),
         double  = row_number() <= round(DOUBLE_SHARE * n()),
         coder   = ifelse(double, "AB", ifelse(row_number() %% 2 == 0, "A", "B")))

dir.create("data/coding", showWarnings = FALSE)
sheet <- function(who) smp %>%
  filter(str_detect(coder, who)) %>% arrange(code_id) %>%
  transmute(code_id, context_before = coalesce(context_before, ""),
            paragraph = text, context_after = coalesce(context_after, ""))
write.csv(sheet("A"), "data/coding/coding_sheet_A.csv", row.names = FALSE)
write.csv(sheet("B"), "data/coding/coding_sheet_B.csv", row.names = FALSE)
write.csv(smp %>% select(code_id, coder, double, corpus, doc_id, t, speaker, country, year,
                         regime, v2x_polyarchy, regime_band, predict, model_band,
                         stratum, N_stratum, n_stratum, weight, words, paragraph = text),
          "data/coding/coding_key.csv", row.names = FALSE)

tab <- smp %>% count(corpus, regime_band, model_band, name = "sampled") %>%
  left_join(pop %>% count(corpus, regime_band, model_band, name = "population"),
            c("corpus", "regime_band", "model_band"))
sink("data/coding/coding_design.txt")
cat("Population (eligible paragraphs):", nrow(pop), "| sampled:", nrow(smp),
    "| double-coded:", sum(smp$double), "\n")
cat("Coder A:", sum(grepl("A", smp$coder)), " Coder B:", sum(grepl("B", smp$coder)), "\n\n")
cat("Parameters: N_PER_STRATUM", N_PER_STRATUM, "| MAX_PER_CLUSTER MS/IAC", paste(MAX_PER_CLUSTER, collapse = "/"),
    "| words", MIN_WORDS, "-", MAX_WORDS, "| seed 20261020\n\n")
print(as.data.frame(tab), row.names = FALSE)
sink()
cat(readLines("data/coding/coding_design.txt"), sep = "\n")
