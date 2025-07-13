# Install if necessary
pacman::p_load("tidyverse", "tidytext", "textdata", 
               "topicmodels", "wordcloud", "ggraph", "igraph")

# Load libraries
library(tidyverse)   # Used for data manipulation (e.g., dplyr, ggplot2 functions).
library(tidytext)    # Used for tokenization and DTM creation (e.g., unnest_tokens, cast_dtm).
library(textdata)    # Indirectly used for sentiment lexicons via get_sentiments().
library(topicmodels) # Used for LDA topic modeling (e.g., LDA()).
library(wordcloud)   # Used for word cloud visualizations (e.g., wordcloud()).
library(igraph)      # Used for graph creation in bigram analysis (e.g., graph_from_data_frame).
library(ggraph)      # Used for network plotting (e.g., ggraph, geom_edge_link).


# Example: Import a CSV of documents
docs <- read_csv("r_nyc_10k.csv")


# 2. Preprocessing & Tokenization
# 2.1 Tidy‐text tokenization
tidy_docs <- docs %>% 
  select(id, body) %>%
  unnest_tokens(word, body)


# 2.2 Remove stop words and non‐alphabetic tokens
data("stop_words")
clean_tokens <- tidy_docs %>%
  filter(!word %in% stop_words$word) %>%
  filter(str_detect(word, "^[a-z']+$"))

#stems <- clean_tokens %>%
 # mutate(stem = SnowballC::wordStem(word, language = "en"))

#head(stems)


# 3. Term Frequency & TF‐IDF
# 3.1 Word counts per document
term_freq <- clean_tokens %>%
  count(id, word, sort = TRUE)

# 3.2 Compute TF-IDF
tfidf <- term_freq %>%
  bind_tf_idf(word, id, n) %>%
  arrange(desc(tf_idf))


# 4. Sentiment Analysis
# Using NRC lexicon
nrc <- get_sentiments("nrc")


sentiment_scores1 <- clean_tokens %>%
  inner_join(nrc, by = "word") %>%
  count(id, word, sentiment) %>%
  spread(sentiment, n, fill = 0)

# or

sentiment_scores <- clean_tokens %>%
  inner_join(nrc,
             by           = "word",
             relationship = "many-to-many") %>%
  count(id, word, sentiment) %>%
  tidyr::pivot_wider(
    names_from  = sentiment,
    values_from = n,
    values_fill = list(n = 0)
  )


# 5. Bigram Analysis & Network
# 5.1 Bigram tokenization
bigrams <- docs %>%
  unnest_tokens(bigram, body, token = "ngrams", n = 2)

# 5.2 Separate words, filter stop words
bigrams_separated <- bigrams %>%
  separate(bigram, into = c("w1","w2"), sep = " ") %>%
  filter(!w1 %in% stop_words$word,
         !w2 %in% stop_words$word)

# 5.3 Count and graph common bigrams
bigram_counts <- bigrams_separated %>%
  count(w1, w2, sort = TRUE)

bigram_graph <- bigram_counts %>% 
  drop_na() |> 
  filter(n > 15) %>%
  graph_from_data_frame()

ggraph(bigram_graph, layout = "treemap") +
  geom_edge_link(aes(edge_alpha = n), show.legend = FALSE) +
  geom_node_point(color = "coral", size = 5) +
  geom_node_text(aes(label = name), repel = TRUE) +
  theme_void()


# 6. Topic Modeling with LDA
# 6.1 Create DTM using quanteda
chapters <- docs %>%
  group_by(author) %>%
  mutate(
    chapter = cumsum(str_detect(body,
                                regex("^chapter [\\divxlc]", ignore_case = TRUE)
    ))
  ) %>%
  ungroup() %>%
  filter(chapter > 0)



data("stop_words")

dtm <- docs %>% 
  unnest_tokens(word, body) |> 
  filter(!word %in% stop_words$word) %>%
  filter(str_detect(word, "^[a-z']+$")) |> 
  count(id, author, word) %>%
  mutate(document = paste(author, id, sep = "_")) %>%
  tidytext::cast_dtm(document, word, n)

lda_model <- LDA(dtm, k = 4, control = list(seed = 369))


# 6.3 Extract top terms per topic
topics <- tidy(lda_model, matrix = "beta") %>%
  group_by(topic) %>%
  top_n(10, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)

# 6.4 Visualize
topics %>%
  mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(term, beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free_y") +
  coord_flip() +
  scale_x_reordered() +
  labs(x = NULL, y = "Beta")


# 6.5 
top_terms <- terms(lda_model, 10)   # Top 10 terms per topic
print(top_terms)


# 6.5 Draw the word cloud
set.seed(369)   # for reproducible layout
wordcloud::wordcloud(
  words       = top_terms,
  scale       = c(5, 1),      # range of word sizes
  min.freq    = 10,           # only words with freq >= 10
  random.order= FALSE,        # plot high-freq words at center
  rot.per     = 0.45,         # 30% of words rotated 90°
  colors      = brewer.pal(8, "Dark2")
)


# 7. Visualizations

# 7.1 Word cloud of top TF-IDF terms
set.seed(369)
tfidf %>%
  with(wordcloud(word = word,
                 scale       = c(5, 1),    # range of word sizes
                 min.freq    = 100,        # only words with freq >= 100
                 max.words   = 100,        # show all words
                 random.order= FALSE,      # plot high-freq words at center
                 rot.per     = 0.45,       # 30% of words rotated 90°
                 colors      = brewer.pal(8, "Dark2")))


# 7.2 Sentiment bar chart
sentiment_scores %>%
  gather(sentiment, count, -id) %>%
  group_by(sentiment) %>%
  summarize(total = sum(count)) %>%
  ggplot(aes(reorder(x = sentiment, total), y = total, fill = sentiment)) +
  geom_col(show.legend = FALSE) +
  coord_flip() +
  labs(x = "Sentiment", y = "Total Count")

