# nlp_tidymodels_analysis.R
# Complete NLP analysis pipeline using tidymodels + textrecipes

# 1. Load Packages ----
library(tidymodels)
library(textrecipes)
library(modeldata)
library(vip)
library(tidyverse)
library(themis)

# 2. Load Data ----
data <- read_csv("Datasets/nyc_sentiments_multi.csv")
colSums(is.na(data))

data <- data |> 
  mutate(sentiment = as.factor(sentiment)) |> 
  drop_na()
colSums(is.na(data))

# 3. Data Split ----
set.seed(369)
split_obj  <- initial_split(data, strata = sentiment, prop = 0.75)
train_data <- training(split_obj)
test_data  <- testing(split_obj)

table(train_data$sentiment)

# 4. Preprocessing Recipe ----
text_rec <- recipe(sentiment ~ text + compound, data = train_data) %>%
  step_tokenize(text) %>%
  step_stopwords(text) %>%
  step_tokenfilter(text, max_tokens = 500) %>%
  step_tfidf(text) |> 
  step_zv() |> 
  step_downsample(sentiment, under_ratio = 1)


p <- prep(text_rec)
jp <- juice(p)
table(jp$sentiment)

colSums((is.na(jp)))


# 5. Model Specification ----
multinom_spec <- multinom_reg(
  penalty = tune(),    # regularization λ
  mixture = tune()     # blend α (L1 vs L2)
) %>%
  set_engine("glmnet") %>%
  set_mode("classification")


# 6. Workflow & Resampling ----
glmnet_wf   <- workflow() %>%
  add_recipe(text_rec) %>%
  add_model(multinom_spec)

set.seed(223)
cv_splits <- vfold_cv(train_data, v = 5, strata = sentiment)

# 7. Hyperparameter Grid & Tuning ----
grid_vals <- grid_random(
  penalty(),  # Log scale for penalty
  mixture(),
  size = 10
)

glmnet_tune <- tune_grid(
  glmnet_wf,
  resamples = cv_splits,
  grid      = grid_vals,
  metrics   = metric_set(roc_auc),
  control   = control_grid(save_pred = TRUE)
)

# Check best metric
show_best(glmnet_tune, metric = "roc_auc")
autoplot(glmnet_tune) + theme_bw()

# Example: Visualization of aggregated metrics
glmnet_tune %>%
  collect_metrics() %>%
  ggplot(aes(x = .metric, y = mean, colour = .metric)) +
  geom_step() +
  labs(
    title = "Aggregated Metrics After Tuning",
    x = "Metric",
    y = "Mean Value"
  )


# 8. Finalize and Refit ----
glmnet_best <- select_best(glmnet_tune, metric = "roc_auc")
glmnet_wf   <- finalize_workflow(glmnet_wf, glmnet_best)
glmnet_fit  <- fit(glmnet_wf, data = train_data)


# 9. Evaluate on Test Set ----
glmnet_test_pred <- test_data %>%
  bind_cols(
    predict(glmnet_fit, test_data),
    predict(glmnet_fit, test_data, type = "prob")
  )

test_metrics <- metric_set(accuracy, 
                           roc_auc, 
                           f_meas, 
                           ppv, 
                           npv, 
                           sensitivity, 
                           specificity)

glmnet_matrics <- glmnet_test_pred %>% 
  test_metrics(truth = sentiment, 
               estimate = .pred_class, 
               .pred_Negative:.pred_Positive)
glmnet_matrics


# 10. Interpret Model ----
glmnet_fit %>%
  extract_fit_parsnip() %>%
  vip(num_features = 10) +
  ggtitle("Top 20 Predictive Tokens")


# Confusion Matrix
glmnet_test_pred %>%
  conf_mat(truth = sentiment, estimate = .pred_class)

# Step 9: Visualize Results
ggplot(glmnet_test_pred, aes(x = sentiment, y = .pred_class)) +
  geom_jitter(width = 0.2, color = "blue", alpha = 0.6) +
  labs(
    title = "Predicted vs Actual Classes",
    x = "Actual Class",
    y = "Predicted Class"
  ) +
  theme_minimal()


# Step 12: Multiclass ROC Curve
roc_curve_data <- glmnet_test_pred %>%
  roc_curve(truth = sentiment, .pred_Negative:.pred_Positive) # Specify columns for all class probabilities
autoplot(roc_curve_data) + labs(title = "ROC Curve: Multinomial gmnet Model")


# End of script