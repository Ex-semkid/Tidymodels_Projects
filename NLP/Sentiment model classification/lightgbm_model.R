# Install required packages (run once)
# install.packages(c("tidymodels", "xgboost", "dials"))

# Load libraries
library(tidymodels)  # Includes parsnip, recipes, workflows, etc.
library(bonsai)      # Tree wrapper
library(lightgbm)    # Engine for XGBoost
library(tidymodels)
library(textrecipes)
library(vip)
library(tidyverse)
library(themis)

# 2. Load Data ----
data <- read_csv("nyc_sentiments_multi.csv")

data <- data |> 
  mutate(sentiment = as.factor(sentiment)) |> 
  drop_na()

# Set seed for reproducibility
set.seed(369)

# 1. Prepare data (using iris dataset)
split <- initial_split(data, prop = 0.75, strata = sentiment)
train <- training(split)
test <- testing(split)

# 2. Create preprocessing recipe
recipe <- recipe(sentiment ~ text + compound, data = train) %>%
  step_tokenize(text) %>%
  step_stopwords(text) %>%
  step_tokenfilter(text, max_tokens = 1000) %>%
  step_tfidf(text) |> 
  step_zv() |> 
  step_downsample(sentiment, under_ratio = 1)


# 3. Define XGBoost model specification
xgb_spec <- boost_tree(
  trees = 250,
  tree_depth = tune(),    # Typical range: 3-8 by default
  learn_rate = tune(),    # Typical range: 0.01-0.3 by default
  loss_reduction = tune() # Minimum loss reduction (gamma)
) |> 
  set_mode("classification") |>
  set_engine("lightgbm",
             objective = "multiclass",
#            tree_method = "hist",# or remove tree_method similar to 'xgboost' engine
             verbose = -1)

# 4. Create workflow
xgb_workflow <- workflow() %>%
  add_recipe(recipe) %>%
  add_model(xgb_spec)

# 5. Create tuning grid
tune_grid <- grid_random(
  tree_depth(range = c(3, 12)),
  learn_rate(range = c(0.02, 0.5)),               
  loss_reduction(range = c(2, 10)),
  size  = 10
) # use grid_regular for manageable datasets and use levels instead of size 

# 6. Perform cross-validation tuning
set.seed(369)
folds <- vfold_cv(train, v = 5)

tune_results <- xgb_workflow %>%
  tune_grid(
    resamples = folds,
    grid = tune_grid,
    metrics = metric_set(accuracy, roc_auc) # Accuracy, Roc_auc as metrics
  )

show_best(tune_results, metric = "accuracy")
autoplot(tune_results) + theme_bw()


# Example: Visualization of aggregated metrics
tune_results %>%
  collect_metrics() %>%
  ggplot(aes(x = .metric, y = mean, colour = .metric)) +
  geom_step(linewidth = 1) +
  labs(
    title = "Aggregated Metrics After Tuning",
    x = "Metric",
    y = "Mean Value"
  )


# 7. Select best model and finalize workflow
best_params <- select_best(tune_results, metric = "accuracy")
final_workflow <- finalize_workflow(xgb_workflow, best_params)

# 8. Train final model
gbm_final_model <- final_workflow %>%
  fit(data = train)

# 9. Evaluate on test set
test_pred_gbm <- test %>%
  bind_cols(
    predict(final_model, test),
    predict(final_model, test, type = "prob")
  )

# Classification metrics
test_metrics <- metric_set(accuracy, precision, ppv, npv, f_meas, roc_auc, sensitivity, specificity)
lightgb_m <- test_pred_gbm %>% test_metrics(truth = sentiment, estimate = .pred_class, .pred_Negative:.pred_Positive)
lightgb_m

# Confusion matrix
test_pred_gbm %>% 
  conf_mat(truth = sentiment, estimate = .pred_class)


# Step 9: Visualize Results
my_cols <- c(Negative = "magenta", Neutral = "coral", Positive = "navy")
ggplot(test_predictions, aes(x = sentiment, y = .pred_class, colour = .pred_class)) +
  geom_jitter(width = 0.2, alpha = 0.6) +
  scale_color_manual(values = my_cols) +
  theme_minimal() +
  labs(
    title = "Predicted vs Actual Classes",
    x = "Actual Class",
    y = "Predicted Class"
  ) +
  theme_minimal()


# 10. Feature importance
extract_fit_engine(gbm_final_model) |> 
vip(num_features = 10) +
  ggtitle("Top 20 Predictive Tokens")


#--- Generate 'ROC Curve' Data
roc_curve_data <- test_pred_gbm %>%
  roc_curve(truth = sentiment, .pred_Negative:.pred_Positive) # Class probabilities

#-- Plot the ROC Curve
autoplot(roc_curve_data)


