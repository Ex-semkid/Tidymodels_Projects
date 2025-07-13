# Install required packages (run once)

# Load libraries
library(tidymodels)  # Includes parsnip, recipes, workflows, etc.
library(tidymodels)
library(textrecipes)
library(tidyverse)
library(themis)
library(ranger)


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



# Pre-processing: tokenize → remove stop words → filter → TF–IDF → balance class
rf_recipe <- recipe(sentiment ~ text + compound, data = train) %>%
  step_tokenize(text) %>%
  step_stopwords(text) %>%
  step_tokenfilter(text, max_tokens = 1000) %>%
  step_tfidf(text) %>%
  step_downsample(sentiment, under_ratio = 1)


p <- prep(rf_recipe)
jp <- juice(p)
head(jp)

# Define model specifications
rf_model <- rand_forest(
  mode = "classification",   # For classification tasks
  mtry = tune(),             # Number of predictors to randomly sample at each split
  trees = tune(),            # Number of trees in the forest
  min_n = tune()             # Minimum number of data points in a node
) %>%
  set_engine("ranger", importance = "impurity") # Use the 'ranger' engine for random forests

# Step 4: Combine recipe and model into a workflow
rf_workflow <- workflow() %>%
  add_model(rf_model) %>%
  add_recipe(rf_recipe)

# Step 5: Create a tuning grid
rf_grid <- grid_random(
  mtry(range = c(1, 4)),        # Adjust based on number of predictors
  trees(range = c(100, 500)),   # Sample trees between 100 and 1000
  min_n(range = c(2, 10)),      # Sample min_n values from 2 to 10
  size = 10                     # Number of random combinations to mtry
) # use grid_regular for manageable datasets and use levels instead of size 

# Step 6: Cross-validation for tuning
cv_folds <- vfold_cv(train, v = 5, strata = sentiment)

# Step 7: Tune the model using grid search
tuned_rf <- tune_grid(
  rf_workflow,
  resamples = cv_folds,
  grid = rf_grid,
  metrics = metric_set(accuracy, f_meas) # Use accuracy and f1_score as metrics
)


# Check best metric
show_best(tuned_rf, metric = "accuracy")
autoplot(tuned_results) + theme_bw()


# Example: Visualization of aggregated metrics
tuned_rf %>%
  collect_metrics() %>%
  ggplot(aes(x = .metric, y = mean, colour = .metric)) +
  geom_step(linewidth = 1) +
  labs(
    title = "Aggregated Metrics After Tuning",
    x = "Metric",
    y = "Mean Value"
  )


# 7. Select best model and finalize workflow
best_params <- select_best(tuned_rf, metric = "accuracy")
final_workflow <- finalize_workflow(rf_workflow, best_params)

# 8. Train final model
rf_final_model <- final_workflow %>%
  fit(data = train)

# 9. Evaluate on test set
test_pred <- test %>%
  bind_cols(
    predict(final_model, test),
    predict(final_model, test, type = "prob")
  )

# Classification metrics
test_metrics <- metric_set(accuracy, npv, roc_auc, ppv, sensitivity, specificity)
rf_m <- test_pred %>% test_metrics(truth = sentiment, estimate = .pred_class, .pred_Negative:.pred_Positive)
rf_m

# Confusion matrix
test_pred %>% 
  conf_mat(truth = sentiment, estimate = .pred_class)


# Step 9: Visualize Results
my_cols <- c(Negative = "#E41A1C", Neutral = "#377EB8", Positive = "#4DAF4A")
ggplot(test_pred, aes(sentiment, .pred_class, color = .pred_class)) +
  geom_jitter(width = 0.2, alpha = 0.6) +
  scale_color_manual(values = my_cols) +
  labs(title = "Predicted vs Actual Classes",
       x = "Actual Class",
       y = "Predicted Class") +
  theme_minimal()


# 10. Feature importance
extract_fit_engine(rf_final_model) |> 
  vip(num_features = 10) +
  ggtitle("Top 20 Predictive Tokens")


#--- Generate 'ROC Curve' Data
roc_curve_data <- test_pred %>%
  roc_curve(truth = sentiment, .pred_Negative:.pred_Positive) # Class probabilities

#-- Plot the ROC Curve
autoplot(roc_curve_data)


