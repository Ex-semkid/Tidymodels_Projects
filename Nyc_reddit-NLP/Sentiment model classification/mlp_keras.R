# Load necessary packages
library(tidyverse)
library(tidymodels)
library(textrecipes)
library(themis)
library(keras)       # Keras interface for R
library(tensorflow)  # TensorFlow backend
library(vip)


data <- read_csv("nyc_sentiments_multi.csv") 
data <- data |> 
  mutate(sentiment = as.factor(sentiment)) |> 
  drop_na()

# Set seed for reproducibility
set.seed(369)

# Split data into training and testing sets (75% train, 25% test)
split <- initial_split(data, prop = 0.75, strata = sentiment)
training <- training(split)
testing <- testing(split)

# Define a recipe for text preprocessing
sent_recipe <- recipe(sentiment ~ text + compound, data = training) %>%
  step_tokenize(text) %>%           # Tokenize the text
  step_stopwords(text) %>%          # Remove common stop words
  step_tokenfilter(text, max_tokens = 1000) %>%  # Keep top 1000 tokens
  step_tfidf(text) |>                   # Apply TF-IDF transformation
  step_zv() |> 
  step_upsample(sentiment, over_ratio = 1)


# 3. Define MLP Model Specification ----------------------------------------
mlp_spec <- mlp(
  hidden_units = 128,         # one hidden layers with 64 and 32 units
  activation = "swish",       # Activation function
  dropout = 0.4,              # Regularization with dropout
  epochs = 250,               # Number of training iterations
  learn_rate = 0.005          # Learning rate for optimizer (Adam optimizer)
) %>%
  set_engine("keras", verbose = 0) %>%  # Use Keras back-end
  set_mode("classification")            # For categorical outcome

# 4. Create Workflow ------------------------------------------------------
mlp_wf <- workflow() %>%
  add_recipe(sent_recipe) %>%
  add_model(mlp_spec)

# 5. Train Model ----------------------------------------------------------
mlp_fit <- fit(mlp_wf, data = training)
mlp_fit

# Make predictions on the test set
# 9. Evaluate on Test Set ----
mlp_test_pred  <- testing %>%
  bind_cols(
    predict(mlp_fit, testing),
    predict(mlp_fit, testing, type = "prob")
  )

test_metrics <- metric_set(accuracy, f_meas, roc_auc, sensitivity, specificity)
mlp_matrics <- mlp_test_pred %>% test_metrics(truth = sentiment, estimate = .pred_class, .pred_Negative:.pred_Positive)
mlp_matrics

#save(mlp_matrics, mlp_fit, file = "keras_mlp_model.RData")

# Evaluate performance
# Confusion Matrix
mlp_test_pred %>%
  conf_mat(truth = sentiment, estimate = .pred_class)

# Confusion matrix
roc_curve_data <- mlp_test_pred %>%
  roc_curve(truth = sentiment, .pred_Negative:.pred_Positive) # Specify columns for all class probabilities
autoplot(roc_curve_data) + labs(title = "ROC Curve: mlp keras Model")


# Extract model coefficients (feature importances)
ggplot(mlp_test_pred, aes(x = sentiment, y = .pred_class)) +
  geom_jitter(width = 0.2, color = "blue", alpha = 0.6) +
  labs(
    title = "Predicted vs Actual Classes",
    x = "Actual Class",
    y = "Predicted Class"
  ) +
  theme_minimal()

