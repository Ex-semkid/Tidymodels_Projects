# Model matrices comparison

load('glmnet_sent_model.RData')
load('lightgbm_sent_model.RData')
load('keras_mlp_model.RData')

glmnet_matrics <- glmnet_matrics %>%
  mutate(model = "glmnet classification") |> 
  select(4, everything())

lightgb_m <- lightgb_m |> 
  mutate(model = "light gbm") |> 
  select(4, everything())

mlp_matrics <- mlp_matrics |> 
  mutate(model = "mlp keras") |> 
  select(4, everything())


all_matrix <- bind_rows(glmnet_matrics, lightgb_m, mlp_matrics ) |> 
  select(model, .metric, .estimate) |> 
  filter(.metric %in% c("accuracy", "sensitivity", "specificity"))


# Plot comaprison
ggplot(all_matrix, aes(
  x    = reorder(model, .estimate),
  y    = .estimate,
  fill = .metric
)) +
  geom_col(position = "dodge") + theme_bw() + 
  theme(legend.title = element_blank()) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1)
  ) + labs(title = 'Model Comaprism', x = 'Model', y = 'Estimate')

