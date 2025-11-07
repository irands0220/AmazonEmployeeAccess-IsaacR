library(tidyverse)
library(tidymodels)
library(embed)
library(vroom)
library(workflows)
library(glmnet)
library(ranger)
library(themis)

# --- READ DATA ---
trainData <- vroom("AmazonEmployeeAccess-IsaacR/amazon-employee-access-challenge/train.csv") %>%
  mutate(ACTION = as.factor(ACTION))

testData <- vroom("AmazonEmployeeAccess-IsaacR/amazon-employee-access-challenge/test.csv")

# --- RECIPE ---
my_recipe <- recipe(ACTION~., data = trainData) %>% 
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>% 
  step_normalize(all_numeric_predictors())

# --- MODEL SPEC ---
rf_bal <- rand_forest(
  mtry = tune(),
  min_n = tune(),
  trees = 500) %>%
  set_mode("classification") %>%
  set_engine("ranger")

# --- WORKFLOW ---
balanced_workflow <- workflow() %>%
  add_model(rf_bal) %>%
  add_recipe(my_recipe)

# --- GRID ---
tuning_grid <- grid_regular(mtry(range=c(1, 9)), min_n(), levels = 3)

# --- FOLDS ---
folds <- vfold_cv(trainData, v = 10, repeats = 1)

# --- TUNE ---
cv_results <- tune_grid(
  balanced_workflow,
  resamples = folds,
  grid = tuning_grid,
  metrics = metric_set(roc_auc)
)

best_tune <- cv_results %>% select_best(metric = "roc_auc")

# --- FINAL FIT ---
rand_for_fit <- balanced_workflow %>%
  finalize_workflow(best_tune) %>%
  fit(data = trainData)

# --- PREDICT ---
preds <- predict(rand_for_fit, new_data = testData, type = "prob") %>%
  bind_cols(testData %>% select(id)) %>%
  select(id, .pred_1) %>%
  rename(Action = .pred_1)

vroom_write(preds, file = "/Users/isaacrands/Documents/Stats/Stat_348/AmazonEmployeeAccess-IsaacR/amazon-employee-access-challenge/predictions_csv/Final_predictions.csv", delim = ",")
