# --- LIBRARIES ---
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

# --- RECIPE (Enhanced Feature Engineering) ---
my_recipe <- recipe(ACTION ~ ., data = trainData) %>%
  step_nzv(all_predictors()) %>%                           # remove near-zero variance
  step_other(all_nominal_predictors(), threshold = 0.001) %>%  
  step_dummy(all_nominal_predictors()) %>%                 
  step_interact( ~ all_numeric_predictors():all_numeric_predictors() ) %>% # add interactions
  step_poly(all_numeric_predictors(), degree = 2) %>%      # add polynomial features
  step_normalize(all_numeric_predictors()) %>%              # normalize numeric features
  step_smote(ACTION, neighbors = 5)                         # balance classes with SMOTE

# --- MODEL SPEC ---
rf_bal <- rand_forest(mtry = tune(), min_n = tune(), trees = 1000) %>%
  set_mode("classification") %>%
  set_engine("ranger", importance = "impurity")

# --- WORKFLOW ---
balanced_workflow <- workflow() %>%
  add_model(rf_bal) %>%
  add_recipe(my_recipe)

# --- GRID ---
tuning_grid <- grid_regular(
  mtry(range = c(3, ncol(trainData) / 2)),  # slightly narrower range for stability
  min_n(range = c(2, 10)),
  levels = 5
)

# --- FOLDS ---
set.seed(123)
folds <- vfold_cv(trainData, v = 10)

# --- TUNE ---
cv_results <- tune_grid(
  balanced_workflow,
  resamples = folds,
  grid = tuning_grid,
  metrics = metric_set(roc_auc),
  control = control_grid(save_pred = TRUE)
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

vroom_write(preds, file = "./RandForSMOTE_PolyInteract_predictions.csv", delim = ",")