# --- LIBRARIES ---
library(tidyverse)
library(tidymodels)
library(vroom)
library(discrim)
library(doParallel)
library(themis)   # for SMOTE

# --- READ DATA ---
trainData <- vroom("AmazonEmployeeAccess-IsaacR/amazon-employee-access-challenge/train.csv", show_col_types = FALSE) %>%
  mutate(ACTION = as.factor(ACTION))

testData <- vroom("AmazonEmployeeAccess-IsaacR/amazon-employee-access-challenge/test.csv", show_col_types = FALSE)

# --- RECIPE (with SMOTE) ---
my_recipe <- recipe(ACTION ~ ., data = trainData) %>%
  step_other(all_nominal_predictors(), threshold = 0.001) %>%  # group infrequent levels
  step_dummy(all_nominal_predictors()) %>%                     # convert factors to dummy numeric
  step_smote(ACTION)                                           # balance classes with SMOTE

# --- MODEL ---
nb_model <- naive_Bayes(
  Laplace = tune(),
  smoothness = tune()
) %>%
  set_mode("classification") %>%
  set_engine("naivebayes")

# --- WORKFLOW ---
nb_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nb_model)

# --- RESAMPLING ---
folds <- vfold_cv(trainData, v = 10, repeats = 2)

# --- TUNING GRID ---
nb_grid <- grid_regular(
  Laplace(range = c(0, 1)),
  smoothness(range = c(0, 1)),
  levels = 3
)

# --- PARALLEL PROCESSING ---
cl <- makePSOCKcluster(parallel::detectCores())
registerDoParallel(cl)

# --- TUNING ---
nb_cv_results <- tune_grid(
  nb_wf,
  resamples = folds,
  grid = nb_grid,
  metrics = metric_set(roc_auc),
  control = control_grid(save_pred = TRUE, parallel_over = "everything")
)

stopCluster(cl)

# --- SELECT BEST PARAMETERS ---
nb_best <- nb_cv_results %>%
  select_best(metric = "roc_auc")

# --- FINALIZE AND FIT WORKFLOW ---
nb_fit <- nb_wf %>%
  finalize_workflow(nb_best) %>%
  fit(data = trainData)

# --- PREDICT ON TEST DATA ---
preds <- predict(nb_fit, new_data = testData, type = "prob") %>%
  bind_cols(testData %>% select(id)) %>%
  select(id, .pred_1) %>%
  rename(Action = .pred_1)

# --- SAVE PREDICTIONS ---
vroom_write(preds, file = "./NaiveBayes_SMOTE_predictions.csv", delim = ",")
