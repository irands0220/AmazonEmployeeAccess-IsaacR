# --- LIBRARIES ---
library(tidyverse)
library(tidymodels)
library(vroom)
library(doParallel)
library(themis)
library(kknn)

# --- READ DATA ---
trainData <- vroom("train.csv", show_col_types = FALSE) %>%
  mutate(ACTION = as.factor(ACTION))

testData <- vroom("test.csv", show_col_types = FALSE)

# --- SETUP PARALLEL PROCESSING ---
number_cores <- 30
cl <- makePSOCKcluster(number_cores)
registerDoParallel(cl)

# --- RECIPE ---
my_recipe <- recipe(ACTION ~ ., data = trainData) %>%
  step_mutate(across(all_numeric_predictors(), as.factor)) %>%
  step_other(all_nominal_predictors(), threshold = 0.001) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

# --- MODEL ---
knn_model <- nearest_neighbor(
  neighbors = tune()   # will tune neighbors
) %>%
  set_mode("classification") %>%
  set_engine("kknn")

# --- WORKFLOW ---
knn_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(knn_model)

# --- CROSS-VALIDATION ---
folds <- vfold_cv(trainData, v = 5, repeats = 1)

# --- TUNING GRID ---
knn_grid <- grid_regular(neighbors(), levels = 10)  # 10 neighbor options

# --- TUNING (parallelized) ---
knn_cv_results <- knn_wf %>%
  tune_grid(
    resamples = folds,
    grid = knn_grid,
    metrics = metric_set(roc_auc),
    control = control_grid(save_pred = TRUE, parallel_over = "everything")
  )

# --- SELECT BEST PARAMETERS ---
knn_best <- knn_cv_results %>%
  select_best(metric = "roc_auc")

# --- FINALIZE AND FIT WORKFLOW ---
knn_fit <- knn_wf %>%
  finalize_workflow(knn_best) %>%
  fit(data = trainData)

# --- MAKE PREDICTIONS ---
preds <- knn_fit %>%
  predict(new_data = testData, type = "prob") %>%
  bind_cols(testData) %>%
  rename(ACTION = .pred_1) %>%
  rename(ID = id) %>%
  select(ID, ACTION)

# --- SAVE PREDICTIONS ---
vroom_write(preds, "KNN_predictions.csv")

# --- STOP PARALLEL CLUSTER ---
stopCluster(cl)
