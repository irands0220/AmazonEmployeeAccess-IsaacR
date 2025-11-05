# library(tidyverse)
# library(tidymodels)
# library(vroom)
# 
# # --- READ DATA ---
# trainData <- vroom("AmazonEmployeeAccess-IsaacR/amazon-employee-access-challenge/train.csv") %>%
#   mutate(ACTION = as.factor(ACTION)) %>%
#   mutate(across(where(is.character), as.factor))
# 
# testData <- vroom("AmazonEmployeeAccess-IsaacR/amazon-employee-access-challenge/test.csv") %>%
#   mutate(across(where(is.character), as.factor)) # make types consistent with train
# 
# # --- RECIPE ---
# amazon_recipe <- recipe(ACTION ~ ., data = trainData) %>%
#   step_other(all_nominal_predictors(), threshold = 0.001, other = "other") %>%
#   step_dummy(all_nominal_predictors(), -all_outcomes()) %>%
#   step_normalize(all_predictors())
# 
# # --- MODEL (standard logistic regression) ---
# logRegModel <- logistic_reg(mixture=tune(), penalty=tune()) %>%
#   set_engine("glmnet") %>%
#   set_mode("classification")
# 
# 
# # --- WORKFLOW and FIT ---
# amazon_wf <- workflow() %>%
#   add_recipe(amazon_recipe) %>%
#   add_model(logRegModel)
# 
# tuning_grid <- grid_regular(
#   penalty(range = c(-8, 0)),   # penalty on log10 scale; grid_regular will expand appropriately
#   mixture(range = c(0, 1)),
#   levels = c(penalty = 5, mixture = 5)
# )
# 
# ## Split data for CV
# folds <- vfold_cv(trainData, v = 10, repeats=1)
# 
# ## Run the CV
# CV_results <- amazon_wf %>%
# tune_grid(resamples=folds,
#           grid=tuning_grid,
#           metrics=NULL)
# 
# bestTune <- CV_results %>%
# select_best("roc_auc")
# 
# ## Finalize the Workflow & fit it
# final_wf <-
# amazon_wf %>%
# finalize_workflow(bestTune) %>%
# fit(data=trainData)
# 
# ## Predict
# final_wf %>%
# predict(new_data = testData, type="class")
# 
# # --- PREDICTIONS ---
# class_preds <- predict(amazon_fit, new_data = testData, type = "class") %>%
#   rename(ACTION = .pred_class) # adjust to Kaggle column name if needed
# prob_preds <- predict(amazon_fit, new_data = testData, type = "prob")
# 
# submission <- bind_cols(testData %>% select(id), class_preds) %>%
#   # ensure ACTION is numeric 0/1 (if Kaggle requires numeric)
#   mutate(ACTION = as.integer(as.character(ACTION)))
# 
# # Write CSV
# write_csv(submission, "amazon_logistic_submission.csv")
# 
# # Print first rows for quick check
# print(head(submission))

# corrected_amazon_logistic.R
library(tidyverse)
library(tidymodels)
library(vroom)
library(embed)      # for target encoding (step_lencode_glm)
library(parsnip)
library(workflows)
library(dials)
library(tune)

# --- READ DATA ---
trainData <- vroom("AmazonEmployeeAccess-IsaacR/amazon-employee-access-challenge/train.csv", show_col_types = FALSE) %>%
  mutate(ACTION = as.factor(ACTION))

testData <- vroom("AmazonEmployeeAccess-IsaacR/amazon-employee-access-challenge/test.csv", show_col_types = FALSE)

# --- RECIPE: use target encoding (via embed::step_lencode_glm) ---
amazon_recipe <- recipe(ACTION ~ ., data = trainData) %>%
  step_lencode_glm(all_nominal_predictors(), outcome = "ACTION") %>%
  step_normalize(all_predictors()) %>%
  step_pca(all_predictors(), threshold=.85) #Threshold is between 0 and 1

# --- MODEL (penalized logistic regression with glmnet) ---
logRegModel <- logistic_reg(penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet") %>%
  set_mode("classification")

# --- WORKFLOW ---
amazon_wf <- workflow() %>%
  add_recipe(amazon_recipe) %>%
  add_model(logRegModel)

# --- TUNING GRID ---
tuning_grid <- grid_regular(
  penalty(range = c(-8, 0)),   # on log10 scale: 10^-8 .. 10^0
  mixture(range = c(0, 1)),
  levels = c(penalty = 5, mixture = 5)
)

# --- RESAMPLES (CV) ---
set.seed(123)
folds <- vfold_cv(trainData, v = 10)

# --- Run the CV: ensure we ask for roc_auc ---
CV_results <- amazon_wf %>%
  tune_grid(
    resamples = folds,
    grid = tuning_grid,
    metrics = metric_set(roc_auc)
  )

# --- get best tune by AUC ---
bestTune <- CV_results %>%
  select_best(metric = "roc_auc")

# --- Finalize the workflow with the best parameters and fit on full training set ---
final_wf <- amazon_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = trainData)

# --- PREDICT on test set ---
prob_preds  <- predict(final_wf, new_data = testData, type = "prob")

prob_preds <- prob_preds %>% 
  select(.pred_1) %>% 
  bind_cols(., testData) %>% #Bind predictions with test data
  select(id, .pred_1) %>% #Just keep resource and predictions
  rename(Action=.pred_1)

# Write CSV
write_csv(prob_preds, "amazon_logistic_submission.csv")
