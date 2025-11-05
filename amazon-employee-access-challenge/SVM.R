## --- Poly ---
# --- LIBRARIES ---
library(tidyverse)
library(tidymodels)
library(vroom)
library(kernlab)

# --- READ DATA ---
trainData <- vroom("AmazonEmployeeAccess-IsaacR/amazon-employee-access-challenge/train.csv") %>%
  mutate(ACTION = as.factor(ACTION))

testData <- vroom("AmazonEmployeeAccess-IsaacR/amazon-employee-access-challenge/test.csv")

# --- RECIPE ---
my_recipe <- recipe(ACTION ~ ., data = trainData) %>%
  step_other(all_nominal_predictors(), threshold = 0.001) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_zv(all_predictors())

# --- MODEL SPEC ---
svmPoly <- svm_poly(degree = 1, cost = 0.0131) %>%
  set_mode("classification") %>%
  set_engine("kernlab", prob.model = TRUE)  # important for probability predictions

# --- WORKFLOW ---
poly_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(svmPoly)

# --- FIT MODEL ---
poly_fit <- fit(poly_wf, data = trainData)

# --- PREDICT ---
preds <- predict(poly_fit, new_data = testData, type = "prob") %>%
  bind_cols(testData %>% select(id)) %>%
  select(id, .pred_1) %>%
  rename(Action = .pred_1)

# --- SAVE ---
vroom_write(preds, file = "./SVM_Poly_predictions.csv", delim = ",")

## --- Radial ---
# --- LIBRARIES ---
library(tidyverse)
library(tidymodels)
library(vroom)
library(kernlab)
library(themis)

# --- READ DATA ---
trainData <- vroom("AmazonEmployeeAccess-IsaacR/amazon-employee-access-challenge/train.csv") %>%
  mutate(ACTION = as.factor(ACTION))

testData <- vroom("AmazonEmployeeAccess-IsaacR/amazon-employee-access-challenge/test.csv")

# --- RECIPE ---
my_recipe <- recipe(ACTION ~ ., data = trainData) %>%
  step_other(all_nominal_predictors(), threshold = 0.001) %>%  # group infrequent levels
  step_dummy(all_nominal_predictors()) %>%                     # convert factors to numeric
  step_normalize(all_numeric_predictors()) %>%                 # normalize numeric predictors
  step_zv(all_predictors()) %>%                                # remove zero variance
  step_smote(ACTION)                                           # apply SMOTE to balance classes

# --- MODEL SPEC ---
svmRbf <- svm_rbf(rbf_sigma = 0.177, cost = 0.00316) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

# --- WORKFLOW ---
rbf_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(svmRbf)

# --- FIT MODEL ---
rbf_fit <- fit(rbf_wf, data = trainData)

# --- PREDICT ---
preds <- predict(rbf_fit, new_data = testData, type = "prob") %>%
  bind_cols(testData %>% select(id)) %>%
  select(id, .pred_1) %>%
  rename(Action = .pred_1)

# --- SAVE ---
vroom_write(preds, file = "./SVM_Rbf_SMOTE_predictions.csv", delim = ",")

## --- Linear ---
# --- LIBRARIES ---
library(tidyverse)
library(tidymodels)
library(vroom)
library(kernlab)

# --- READ DATA ---
trainData <- vroom("AmazonEmployeeAccess-IsaacR/amazon-employee-access-challenge/train.csv") %>%
  mutate(ACTION = as.factor(ACTION))

testData <- vroom("AmazonEmployeeAccess-IsaacR/amazon-employee-access-challenge/test.csv")

# --- RECIPE ---
my_recipe <- recipe(ACTION ~ ., data = trainData) %>%
  step_other(all_nominal_predictors(), threshold = 0.001) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_zv(all_predictors())

# --- MODEL SPEC ---
svmLin <- svm_linear(cost = 0.0131) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

# --- WORKFLOW ---
lin_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(svmLin)

# --- FIT MODEL ---
lin_fit <- fit(lin_wf, data = trainData)

# --- PREDICT ---
preds <- predict(lin_fit, new_data = testData, type = "prob") %>%
  bind_cols(testData %>% select(id)) %>%
  select(id, .pred_1) %>%
  rename(Action = .pred_1)

# --- SAVE ---
vroom_write(preds, file = "./SVM_Lin_predictions.csv", delim = ",")
