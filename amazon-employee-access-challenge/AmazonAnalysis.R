# library(tidyverse)
# library(tidymodels)
# library(vroom)
# library(patchwork)
# library(DataExplorer)
# library(GGally)
# library(ggmosaic)
# 
# trainData <- vroom("AmazonEmployeeAccess-IsaacR/amazon-employee-access-challenge/train.csv") %>%
#   mutate(across(everything(), as.factor))
# testData <- vroom("AmazonEmployeeAccess-IsaacR/amazon-employee-access-challenge/test.csv")
# 
# ggplot(trainData, aes(x = factor(ACTION))) +
#   geom_bar(fill = "darkgreen") +
#   labs(title = "Distribution of ACTION", x = "ACTION", y = "Count")
# 
# trainData %>%
#   count(ROLE_FAMILY_DESC, sort = TRUE) %>%
#   slice_head(n = 10) %>%
#   ggplot(aes(x = reorder(ROLE_FAMILY_DESC, n), y = n)) +
#   geom_col(fill = "steelblue") +
#   coord_flip() +
#   labs(title = "Top 10 ROLE_FAMILY_DESC counts", x = "ROLE_FAMILY_DESC", y = "Count")
# 
# amazon_recipe <- recipe(ACTION ~ ., data = trainData) %>%
#   step_other(all_nominal_predictors(), threshold = 0.001, other = "other") %>%
#   step_dummy(all_nominal_predictors())
# 
# prep_recipe <- prep(amazon_recipe)
# baked_train <- bake(prep_recipe, new_data = trainData)
# 
# ## LOGISTIC REGRESSION
# logRegModel <- logistic_reg() %>% #Type of model
#   set_engine("glm")
# 
# ## Put into a workflow here
# amazon_wf <- workflow() %>%
#   add_recipe(amazon_recipe) %>%
#   add_model(logRegModel) %>%
#   fit(data=trainData)
# 
# log_preds <- predict(amazon_wf, new_data = testData)
# 
# ## Make predictions
# amazon_predictions <- predict(amazon_wf,
#                               new_data=testData,
#                               type=prob) # "class" or "prob"

library(tidyverse)
library(tidymodels)
library(vroom)

# --- READ DATA ---
trainData <- vroom("AmazonEmployeeAccess-IsaacR/amazon-employee-access-challenge/train.csv") %>%
  mutate(ACTION = as.factor(ACTION)) %>%
  mutate(across(where(is.character), as.factor))

testData <- vroom("AmazonEmployeeAccess-IsaacR/amazon-employee-access-challenge/test.csv") %>%
  mutate(across(where(is.character), as.factor)) # make types consistent with train

# --- RECIPE ---
amazon_recipe <- recipe(ACTION ~ ., data = trainData) %>%
  step_other(all_nominal_predictors(), threshold = 0.001, other = "other") %>%
  step_dummy(all_nominal_predictors(), -all_outcomes())

# --- MODEL (standard logistic regression) ---
logRegModel <- logistic_reg() %>% 
  set_engine("glm") %>% 
  set_mode("classification")

# --- WORKFLOW and FIT ---
amazon_wf <- workflow() %>%
  add_recipe(amazon_recipe) %>%
  add_model(logRegModel)

set.seed(123)
amazon_fit <- fit(amazon_wf, data = trainData)

# --- PREDICTIONS ---
class_preds <- predict(amazon_fit, new_data = testData, type = "class") %>%
  rename(ACTION = .pred_class) # adjust to Kaggle column name if needed
prob_preds <- predict(amazon_fit, new_data = testData, type = "prob")

submission <- bind_cols(testData %>% select(id), class_preds) %>%
  # ensure ACTION is numeric 0/1 (if Kaggle requires numeric)
  mutate(ACTION = as.integer(as.character(ACTION)))

# Write CSV
write_csv(submission, "amazon_logistic_submission.csv")

# Print first rows for quick check
print(head(submission))
