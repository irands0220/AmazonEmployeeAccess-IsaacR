trainData <- vroom("AmazonEmployeeAccess-IsaacR/amazon-employee-access-challenge/train.csv") %>%
  mutate(ACTION = as.factor(ACTION)) %>%
  mutate(across(where(is.character), as.factor))

testData <- vroom("AmazonEmployeeAccess-IsaacR/amazon-employee-access-challenge/test.csv") %>%
  mutate(across(where(is.character), as.factor)) # make types consistent with train

my_mod <- rand_forest(mtry = tune(),
                      min_n=tune(),
                      trees=500) %>%
set_engine("ranger") %>%
set_mode("classification")

## Create a workflow with model & recipe


## Set up grid of tuning values

## Set up K-fold CV

## Find best tuning parameters

## Finalize workflow and predict