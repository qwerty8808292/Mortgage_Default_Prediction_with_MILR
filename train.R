## Import Packages ##
library(dplyr)
library(caret)
library(milr)
library(glmnet)
library(randomForest)
set.seed(1)


## Load the Data ##
data <- read.csv("mortgage.csv", header = TRUE, sep = ",")


## Preprocess the data ##
data <- data %>%
  mutate(across(where(is.numeric), ~ifelse(is.na(.), median(., na.rm = TRUE), .)),  
         loan_age = time - orig_time,                                                
         time_to_maturity = mat_time - time,                                        
         time_since_first_obs = time - first_time) %>%
  mutate(default_label = ifelse(status_time == 1, 1, 0)) %>%
  select(-time, -orig_time, -first_time, -mat_time, -status_time, -default_time,
         -payoff_time) %>%
  group_by(id) %>%
  mutate(default_label = max(default_label)) %>%
  ungroup()

# Standardization
numeric_features <- c('balance_time', 'LTV_time', 'interest_rate_time', 'hpi_time',
                      'gdp_time', 'uer_time', 'balance_orig_time', 'FICO_orig_time',
                      'LTV_orig_time', 'Interest_Rate_orig_time', 'hpi_orig_time',
                      'loan_age', 'time_to_maturity', 'time_since_first_obs')
data[numeric_features] <- scale(data[numeric_features])

# Split the data
unique_ids <- unique(data$id)
train_ids <- createDataPartition(unique_ids, p = 0.7, list = FALSE)
train_data <- data %>% filter(id %in% train_ids)
test_data <- data %>% filter(!id %in% train_ids)
train_Z <- train_data$default_label
test_Z <- test_data$default_label
train_X <- as.matrix(train_data %>% select(-id, -default_label))
test_X <- as.matrix(test_data %>% select(-id, -default_label))

# Reassign IDs
reassign_ids <- function(df) {
  unique_ids <- unique(df$id)
  id_map <- setNames(seq_along(unique_ids), unique_ids)
  df <- df %>%
    mutate(id = id_map[as.character(id)])
  return(df)
}
train_data <- reassign_ids(train_data)
test_data <- reassign_ids(test_data)
train_ID <- train_data$id
test_ID <- test_data$id


## Fit MILR model ## 
model <- milr(train_Z, train_X, train_ID, lambda = 0)
summary(model)


## Prediction of MILR model ##
pred <- predict(model, test_X, test_ID, type = "bag")
result <- table(DATA = tapply(test_Z, test_ID, function(x) sum(x) > 0) %>% as.numeric,
                PRED = pred)
accuracy <- sum(diag(result)) / sum(result)
precision <- result[2,2] / sum(result[,2])
sensitivity <- result[2,2] / sum(result[2,])
f1_score <- 2 * ((precision * sensitivity) / (precision + sensitivity))


## Fit logistic model ##
getMilrProb <- function(beta, X, bag) {
  .Call('_milr_getMilrProb', PACKAGE = 'milr', beta, X, bag)
}
model_glm <- cv.glmnet(train_X, train_Z, family = "binomial", alpha = 1)
coef_glm <- as.vector(coef(model_glm, s = "lambda.min"))


## Prediction of logistic model ##
pred_probs_glm <- getMilrProb(coef_glm, cbind(1, test_X), test_ID)
pred_glm <- ifelse(pred_probs_glm > 0.5, 1, 0)
result_glm <- table(True = tapply(test_Z, test_ID, function(x) sum(x) > 0) %>% as.numeric,
                    Predicted = pred_glm)
accuracy_glm <- sum(diag(result_glm)) / sum(result_glm)
precision_glm <- result_glm[2,2] / sum(result_glm[,2])
sensitivity_glm <- result_glm[2,2] / sum(result_glm[2,])
f1_score_glm <- 2 * ((precision_glm * sensitivity_glm) / (precision_glm + sensitivity_glm))


## Comparison ##
cat("MILR", "\n")
cat("Accuracy: ", accuracy, "\n")
cat("Precision: ", precision, "\n")
cat("Sensitivity (Recall): ", sensitivity, "\n")
cat("F1 Score: ", f1_score, "\n\n")

cat("Logistic regression with lasso", "\n")
cat("Accuracy: ", accuracy_glm, "\n")
cat("Precision: ", precision_glm, "\n")
cat("Sensitivity (Recall): ", sensitivity_glm, "\n")
cat("F1 Score: ", f1_score_glm, "\n")


