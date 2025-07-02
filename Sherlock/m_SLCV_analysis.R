library(SuperLearner)
library(readr)
library(dplyr)
library(PRROC)

# Load data
m_SL <- readRDS("m_SLCV.rds")
data <- read_csv("data_5yr_base_omit_TRAIN_yearFE.csv")

train_yearFE_SL <- data |> 
  mutate(extreme_closure_10pct_over_5yr = 
           if_else(extreme_closure_10pct_over_5yr == 'X1', 1, 0),
         year = as.numeric(as.character(year)))

# Pull out outcome 
Y <- train_yearFE_SL[["extreme_closure_10pct_over_5yr"]]

# Evaluate class-weighted model performance
# -----------------------------------------
# Get cross-validated predictions
cv_preds <- m_SL$SL.predict

# Calculate performance metrics with emphasis on minority class
conf_matrix <- table(Predicted = ifelse(cv_preds > 0.5, 1, 0), 
                     Actual = Y)
print("Confusion Matrix:")
print(conf_matrix)

# Calculate sensitivity (recall) and specificity
sensitivity <- conf_matrix[2,2] / sum(conf_matrix[,2])
specificity <- conf_matrix[1,1] / sum(conf_matrix[,1])

print(paste("Sensitivity (True Positive Rate):", round(sensitivity, 4)))
print(paste("Specificity (True Negative Rate):", round(specificity, 4)))

# Calculate balanced accuracy
balanced_acc <- (sensitivity + specificity) / 2
print(paste("Balanced Accuracy:", round(balanced_acc, 4)))

# Calculate F1 score (harmonic mean of precision and recall)
precision <- conf_matrix[2,2] / sum(conf_matrix[2,])
f1_score <- 2 * (precision * sensitivity) / (precision + sensitivity)
print(paste("F1 Score:", round(f1_score, 4)))

# Calculate AUC-PR
print("Checking AUC-PR pre-fold examination...")
scores_positive <- m_SL$SL.predict[Y == 1]
scores_negative <- m_SL$SL.predict[Y == 0]

pr_curve_SL <- PRROC::pr.curve(
  scores.class0 = cv_preds,   # all predicted probabilities
  weights.class0 = Y,         # true labels: 1 = positive, 0 = negative
  curve = TRUE)

cv_auc_pr_SL <- pr_curve_SL$auc.integral
print(paste("Area Under PR Curve (AUC-PR) (Cross-validated model):", 
            round(cv_auc_pr_SL, 4)))

# Export pr_curve info
saveRDS(pr_curve_SL, file = 'm_SLCV_prCurve.rds')

# Get fold performance
# -----------------------------------------
get_SLevals_by_fold <- function(cvsl_object) {
  # Initialize storage for PR-AUC and Recall values
  V <- cvsl_object$V
  fold_prauc <- numeric(V)
  fold_recall <- numeric(V)
  
  # Loop through each fold
  for (i in 1:V) {
    # Get indices for validation fold
    val_indices <- cvsl_object$folds[[i]]
    
    # Get SuperLearner predictions for this fold
    sl_preds <- cvsl_object$SL.predict[val_indices]
    
    # Get actual outcomes for this fold
    y_actual <- cvsl_object$Y[val_indices]
    
    # Calculate PR-AUC for this fold if both classes are present
    if (length(unique(y_actual)) > 1) {
      pr_curve <- PRROC::pr.curve(
        scores.class0 = sl_preds,  # Predicted probabilities for class 1
        weights.class0 = y_actual,  # True labels (0/1)
        curve = TRUE
      )
      fold_prauc[i] <- pr_curve$auc.integral
    } else {
      fold_prauc[i] <- NA
    }
    
    # Calculate recall for this fold using a threshold of 0.5
    # Only calculate recall if there is at least one positive class; otherwise, assign NA.
    if (sum(y_actual) > 0) {
      pred_class <- ifelse(sl_preds >= 0.5, 1, 0)
      TP <- sum(pred_class == 1 & y_actual == 1)
      FN <- sum(pred_class == 0 & y_actual == 1)
      fold_recall[i] <- TP / (TP + FN)
    } else {
      fold_recall[i] <- NA
    }
  }
  
  # Return a data frame with fold numbers, their corresponding PR-AUC and Recall
  result <- data.frame(
    fold = 1:V,
    prauc = fold_prauc,
    recall = fold_recall
  )
  
  # Add summary statistics for PR-AUC and Recall
  result$mean_prauc <- mean(result$prauc, na.rm = TRUE)
  result$sd_prauc <- sd(result$prauc, na.rm = TRUE)
  result$mean_recall <- mean(result$recall, na.rm = TRUE)
  result$sd_recall <- sd(result$recall, na.rm = TRUE)
  
  return(result)
}
print("Evaluating folds...")
m_SL_evals <- get_SLevals_by_fold(m_SL)
saveRDS(m_SL_evals, file = 'm_SLCV_evals.rds')
print("Eval complete and object saved")

# This is the value comparable to caret's averaged prSummary result
avg_cv_auc_pr_SL <- m_SL_evals$mean_prauc[1] # Extract the mean value

print(paste("Average Area Under PR Curve (AUC-PR) across Folds (caret-comparable):",
            round(avg_cv_auc_pr_SL, 4)))
print(paste("SD Area Under PR Curve (AUC-PR) across Folds:",
            round(m_SL_evals$sd_prauc[1], 4)))
