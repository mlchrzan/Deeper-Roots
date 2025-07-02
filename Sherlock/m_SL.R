library(SuperLearner) 
library(readr)
library(doParallel)
library(xgboost)
library(ranger)
library(glmnet)
library(dplyr)
library(caret)

# Load data
train_yearFE <- read_csv("data_5yr_base_omit_TRAIN_yearFE.csv")
folds <- readRDS("folds.rds")
folds_df <- read_csv("folds_LSTM.csv")

create_inner_cvControls <- function(df) {
  # Identify fold indicator columns assuming only "agency_id" is non-fold indicator.
  fold_cols <- setdiff(names(df), "agency_id")
  
  # Initialize the list to store cvControl lists for each outer fold.
  cv_control_list <- vector("list", length = length(fold_cols))
  
  # Loop over each fold indicator column.
  for (i in seq_along(fold_cols)) {
    current_fold <- fold_cols[i]
    
    # Extract the indicator that sets if row is in the training set of the outer fold
    train_indicator <- df[[current_fold]]
    
    # Identify the agencies that are in the outer fold training set.
    df_fold <- df[train_indicator == 1, ]
    inner_folds <- groupKFold(df_fold$agency_id, k = 5)
    
    # Invert the folds to represent validation indices of the outer fold, not training
    n_total <- nrow(df_fold) # Total number of observations
    train_indices_list <- inner_folds
    
    # Get all possible row indices
    all_indices <- seq_len(n_total)
    
    # Generate the validation indices list using setdiff
    #    lapply iterates through each fold's training indices in train_indices_list
    #    For each set of training indices (train_idx), it finds the indices in
    #    all_indices that are NOT present in train_idx.
    folds_validation_indices <- lapply(train_indices_list, function(train_idx) {
      setdiff(all_indices, train_idx)
    })
    
    # Store the cvControl list for this outer fold.
    # Set V = 5 inner folds, and validRows as a list of indices for each fold.
    cv_control_list[[i]] <- list(V = 5, validRows = folds_validation_indices)
  }
  
  return(cv_control_list)
}


# Define Variable Names
# --------------------------
outcome_variable_name <- "extreme_closure_10pct_over_5yr" 
cluster_id_variable_name <- "agency_id"

# Automatically determine predictor variables (all columns except outcome and cluster ID)
all_column_names <- colnames(train_yearFE)
predictor_variable_names <- setdiff(all_column_names, 
                                    c(outcome_variable_name, cluster_id_variable_name))

# Prepare Data Subsets for SuperLearner
# -----------------------------------------
# Ensure outcome and year are numeric 
train_yearFE_SL <- train_yearFE |> 
  mutate(extreme_closure_10pct_over_5yr = 
           if_else(extreme_closure_10pct_over_5yr == 'X1', 1, 0),
         year = as.numeric(as.character(year)))

# Pull out outcome and design matrix
Y <- train_yearFE_SL[[outcome_variable_name]]
X <- train_yearFE_SL[, predictor_variable_names, drop = FALSE]
id <- train_yearFE_SL[[cluster_id_variable_name]]

# Calculate class weights based on class imbalance
# -------------------------------------------------
# Compute the ratio of negative to positive cases
n_pos <- sum(Y == 1)
n_neg <- sum(Y == 0)
pos_weight <- n_neg / n_pos  # Higher weight for the minority class

# Print class balance information
print(paste("Positive cases:", n_pos, "(", round(100 * n_pos / length(Y), 2), "%)"))
print(paste("Negative cases:", n_neg, "(", round(100 * n_neg / length(Y), 2), "%)"))
print(paste("Positive class weight:", round(pos_weight, 2)))

# Create observation weights vector (1 for negative class, pos_weight for positive class)
obs_weights <- ifelse(Y == 1, pos_weight, 1)

# Define the Library of Base Learners
# --------------------------------------
# Create weighted versions of algorithms
# Note: Different algorithms handle weights differently
learner_library <- c(
  "SL.mean",
  "SL.glmnet",  # glmnet can use weights directly
  "SL.ranger.wt",  # Custom wrapper for weighted random forest
  "SL.xgboost.wt"  # Custom wrapper for weighted xgboost
)

# Define custom wrappers that incorporate weights
# -----------------------------------------------
# Weighted Ranger (Random Forest) wrapper
SL.ranger.wt <- function(Y, X, newX = NULL, family = gaussian(), obsWeights = NULL, ...) {
  SL.ranger(Y = Y, X = X, newX = newX, family = family, 
            obsWeights = obsWeights, 
            mtry = 10,
            splitrule = 'extratrees',
            min.node.size = 30,...)
}

# Weighted XGBoost wrapper
SL.xgboost.wt <- function(Y, X, newX = NULL, family = gaussian(), obsWeights = NULL, ...) {
  # Convert to matrix format required by xgboost
  xgmat <- model.matrix(~ . - 1, data = X)
  
  # Default parameters
  params <- list(
    objective = ifelse(family$family == "binomial", "binary:logistic", "reg:squarederror"),
    eval_metric = ifelse(family$family == "binomial", "logloss", "rmse"),
    eta = 0.05,
    gamma = 0.5,
    colsample_bytree = 1, 
    min_child_weight = 10,
    subsample = 0.8,
    max_depth = 5,
    verbose = 0
  )
  
  
  # If no weights provided, use equal weights
  if (is.null(obsWeights)) {
    obsWeights <- rep(1, length(Y))
  }
  
  # Train with weights
  fit <- xgboost::xgboost(
    data = xgmat,
    label = Y,
    weight = obsWeights,
    params = params,
    nrounds = 300, # changed from 100, loss was still reducing... 
    ...
  )
  
  # Predict
  pred <- NULL
  if (!is.null(newX)) {
    newX <- model.matrix(~ . - 1, data = newX)
    pred <- predict(fit, newX)
  }
  
  # Return
  fit <- list(object = fit)
  class(fit) <- "SL.xgboost"
  out <- list(fit = fit, pred = pred)
  return(out)
}

# Specify the Model Family
# ---------------------------
family_choice <- binomial()

# Train the SuperLearner Model
# -------------------------------
set.seed(1234) # Choose any seed number

# Run SuperLearner
print("Starting SuperLearner training with class weights...")

registerDoParallel(cores = (Sys.getenv("SLURM_NTASKS_PER_NODE")))

print("Generating inner and outer valdiation fold indices from pre-generated training folds...")
# Get Indices for Outer CV Validation Set from Training Set
# --------------------------------------
n_total <- nrow(train_yearFE_SL) # Total number of observations
train_indices_list <- folds

# Get all possible row indices
all_indices <- seq_len(n_total)

# Generate the validation indices list using setdiff
#    lapply iterates through each fold's training indices in train_indices_list
#    For each set of training indices (train_idx), it finds the indices in
#    all_indices that are NOT present in train_idx.
folds_validation_indices <- lapply(train_indices_list, function(train_idx) {
  setdiff(all_indices, train_idx)
})
rm(n_total, train_indices_list, all_indices)

# Set up Outer Cross-Validation Control
# --------------------------------------
num_folds <- length(folds)
cv_control <- list(V = num_folds, 
                   validRows = folds_validation_indices)
print(paste("Using pre-defined folds list with", num_folds, "folds."))

# Set up Inner CV folds
# --------------------------------------
# Take in set of indicies from outer folds (stored in folds_df)
# Returns validation sets for each fold based on 5-fold split
inner_cv_control <- create_inner_cvControls(folds_df)

# Train 
# --------------------------------------
m_CVSL <- CV.SuperLearner(
  Y = Y,                  # Outcome variable vector (0/1)
  X = X,                  # Predictor variable data frame
  family = family_choice, # binomial() for binary outcome
  SL.library = learner_library, # List of base learning algorithms
  id = id,                # Cluster identifier for V-fold CV
  cvControl = cv_control, # Control parameters for CV (includes V=5 folds)
  control = SuperLearner.control(saveCVFitLibrary = T),
  method = 'method.CC_nloglik',
  verbose = T,
  innerCvControl = inner_cv_control,
  obsWeights = obs_weights # Add observation weights based on class imbalance
)

print("SuperLearner cross-validation complete.")
print("Saving model...")
saveRDS(m_CVSL, file = "m_CVSL.rds")
print("Model saved")

# Train model for predictions 
m_SL <- SuperLearner(
  Y = Y,                  # Outcome variable vector (0/1)
  X = X,                  # Predictor variable data frame
  family = family_choice, # binomial() for binary outcome
  SL.library = learner_library, # List of base learning algorithms
  id = id,                # Cluster identifier for V-fold CV
  cvControl = cv_control, # Control parameters for CV (includes V=5 folds)
  method = 'method.CC_nloglik',
  verbose = T,
  obsWeights = obs_weights # Add observation weights based on class imbalance
)

print("SuperLearner training complete.")
print("Saving model...")
# Save models for output
saveRDS(m_SL, file = "m_SL.rds")
print("Models saved")