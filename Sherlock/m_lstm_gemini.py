# =========================
# Load packages
# =========================
import pandas as pd
import sys
import numpy as np
import os
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterGrid

# =========================
# Speed Up Training: Enable Mixed Precision (Optional, requires compatible GPU)
# =========================
# Use mixed precision to potentially speed up training on compatible GPUs (e.g., NVIDIA Volta, Turing, Ampere+)
# Comment this out if you don't have a compatible GPU or encounter issues.
try:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("Mixed precision policy set to 'mixed_float16'.")
except Exception as e:
    print(f"Could not set mixed precision policy: {e}. Using default float32.")
    tf.keras.mixed_precision.set_global_policy('float32')


# =========================
# Load data
# =========================

# --- Load your data ---
# Ensure these files are in the same directory as the script or provide full paths
try:
    data = pd.read_csv('data_5yr_base_omit_TRAIN_LSTM.csv')
    print(f"Successfully loaded data_df with shape: {data.shape}")
except FileNotFoundError:
    print("Error: 'data_5yr_base_omit_TRAIN_LSTM.csv' not found. Please ensure the file exists.")
    sys.exit(1) # Exit if data file is missing

try:
    folds_df = pd.read_csv('folds_LSTM.csv')
    print(f"Successfully loaded folds_df with shape: {folds_df.shape}")
except FileNotFoundError:
    print("Error: 'folds_LSTM.csv' not found. Please ensure the file exists.")
    sys.exit(1) # Exit if folds file is missing


# Convert from dataframe format to list of indices format (Original code - seems unused, keeping for context)
# folds = []
# for col in folds_df.columns:
#     if col.startswith('Fold'):
#         indices = folds_df.index[folds_df[col] == 1].tolist()
#         folds.append(indices)
# print(f"Successfully converted {len(folds)} folds (Note: This list 'folds' is not directly used in the CV logic below)")


# =========================
# Data Preprocessing
# =========================
def preprocess_data(data):
    """
    Preprocesses the data for LSTM model training.

    Args:
        data: pandas DataFrame containing the data (already one-hot encoded).

    Returns:
        Tuple containing:
            X: NumPy array of input features (float32).
            y: NumPy array of target variable (float32).
            groups: NumPy array of district IDs for grouping.
    """
    # Separate features (X), target (y), and group IDs
    features = [col for col in data.columns if col not in ['agency_id', 'extreme_closure_10pct_over_5yr']]
    X = data[features].values.astype(np.float32)  # Ensure float32 for TensorFlow
    y = data['extreme_closure_10pct_over_5yr'].values.astype(np.float32)
    groups = data['agency_id'].values # Keep original dtype for agency_id grouping

    # Feature Scaling (Optional but recommended)
    # Apply scaling *after* separating features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X) # Fit and transform

    print(f"Preprocessing complete. X shape: {X_scaled.shape}, y shape: {y.shape}, groups shape: {groups.shape}")

    return X_scaled, y, groups

# =========================
# Sequence Creation
# =========================
def create_sequences_grouped(X, y, entities, seq_length):
    """
    Create sequences for X and corresponding target values from y, grouped by entity.

    Args:
        X: NumPy array of shape (total_samples, num_features) - Scaled features.
        y: NumPy array of target values of shape (total_samples,)
        entities: NumPy array of entity IDs (e.g., district IDs) for each sample.
        seq_length: Desired sequence length.

    Returns:
        X_seq: NumPy array of shape (n_sequences, seq_length, num_features), dtype=float32.
        y_seq: NumPy array of shape (n_sequences, seq_length), dtype=float32 - the target for each step in the sequence.
        sequence_groups: NumPy array containing the entity ID for each generated sequence.
    """
    X_seq_list = []
    y_seq_list = []
    sequence_groups_list = [] # List to track groups for each sequence

    unique_entities = np.unique(entities)
    print(f"Processing {len(unique_entities)} unique entities...")

    for entity_count, entity in enumerate(unique_entities):
        idx = np.where(entities == entity)[0]
        X_entity = X[idx]
        y_entity = y[idx]

        num_entity_rows = X_entity.shape[0]
        if num_entity_rows < seq_length:
            # print(f"Skipping entity {entity}: Not enough data ({num_entity_rows} rows) for sequence length {seq_length}")
            continue # Skip entities with insufficient data

        # Generate sequences for this entity
        for i in range(num_entity_rows - seq_length + 1):
            sequence = X_entity[i:i+seq_length]
            target_sequence = y_entity[i:i+seq_length] # Target for each time step
            X_seq_list.append(sequence)
            y_seq_list.append(target_sequence)
            sequence_groups_list.append(entity) # Add the entity ID for this sequence

        # Optional: Print progress periodically
        # if (entity_count + 1) % 100 == 0:
        #     print(f"Processed {entity_count + 1}/{len(unique_entities)} entities...")

    if not X_seq_list:
        print("Warning: No sequences were generated. Check data and sequence length.")
        return np.array([]), np.array([]), np.array([])

    # Convert lists to numpy arrays with appropriate data types
    X_seq_np = np.array(X_seq_list, dtype=np.float32)
    y_seq_np = np.array(y_seq_list, dtype=np.float32)
    seq_groups_np = np.array(sequence_groups_list) # Dtype depends on original group IDs

    print(f"Sequence creation finished. X_seq shape: {X_seq_np.shape}, y_seq shape: {y_seq_np.shape}, seq_groups shape: {seq_groups_np.shape}")

    return X_seq_np, y_seq_np, seq_groups_np


# =========================
# Custom Weighted Binary Cross-Entropy Loss
# =========================
def weighted_binary_crossentropy(pos_weight):
    """
    Returns a loss function that applies a weight to the positive examples.
    Handles potential float16 inputs if using mixed precision.
    """
    def loss(y_true, y_pred):
        # Ensure y_true is float32 for calculations, as it might not be handled by mixed precision policy
        y_true = tf.cast(y_true, tf.float32)
        # y_pred should be float32 *output* from the model before activation usually,
        # but casting here ensures compatibility if the final layer output is float16.
        y_pred = tf.cast(y_pred, tf.float32)

        # Clip predictions to avoid log(0)
        epsilon_ = tf.keras.backend.epsilon() # Use TensorFlow's epsilon
        y_pred = tf.clip_by_value(y_pred, epsilon_, 1.0 - epsilon_)

        # Compute weighted binary cross-entropy
        bce = -(pos_weight * y_true * tf.math.log(y_pred) +
                (1.0 - y_true) * tf.math.log(1.0 - y_pred))

        # Return the mean loss over the batch
        return tf.reduce_mean(bce)

    return loss

# =========================
# Model Building
# =========================
def build_model(seq_length, num_features, lstm_units,
                dropout_rate, learning_rate, positive_weight):
    """
    Build LSTM model with attention mechanism for time series prediction.

    Args:
        seq_length: Length of input sequences
        num_features: Number of input features
        lstm_units: Number of LSTM units in first layer
        dropout_rate: Dropout rate for regularization
        learning_rate: Learning rate for optimizer
        positive_weight: Weight for positive class in loss function
    """
    # Input layer
    inputs = layers.Input(shape=(seq_length, num_features), dtype=tf.float32) # Specify dtype

    # First LSTM layer with regularization
    # Consider using CuDNNLSTM if running on GPU for potential speedup,
    # but standard LSTM is more general.
    lstm1 = layers.LSTM(lstm_units,
                        return_sequences=True,
                        kernel_regularizer=keras.regularizers.l2(0.01))(inputs)
    lstm1 = layers.Dropout(dropout_rate)(lstm1)

    # Second LSTM layer with increased units
    lstm2 = layers.LSTM(lstm_units * 2,
                        return_sequences=True)(lstm1)
    lstm2 = layers.Dropout(dropout_rate)(lstm2)

    # Self-attention mechanism
    # Ensure key_dim is reasonable, especially with mixed precision
    attention_key_dim = max(1, lstm_units // 4) # Ensure key_dim is at least 1
    attention = layers.MultiHeadAttention(
        num_heads=4, # Number of attention heads
        key_dim=attention_key_dim
    )(query=lstm2, value=lstm2, key=lstm2) # Use lstm2 for query, key, and value

    # Skip connection (residual connection)
    attention_output = layers.Add()([attention, lstm2])

    # Layer normalization (generally preferred over BatchNormalization for sequence data)
    normalized = layers.LayerNormalization()(attention_output)

    # TimeDistributed Dense layers for processing each time step
    dense1 = layers.TimeDistributed(
        layers.Dense(32, activation='relu')
    )(normalized)
    dense1 = layers.Dropout(dropout_rate / 2)(dense1) # Reduced dropout

    # Output layer - Sigmoid activation for binary classification per time step
    # The Dense layer inside TimeDistributed will have float32 output by default
    # unless the mixed precision policy forces it to float16.
    # The loss function handles casting back to float32 if needed.
    outputs = layers.TimeDistributed(
        layers.Dense(1, activation='sigmoid') # Output one probability per time step
    )(dense1)

    # Squeeze the last dimension if needed, but the current shape (batch, seq_length, 1)
    # might be compatible with the loss function depending on how y_seq is shaped.
    # Let's keep the shape (batch, seq_length, 1) and ensure y_seq matches (batch, seq_length)
    # The loss function expects y_true (batch, seq_length) and y_pred (batch, seq_length, 1) or (batch, seq_length)
    # Reshape to (batch, seq_length) to match typical y_true shape.
    outputs = layers.Reshape((seq_length,))(outputs) # Shape: (batch_size, seq_length)

    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    # If using mixed precision, wrap the optimizer
    if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
         optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

    model.compile(
        optimizer=optimizer,
        loss=weighted_binary_crossentropy(positive_weight),
        metrics=[
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='pr_auc', curve='PR'),
            # Add Loss as a metric to track it alongside custom loss function value
            tf.keras.metrics.BinaryCrossentropy(name='bce_loss') # Standard BCE for reference
        ]
    )

    return model

# =========================
# Sequence Splitting Logic
# =========================
def get_sequence_split_indices(seq_groups, current_fold, folds_df):
    """
    Get indices for training and validation splits for sequences based on pre-defined folds.

    Args:
        seq_groups: Array containing group ID for each sequence.
        current_fold: Current fold number (1-based).
        folds_df: DataFrame with fold assignments (agency_id vs Fold columns).

    Returns:
        train_indices, val_indices: NumPy arrays of indices for training and validation sequences.
    """
    fold_col = f"Fold{current_fold}"
    if fold_col not in folds_df.columns:
        raise ValueError(f"Column {fold_col} not found in folds_df.")

    # Get agency IDs designated for *training* in this fold (where FoldX == 1)
    # Make sure 'agency_id' column exists and handle potential type mismatches
    if 'agency_id' not in folds_df.columns:
        raise ValueError("'agency_id' column not found in folds_df.")

    training_groups = folds_df.loc[folds_df[fold_col] == 1, 'agency_id'].unique()

    # Ensure consistent types for comparison (e.g., if one is int and other is str)
    try:
        if seq_groups.dtype != training_groups.dtype:
            print(f"Type mismatch: seq_groups ({seq_groups.dtype}) vs training_groups ({training_groups.dtype}). Attempting conversion.")
            # Attempt to convert seq_groups to the type of training_groups
            seq_groups_typed = seq_groups.astype(training_groups.dtype)
        else:
             seq_groups_typed = seq_groups
    except Exception as e:
         print(f"Warning: Could not ensure type consistency between seq_groups and training_groups: {e}. Trying string conversion as fallback.")
         seq_groups_typed = seq_groups.astype(str)
         training_groups = training_groups.astype(str)


    # Find indices where the sequence's group is in the set of training groups for this fold
    try:
        train_indices = np.where(np.isin(seq_groups_typed, training_groups))[0]
    except TypeError as e:
        print(f"Error during np.isin comparison: {e}. This might indicate incompatible types even after conversion attempts.")
        # Fallback: Iterate and compare (slower)
        train_indices_list = []
        training_groups_set = set(training_groups)
        for i, group in enumerate(seq_groups_typed):
            if group in training_groups_set:
                train_indices_list.append(i)
        train_indices = np.array(train_indices_list)
        print("Used fallback comparison method.")


    # Validation indices are all sequence indices *not* in the training set for this fold
    all_indices = np.arange(len(seq_groups))
    val_indices = np.setdiff1d(all_indices, train_indices, assume_unique=True) # assume_unique can speed up if indices are indeed unique

    if len(train_indices) == 0 or len(val_indices) == 0:
         print(f"Warning: Fold {current_fold} resulted in an empty split. Train size: {len(train_indices)}, Val size: {len(val_indices)}. Check fold definitions and group assignments.")

    return train_indices, val_indices


# =========================
# Cross-Validation Function
# =========================
def grouped_cv(X, y, groups, folds_df, n_splits, lstm_units, seq_length,
               dropout_rate, learning_rate, batch_size, num_epochs, patience,
               early_stopping_monitor='val_loss'): # Added patience and monitor params
    """
    Performs group-based cross-validation using pre-defined folds and pre-computed sequences.

    Args:
        X, y, groups: Preprocessed data.
        folds_df: DataFrame defining the fold splits by group ('agency_id').
        n_splits: Number of folds (should match folds_df).
        lstm_units, seq_length, dropout_rate, learning_rate, batch_size, num_epochs: Hyperparameters.
        patience: Patience for EarlyStopping and ReduceLROnPlateau.
        early_stopping_monitor: Metric to monitor for early stopping ('val_loss' or 'val_pr_auc').

    Returns:
        avg_loss, avg_recall, avg_pr: Average metrics across folds.
        fold_details: List of dictionaries, each containing metrics for a single fold.
                      Example: [{'fold': 1, 'loss': X, 'recall': Y, 'pr_auc': Z}, ...]
    """
    num_features = X.shape[1]
    print(f"Number of features: {num_features}")

    # Compute class balance from original data (before sequencing)
    positive_count = np.sum(y == 1)
    negative_count = y.size - positive_count
    if positive_count == 0:
        print("Warning: No positive examples found in the dataset. Check target variable 'y'.")
        positive_weight = 1.0 # Avoid division by zero, but weighting is meaningless
    else:
        positive_weight = negative_count / positive_count
    print(f"Positive class weight calculated: {positive_weight:.4f} (negatives: {negative_count}, positives: {positive_count})")

    # Pre-compute all sequences once
    print(f"Pre-computing sequences with seq_length={seq_length}...")
    X_seq, y_seq, seq_groups = create_sequences_grouped(X, y, groups, seq_length)

    if X_seq.size == 0:
         print("Error: No sequences were generated. Cannot proceed with CV.")
         # Return dummy values or raise an error
         return np.nan, np.nan, np.nan, []


    print(f"Created {len(X_seq)} sequences.")
    # Note: y_seq shape is (n_sequences, seq_length). Ensure model output and loss handle this.

    # --- Callbacks ---
    # UPDATED: Monitor val_loss and increased patience
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor=early_stopping_monitor, # Monitor validation loss
        patience=patience,              # Increased patience
        restore_best_weights=True,      # Restore weights from the epoch with the best monitored value
        verbose=1
    )
    # UPDATED: Monitor val_loss for ReduceLROnPlateau as well
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=early_stopping_monitor, # Monitor the same metric as EarlyStopping
        factor=0.5,                     # Reduce LR by half
        patience=patience // 2,         # Reduce LR sooner than stopping (e.g., half the patience)
        min_lr=1e-6,                    # Minimum learning rate
        verbose=1
    )
    callbacks = [early_stopping, reduce_lr]

    fold_details = [] # Store detailed metrics for each fold

    for fold in range(1, n_splits + 1):
        print(f"\n--- Starting Fold {fold}/{n_splits} ---")

        # Split the *sequences* based on group assignments in folds_df
        try:
            train_indices, val_indices = get_sequence_split_indices(seq_groups, fold, folds_df)
        except ValueError as e:
            print(f"Error getting split indices for fold {fold}: {e}. Skipping fold.")
            continue # Skip this fold if there's an error

        if len(train_indices) == 0 or len(val_indices) == 0:
            print(f"Skipping Fold {fold} due to empty train or validation set after splitting sequences.")
            # Append NaN results for this fold to maintain structure if needed, or just skip
            fold_metrics = {'fold': fold, 'loss': np.nan, 'recall': np.nan, 'pr_auc': np.nan, 'bce_loss': np.nan}
            fold_details.append(fold_metrics)
            continue


        X_train, y_train = X_seq[train_indices], y_seq[train_indices]
        X_val, y_val = X_seq[val_indices], y_seq[val_indices]

        print(f"Fold {fold}: Training sequences: {len(X_train)}, Validation sequences: {len(X_val)}")
        # Check class balance in fold splits (optional but useful)
        print(f"Fold {fold} Train: {np.sum(y_train):.0f} positive steps out of {y_train.size} total steps")
        print(f"Fold {fold} Val:   {np.sum(y_val):.0f} positive steps out of {y_val.size} total steps")


        # Build and train model for the current fold
        # Ensure the model is rebuilt for each fold to reset weights
        tf.keras.backend.clear_session() # Clear previous model graph
        model = build_model(seq_length, num_features, lstm_units,
                            dropout_rate, learning_rate, positive_weight)

        print(f"Fold {fold}: Starting training...")
        history = model.fit(X_train, y_train,
                            batch_size=batch_size,
                            epochs=num_epochs,
                            validation_data=(X_val, y_val),
                            callbacks=callbacks,
                            verbose=2) # Use verbose=2 for less output per epoch

        # Evaluate on validation set (using weights restored by EarlyStopping)
        print(f"Fold {fold}: Evaluating model on validation data...")
        # model.evaluate returns: [loss, recall, pr_auc, bce_loss] based on compile metrics order
        val_metrics = model.evaluate(X_val, y_val, batch_size=batch_size, verbose=0) # Verbose 0 for cleaner logs

        # Store metrics for this fold
        fold_metrics = {
            'fold': fold,
            'loss': val_metrics[0], # The custom weighted loss value
            'recall': val_metrics[1],
            'pr_auc': val_metrics[2],
            'bce_loss': val_metrics[3] # Standard BCE loss metric
        }
        fold_details.append(fold_metrics)

        print(f"Fold {fold} - Validation Metrics -> Loss: {fold_metrics['loss']:.4f}, Recall: {fold_metrics['recall']:.4f}, PR AUC: {fold_metrics['pr_auc']:.4f}, BCE Loss: {fold_metrics['bce_loss']:.4f}")
        print(f"Fold {fold}: Training stopped after {len(history.history['loss'])} epochs (stopped early: {early_stopping.stopped_epoch > 0}).")


    # Calculate average metrics across successful folds
    valid_folds = [f for f in fold_details if not np.isnan(f['loss'])] # Filter out folds with NaN results
    if not valid_folds:
        print("Warning: No folds completed successfully.")
        return np.nan, np.nan, np.nan, fold_details

    avg_loss = np.mean([f['loss'] for f in valid_folds])
    avg_recall = np.mean([f['recall'] for f in valid_folds])
    avg_pr = np.mean([f['pr_auc'] for f in valid_folds])
    avg_bce = np.mean([f['bce_loss'] for f in valid_folds])

    print("\n--- Cross-Validation Summary ---")
    print(f"Average Validation Loss (Weighted): {avg_loss:.4f}")
    print(f"Average Validation Recall:          {avg_recall:.4f}")
    print(f"Average Validation PR AUC:          {avg_pr:.4f}")
    print(f"Average Validation BCE Loss:        {avg_bce:.4f}")

    return avg_loss, avg_recall, avg_pr, fold_details # Return detailed fold metrics


# =========================
# Data Preparation & Grid Search Setup
# =========================
print("\n--- Preparing Data for Training ---")
# Identify categorical columns (assuming 'agency_id' is identifier, not feature)
categorical_cols = data.select_dtypes(include=['object', 'category']).columns
# Exclude agency_id if it's present and treated as identifier
categorical_cols = [col for col in categorical_cols if col != 'agency_id']

print(f"Identified categorical columns for encoding: {list(categorical_cols)}")

# Apply one-hot encoding
if list(categorical_cols): # Check if there are any categorical columns to encode
    data_dummies = pd.get_dummies(data, columns=categorical_cols, drop_first=True, dummy_na=False) # drop_first=True avoids multicollinearity
    print("Applied one-hot encoding.")
else:
    data_dummies = data.copy() # No categorical columns to encode
    print("No categorical columns found to encode.")


# Convert for LSTM input (includes scaling)
X, y, groups = preprocess_data(data_dummies)

# Define the hyperparameter grid
param_grid = {
    'lstm_units': [64, 128],             # Reduced options for faster testing
    'dropout_rate': [0.3, 0.4],         # Reduced options
    'learning_rate': [0.001, 0.0005],   # Adjusted learning rates
    'batch_size': [64, 128],            # Larger batch sizes might speed up epoch time
    'num_epochs': [150],                # Increased max epochs, relies on early stopping
    'seq_length': [4, 5],               # Sequence lengths to test
    'patience': [30],                   # UPDATED: Increased patience for early stopping
    'early_stopping_monitor': ['val_loss'] # UPDATED: Monitor validation loss
}

# Convert grid to list of parameter dictionaries
grid = list(ParameterGrid(param_grid))
print(f"\n--- Starting Grid Search with {len(grid)} Parameter Combinations ---")

# =========================
# Grid Search Execution
# =========================
results = []
all_fold_metrics = [] # List to store the detailed fold metrics for *each* param set

for i, params in enumerate(grid):
    print(f"\n--- Evaluating Parameter Set {i+1}/{len(grid)} ---")
    print(f"Parameters: {params}")

    # Clear session before each CV run for different params
    tf.keras.backend.clear_session()

    # Run cross-validation for the current parameter set
    avg_loss, avg_rec, avg_pr, fold_details = grouped_cv(
        X, y, groups, folds_df, n_splits=5, **params # Pass folds_df here
    )

    print(f"\nParameter Set {i+1} Results:")
    print(f"  Average Validation Loss (Weighted): {avg_loss:.4f}")
    print(f"  Average Validation Recall:          {avg_rec:.4f}")
    print(f"  Average Validation PR AUC:          {avg_pr:.4f}")

    # Store average results for this parameter combination
    results.append({
        'params': params,
        'avg_loss': avg_loss,
        'avg_rec': avg_rec,
        'avg_pr': avg_pr
    })

    # Store the detailed fold metrics for this parameter set
    # Add the parameters to each fold's dictionary for easy tracking in the final CSV
    for fold_data in fold_details:
        fold_data.update(params) # Add current hyperparameters to the fold's metrics dict
    all_fold_metrics.extend(fold_details) # Add the list of fold dicts to the overall list


# =========================
# Save Results
# =========================
def save_lstm_results(results, all_fold_metrics, timestamp=None):
    """
    Save LSTM model grid search average results and detailed fold metrics to CSV files.

    Args:
        results: List of dictionaries with params and average metrics per param set.
        all_fold_metrics: List of dictionaries, each containing metrics for a specific fold
                          and the parameters used for that fold's run.
        timestamp: Optional timestamp string for filename.
    """
    # Create Data directory if it doesn't exist
    data_dir = './Data'
    if not os.path.exists(data_dir):
        try:
            os.makedirs(data_dir)
            print(f"Created directory: {data_dir}")
        except OSError as e:
            print(f"Error creating directory {data_dir}: {e}. Saving to current directory.")
            data_dir = '.' # Fallback to current directory

    # Generate timestamp if not provided
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- Save Average Results ---
    if results:
        results_df = pd.DataFrame()
        for i, res in enumerate(results):
            params = res['params']
            metrics = {k: v for k, v in res.items() if k != 'params'}
            row_dict = {**params, **metrics} # Combine params and avg metrics
            results_df = pd.concat([results_df, pd.DataFrame([row_dict])], ignore_index=True)

        # Sort by the primary metric (e.g., avg_pr descending)
        if 'avg_pr' in results_df.columns:
             results_df = results_df.sort_values(by='avg_pr', ascending=False)

        results_path = os.path.join(data_dir, f'LSTM_grid_search_avg_results_{timestamp}.csv')
        try:
            results_df.to_csv(results_path, index=False)
            print(f"Saved average grid search results to {results_path}")
        except Exception as e:
            print(f"Error saving average results CSV: {e}")
    else:
        print("No average results to save.")


    # --- Save Detailed Fold Metrics ---
    if all_fold_metrics:
        # Convert the list of fold dictionaries directly to a DataFrame
        fold_metrics_df = pd.DataFrame(all_fold_metrics)

        # Reorder columns for clarity (optional)
        param_cols = list(param_grid.keys()) # Get parameter names
        metric_cols = ['fold', 'loss', 'recall', 'pr_auc', 'bce_loss']
        # Ensure all expected columns exist before reordering
        existing_cols = [col for col in param_cols + metric_cols if col in fold_metrics_df.columns]
        fold_metrics_df = fold_metrics_df[existing_cols]


        fold_results_path = os.path.join(data_dir, f'LSTM_grid_search_fold_details_{timestamp}.csv')
        try:
            fold_metrics_df.to_csv(fold_results_path, index=False)
            print(f"Saved detailed fold metrics to {fold_results_path}")
        except Exception as e:
            print(f"Error saving detailed fold metrics CSV: {e}")

    else:
        print("No detailed fold metrics to save.")


    # --- Save Best Model Configuration (based on average PR AUC) ---
    if results:
        # Find the result with the highest average PR AUC (handle NaNs)
        valid_results = [r for r in results if 'avg_pr' in r and not np.isnan(r['avg_pr'])]
        if valid_results:
            best_idx = max(range(len(valid_results)), key=lambda i: valid_results[i]['avg_pr'])
            best_config = valid_results[best_idx]
            best_config_path = os.path.join(data_dir, f'LSTM_best_model_config_{timestamp}.txt')

            try:
                with open(best_config_path, 'w') as f:
                    f.write(f"Best LSTM Model Configuration (based on Average PR-AUC: {best_config['avg_pr']:.4f})\n")
                    f.write("=" * 60 + "\n")
                    f.write("Hyperparameters:\n")
                    for param, value in best_config['params'].items():
                        f.write(f"  {param}: {value}\n")
                    f.write("\nAverage Performance Metrics across Folds:\n")
                    f.write(f"  Average Loss (Weighted): {best_config['avg_loss']:.4f}\n")
                    f.write(f"  Average Recall:          {best_config['avg_rec']:.4f}\n")
                    f.write(f"  Average PR-AUC:          {best_config['avg_pr']:.4f}\n")
                print(f"Saved best model configuration summary to {best_config_path}")
            except Exception as e:
                 print(f"Error saving best model config text file: {e}")
        else:
            print("Could not determine best model configuration (no valid PR AUC scores found).")


# --- Run the saving function after the grid search loop completes ---
print("\n--- Grid Search Complete. Saving Results... ---")
save_lstm_results(results, all_fold_metrics)

print("\n--- Script Finished ---")