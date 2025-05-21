# =========================
# Load packages
# =========================
import pandas as pd
import sys
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterGrid

# =========================
# Load data
# =========================

data = pd.read_csv('.Data/data_5yr_base_omit_TRAIN_LSTM.csv')

# Load the CSV file containing the folds
folds_df = pd.read_csv('./Data/folds_LSTM.csv')
print(f"Successfully loaded folds_df with shape: {folds_df.shape}")

# Convert from dataframe format to list of indices format
folds = []
for col in folds_df.columns:
    if col.startswith('Fold'):
        indices = folds_df.index[folds_df[col] == 1].tolist()
        folds.append(indices)

print(f"Successfully converted {len(folds)} folds")

# Quick validation check after loading folds
if len(folds) != 5:  # Assuming 5-fold CV
    print("Warning: Expected 5 folds but got", len(folds))
    
# Make sure each fold has a reasonable number of indices
for i, fold in enumerate(folds):
    print(f"Fold {i+1} has {len(fold)} training indices")
    
def preprocess_data(data):
  """
  Preprocesses the data for LSTM model training.

  Args:
      data: pandas DataFrame containing the data.

  Returns:
      Tuple containing:
          X: NumPy array of input features.
          y: NumPy array of target variable.
          groups: NumPy array of district IDs for GroupKFold.
  """

  # Sort by agency_id and year
  # data = data.sort_values(['agency_id', 'year']) 
  # # Separate features (X), target (y), and group IDs
  # features = [col for col in data.columns if col not in ['agency_id', 'extreme_closure_10pct_over_5yr']]
  # X = data[features]
  # y = data['extreme_closure_10pct_over_5yr']
  # groups = data['agency_id']

  # Handle missing values (handled in R)
  # X = X.fillna(X.mean())

  # Scale features
  # scaler = StandardScaler()
  # X_scaled = scaler.fit_transform(X)
  
  # return X.values, y.values, groups.values 

  # Separate features (X), target (y), and group IDs
  features = [col for col in data.columns if col not in ['agency_id', 'extreme_closure_10pct_over_5yr']]
  X = data[features].values.astype(np.float32)  # Ensure float32 for TensorFlow
  y = data['extreme_closure_10pct_over_5yr'].values.astype(np.float32)
  groups = data['agency_id'].values  # Keep original dtype for agency_id

  return X, y, groups
    
def create_sequences_grouped(X, y, entities, seq_length):
    """
    Create sequences for X and corresponding target values from y.

    Args:
      X: NumPy array of shape (total_samples, num_features)
      y: NumPy array of target values of shape (total_samples,)
      entities: NumPy array of entity IDs (e.g., district IDs) for each sample.
      seq_length: Desired sequence length.
      
    Returns:
      X_seq: NumPy array of shape (n_sequences, seq_length, num_features)
      y_seq: NumPy array of shape (n_sequences, seq_length) -- the target for each sequence.
      Now also returns sequence_groups to maintain group alignment.
    """
    X_seq = []
    y_seq = []
    sequence_groups = []  # New list to track groups for each sequence

    # Split data by district
    unique_entities = np.unique(entities)
    for entity in unique_entities:
        idx = np.where(entities == entity)[0]
        X_entity = X[idx]
        y_entity = y[idx]

        num_entity_rows = X_entity.shape[0]
        if num_entity_rows < seq_length:
            continue

        # Generate sequences for this district
        for i in range(num_entity_rows - seq_length + 1):
            sequence = X_entity[i:i+seq_length]
            target_sequence = y_entity[i:i+seq_length]
            X_seq.append(sequence)
            y_seq.append(target_sequence)
            sequence_groups.append(entity)  # Add the district ID for this sequence

    # Convert to numpy arrays with appropriate data types
    X_seq_np = np.array(X_seq, dtype=np.float32)
    y_seq_np = np.array(y_seq, dtype=np.float32)
    
    # For sequence groups, store as separate array
    seq_groups_np = np.array(sequence_groups)
    
    return X_seq_np, y_seq_np, seq_groups_np

# =========================
# Custom Weighted Binary Cross-Entropy Loss
# =========================
def weighted_binary_crossentropy(pos_weight):
    """
    Returns a loss function that applies a weight to the positive examples.
    """
    def loss(y_true, y_pred):
        # Clip predictions to avoid log(0)
        y_pred = tf.clip_by_value(y_pred, keras.backend.epsilon(), 1 - keras.backend.epsilon())

        # Compute weighted binary cross-entropy for each element
        bce = -(pos_weight * y_true * tf.math.log(y_pred) +
                (1 - y_true) * tf.math.log(1 - y_pred))
        return tf.reduce_mean(bce)

    return loss

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
    inputs = layers.Input(shape=(seq_length, num_features))
    
    # First LSTM layer with regularization
    lstm1 = layers.LSTM(lstm_units, 
                        return_sequences = True,
                        kernel_regularizer = keras.regularizers.l2(0.01))(inputs)
    lstm1 = layers.Dropout(dropout_rate)(lstm1)
    
    # Second LSTM layer with increased units
    lstm2 = layers.LSTM(lstm_units * 2,
                        return_sequences=True)(lstm1)
    lstm2 = layers.Dropout(dropout_rate)(lstm2)
    
    # Self-attention mechanism
    attention = layers.MultiHeadAttention(
        num_heads = 4,  # Multiple attention heads to capture different patterns
        key_dim = lstm_units // 4  # Dimension of key/query vectors
    )(lstm2, lstm2)
    
    # Skip connection (residual connection)
    attention_output = layers.Add()([attention, lstm2])
    
    # Layer normalization (more stable than batch normalization for transformers)
    normalized = layers.LayerNormalization()(attention_output)
    
    # Dense layers for final processing
    dense1 = layers.TimeDistributed(
        layers.Dense(32, activation = 'relu')
    )(normalized)
    
    # Additional dropout for dense layer
    dense1 = layers.Dropout(dropout_rate/2)(dense1)
    
    # Output layer
    outputs = layers.TimeDistributed(
        layers.Dense(1, activation = 'sigmoid')
    )(dense1)
    
    # Squeeze out the last dimension
    outputs = layers.Reshape((seq_length,))(outputs)
    
    # Create model
    model = keras.Model(inputs = inputs, outputs = outputs)
    
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate = learning_rate)
    model.compile(
        optimizer = optimizer,
        loss = weighted_binary_crossentropy(positive_weight),
        metrics = [
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='pr_auc', curve='PR')
        ]
    )
    
    return model

def get_sequence_split_indices(seq_groups, current_fold, folds_df):
    """
    Get indices for training and validation splits for sequences.
    
    Args:
        seq_groups: Array containing group ID for each sequence
        current_fold: Current fold number (1-based)
        folds_df: DataFrame with fold assignments
        
    Returns:
        train_indices, val_indices: Arrays of indices for training and validation
    """
    # unique_groups = np.unique(seq_groups)
    
    # Get groups for training (from fold column)
    # fold_col = f"Fold{current_fold}"
    # training_groups = folds_df[folds_df[fold_col] == 1]['agency_id'].values
    # 
    # # Find indices for sequences that belong to training groups
    # train_indices = np.where(np.isin(seq_groups, training_groups))[0]
    # 
    # # Validation indices are all others
    # all_indices = np.arange(len(seq_groups))
    # val_indices = np.setdiff1d(all_indices, train_indices)
    
    # return train_indices, val_indices

    # Get groups for training (from fold column)
    fold_col = f"Fold{current_fold}"
    training_groups = folds_df[folds_df[fold_col] == 1]['agency_id'].values
    
    # Ensure consistent type comparison
    if seq_groups.dtype != training_groups.dtype:
        training_groups = np.array([str(g) for g in training_groups])
        seq_groups_str = np.array([str(g) for g in seq_groups])
        train_indices = np.where(np.isin(seq_groups_str, training_groups))[0]
    else:
        train_indices = np.where(np.isin(seq_groups, training_groups))[0]
    
    # Validation indices are all others
    all_indices = np.arange(len(seq_groups))
    val_indices = np.setdiff1d(all_indices, train_indices)
    
    return train_indices, val_indices

# =========================
# Design Unit-Based CV Approach
# =========================

# Function to run the cross-validation training
def grouped_cv(X, y, groups, n_splits, lstm_units, seq_length,
               dropout_rate, learning_rate, batch_size, num_epochs):
    """
    Performs group-based cross-validation with pre-computed sequences.
    """
    # Get dimensions for model building
    num_features = X.shape[1]
    print(f"Number of features: {num_features}")

    # Compute class balance from original data
    positive_count = np.sum(y)
    negative_count = y.size - positive_count
    positive_weight = negative_count / (positive_count + 1e-6)
    print(f"Positive class weight: {positive_weight}")

    # Pre-compute all sequences once
    print("Pre-computing sequences...")
    X_seq, y_seq, seq_groups = create_sequences_grouped(X, y, groups, seq_length)
    print(f"Created {len(X_seq)} sequences from {len(X)} samples")

    # Add callbacks for better training
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_pr_auc',
            patience=15,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_pr_auc',
            factor=0.5,
            patience=7,
            min_lr=0.00001
        )
    ]

    fold_metrics = []
    for fold in range(1, n_splits+1):
        print(f"\nStarting Fold {fold}")
        
        # Split the sequences based on groups
        train_indices, val_indices = get_sequence_split_indices(seq_groups, fold, folds_df)
        
        X_train, y_train = X_seq[train_indices], y_seq[train_indices]
        X_val, y_val = X_seq[val_indices], y_seq[val_indices]
        
        print(f"Training with {len(X_train)} sequences, validating with {len(X_val)} sequences")

        # Build and train model
        model = build_model(seq_length, num_features, lstm_units,
                            dropout_rate, learning_rate, positive_weight)
        
        history = model.fit(X_train, y_train,
                            batch_size=batch_size,
                            epochs=num_epochs,
                            validation_data=(X_val, y_val),
                            callbacks=callbacks,
                            verbose=2)
        
        # Evaluate
        val_metrics = model.evaluate(X_val, y_val, verbose=2)
        val_loss, val_recall, val_pr = val_metrics  
        
        fold_metrics.append((val_loss, val_recall, val_pr))
        print(f"Fold {fold} - Loss: {val_loss:.4f}, PR AUC: {val_pr:.4f}, Recall: {val_recall:.4f}")
    
    # Average metrics
    avg_loss = np.mean([m[0] for m in fold_metrics])
    avg_recall = np.mean([m[1] for m in fold_metrics])
    avg_pr = np.mean([m[2] for m in fold_metrics])
    
    return avg_loss, avg_recall, avg_pr, fold_metrics

# =========================
# TRAIN MODEL
# Prep data
# =========================

# Identify categorical columns
categorical_cols = data.select_dtypes(include = ['object', 'category']).columns

# Apply one-hot encoding to all categorical columns
data_dummies = pd.get_dummies(data, columns = categorical_cols, drop_first = True)

# Convert for LSTM input
X, y, groups = preprocess_data(data_dummies)

# COMMENTED OUT FULL HYPERPARAMETER GRID SEARCH
'''
# Define the hyperparameter grid
param_grid = {
    'lstm_units': [32, 64, 128],
    'dropout_rate': [0.2, 0.3, 0.4],
    'learning_rate': [0.0001, 0.001],
    'batch_size': [32, 64],
    'num_epochs': [50, 100],
    'seq_length': [3, 4, 5]
}

# Perform grid search with performance tracking
results = []
fold_metrics = []

for params in ParameterGrid(param_grid):
    print(f"Evaluating with parameters: {params}")
    
    # Use the optimized version with pre-computed sequences
    avg_loss, avg_rec, avg_pr, new_folds = grouped_cv(
        X, y, groups, n_splits=5, **params
    )
    
    print(f"Average PR AUC: {avg_pr:.4f}")

    # Store results for this parameter combination
    fold_metrics.append(new_folds)
    results.append({
        'params': params,
        'avg_loss': avg_loss,
        'avg_rec': avg_rec,
        'avg_pr': avg_pr
    })

# Save grid search results
def save_lstm_results(results, fold_metrics, timestamp=None):
    """
    Save LSTM model grid search results and fold metrics to CSV files
    
    Args:
        results: List of dictionaries with params and metrics
        fold_metrics: List of fold-specific metrics
        timestamp: Optional timestamp string for filename
    """
    import os
    import pandas as pd
    from datetime import datetime
    
    # Create Data directory if it doesn't exist
    data_dir = './Data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Generate timestamp if not provided
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame()
    for i, res in enumerate(results):
        # Extract parameters and metrics
        params = res['params']
        metrics = {k: v for k, v in res.items() if k != 'params'}
        
        # Combine parameters and metrics into a single row
        row_dict = {**params, **metrics}
        results_df = pd.concat([results_df, pd.DataFrame([row_dict])], ignore_index=True)
    
    # Save results
    results_path = os.path.join(data_dir, f'LSTM_grid_search_results_{timestamp}.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Saved grid search results to {results_path}")
    
    # Save best model configuration
    if results:
        best_idx = max(range(len(results)), key=lambda i: results[i]['avg_pr'])
        best_config = results[best_idx]
        best_config_path = os.path.join(data_dir, f'LSTM_best_model_config_{timestamp}.txt')
        
        with open(best_config_path, 'w') as f:
            f.write(f"Best LSTM Model Configuration (PR-AUC: {best_config['avg_pr']:.4f})\n")
            f.write("=" * 50 + "\n")
            for param, value in best_config['params'].items():
                f.write(f"{param}: {value}\n")
            f.write("\nPerformance Metrics:\n")
            f.write(f"Average Loss: {best_config['avg_loss']:.4f}\n")
            f.write(f"Average Recall: {best_config['avg_rec']:.4f}\n")
            f.write(f"Average PR-AUC: {best_config['avg_pr']:.4f}\n")
        
        print(f"Saved best model configuration to {best_config_path}")

# Add this after your grid search loop
save_lstm_results(results, fold_metrics)
'''

# Single CV run with specified hyperparameters
params = {
    'lstm_units': 128,
    'dropout_rate': 0.2,
    'learning_rate': 0.0001,
    'batch_size': 32,
    'num_epochs': 100,
    'seq_length': 5
}

avg_loss, avg_recall, avg_pr, fold_metrics = grouped_cv(
    X, y, groups,
    n_splits=5,
    **params
)

print("=== Cross-Validation Results ===")
print(f"Average Loss    : {avg_loss:.4f}")
print(f"Average Recall  : {avg_recall:.4f}")
print(f"Average PR-AUC  : {avg_pr:.4f}")

# Save fold_metrics for single CV run
import os
import pandas as pd

# Convert fold_metrics to DataFrame
fold_metrics_df = pd.DataFrame(fold_metrics, columns=['val_loss', 'val_recall', 'val_pr_auc'])
fold_metrics_df['fold'] = fold_metrics_df.index + 1

# Ensure Data directory exists
output_dir = './Data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Write to CSV
output_path = os.path.join(output_dir, 'fold_metrics_single_run.csv')
fold_metrics_df.to_csv(output_path, index=False)
print(f"Saved single-run fold metrics to {output_path}")



# =========================
# FINAL MODEL TRAINING & PREDICTION
# =========================
'''
# 1. Pre-compute sequences on the full dataset
X_seq_full, y_seq_full, seq_groups_full = create_sequences_grouped(
    X, y, groups, params['seq_length']
)
print(f"Full dataset: {len(X_seq_full)} sequences of length {params['seq_length']}")

# 2. Recompute positive class weight on the full data
pos_count = y.sum()
neg_count = y.size - pos_count
positive_weight = neg_count / (pos_count + 1e-6)

# 3. Build and train final model on all sequences
final_model = build_model(
    seq_length=params['seq_length'],
    num_features=X.shape[1],
    lstm_units=params['lstm_units'],
    dropout_rate=params['dropout_rate'],
    learning_rate=params['learning_rate'],
    positive_weight=positive_weight
)

final_model.fit(
    X_seq_full, 
    y_seq_full,
    batch_size=params['batch_size'],
    epochs=params['num_epochs'],
    verbose=2
)

# 4. Predict on the full sequence set
y_pred_seq = final_model.predict(
    X_seq_full,
    batch_size=params['batch_size']
)   # shape: (n_sequences, seq_length)

# 5. Flatten predictions and true labels
y_pred_flat = y_pred_seq.reshape(-1)
y_true_flat = y_seq_full.reshape(-1)
agency_ids   = np.repeat(seq_groups_full, params['seq_length'])

# 6. Save to a DataFrame
predictions_df = pd.DataFrame({
    'agency_id': agency_ids,
    'y_true'   : y_true_flat,
    'y_pred'   : y_pred_flat
})

predictions_df.to_csv('predictions_full_model.csv', index=False)
print("Saved predictions to predictions_full_model.csv")
'''