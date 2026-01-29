# utils/model_utils.py
import numpy as np
import tensorflow as tf
import pandas as pd
import os
import joblib
import sqlite3
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from backend import config
import matplotlib.pyplot as plt

def _fetch_data(config, db_path: str, last_retraining_datetime: str, train_from_scratch: bool = False):
    """
    Legacy helper: fetches historical data from the ground_truth_table for retraining.
    Filters out temperature data marked as forecasted.
    NOTE: RetrainingAgent now uses its own _fetch_full_training_data method instead.
    """
    print(f"[ModelUtils] Loading historical data from '{config.GROUND_TRUTH_TABLE}'...")
    conn = None

    try:
        conn = sqlite3.connect(db_path)
        
        if train_from_scratch:
            print("[ModelUtils] Fetching ALL historical data for training from scratch (ONLY non-forecasted temp).")
            # Fetch all data, still filter out forecasted temp
            query = f"""
                SELECT * 
                FROM {config.GROUND_TRUTH_TABLE} 
                WHERE temp_is_forecasted = 0 
                ORDER BY {config.TIME_COLUMN}
            """
            df = pd.read_sql_query(query, conn, parse_dates=[config.TIME_COLUMN])
        else:
            print(f"[ModelUtils] Fetching data since {last_retraining_datetime}, excluding forecasted temp.")
            query = f"""
                SELECT * 
                FROM {config.GROUND_TRUTH_TABLE} 
                WHERE {config.TIME_COLUMN} > ? AND temp_is_forecasted = 0 
                ORDER BY {config.TIME_COLUMN}
            """
            df = pd.read_sql_query(query, conn, params=(last_retraining_datetime,), parse_dates=[config.TIME_COLUMN])
        
        conn.close()
        print(f"[ModelUtils] ✅ Data loaded. Total rows fetched for training: {len(df)}")
        return df
        
    except Exception as e:
        print(f"[ModelUtils] ❌ Error loading data from DB: {e}")
        return None

def _process_features_and_target(df: pd.DataFrame):
    """
    Standard feature engineering for retraining.

    - Ensures TIME_COLUMN is datetime
    - Creates:
        - hour_extracted
        - minute_extracted
        - weekday (0=Mon,...,6=Sun)
        - is_weekend (1 if Sat/Sun, else 0)
        - eprice_lag1 (log1p-lag of TARGET_COLUMN)
    - Selects features using config.LSTM_FEATURE_COLUMNS
    - Applies log1p to TARGET_COLUMN and to eprice_lag1
    """
    if df is None or df.empty:
        print("[ModelUtils] ❌ Input DataFrame is empty.")
        return None, None
    
    # 1. Time handling
    if config.TIME_COLUMN not in df.columns:
        print(f"[ModelUtils] ❌ Time column '{config.TIME_COLUMN}' not found in DataFrame.")
        return None, None

    if not pd.api.types.is_datetime64_any_dtype(df[config.TIME_COLUMN]):
        try:
            df[config.TIME_COLUMN] = pd.to_datetime(df[config.TIME_COLUMN], utc=True, errors="coerce")
            print(f"[ModelUtils] ✅ Converted '{config.TIME_COLUMN}' to datetime type.")
        except Exception as e:
            print(f"[ModelUtils] ❌ Error converting '{config.TIME_COLUMN}' to datetime: {e}. Aborting.")
            return None, None

    df = df.sort_values(config.TIME_COLUMN).reset_index(drop=True)

    # 2. Feature engineering from time 
    if 'hour_extracted' not in df.columns:
        df['hour_extracted'] = df[config.TIME_COLUMN].dt.hour
        print("[ModelUtils] ✅ Created 'hour_extracted' feature.")

    if 'minute_extracted' not in df.columns:
        df['minute_extracted'] = df[config.TIME_COLUMN].dt.minute
        print("[ModelUtils] ✅ Created 'minute_extracted' feature.")

    # Weekday / weekend always created on the fly
    df['weekday'] = df[config.TIME_COLUMN].dt.dayofweek  # 0=Mon, 6=Sun
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)

    # 3. Target handling
    if config.TARGET_COLUMN not in df.columns:
        print(f"[ModelUtils] ❌ Target column '{config.TARGET_COLUMN}' missing.")
        return None, None

    if not pd.api.types.is_numeric_dtype(df[config.TARGET_COLUMN]):
        try:
            df[config.TARGET_COLUMN] = pd.to_numeric(df[config.TARGET_COLUMN], errors='coerce')
            print(f"[ModelUtils] ✅ Converted '{config.TARGET_COLUMN}' to numeric type.")
        except Exception as e:
            print(f"[ModelUtils] ❌ Error converting '{config.TARGET_COLUMN}' to numeric: {e}.")
            return None, None
    
    # 4. Lag feature in log-domain
    # First convert target to log1p for lag creation
    df['target_log1p'] = np.log1p(df[config.TARGET_COLUMN])
    df['eprice_lag1'] = df['target_log1p'].shift(1)
    df['eprice_lag1'].bfill(inplace=True)

    # 5. Ensure required feature columns exist
    missing_features = [col for col in config.LSTM_FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        print(f"[ModelUtils] ❌ Missing required feature columns in data: {missing_features}. Aborting retraining.")
        return None, None

    features_raw = df[config.LSTM_FEATURE_COLUMNS].copy()
    target_raw = df[[config.TARGET_COLUMN]].copy()

    # 6. Clean NaNs before scaling
    df_clean = pd.concat([features_raw, target_raw], axis=1).dropna()
    if df_clean.empty:
        print("[ModelUtils] ❌ All rows dropped due to NaNs.")
        return None, None
    
    # Apply log1p to target and to lag (consistent with ForecastingAgent)
    df_clean[config.TARGET_COLUMN] = np.log1p(df_clean[config.TARGET_COLUMN])

    features_clean = df_clean[features_raw.columns] # Re-select to ensure order and avoid extra cols
    target_clean = df_clean[[config.TARGET_COLUMN]]

    if features_clean.empty or target_clean.empty:
        print("[ModelUtils] ❌ Features or target are empty after dropping NaNs. Cannot proceed with scaling.")
        return None, None
    
    print(f"[ModelUtils] ✅ Data cleaned. Rows after NaN drop: {len(features_clean)}")

    return features_clean, target_clean

def _get_fitted_scalers(features_df: pd.DataFrame, target_df: pd.DataFrame):
    """
    Fits and returns scalers for the given data.

    - Uses config.MM_SCALED_COLUMNS and config.R_SCALED_COLUMNS.
    - Any feature in config.LSTM_FEATURE_COLUMNS that is not listed in either
      scaler group is automatically added to MinMax scaling.
    """    
    r_scaler = RobustScaler()
    mm_scaler = MinMaxScaler()
    target_scaler = RobustScaler() 
    
    # Columns explicitly configured
    r_cols = [c for c in config.R_SCALED_COLUMNS if c in features_df.columns]
    mm_cols = [c for c in config.MM_SCALED_COLUMNS if c in features_df.columns]
    
    # Ensure all features used by the LSTM are scaled by one of the two scalers
    for col in config.LSTM_FEATURE_COLUMNS:
        if col in features_df.columns and col not in mm_cols and col not in r_cols:
            mm_cols.append(col)
            print(f"[ModelUtils] ℹ️ Auto-assigning '{col}' to MinMaxScaler group.")

    if not mm_cols and not r_cols:
        print("[ModelUtils] ⚠️ No feature columns selected for scaling. Check config.MM_SCALED_COLUMNS / R_SCALED_COLUMNS.")

    print(f"[ModelUtils] Fitting MinMaxScaler on: {mm_cols}")
    if mm_cols:
        mm_scaler.fit(features_df[mm_cols])

    print(f"[ModelUtils] Fitting RobustScaler on: {r_cols}")
    if r_cols:
        r_scaler.fit(features_df[r_cols])
        
    # Target scaler is fit on log-space targets
    target_scaler.fit(target_df)

    # Apply scalers
    scaled_features_df = features_df.copy()
    if mm_cols:
        scaled_features_df[mm_cols] = mm_scaler.transform(features_df[mm_cols])
    if r_cols:
        scaled_features_df[r_cols] = r_scaler.transform(features_df[r_cols])

    target_scaled = target_scaler.transform(target_df)

    print("[ModelUtils] ✅ Scalers fitted and data transformed successfully.")
    return mm_scaler, r_scaler, target_scaler, scaled_features_df, target_scaled

def _create_sequences(X, y, sequence_length=16, forecast_horizon=4):
    """
    Generate overlapping sequences for DIRECT multi-step forecasting.

    X: np.ndarray or pd.DataFrame of features, shape (N, num_features)
    y: np.ndarray or pd.Series of target values (scaled log-domain), shape (N, 1) or (N,)

    Returns:
        X_seq: (num_seq, sequence_length, num_features)
        y_seq: (num_seq, forecast_horizon)  # multi-output target
    """
    X_seq, y_seq = [], []

    X_arr = np.asarray(X)
    y_arr = np.asarray(y).reshape(-1)  # flatten target to 1D

    required_length = sequence_length + forecast_horizon
    if len(X_arr) < required_length:
        print(
            f"[ModelUtils] ❌ Not enough data to generate sequences: "
            f"required at least {required_length}, got {len(X_arr)}."
        )
        num_features = X_arr.shape[1] if X_arr.ndim > 1 else 0
        # Return correctly shaped empties
        return (
            np.empty((0, sequence_length, num_features)),
            np.empty((0, forecast_horizon))
        )

    # Sliding window over the time series
    for i in range(len(X_arr) - sequence_length - forecast_horizon + 1):
        x_i = X_arr[i : i + sequence_length]  # (sequence_length, num_features)

        # Multi-step target: next forecast_horizon points in the target series
        y_window = y_arr[i + sequence_length : i + sequence_length + forecast_horizon]
        if len(y_window) == forecast_horizon:
            y_seq.append(y_window)  # shape (forecast_horizon,)

        X_seq.append(x_i)

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)  # (num_seq, forecast_horizon)

    print(
        f"[ModelUtils] [DEBUG] Generated {len(X_seq)} sequences of length "
        f"{sequence_length} with horizon {forecast_horizon}."
    )
    print(f"[ModelUtils] [DEBUG] X_seq shape: {X_seq.shape}, y_seq shape: {y_seq.shape}")
    return X_seq, y_seq

def _create_tf_dataset(X, y, batch_size, shuffle=True, seed=None):
    if X.size == 0 or y.size == 0:
        print("[ModelUtils] ⚠️ Attempted to create tf.data.Dataset with empty input array.")
        x_dtype = tf.float32
        y_dtype = tf.float32
        x_empty_shape = (
            (0, X.shape[1], X.shape[2])
            if X.ndim == 3
            else (0, X.shape[1])
            if X.ndim == 2
            else (0,)
        )
        y_empty_shape = (0, y.shape[1]) if y.ndim == 2 else (0,)

        x_tensor = tf.constant([], dtype=x_dtype, shape=x_empty_shape)
        y_tensor = tf.constant([], dtype=y_dtype, shape=y_empty_shape)

        return tf.data.Dataset.from_tensor_slices((x_tensor, y_tensor)).batch(batch_size)

    dataset = tf.data.Dataset.from_tensor_slices((X, y)).cache()

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X), seed=seed, reshuffle_each_iteration=False)

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Model building function for multi-output model:
def _build_cnn_lstm_model_reduced_lr(input_shape, output_size=None):
    """
    CNN-LSTM model with configurable multi-output head.

    input_shape: (sequence_length, num_features)
    output_size: number of forecast steps (e.g. config.FORECAST_HORIZON)
    """
    from backend import config as _cfg  # avoid circular imports at module load time

    if output_size is None:
        output_size = _cfg.FORECAST_HORIZON

    model = tf.keras.Sequential([
        tf.keras.Input(shape=input_shape),  # (sequence_length, num_features)

        # 1D Convolution layer
        tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),

        # Sequence modeling
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.2),

        # Multi-output head: one value per forecast step, in the same scaled log-space
        tf.keras.layers.Dense(output_size)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='mae',
        metrics=['mae']
    )

    return model



def _load_model_and_scalers(version_id):
    """
    Loads a specific version of the model and its associated scalers.
    """
    try:
        model_path = config.get_versioned_model_path(version_id)
        model = tf.keras.models.load_model(model_path)
    
        mm_scaler = joblib.load(config.get_versioned_scaler_path(version_id, config.MM_SCALER_NAME)) 
        r_scaler = joblib.load(config.get_versioned_scaler_path(version_id, config.R_SCALER_NAME))    
        target_scaler = joblib.load(config.get_versioned_scaler_path(version_id, config.TARGET_SCALER_NAME))

        
        print(f"[ModelUtils] ✅ Loaded model and scalers for version: {version_id}")
        return model, mm_scaler, r_scaler, target_scaler

    except Exception as e:
        print(f"[ModelUtils] ❌ Error loading model or scalers for version {version_id}: {e}")
        return None, None, None, None



def plot_learning_curves(history, metric='mae', title="Model Learning Curves", save_path=None):
    """
    Plots training and validation loss and metric curves from Keras history,
    and optionally saves the figure to disk.
    """
    if history is None or not hasattr(history, "history"):
        print("❌ Invalid history object provided.")
        return

    plt.figure(figsize=(12, 5))

    # Loss subplot
    plt.subplot(1, 2, 1)
    if "loss" in history.history:
        plt.plot(history.history["loss"], label="Train Loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    # Metric subplot
    plt.subplot(1, 2, 2)
    if metric in history.history:
        plt.plot(history.history[metric], label=f"Train {metric.upper()}")
    if f"val_{metric}" in history.history:
        plt.plot(history.history[f"val_{metric}"], label=f"Val {metric.upper()}")
    plt.title(f"{metric.upper()} Curve")
    plt.xlabel("Epoch")
    plt.ylabel(metric.upper())
    plt.grid(True)
    plt.legend()

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"✅ Learning curves saved to: {save_path}")

    plt.close()

def compute_stepwise_mae(model, X_test_seq, y_test_seq):
    """
    Compute mean absolute error per forecast step across the entire test set.

    X_test_seq: (num_seq, seq_len, num_features)
    y_test_seq: (num_seq, horizon)
    Returns: 1D array of length = horizon with MAE for step 1..H.
    """
    if X_test_seq.size == 0 or y_test_seq.size == 0:
        print("[ModelUtils] ⚠️ Empty arrays passed to compute_stepwise_mae.")
        return np.array([])

    y_pred = model.predict(X_test_seq, verbose=0)
    errors = np.abs(np.squeeze(y_test_seq) - np.squeeze(y_pred))
    if errors.ndim == 1:
        errors = errors.reshape(1, -1)
    return errors.mean(axis=0)

def plot_stepwise_mae(step_mae, save_path, title="Step-wise MAE"):
    """
    Bar plot of MAE per step in the forecast horizon.
    """
    if step_mae is None or len(step_mae) == 0:
        print("[ModelUtils] ⚠️ Empty step-wise MAE; skipping plot.")
        return

    plt.figure(figsize=(7, 4))
    plt.bar(np.arange(1, len(step_mae) + 1), step_mae)
    plt.xlabel("Step")
    plt.ylabel("MAE")
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"[ModelUtils] ✅ Step-wise MAE plot saved to {save_path}")



