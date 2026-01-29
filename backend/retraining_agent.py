# backend/retraining_agent.py
import os
import pandas as pd
import numpy as np
import random
import joblib
import sqlite3
from datetime import datetime, timezone
import tensorflow as tf
import json
from tensorflow.keras.callbacks import EarlyStopping

from utils.model_utils import (
    _process_features_and_target,
    _create_sequences,
    _create_tf_dataset,
    _get_fitted_scalers,
    _build_cnn_lstm_model_reduced_lr,
    plot_learning_curves,
    compute_stepwise_mae,
    plot_stepwise_mae,
)
from . import config

# Suppress TensorFlow logging for cleaner output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")

PLOTS_DIR = os.path.join("data", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


class RetrainingAgent:
    def __init__(self, logger=print):
        self.logger = logger
        self.log("\n🔄 Retraining Agent: Initializing...")
        self.db_path = config.FORECASTING_DB_PATH

    def log(self, msg):
        self.logger(f"[RetrainingAgent] {msg}")

    
    # Full-dataset fetch (no metadata / no last_training_datetime)
    def _fetch_full_training_data(self) -> pd.DataFrame:
        """
        Pull ALL rows from ground_truth_table in chronological order.
        Filters to historical rows (datetime <= now) and (optionally) excludes
        forecasted temperature rows (temp_is_forecasted = 0).

        IMPORTANT: only core columns are selected; engineered features like
        hour_extracted, minute_extracted, weekday, is_weekend are created
        later in _process_features_and_target.
        """
        with sqlite3.connect(self.db_path) as conn:
            q = f"""
                SELECT
                  {config.TIME_COLUMN}          AS {config.TIME_COLUMN},
                  {config.TARGET_COLUMN}        AS {config.TARGET_COLUMN},
                  {config.TEMPERATURE_COL_NAME} AS temp,
                  {config.FINGRID_COL_NAME}     AS windEnergy,
                  temp_is_forecasted
                FROM {config.GROUND_TRUTH_TABLE}
                WHERE {config.TARGET_COLUMN} IS NOT NULL
                ORDER BY {config.TIME_COLUMN} ASC
            """
            df = pd.read_sql_query(q, conn, parse_dates=[config.TIME_COLUMN])

        if df.empty:
            return df

        # Normalize datetimes
        df[config.TIME_COLUMN] = pd.to_datetime(
            df[config.TIME_COLUMN], utc=True, errors="coerce"
        )
        df = df.dropna(subset=[config.TIME_COLUMN]).sort_values(config.TIME_COLUMN).reset_index(drop=True)

        # Keep only historical rows (no future leakage)
        now_utc = pd.Timestamp.now(tz="UTC")
        df = df[df[config.TIME_COLUMN] <= now_utc]

        if df.empty:
            return df

        # Exclude forecasted temperatures from training
        if "temp_is_forecasted" in df.columns:
            df = df[df["temp_is_forecasted"].fillna(0) == 0]

        # Ensure numeric columns
        for c in [config.TARGET_COLUMN, "temp", "windEnergy"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna(subset=[config.TARGET_COLUMN])
        return df

    def retrain_model(self, train_from_scratch: bool = True):
        """
        Always retrain on the full dataset in ground_truth_table.
        `train_from_scratch` is ignored (kept only for signature compatibility).
        """
        self.log("Starting full model re-training cycle (full history)...")

        # Deterministic setup
        os.environ["PYTHONHASHSEED"] = str(config.SEED)
        random.seed(config.SEED)
        np.random.seed(config.SEED)
        tf.random.set_seed(config.SEED)

        version_id = config.get_version_id()
        self.log(f"Generated new model version ID: {version_id}")

        # Loading full history
        raw_data = self._fetch_full_training_data()
        if raw_data is None or raw_data.empty:
            self.log("❌ No data available for retraining.")
            return None, None

        self.log(f"Loaded {len(raw_data)} historical rows for training.")

        # Feature/target processing
        processed_features, processed_target = _process_features_and_target(raw_data)

        if processed_features is None or processed_features.empty:
            self.log("❌ Feature processing failed or produced empty data.")
            return None, None
        

        # Fitting scalers
        (
            mm_scaler,
            r_scaler,
            target_scaler,
            scaled_features,
            scaled_target,
        ) = _get_fitted_scalers(processed_features, processed_target)

        # Save raw vs processed target semantics (diagnostic artifact)
        self._save_target_semantics_artifact(
            version_id=version_id,
            raw_target_series=raw_data[config.TARGET_COLUMN],
            processed_target_series=processed_target,
            target_scaler=target_scaler,
        )

        # Saving the scalers
        versioned_scaler_dir = config.get_versioned_scaler_dir(version_id)
        os.makedirs(versioned_scaler_dir, exist_ok=True)

        joblib.dump(
            mm_scaler,
            config.get_versioned_scaler_path(version_id, config.MM_SCALER_NAME),
        )
        joblib.dump(
            r_scaler,
            config.get_versioned_scaler_path(version_id, config.R_SCALER_NAME),
        )
        joblib.dump(
            target_scaler,
            config.get_versioned_scaler_path(version_id, config.TARGET_SCALER_NAME),
        )
        self.log(f"✅ Scalers saved for version {version_id} → {versioned_scaler_dir}")

        # Time-based split
        split_index = int(len(scaled_features) * config.SPLIT_RATIO)
        X_train_scaled = scaled_features[:split_index].values
        y_train_scaled = scaled_target[:split_index]

        X_test_scaled = scaled_features[split_index:].values
        y_test_scaled = scaled_target[split_index:]

        # Ensure making sequences
        min_data_for_sequence = config.SEQUENCE_LENGTH + config.FORECAST_HORIZON
        if len(X_train_scaled) < min_data_for_sequence:
            self.log(
                f"❌ Not enough training rows ({len(X_train_scaled)}). "
                f"Need at least {min_data_for_sequence} after split."
            )
            return None, None

        # Creating sequences (multi-output)
        X_train_seq, y_train_seq = _create_sequences(
            X_train_scaled,
            y_train_scaled,
            sequence_length=config.SEQUENCE_LENGTH,
            forecast_horizon=config.FORECAST_HORIZON,
        )

        X_test_seq, y_test_seq = _create_sequences(
            X_test_scaled,
            y_test_scaled,
            sequence_length=config.SEQUENCE_LENGTH,
            forecast_horizon=config.FORECAST_HORIZON,
        )

        # Saving diagnostic sequences for ForecastingAgent
        try:
            if X_test_seq.size > 0 and y_test_seq.size > 0:
                os.makedirs("data", exist_ok=True)
                diag_path = os.path.join("data", "test_sequences.npz")

                MAX_SEQS = 200  # Max number of diag sequences
                X_diag = X_test_seq[:MAX_SEQS]
                y_diag = y_test_seq[:MAX_SEQS]

                np.savez(diag_path, X_test_seq=X_diag, y_test_seq=y_diag)
                self.log(
                    f"[Diag] ✅ Saved diagnostic test sequences to {diag_path} "
                    f"(X_test_seq={X_diag.shape}, y_test_seq={y_diag.shape})."
                )
            else:
                self.log("[Diag] ℹ️ Not enough test sequences to save diagnostics.")
        except Exception as e:
            self.log(f"[Diag] ❌ Error while creating/saving diagnostic sequences: {e}")

        # Creating tf.data datasets
        train_dataset = _create_tf_dataset(
            X_train_seq, y_train_seq, config.BATCH_SIZE, shuffle=True, seed=config.SEED
        )

        test_dataset = None
        if X_test_seq.size > 0 and y_test_seq.size > 0:
            test_dataset = _create_tf_dataset(
                X_test_seq, y_test_seq, config.BATCH_SIZE, shuffle=False, seed=config.SEED
            )
        else:
            self.log("⚠️ No validation sequences — training without validation.")

        if X_train_seq.size == 0:
            self.log("❌ No training sequences generated.")
            return None, None

        # Building model
        input_shape = None
        for x_batch, _ in train_dataset.take(1):
            input_shape = x_batch.shape[1:]
            break
        if input_shape is None:
            self.log("❌ Could not determine input shape from training dataset.")
            return None, None

        self.log(f"🏗️ Building model with input shape: {input_shape}")

        # Multi-output model
        model = _build_cnn_lstm_model_reduced_lr(
            input_shape=input_shape, output_size=config.FORECAST_HORIZON
        )

        self.log("Model Summary:")
        model.summary(print_fn=self.log)
        self.log("-" * 30)

        # Training
        monitor_metric = "val_mae" if test_dataset else "mae"
        early_stopping = EarlyStopping(
            monitor=monitor_metric,
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            mode="min",
            verbose=1,
        )

        self.log(f"\n🧠 Training for {config.EPOCHS} epochs (monitor='{monitor_metric}')...")
        fit_kwargs = {
            "x": train_dataset,
            "epochs": config.EPOCHS,
            "callbacks": [early_stopping],
            "verbose": 1,
        }
        if test_dataset:
            fit_kwargs["validation_data"] = test_dataset

        try:
            history = model.fit(**fit_kwargs)
            if not history.history or not history.history.get(monitor_metric):
                self.log("⚠️ Training history missing monitor metric (possibly early exit).")
        except Exception as e:
            self.log(f"❌ Error during model training: {e}")
            return None, None

        # Saving learning curves for model diagnostics
        try:
            lc_path = os.path.join(PLOTS_DIR, f"learning_curves_{version_id}.png")
            plot_learning_curves(
                history,
                metric="mae",
                title=f"Learning Curves (version {version_id})",
                save_path=lc_path,
            )
            self.log(f"[Diag] ✅ Learning curves plot saved to {lc_path}")
        except Exception as e:
            self.log(f"[Diag] ⚠️ Could not save learning curves plot: {e}")

        # Saving step-wise MAE for multi-output diagnostics
        try:
            if X_test_seq.size > 0 and y_test_seq.size > 0:
                step_mae = compute_stepwise_mae(model, X_test_seq, y_test_seq)
                if step_mae is not None and len(step_mae) > 0:
                    sw_path = os.path.join(PLOTS_DIR, f"stepwise_mae_{version_id}.png")
                    plot_stepwise_mae(
                        step_mae,
                        save_path=sw_path,
                        title=f"Step-wise MAE (version {version_id})",
                    )
                    self.log(f"[Diag] ✅ Step-wise MAE plot saved to {sw_path}")
                else:
                    self.log("[Diag] ℹ️ Step-wise MAE array empty; no plot saved.")
            else:
                self.log("[Diag] ℹ️ No test sequences available for step-wise MAE.")
        except Exception as e:
            self.log(f"[Diag] ⚠️ Could not compute/save step-wise MAE plot: {e}")

        # 9) Save versioned model
        versioned_model_path = config.get_versioned_model_path(version_id)
        os.makedirs(os.path.dirname(versioned_model_path), exist_ok=True)
        model.save(versioned_model_path)
        self.log(f"\n✅ Model retrained and saved as version {version_id} → {versioned_model_path}")
        self.log("🔄 Retraining cycle complete. Model ready for forecasting.")

        return model, version_id
    
    
    def _save_target_semantics_artifact(
        self,
        version_id: str,
        raw_target_series: pd.Series,
        processed_target_series: pd.Series,
        target_scaler,
    ):
        """
        Saves a small JSON artifact describing the target unit + transform
        used for training, plus basic stats, to make future debugging deterministic.
        """
        try:
            out_dir = config.get_versioned_scaler_dir(version_id)
            os.makedirs(out_dir, exist_ok=True)

            def to_1d_numeric(x, name: str) -> pd.Series:
                """
                Convert x to a 1-D numeric pandas Series (best-effort).
                Supports Series, DataFrame (1 col), list, 1-D/2-D ndarray.
                """
                if x is None:
                    return pd.Series(dtype="float64")

                # Series -> ok
                if isinstance(x, pd.Series):
                    s = x

                # DataFrame -> take first column (common for (n,1) targets)
                elif isinstance(x, pd.DataFrame):
                    if x.shape[1] == 0:
                        return pd.Series(dtype="float64")
                    s = x.iloc[:, 0]

                # ndarray/list/tuple -> flatten if needed
                else:
                    arr = np.asarray(x)
                    if arr.ndim == 0:
                        # scalar
                        s = pd.Series([arr.item()])
                    elif arr.ndim == 1:
                        s = pd.Series(arr)
                    else:
                        # 2-D or more: take first column, then flatten
                        s = pd.Series(arr.reshape(arr.shape[0], -1)[:, 0])

                s = pd.to_numeric(s, errors="coerce").dropna()
                return s

            raw = to_1d_numeric(raw_target_series, "raw_target")
            proc = to_1d_numeric(processed_target_series, "processed_target")

            def stats(s: pd.Series):
                if s.empty:
                    return {}
                return {
                    "count": int(s.shape[0]),
                    "min": float(s.min()),
                    "p01": float(s.quantile(0.01)),
                    "p50": float(s.quantile(0.50)),
                    "p99": float(s.quantile(0.99)),
                    "max": float(s.max()),
                    "mean": float(s.mean()),
                    "std": float(s.std(ddof=0)),
                }

            def safe_attr(obj, attr):
                return getattr(obj, attr, None) if obj is not None and hasattr(obj, attr) else None

            def safe_list(obj, attr):
                val = safe_attr(obj, attr)
                if val is None:
                    return None
                try:
                    return val.tolist()
                except Exception:
                    try:
                        return list(val)
                    except Exception:
                        return None

            payload = {
                "version_id": version_id,
                "time_utc": datetime.now(timezone.utc).isoformat(),
                "target_column": config.TARGET_COLUMN,

                # Explicit semantics
                "intended_unit": "snt_per_kwh",
                "training_target_transform": "unknown",

                # Stats
                "raw_target_stats": stats(raw),
                "processed_target_stats": stats(proc),

                # Scaler metadata (SAFE)
                "target_scaler_type": type(target_scaler).__name__ if target_scaler is not None else None,
                "target_scaler_n_features_in": safe_attr(target_scaler, "n_features_in_"),
                "target_scaler_feature_names_in": list(safe_attr(target_scaler, "feature_names_in_") or []),

                "target_scaler_center": safe_list(target_scaler, "center_"),
                "target_scaler_scale": safe_list(target_scaler, "scale_"),
                "target_scaler_mean": safe_list(target_scaler, "mean_"),
                "target_scaler_var": safe_list(target_scaler, "var_"),
                "target_scaler_data_min": safe_list(target_scaler, "data_min_"),
                "target_scaler_data_max": safe_list(target_scaler, "data_max_"),
            }

            # Heuristic inference: if processed values look like log1p(raw) -> mark it.
            if (not raw.empty) and (not proc.empty):
                raw_med = float(raw.quantile(0.5))
                proc_med = float(proc.quantile(0.5))
                if raw_med > -0.999:
                    if abs(np.log1p(raw_med) - proc_med) < 0.05:
                        payload["training_target_transform"] = "log1p"
                    else:
                        payload["training_target_transform"] = "linear_or_other"

            out_path = os.path.join(out_dir, "target_semantics.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

            self.log(f"[TargetSemantics] ✅ Saved {out_path}")

        except Exception as e:
            self.log(f"[TargetSemantics] ⚠️ Failed to save target semantics artifact: {e}")

