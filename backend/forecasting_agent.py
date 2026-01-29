# backend/forecasting_agent.py
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import sqlite3
import joblib
import matplotlib.pyplot as plt
import re
from datetime import datetime, timezone, timedelta
from typing import Dict, Any

from .forecast_best_helper import rebuild_eprice_forecasts_best
from . import config

PLOTS_DIR = os.path.join("data", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


class ForecastingAgent:
    def __init__(
        self,
        sequence_length=config.SEQUENCE_LENGTH,
        forecast_horizon=config.FORECAST_HORIZON,
        batch_size=config.BATCH_SIZE,
        logger=None,
    ):
        if logger:
            self._log_operation = logger
        else:
            self._log_operation = lambda msg: print(f"ForecastingAgent Log: {msg}")

        self.model = None
        self.model_path = None
        self.scalers = {}
        self.version_id = None
        self.sequence_length = sequence_length
        self.forecast_horizon = config.FORECAST_HORIZON
        self.batch_size = batch_size
        self.db_path = config.FORECASTING_DB_PATH
        self.enable_diagnostic_plots = True
        self.test_sequences_path = os.path.join("data", "test_sequences.npz")

        # Try loading artifacts
        try:
            self._load_latest_artifacts_internal()
        except RuntimeError as e:
            self._log_operation(f"⚠️ Warning: {e}. Agent initialized without model.")
            self.model = None
            self.model_path = None
            self.version_id = "no_model_yet"

        self._init_db()
        self._log_operation("ForecastingAgent initialized...")

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            # Table for Forecasts
            conn.execute(
                """CREATE TABLE IF NOT EXISTS eprice_forecasts (
                forecast_generation_time TEXT,
                forecasted_for_timestamp TEXT,
                forecasted_for_timestamp_local TEXT,
                predicted_eprice REAL,
                actual_eprice REAL,
                model_version TEXT,
                PRIMARY KEY (forecast_generation_time, forecasted_for_timestamp)
            )"""
            )
            # Table for Historical Observations
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS historical_observations (
                    datetime TEXT PRIMARY KEY,
                    timestamp_local TEXT,
                    temp REAL,
                    windEnergy REAL,
                    eprice_15min REAL,
                    is_used_for_finetuning INTEGER DEFAULT 0,
                    data_source_date TEXT
                )
            """
            )
            self._ensure_eprice_table_schema(conn)

    def _ensure_eprice_table_schema(self, conn):
        cursor = conn.cursor()
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS eprice_15min (datetime TEXT PRIMARY KEY, eprice REAL)"
        )
        cursor.execute("PRAGMA table_info(eprice_15min)")
        columns = [row[1] for row in cursor.fetchall()]
        if "temp" not in columns:
            cursor.execute("ALTER TABLE eprice_15min ADD COLUMN temp REAL")
        if "windEnergy" not in columns:
            cursor.execute("ALTER TABLE eprice_15min ADD COLUMN windEnergy REAL")
        conn.commit()

    def _load_model_and_scalers(self):
        self._log_operation("Reloading latest model/scalers...")
        self._load_latest_artifacts_internal()

    
    # Discover versions from scaler version folders
    def _get_scaler_versions_base_dir(self) -> str:
        """
        Returns the base folder where scaler versions live.
        Prefer config.SCALER_VERSION_DIR if present; otherwise derive it.
        Expected structure:
            <base>/YYYYMMDD_HHMMSS/<scaler files>
        """
        if hasattr(config, "SCALER_VERSION_DIR"):
            return getattr(config, "SCALER_VERSION_DIR")

        # Derive from get_versioned_scaler_dir()
        try:
            dummy = "00000000_000000"
            return os.path.dirname(config.get_versioned_scaler_dir(dummy))
        except Exception:
            # last resort: keep older behavior
            return getattr(config, "MODEL_VERSION_DIR", "backend/scaler_artifacts/versions")

    def _list_version_ids_from_scaler_dirs(self) -> list:
        base_dir = self._get_scaler_versions_base_dir()
        if not os.path.exists(base_dir):
            return []

        ids = []
        for name in os.listdir(base_dir):
            full = os.path.join(base_dir, name)
            if os.path.isdir(full) and re.match(r"^\d{8}_\d{6}$", name):
                ids.append(name)
        return sorted(ids)

    def _load_latest_artifacts_internal(self):
        """
        Loads the newest version where:
          - scaler dir exists and contains required scaler files
          - corresponding model file exists
        """
        version_ids = self._list_version_ids_from_scaler_dirs()
        valid_versions = []

        for v in version_ids:
            s_dir = config.get_versioned_scaler_dir(v)
            m_path = config.get_versioned_model_path(v)

            if not os.path.isdir(s_dir):
                continue
            if not os.path.exists(m_path):
                continue

            # Require at least mm + target scaler. r_scaler is optional.
            mm_path = config.get_versioned_scaler_path(v, config.MM_SCALER_NAME)
            tgt_path = config.get_versioned_scaler_path(v, config.TARGET_SCALER_NAME)
            if not (os.path.exists(mm_path) and os.path.exists(tgt_path)):
                continue

            valid_versions.append(v)

        if not valid_versions:
            raise RuntimeError(
                "No valid model/scaler versions found. "
                "Checked scaler version folders and matching model files."
            )

        latest_id = sorted(valid_versions)[-1]
        self.version_id = latest_id

        print("\n================= MODEL and scaler LOAD DEBUG =================")
        print("Latest version selected:", latest_id)
        print("Target scaler path:", config.get_versioned_scaler_path(latest_id, config.TARGET_SCALER_NAME))

        # Load Model
        tf.keras.backend.clear_session()
        m_path = config.get_versioned_model_path(latest_id)
        print("Model path:", m_path)
        self.model = tf.keras.models.load_model(m_path)
        self.model_path = m_path

        # Load Scalers (use the same versioned path helpers used in retraining)
        mm_path = config.get_versioned_scaler_path(latest_id, config.MM_SCALER_NAME)
        tgt_path = config.get_versioned_scaler_path(latest_id, config.TARGET_SCALER_NAME)
        r_path  = config.get_versioned_scaler_path(latest_id, config.R_SCALER_NAME)

        print("MM scaler path:", mm_path)
        print("Target scaler path:", tgt_path)
        print("R scaler path:", r_path)

        self.scalers["mm_scaler"] = joblib.load(mm_path)
        self.scalers["target_scaler"] = joblib.load(tgt_path)

        if os.path.exists(r_path):
            self.scalers["r_scaler"] = joblib.load(r_path)

        #------------------------------------>
        for name, sc in self.scalers.items():
            if hasattr(sc, "n_features_in_"):
                self._log_operation(f"{name} n_features_in_={sc.n_features_in_}")
            if hasattr(sc, "feature_names_in_"):
                self._log_operation(f"{name} feature_names_in_={list(sc.feature_names_in_)}")

        #-------------------------------------<

        # >>> ADD THESE LINES <<<
        ts = self.scalers["target_scaler"]
        # Log known scaler attributes safely
        for attr in ["center_", "scale_", "mean_", "var_", "data_min_", "data_max_"]:
            if hasattr(ts, attr):
                self._log_operation(f"TARGET SCALER {attr}: {getattr(ts, attr)}")

        self._log_operation(f"✅ Loaded model and scalers for version: {latest_id}")

    # FEATURE GENERATION AND SELECTION 
    def _process_and_sequence_for_forecast(self, df: pd.DataFrame, source: str = "UNKNOWN"):
        if df is None or df.empty:
            return None, None

        # 1. Prepare Index
        if not pd.api.types.is_datetime64_any_dtype(df[config.TIME_COLUMN]):
            df[config.TIME_COLUMN] = pd.to_datetime(
                df[config.TIME_COLUMN], errors="coerce", utc=True
            )
            df.dropna(subset=[config.TIME_COLUMN], inplace=True)

        df = df.set_index(config.TIME_COLUMN).sort_index()

        #------muutoksia>
        # 2. Filter for known targets (HISTORICAL ONLY) 

        if config.TARGET_COLUMN not in df.columns:
            return None, None
        
        df_known = df[df[config.TARGET_COLUMN].notna()].copy()
        if df_known.empty:
            return None, None

        # Anchor timestamp: last timestamp with a known real target (can be > now)
        last_known_ts = df_known.index.max()

        # Keep only up to the anchor timestamp for sequencing
        df = df.loc[:last_known_ts].copy()

        # FEATURE ENGINEERING (on-the-fly, consistent with retraining)
        df["hour_extracted"] = df.index.hour
        df["minute_extracted"] = df.index.minute

        # weekday/weekend are NOT in ground_truth_table; always compute them here
        df["weekday"] = df.index.dayofweek
        df["is_weekend"] = (df["weekday"] >= 5).astype(int)

        # Lags (multi-output still uses lag as feature)
        df["eprice_log"] = np.log1p(df[config.TARGET_COLUMN])
        df["eprice_lag1"] = df["eprice_log"].shift(1)
        df["eprice_lag1"].bfill(inplace=True)
        df.dropna(subset=["eprice_lag1"], inplace=True)

        #DEBUG:
        tail_n = 48
        try:
            self._log_operation(
                f"[{source}] DEBUG last {tail_n} TARGET (linear): "
                f"{df[config.TARGET_COLUMN].tail(tail_n).tolist()}"
            )
            self._log_operation(
                f"[{source}] DEBUG last {tail_n} eprice_lag1 (log1p): "
                f"{df['eprice_lag1'].tail(tail_n).tolist()}"
            )
            self._log_operation(
                f"[{source}] DEBUG last {tail_n} temp: "
                f"{df['temp'].tail(tail_n).tolist()}"
            )
            self._log_operation(
                f"[{source}] DEBUG last {tail_n} windEnergy: "
                f"{df['windEnergy'].tail(tail_n).tolist()}"
            )
        except Exception as e:
            self._log_operation(f"[{source}] DEBUG block failed: {e}")

        
        # Feature list comes from config
        features_to_use = list(config.LSTM_FEATURE_COLUMNS)

        missing = [c for c in features_to_use if c not in df.columns]
        if missing:
            self._log_operation(f"❌ Missing features: {missing}")
            return None, None
        
        # ===== DEBUG: inspect forecast input tail =====
        tail_n = self.sequence_length
        tail_df = df.tail(tail_n)

        if "temp_source" in tail_df.columns:
            src_counts_tail = tail_df["temp_source"].value_counts(dropna=False).to_dict()
            self._log_operation(f"[FORECAST_INPUT] temp_source tail dist: {src_counts_tail}")

        if config.TEMPERATURE_COL_NAME in tail_df.columns:
            nan_temp_tail = int(tail_df[config.TEMPERATURE_COL_NAME].isna().sum())
            unique_temp_tail = int(tail_df[config.TEMPERATURE_COL_NAME].nunique(dropna=True))
            self._log_operation(
                f"[FORECAST_INPUT] temp NaNs in tail: {nan_temp_tail}/{len(tail_df)}, "
                f"unique temps: {unique_temp_tail}"
            )

            self._log_operation(
                f"[FORECAST_INPUT] last 48 temps: "
                f"{tail_df[config.TEMPERATURE_COL_NAME].tail(48).tolist()}"
            )

        # Optional: wind check (since it strongly affects price)
        if config.FINGRID_COL_NAME in tail_df.columns:
            self._log_operation(
                f"[FORECAST_INPUT] last 48 windEnergy: "
                f"{tail_df[config.FINGRID_COL_NAME].tail(48).tolist()}"
            )
        # ===== END DEBUG =====
        

        # Scaling
        features_raw = df[features_to_use].copy()
        mm_scaler = self.scalers["mm_scaler"]
        r_scaler = self.scalers.get("r_scaler")

        scaled = features_raw.copy()

        #---------------------------->
        '''
        if True:  # keep simple; remove later
            tail_n = 32
            raw_tail = features_raw.tail(tail_n)
            scaled_tail = scaled.tail(tail_n)

            # raw ranges
            for c in features_to_use:
                self._log_operation(
                    f"[{source}] RAW {c}: min={float(raw_tail[c].min()):.6f} "
                    f"max={float(raw_tail[c].max()):.6f} last={float(raw_tail[c].iloc[-1]):.6f}"
                )

            # scaled ranges
            for c in features_to_use:
                self._log_operation(
                    f"[{source}] SCALED {c}: min={float(scaled_tail[c].min()):.6f} "
                    f"max={float(scaled_tail[c].max()):.6f} last={float(scaled_tail[c].iloc[-1]):.6f}"
                )
        #-----------------------------<
        '''
        

        mm_cols = [c for c in config.MM_SCALED_COLUMNS if c in scaled.columns]
        r_cols = [c for c in config.R_SCALED_COLUMNS if c in scaled.columns]

        if mm_cols:
            before = features_raw[mm_cols].iloc[-1].to_dict()
            scaled[mm_cols] = mm_scaler.transform(features_raw[mm_cols])
            after = scaled[mm_cols].iloc[-1].to_dict()
            self._log_operation(f"[{source}] MM transform last-row before={before} after={after}")

        if r_cols and r_scaler:
            before = features_raw[r_cols].iloc[-1].to_dict()
            scaled[r_cols] = r_scaler.transform(features_raw[r_cols])
            after = scaled[r_cols].iloc[-1].to_dict()
            self._log_operation(f"[{source}] R transform last-row before={before} after={after}")

        features_scaled = scaled[features_to_use].values

        #-------------------------->
        #z = np.abs(features_scaled[-tail_n:, :])
        #if np.nanmax(z) > 10:
        #    self._log_operation(f"[{source}] ⚠️ Very large scaled feature detected: max_abs={float(np.nanmax(z)):.3f}")
        #--------------------------<

        if len(features_scaled) < self.sequence_length:
            self._log_operation(
                f"❌ Not enough data ({len(features_scaled)} < {self.sequence_length})"
            )
            return None, None
        
        tail = df[config.TARGET_COLUMN].tail(self.sequence_length)
        self._log_operation(
            f"[{source}] TARGET tail min/max: {float(tail.min()):.6f}/{float(tail.max()):.6f}"
        )
        self._log_operation(
            f"[{source}] TARGET tail last 8: {tail.tail(8).tolist()}"
        )

        # Anchor timestamp: last timestamp with a known target
        last_known_ts = df.index.max()

        return features_scaled[-self.sequence_length:], last_known_ts
    
    def _perform_direct_multistep_forecast(self, initial_sequence: np.ndarray, call_tag: str = "RUN"):
        input_seq = initial_sequence.reshape(
            1, self.sequence_length, initial_sequence.shape[1]
        ).astype(np.float32)

        #INPUT SANITY CHECK
        #--------------------------------------------------->

        # ====== HARD SANITY CHECK: is the input scaled? ======
        try:
            feats = list(config.LSTM_FEATURE_COLUMNS)
            last = input_seq[0, -1, :]
            mins = input_seq[0].min(axis=0)
            maxs = input_seq[0].max(axis=0)

            def fmt(i):
                return f"{feats[i]} min={mins[i]:.6f} max={maxs[i]:.6f} last={last[i]:.6f}"

            # Log all features (compact)
            self._log_operation(f"[{call_tag}] INPUT_SEQ feature ranges (last timestep + seq min/max):")
            for i in range(len(feats)):
                self._log_operation(f"[{call_tag}]   {fmt(i)}")

            # Abort early if clearly unscaled (these thresholds are intentionally conservative)
            idx_hour = feats.index("hour_extracted") if "hour_extracted" in feats else None
            idx_min  = feats.index("minute_extracted") if "minute_extracted" in feats else None
            idx_wind = feats.index("windEnergy") if "windEnergy" in feats else None

            unscaled_flags = []
            if idx_hour is not None and maxs[idx_hour] > 2.0:
                unscaled_flags.append(f"hour_extracted looks unscaled (max={maxs[idx_hour]:.3f})")
            if idx_min is not None and maxs[idx_min] > 2.0:
                unscaled_flags.append(f"minute_extracted looks unscaled (max={maxs[idx_min]:.3f})")
            if idx_wind is not None and maxs[idx_wind] > 50.0:
                unscaled_flags.append(f"windEnergy looks unscaled (max={maxs[idx_wind]:.3f})")

            if unscaled_flags:
                self._log_operation(f"[{call_tag}] ❌ INPUT SCALE CHECK FAILED: " + " | ".join(unscaled_flags))
                # Raise to stop silently producing nonsense forecasts
                raise ValueError("Input sequence appears unscaled; aborting forecast to prevent invalid output.")
        except Exception as e:
            # If you prefer not to abort, comment out the raise above and keep only logging
            self._log_operation(f"[{call_tag}] ⚠️ Input scale check encountered issue: {e}")
            raise
        # =====================================================
        #---------------------------------------------------<

        preds_scaled_log = self.model.predict(input_seq, verbose=0)[0]
        self._log_operation(
            f"[{call_tag}] preds_scaled_log shape={preds_scaled_log.shape} "
            f"min={float(np.min(preds_scaled_log)):.6f} max={float(np.max(preds_scaled_log)):.6f} "
            f"sample={preds_scaled_log[:4]}"
        )

        ts = self.scalers["target_scaler"]
        preds_log = ts.inverse_transform(preds_scaled_log.reshape(-1, 1))
        self._log_operation(
            f"[{call_tag}] preds_log shape={preds_log.shape} "
            f"min={float(np.min(preds_log)):.6f} max={float(np.max(preds_log)):.6f} "
            f"sample={preds_log[:4].flatten()}"
        )

        preds = np.expm1(preds_log).flatten()
        self._log_operation(
            f"[{call_tag}] expm1(inverse) min/max={float(np.min(preds)):.6f}/{float(np.max(preds)):.6f} "
            f"sample={preds[:4]}"
        )
        return preds
        

    def _run_multi_output_diagnostics(self):
        """Generates Forecast vs Actual plot for UI."""
        if (
            not self.enable_diagnostic_plots
            or not os.path.exists(self.test_sequences_path)
            or self.model is None
            or not self.scalers
        ):
            return

        try:
            data = np.load(self.test_sequences_path)
            X_seq, y_seq = data["X_test_seq"], data["y_test_seq"]
            if X_seq.size == 0:
                return

            preds = self._perform_direct_multistep_forecast(X_seq[0], call_tag="DIAG_TESTSEQ")

            act_log = self.scalers["target_scaler"].inverse_transform(
                y_seq[0].reshape(-1, 1)
            )
            acts = np.expm1(act_log).flatten()

            min_len = min(len(preds), len(acts))
            preds, acts = preds[:min_len], acts[:min_len]
            mae = float(np.mean(np.abs(preds - acts)))

            plt.figure(figsize=(10, 4))
            plt.plot(acts, marker="o", label="Actual")
            plt.plot(preds, marker="x", label="Forecast")
            plt.title(f"Forecast vs Actual (Model {self.version_id}) | MAE={mae:.4f}")
            plt.xlabel("Step")
            plt.ylabel("Price")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()

            path = os.path.join(PLOTS_DIR, f"diagnostic_forecast_{self.version_id}.png")
            plt.savefig(path)
            plt.close()
            self._log_operation(f"[Diag] Saved plot: {path}")

        except Exception as e:
            self._log_operation(f"[Diag] Error: {e}")

    def run(self, merged_df: pd.DataFrame, model_version_tag: str) -> Dict[str, Any]:
        if self.model is None or not self.scalers:
            self._log_operation("❌ No model loaded.")
            return {"forecast_df": pd.DataFrame(), "interpretation": "Error: No model."}

        tag = model_version_tag or self.version_id
        self._log_operation(f"--- Running Direct Forecast ({tag}) ---")

        seq, last_ts = self._process_and_sequence_for_forecast(merged_df, source="RUN_MAIN")
        if seq is None:
            return {
                "forecast_df": pd.DataFrame(),
                "interpretation": "Error: Data processing failed.",
            }

        try:
            forecast_vals = self._perform_direct_multistep_forecast(seq, call_tag="RUN_MAIN")
        except Exception as e:
            self._log_operation(f"❌ Prediction failed: {e}")
            return {
                "forecast_df": pd.DataFrame(),
                "interpretation": "Error: Prediction failed.",
            }

        timestamps = pd.date_range(
            start=last_ts + pd.Timedelta(minutes=15),
            periods=len(forecast_vals),
            freq="15min",
            tz="UTC",
        )

        forecast_df = pd.DataFrame(
            {config.TIME_COLUMN: timestamps, config.TARGET_COLUMN: forecast_vals}
        )

        self._save_forecasts(forecast_df, tag)
        try:
            rebuild_eprice_forecasts_best(self.db_path)
        except Exception as e:
            self._log_operation(f"⚠️ Could not rebuild eprice_forecasts_best: {e}")
            
        try:
            self._run_multi_output_diagnostics()
        except Exception:
            pass

        return {"forecast_df": forecast_df, "interpretation": "Success"}

    def _save_forecasts(self, df, version):
        if df.empty:
            return
        try:
            with sqlite3.connect(self.db_path) as conn:
                df_save = df.copy()
                df_save["forecast_generation_time"] = datetime.now(timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                )
                df_save["forecasted_for_timestamp"] = df_save[config.TIME_COLUMN].dt.strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                )
                df_save["forecasted_for_timestamp_local"] = (
                    df_save[config.TIME_COLUMN]
                    .dt.tz_convert(config.LOCAL_TIME_ZONE)
                    .dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                )
                df_save.rename(columns={config.TARGET_COLUMN: "predicted_eprice"}, inplace=True)
                df_save["model_version"] = version
                df_save["actual_eprice"] = np.nan
                cols = [
                    "forecast_generation_time",
                    "forecasted_for_timestamp",
                    "forecasted_for_timestamp_local",
                    "predicted_eprice",
                    "actual_eprice",
                    "model_version",
                ]
                df_save[cols].to_sql("eprice_forecasts", conn, if_exists="append", index=False)
                # NEW: rebuild "best" table (shortest lead time wins)
                rebuild_eprice_forecasts_best(self.db_path)
            self._log_operation(f"✅ Saved {len(df)} forecasts.")
        except Exception as e:
            self._log_operation(f"❌ DB Error: {e}")

    # Keep _save_historical_data_for_finetuning (unchanged)
    def _save_historical_data_for_finetuning(self, historical_df: pd.DataFrame):
        if historical_df.empty:
            return
        df_to_save = historical_df.copy()
        df_to_save[config.TIME_COLUMN] = pd.to_datetime(df_to_save[config.TIME_COLUMN])
        df_to_save["timestamp_local"] = (
            df_to_save[config.TIME_COLUMN]
            .dt.tz_convert(config.LOCAL_TIME_ZONE)
            .dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        df_to_save[config.TIME_COLUMN] = df_to_save[config.TIME_COLUMN].dt.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        cols_for_db = [config.TIME_COLUMN, "timestamp_local", "temp", "windEnergy", config.TARGET_COLUMN]
        final_df_for_sql = df_to_save[[col for col in cols_for_db if col in df_to_save.columns]].copy()
        final_df_for_sql.rename(columns={config.TARGET_COLUMN: "eprice_15min"}, inplace=True)
        final_df_for_sql["data_source_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        final_df_for_sql["is_used_for_finetuning"] = 0
        final_df_for_sql.dropna(subset=[config.TIME_COLUMN, "eprice_15min"], inplace=True)
        if final_df_for_sql.empty:
            return
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                timestamps_to_replace = final_df_for_sql[config.TIME_COLUMN].tolist()
                if timestamps_to_replace:
                    placeholders = ",".join(["?"] * len(timestamps_to_replace))
                    cursor.execute(
                        f"DELETE FROM historical_observations WHERE datetime IN ({placeholders})",
                        tuple(timestamps_to_replace),
                    )
                final_df_for_sql.to_sql(
                    "historical_observations", conn, if_exists="append", index=False, method="multi"
                )
            self._log_operation(
                f"✅ Upserted {len(final_df_for_sql)} rows of historical data into 'historical_observations'."
            )
        except Exception as e:
            self._log_operation(f"❌ Error saving historical data to 'historical_observations': {e}")

        cols_for_eprice_table_update = [config.TIME_COLUMN, "temp", "windEnergy", config.TARGET_COLUMN]
        df_for_update = historical_df[cols_for_eprice_table_update].copy()
        df_for_update.rename(columns={config.TARGET_COLUMN: "eprice"}, inplace=True)
        df_for_update.dropna(subset=[config.TIME_COLUMN, "eprice", "temp", "windEnergy"], inplace=True)
        if not df_for_update.empty:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    self._ensure_eprice_table_schema(conn)
                    df_for_update[config.TIME_COLUMN] = (
                        pd.to_datetime(df_for_update[config.TIME_COLUMN], utc=True)
                        .dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                    )
                    df_for_update.to_sql("eprice_15min", conn, if_exists="replace", index=False)
                    self._log_operation(
                        f"✅ Upserted {len(df_for_update)} rows into 'eprice_15min' with the latest prices and weather data."
                    )
            except Exception as e:
                self._log_operation(f"❌ Error updating 'eprice_15min' table: {e}")
