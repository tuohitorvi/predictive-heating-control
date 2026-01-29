# backend/orchestrating_agent.py
import pandas as pd
import os
import numpy as np
import sqlite3
from datetime import datetime, timezone, timedelta
from typing import Optional
import traceback
import time
import sys
from . import config

# Agent classes
from .sensor_monitor_agent import SensorMonitorAgent
from .data_fetcher_agent import DataFetcherAgent
from .preprocessor_agent import PreprocessorAgent
from .forecasting_agent import ForecastingAgent
from .actuator_control_agent import ActuatorCtrlAgent
from .retraining_agent import RetrainingAgent
from .llm_agent import LLMAgent

# Tool functions
from .mcp_tools.fingrid_tool import fetch_fingrid_data
from .mcp_tools.fmi_tool import fetch_temp_data, fetch_weather_forecast
from .mcp_tools.elering_tool import fetch_elering_prices
from utils.data_transformation_utils import add_periodic_time_features
from utils.model_utils import _fetch_data, _process_features_and_target, _create_sequences, _create_tf_dataset, _build_cnn_lstm_model_reduced_lr, _get_fitted_scalers, _load_model_and_scalers
from .forecast_best_helper import rebuild_eprice_forecasts_best

# Import of the upsert function for ground_truth_table for initial creating of the ground_truth_table
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')))
from scripts.upsert_ground_truth_table import upsert_ground_truth_table as upsert_to_ground_truth



class OrchestratingAgent:
    def __init__(self):
        # 1. Provide a direct, print-based logger that works immediately.
        self._temp_logger = lambda msg: print(f"[{datetime.now(timezone.utc).isoformat()}] Orchestrator: {msg}")
        self._temp_logger("OrchestratingAgent initializing (basic logger active)...")

        # 2. Initialize system_state immediately.
        self.system_state = {
            "last_forecast_cycle_status": "Idle",
            "last_finetune_cycle_status": "Idle",
            "last_retraining_cycle_status": "Idle",
            "last_actuator_cycle_status": "Idle", # Status for actuator cycle
            "current_model_path": None,
            "current_model_version_tag": "initializing",
            "preprocessed_df": pd.DataFrame(),
            "forecast_df": pd.DataFrame(),
            "forecast_interpretation": "",
            # Initialize Eprice Upper Limit in control_parameters
            "control_parameters": {
                **config.CONTROL_PARAMETERS.copy(), # Control parameters
                "EPRICE_UPPER_LIMIT": config.DEFAULT_EPRICE_UPPER_LIMIT # Default upper limit
            },
            "control_signals": {},
            "last_processed_eprice_timestamp": None,
            "operational_log": []
        }
        
        # 3. Assign the "real" logger.
        self.logger = self._full_logger
        self.logger("OrchestratingAgent initializing (full logger active)...")

        # Initialize sub-agents
        self.data_fetcher = DataFetcherAgent(logger=self.logger)
        self.preprocessor = PreprocessorAgent(logger=self.logger)
        self.forecaster = ForecastingAgent(logger=self.logger)
        self.retrainer = RetrainingAgent(logger=self.logger)
        
        self.sensor_monitor = SensorMonitorAgent(db_path=config.SENSOR_DB_PATH, logger=self.logger)
        self.actuator_controller = ActuatorCtrlAgent(logger=self.logger)
        self.actuator_controller.connect()

        self.system_state["current_model_path"] = self.forecaster.model_path
        self.system_state["current_model_version_tag"] = self.forecaster.version_id
        
        self.sensor_monitor.start()

        self.llm_agent = LLMAgent(llm_service_url="http://localhost:5001", logger=self.logger) # Initialize LLMAgent
        
        self.logger(f"OrchestratingAgent initialized. Current model version: {self.system_state['current_model_version_tag']}")
    
    def _full_logger(self, message: str):
        timestamp = datetime.now(timezone.utc).isoformat()
        log_entry = f"[{timestamp}] Orchestrator: {message}"
        print(log_entry)
        self.system_state["operational_log"].insert(0, log_entry)
        if len(self.system_state["operational_log"]) > 200:
            self.system_state["operational_log"].pop()

    def run_forecasting_cycle(self):
        self.logger("Starting forecasting cycle...")
        self.system_state["last_forecast_cycle_status"] = "Running..."
        try:
            # 0. Define rolling lookback window for fetch 
            # Ensure that at least N days of history + day-ahead span is always pulled
            lookback_days = getattr(config, "FORECAST_LOOKBACK_DAYS", 7)

            now_utc = pd.Timestamp.now(tz="UTC")
            start_utc = (now_utc - pd.Timedelta(days=lookback_days)).normalize()  # 00:00 UTC N days ago
            end_utc = (now_utc + pd.Timedelta(days=1)).replace(hour=23, minute=59, second=59, microsecond=0)

            self.logger(f"Using lookback window: {start_utc.isoformat()} → {end_utc.isoformat()}")

            # 1. Fetch data
            raw_data = self.data_fetcher.run(
                location="Hämeenlinna",
                startTime=start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                endTime=end_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            )
            self.logger("Data fetched successfully.")

            # 2. Preprocess and align data
            preprocessed_df = self.preprocessor.run(raw_data)
            if preprocessed_df.empty:
                self.logger("❌ Cycle Failed: Preprocessing resulted in an empty DataFrame.")
                self.system_state["last_forecast_cycle_status"] = "Failed: Empty DataFrame after preprocessing."
                return
            
            
            
            self.logger("Step 3: Checking for new day-ahead price data...")
            now_utc = pd.Timestamp.now(tz="UTC")
            tmp = preprocessed_df.copy()
            tmp[config.TIME_COLUMN] = pd.to_datetime(tmp[config.TIME_COLUMN], utc=True, errors="coerce")

            tmp_hist = tmp[tmp[config.TIME_COLUMN] <= now_utc]
            last_available_eprice_ts = tmp_hist.dropna(subset=[config.TARGET_COLUMN])[config.TIME_COLUMN].max()

            has_new_price_data = True
            if self.system_state.get("last_processed_eprice_timestamp") is not None:
                last_processed_ts_dt = pd.to_datetime(
                    self.system_state["last_processed_eprice_timestamp"]
                )
                last_available_eprice_ts_dt = last_available_eprice_ts  # already tz-aware

                if last_available_eprice_ts_dt > last_processed_ts_dt:
                    self.logger(
                        f"✅ New price data found up to {last_available_eprice_ts_dt.isoformat()} "
                        f"(previously processed up to {last_processed_ts_dt.isoformat()})."
                    )
                    has_new_price_data = True
                else:
                    # Log but don't return
                    self.logger(
                        f"ℹ️ No *newer* price data available than "
                        f"{last_processed_ts_dt.isoformat()}. "
                        f"Proceeding anyway to generate a new forecast based on updated features."
                    )
                    has_new_price_data = False
            else:
                self.logger(
                    f"✅ First run or no previous timestamp. Latest price data up to {last_available_eprice_ts.isoformat()}."
                )
                has_new_price_data = True

            # For debugging purposes:
            self.logger(f"Proceeding with forecasting cycle (has_new_price_data={has_new_price_data}).")

            enhanced_df = self.preprocessor.enhance_with_fmi_forecast(preprocessed_df, self.data_fetcher)
            df_for_prediction = self.preprocessor.fill_prediction_gaps(enhanced_df)

            # --- DEBUG: verify target timestamp range BEFORE upsert ---
            try:
                _df = df_for_prediction.copy()
                _df[config.TIME_COLUMN] = pd.to_datetime(_df[config.TIME_COLUMN], utc=True, errors="coerce")
                _df = _df.dropna(subset=[config.TIME_COLUMN]).sort_values(config.TIME_COLUMN)

                # last few non-null target timestamps
                nonnull = _df[_df[config.TARGET_COLUMN].notna()][[config.TIME_COLUMN, config.TARGET_COLUMN]].tail(10)

                self.logger(f"[TS_DEBUG] df_for_prediction time min={_df[config.TIME_COLUMN].min()} max={_df[config.TIME_COLUMN].max()} rows={len(_df)}")
                self.logger(f"[TS_DEBUG] non-null target count={int(_df[config.TARGET_COLUMN].notna().sum())}")

                if not nonnull.empty:
                    self.logger(f"[TS_DEBUG] last 10 non-null targets:\n{nonnull.to_string(index=False)}")
                    self.logger(f"[TS_DEBUG] max non-null target ts={nonnull[config.TIME_COLUMN].max()}")
                else:
                    self.logger("[TS_DEBUG] no non-null targets in df_for_prediction")
            except Exception as e:
                self.logger(f"[TS_DEBUG] failed: {e}")
            #-----------------------

            self.system_state["preprocessed_df"] = df_for_prediction.tail(20)
            self.logger("Data preprocessed, enhanced, and gaps filled.")

            # Reset temp_is_forecasted=0 for historical rows in the in-memory frame
            now_utc_moment = pd.Timestamp.now(tz='UTC')
            if 'temp_is_forecasted' in df_for_prediction.columns:
                df_for_prediction.loc[df_for_prediction[config.TIME_COLUMN] <= now_utc_moment, 'temp_is_forecasted'] = 0
                self.logger(f"Reset 'temp_is_forecasted' to 0 for all historical data up to {now_utc_moment.isoformat()}.")

            # DEBUG: target values to upsert (units/scale sanity) ---
            try:
                if config.TARGET_COLUMN in df_for_prediction.columns:
                    tail = df_for_prediction[[config.TIME_COLUMN, config.TARGET_COLUMN]].tail(12).copy()
                    tail[config.TIME_COLUMN] = pd.to_datetime(tail[config.TIME_COLUMN], utc=True, errors="coerce")
                    self.logger(
                        "[UNITCHECK] pre-upsert target tail:\n"
                        + tail.to_string(index=False)
                    )
                    self.logger(
                        f"[UNITCHECK] pre-upsert target stats: "
                        f"min={float(pd.to_numeric(tail[config.TARGET_COLUMN], errors='coerce').min()):.6f} "
                        f"max={float(pd.to_numeric(tail[config.TARGET_COLUMN], errors='coerce').max()):.6f}"
                    )
            except Exception as e:
                self.logger(f"[UNITCHECK] ⚠️ pre-upsert logging failed: {e}")
            #--------------------

            def ensure_utc_datetime(series: pd.Series, local_tz: str = "Europe/Helsinki") -> pd.Series:
                s = pd.to_datetime(series, errors="coerce")
                # tz-aware -> convert
                if getattr(s.dt, "tz", None) is not None:
                    return s.dt.tz_convert("UTC")
                # naive -> localize -> convert
                return s.dt.tz_localize(local_tz).dt.tz_convert("UTC")    


            df_for_prediction[config.TIME_COLUMN] = ensure_utc_datetime(df_for_prediction[config.TIME_COLUMN], config.LOCAL_TIME_ZONE)

            # Upserting to ground_truth_table
            upsert_to_ground_truth(df_for_prediction, self.db_path, config.GROUND_TRUTH_TABLE)
            self.logger(f"Data upserted to {config.GROUND_TRUTH_TABLE}.")

            # DB-wide reset of temp_is_forecasted AFTER upsert 
            try:
                import sqlite3
                now_utc_str = pd.Timestamp.now(tz="UTC").isoformat()

                with sqlite3.connect(config.FORECASTING_DB_PATH) as conn:
                    conn.execute(f"""
                        CREATE INDEX IF NOT EXISTS ix_gt_datetime
                        ON {config.GROUND_TRUTH_TABLE} ({config.TIME_COLUMN});
                    """)
                    conn.execute(f"""
                        CREATE INDEX IF NOT EXISTS ix_gt_temp_flag
                        ON {config.GROUND_TRUTH_TABLE} (temp_is_forecasted);
                    """)

                    cur = conn.execute(f"""
                        UPDATE {config.GROUND_TRUTH_TABLE}
                        SET temp_is_forecasted = 0
                        WHERE {config.TIME_COLUMN} <= ?
                        AND {config.TEMPERATURE_COL_NAME} IS NOT NULL
                        AND temp_is_forecasted <> 0
                    """, (now_utc_str,))
                    self.logger(
                        f"DB-wide reset: cleared temp_is_forecasted for {cur.rowcount} rows up to {now_utc_str}."
                    )
            except Exception as e:
                self.logger(f"⚠️ DB-wide temp_is_forecasted reset failed: {e}")

            # Validating data for forecast before passing to agent (historical tail only)
            self.logger("Step 6: Validating data for forecast before passing to agent...")

            required_cols = [
                config.TIME_COLUMN,
                config.TARGET_COLUMN,
                config.TEMPERATURE_COL_NAME,
                config.FINGRID_COL_NAME,
            ]
            missing = [c for c in required_cols if c not in df_for_prediction.columns]
            if missing:
                self.logger(f"❌ Forecast aborted. Missing columns: {missing}")
                self.system_state["last_forecast_cycle_status"] = "Failed: Missing base columns."
                return

            now_utc = pd.Timestamp.now(tz="UTC")

            df_hist = df_for_prediction.copy()
            df_hist[config.TIME_COLUMN] = pd.to_datetime(df_hist[config.TIME_COLUMN], utc=True, errors="coerce")
            df_hist = df_hist.dropna(subset=[config.TIME_COLUMN]).sort_values(config.TIME_COLUMN)

            df_hist = df_for_prediction.copy()
            df_hist[config.TIME_COLUMN] = pd.to_datetime(df_hist[config.TIME_COLUMN], utc=True, errors="coerce")
            df_hist = df_hist.dropna(subset=[config.TIME_COLUMN]).sort_values(config.TIME_COLUMN)

            # Use ALL known target rows (including known day-ahead future), not only <= now.
            df_known = df_hist.dropna(subset=[config.TARGET_COLUMN]).copy()

            if df_known.empty:
                self.logger("❌ Forecast aborted. No known target values available for sequencing.")
                self.system_state["last_forecast_cycle_status"] = "Failed: No known targets."
                return

            # Anchoring the last known real price timestamp
            last_known_ts = df_known[config.TIME_COLUMN].max()

            # Build the sequence tail ending at the anchor timestamp
            df_upto_anchor = df_known[df_known[config.TIME_COLUMN] <= last_known_ts].copy()
            df_tail = df_upto_anchor.tail(self.forecaster.sequence_length).copy()

            if len(df_tail) < self.forecaster.sequence_length:
                self.logger(
                    f"❌ Forecast aborted. Not enough rows for sequence "
                    f"({len(df_tail)} < {self.forecaster.sequence_length}) ending at last known price {last_known_ts}."
                )
                self.system_state["last_forecast_cycle_status"] = "Failed: Insufficient sequence length."
                return

            # On that tail, required inputs must be non-null
            if df_tail[[config.TARGET_COLUMN, config.TEMPERATURE_COL_NAME, config.FINGRID_COL_NAME]].isnull().values.any():
                self.logger("❌ Forecast aborted. NaNs found inside sequencing tail ending at last known price timestamp.")
                self.logger(
                    df_tail[[config.TIME_COLUMN, config.TARGET_COLUMN, config.TEMPERATURE_COL_NAME, config.FINGRID_COL_NAME]]
                    .tail(12)
                    .to_string(index=False)
                )
                self.system_state["last_forecast_cycle_status"] = "Failed: NaNs in sequencing tail."
                return

            self.logger(f"✅ Sequencing tail validated. Anchor(last known price)={last_known_ts.isoformat()}")


            # Generating Forecast
            self.logger("Step 7: Passing data to ForecastingAgent for prediction...")
            model_tag = self.system_state.get("current_model_version_tag")

            if len(df_for_prediction) < self.forecaster.sequence_length:
                self.logger(f"❌ Not enough data for forecasting (need at least {self.forecaster.sequence_length} points for sequence). Have {len(df_for_prediction)}.")
                self.system_state["last_forecast_cycle_status"] = "Failed: Insufficient data for sequencing."
                return

            forecast_result = self.forecaster.run(merged_df=df_for_prediction, model_version_tag=model_tag)

            try:
                fdf = forecast_result.get("forecast_df", pd.DataFrame())
                if not fdf.empty:
                    f_min = pd.to_datetime(fdf[config.TIME_COLUMN].min(), utc=True)
                    max_known = pd.to_datetime(df_for_prediction.dropna(subset=[config.TARGET_COLUMN])[config.TIME_COLUMN].max(), utc=True)
                    self.logger(f"[ANCHOR_CHECK] max_known_target_ts={max_known} | first_forecast_ts={f_min}")
                    if f_min <= max_known:
                        self.logger("[ANCHOR_CHECK] ❌ Forecast still overlaps known targets. Investigate target NaNs / fill behavior.")
                    else:
                        self.logger("[ANCHOR_CHECK] ✅ Forecast starts after last known target.")
            except Exception as e:
                self.logger(f"[ANCHOR_CHECK] ⚠️ Failed: {e}")

            self.system_state["forecast_df"] = forecast_result.get("forecast_df", pd.DataFrame())

            try:
                rebuild_eprice_forecasts_best(self.db_path)
                self.logger("Orchestrator: eprice_forecasts_best rebuilt (shortest lead time wins).")
            except Exception as e:
                self.logger(f"❌ Failed to rebuild eprice_forecasts_best: {e}")

            self.logger("Forecast generated.")


            self.logger(f"DEBUG (OrchestratingAgent.run_forecasting_cycle): system_state['forecast_df'] has {len(self.system_state['forecast_df'])} records.")
            if not self.system_state["forecast_df"].empty:
                self.logger(f"DEBUG (OrchestratingAgent.run_forecasting_cycle): system_state['forecast_df'] head:\n{self.system_state['forecast_df'].head()}")

            # Generating interpretation
            self.logger("Step 7.1: Generating forecast interpretation...")
            interpretation = self.explain_recent_forecast()
            self.system_state["forecast_interpretation"] = interpretation
            self.logger("✅ Forecast interpretation generated.")

            # Final state update
            self.system_state["last_processed_eprice_timestamp"] = last_available_eprice_ts.isoformat()
            self.system_state["last_forecast_cycle_status"] = "Success"
            self.logger("✅ Forecasting cycle completed successfully.")

        except Exception as e:
            self.logger(f"❌ Forecasting Cycle Exception: {e}")
            import traceback; self.logger(traceback.format_exc())
            self.system_state["last_forecast_cycle_status"] = f"Failed - Exception: {e}"
            self.system_state["forecast_interpretation"] = f"Error during forecast: {e}"

   
    # Actuator Control Cycle
    def run_actuator_control_cycle(self):
        self.logger("Starting actuator control cycle...")
        self.system_state["last_actuator_cycle_status"] = "Running..."
        try:
            # Pull the most recent sensor values from the in-memory cache
            raw = getattr(self.sensor_monitor, "latest_sensor_data", {}) or {}
            # Expecting entries like {"temp_outdoor": {"value": 12.3, "timestamp": ...}, ...}
            live_temps = {
                k: (v.get("value") if isinstance(v, dict) else v)
                for k, v in raw.items()
            }

            # DEBUG: verify sensor snapshot passed to actuator control
            self.logger(
                "[ACT_CTRL_DEBUG] sensor_monitor.latest_sensor_data="
                f"{self.sensor_monitor.latest_sensor_data}"
            )

            self.logger(f"[ACT_CTRL_DEBUG] live_temps keys={list(live_temps.keys()) if isinstance(live_temps, dict) else type(live_temps)} raw={live_temps}")
            
            # The ActuatorCtrlAgent fetches prices from ground_truth_table itself.
            control_signals = self.actuator_controller.run_control_logic(
                live_temps=live_temps,
                control_params=self.system_state["control_parameters"] # Pass current control parameters, including the new limit
            )

            self.system_state["control_signals"] = control_signals
            self.system_state["last_actuator_cycle_status"] = "Success"
            self.logger(f"✅ Actuator control cycle completed successfully. Signals: {control_signals}")

        except Exception as e:
            self.system_state["last_actuator_cycle_status"] = f"Failed: {e}"
            self.logger(f"❌ Actuator control cycle failed: {e}")
            import traceback
            self.logger(traceback.format_exc())        

             
            
    def run_retraining_cycle(self, train_from_scratch: bool = False) -> Optional[str]:
        self.logger(f"Starting full model re-training cycle (train_from_scratch={train_from_scratch})...")
        self.system_state["last_retraining_cycle_status"] = "Running..."

        try:
            model, version_id = self.retrainer.retrain_model(train_from_scratch=train_from_scratch)

            if model is None:
                self.logger("❌ Retraining failed. No model was returned.")
                self.system_state["last_retraining_cycle_status"] = "Failed - Retraining returned None"
                return None
            
            self.forecaster._load_model_and_scalers()

            self.system_state["current_model_path"] = self.forecaster.model_path
            self.system_state["current_model_version_tag"] = self.forecaster.version_id
            self.system_state["last_retraining_cycle_status"] = f"Success - New retrained model active! (Version: {self.forecaster.version_id})"

            self.logger(f"✅ Retraining complete. New model active: {self.forecaster.model_path}, Version: {self.forecaster.version_id}")
            return self.forecaster.model_path
        
        except Exception as e:
            self.logger(f"❌ Retraining Cycle: Exception occurred - {e}")
            import traceback; self.logger(traceback.format_exc())
            self.system_state["last_retraining_cycle_status"] = f"Failed - Exception: {e}"
            return None
            

    def run_finetuning_cycle(self, days_of_data: int = 7) -> Optional[str]:
        self.logger(f"Starting fine-tuning cycle using last {days_of_data} days of data...")
        self.system_state["last_finetune_cycle_status"] = "Running..."

        # Explicitly disable fine-tuning placeholder ---
        self.logger("--- FINE-TUNING IS CURRENTLY DISABLED. SKIPPING OPERATION. ---")
        self.logger("Fine-tuning requires sufficient new data and further implementation. This cycle is a no-op.")
        self.system_state["last_finetune_cycle_status"] = "Disabled" # Update final status
        return None # Immediately return None, no further logic.
        # Fine-tuning agent will be used when there's enough 15min eprice data for the model, until that the model is fully retrained
        try:
            self.logger("Fine-tuning agent is not yet fully integrated/enabled. Placeholder.")
            new_model_path = None

            if new_model_path and os.path.exists(new_model_path):
                self.logger(f"✅ Fine-tuning successful. New model candidate: {new_model_path}")
                self.logger(f"Attempting to activate new model: {new_model_path}")
                
                previous_model_version = self.forecaster.version_id
                self.forecaster._load_model_and_scalers()
                
                if self.forecaster.version_id == os.path.basename(os.path.dirname(new_model_path)):
                    self.system_state["current_model_path"] = self.forecaster.model_path
                    self.system_state["current_model_version_tag"] = self.forecaster.version_id
                    status_msg = f"Success - New model active: {self.forecaster.model_path}"
                    self.logger(f"New model {self.forecaster.version_id} is now active.")
                    self.system_state["last_finetune_cycle_status"] = status_msg
                    return self.forecaster.model_path
                else:
                    self.logger(f"❌ Failed to load/activate new model {new_model_path}. Reverting to {previous_model_version}.")
                    self.system_state["last_finetune_cycle_status"] = "Failed - Model activation error"
                    return None
            else:
                self.logger("❌ Fine-tuning skipped, failed, or no new model file produced.")
                self.system_state["last_finetune_cycle_status"] = "Failed - No new model produced or agent disabled."
                return None
        except Exception as e:
            self.logger(f"❌ Fine-tuning Cycle: Exception occurred - {e}")
            import traceback; self.logger(traceback.format_exc())
            self.system_state["last_finetune_cycle_status"] = f"Failed - Exception: {e}"
            return None


    
    
    def explain_recent_forecast(self, forecast_df: Optional[pd.DataFrame] = None) -> str:
        """
        Generates an interpretation of recent electricity price data using the LLM agent
        or a programmatic fallback. Fetches relevant data from ground_truth_table,
        including all relevant features for a richer analysis.
        The 'forecast_df' argument is ignored as data is fetched directly for robustness.
        """
        self.logger("Orchestrator: Attempting to explain recent forecast...")

        now_utc = pd.Timestamp.now(tz='UTC')
        
        # Define the interpretation window: from 24 hours backwards from 'now'
        # until the absolute last datetime in ground_truth_table.
        history_start_time = now_utc - pd.Timedelta(hours=config.HOURS_FOR_INTERPRETATION_HISTORY)
        
        
        # Fetching until 2 days from now. The LLM will process what's available.        
        try:
            conn = sqlite3.connect(self.db_path)
            # Find the absolute latest timestamp in ground_truth_table
            max_db_datetime_str = conn.execute(f"SELECT MAX({config.TIME_COLUMN}) FROM {config.GROUND_TRUTH_TABLE}").fetchone()[0]
            max_db_datetime = pd.to_datetime(max_db_datetime_str, utc=True)
            conn.close()
            self.logger(f"Orchestrator: Latest data in ground_truth_table is {max_db_datetime.isoformat()}")
        except Exception as e:
            self.logger(f"Error fetching max datetime from ground_truth_table: {str(e)}. Defaulting to now + 2 days.")
            max_db_datetime = now_utc + pd.Timedelta(days=2) # Fallback to cover future
        
        interpretation_end_time = max_db_datetime # Use the actual max datetime from the table

        # Interpreting the BEST forecasts (shortest lead-time wins)
        try:
            start_str = (now_utc - pd.Timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
            end_str = interpretation_end_time.strftime("%Y-%m-%dT%H:%M:%SZ")

            with sqlite3.connect(self.db_path) as conn:
                effective_interpretation_df = pd.read_sql_query(
                    f"""
                    SELECT
                        b.forecasted_for_timestamp AS datetime,
                        b.predicted_eprice         AS eprice_15min,
                        gt.temp                    AS temp,
                        gt.windEnergy              AS windEnergy,
                        gt.temp_is_forecasted      AS temp_is_forecasted
                    FROM eprice_forecasts_best b
                    LEFT JOIN {config.GROUND_TRUTH_TABLE} gt
                        ON gt.{config.TIME_COLUMN} = b.forecasted_for_timestamp
                    WHERE b.forecasted_for_timestamp > ?
                    AND b.forecasted_for_timestamp <= ?
                    ORDER BY b.forecasted_for_timestamp ASC
                    """,
                    conn,
                    params=(start_str, end_str),
                    parse_dates=["datetime"],
                )

        except Exception as e:
            self.logger(f"Error fetching best forecasts for interpretation: {e}")
            effective_interpretation_df = pd.DataFrame()

        # Guard: prevent KeyError('datetime') when SQL fails or returns no rows
        if (
            effective_interpretation_df is None
            or effective_interpretation_df.empty
            or config.TIME_COLUMN not in effective_interpretation_df.columns
        ):
            self.logger("Orchestrator: No forecast rows available for interpretation window.")
            return "No forecast data available to generate an explanation."
        
        '''
        # Ensure datetimes are properly parsed and timezone-aware
        if not effective_interpretation_df.empty:
            effective_interpretation_df[config.TIME_COLUMN] = pd.to_datetime(effective_interpretation_df[config.TIME_COLUMN], utc=True, errors='coerce')
            effective_interpretation_df.dropna(subset=["datetime", config.TARGET_COLUMN], inplace=True)# Ensure critical columns are not null
            effective_interpretation_df.sort_values(by=config.TIME_COLUMN, inplace=True) # Ensure sorted
        '''

        effective_interpretation_df[config.TIME_COLUMN] = pd.to_datetime(
            effective_interpretation_df[config.TIME_COLUMN], utc=True, errors="coerce"
        )
        effective_interpretation_df.dropna(
            subset=[config.TIME_COLUMN, config.TARGET_COLUMN], inplace=True
        )
        effective_interpretation_df.sort_values(by=config.TIME_COLUMN, inplace=True)

        

        # Keep only the most recent 96 rows (24h at 15-min cadence)
        MAX_ROWS_FOR_LLM = 96
        effective_interpretation_df.sort_values(by=config.TIME_COLUMN, inplace=True)
        if len(effective_interpretation_df) > MAX_ROWS_FOR_LLM:
            effective_interpretation_df = effective_interpretation_df.tail(MAX_ROWS_FOR_LLM).copy()
        
        explanation_to_return = "No relevant price data available from ground_truth_table for interpretation."

        if effective_interpretation_df is not None and not effective_interpretation_df.empty:
            self.logger(f"Orchestrator: Retrieved {len(effective_interpretation_df)} rows of data from ground_truth_table for interpretation.")
            
            # Safe building of context_data with debug logs 
            try:
                context_data = {
                    k: v['value']
                    for k, v in self.sensor_monitor.latest_sensor_data.items()
                    if isinstance(v, dict) and 'value' in v
                }
                self.logger(f"[LLM DEBUG] Built context_data with {len(context_data)} sensor entries.")
            except Exception as e_ctx:
                self.logger(f"[LLM DEBUG] Error building context_data for LLM: {e_ctx}. Using empty context.")
                context_data = {}

            # LLM call with detailed debug logs 
            try:
                self.logger(
                    f"[LLM DEBUG] Calling LLMAgent.interpret_forecast with "
                    f"{len(effective_interpretation_df)} rows, target={config.TARGET_COLUMN}."
                )
                explanation_to_return = self.llm_agent.interpret_forecast(
                    effective_interpretation_df,
                    config.TARGET_COLUMN,
                    context_data=context_data,
                )
                self.logger(
                    f"[LLM DEBUG] LLMAgent returned explanation (len={len(explanation_to_return) if explanation_to_return else 0})."
                )
            except Exception as e_llm_interpret:
                self.logger(
                    f"[LLM DEBUG] Exception calling LLMAgent.interpret_forecast: {e_llm_interpret}. "
                    f"Falling back to programmatic summary."
                )
                explanation_to_return = self._programmatic_explanation(
                    effective_interpretation_df, config.TARGET_COLUMN
                )

            # Detect fallback text coming back from LLMAgent 
            if (
                not explanation_to_return
                or "LLM interpretation failed" in explanation_to_return
                or "No explanation in response" in explanation_to_return
            ):
                self.logger(
                    "[LLM DEBUG] LLMAgent returned fallback/failed explanation. "
                    "Generating programmatic forecast explanation."
                )
                explanation_to_return = self._programmatic_explanation(
                    effective_interpretation_df, config.TARGET_COLUMN
                )

        else:
            self.logger("Orchestrator: No relevant price data available from ground_truth_table for interpretation.")

        return explanation_to_return
    
    

    
    
    def _programmatic_explanation(self, df: pd.DataFrame, target_col: str) -> str:
        if df.empty or target_col not in df.columns or df[target_col].isna().all():
            return f"Programmatic Summary: No valid data for {target_col} to summarize."
        
        avg_price = df[target_col].mean()
        min_price_row = df.loc[df[target_col].idxmin()]
        max_price_row = df.loc[df[target_col].idxmax()]
        
        # Scalar timestamp handling 
        min_ts_str = "N/A"
        max_ts_str = "N/A"
        min_hour_min_str = ""
        max_hour_min_str = ""

        min_val = min_price_row.get(config.TIME_COLUMN)
        max_val = max_price_row.get(config.TIME_COLUMN)

        if isinstance(min_val, pd.Timestamp):
            min_ts_dt = min_val.tz_convert('Europe/Helsinki') if min_val.tzinfo else min_val.tz_localize('UTC').tz_convert('Europe/Helsinki')
            min_ts_str = min_ts_dt.strftime('%Y-%m-%d %H:%M')
            min_hour_min_str = f" (Hour: {min_ts_dt.hour:02d}, Minute: {min_ts_dt.minute:02d})"

        if isinstance(max_val, pd.Timestamp):
            max_ts_dt = max_val.tz_convert('Europe/Helsinki') if max_val.tzinfo else max_val.tz_localize('UTC').tz_convert('Europe/Helsinki')
            max_ts_str = max_ts_dt.strftime('%Y-%m-%d %H:%M')
            max_hour_min_str = f" (Hour: {max_ts_dt.hour:02d}, Minute: {max_ts_dt.minute:02d})"
        
        
        return (f"Programmatic Summary: Data for {len(df)} 15-min intervals (history + day-ahead real prices).\n"
                f"Average {target_col}: {avg_price:.2f} snt/kWh.\n"
                f"Lowest {target_col}: {min_price_row[target_col]:.2f} snt/kWh at {min_ts_str}{min_hour_min_str}.\n"
                f"Highest {target_col}: {max_price_row[target_col]:.2f} snt/kWh at {max_ts_str}{max_hour_min_str}.\n")

    # The method accepts start_time_str and end_time_str directly:
    def query_stored_data(self, hours: Optional[int] = None, days: Optional[int] = None, 
                          start_time_str: Optional[str] = None, end_time_str: Optional[str] = None,
                          data_type: str = "historical") -> pd.DataFrame:
        self.logger(f"Querying DB for stored {data_type} data from {start_time_str} to {end_time_str}...")
        df = pd.DataFrame()
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.isolation_level = None 

            end_time_dt = datetime.now(timezone.utc) if end_time_str is None else pd.to_datetime(end_time_str, utc=True)
            
            if start_time_str is None:
                if days is not None:
                    start_time_dt = end_time_dt - timedelta(days=days)
                elif hours is not None:
                    start_time_dt = end_time_dt - timedelta(hours=hours)
                else:
                    start_time_dt = end_time_dt - timedelta(hours=24)
            else:
                start_time_dt = pd.to_datetime(start_time_str, utc=True)

            start_time_iso = start_time_dt.strftime('%Y-%m-%dT%H:%M:%SZ')
            end_time_iso = end_time_dt.strftime('%Y-%m-%dT%H:%M:%SZ')

            if data_type == "forecasts":
                recent_generation_threshold = end_time_dt - timedelta(minutes=5)
                
                query_recent_forecasts = f"""
                    SELECT forecast_generation_time, forecasted_for_timestamp, predicted_eprice, model_version 
                    FROM eprice_forecasts 
                    WHERE forecast_generation_time >= ? 
                    ORDER BY forecast_generation_time DESC, forecasted_for_timestamp ASC
                """
                
                temp_df = pd.read_sql_query(query_recent_forecasts, conn, params=(recent_generation_threshold.strftime('%Y-%m-%dT%H:%M:%SZ'),))
                
                if not temp_df.empty:
                    temp_df['forecast_generation_time'] = pd.to_datetime(temp_df['forecast_generation_time'], utc=True)
                    temp_df['forecasted_for_timestamp'] = pd.to_datetime(temp_df['forecasted_for_timestamp'], utc=True)
                    
                    latest_generation_time = temp_df['forecast_generation_time'].max()
                    df = temp_df[temp_df['forecast_generation_time'] == latest_generation_time].copy()
                    
                if df.empty:
                    self.logger(f"ℹ️ No *recent* forecasts found (generated in last 5 min). Falling back to general time window query for forecasts.")
                    query_general_forecasts = f"""
                        SELECT forecast_generation_time, forecasted_for_timestamp, predicted_eprice, model_version 
                        FROM eprice_forecasts 
                        WHERE forecasted_for_timestamp BETWEEN ? AND ? 
                        ORDER BY forecast_generation_time DESC, forecasted_for_timestamp ASC 
                    """
                    temp_df = pd.read_sql_query(query_general_forecasts, conn, params=(start_time_iso, end_time_iso))
                    if not temp_df.empty:
                        temp_df['forecast_generation_time'] = pd.to_datetime(temp_df['forecast_generation_time'], utc=True)
                        temp_df['forecasted_for_timestamp'] = pd.to_datetime(temp_df['forecasted_for_timestamp'], utc=True)
                        latest_gen_time_in_window = temp_df['forecast_generation_time'].max()
                        df = temp_df[temp_df['forecast_generation_time'] == latest_gen_time_in_window].copy()
                    else:
                        df = pd.DataFrame()
                
            elif data_type == "historical": # <--> ground_truth_table
                # Fetch all relevant columns for interpretation
                columns_to_fetch = [
                    config.TIME_COLUMN,
                    config.TARGET_COLUMN,
                    config.TEMPERATURE_COL_NAME, # 'temp'
                    config.FINGRID_COL_NAME,     # 'windEnergy'
                    'hour_extracted',
                    'minute_extracted',
                    'temp_is_forecasted'         # Include the flag for LLM context
                ]
                columns_str = ", ".join(columns_to_fetch)

                query = f"""
                    SELECT {columns_str}
                    FROM {config.GROUND_TRUTH_TABLE}
                    WHERE {config.TIME_COLUMN} BETWEEN ? AND ?
                    ORDER BY {config.TIME_COLUMN} ASC
                """
                df = pd.read_sql_query(query, conn, params=(start_time_iso, end_time_iso), parse_dates=[config.TIME_COLUMN])

            else:
                self.logger(f"Invalid data_type: {data_type}")
        except Exception as e:
            self.logger(f"Error retrieving {data_type} data: {str(e)}")
        finally:
            if conn: conn.close()
        
        if df.empty:
            self.logger(f"ℹ️ No {data_type} data found.")
        else:
            self.logger(f"✅ Retrieved {len(df)} rows of {data_type} data.")
            if config.TIME_COLUMN in df.columns:
                s = pd.to_datetime(df[config.TIME_COLUMN], errors='coerce')
                # Localize naive to UTC, or convert aware to UTC
                if s.dt.tz is None:
                    s = s.dt.tz_localize('UTC')
                else:
                    s = s.dt.tz_convert('UTC')
                df[config.TIME_COLUMN] = s
        return df

    @property
    def db_path(self):
        return config.FORECASTING_DB_PATH






    