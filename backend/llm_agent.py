# backend/llm_agent.py
import requests
import pandas as pd
from typing import Dict, Any, Optional
import time
import sqlite3
from datetime import datetime, timezone
from . import config

class LLMAgent:
    def __init__(self, llm_service_url: str, logger=None):
        self.logger = logger if logger else print
        self.llm_service_url = llm_service_url
        self.interpret_endpoint = f"{llm_service_url}/interpret-forecast"
        self.logger(f"LLMAgent initialized, connecting to LLM service at {llm_service_url}")

    def _load_best_forecast_df_from_db(
        self,
        db_path: str,
        horizon_steps: int,
        start_after_utc: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        Loads best-per-timestamp forecasts from eprice_forecasts_best.

        Returns a DataFrame with at least:
        - datetime
        - eprice_15min (or predicted_eprice, depending on how you want to pass it)
        - model_version (optional)
        """
        if start_after_utc is None:
            start_after_utc = pd.Timestamp.now(tz="UTC")

        # UTC Zulu strings, DB stores TEXT like 'YYYY-mm-ddTHH:MM:SSZ'
        start_after_str = start_after_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

        try:
            with sqlite3.connect(db_path) as conn:
                df = pd.read_sql_query(
                    """
                    SELECT
                        forecasted_for_timestamp AS datetime,
                        predicted_eprice,
                        model_version
                    FROM eprice_forecasts_best
                    WHERE forecasted_for_timestamp > ?
                    ORDER BY forecasted_for_timestamp ASC
                    LIMIT ?
                    """,
                    conn,
                    params=(start_after_str, int(horizon_steps)),
                    parse_dates=["datetime"],
                )
        except Exception as e:
            self.logger(f"❌ Failed to load eprice_forecasts_best from DB: {e}")
            return pd.DataFrame()

        if df.empty:
            return df

        # Ensure tz-aware UTC
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
        df = df.loc[df["datetime"].notna()].copy()

        df[config.TARGET_COLUMN] = pd.to_numeric(df["predicted_eprice"], errors="coerce")
        df.drop(columns=["predicted_eprice"], inplace=True, errors="ignore")

        return df


    def interpret_best_forecast_from_db(
        self,
        db_path: str,
        target_column: str,
        horizon_steps: int,
        context_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Convenience wrapper: loads best forecasts from DB and calls interpret_forecast().
        """
        df_best = self._load_best_forecast_df_from_db(
            db_path=db_path,
            horizon_steps=horizon_steps,
            start_after_utc=pd.Timestamp.now(tz="UTC"),
        )

        if df_best.empty:
            return "No best-forecast data available to generate an explanation."

        return self.interpret_forecast(
            forecast_df=df_best,
            target_column=target_column,
            context_data=context_data,
        )


    def interpret_forecast(
        self,
        forecast_df: pd.DataFrame,
        target_column: str,
        context_data: Optional[Dict[str, Any]] = None
    ) -> str:
        if forecast_df.empty:
            return "No forecast data available to generate an explanation."

        df_to_serialize = forecast_df.copy()

        if 'datetime' in df_to_serialize.columns:
            dt = pd.to_datetime(df_to_serialize['datetime'], errors='coerce', utc=True)
            df_to_serialize = df_to_serialize.loc[dt.notna()].copy()
            df_to_serialize['datetime'] = dt.loc[dt.notna()].dt.strftime('%Y-%m-%dT%H:%M:%SZ')

        df_to_serialize = df_to_serialize.where(df_to_serialize.notna(), None)

        needed_cols = [
            'datetime',
            target_column,
            'temp',
            'windEnergy',
            'temp_is_forecasted',
            'hour_extracted',
            'minute_extracted',
        ]
        existing = [c for c in needed_cols if c in df_to_serialize.columns]
        df_to_serialize = df_to_serialize[existing].copy()

        # Limit rows defensively
        MAX_ROWS = 64
        if len(df_to_serialize) > MAX_ROWS:
            self.logger(
                f"[LLM DEBUG] Truncating forecast rows from {len(df_to_serialize)} to last {MAX_ROWS}."
            )
            df_to_serialize = df_to_serialize.tail(MAX_ROWS)

        forecast_data_list = df_to_serialize.to_dict(orient="records")

        request_payload = {
            "forecast_data": forecast_data_list,
            "target_column": target_column,
            "context_data": context_data,
        }

        TIMEOUT_S = 120
        MAX_RETRIES = 2
        backoff = 2

        self.logger(
            f"[LLM DEBUG] Sending request to {self.interpret_endpoint} with "
            f"{len(forecast_data_list)} rows, timeout={TIMEOUT_S}s"
        )

        for attempt in range(1, MAX_RETRIES + 2):
            try:
                response = requests.post(
                    self.interpret_endpoint,
                    json=request_payload,
                    timeout=TIMEOUT_S,
                )
                self.logger(
                    f"[LLM DEBUG] HTTP status from LLM service (attempt {attempt}): {response.status_code}"
                )
                response.raise_for_status()
                result = response.json()
                explanation = result.get("explanation")
                self.logger(
                    f"[LLM DEBUG] Received explanation length: {len(explanation) if explanation else 0}"
                )
                return explanation or "LLM interpretation failed: No explanation in response."
            except requests.exceptions.Timeout:
                self.logger(
                    f"❌ LLM service request timed out after {TIMEOUT_S} seconds (attempt {attempt})."
                )
            except requests.exceptions.ConnectionError:
                self.logger(
                    f"❌ LLM service connection error. Is the LLM service running at {self.llm_service_url}?"
                )
                break
            except requests.exceptions.RequestException as e:
                self.logger(f"❌ LLM service request failed (attempt {attempt}): {e}")

            if attempt <= MAX_RETRIES:
                self.logger(f"[LLM DEBUG] Backing off for {backoff}s before retry...")
                time.sleep(backoff)
                backoff *= 2

        return "LLM interpretation failed: Request timed out or errored."