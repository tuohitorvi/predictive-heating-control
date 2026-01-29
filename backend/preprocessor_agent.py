#backend/preprocessor_agent.py
import pandas as pd
from typing import Dict, Any
import numpy as np
from . import config
import sqlite3
import itertools

from .data_fetcher_agent import DataFetcherAgent

def preprocess_and_align_data(raw_data_from_fetcher: Dict[str, Any]) -> pd.DataFrame:
    print("🔄 Starting data preprocessing and alignment...")
    
    processed_dfs = {}

    # 1. Process Fingrid Data
    fingrid_data = raw_data_from_fetcher.get("fingrid", [])
    valid_fingrid_data = [item for item in fingrid_data if isinstance(item, dict)]
    if valid_fingrid_data:
        df = pd.DataFrame(valid_fingrid_data)
        print("🔍 Fingrid DataFrame preview:")
        print(df.head())
        print("Columns:", df.columns.tolist())

        if "startTime" in df.columns:
            df = df.rename(columns={
                "startTime": config.TIME_COLUMN,
                "Tuulivoiman tuotantoennuste - päivitys 15 minuutin välein": config.FINGRID_COL_NAME
            })

            df[config.TIME_COLUMN] = pd.to_datetime(df[config.TIME_COLUMN], utc=True, errors='coerce')
            df[config.FINGRID_COL_NAME] = pd.to_numeric(df[config.FINGRID_COL_NAME], errors='coerce')
            df.dropna(subset=[config.TIME_COLUMN, config.FINGRID_COL_NAME], inplace=True) # Drop NaNs here explicitly for Fingrid

            
            if not df.empty:
                # Resample to 15min and forward-fill to maintain/upsample resolution
                df = df.set_index(config.TIME_COLUMN).resample("15min").ffill().reset_index()
                processed_dfs["fingrid"] = df
                print("📊 Processed Fingrid data to 15-min resolution.")
                print("✅ Processed Fingrid data preview:")
                print(df.head())
                

            else:
                print("[PreprocessorAgent] ℹ️ Fingrid DataFrame is empty after cleaning. Skipping to next source.")
                processed_dfs["fingrid"] = pd.DataFrame(columns=[config.TIME_COLUMN, config.FINGRID_COL_NAME])

    else:
        print("[PreprocessorAgent] ℹ️ Fingrid DataFrame is empty after cleaning. Skipping to next source.")
        processed_dfs["fingrid"] = pd.DataFrame(columns=[config.TIME_COLUMN, config.FINGRID_COL_NAME])
        
    # 2. Process FMI Data
    fmi_data = raw_data_from_fetcher.get("fmi", {}).get("observations", [])
    if isinstance(fmi_data, list) and fmi_data:
        df = pd.DataFrame(fmi_data)
        if "time" in df.columns:
            df = df.rename(columns={"time": config.TIME_COLUMN, "t2m": config.TEMPERATURE_COL_NAME})
            df[config.TIME_COLUMN] = pd.to_datetime(df[config.TIME_COLUMN], utc=True, errors='coerce')
            df[config.TEMPERATURE_COL_NAME] = pd.to_numeric(df[config.TEMPERATURE_COL_NAME], errors='coerce')
            df.dropna(subset=[config.TIME_COLUMN, config.TEMPERATURE_COL_NAME], inplace=True)
            
            if not df.empty:
                # Resample to 15min and forward-fill to maintain/upsample resolution
                df = df.set_index(config.TIME_COLUMN).resample("15min").ffill().reset_index()
                processed_dfs["fmi"] = df
                print("🌦️ Processed FMI weather data to 15-min resolution.")
                print(f"Preprocessed fmi data: {df.head()}")

            else:
                print("[PreprocessorAgent] ℹ️ FMI DataFrame is empty after cleaning. Skipping to next source.")
                processed_dfs["fmi"] = pd.DataFrame(columns=[config.TIME_COLUMN, config.TEMPERATURE_COL_NAME])

        else:
            print("[PreprocessorAgent] ℹ️ No valid raw FMI data to process.")
            processed_dfs["fmi"] = pd.DataFrame(columns=[config.TIME_COLUMN, config.TEMPERATURE_COL_NAME])
    

    # 3. Process Elering Data
    elering_data = raw_data_from_fetcher.get("elering", {}).get("data", [])
    if isinstance(elering_data, list) and elering_data:
        df = pd.DataFrame(elering_data)
        if "datetime" in df.columns and "price_eur_per_mwh" in df.columns:
            df = df.rename(columns={"datetime": config.TIME_COLUMN})
            df[config.TIME_COLUMN] = pd.to_datetime(df[config.TIME_COLUMN], utc=True, errors='coerce')
            df['price_eur_per_mwh'] = pd.to_numeric(df['price_eur_per_mwh'], errors='coerce')
            df.dropna(subset=[config.TIME_COLUMN, 'price_eur_per_mwh'], inplace=True)

            df[config.TARGET_COLUMN] = (df['price_eur_per_mwh'] * (1 + config.ALV)) / 10.0
            
            if not df.empty:
                elering_series = df.set_index(config.TIME_COLUMN)[config.TARGET_COLUMN].resample("15min").ffill()
                resampled_df = elering_series.reset_index()
                processed_dfs["elering"] = resampled_df[[config.TIME_COLUMN, config.TARGET_COLUMN]]
                print("💡 Processed Elering price data to 15-min resolution.")
                print(f"Preprocessed elering data: {df.head()}")
            else:
                print("[PreprocessorAgent] ℹ️ Elering DataFrame is empty after price calculation. Skipping to next source.")
                processed_dfs["elering"] = pd.DataFrame(columns=[config.TIME_COLUMN, config.TARGET_COLUMN])     
            
    else: # The cases where elering_data is not a valid list or is empty initially
        print("[PreprocessorAgent] ℹ️ No valid raw Elering data to process.")
        processed_dfs["elering"] = pd.DataFrame(columns=[config.TIME_COLUMN, config.TARGET_COLUMN])

    # 4. Merge DataFrames
    if not processed_dfs:
        print("❌ No data available from any source to merge.")
        return pd.DataFrame()

    non_empty_dfs = [df for df in processed_dfs.values() if not df.empty]
    if not non_empty_dfs:
        print("❌ All processed DataFrames are empty. No data to merge.")
        return pd.DataFrame()

    merged_df = non_empty_dfs[0]
    for df_to_merge in non_empty_dfs[1:]:
        merged_df = pd.merge(merged_df, df_to_merge, on=config.TIME_COLUMN, how="outer")

    print(f"Preprocessed and merged data, head: {merged_df.head()}")

    for col in [config.TARGET_COLUMN, config.TEMPERATURE_COL_NAME, config.FINGRID_COL_NAME]:
        if col not in merged_df.columns:
            merged_df[col] = np.nan
            print(f"⚠️ Created placeholder column for missing data source: '{col}'")

    if 'temp_is_forecasted' not in merged_df.columns:
        merged_df['temp_is_forecasted'] = 0

    if merged_df[config.TIME_COLUMN].dt.tz is None:
        merged_df[config.TIME_COLUMN] = merged_df[config.TIME_COLUMN].dt.tz_localize('UTC')
    else:
        merged_df[config.TIME_COLUMN] = merged_df[config.TIME_COLUMN].dt.tz_convert('UTC')

    merged_df['datetime_local'] = merged_df[config.TIME_COLUMN].dt.tz_convert('Europe/Helsinki')
    
    merged_df['hour_extracted'] = merged_df[config.TIME_COLUMN].dt.hour
    merged_df['minute_extracted'] = merged_df[config.TIME_COLUMN].dt.minute

    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

    print(f"✅ Data preprocessing complete. Final DataFrame shape: {merged_df.shape}.")

        
    return merged_df


class PreprocessorAgent:
    def __init__(self, logger=None):
        if logger:
            self._log_operation = logger
        else:
            self._log_operation = lambda msg: print(f"PreprocessorAgent Log: {msg}")
        self._log_operation("PreprocessorAgent initialized.")
        

    def run(self, raw_data: Dict[str, Any]) -> pd.DataFrame:
        self._log_operation("PreprocessorAgent: Processing raw data...")
        try:
            aligned_df = preprocess_and_align_data(raw_data)
            self._log_operation(f"Aligned_df after initial preprocessing: {aligned_df.head()}")
            return aligned_df
        except Exception as e:
            self._log_operation(f"PreprocessorAgent: Error during initial processing - {e}")
            import traceback
            self._log_operation(traceback.format_exc())
            return pd.DataFrame()
    

    # enhance_with_fmi_forecast()
    def enhance_with_fmi_forecast(self, merged_df: pd.DataFrame, data_fetcher: DataFetcherAgent) -> pd.DataFrame:
        self._log_operation("Enhancing data with FMI weather forecast...")
        enhanced_df = merged_df.copy()

        # Ensure temp_source column exists
        if "temp_source" not in enhanced_df.columns:
            enhanced_df["temp_source"] = pd.Series(index=enhanced_df.index, dtype="object")

        # Mark existing (observed) temps
        if config.TEMPERATURE_COL_NAME in enhanced_df.columns:
            obs_mask = enhanced_df[config.TEMPERATURE_COL_NAME].notna()
            enhanced_df.loc[obs_mask, "temp_source"] = "obs"
        else:
            enhanced_df[config.TEMPERATURE_COL_NAME] = np.nan

        fmi_forecast_obs = data_fetcher.fetch_fmi_forecast()
        if not fmi_forecast_obs:
            self._log_operation("⚠️ No FMI forecast data returned. Skipping enhancement.")
            return enhanced_df

        df_fmi_fc = pd.DataFrame(fmi_forecast_obs)
        if "time" not in df_fmi_fc.columns or "t2m" not in df_fmi_fc.columns:
            self._log_operation("⚠️ FMI forecast data is missing required 'time' or 't2m' columns.")
            return enhanced_df

        df_fmi_fc = df_fmi_fc.rename(columns={"time": config.TIME_COLUMN, "t2m": "forecast_temp"})
        df_fmi_fc[config.TIME_COLUMN] = (
            pd.to_datetime(df_fmi_fc[config.TIME_COLUMN], errors="coerce", utc=True).dt.floor("15min")
        )
        df_fmi_fc["forecast_temp"] = pd.to_numeric(df_fmi_fc["forecast_temp"], errors="coerce")
        df_fmi_fc.dropna(subset=[config.TIME_COLUMN, "forecast_temp"], inplace=True)

        if df_fmi_fc.empty:
            self._log_operation("⚠️ FMI forecast DataFrame is empty after cleaning. Skipping enhancement.")
            return enhanced_df

        df_fmi_fc = (
            df_fmi_fc.set_index(config.TIME_COLUMN)
            .resample("15min")
            .mean()
            .interpolate("linear")
            .ffill()
            .bfill()
            .reset_index()
        )
        self._log_operation(
            f"[FMI_FC] forecast range: {df_fmi_fc[config.TIME_COLUMN].min()} → {df_fmi_fc[config.TIME_COLUMN].max()} "
            f"rows={len(df_fmi_fc)}"
        )
        self._log_operation("✅ FMI weather forecast data interpolated to 15-min resolution.")

        enhanced_df = pd.merge(
            enhanced_df,
            df_fmi_fc[[config.TIME_COLUMN, "forecast_temp"]],
            on=config.TIME_COLUMN,
            how="outer",
            suffixes=("", "_fmi_fc"),
        )

        # Computing mask after merge on the merged frame
        forecast_fill_mask = (
            enhanced_df[config.TEMPERATURE_COL_NAME].isna()
            & enhanced_df["forecast_temp"].notna()
        )

        enhanced_df.loc[forecast_fill_mask, config.TEMPERATURE_COL_NAME] = enhanced_df.loc[
            forecast_fill_mask, "forecast_temp"
        ]
        enhanced_df.loc[forecast_fill_mask, "temp_source"] = "forecast"

        if "temp_is_forecasted" in enhanced_df.columns:
            enhanced_df.loc[forecast_fill_mask, "temp_is_forecasted"] = 1

        # Ensure numeric
        enhanced_df[config.TEMPERATURE_COL_NAME] = pd.to_numeric(
            enhanced_df[config.TEMPERATURE_COL_NAME], errors="coerce"
        )

        # Forward fill with limit
        ffill_mask = enhanced_df[config.TEMPERATURE_COL_NAME].isna()
        enhanced_df[config.TEMPERATURE_COL_NAME] = enhanced_df[config.TEMPERATURE_COL_NAME].ffill(limit=8)
        newly_ffilled = ffill_mask & enhanced_df[config.TEMPERATURE_COL_NAME].notna()
        enhanced_df.loc[newly_ffilled & enhanced_df["temp_source"].isna(), "temp_source"] = "ffill"

        # Backward fill (last resort)
        bfill_mask = enhanced_df[config.TEMPERATURE_COL_NAME].isna()
        enhanced_df[config.TEMPERATURE_COL_NAME] = enhanced_df[config.TEMPERATURE_COL_NAME].bfill(limit=8)
        newly_bfilled = bfill_mask & enhanced_df[config.TEMPERATURE_COL_NAME].notna()
        enhanced_df.loc[newly_bfilled & enhanced_df["temp_source"].isna(), "temp_source"] = "bfill"

        # Clean up
        if "forecast_temp" in enhanced_df.columns:
            enhanced_df.drop(columns=["forecast_temp"], inplace=True)

        # Keep flag as int
        if "temp_is_forecasted" in enhanced_df.columns:
            enhanced_df["temp_is_forecasted"] = enhanced_df["temp_is_forecasted"].astype("Int64", errors="ignore")
            enhanced_df.loc[forecast_fill_mask, "temp_is_forecasted"] = 1

        # temp_source finalization --> outer merge
        # Ensure column exists post-merge
        if "temp_source" not in enhanced_df.columns:
            enhanced_df["temp_source"] = pd.Series(index=enhanced_df.index, dtype="object")

        # Any row where temp is present but temp_source is still null:
        mask_temp_present_src_missing = (
            enhanced_df[config.TEMPERATURE_COL_NAME].notna() & enhanced_df["temp_source"].isna()
        )
        enhanced_df.loc[mask_temp_present_src_missing, "temp_source"] = "obs"

        # Any row where temp is still missing -> mark explicitly
        mask_temp_missing = enhanced_df[config.TEMPERATURE_COL_NAME].isna()
        enhanced_df.loc[mask_temp_missing, "temp_source"] = "missing"

        # Final diagnostics
        src_counts = enhanced_df["temp_source"].value_counts(dropna=False).to_dict()
        self._log_operation(f"🌡️ temp_source distribution: {src_counts}")
        self._log_operation(
            f"🌡️ temp unique last 96: {enhanced_df[config.TEMPERATURE_COL_NAME].tail(96).nunique(dropna=True)}"
        )

        self._log_operation("✅ Successfully updated 'temp' column with forecast data and temp_source flags.")
        self._log_operation(f"Enhanced forecast temp data head:\n{enhanced_df.head()}")

        return enhanced_df

    
    # fill_prediction_gaps()
    # - After ffill/bfill/0-fill ->  update temp_source for rows that were previously "missing"
    def fill_prediction_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        self._log_operation("Filling final gaps for prediction (NO target forward-fill)...")
        df_filled = df.copy()

        # Ensure numeric dtypes (best-effort)
        for col in [config.TARGET_COLUMN, config.TEMPERATURE_COL_NAME, config.FINGRID_COL_NAME]:
            if col in df_filled.columns:
                df_filled[col] = pd.to_numeric(df_filled[col], errors="coerce")

        # Ensure temp_source exists (don't overwrite existing values like "forecast")
        if "temp_source" not in df_filled.columns:
            df_filled["temp_source"] = pd.Series(index=df_filled.index, dtype="object")

        # fill/extrapolate only FEATURES. The target must remain historical-only.
        feature_cols = []
        if config.TEMPERATURE_COL_NAME in df_filled.columns:
            feature_cols.append(config.TEMPERATURE_COL_NAME)
        if config.FINGRID_COL_NAME in df_filled.columns:
            feature_cols.append(config.FINGRID_COL_NAME)

        if not feature_cols:
            self._log_operation("⚠️ No feature columns found for gap filling.")
            return df_filled

        
        # Forward-fill only features
        temp_nan_before_ffill = (
            df_filled[config.TEMPERATURE_COL_NAME].isnull().copy()
            if config.TEMPERATURE_COL_NAME in df_filled.columns
            else None
        )

        df_filled[feature_cols] = df_filled[feature_cols].ffill()

        # Label temp_source for values that were filled by ffill
        if temp_nan_before_ffill is not None and config.TEMPERATURE_COL_NAME in df_filled.columns:
            filled_by_ffill = temp_nan_before_ffill & df_filled[config.TEMPERATURE_COL_NAME].notnull()

            # allow overwriting "missing" (and only "missing"/null), but keep "forecast"/"obs"
            can_overwrite = df_filled["temp_source"].isna() | (df_filled["temp_source"] == "missing")
            df_filled.loc[filled_by_ffill & can_overwrite, "temp_source"] = "ffill"

            # Keep your existing flag behavior
            if "temp_is_forecasted" in df_filled.columns:
                df_filled.loc[filled_by_ffill, "temp_is_forecasted"] = 1

        # Extrapolate wind beyond last real
        wind_col = config.FINGRID_COL_NAME
        if wind_col in df_filled.columns and pd.api.types.is_numeric_dtype(df_filled[wind_col]):
            last_real_wind_idx = df_filled[wind_col].last_valid_index()
            if last_real_wind_idx is not None and last_real_wind_idx < len(df_filled) - 1:
                self._log_operation("Extrapolating wind forecast...")
                slope = 0.0
                if last_real_wind_idx >= 3:
                    slopes = [
                        df_filled.loc[last_real_wind_idx - i, wind_col]
                        - df_filled.loc[last_real_wind_idx - i - 1, wind_col]
                        for i in range(3)
                    ]
                    valid_slopes = [s for s in slopes if not np.isnan(s)]
                    slope = float(np.mean(valid_slopes)) if valid_slopes else 0.0
                elif last_real_wind_idx > 0:
                    p1 = df_filled.loc[last_real_wind_idx, wind_col]
                    p0 = df_filled.loc[last_real_wind_idx - 1, wind_col]
                    slope = float(p1 - p0)

                for i in range(last_real_wind_idx + 1, len(df_filled)):
                    prev = df_filled.loc[i - 1, wind_col]
                    df_filled.loc[i, wind_col] = max(0.0, float(prev) + slope)

                self._log_operation("✅ Wind energy forecast extrapolation complete.")

        
        # Back-fill ONLY features (leading NaNs)
        temp_nan_before_bfill = (
            df_filled[config.TEMPERATURE_COL_NAME].isnull().copy()
            if config.TEMPERATURE_COL_NAME in df_filled.columns
            else None
        )

        df_filled[feature_cols] = df_filled[feature_cols].bfill()

        # Label temp_source for values that were filled by bfill
        if temp_nan_before_bfill is not None and config.TEMPERATURE_COL_NAME in df_filled.columns:
            filled_by_bfill = temp_nan_before_bfill & df_filled[config.TEMPERATURE_COL_NAME].notnull()

            can_overwrite = df_filled["temp_source"].isna() | (df_filled["temp_source"] == "missing")
            df_filled.loc[filled_by_bfill & can_overwrite, "temp_source"] = "bfill"

            # Keep your existing flag behavior
            if "temp_is_forecasted" in df_filled.columns:
                df_filled.loc[filled_by_bfill, "temp_is_forecasted"] = 1

        # Final fill for features only
        temp_nan_before_zero = (
            df_filled[config.TEMPERATURE_COL_NAME].isnull().copy()
            if config.TEMPERATURE_COL_NAME in df_filled.columns
            else None
        )

        df_filled[feature_cols] = df_filled[feature_cols].fillna(0.0)

        # Tag any remaining missing that got zero-filled 
        if temp_nan_before_zero is not None and config.TEMPERATURE_COL_NAME in df_filled.columns:
            filled_by_zero = temp_nan_before_zero & df_filled[config.TEMPERATURE_COL_NAME].notnull()

            can_overwrite = df_filled["temp_source"].isna() | (df_filled["temp_source"] == "missing")
            df_filled.loc[filled_by_zero & can_overwrite, "temp_source"] = "zero_fill"

            if "temp_is_forecasted" in df_filled.columns:
                df_filled.loc[filled_by_zero, "temp_is_forecasted"] = 1


        return df_filled


    