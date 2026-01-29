# scripts/upsert_ground_truth_table.py
import os
import sys
import sqlite3
import argparse
import pandas as pd

# Ensure backend import works when running as a script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend import config


def _normalize_datetime_utc(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series, errors='coerce', utc=True)
    return s


def upsert_ground_truth_table(
    df_to_upsert: pd.DataFrame,
    db_path: str,
    table_name: str = "ground_truth_table",
    start_dt: str | None = None,
    end_dt: str | None = None,
):
    """
    Upserts preprocessed data (DataFrame) into a dedicated 15min resolution table.

    Required/expected columns (names from backend.config):
      - config.TIME_COLUMN          (e.g., 'datetime')
      - config.TARGET_COLUMN        (e.g., 'eprice_15min')
      - 'hour_extracted' (int)
      - 'minute_extracted' (int)
      - config.TEMPERATURE_COL_NAME (e.g., 'temp')
      - config.FINGRID_COL_NAME     (e.g., 'windEnergy')
      - 'temp_is_forecasted' (0/1)

    New: You can restrict the upsert to a time window by providing start_dt and end_dt.
         Example formats accepted: '2025-10-01T00:00:00Z', '2025-10-01 00:00', etc.
    """
    print(f"--- Upserting data into {table_name} ---")

    if df_to_upsert.empty:
        print("[Upsert] No data to upsert into ground_truth_table (empty DataFrame).")
        return

    # Ensure there is a datetime column with the configured name
    if config.TIME_COLUMN not in df_to_upsert.columns:
        raise ValueError(f"Input DataFrame missing '{config.TIME_COLUMN}' column.")

    # Normalize datetime to UTC tz-aware
    df_to_upsert[config.TIME_COLUMN] = _normalize_datetime_utc(df_to_upsert[config.TIME_COLUMN])

    # Optional: filter by time window
    if start_dt is not None:
        start_ts = pd.to_datetime(start_dt, errors='raise', utc=True)
        df_to_upsert = df_to_upsert[df_to_upsert[config.TIME_COLUMN] >= start_ts]
    if end_dt is not None:
        end_ts = pd.to_datetime(end_dt, errors='raise', utc=True)
        df_to_upsert = df_to_upsert[df_to_upsert[config.TIME_COLUMN] <= end_ts]

    if df_to_upsert.empty:
        print("[Upsert] No rows fall inside the requested time window. Nothing to do.")
        return

    # Compute hour/minute if missing
    if 'hour_extracted' not in df_to_upsert.columns:
        df_to_upsert['hour_extracted'] = df_to_upsert[config.TIME_COLUMN].dt.tz_convert('UTC').dt.hour
    if 'minute_extracted' not in df_to_upsert.columns:
        df_to_upsert['minute_extracted'] = df_to_upsert[config.TIME_COLUMN].dt.tz_convert('UTC').dt.minute

    # Ensure temp_is_forecasted exists (default 0)
    if 'temp_is_forecasted' not in df_to_upsert.columns:
        df_to_upsert['temp_is_forecasted'] = 0

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # Create table if needed (TEXT PK on datetime ISO string)
            expected_table_columns_schema = [
                f"{config.TIME_COLUMN} TEXT PRIMARY KEY",
                f"{config.TARGET_COLUMN} REAL",
                "hour_extracted INTEGER",
                "minute_extracted INTEGER",
                f"{config.TEMPERATURE_COL_NAME} REAL",
                f"{config.FINGRID_COL_NAME} REAL",
                "temp_is_forecasted INTEGER DEFAULT 0",
            ]
            cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(expected_table_columns_schema)})")

            # Migration: ensure temp_is_forecasted exists
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns_info = [col[1] for col in cursor.fetchall()]
            if 'temp_is_forecasted' not in columns_info:
                print(f"Adding 'temp_is_forecasted' column to {table_name}.")
                cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN temp_is_forecasted INTEGER DEFAULT 0")
                conn.commit()

            # Prepare final DataFrame with required columns
            required_cols_for_upsert = [
                config.TIME_COLUMN,
                config.TARGET_COLUMN,
                'hour_extracted',
                'minute_extracted',
                config.TEMPERATURE_COL_NAME,
                config.FINGRID_COL_NAME,
                'temp_is_forecasted',
            ]

            # Coerce renames if source columns differ (safe no-op if same)
            rename_map = {}
            if 'eprice' in df_to_upsert.columns and config.TARGET_COLUMN not in df_to_upsert.columns:
                rename_map['eprice'] = config.TARGET_COLUMN
            if 'temp' in df_to_upsert.columns and config.TEMPERATURE_COL_NAME not in df_to_upsert.columns:
                rename_map['temp'] = config.TEMPERATURE_COL_NAME
            if 'windEnergy' in df_to_upsert.columns and config.FINGRID_COL_NAME not in df_to_upsert.columns:
                rename_map['windEnergy'] = config.FINGRID_COL_NAME
            if rename_map:
                df_to_upsert = df_to_upsert.rename(columns=rename_map)

            # Ensure all required columns now exist
            missing_cols = [c for c in required_cols_for_upsert if c not in df_to_upsert.columns]
            if missing_cols:
                raise ValueError(f"Input DataFrame missing required columns: {missing_cols}")

            df_final_upsert = df_to_upsert[required_cols_for_upsert].copy()

            # Numeric coercions (keep datetime as Timestamp for formatting below)
            for col in [c for c in required_cols_for_upsert if c != config.TIME_COLUMN]:
                if col == 'temp_is_forecasted':
                    df_final_upsert[col] = pd.to_numeric(df_final_upsert[col], errors='coerce').fillna(0).astype(int)
                else:
                    df_final_upsert[col] = pd.to_numeric(df_final_upsert[col], errors='coerce')

            # Drop rows missing datetime / key features.
            # NOTE: target is allowed to be NULL (future rows).
            crit_subset = [
                config.TIME_COLUMN,
                'hour_extracted',
                'minute_extracted',
                config.TEMPERATURE_COL_NAME,
                config.FINGRID_COL_NAME,
                'temp_is_forecasted',
            ]
            df_final_upsert = df_final_upsert.dropna(subset=crit_subset)

            if df_final_upsert.empty:
                print("[Upsert] No valid records after cleaning — nothing to insert.")
                return

            # Format datetime to ISO Z string for TEXT PK
            df_final_upsert[config.TIME_COLUMN] = (
                pd.to_datetime(df_final_upsert[config.TIME_COLUMN], utc=True, errors='coerce')
                .dt.strftime('%Y-%m-%dT%H:%M:%SZ')
            )

            records = df_final_upsert.values.tolist()
            placeholders = ', '.join(['?' for _ in required_cols_for_upsert])
            columns_str = ', '.join(required_cols_for_upsert)

            insert_sql = f"INSERT OR REPLACE INTO {table_name} ({columns_str}) VALUES ({placeholders})"
            cursor.executemany(insert_sql, records)
            conn.commit()
            print(f"[Upsert] ✅ Upserted {len(records)} records into {table_name}.")

    except Exception as e:
        print(f"[Upsert] ❌ Failed to upsert ground truth data: {e}")
        import traceback
        traceback.print_exc()


def _load_csv_for_initial(csv_path: str) -> pd.DataFrame:
    df_initial = pd.read_csv(csv_path)
    # Drop legacy 'time' column if present
    if 'time' in df_initial.columns:
        df_initial = df_initial.drop(columns=['time'])
    # Ensure datetime exists
    if 'datetime' not in df_initial.columns:
        raise ValueError("CSV must include a 'datetime' column.")
    # Make sure it's parsed (UTC will be applied in upsert)
    df_initial['datetime'] = pd.to_datetime(df_initial['datetime'], errors='coerce')
    # Add hour/minute if missing
    if 'hour_extracted' not in df_initial.columns:
        df_initial['hour_extracted'] = pd.to_datetime(df_initial['datetime'], utc=True, errors='coerce').dt.hour
    if 'minute_extracted' not in df_initial.columns:
        df_initial['minute_extracted'] = pd.to_datetime(df_initial['datetime'], utc=True, errors='coerce').dt.minute
    # Ensure temp_is_forecasted exists (assume historical)
    if 'temp_is_forecasted' not in df_initial.columns:
        df_initial['temp_is_forecasted'] = 0
    return df_initial


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upsert data into ground_truth_table with optional time window.")
    parser.add_argument("--csv", type=str, default=os.path.join(os.path.dirname(__file__), '..', 'data', 'merged_oct6.csv'),
                        help="Path to source CSV (default: ../data/merged_oct5.csv)")
    parser.add_argument("--db", type=str, default=os.path.join(os.path.dirname(__file__), '..', 'data', 'forecasting_data.db'),
                        help="Path to sqlite DB (default: ../data/forecasting_data.db)")
    parser.add_argument("--table", type=str, default="ground_truth_table",
                        help="Target table name (default: ground_truth_table)")
    parser.add_argument("--start", type=str, default=None, help="Start datetime (inclusive), e.g. 2025-10-01T00:00:00Z")
    parser.add_argument("--end", type=str, default=None, help="End datetime (inclusive), e.g. 2025-11-05T17:45:00Z")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"CSV not found: {args.csv}")
        sys.exit(1)

    df_src = _load_csv_for_initial(args.csv)

    upsert_ground_truth_table(
        df_to_upsert=df_src,
        db_path=args.db,
        table_name=args.table,
        start_dt=args.start,
        end_dt=args.end,
    )

