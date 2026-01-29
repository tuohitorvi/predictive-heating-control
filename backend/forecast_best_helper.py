# backend/forecast_best_helper.py

import sqlite3

def rebuild_eprice_forecasts_best(db_path: str) -> None:
    """
    Creates/refreshes a VIEW named eprice_forecasts_best that picks exactly one
    forecast per forecasted_for_timestamp using:
      - shortest lead time wins
      - tie-breaker: latest forecast_generation_time wins
    Requires SQLite with window functions (3.25+).
    """
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()

        # Remove any legacy table/view with the same name
        cur.execute("DROP VIEW IF EXISTS eprice_forecasts_best")
        cur.execute("DROP TABLE IF EXISTS eprice_forecasts_best")

        cur.execute(
            """
            CREATE VIEW eprice_forecasts_best AS
            WITH ranked AS (
                SELECT
                    f.forecast_generation_time,
                    f.forecasted_for_timestamp,
                    f.forecasted_for_timestamp_local,
                    f.predicted_eprice,
                    f.actual_eprice,
                    f.model_version,
                    (julianday(f.forecasted_for_timestamp) - julianday(f.forecast_generation_time)) AS lead_days,
                    ROW_NUMBER() OVER (
                        PARTITION BY f.forecasted_for_timestamp
                        ORDER BY
                            (julianday(f.forecasted_for_timestamp) - julianday(f.forecast_generation_time)) ASC,
                            f.forecast_generation_time DESC
                    ) AS rn
                FROM eprice_forecasts f
                WHERE f.predicted_eprice IS NOT NULL
                  AND f.forecast_generation_time IS NOT NULL
                  AND f.forecasted_for_timestamp IS NOT NULL
                  AND julianday(f.forecast_generation_time) <= julianday(f.forecasted_for_timestamp)
            )
            SELECT
                forecast_generation_time,
                forecasted_for_timestamp,
                forecasted_for_timestamp_local,
                predicted_eprice,
                actual_eprice,
                model_version
            FROM ranked
            WHERE rn = 1;
            """
        )

        conn.commit()
