# backend/mcp_tools/elering_tool.py
import requests
from datetime import datetime, timezone
from typing import Optional
import pandas as pd


def fetch_elering_prices(startTime: str, endTime: str, forecasting_db_path: str = None, logger=print) -> dict:
    """
    Fetches 15-minute electricity prices from Elering and stores them in forecasting_data.db.
    Converts EUR/MWh -> snt/kWh (with 24% VAT) and writes both native and derived tables.
    """
    base_url = "https://dashboard.elering.ee/api/nps/price"
    query_params = {"start": startTime, "end": endTime}

    try:
        response = requests.get(base_url, params=query_params)
        response.raise_for_status()
        data = response.json()

        # Handle both 'fi' and 'ee' keys (for Finnish or Estonian markets)
        data_key = "fi" if "fi" in data.get("data", {}) else "ee"
        raw_data = data.get("data", {}).get(data_key, [])

        if not raw_data:
            logger("No price data returned from Elering.")
            return {"success": False, "error": "No price data returned from Elering."}

        converted_prices = []
        for entry in raw_data:
            if "timestamp" in entry and "price" in entry:
                dt_object = datetime.fromtimestamp(entry["timestamp"], tz=timezone.utc)
                iso_timestamp = dt_object.strftime("%Y-%m-%dT%H:%M:%SZ")
                converted_prices.append({
                    "datetime": iso_timestamp,
                    "price_eur_per_mwh": entry["price"]
                })

        df = pd.DataFrame(converted_prices)
        if df.empty:
            logger("No valid price entries after conversion.")
            return {"success": False, "error": "No valid price entries after conversion."}

        return {"success": True, "data": df.to_dict(orient="records")}


    except Exception as e:
        logger(f"Error fetching Elering data: {str(e)}")
        return {"success": False, "error": str(e)}
