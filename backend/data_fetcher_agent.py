# backend/data_fetcher_agent.py
from .mcp_tools.fingrid_tool import fetch_fingrid_data
from .mcp_tools.fmi_tool import fetch_temp_data, fetch_weather_forecast
from .mcp_tools.elering_tool import fetch_elering_prices
from utils.parse_weather_xml import parse_weather_xml
from datetime import datetime, timedelta, timezone
import pandas as pd
from typing import Optional


class DataFetcherAgent:
    def __init__(self, forecasting_data_db_path: str = None, logger = None):
        self.logger = logger if logger else print # Use provided logger or default to print
        self.logger("DataFetcherAgent initialized.")
        self.forecasting_data_db_path = forecasting_data_db_path

    
    def run(self, location: str = "Hämeenlinna", startTime: Optional[str] = None, endTime: Optional[str] = None) -> dict:
            self.logger("DataFetcherAgent: Fetching data for all sources...")
            datasetId = 245
            now_utc = datetime.now(timezone.utc)

            if startTime and endTime:
                # Use provided window for ALL sources <--> consistent with orchestrator lookback
                start_iso_wide = startTime
                end_iso_wide   = endTime
                start_iso_hist = startTime
                end_iso_hist   = endTime
            else:
                # Fallback to previous default behavior
                start_iso_wide = (now_utc - timedelta(days=1)).replace(hour=0, minute=0, second=0).strftime("%Y-%m-%dT%H:%M:%SZ")
                end_iso_wide   = (now_utc + timedelta(days=1)).replace(hour=23, minute=59, second=59).strftime("%Y-%m-%dT%H:%M:%SZ")
                start_iso_hist = (now_utc - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
                end_iso_hist   = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

            fetched_data = {}

            # Fingrid
            try:
                fingrid_list = fetch_fingrid_data(
                    datasetId=datasetId,
                    startTime=start_iso_wide,
                    endTime=end_iso_wide,
                    logger=self.logger
                )
                fetched_data["fingrid"] = fingrid_list
                if isinstance(fingrid_list, list) and fingrid_list:
                    self.logger(f"📦 Fingrid fetch returned {len(fingrid_list)} records.")
                    self.logger(f"📦 Raw Fingrid data (first 3 rows): {fingrid_list[:3]}")
            except Exception as e:
                self.logger(f"Error fetching Fingrid data: {str(e)}")
                fetched_data["fingrid"] = []

            # FMI (observations)

            '''
            try:
                fmi_dict = fetch_temp_data(
                    place=location,
                    startTime=start_iso_hist,
                    endTime=end_iso_hist,
                    parameters="t2m",
                    logger=self.logger
                )
                fetched_data["fmi"] = fmi_dict
            except Exception as e:
                self.logger(f"Error fetching FMI data: {str(e)}")
                fetched_data["fmi"] = {"observations": []}
            '''
            try:
                # FMI obs has a max window limit (typically 168 hours = 7 days).
                # Cap obs request to last 7 days ending at "now" (or end_iso_hist).
                end_hist_dt = pd.to_datetime(end_iso_hist, utc=True)
                start_hist_dt = pd.to_datetime(start_iso_hist, utc=True)

                max_hours = 168
                capped_start_dt = max(start_hist_dt, end_hist_dt - pd.Timedelta(hours=max_hours))
                start_iso_hist_capped = capped_start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                end_iso_hist_capped = end_hist_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

                fmi_dict = fetch_temp_data(
                    place=location,
                    startTime=start_iso_hist_capped,
                    endTime=end_iso_hist_capped,
                    parameters="t2m",
                    logger=self.logger
                )
                fetched_data["fmi"] = fmi_dict
            except Exception as e:
                self.logger(f"Error fetching FMI data: {str(e)}")
                fetched_data["fmi"] = {"observations": []}
            

            # Elering
            try:
                elering_dict = fetch_elering_prices(
                    startTime=start_iso_wide,
                    endTime=end_iso_wide,
                    forecasting_db_path="data/forecasting_data.db",
                    logger=self.logger
                )
                fetched_data["elering"] = elering_dict
            except Exception as e:
                self.logger(f"Error fetching Elering data: {str(e)}")
                fetched_data["elering"] = {"success": False, "error": str(e)}

            return fetched_data


    def fetch_fmi_forecast(self, location: str = "Hämeenlinna") -> list:
        self.logger(f"🌦️ Fetching FMI FORECAST for: {location}") 
        try:
            forecast_result = fetch_weather_forecast(place=location, logger=self.logger)
            return forecast_result.get("observations", [])
        except Exception as e:
            self.logger(f"Error during FMI forecast fetch: {str(e)}")
            return []
    
    



