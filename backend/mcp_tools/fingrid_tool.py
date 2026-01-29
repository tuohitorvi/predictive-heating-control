#backend/mcp_tools/fingrid_tool.py
import os
import requests
import time
import random
from dotenv import load_dotenv
from typing import List, Dict, Any

# Load environment variables from .env file if it exists
load_dotenv()
FINGRID_API_KEY = os.getenv("FINGRID_API_KEY")
if not FINGRID_API_KEY:
    raise ValueError("API key FINGRID_API_KEY is missing! Please set it in your environment or .env file.")

# module-level cache (last successful response per (datasetId, start, end))
_FG_CACHE: Dict[tuple, List[Dict[str, Any]]] = {}

MAX_RETRIES = 4
BACKOFFS = [1, 2, 4, 8]  # seconds
TIMEOUT = 15  # seconds

def fetch_fingrid_data(datasetId: int, startTime: str, endTime: str, logger=print) -> List[Dict[str, Any]]:
    """
    Fetch Fingrid time-series data with datasetId, startTime, and endTime.
    Returns a list (same as your original) or [] on failure.
    Adds retry with exponential backoff and a tiny cache fallback.
    """
    logger(f"📡 Fetching Fingrid data for dataset {datasetId} from {startTime} to {endTime}...")
    url = "https://data.fingrid.fi/api/datasets/{}/data".format(datasetId)
    headers = {"x-api-key": FINGRID_API_KEY}
    params = {
        "datasetId": datasetId,
        "startTime": startTime,
        "endTime": endTime,
        "format": "json",
        "oneRowPerTimePeriod": "true",
        "page": 1,
        "pageSize": 20000,
    }

    cache_key = (datasetId, startTime, endTime)
    last_err_text = None

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=TIMEOUT)
            # Explicitly raise for non-2xx
            resp.raise_for_status()

            # Fingrid sometimes returns {"data":[...]} but handle bare lists just in case
            payload = resp.json()
            data = payload.get("data", payload if isinstance(payload, list) else [])

            if not isinstance(data, list):
                logger("⚠️ Fingrid: response JSON doesn't look like a list or contain 'data'; treating as empty.")
                data = []

            logger(f"📦 Fingrid fetch returned {len(data)} records.")
            _FG_CACHE[cache_key] = data  # update cache only on success
            return data

        except requests.exceptions.HTTPError as http_err:
            # Save body for the last attempt log
            last_err_text = getattr(resp, "text", "")
            status = getattr(resp, "status_code", "N/A")
            logger(f"❌ Fingrid HTTP {status} on attempt {attempt+1}/{MAX_RETRIES}: {http_err}")
        except requests.exceptions.RequestException as req_err:
            logger(f"🌐 Fingrid network error on attempt {attempt+1}/{MAX_RETRIES}: {req_err}")
        except Exception as e:
            logger(f"Unexpected Fingrid error on attempt {attempt+1}/{MAX_RETRIES}: {e}")

        # Backoff before next attempt (except after last)
        if attempt < MAX_RETRIES - 1:
            time.sleep(BACKOFFS[attempt] + random.random())

    # cache
    cached = _FG_CACHE.get(cache_key)
    if cached:
        logger("⚠️ Fingrid API failed; returning cached data for this window.")
        return cached

    # No cache
    if last_err_text:
        logger(f"❌ FINGRID API HTTP ERROR (final): Response Body: {last_err_text}")
    else:
        logger("❌ Fingrid: giving up after retries; returning empty list.")
    return []


