# backend/mcp_tools/fmi_tool.py
import requests
from datetime import datetime, timezone, timedelta
import traceback 
import xml.etree.ElementTree as ET
from utils.parse_weather_xml import parse_weather_xml


def fetch_temp_data(place: str = "Hämeenlinna", startTime: str = "", endTime: str = "", parameters: str = "t2m", logger=print) -> dict:
    """Fetch weather data from FMI for a given place and parameters.
    Returns a structured JSON with observations.
    startTime and endTime should be in ISO 8601 format (e.g., YYYY-MM-DDTHH:MM:SSZ).
    Parameters is a comma-separated string (e.g., 't2m,ws_10min').
    """
    logger(f"🌦️ Fetching FMI data for: {place} (Parameters: {parameters}, Range: '{startTime}' to '{endTime}')")

    original_start_time, original_end_time = startTime, endTime

    if not startTime and not endTime:
        now_utc = datetime.now(timezone.utc)
        default_end_time_obj = now_utc
        default_start_time_obj = now_utc - timedelta(hours=1)
        startTime = default_start_time_obj.strftime('%Y-%m-%dT%H:%M:%SZ')
        endTime = default_end_time_obj.strftime('%Y-%m-%dT%H:%M:%SZ')
        logger(f"🕰️ Using default time range: {startTime} to {endTime}")
    elif not startTime and endTime:
        try:
            end_time_obj = datetime.fromisoformat(endTime.replace('Z', '+00:00'))
            startTime = (end_time_obj - timedelta(hours=1)).strftime('%Y-%m-%dT%H:%M:%SZ')
        except ValueError:
            return {"error": f"Invalid endTime format: {endTime}. Expected ISO 8601."}
    elif startTime and not endTime:
        now_utc = datetime.now(timezone.utc)
        endTime = now_utc.strftime('%Y-%m-%dT%H:%M:%SZ')

    base_url = "http://opendata.fmi.fi/wfs/fin"
    api_params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "getFeature",
        "storedquery_id": "fmi::observations::weather::timevaluepair",
        "place": place,
        "parameters": parameters 
    }

    if startTime: api_params["starttime"] = startTime
    if endTime: api_params["endtime"] = endTime

    try:
        response = requests.get(base_url, params=api_params)
        logger(f"--- FMI Request URL: {response.url} ---")
        logger(f"--- FMI Response Status Code: {response.status_code} ---")
        response.raise_for_status()

        raw_xml_text = response.text
        #logger(f"--- Raw FMI XML Response Snippet ---\n{raw_xml_text[:500]}\n------------------------------------") # Uncomment for debugging XML

        # Use the parsing function
        json_observations = parse_weather_xml(raw_xml_text)

        # Check if parsing returned an error structure 
        if isinstance(json_observations, list) and len(json_observations) > 0 and isinstance(json_observations[0], dict) and "error" in json_observations[0]:
            return {
                "error": "Failed to parse FMI XML using custom parser.",
                "details": json_observations[0].get("details", "No details from parser."),
                "place": place,
                "parameters_requested": parameters
            }


        return {
            "place": place,
            "parameters_requested": parameters,
            "requested_startTime": original_start_time or "N/A (defaulted)",
            "requested_endTime": original_end_time or "N/A (defaulted)",
            "actual_startTime_used": startTime if (original_start_time or original_end_time) else "Defaulted",
            "actual_endTime_used": endTime if (original_start_time or original_end_time) else "Defaulted",
            "data_format": "json",
            "observations": json_observations
        }
    except requests.exceptions.HTTPError as http_err:
        error_detail = response.text
        logger(f"FMI API HTTP error occurred: {str(http_err)}. Details: {error_detail}")
        if hasattr(response, 'text') and "<ExceptionText>" in response.text: # Check if response.text exists
             try:
                start = response.text.find("<ExceptionText>") + len("<ExceptionText>")
                end = response.text.find("</ExceptionText>")
                error_detail = response.text[start:end].strip()
             except Exception: pass
        else:
            error_detail = str(http_err)
        return {"error": f"FMI API request failed: {response.status_code}", "details": error_detail}
    except requests.exceptions.RequestException as req_err:
        logger(f"FMI API Request exception occurred: {str(req_err)}")
        return {"error": f"FMI API request failed due to a network issue: {str(req_err)}"}
    except ET.ParseError as xml_err: # Catch ElementTree specific parsing errors
        #Ensure logger call is with a single string
        logger(f"FMI XML parsing error (ElementTree): {str(xml_err)}. Raw XML snippet: {raw_xml_text[:500] if 'raw_xml_text' in locals() else 'XML not available'}")
        return {"error": "Failed to parse XML response from FMI (ElementTree).", "details": str(xml_err), "raw_xml_snippet": raw_xml_text[:500] if 'raw_xml_text' in locals() else "XML not available"}
    except NotImplementedError as nie: # Catch if parser couldn't be imported
        return {"error": str(nie)}
    except Exception as e:
        logger(f"An unexpected error occurred in fetch_temp_data: {str(e)}")
        traceback.print_exc() # traceback.print_exc() prints to stderr, not logger
        return {"error": f"An unexpected error occurred: {str(e)}"}
    
def fetch_weather_forecast(place: str = "Hämeenlinna", logger=print) -> dict: 
    """Fetch the latest weather forecast from FMI using the HARMONIE model."""
    logger(f"🌦️ Fetching FMI FORECAST for: {place}")
    
    # 1. Use the HARMONIE model stored query.
    storedquery_id = "fmi::forecast::harmonie::surface::point::timevaluepair"

    # 2. This specific query uses "Temperature" (capital T) as the parameter name.
    parameters_to_request = "Temperature"
    
    
    base_url = "http://opendata.fmi.fi/wfs/fin"
    api_params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "getFeature",
        "storedquery_id": storedquery_id,
        "place": place,
        "parameters": parameters_to_request
    }
    
    try:
        response = requests.get(base_url, params=api_params)
        response.raise_for_status()

        # The parser will find the 't2m' key 
        json_observations = parse_weather_xml(response.text)
        
        
        # Renaming Temperature to t2m -> PreprocessorAgent can recognize it and rename it to "temp".
        cleaned_observations = []
        for obs in json_observations:
            if "Temperature" in obs:
                obs["t2m"] = obs.pop("Temperature") # Rename the key
            cleaned_observations.append(obs)
        
        return {"observations": cleaned_observations}
    except Exception as e:
        logger(f"An unexpected error occurred in fetch_weather_forecast: {str(e)}")
        return {"error": str(e), "observations": []}