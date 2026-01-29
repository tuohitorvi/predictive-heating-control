# run_all.py
import threading
import time
import os
import numpy as np
import requests
import uvicorn
import gradio as gr
import pandas as pd
import sqlite3
from typing import Dict, Any
from fastapi import FastAPI, BackgroundTasks, Body
from fastapi.responses import FileResponse
import plotly.graph_objects as go
import glob
from pathlib import Path
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt 
from backend.orchestrating_agent import OrchestratingAgent 
import backend.config as config
from backend.analytics_agent import AnalyticsAgent


orchestrator = OrchestratingAgent() 
app = FastAPI() 


DARK_CSS = """
:root { color-scheme: dark; }

/* page + container */
html, body, .gradio-container {
  background: #0e1117 !important;
  color: #e6edf3 !important;
}

/* common panels/boxes */
.gradio-container .block,
.gradio-container .form,
.gradio-container .panel,
.gradio-container .group {
  background: #111827 !important;
}

/* text elements */
.gradio-container label,
.gr-markdown,
.gr-markdown * {
  color: #e6edf3 !important;
}

/* stronger headings + link contrast only in markdown */
.gradio-container .gr-markdown h1,
.gradio-container .gr-markdown h2,
.gradio-container .gr-markdown h3,
.gradio-container .gr-markdown h4 {
  color: #f0f3f6 !important;
}
.gradio-container .gr-markdown a {
  color: #7aa2f7 !important;
  text-decoration: underline;
}

/* inputs */
.gradio-container input,
.gradio-container textarea,
.gradio-container select {
  background: #0b1623 !important;
  color: #e6edf3 !important;
  border-color: #1f2937 !important;
}

/* buttons (let theme handle primary colors, just nudge borders) */
.gradio-container button {
  border-color: #1f2937 !important;
}

/* ---- Plotly readability tweaks ---- */
/* axis tick labels */
.gradio-container .js-plotly-plot .xtick text,
.gradio-container .js-plotly-plot .ytick text {
  fill: #e6edf3 !important;
}

/* axis titles */
.gradio-container .js-plotly-plot .g-xtitle text,
.gradio-container .js-plotly-plot .g-ytitle text {
  fill: #e6edf3 !important;
  font-weight: 600;
}

/* legend text */
.gradio-container .js-plotly-plot .legend text {
  fill: #e6edf3 !important;
}

/* gridlines & zero lines (subtle but visible) */
.gradio-container .js-plotly-plot .xgrid,
.gradio-container .js-plotly-plot .ygrid {
  stroke: #3b4252 !important;
  stroke-opacity: 0.6 !important;
}
.gradio-container .js-plotly-plot .xzeroline,
.gradio-container .js-plotly-plot .yzeroline {
  stroke: #9aa4b2 !important;
  stroke-width: 1.2px !important;
}

/* hover labels: keep text dark on light tooltip */
.gradio-container .js-plotly-plot .hovertext text {
  fill: #0d1117 !important;
}

/* Style the "Predictive Heating System Control Panel" title */
#panel_title h2,
#panel_title .prose h2 {
  color: #ffffff !important;
}
/* "System status" title */
#system_status h3,
#system_status .prose h3 {
    color: #ffffff !important;
}
/* "Eprice Upper Limit" title */
#eprice_limit h3,
#eprice_limit .prose h3 {
    color: #ffffff !important;
}
/* "Actual eprice" title */
#actual_eprice h3,
#actual_eprice .prose h3 {
    color: #ffffff !important;
}
/* "Forecast Interpretation" title */
#interpretation h3,
#interpretation .prose h3 {
    color: #ffffff !important;
}
/* "Price Forecast" title */
#price_forecast h3,
#price_forecast .prose h3 {
    color: #ffffff !important;
}
/* "Operational Log" title */
#operational_log h3,
#operational_log .prose h3 {
    color: #ffffff !important;
}
/* Improve visibility of Plotly axes, ticks, labels, and grid */
#forecast_plot .xtick text,
#forecast_plot .ytick text {
    fill: #ffffff !important;
    font-size: 14px !important;
}
#forecast_plot .xaxes-title text,
#forecast_plot .yaxes-title text,
#forecast_plot .xtitle text,
#forecast_plot .ytitle text {
    fill: #ffffff !important;
    font-weight: bold !important;
}

#forecast_plot .gridline {
    stroke: #555555 !important;
}

/* axis line */
#forecast_plot .xaxis path,
#forecast_plot .yaxis path {
    stroke: #ffffff !important;
}

#cycle-status-box textarea, 
#cycle-status-box .wrap, 
#cycle-status-box input {
    color: #ffffff !important;            /* force white text */
    background-color: #1e293b !important; /* dark slate background */
    border: 1px solid #4b5563 !important; /* visible border */
}

/* Cycle Trigger Status: force readable text regardless of rendering mode */
#cycle-status-box textarea,
#cycle-status-box [data-testid="textbox"],
#cycle-status-box .wrap,
#cycle-status-box input,
#cycle-status-box div {
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important; /* Safari */
  background-color: #1e293b !important;
  border: 1px solid #4b5563 !important;
}

/* Make the label readable too */
#cycle-status-box label {
  color: #e6edf3 !important;
}

/* caption styling for the plot */
/* Make the forecast caption clearly visible */
#forecast_caption,
#forecast_caption * ,
#forecast_caption .prose,
#forecast_caption .prose * {
  color: #e6edf3 !important;   /* bright white */
  opacity: 1 !important;        /* defeat dimming */
}

/* "Model diagnostics" title */
#model_diagnostics h3,
#model_diagnostics .prose h3 {
    color: #ffffff !important;
}

/* "Price spike diagnostics" title */
#price_spike_detector h3,
#price_spike_detector .prose h3 {
    color: #ffffff !important;
}

/* "Actuator Control & Spike Guard" title */
#actuator_ctrl_with_spike_guard h2,
#actuator_ctrl_with_spike_guard .prose h2 {
    color: #ffffff !important;
}

/* ---- Make checkboxes clearly visible ---- */
.gradio-container input[type="checkbox"] {
  accent-color: #7aa2f7 !important;   /* bright blue tick/fill */
  width: 18px;
  height: 18px;
  cursor: pointer;
  border: 2px solid #e5e7eb !important;
  border-radius: 3px;
}

/* ---- Make checkboxes clearly visible ---- */
.gradio-container input[type="checkbox"] {
  accent-color: #7aa2f7 !important;   /* bright blue tick/fill */
  width: 18px;
  height: 18px;
  cursor: pointer;
  border: 2px solid #e5e7eb !important;
  border-radius: 3px;
}

/* "AI-Assisted Heating Analytics" title */
#ai-assisted-heating-analytics h2,
#ai-assisted-heating-analytics .prose h2 {
    color: #ffffff !important;
}

/* "Decision thresholds (recommended)" title */
#decision_thresholds h2,
#decision_thresholds .prose h2 {
    color: #ffffff !important;
}
"""



@app.get("/sensors")
def get_all_sensor_data():
    """Returns the most recent reading for all sensors."""
    # The data lives inside the sensor_monitor instance
    return orchestrator.sensor_monitor.latest_sensor_data

@app.get("/sensors/{sensor_key}")
def get_sensor_by_key(sensor_key: str):
    """Returns the most recent reading for a specific sensor."""
    # Get the data from the correct location
    sensor = orchestrator.sensor_monitor.latest_sensor_data.get(sensor_key)
    if sensor:
        return sensor
    else:
        return {"error": "Sensor not found"}
    
@app.get("/alerts")
def get_alerts():
    """Returns the last 50 alerts from the database."""
    # This endpoint talks directly to the DB
    with sqlite3.connect(config.SENSOR_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM alerts ORDER BY timestamp DESC LIMIT 50").fetchall()
        return [dict(row) for row in rows]
    
@app.get("/sensor_history/{sensor_key}")
def get_sensor_history(sensor_key: str, limit: int = 100):
    """Returns the historical readings for a specific sensor from the database."""
    # This endpoint also talks directly to the DB.
    with sqlite3.connect(config.SENSOR_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT timestamp, value FROM sensor_readings WHERE sensor_key = ? ORDER BY timestamp DESC LIMIT ?",
            (sensor_key, limit)
        ).fetchall()
    # Reverse the list so the data is chronological for plotting
    return list(reversed([dict(row) for row in rows]))

# ORCHESTRATOR CONTROL ENDPOINTS 
@app.get("/orchestrator/status")
def get_orchestrator_status():
    """Returns the current state of the orchestrator for UI updates."""
    # The system state lives inside the orchestrator instance
    state = orchestrator.system_state.copy()
    # Convert any pandas DataFrames to a JSON-serializable format before returning
    for key in ["preprocessed_df", "forecast_df", "current_sensor_data"]:
        if isinstance(state.get(key), pd.DataFrame):
            df = state[key].replace({np.nan: None})  # Replace NaN with None
            state[key] = df.to_dict(orient='records')

    # Include actual eprice (snt/kWh) from actuator controller
    try:
        actual_eprice = orchestrator.actuator_controller.get_current_eprice_now()
    except Exception:
        actual_eprice = None
    state["actual_eprice_snt_per_kwh"] = actual_eprice
    
    return state
        
@app.post("/orchestrator/run_cycle")
def trigger_forecast_cycle():
    """Triggers a full forecast-to-actuation cycle in the background."""
    print("API: Received request to run forecasting cycle.")
    # Using a simple threading (easier than FastAPI's BackgroundTasks for long-running jobs that don't need to be tied to the request-response cycle).
    threading.Thread(target=orchestrator.run_forecasting_cycle, daemon=True).start()
    return {"status": "ok", "message": "Forecasting cycle initiated in the background."}

# Endpoint to update Eprice Upper Limit
@app.post("/orchestrator/set_eprice_upper_limit")
def set_eprice_upper_limit(new_limit: float = Body(..., embed=True, alias="eprice_upper_limit")):
    """Sets the Eprice upper limit for actuator control."""
    orchestrator.system_state["control_parameters"]["EPRICE_UPPER_LIMIT"] = new_limit
    print(f"API: Eprice upper limit set to {new_limit} snt/kWh.")
    return {"status": "ok", "message": f"Eprice upper limit set to {new_limit}."}

# Endpoint to trigger Actuator Control Cycle
@app.post("/orchestrator/run_actuator_control_cycle")
def trigger_actuator_control_cycle():
    """Triggers an actuator control cycle in the background."""
    print("API: Received request to run actuator control cycle.")
    threading.Thread(target=orchestrator.run_actuator_control_cycle, daemon=True).start()
    return {"status": "ok", "message": "Actuator control cycle initiated in the background."}

@app.post("/orchestrator/run_finetune")
def trigger_finetune_cycle():
    """Triggers a fine-tuning cycle in the background."""
    print("API: Received request to run fine-tuning cycle.")
    threading.Thread(target=orchestrator.run_finetuning_cycle, daemon=True).start()
    return {"status": "ok", "message": "Fine-tuning cycle initiated in the background."}

@app.post("/orchestrator/run_retraining")
# Explicitly expect a dictionary in the request body
def trigger_retraining_cycle(data: Dict[str, bool] = Body(...)): # '...' makes it required
    train_from_scratch = data.get("train_from_scratch", False) # Extract the boolean, with a fallback default
    """Triggers full model re-training in the background."""
    print(f"API: Received request to run re-training cycle (train_from_scratch={train_from_scratch}).")
    print(f"DEBUG: FastAPI received raw body data: {data}, extracted train_from_scratch as {train_from_scratch}") # Added debug
    threading.Thread(target=orchestrator.run_retraining_cycle, args=(train_from_scratch,), daemon=True).start()
    return {"status": "ok", "message": "Re-training cycle initiated in the background."}


@app.post("/orchestrator/set_spike_guard")
def set_spike_guard(enabled: bool = Body(..., embed=True)):
    """
    Enable or disable automatic spike guard.
    """
    orchestrator.actuator_controller.set_spike_guard_enabled(enabled)
    return {"status": "ok", "spike_guard_enabled": enabled}


@app.post("/orchestrator/set_spike_override")
def set_spike_override(override: bool = Body(..., embed=True)):
    """
    Manual override: if True, ignore spike guard and allow heating
    even during spikes.
    """
    orchestrator.actuator_controller.set_spike_manual_override(override)
    return {"status": "ok", "spike_manual_override": override}

@app.post("/orchestrator/set_control_parameters")
def set_control_parameters(payload: Dict[str, Any] = Body(...)):
    """
    Expected JSON:
      {"control_parameters": {"KEY": value, ...}}
    Merges into orchestrator.system_state["control_parameters"].
    """
    updates = payload.get("control_parameters", {})
    if not isinstance(updates, dict):
        return {"status": "error", "message": "control_parameters must be a dict"}

    orchestrator.system_state.setdefault("control_parameters", {})
    orchestrator.system_state["control_parameters"].update(updates)

    return {"status": "ok", "updated": updates}


def launch_gradio():

    # Safe analytics agent init 
    try:
        analytics_agent = AnalyticsAgent(logger=print)
    except Exception as e:
        print(f"[AnalyticsAgent] WARNING: failed to initialize analytics agent: {e}")
        analytics_agent = None


    # Actions
    def run_orchestrator_cycle():
        requests.post("http://localhost:5000/orchestrator/run_cycle")
        return "Forecast cycle initiated. Check logs for progress."

    def run_actuator_control_cycle():
        requests.post("http://localhost:5000/orchestrator/run_actuator_control_cycle")
        return "Actuator control cycle initiated. Check logs for progress."

    def set_eprice_limit(new_limit: float):
        requests.post("http://localhost:5000/orchestrator/set_eprice_upper_limit",
                      json={"eprice_upper_limit": new_limit})
        return f"Eprice upper limit set to {new_limit} snt/kWh."

    def run_orchestrator_finetune():
        requests.post("http://localhost:5000/orchestrator/run_finetune")
        return "Fine-tuning initiated. Check logs for progress."

    def run_orchestrator_retraining(train_from_scratch_flag: bool):
        requests.post("http://localhost:5000/orchestrator/run_retraining",
                      json={"train_from_scratch": train_from_scratch_flag})
        return "Re-training initiated. Check logs for progress."

    # Polling (Returns actual price too)
    def update_orchestrator_display():
        try:
            resp = requests.get("http://localhost:5000/orchestrator/status")
            resp.raise_for_status()
            status = resp.json()

            log_output = "\n".join(status.get("operational_log", []))
            status_summary = (
                f"Forecast Cycle: {status.get('last_forecast_cycle_status', 'N/A')}\n"
                f"Actuator Cycle: {status.get('last_actuator_cycle_status', 'N/A')}\n"
                f"Fine-Tune Status: {status.get('last_finetune_cycle_status', 'N/A')}\n"
                f"Retraining Status: {status.get('last_retraining_cycle_status', 'N/A')}\n"
                f"Active Model: {status.get('current_model_version_tag', 'N/A')}\n"
                f"Eprice Upper Limit: {status.get('control_parameters', {}).get('EPRICE_UPPER_LIMIT', 'N/A')} snt/kWh\n"
                f"Control Signals: {status.get('control_signals', {})}"
            )

            fig = go.Figure()

            df_best = load_best_forecast_for_plot(n_points=96)  # last 24h of 15-min points
            if df_best is not None and not df_best.empty:
                fig.add_trace(go.Scatter(
                    x=df_best["datetime"],
                    y=df_best["eprice_15min"],
                    mode="lines+markers",
                    name="Best Forecast (shortest lead time)",
                ))
            else:
                # fallback to in-memory forecast_df if best table has no data yet
                forecast_data = status.get("forecast_df")
                if forecast_data:
                    df = pd.DataFrame(forecast_data)
                    if not df.empty and "datetime" in df.columns and "eprice_15min" in df.columns:
                        df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
                        df = df.dropna(subset=["datetime"])
                        fig.add_trace(go.Scatter(
                            x=df["datetime"], y=df["eprice_15min"], mode="lines+markers", name="Forecast (latest run)"
                        ))

            # Dark theme axis/title/tick styling
            fig.update_layout(
                title=dict(text=""),
                margin=dict(t=0),
                showlegend=False,
                paper_bgcolor="#0e1117",
                plot_bgcolor="#0e1117",
                font=dict(color="#e6edf3", size=14),
                xaxis=dict(
                    title="Time",
                    tickfont=dict(color="#ffffff", size=13),
                    gridcolor="#333333",
                    zerolinecolor="#555555",
                    linecolor="#ffffff",
                    showline=True,
                ),
                yaxis=dict(
                    title="snt/kWh",
                    tickfont=dict(color="#ffffff", size=13),
                    gridcolor="#333333",
                    zerolinecolor="#555555",
                    linecolor="#ffffff",
                    showline=True,
                ),
                xaxis_title_font=dict(color="#ffffff", size=16),
                yaxis_title_font=dict(color="#ffffff", size=16),
                hoverlabel=dict(font_color="#ffffff", bgcolor="#111827", bordercolor="#444444"),
            )

            forecast_interpretation = status.get('forecast_interpretation', 'No interpretation available.')
            actual_eprice = status.get("actual_eprice_snt_per_kwh", None)
            actual_eprice_text = "N/A" if actual_eprice is None else f"{float(actual_eprice):.3f}"

            return log_output, status_summary, fig, forecast_interpretation, actual_eprice_text

        except requests.RequestException as e:
            err = f"Failed to fetch orchestrator status: {e}"
            empty_fig = go.Figure().update_layout(
                paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                font=dict(color="#e6edf3")
            )
            return err, err, empty_fig, "", "N/A"

            
    
    def load_latest_diagnostics_plot():
        """
        Returns:
        (forecast_path, learning_curve_path, stepwise_mae_path, status_text)
        """
        plots_dir = Path("data") / "plots"
        if not plots_dir.exists():
            return None, None, None, "Diagnostics folder 'data/plots' does not exist."

        # A. Forecast vs Actual (Look for the NEW multi-output file first)
        # The agent saves it as: diagnostic_forecast_VERSION.png
        forecast_pngs = list(plots_dir.glob("*diagnostic_forecast*.png"))
    
        # Fallback to old name if new one doesn't exist yet, but prioritize new
        if not forecast_pngs:
            forecast_pngs = list(plots_dir.glob("*forecast*.png"))
        
        latest_forecast = max(forecast_pngs, key=os.path.getmtime) if forecast_pngs else None

        # B. Learning Curves
        lc_pngs = list(plots_dir.glob("*learning_curves*.png"))
        latest_lc = max(lc_pngs, key=os.path.getmtime) if lc_pngs else None

        # C. Stepwise MAE
        mae_pngs = list(plots_dir.glob("*stepwise_mae*.png"))
        latest_mae = max(mae_pngs, key=os.path.getmtime) if mae_pngs else None

        parts = []
        if latest_forecast: parts.append("Forecast Plot Found")
        if latest_lc: parts.append("Learning Curves Found")
        if latest_mae: parts.append("MAE Plot Found")
    
        status = " | ".join(parts) if parts else "No plots found."

        return (
            str(latest_forecast) if latest_forecast else None,
            str(latest_lc) if latest_lc else None,
            str(latest_mae) if latest_mae else None,
            status
        )
 
    
    def plot_price_spikes(
        z_threshold: float = 1.0,
        pct_threshold: float = 0.006,
        abs_min_price: float = 5.0,
        window_days: float = 3.0,
    ):
        """
        Load eprice_15min history from ground_truth_table and plot:
        - full price series
        - spikes highlighted based on robust stats.

        Parameters are tuned for Finnish 15-min prices by default, but 
        can be adjusted from the UI.

        Returns:
        - Plotly figure
        - Info string with spike count + parameters
        """

        try:
            with sqlite3.connect(config.FORECASTING_DB_PATH) as conn:
                df = pd.read_sql_query(
                    f"""
                    SELECT {config.TIME_COLUMN} AS datetime, {config.TARGET_COLUMN} AS eprice_15min
                    FROM {config.GROUND_TRUTH_TABLE}
                    WHERE {config.TARGET_COLUMN} IS NOT NULL
                    ORDER BY {config.TIME_COLUMN} ASC
                    """,
                    conn,
                    parse_dates=["datetime"],
                )
        except Exception as e:
            msg = f"[SpikeDetector] Error loading data: {e}"
            print(msg)
            fig = go.Figure()
            fig.update_layout(
                title="Error loading data for spike detection",
                paper_bgcolor="#0e1117",
                plot_bgcolor="#0e1117",
                font=dict(color="#e6edf3"),
            )
            return fig, msg

        if df.empty:
            msg = "[SpikeDetector] No eprice_15min data available."
            print(msg)
            fig = go.Figure()
            fig.update_layout(
                title="No eprice_15min data available for spike detection",
                paper_bgcolor="#0e1117",
                plot_bgcolor="#0e1117",
                font=dict(color="#e6edf3"),
            )
            return fig, msg

        # Ensure timezone-aware UTC
        if df["datetime"].dt.tz is None:
            df["datetime"] = df["datetime"].dt.tz_localize("UTC")
        else:
            df["datetime"] = df["datetime"].dt.tz_convert("UTC")

        # Use datetime as index for time-based rolling
        df = df.set_index("datetime").sort_index()

        # Limit to recent window for plotting & detection
        lookback_days = 21  # 3 weeks
        end_ts = df.index.max()
        start_ts = end_ts - pd.Timedelta(days=lookback_days)
        df = df.loc[start_ts:end_ts]

        price = df["eprice_15min"].astype(float)

        # Rolling window in days
        window_days = max(1.0, float(window_days))
        window_str = f"{max(1, int(window_days))}D"

        # Rolling median and MAD (robust)
        roll_median = price.rolling(window=window_str, min_periods=40).median()
        roll_mad = (price - roll_median).abs().rolling(window=window_str, min_periods=40).median()

        # Robust scale estimate (MAD -> sigma approx)
        mad_scaled = 1.4826 * roll_mad

        # Floor the scale to avoid huge z during ultra-flat regimes (tune 0.2–0.5)
        mad_scaled = mad_scaled.clip(lower=0.2)

        # One-sided (upward) deviation z-score
        up_dev = (price - roll_median)
        z_up = up_dev / mad_scaled

        # Percentage change vs previous 15-min (use a meaningful threshold for spikes)
        pct_change = price.pct_change()

        '''
        spike_mask = (
            (price >= abs_min_price) &
            (z_up >= z_threshold) &
            (pct_change >= pct_threshold)
        ).fillna(False)
        '''

        # --- Strategic spike detection (Finland-tuned) ---

        HIGH_PRICE = 20.0        # start caring seriously
        EXTREME_PRICE = 30.0     # always a spike
        ABS_JUMP = 3.0           # snt in 15 min
        RAMP_1H = 8.0             # snt over 1 hour
        ramp_window = 4          # 4 * 15min = 1h

        delta = price.diff()
        ramp = price.diff(ramp_window)

        mode_outlier = (price >= HIGH_PRICE) & (z_up >= z_threshold)

        mode_shock = (
            (price >= HIGH_PRICE) &
            (
                (pct_change >= pct_threshold) |
                (delta >= ABS_JUMP) |
                (ramp >= RAMP_1H)
            )
        )

        mode_extreme = (price >= EXTREME_PRICE)

        spike_mask = (mode_outlier | mode_shock | mode_extreme).fillna(False)

        

        df_out = pd.DataFrame({
            "eprice_15min": price,
            "z_score_up": z_up,
            "pct_change": pct_change,
            "delta": delta,
            "ramp_1h": ramp,
            "mode_outlier": mode_outlier.fillna(False),
            "mode_shock": mode_shock.fillna(False),
            "mode_extreme": mode_extreme.fillna(False),
            "is_spike": spike_mask,
        })

        n_spikes = int(spike_mask.sum())
        n_points = len(df_out)

        info = (
            f"[SpikeDetector] Detected {n_spikes} spikes over {n_points} points "
            f"(window={window_str}, z_thr={z_threshold}, pct_thr={pct_threshold}, "
            f"abs_min={abs_min_price})."
        )
        print(info)

        # Convert index to local time for plotting
        dt_local = df_out.index.tz_convert(config.LOCAL_TIME_ZONE)

        # Build Plotly figure
        fig = go.Figure()

        # Base price series
        fig.add_trace(
            go.Scatter(
                x=dt_local,
                y=df_out["eprice_15min"],
                mode="lines",
                name="eprice_15min",
            )
        )

        # Spike markers
        if n_spikes > 0:
            spikes = df_out[df_out["is_spike"]]
            fig.add_trace(
                go.Scatter(
                    x=spikes.index.tz_convert(config.LOCAL_TIME_ZONE),
                    y=spikes["eprice_15min"],
                    mode="markers",
                    name="spikes",
                    marker=dict(
                        size=9,
                        symbol="circle-open-dot",
                        color="red",          # explicitly visible
                        line=dict(width=1.5),
                    ),
                )
            )

        # Dark theme styling
        fig.update_layout(
            title=f"Price Spikes (found {n_spikes} spikes)",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font=dict(color="#e6edf3", size=14),
            xaxis=dict(
                title="Local time",
                tickfont=dict(color="#ffffff", size=13),
                gridcolor="#333333",
                zerolinecolor="#555555",
            linecolor="#ffffff",
                showline=True,
            ),
            yaxis=dict(
                title="eprice_15min (snt/kWh)",
                tickfont=dict(color="#ffffff", size=13),
                gridcolor="#333333",
                zerolinecolor="#555555",
                linecolor="#ffffff",
                showline=True,
            ),
            hoverlabel=dict(font_color="#ffffff", bgcolor="#111827", bordercolor="#444444"),
            legend=dict(
                bgcolor="#0e1117",
                bordercolor="#444444",
                borderwidth=1,
            ),
        )

        return fig, info
    

    def plot_last_day_all_sensors():
        """
        Reads the last 24h of sensor_readings from data/sensor_log.db
        and plots the temperature values of:

            temp_outdoor
            temp_tank_lower
            temp_tank_upper
            temp_supply

        All curves go into one figure, using sensor_key values directly
        as the legend labels.

        NOTE: No UI-side outlier filtering anymore – firmware already
        filters spikes. We only drop NaN / non-finite values.
        """

        SENSOR_KEYS = [
            "temp_outdoor",
            "temp_tank_lower",
            "temp_tank_upper",
            "temp_supply",
        ]

        try:
            now_utc = datetime.now(timezone.utc)
            cutoff_utc = now_utc - timedelta(days=1)
            cutoff_ts = int(cutoff_utc.timestamp())

            with sqlite3.connect(config.SENSOR_DB_PATH) as conn:
                query = f"""
                    SELECT timestamp, sensor_key, value
                    FROM sensor_readings
                    WHERE sensor_key IN ({','.join(['?'] * len(SENSOR_KEYS))})
                    AND timestamp >= ?
                    ORDER BY timestamp ASC
                """
                df = pd.read_sql_query(query, conn, params=(*SENSOR_KEYS, cutoff_ts))

        except Exception as e:
            print(f"[SensorMonitor] Error loading sensor data: {e}")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center")
            ax.axis("off")
            return fig

        if df.empty:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.text(0.5, 0.5, "No sensor data in the last 24 hours.", ha="center", va="center")
            ax.axis("off")
            return fig

        # Keep only finite numeric values; no “outlier” thresholds here
        df = df[pd.to_numeric(df["value"], errors="coerce").notna()].copy()
        df["value"] = df["value"].astype(float)

        # Second safety net: discard physically impossible temperatures
        # (defensive only – STM32 firmware already filters most spikes).
        df = df[(df["value"] >= -40.0) & (df["value"] <= 100.0)]

        # Convert to local timezone for plotting
        df["datetime"] = (
            pd.to_datetime(df["timestamp"], unit="s", utc=True)
            .dt.tz_convert(config.LOCAL_TIME_ZONE)
        )

        fig, ax = plt.subplots(figsize=(10, 4))

        for sensor_key, sub in df.groupby("sensor_key"):
            if sub.empty:
                continue
            ax.plot(sub["datetime"], sub["value"], label=sensor_key)

        ax.set_title("Last 24 Hours: Temperature Sensors")
        ax.set_xlabel("Local Time")
        ax.set_ylabel("Temperature (°C)")
        ax.grid(True)
        ax.legend()
        fig.autofmt_xdate()

        # Let Matplotlib choose y-limits automatically, no clamping here
        return fig
    
    def plot_heating_duration_vs_outdoor(days_back: int = 7):
        """
        Use heating_cycles table to plot duration vs avg_outdoor_temp
        for the last N days.
        """
        try:
            df = analytics_agent.get_duration_vs_outdoor(days_back=days_back)
        except Exception as e:
            print(f"[AnalyticsPlot] Error loading heating cycles: {e}")
            fig = go.Figure()
            fig.update_layout(
                title="Error loading heating cycle analytics",
                paper_bgcolor="#0e1117",
                plot_bgcolor="#0e1117",
                font=dict(color="#e6edf3"),
            )
            return fig

        fig = go.Figure()
        if df is None or df.empty:
            fig.update_layout(
                title="No heating cycles found for this period.",
                paper_bgcolor="#0e1117",
                plot_bgcolor="#0e1117",
                font=dict(color="#e6edf3"),
            )
            return fig

        # Convert to local time for hover
        df["start_time_local"] = df["start_time"].dt.tz_localize("UTC").dt.tz_convert(
            config.LOCAL_TIME_ZONE
        )

        fig.add_trace(
            go.Scatter(
                x=df["avg_outdoor_temp"],
                y=df["duration_minutes"],
                mode="markers",
                text=df["start_time_local"].dt.strftime("%Y-%m-%d %H:%M"),
                name="cycle",
            )
        )

        fig.update_layout(
            title="Heating duration vs outdoor temperature",
            xaxis_title="Avg outdoor temp (°C)",
            yaxis_title="Duration (minutes)",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font=dict(color="#e6edf3", size=14),
            xaxis=dict(
                tickfont=dict(color="#ffffff", size=13),
                gridcolor="#333333",
                zerolinecolor="#555555",
                linecolor="#ffffff",
                showline=True,
            ),
            yaxis=dict(
                tickfont=dict(color="#ffffff", size=13),
                gridcolor="#333333",
                zerolinecolor="#555555",
                linecolor="#ffffff",
                showline=True,
            ),
            hoverlabel=dict(font_color="#ffffff", bgcolor="#111827", bordercolor="#444444"),
        )
        return fig
          

    # Sensor plot helper
    def plot_sensor(sensor_key: str):
        if not sensor_key:
            return gr.LinePlot(value=pd.DataFrame())
        r = requests.get(f"http://localhost:5000/sensor_history/{sensor_key}?limit=200")
        if r.status_code != 200:
            return gr.LinePlot(value=pd.DataFrame())
        data = r.json()
        if not data:
            return gr.LinePlot(value=pd.DataFrame())
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        return gr.LinePlot(value=df, x="timestamp", y="value", title=sensor_key)
    
    
    # AI Analytics helpers
    def analytics_chat_fn(user_msg, history):
        """
        Simple chat wrapper:
        - history is a list of [user, assistant] turns.
        - returns updated history with new assistant message appended.
        """
        if not user_msg:
            return history

        if analytics_agent is None:
            reply = "Analytics agent is not available on the server."
        else:
            try:
                reply = analytics_agent.answer_question(user_msg)
            except Exception as e:
                print(f"[AI Analytics] Error in answer_question: {e}")
                reply = f"Error while answering analytics question: {e}"

        history = history + [[user_msg, reply]]
        return history

    def refresh_analytics_plot(window_days: float = 7.0):
        if analytics_agent is None:
            fig = go.Figure()
            fig.update_layout(
                title="Analytics agent not available.",
                paper_bgcolor="#0e1117",
                plot_bgcolor="#0e1117",
                font=dict(color="#e6edf3"),
            )
            return fig

        try:
            return analytics_agent.build_duration_vs_outdoor_bar(
                window_days=int(window_days)
            )
        except Exception as e:
            print(f"[AI Analytics] Error building analytics plot: {e}")
            fig = go.Figure()
            fig.update_layout(
                title=f"Error building analytics plot: {e}",
                paper_bgcolor="#0e1117",
                plot_bgcolor="#0e1117",
                font=dict(color="#e6edf3"),
            )
            return fig
        
    def load_best_forecast_for_plot(n_points: int = 96):
        """
        Loads the latest best-per-timestamp forecasts for plotting.
        n_points: number of 15-min points to show (e.g., 96 = 24 hours).
        """
        try:
            with sqlite3.connect(config.FORECASTING_DB_PATH) as conn:
                df = pd.read_sql_query(
                    """
                    SELECT forecasted_for_timestamp AS datetime, predicted_eprice AS eprice_15min, model_version
                    FROM eprice_forecasts_best
                    ORDER BY forecasted_for_timestamp DESC
                    LIMIT ?
                    """,
                    conn,
                    params=(int(n_points),),
                    parse_dates=["datetime"],
                )
        except Exception as e:
            print(f"[UI] Failed to load eprice_forecasts_best: {e}")
            return None

        if df.empty:
            return None

        df = df.sort_values("datetime")
        # Ensure timezone-aware for Plotly consistency
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        df = df.dropna(subset=["datetime"])
        return df


    with gr.Blocks(
        title="Unified Control System",
        theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"),
        css=DARK_CSS
    ) as demo:
        with gr.Tabs():
            with gr.TabItem("🧠 Control System"):
                gr.Markdown("## Predictive Heating System Control Panel", elem_id="panel_title")

                # ROW 1: Four equal buttons
                with gr.Row():
                    with gr.Column(scale=1):
                        cycle_btn = gr.Button("Run Forecast Cycle", variant="primary")
                    with gr.Column(scale=1):
                        actuator_btn = gr.Button("Run Actuator Control Cycle")
                    with gr.Column(scale=1):
                        finetune_btn = gr.Button("Run Fine-tuning Cycle")
                    with gr.Column(scale=1):
                        retrain_btn = gr.Button("Run Retraining Cycle")
                        # ROW 2 within the same column for checkbox under retrain
                        train_from_scratch_checkbox = gr.Checkbox(
                            label="Train from scratch", value=False
                        )

                # Bind actions
                status_output = gr.Textbox(
                    label="Cycle Trigger Status",
                    value="—",                 # seed something visible
                    lines=2,
                    interactive=True,          # <- render as <textarea>
                    elem_id="cycle-status-box"
                )
                cycle_btn.click(fn=run_orchestrator_cycle, outputs=[status_output])
                actuator_btn.click(fn=run_actuator_control_cycle, outputs=[status_output])
                finetune_btn.click(fn=run_orchestrator_finetune, outputs=[status_output])
                retrain_btn.click(fn=run_orchestrator_retraining,
                                  inputs=[train_from_scratch_checkbox],
                                  outputs=[status_output])

                # ROW 3: Status + Eprice limit + Actual Eprice
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### System Status", elem_id="system_status")
                        summary_display = gr.Textbox(label="Current State", lines=8, interactive=False)

                    with gr.Column(scale=1):
                        gr.Markdown("### Eprice Upper Limit", elem_id="eprice_limit")
                        eprice_limit_input = gr.Number(
                            label="Eprice Upper Limit (snt/kWh)",
                            value=config.DEFAULT_EPRICE_UPPER_LIMIT
                        )
                        set_eprice_limit_btn = gr.Button("Set Eprice Limit")
                        set_eprice_limit_btn.click(fn=set_eprice_limit,
                                                   inputs=[eprice_limit_input],
                                                   outputs=status_output)

                    with gr.Column(scale=1):
                        gr.Markdown("### Actual eprice", elem_id="actual_eprice")
                        actual_eprice_box = gr.Textbox(
                            label="Current 15-min price (snt/kWh)",
                            interactive=False
                        )

                # ROW 4: Interpretation (left) + Plot (right)
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Forecast Interpretation", elem_id="interpretation")
                        interp_display = gr.Textbox(label="LLM/Programmatic Summary", lines=10, interactive=False)
                    with gr.Column(scale=1):
                        gr.Markdown("### Price Forecast", elem_id="price_forecast")
                        gr.Markdown("**Forecasted for: 24 timesteps**", elem_id="forecast_caption")
                        forecast_plot_display = gr.Plot(
                            show_label=False,   # hide the component's label box entirely
                            elem_id="forecast_plot"
                        )

                # ROW 5: Log
                gr.Markdown("### Operational Log", elem_id="operational_log")
                log_display = gr.Textbox(label="Live Log", lines=15, max_lines=20, interactive=False)

                # Timer updates: now with actual price too
                loop_timer = gr.Timer(5)
                loop_timer.tick(
                    fn=update_orchestrator_display,
                    outputs=[log_display, summary_display, forecast_plot_display, interp_display, actual_eprice_box]
                )

            # Sensor tab
            with gr.TabItem("🌡️ Sensor Monitor"):
                gr.Markdown("### Sensor Monitor")
                
                with gr.Row():
                    refresh_btn = gr.Button("Refresh sensor plot", variant="primary")

                sensor_plot = gr.Plot(
                    show_label=False
                )

                
                # Refresh button to re-query DB and re-draw the figure
                refresh_btn.click(
                    fn=plot_last_day_all_sensors,
                    inputs=None,
                    outputs=sensor_plot,
                )

            # Model Diagnostics tab
            with gr.TabItem("📊 Diagnostics"):
                gr.Markdown("### Model Diagnostics", elem_id="model_diagnostics")

                # Row 1: Forecast vs Actual AND Learning Curves
                with gr.Row():
                    diag_forecast_img = gr.Image(
                        show_label=False,
                        type="filepath",
                        interactive=False,
                    )
                    learning_curves_img = gr.Image(
                        show_label=False,
                        type="filepath",
                        interactive=False,
                    )

                # Row 2: Stepwise MAE
                with gr.Row():
                    stepwise_mae_img = gr.Image(
                        show_label=False,
                        type="filepath",
                        interactive=False,
                    )

                diag_status_box = gr.Textbox(label="Status", lines=1, interactive=False)
                refresh_diag_btn = gr.Button("Refresh diagnostics")

                # Wiring the button to the function
                refresh_diag_btn.click(
                    fn=load_latest_diagnostics_plot,
                    inputs=[],
                    outputs=[diag_forecast_img, learning_curves_img, stepwise_mae_img, diag_status_box],
                )

                # Price Spike Diagnostics
                gr.Markdown("### Price Spike Diagnostics", elem_id="price_spike_detector")

                # derive default window in days from config.WINDOW (e.g. "3D")
                default_window_days = 3.0
                try:
                    if isinstance(config.WINDOW, str) and config.WINDOW.endswith("D"):
                        default_window_days = float(config.WINDOW[:-1])
                except Exception:
                    default_window_days = 3.0

                with gr.Row():
                    z_thr_input = gr.Number(
                        label="Z-score threshold",
                        value=getattr(config, "Z_THRESHOLD", 1.0),
                        precision=3,
                        info="Robust sigma threshold; higher → fewer spikes.",
                    )
                    pct_thr_input = gr.Number(
                        label="Min relative jump (factor)",
                        value=getattr(config, "PCT_THRESHOLD", 0.006),
                        precision=4,
                        info="e.g. 0.6 / 100 jump vs previous 15-min.",
                    )
                    abs_min_price_input = gr.Number(
                        label="Absolute minimum price (snt/kWh)",
                        value=getattr(config, "ABS_MIN_PRICE", 5.0),
                        precision=2,
                        info="Ignore low prices below this level.",
                    )
                    window_days_input = gr.Number(
                        label="Rolling window (days)",
                        value=float(config.WINDOW.replace('D','')) if hasattr(config, "WINDOW") else 3.0,
                        precision=1,
                        info="Length of rolling window for baseline & MAD.",
                    )

                spike_plot = gr.Plot(
                    label="Price series with spikes highlighted",
                    show_label=False,
                )

                spike_info_box = gr.Textbox(
                    label="Spike detector log",
                    lines=3,
                    interactive=False,
                )

                run_spike_btn = gr.Button("Run spike detection")

                run_spike_btn.click(
                    fn=plot_price_spikes,
                    inputs=[z_thr_input, pct_thr_input, abs_min_price_input, window_days_input],
                    outputs=[spike_plot, spike_info_box],
                )

            # Actuator Control tab
            with gr.TabItem("⚙️ Actuator Control"):
                gr.Markdown("## Actuator Control & Spike Guard", elem_id="actuator_ctrl_with_spike_guard")

                # -------------------------
                # Spike Guard (existing)
                # -------------------------
                with gr.Row():
                    spike_guard_checkbox = gr.Checkbox(
                        label="Enable automatic spike guard",
                        value=True,
                        interactive=True,
                        info="When ON, GSHP + heater are forced OFF during strong price spikes.",
                    )
                    spike_override_checkbox = gr.Checkbox(
                        label="Manual override (ignore spike guard)",
                        value=False,
                        interactive=True,
                        info="When ON, allow heating even during spikes.",
                    )

                spike_status_box = gr.Textbox(label="Spike guard status", lines=4, interactive=False)

                def update_spike_settings(enable_guard: bool, override: bool):
                    try:
                        requests.post(
                            "http://localhost:5000/orchestrator/set_spike_guard",
                            json={"enabled": enable_guard},
                        )
                        requests.post(
                            "http://localhost:5000/orchestrator/set_spike_override",
                            json={"override": override},
                        )
                        return (
                            f"Spike guard enabled: {enable_guard}\n"
                            f"Manual override (ignore guard): {override}"
                        )
                    except requests.RequestException as e:
                        return f"Error updating spike settings: {e}"

                apply_spike_settings_btn = gr.Button("Apply spike settings")
                apply_spike_settings_btn.click(
                    fn=update_spike_settings,
                    inputs=[spike_guard_checkbox, spike_override_checkbox],
                    outputs=[spike_status_box],
                )

                gr.Markdown("---")
                gr.Markdown("## Decision thresholds (recommended)", elem_id="decision_thresholds")

                # -------------------------
                # Group 1: Decision thresholds (user-facing)
                # -------------------------
                with gr.Row():
                    eprice_upper_limit = gr.Number(
                        label="EPRICE_UPPER_LIMIT (snt/kWh)",
                        value=config.DEFAULT_EPRICE_UPPER_LIMIT,
                        interactive=True,
                    )
                    eprice_very_low = gr.Number(
                        label="EPRICE_VERY_LOW_THRESHOLD (snt/kWh)",
                        value=config.CONTROL_PARAMETERS.get("EPRICE_VERY_LOW_THRESHOLD", 2.0),
                        interactive=True,
                    )

                with gr.Row():
                    tank_target_upper = gr.Number(
                        label="TANK_TARGET_TEMP_UPPER (°C)",
                        value=config.CONTROL_PARAMETERS.get("TANK_TARGET_TEMP_UPPER", 65.0),
                        interactive=True,
                    )
                    tank_hysteresis = gr.Number(
                        label="TANK_HYSTERESIS (°C)",
                        value=config.CONTROL_PARAMETERS.get("TANK_HYSTERESIS", 5.0),
                        interactive=True,
                    )
                    tank_critical_upper = gr.Number(
                        label="TANK_CRITICAL_TEMP_UPPER (°C)",
                        value=config.CONTROL_PARAMETERS.get("TANK_CRITICAL_TEMP_UPPER", 30.0),
                        interactive=True,
                    )

                with gr.Row():
                    room_target = gr.Number(
                        label="ROOM_TARGET_TEMP (°C)",
                        value=config.CONTROL_PARAMETERS.get("ROOM_TARGET_TEMP", 21.0),
                        interactive=True,
                    )
                    room_hysteresis = gr.Number(
                        label="ROOM_HYSTERESIS (°C)",
                        value=config.CONTROL_PARAMETERS.get("ROOM_HYSTERESIS", 0.5),
                        interactive=True,
                    )

                with gr.Row():
                    solar_on = gr.Number(
                        label="SOLAR_DELTA_T_ON (°C)",
                        value=config.CONTROL_PARAMETERS.get("SOLAR_DELTA_T_ON", 5.0),
                        interactive=True,
                    )
                    solar_off = gr.Number(
                        label="SOLAR_DELTA_T_OFF (°C)",
                        value=config.CONTROL_PARAMETERS.get("SOLAR_DELTA_T_OFF", 2.0),
                        interactive=True,
                    )

                ctrl_status_box = gr.Textbox(label="Actuator control parameters status", lines=6, interactive=False)

                def update_actuator_decision_thresholds(
                    eprice_upper_limit_val: float,
                    eprice_very_low_val: float,
                    tank_target_upper_val: float,
                    tank_hysteresis_val: float,
                    tank_critical_upper_val: float,
                    room_target_val: float,
                    room_hysteresis_val: float,
                    solar_on_val: float,
                    solar_off_val: float,
                ):
                    """
                    Requires backend support:
                    - POST /orchestrator/set_eprice_upper_limit  (already exists)
                    - POST /orchestrator/set_control_parameters  (you need to add; see notes below)
                    """
                    try:
                        # 1) existing endpoint
                        requests.post(
                            "http://localhost:5000/orchestrator/set_eprice_upper_limit",
                            json={"eprice_upper_limit": float(eprice_upper_limit_val)},
                        )

                        # 2) bulk control parameters endpoint (recommended)
                        payload = {
                            "control_parameters": {
                                "EPRICE_VERY_LOW_THRESHOLD": float(eprice_very_low_val),
                                "TANK_TARGET_TEMP_UPPER": float(tank_target_upper_val),
                                "TANK_HYSTERESIS": float(tank_hysteresis_val),
                                "TANK_CRITICAL_TEMP_UPPER": float(tank_critical_upper_val),
                                "ROOM_TARGET_TEMP": float(room_target_val),
                                "ROOM_HYSTERESIS": float(room_hysteresis_val),
                                "SOLAR_DELTA_T_ON": float(solar_on_val),
                                "SOLAR_DELTA_T_OFF": float(solar_off_val),
                            }
                        }
                        requests.post(
                            "http://localhost:5000/orchestrator/set_control_parameters",
                            json=payload,
                        )

                        return (
                            "Updated decision thresholds:\n"
                            f"- EPRICE_UPPER_LIMIT={float(eprice_upper_limit_val):.2f}\n"
                            f"- EPRICE_VERY_LOW_THRESHOLD={float(eprice_very_low_val):.2f}\n"
                            f"- TANK_TARGET_TEMP_UPPER={float(tank_target_upper_val):.2f}, "
                            f"TANK_HYSTERESIS={float(tank_hysteresis_val):.2f}, "
                            f"TANK_CRITICAL_TEMP_UPPER={float(tank_critical_upper_val):.2f}\n"
                            f"- ROOM_TARGET_TEMP={float(room_target_val):.2f}, ROOM_HYSTERESIS={float(room_hysteresis_val):.2f}\n"
                            f"- SOLAR_DELTA_T_ON={float(solar_on_val):.2f}, SOLAR_DELTA_T_OFF={float(solar_off_val):.2f}"
                        )
                    except requests.RequestException as e:
                        return f"Error updating actuator decision thresholds: {e}"

                apply_ctrl_params_btn = gr.Button("Apply decision thresholds", variant="primary")
                apply_ctrl_params_btn.click(
                    fn=update_actuator_decision_thresholds,
                    inputs=[
                        eprice_upper_limit,
                        eprice_very_low,
                        tank_target_upper,
                        tank_hysteresis,
                        tank_critical_upper,
                        room_target,
                        room_hysteresis,
                        solar_on,
                        solar_off,
                    ],
                    outputs=[ctrl_status_box],
                )

                gr.Markdown("---")

                # -------------------------
                # Group 2: Safety tuning (advanced)
                # -------------------------
                with gr.Accordion("Advanced safety tuning (ROC + max temp)", open=False):
                    with gr.Row():
                        tank_temp_upper_max = gr.Number(
                            label="TANK_TEMP_UPPER_MAX (°C)",
                            value=getattr(config, "TANK_TEMP_UPPER_MAX", 90.0),
                            interactive=True,
                        )

                    gr.Markdown("### Tank ROC safety")
                    with gr.Row():
                        tank_roc_window = gr.Number(
                            label="TANK_ROC_WINDOW_SECONDS",
                            value=getattr(config, "TANK_ROC_WINDOW_SECONDS", 300),
                            interactive=True,
                        )
                        tank_roc_thr = gr.Number(
                            label="TANK_ROC_THRESHOLD_C_PER_SEC",
                            value=getattr(config, "TANK_ROC_THRESHOLD_C_PER_SEC", -0.003),
                            interactive=True,
                        )
                        tank_roc_min_delta = gr.Number(
                            label="TANK_ROC_MIN_DELTA_C",
                            value=getattr(config, "TANK_ROC_MIN_DELTA_C", 0.2),
                            interactive=True,
                        )

                    gr.Markdown("### Room ROC comfort guard")
                    with gr.Row():
                        room_roc_window = gr.Number(
                            label="ROOM_ROC_WINDOW_SECONDS",
                            value=getattr(config, "ROOM_ROC_WINDOW_SECONDS", 600),
                            interactive=True,
                        )
                        room_roc_thr = gr.Number(
                            label="ROOM_ROC_COOL_THRESHOLD_C_PER_SEC",
                            value=getattr(config, "ROOM_ROC_COOL_THRESHOLD_C_PER_SEC", -0.0005),
                            interactive=True,
                        )
                        room_roc_min_delta = gr.Number(
                            label="ROOM_ROC_MIN_DELTA_C",
                            value=getattr(config, "ROOM_ROC_MIN_DELTA_C", 0.1),
                            interactive=True,
                        )

                    gr.Markdown("### Outdoor ROC preheat guard")
                    with gr.Row():
                        outdoor_roc_window = gr.Number(
                            label="OUTDOOR_ROC_WINDOW_SECONDS",
                            value=getattr(config, "OUTDOOR_ROC_WINDOW_SECONDS", 900),
                            interactive=True,
                        )
                        outdoor_roc_thr = gr.Number(
                            label="OUTDOOR_ROC_COOL_THRESHOLD_C_PER_SEC",
                            value=getattr(config, "OUTDOOR_ROC_COOL_THRESHOLD_C_PER_SEC", -0.001),
                            interactive=True,
                        )
                        outdoor_roc_min_delta = gr.Number(
                            label="OUTDOOR_ROC_MIN_DELTA_C",
                            value=getattr(config, "OUTDOOR_ROC_MIN_DELTA_C", 0.2),
                            interactive=True,
                        )

                    safety_status_box = gr.Textbox(label="Safety parameters status", lines=8, interactive=False)

                    def update_advanced_safety_params(
                        tank_temp_upper_max_val: float,
                        tank_roc_window_val: float,
                        tank_roc_thr_val: float,
                        tank_roc_min_delta_val: float,
                        room_roc_window_val: float,
                        room_roc_thr_val: float,
                        room_roc_min_delta_val: float,
                        outdoor_roc_window_val: float,
                        outdoor_roc_thr_val: float,
                        outdoor_roc_min_delta_val: float,
                    ):
                        """
                        Requires backend support:
                        - POST /orchestrator/set_control_parameters (bulk)
                        Note: For these to affect ActuatorCtrlAgent at runtime, the agent must
                        read them from control_params (preferred) rather than getattr(config, ...).
                        """
                        try:
                            payload = {
                                "control_parameters": {
                                    "TANK_TEMP_UPPER_MAX": float(tank_temp_upper_max_val),

                                    "TANK_ROC_WINDOW_SECONDS": float(tank_roc_window_val),
                                    "TANK_ROC_THRESHOLD_C_PER_SEC": float(tank_roc_thr_val),
                                    "TANK_ROC_MIN_DELTA_C": float(tank_roc_min_delta_val),

                                    "ROOM_ROC_WINDOW_SECONDS": float(room_roc_window_val),
                                    "ROOM_ROC_COOL_THRESHOLD_C_PER_SEC": float(room_roc_thr_val),
                                    "ROOM_ROC_MIN_DELTA_C": float(room_roc_min_delta_val),

                                    "OUTDOOR_ROC_WINDOW_SECONDS": float(outdoor_roc_window_val),
                                    "OUTDOOR_ROC_COOL_THRESHOLD_C_PER_SEC": float(outdoor_roc_thr_val),
                                    "OUTDOOR_ROC_MIN_DELTA_C": float(outdoor_roc_min_delta_val),
                                }
                            }
                            requests.post(
                                "http://localhost:5000/orchestrator/set_control_parameters",
                                json=payload,
                            )
                            return (
                                "Updated advanced safety params (pending agent support):\n"
                                f"- TANK_TEMP_UPPER_MAX={float(tank_temp_upper_max_val):.2f}\n"
                                f"- TANK_ROC: window={float(tank_roc_window_val):.0f}s, thr={float(tank_roc_thr_val):.6f}, minΔ={float(tank_roc_min_delta_val):.2f}\n"
                                f"- ROOM_ROC: window={float(room_roc_window_val):.0f}s, thr={float(room_roc_thr_val):.6f}, minΔ={float(room_roc_min_delta_val):.2f}\n"
                                f"- OUTDOOR_ROC: window={float(outdoor_roc_window_val):.0f}s, thr={float(outdoor_roc_thr_val):.6f}, minΔ={float(outdoor_roc_min_delta_val):.2f}"
                            )
                        except requests.RequestException as e:
                            return f"Error updating advanced safety params: {e}"

                    apply_safety_params_btn = gr.Button("Apply advanced safety tuning")
                    apply_safety_params_btn.click(
                        fn=update_advanced_safety_params,
                        inputs=[
                            tank_temp_upper_max,
                            tank_roc_window, tank_roc_thr, tank_roc_min_delta,
                            room_roc_window, room_roc_thr, room_roc_min_delta,
                            outdoor_roc_window, outdoor_roc_thr, outdoor_roc_min_delta,
                        ],
                        outputs=[safety_status_box],
                    )
            # AI Analytics tab
            with gr.TabItem("📈 AI Analytics"):
                gr.Markdown("## AI-Assisted Heating Analytics", elem_id="ai-assisted-heating-analytics")

                # Row 1: 
                with gr.Row():
                    with gr.Column(scale=1):
                        analytics_chatbot = gr.Chatbot(
                            label="AI Analytics Assistant",
                            show_label=False,
                            height=300,
                        )
                        analytics_input = gr.Textbox(
                            label="Ask a question about system performance",
                            placeholder=(
                                "Examples:\n"
                                "- How many heating cycles have there been in the last 7 days?\n"
                                "- How is heating performing during freezing nights?"
                                "- Should i make some adjustments to the system?"
                                "- How could I minimize electricity consumption?"

                            ),
                            lines=4,
                        )
                        send_btn = gr.Button("Ask")

                send_btn.click(
                    fn=analytics_chat_fn,
                    inputs=[analytics_input, analytics_chatbot],
                    outputs=[analytics_chatbot],
                )
                
                # Row 2: Live analytics plot
                gr.Markdown(
                    "### Heating duration vs outdoor temperature (last 7 days)"
                    )
                with gr.Row():
                    analytics_window_days = gr.Number(
                        label="Window (days)",
                        value=7,
                        precision=0,
                        info="How many past days to include in the analytics plot.",
                    )
                    refresh_plot_btn = gr.Button("Refresh analytics plot")

                analytics_plot = gr.Plot(
                    label="Heating duration vs outdoor temperature",
                    show_label=False,
                )

                refresh_plot_btn.click(
                    fn=refresh_analytics_plot,
                    inputs=[analytics_window_days],
                    outputs=[analytics_plot],
                )

                def analytics_chat_fn(user_msg, history):
                    """
                    Simple chat wrapper:
                    - history is a list of [user, assistant] turns.
                    - returns updated history with new assistant message appended.
                    """
                    if not user_msg:
                        return history

                    if analytics_agent is None:
                        reply = "Analytics agent is not available on the server."
                    else:
                        try:
                            reply = analytics_agent.answer_question(user_msg)
                        except Exception as e:
                            print(f"[AI Analytics] Error in answer_question: {e}")
                            reply = f"Error while answering analytics question: {e}"

                    history = history + [[user_msg, reply]]
                    return history

                def refresh_analytics_plot(window_days: float = 7.0):
                    if analytics_agent is None:
                        fig = go.Figure()
                        fig.update_layout(
                            title="Analytics agent not available.",
                            paper_bgcolor="#0e1117",
                            plot_bgcolor="#0e1117",
                            font=dict(color="#e6edf3"),
                        )
                        return fig

                    try:
                        return analytics_agent.build_duration_vs_outdoor_plot(
                            window_days=int(window_days)
                        )
                    except Exception as e:
                        print(f"[AI Analytics] Error building analytics plot: {e}")
                        fig = go.Figure()
                        fig.update_layout(
                            title=f"Error building analytics plot: {e}",
                            paper_bgcolor="#0e1117",
                            plot_bgcolor="#0e1117",
                            font=dict(color="#e6edf3"),
                        )
                        return fig


    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

    
    
# SYSTEM LAUNCH
def auto_run_forecast_cycle():
    """A thread to run the forecast cycle periodically."""
    print("🕒 Automatic forecast cycle thread started. Will run every hour.")
    while True:
        print("Auto-run: Kicking off hourly forecast cycle via API...")
        try:
            requests.post("http://localhost:5000/orchestrator/run_cycle")
            print("Auto-run: Successfully triggered forecast cycle via API.")
        except requests.RequestException as e:
            print(f"Auto-run: Failed to trigger forecast cycle via API: {e}")
        time.sleep(3600) # Wait for 1 hour before the next run

# Auto-run thread for actuator control cycle
def auto_run_actuator_control_cycle():
    """A thread to run the actuator control cycle periodically."""
    print(f"🕒 Automatic actuator control cycle thread started. Will run every {config.ACTUATOR_CONTROL_INTERVAL_SECONDS / 60} minutes.")
    while True:
        print("Auto-run: Kicking off actuator control cycle via API...")
        try:
            requests.post("http://localhost:5000/orchestrator/run_actuator_control_cycle")
            print("Auto-run: Successfully triggered actuator control cycle via API.")
        except requests.RequestException as e:
            print(f"Auto-run: Failed to trigger actuator control cycle via API: {e}")
        time.sleep(config.ACTUATOR_CONTROL_INTERVAL_SECONDS) # Configurable interval

def launch_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")

def run_all():
    print("🚀 Initializing the Orchestrating Agent...")
    
    # Optional: Uncomment to run the forecast cycle automatically in the background
    print("⏰ Starting automatic forecast cycle thread (commented out by default, uncomment if needed)...")
    threading.Thread(target=auto_run_forecast_cycle, daemon=True).start() #un-comment to stop

    # Launch actuator control auto-run thread
    print(f"⚙️ Starting automatic actuator control cycle thread...")
    threading.Thread(target=auto_run_actuator_control_cycle, daemon=True).start()

    print("🚀 Starting FastAPI backend...")
    threading.Thread(target=launch_fastapi, daemon=True).start()

    time.sleep(2) # Give FastAPI a moment to start

    print("📊 Starting Gradio UI on http://localhost:7860")
    launch_gradio()

if __name__ == "__main__":
    run_all()








