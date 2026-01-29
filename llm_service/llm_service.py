# llm_service.py
import os
import json
import traceback
from io import StringIO
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
import time
from dataclasses import dataclass

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
from ollama import Client

from contextlib import asynccontextmanager



# Ollama / generation config
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://...")
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", ...)
llm_client = Client(host=OLLAMA_HOST)

GEN_OPTS = {
    "num_predict": 220,   
    "num_ctx": 512,       
    "temperature": 0.2,
    "top_k": 40,
    "top_p": 0.9,
    "repeat_penalty": 1.05,
    "keep_alive": "5m",
}

# Streaming time rules:
FIRST_TOKEN_TIMEOUT = 45.0  # must get first token in this time or fallback
SOFT_TOTAL_CAP      = 90.0  # return partial accumulated tokens by this time cap

EXECUTOR = ThreadPoolExecutor(max_workers=2)



# API schemas
class InterpretRequest(BaseModel):
    # Preferred: CSV text including 'datetime'
    dataframe: Optional[str] = None
    forecast_data: Optional[List[Dict[str, Any]]] = None
    target_column: str
    # Ignored but accepted for old clients
    context_data: Optional[Dict[str, Any]] = None


class InterpretResponse(BaseModel):
    explanation: str

# Helpers
def _df_from_request(req: InterpretRequest) -> pd.DataFrame:
    """Build a DataFrame from CSV or list-of-dicts."""
    if req.dataframe:
        try:
            return pd.read_csv(StringIO(req.dataframe))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")
    if req.forecast_data:
        try:
            return pd.DataFrame(req.forecast_data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse forecast_data: {e}")
    raise HTTPException(status_code=400, detail="Provide 'dataframe' (CSV) or 'forecast_data' (list).")


def _tz_aware_utc(ts: pd.Series) -> pd.Series:
    s = pd.to_datetime(ts, errors="coerce", utc=True)
    return s


def _slope_per_hour(t_utc: pd.Series, y: pd.Series) -> Optional[float]:
    """Linear slope (units per hour) over valid points; returns None if not enough."""
    try:
        t = pd.to_datetime(t_utc, errors="coerce", utc=True)
        mask = ~(t.isna() | y.isna())
        if mask.sum() < 3:
            return None
        x = (t[mask].astype("int64") / 1e9).astype(float)  # seconds
        coeffs = np.polyfit(x, y[mask].astype(float), 1)
        return float(coeffs[0] * 3600.0)  # per hour
    except Exception:
        return None


@dataclass
class TimeWindowStats:
    count: int
    mean: Optional[float]
    std: Optional[float]
    min_val: Optional[float]
    min_time_local: Optional[str]
    max_val: Optional[float]
    max_time_local: Optional[str]
    slope_per_hour: Optional[float]


def _window_stats(df: pd.DataFrame, target_col: str) -> TimeWindowStats:
    s = pd.to_numeric(df[target_col], errors="coerce")
    t = _tz_aware_utc(df["datetime"])
    mask = ~(s.isna() | t.isna())
    if mask.sum() == 0:
        return TimeWindowStats(0, None, None, None, None, None, None, None)

    sv = s[mask]
    tv = t[mask]

    mean = float(sv.mean()) if len(sv) else None
    std = float(sv.std()) if len(sv) > 1 else None

    min_val = None; min_time_local = None
    max_val = None; max_time_local = None
    try:
        idx_min = sv.idxmin(); idx_max = sv.idxmax()
        min_val = float(sv.loc[idx_min]) if pd.notna(sv.loc[idx_min]) else None
        max_val = float(sv.loc[idx_max]) if pd.notna(sv.loc[idx_max]) else None

        tmin = tv.loc[idx_min]; tmax = tv.loc[idx_max]
        if isinstance(tmin, pd.Timestamp):
            tmin_local = tmin.tz_convert("Europe/Helsinki").strftime("%Y-%m-%d %H:%M")
            min_time_local = tmin_local
        if isinstance(tmax, pd.Timestamp):
            tmax_local = tmax.tz_convert("Europe/Helsinki").strftime("%Y-%m-%d %H:%M")
            max_time_local = tmax_local
    except Exception:
        pass

    slope = _slope_per_hour(tv, sv)
    return TimeWindowStats(mask.sum(), mean, std, min_val, min_time_local, max_val, max_time_local, slope)


def _hourly_aggregate(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Resample to hourly means to get longer perspective cheaply."""
    tmp = df[["datetime", target_col]].copy()
    tmp["datetime"] = _tz_aware_utc(tmp["datetime"])
    tmp.dropna(subset=["datetime"], inplace=True)
    tmp.set_index("datetime", inplace=True)
    hr = tmp.resample("1h").mean()
    hr.reset_index(inplace=True)
    return hr


def _rolling_features(hr: pd.DataFrame, target_col: str) -> Dict[str, Optional[float]]:
    """Rolling means to expose broader trends."""
    out: Dict[str, Optional[float]] = {
        "mean_6h": None,
        "mean_24h": None,
        "delta_last_hour_vs_6h": None,
        "delta_last_hour_vs_24h": None,
    }
    if hr.empty or target_col not in hr.columns:
        return out

    s = pd.to_numeric(hr[target_col], errors="coerce")
    mean_6h = s.rolling(6, min_periods=3).mean().iloc[-1]
    mean_24h = s.rolling(24, min_periods=6).mean().iloc[-1]
    last = s.iloc[-1] if s.size else np.nan

    out["mean_6h"] = float(mean_6h) if pd.notna(mean_6h) else None
    out["mean_24h"] = float(mean_24h) if pd.notna(mean_24h) else None
    out["delta_last_hour_vs_6h"] = float(last - mean_6h) if pd.notna(last) and pd.notna(mean_6h) else None
    out["delta_last_hour_vs_24h"] = float(last - mean_24h) if pd.notna(last) and pd.notna(mean_24h) else None
    return out


def _day_over_day(hr: pd.DataFrame, target_col: str) -> Dict[str, Optional[float]]:
    """Compare last 24h average vs previous 24h average."""
    out = {"mean_last_24h": None, "mean_prev_24h": None, "delta_mean": None}
    if len(hr) < 30:
        return out
    s = pd.to_numeric(hr[target_col], errors="coerce")
    s = s.dropna()
    if len(s) < 30:
        return out
    last_24 = s.iloc[-24:]
    prev_24 = s.iloc[-48:-24] if len(s) >= 48 else pd.Series([], dtype=float)
    if len(last_24) >= 6:
        out["mean_last_24h"] = float(last_24.mean())
    if len(prev_24) >= 6:
        out["mean_prev_24h"] = float(prev_24.mean())
    if out["mean_last_24h"] is not None and out["mean_prev_24h"] is not None:
        out["delta_mean"] = float(out["mean_last_24h"] - out["mean_prev_24h"])
    return out


def _compact_facts_text(target_col: str,
                        full_stats: TimeWindowStats,
                        hr_stats: TimeWindowStats,
                        roll: Dict[str, Optional[float]],
                        dday: Dict[str, Optional[float]],
                        aux: Dict[str, Optional[float]]) -> str:
    """Compose a compact fact set for the LLM prompt."""
    lines = []
    lines.append(f"Target column: {target_col}")
    lines.append(f"Total points (raw): {full_stats.count}")
    if full_stats.mean is not None:
        lines.append(f"Mean (raw): {full_stats.mean:.3f} snt/kWh")
    if full_stats.std is not None:
        lines.append(f"Volatility (raw std): {full_stats.std:.3f}")
    if full_stats.slope_per_hour is not None:
        lines.append(f"Overall slope: {full_stats.slope_per_hour:+.3f} snt/kWh per hour")
    if full_stats.min_val is not None:
        lines.append(f"Min: {full_stats.min_val:.3f} at {full_stats.min_time_local}")
    if full_stats.max_val is not None:
        lines.append(f"Max: {full_stats.max_val:.3f} at {full_stats.max_time_local}")

    # hourly (smoother)
    if hr_stats.count:
        lines.append(f"Hourly points: {hr_stats.count}")
    if hr_stats.slope_per_hour is not None:
        lines.append(f"Hourly slope: {hr_stats.slope_per_hour:+.3f} snt/kWh per hour")

    # rolling
    if roll.get("mean_6h") is not None:
        lines.append(f"Mean(6h): {roll['mean_6h']:.3f}")
    if roll.get("mean_24h") is not None:
        lines.append(f"Mean(24h): {roll['mean_24h']:.3f}")
    if roll.get("delta_last_hour_vs_6h") is not None:
        lines.append(f"LastHour - Mean(6h): {roll['delta_last_hour_vs_6h']:+.3f}")
    if roll.get("delta_last_hour_vs_24h") is not None:
        lines.append(f"LastHour - Mean(24h): {roll['delta_last_hour_vs_24h']:+.3f}")

    # day-over-day
    if dday.get("mean_last_24h") is not None:
        lines.append(f"Avg(last 24h): {dday['mean_last_24h']:.3f}")
    if dday.get("mean_prev_24h") is not None:
        lines.append(f"Avg(prev 24h): {dday['mean_prev_24h']:.3f}")
    if dday.get("delta_mean") is not None:
        lines.append(f"ΔAverage (last 24h - prev 24h): {dday['delta_mean']:+.3f}")

    # optional auxiliaries (temperature/wind means)
    if aux.get("mean_temp") is not None:
        lines.append(f"Mean temp: {aux['mean_temp']:.2f} °C")
    if aux.get("mean_windEnergy") is not None:
        lines.append(f"Mean windEnergy: {aux['mean_windEnergy']:.0f} MW")

    return "\n".join(lines)


def _programmatic_long_summary(target_col: str,
                               facts: str) -> str:
    """Stronger programmatic fallback with long-window perspective."""
    return (
        "## Price trend summary (programmatic)\n\n"
        f"{facts}\n\n"
        "Interpretation:\n"
        "- The slope values show the general direction (positive = rising, negative = falling).\n"
        "- Day-over-day delta indicates whether the recent 24h average improved or worsened vs the prior day.\n"
        "- Rolling 6h and 24h means help distinguish short-term vs broader trend.\n"
        "- Extremes mark the local best/worst price windows and their local times (Europe/Helsinki).\n"
        "\n*(LLM not used due to latency; server performed analysis programmatically.)*"
    )


def _stream_ollama_to_queue(prompt: str, q: Queue):
    """Run Ollama streaming in a worker thread and push chunks to a queue."""
    try:
        for part in llm_client.generate(
            model=OLLAMA_MODEL_NAME,
            prompt=prompt,
            options=GEN_OPTS,
            stream=True,
        ):
            r = part.get("response", "")
            if r:
                q.put(("chunk", r))
            if part.get("done"):
                q.put(("done", None))
                break
    except Exception as e:
        q.put(("error", str(e)))

# Startup / Shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"LLM Service: Connected to {OLLAMA_HOST}, model={OLLAMA_MODEL_NAME}")
    try:
        llm_client.generate(
            model=OLLAMA_MODEL_NAME,
            prompt="Hello",
            options={**GEN_OPTS, "temperature": 0.0},
            stream=False,
        )
        print("LLM Service: Warm-up OK.")
    except Exception:
        print("LLM Service: Warm-up FAILED.")
        traceback.print_exc()

    yield 

    print("LLM Service: Shutting down.")


app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/interpret-forecast", response_model=InterpretResponse)
async def interpret_forecast_endpoint(request: InterpretRequest):
    # 1) Build DataFrame
    df = _df_from_request(request)
    if "datetime" not in df.columns:
        return InterpretResponse(explanation="No 'datetime' column present in data.")

    # Normalize datetimes
    df["datetime"] = _tz_aware_utc(df["datetime"])

    # Make sure target column exists
    target_col = request.target_column
    if target_col not in df.columns:
        return InterpretResponse(explanation=f"Target column '{target_col}' not found in data.")

    # 2) Compute long-window programmatic features fast
    full_stats = _window_stats(df, target_col)

    # Aux means (temp, windEnergy) on valid rows
    aux_means = {"mean_temp": None, "mean_windEnergy": None}
    for col in ["temp", "windEnergy"]:
        if col in df.columns:
            sc = pd.to_numeric(df[col], errors="coerce")
            aux_means[f"mean_{col}"] = float(sc.mean()) if sc.dropna().size else None

    # Hourly aggregate + rolling + day-over-day
    hr = _hourly_aggregate(df, target_col)
    hr_stats = _window_stats(hr, target_col) if not hr.empty else TimeWindowStats(0, None, None, None, None, None, None, None)
    roll = _rolling_features(hr, target_col)
    dday = _day_over_day(hr, target_col)

    # 3) Build compact facts text
    facts = _compact_facts_text(target_col, full_stats, hr_stats, roll, dday, aux_means)

    # 4) Build prompt
    prompt = f"""
        You are an expert electricity-market analyst.
        Your analysis is targeted at people who want to minimize their electricity bills.
        Write ONE tight paragraph (90–140 words) interpreting the electricity price trend.

        Use ONLY the facts below (already aggregated):
        {facts}

        Guidelines:
        - Start with overall direction and magnitude (use the slope numbers).
        - Contrast short-term (6h) vs broad (24h) trend.
        - Mention extremes with local times (Europe/Helsinki) and implications.
        - Include day-over-day change and what it suggests for the next 24–48 hours.
        - If temp/windEnergy are available, relate them qualitatively (no overclaiming).
        - Keep it concise, declarative, and self-contained.
        """

    # 5) Stream from LLM; return partial if slow, otherwise fallback to strong programmatic text
    q: Queue = Queue()
    EXECUTOR.submit(_stream_ollama_to_queue, prompt, q)

    collected: List[str] = []
    start = time.time()
    got_first = False

    while True:
        elapsed = time.time() - start
        timeout = 0.25 if got_first else FIRST_TOKEN_TIMEOUT

        try:
            kind, payload = q.get(timeout=timeout)
        except Empty:
            if not got_first:
                print(f"[LLM] No first token within {FIRST_TOKEN_TIMEOUT}s → fallback.")
                return InterpretResponse(explanation=_programmatic_long_summary(target_col, facts))
            else:
                if elapsed >= SOFT_TOTAL_CAP:
                    text = "".join(collected).strip()
                    print(f"[LLM] Soft cap {SOFT_TOTAL_CAP}s reached (after first token) — returning {len(text)} chars.")
                    if not text:
                        text = _programmatic_long_summary(target_col, facts)
                    return InterpretResponse(explanation=text)
                continue

        if kind == "chunk":
            if not got_first:
                print(f"[LLM] First token received at {elapsed:.1f}s")
            got_first = True
            collected.append(payload)

            if elapsed >= SOFT_TOTAL_CAP:
                text = "".join(collected).strip()
                print(f"[LLM] Soft cap {SOFT_TOTAL_CAP}s reached — returning {len(text)} characters.")
                if not text:
                    print("[LLM] Soft cap hit but text empty → fallback summary.")
                    text = _programmatic_long_summary(target_col, facts)
                return InterpretResponse(explanation=text)

        elif kind == "done":
            text = "".join(collected).strip()
            print(f"[LLM] Done. Received {len(text)} characters.")
            if not text:
                print("[LLM] Done but output empty → fallback summary.")
                text = _programmatic_long_summary(target_col, facts)
            return InterpretResponse(explanation=text)

        elif kind == "error":
            return InterpretResponse(explanation=_programmatic_long_summary(target_col, facts))


# Local run
if __name__ == "__main__":
    uvicorn.run("llm_service:app", host="0.0.0.0", port=5001)