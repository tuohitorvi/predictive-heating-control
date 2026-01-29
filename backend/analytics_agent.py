# backend/analytics_agent.py

import os
import re
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timezone

import sqlite3
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from ollama import Client as OllamaClient

from . import config

# RAG: optional TF-IDF backend
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    TfidfVectorizer = None


@dataclass
class HeatingStats:
    window_label: str
    n_cycles: int
    total_minutes: float
    avg_minutes: Optional[float]
    min_outdoor: Optional[float]
    max_outdoor: Optional[float]
    avg_outdoor: Optional[float]
    avg_efficiency: Optional[float]


class AnalyticsAgent:
    """
    Advanced local analytics + LLM agent for heating system diagnostics.
    
    Capabilities:
    1. Statistical Analysis: detailed aggregation of heating cycles.
    2. RAG (Retrieval-Augmented Generation): Context-aware answers from local docs.
    3. Anomaly Detection: Heuristic logic to identify potential system failures 
       before asking the LLM.
    """

    def __init__(self, logger=None):
        self._log = logger if logger is not None else print

        # LLM config
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.model_name = os.getenv("OLLAMA_MODEL_NAME", "llama2")
        self.llm_client = OllamaClient(host=self.ollama_host)

        # Database Configuration
        self.heating_db_path = getattr(
            config,
            "HEATING_CYCLES_DB_PATH",
            getattr(config, "FORECASTING_DB_PATH", "data/forecasting_data.db"),
        )

        # RAG State
        self._rag_enabled: bool = False
        self._rag_docs: List[str] = []
        self._rag_meta: List[Tuple[str, int]] = []
        self._rag_vectorizer = None
        self._rag_matrix = None

        self._log(f"[AnalyticsAgent] Init: DB={self.heating_db_path}, Model={self.model_name}")
        self._init_rag()

    # RAG: Document Ingestion
    def _init_rag(self):
        """Initializes TF-IDF based RAG with paragraph-aware chunking."""
        if TfidfVectorizer is None:
            self._log("[AnalyticsAgent][RAG] scikit-learn missing. RAG disabled.")
            return

        default_paths = [
            "README.md",
            "docs/system_description.txt",
            "docs/config_docs.md",
            "docs/Hardware_components.txt",
            "docs/troubleshooting.md" # Added hypothetical troubleshooting doc
        ]

        doc_paths = [p for p in default_paths if os.path.exists(p)]
        
        if not doc_paths:
            self._log("[AnalyticsAgent][RAG] No documentation files found.")
            return

        chunks: List[str] = []
        meta: List[Tuple[str, int]] = []

        for path in doc_paths:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                
                # Improved chunking: Respect paragraph boundaries
                file_chunks = self._smart_chunk_text(text, target_size=1000)
                for idx, chunk in enumerate(file_chunks):
                    chunks.append(chunk)
                    meta.append((path, idx))
            except Exception as e:
                self._log(f"[AnalyticsAgent][RAG] Error reading {path}: {e}")

        if not chunks:
            return

        try:
            # Create TF-IDF Matrix
            self._rag_vectorizer = TfidfVectorizer(stop_words="english")
            self._rag_matrix = self._rag_vectorizer.fit_transform(chunks)
            self._rag_enabled = True
            self._rag_docs = chunks
            self._rag_meta = meta
            self._log(f"[AnalyticsAgent][RAG] Indexed {len(chunks)} chunks from {len(doc_paths)} files.")
        except Exception as e:
            self._log(f"[AnalyticsAgent][RAG] Indexing failed: {e}")
            self._rag_enabled = False

    @staticmethod
    def _smart_chunk_text(text: str, target_size: int = 1000) -> List[str]:
        """
        Splits text by double newlines (paragraphs) to preserve semantic meaning,
        then aggregates them until target_size is reached.
        """
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_len = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If a single paragraph is massive, force split it (fallback)
            if len(para) > target_size:
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_len = 0
                chunks.append(para) 
                continue

            if current_len + len(para) > target_size:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_len = len(para)
            else:
                current_chunk.append(para)
                current_len += len(para)

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
        
        return chunks

    def _retrieve_doc_context(self, question: str, top_k: int = 4) -> str:
        """Retrieves most relevant documentation chunks."""
        if not self._rag_enabled or not question.strip():
            return ""

        try:
            q_vec = self._rag_vectorizer.transform([question])
            sims = cosine_similarity(q_vec, self._rag_matrix).flatten()
            
            # Get top indices
            top_indices = sims.argsort()[-top_k:][::-1]
            
            pieces = []
            for idx in top_indices:
                if sims[idx] < 0.1: # Threshold to ignore irrelevant noise
                    continue
                source, _ = self._rag_meta[idx]
                chunk = self._rag_docs[idx]
                pieces.append(f"--- SOURCE: {os.path.basename(source)} ---\n{chunk}")

            return "\n\n".join(pieces)
        except Exception as e:
            self._log(f"[AnalyticsAgent] Retrieval error: {e}")
            return ""

    # Data Loading & Analytics
    def _load_heating_cycles(self) -> pd.DataFrame:
        """Loads heating cycles from SQLite with correct type casting."""
        try:
            conn = sqlite3.connect(self.heating_db_path)
            df = pd.read_sql("SELECT * FROM heating_cycles ORDER BY start_time ASC", conn)
            conn.close()
        except Exception as e:
            self._log(f"[AnalyticsAgent] DB Error: {e}")
            return pd.DataFrame()

        if df.empty: 
            return df

        # UTC conversion
        cols_to_date = ["start_time", "end_time"]
        for col in cols_to_date:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

        return df

    def _detect_anomalies(self, df: pd.DataFrame) -> List[str]:
        """
        Pythonic heuristic logic to detect system faults BEFORE sending to LLM.
        This provides the 'Consultant' capabilities.
        """
        anomalies = []
        if df.empty:
            return ["No data available to detect anomalies."]

        now_utc = pd.Timestamp.now(tz="UTC")
        last_24h = df[df["start_time"] > (now_utc - pd.Timedelta(hours=24))]

        # Heuristic 1: No heating in cold weather
        # (Assuming the current temp can be accessed, otherwise use recent cycle avg)
        recent_temps = df["avg_outdoor_temp"].tail(5).dropna()
        if not recent_temps.empty:
            avg_recent_temp = recent_temps.mean()
            if avg_recent_temp < 10.0 and last_24h.empty:
                anomalies.append(
                    f"CRITICAL: Outdoor temp is approx {avg_recent_temp:.1f}°C, "
                    "but NO heating cycles occurred in the last 24 hours."
                )

        # Heuristic 2: Short cycling (cycles < 5 mins)
        short_cycles = last_24h[last_24h["duration_minutes"] < 5]
        if not short_cycles.empty:
            count = len(short_cycles)
            anomalies.append(
                f"WARNING: {count} short-cycles detected (under 5 mins) in last 24h. "
                "Check pump flow rate or sensor placement."
            )

        return anomalies

    def _stats_for_window(self, df: pd.DataFrame, days: int, label: str) -> HeatingStats:
        if df.empty or "start_time" not in df.columns:
            return HeatingStats(label, 0, 0.0, None, None, None, None, None)

        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
        df_w = df[df["start_time"] >= cutoff]

        if df_w.empty:
            return HeatingStats(label, 0, 0.0, None, None, None, None, None)

        return HeatingStats(
            window_label=label,
            n_cycles=len(df_w),
            total_minutes=float(df_w["duration_minutes"].sum()),
            avg_minutes=float(df_w["duration_minutes"].mean()),
            min_outdoor=float(df_w["avg_outdoor_temp"].min()),
            max_outdoor=float(df_w["avg_outdoor_temp"].max()),
            avg_outdoor=float(df_w["avg_outdoor_temp"].mean()),
            avg_efficiency=None # Add efficiency calculation logic here if metric exists
        )

    def get_recent_stats(self) -> Dict[str, Any]:
        df = self._load_heating_cycles()
        return {
            "last_24h": self._stats_for_window(df, 1, "Last 24 Hours"),
            "last_7d": self._stats_for_window(df, 7, "Last 7 Days"),
            "last_30d": self._stats_for_window(df, 30, "Last 30 Days"),
            "anomalies": self._detect_anomalies(df)
        }
    
    def build_duration_vs_outdoor_plot(self, window_days: int = 7) -> go.Figure:
        """
        Backwards-compatible alias for the UI.
        """
        return self.build_duration_vs_outdoor_bar(window_days=window_days)

    # Visualization
    def build_duration_vs_outdoor_bar(self, window_days: int = 7) -> go.Figure:
        """Generates Plotly figure for UI."""
        now_utc = pd.Timestamp.now(tz="UTC")
        cutoff = now_utc - pd.Timedelta(days=window_days)

        # Use the same DB path as _load_heating_cycles
        with sqlite3.connect(self.heating_db_path) as conn:
            df = pd.read_sql_query(
                """
                SELECT
                    cycle_id,
                    actuator,
                    start_time,
                    duration_minutes,
                    avg_outdoor_temp
                FROM heating_cycles
                """,
                conn,
            )

        if df.empty:
            fig = go.Figure()
            fig.update_layout(
                title="No heating cycles found.",
                paper_bgcolor="#0e1117",
                plot_bgcolor="#0e1117",
                font=dict(color="#e6edf3"),
            )
            return fig

        # Parse times and filter last `window_days`
        df["start_time"] = pd.to_datetime(df["start_time"], utc=True, errors="coerce")
        df = df.dropna(subset=["start_time"])
        df = df[df["start_time"] >= cutoff]

        if df.empty:
            fig = go.Figure()
            fig.update_layout(
                title=f"No heating cycles in last {window_days} days.",
                paper_bgcolor="#0e1117",
                plot_bgcolor="#0e1117",
                font=dict(color="#e6edf3"),
            )
            return fig

        # Prepare labels & values
        df["start_local"] = df["start_time"].dt.tz_convert(config.LOCAL_TIME_ZONE)
        df["start_label"] = df["start_local"].dt.strftime("%m-%d %H:%M")

        def _fmt_temp(v):
            try:
                return f"{float(v):.1f}°C"
            except Exception:
                return "NA"

        x_labels = [
            f"{row['start_label']} ({_fmt_temp(row['avg_outdoor_temp'])})"
            for _, row in df.iterrows()
        ]
        durations = df["duration_minutes"].astype(float).tolist()

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=x_labels,
                y=durations,
                name="Heating duration",
            )
        )

        fig.update_layout(
            title=f"Heating cycle duration vs. avg outdoor temp (last {window_days} days)",
            xaxis_title="Cycle start time (local) and avg outdoor temp",
            yaxis_title="Duration (minutes)",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font=dict(color="#e6edf3", size=14),
            xaxis=dict(
                tickangle=-45,
                tickfont=dict(color="#ffffff", size=11),
                gridcolor="#333333",
            ),
            yaxis=dict(
                tickfont=dict(color="#ffffff", size=13),
                gridcolor="#333333",
            ),
            hoverlabel=dict(font_color="#ffffff", bgcolor="#111827", bordercolor="#444444"),
        )

        return fig
        
    # LLM & Analysis Logic
    def answer_question(self, user_question: str) -> str:
        """
        Main interface. Combines Stats + Heuristics + RAG -> LLM.
        """
        # 1. Gather Data
        stats_data = self.get_recent_stats()
        
        # 2. Gather Docs (RAG)
        # Increase the context retrieved if query implies a complex fault
        doc_context = self._retrieve_doc_context(user_question, top_k=5)

        # 3. Format Data for LLM
        def fmt_stat(s: HeatingStats):
            if s.n_cycles == 0: return f"[{s.window_label}]: No heating cycles recorded."
            return (f"[{s.window_label}]: {s.n_cycles} cycles, Total: {s.total_minutes:.1f} min. "
                    f"Avg Duration: {s.avg_minutes:.1f} min. "
                    f"Outdoor Temp: Avg {s.avg_outdoor:.1f}C (Min {s.min_outdoor:.1f}C).")

        stats_text = "\n".join([
            fmt_stat(stats_data["last_24h"]),
            fmt_stat(stats_data["last_7d"]),
            fmt_stat(stats_data["last_30d"])
        ])

        anomalies_text = "NONE DETECTED."
        if stats_data["anomalies"]:
            anomalies_text = "\n- ".join(stats_data["anomalies"])

        # 4. Construct Professional Prompt
        prompt = f"""
You are an expert Control Systems Engineer analyzing a residential heating system.
Your goal is to diagnose faults, explain behaviour, and optimize performance.

### SYSTEM HEALTH & ANOMALIES (Detected by Code)
{anomalies_text}

### HEATING DATA (Fact-Based)
{stats_text}

### DOCUMENTATION CONTEXT (Reference Manuals)
{doc_context if doc_context else "No specific manuals found for this query."}

### USER QUERY
"{user_question}"

### INSTRUCTIONS
1. **Prioritize Faults**: If 'SYSTEM HEALTH' shows anomalies (e.g., short-cycling, no heat in cold weather), address those FIRST.
2. **Use Data**: Cite the exact numbers from HEATING DATA. Do not invent timestamps or durations.
3. **Reference Docs**: Use the DOCUMENTATION CONTEXT to explain *why* the system behaves this way (e.g., control logic, sensors).
4. **Tone**: Professional, concise, and analytical. 

Answer:
"""
        
        # 5. Call LLM with EXPANDED Context Window
        try:
            result = self.llm_client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "num_ctx": 4096,       # CRITICAL: Llama2 needs space for Docs + Stats
                    "temperature": 0.1,    # Low temp for factual consistency
                    "repeat_penalty": 1.1,
                    "top_k": 40,
                },
                stream=False,
            )
            return result.get("response", "").strip()
        except Exception as e:
            self._log(f"[AnalyticsAgent] LLM Critical Error: {e}")
            return f"Error analyzing system: {e}. (Raw Data: {stats_text})"





