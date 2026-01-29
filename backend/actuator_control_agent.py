# backend/actuator_control_agent.py

import paho.mqtt.client as mqtt
from typing import Dict, Any
import pandas as pd
import sqlite3
from datetime import datetime, timezone

from . import config 

class ActuatorCycleLogger:
    """
    Tracks ON→OFF cycles for heating actuators (gshp, heater_element)
    and writes summarized rows into the heating_cycles table.
    """

    def __init__(self, forecasting_db_path: str, logger=None):
        self.db_path = forecasting_db_path
        self._log = logger or (lambda msg: None)
        # per-actuator active cycle state
        self._active: dict[str, dict] = {}
        self._ensure_table()

    def _ensure_table(self):
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS heating_cycles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                actuator TEXT NOT NULL,
                cycle_id TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT,
                duration_minutes REAL,
                start_temp_outdoor REAL,
                start_temp_tank_lower REAL,
                start_temp_tank_upper REAL,
                start_temp_room_main REAL,
                end_temp_outdoor REAL,
                end_temp_tank_lower REAL,
                end_temp_tank_upper REAL,
                end_temp_room_main REAL,
                temp_delta_tank_upper REAL,
                avg_outdoor_temp REAL,
                efficiency_min_per_degree REAL
            );
            """
        )
        conn.commit()
        conn.close()

    def _read_outdoor_avg_between(self, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> float | None:
        """
        Compute average temp_outdoor from sensor_log.db between two times.
        Uses config.SENSOR_DB_PATH and sensor_key='temp_outdoor'.
        """
        import sqlite3
        if end_ts <= start_ts:
            return None

        try:
            with sqlite3.connect(config.SENSOR_DB_PATH) as conn:
                start_unix = int(start_ts.timestamp())
                end_unix = int(end_ts.timestamp())
                df = pd.read_sql_query(
                    """
                    SELECT value FROM sensor_readings
                    WHERE sensor_key = 'temp_outdoor'
                      AND timestamp BETWEEN ? AND ?
                    """,
                    conn,
                    params=(start_unix, end_unix),
                )
            if df.empty:
                return None
            vals = pd.to_numeric(df["value"], errors="coerce").dropna()
            if vals.empty:
                return None
            return float(vals.mean())
        except Exception as e:
            self._log(f"[CycleLogger] Error computing avg outdoor temp: {e}")
            return None

    def update_cycles(
        self,
        now_utc: pd.Timestamp,
        live_temps: dict,
        new_signals: dict[str, bool],
    ):
        """
        Call once per control cycle, AFTER you have decided final actuator signals.

        - Detect ON→OFF transitions for 'gshp' and 'heater_element'.
        - On ON: create active cycle, store start temps.
        - On OFF: close cycle, compute metrics, write row to heating_cycles.
        """
        tracked = ["gshp", "heater_element"]

        # Normalize now_utc
        if not isinstance(now_utc, pd.Timestamp):
            now_utc = pd.Timestamp(now_utc, tz="UTC").tz_convert("UTC")
        elif now_utc.tz is None:
            now_utc = now_utc.tz_localize("UTC")

        for actuator in tracked:
            new_state = bool(new_signals.get(actuator, False))
            active = self._active.get(actuator)

            # Transition OFF -> ON: start new cycle
            if new_state and active is None:
                start_time = now_utc
                st_out = live_temps.get("temp_outdoor")
                st_low = live_temps.get("temp_tank_lower")
                st_up  = live_temps.get("temp_tank_upper")
                st_room = live_temps.get("temp_room_main")

                self._active[actuator] = {
                    "start_time": start_time,
                    "start_temp_outdoor": float(st_out) if st_out is not None else None,
                    "start_temp_tank_lower": float(st_low) if st_low is not None else None,
                    "start_temp_tank_upper": float(st_up) if st_up is not None else None,
                    "start_temp_room_main": float(st_room) if st_room is not None else None,
                }
                self._log(
                    f"[CycleLogger] {actuator} ON → starting cycle at {start_time.isoformat()}"
                )

            # Transition ON -> OFF: finalize cycle
            elif (not new_state) and active is not None:
                end_time = now_utc
                duration_min = (end_time - active["start_time"]).total_seconds() / 60.0

                end_out = live_temps.get("temp_outdoor")
                end_low = live_temps.get("temp_tank_lower")
                end_up  = live_temps.get("temp_tank_upper")
                end_room = live_temps.get("temp_room_main")

                start_up = active.get("start_temp_tank_upper")
                temp_delta = None
                if start_up is not None and end_up is not None:
                    temp_delta = float(end_up) - float(start_up)

                # avg outdoor between start & end using sensor_log.db
                avg_outdoor = self._read_outdoor_avg_between(
                    active["start_time"], end_time
                )
                if avg_outdoor is None:
                    # fallback: simple mean of start/end temps
                    start_out = active.get("start_temp_outdoor")
                    if (start_out is not None) and (end_out is not None):
                        avg_outdoor = float(start_out + end_out) / 2.0

                eff = None
                if temp_delta is not None and abs(temp_delta) > 0.05:
                    eff = duration_min / temp_delta  # minutes per °C gained (or lost)

                cycle_id = f"{actuator}_{active['start_time'].isoformat()}"

                import sqlite3
                conn = sqlite3.connect(self.db_path)
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT INTO heating_cycles (
                        actuator, cycle_id, start_time, end_time, duration_minutes,
                        start_temp_outdoor, start_temp_tank_lower, start_temp_tank_upper,
                        start_temp_room_main,
                        end_temp_outdoor, end_temp_tank_lower, end_temp_tank_upper,
                        end_temp_room_main,
                        temp_delta_tank_upper, avg_outdoor_temp, efficiency_min_per_degree
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        actuator,
                        cycle_id,
                        active["start_time"].isoformat(),
                        end_time.isoformat(),
                        float(duration_min),
                        active.get("start_temp_outdoor"),
                        active.get("start_temp_tank_lower"),
                        active.get("start_temp_tank_upper"),
                        active.get("start_temp_room_main"),
                        float(end_out) if end_out is not None else None,
                        float(end_low) if end_low is not None else None,
                        float(end_up) if end_up is not None else None,
                        float(end_room) if end_room is not None else None,
                        temp_delta,
                        avg_outdoor,
                        eff,
                    ),
                )
                conn.commit()
                conn.close()

                self._log(
                    f"[CycleLogger] {actuator} OFF → stored cycle {cycle_id}: "
                    f"duration={duration_min:.1f} min, ΔT_upper={temp_delta}, "
                    f"avg_outdoor={avg_outdoor}, eff={eff}"
                )

                # clear active state
                self._active[actuator] = None

class ActuatorCtrlAgent:
    def __init__(self, logger=None):
        if logger:
            self._log_operation = logger
        else:
            self._log_operation = lambda msg: print(f"ActuatorCtrlAgent Log: {msg}")


        self.client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION1)
        self.latest_states = {}
        self.mode = "auto"
        self._log_operation("ActuatorCtrlAgent initialized.")
        self.db_path = config.FORECASTING_DB_PATH
        self.cycle_logger = ActuatorCycleLogger(
            forecasting_db_path=self.db_path,
            logger=self._log_operation,
        )
        # State for ROC safety logic (tank upper)
        self._roc_state = {
            "last_temp": None,     # last temp_tank_upper sample (°C)
            "last_ts": None,       # timestamp of last sample (UTC pd.Timestamp)
            "engaged": False       # True while safety forcing is active
        }

        # ROC state for room comfort guard
        self._room_roc_state = {
            "last_temp": None,
            "last_ts": None,
            "engaged": False,
        }

        # ROC state for outdoor preheat guard
        self._outdoor_roc_state = {
            "last_temp": None,
            "last_ts": None,
            "engaged": False,
        }

        # Price spike guard state
        # If True, "spike mode" is applied when a price spike is detected.
        self.spike_guard_enabled = True

        # If True, spike guard (allow heating even during spikes) is ignored.
        # Controlled by manual override button in UI.
        self.spike_manual_override = False

        # The last spike info for UI debugging
        self.last_spike_info: None 

    def connect(self):
        self.client.connect(config.MQTT_BROKER_IP, config.MQTT_BROKER_PORT, 60)
        self.client.loop_start()
        self._log_operation("MQTT client connected and loop started.")

    def stop(self):
        self.client.loop_stop()
        self.client.disconnect()
        self._log_operation("MQTT client stopped.")


    def _publish_command(self, actuator: str, command: str):
        topic = f"actuators/{actuator}/set"
        self.client.publish(topic, command)
        self._log_operation(f"Sent command to {topic}: {command}")


    def _get_eprice_data_from_ground_truth(self) -> pd.DataFrame:
        """
        Fetches the latest eprice_15min data from the ground_truth_table.
        This includes both historical and day-ahead (real) prices available.
        """
        try:
            conn = sqlite3.connect(self.db_path) 
            # Select only datetime and eprice_15min, ensuring it's not null
            df = pd.read_sql_query(f"""
                SELECT {config.TIME_COLUMN}, {config.TARGET_COLUMN}
                FROM {config.GROUND_TRUTH_TABLE}
                WHERE {config.TARGET_COLUMN} IS NOT NULL
                ORDER BY {config.TIME_COLUMN} ASC
            """, conn)
            conn.close()

            if not df.empty:
                df[config.TIME_COLUMN] = pd.to_datetime(df[config.TIME_COLUMN], errors='coerce', utc=True)
                df = df.dropna(subset=[config.TIME_COLUMN, config.TARGET_COLUMN]) # Drop rows where datetime conversion failed or price is NaN
                return df
            else:
                self._log_operation("⚠️ No eprice_15min data found in ground_truth_table. Returning empty DataFrame.")
                return pd.DataFrame(columns=[config.TIME_COLUMN, config.TARGET_COLUMN])
        except Exception as e:
            self._log_operation(f"❌ Failed to fetch eprice_15min data from ground_truth_table: {e}")
            return pd.DataFrame(columns=[config.TIME_COLUMN, config.TARGET_COLUMN])
        
    def _get_last_known_day_ahead_end_utc(self, now_utc: pd.Timestamp) -> pd.Timestamp:
        """
        Your corrected market rule:
        - Before 13:00 UTC: day-ahead real prices known until TODAY 22:45 UTC
        - After  13:00 UTC: day-ahead real prices known until TOMORROW 22:45 UTC
        """
        if now_utc.tz is None:
            now_utc = now_utc.tz_localize("UTC")
        else:
            now_utc = now_utc.tz_convert("UTC")

        if now_utc.hour < 13:
            return now_utc.replace(hour=22, minute=45, second=0, microsecond=0)
        return (now_utc + pd.Timedelta(days=1)).replace(hour=22, minute=45, second=0, microsecond=0)
    
    def _fetch_forecast_vintages(
        self,
        start_utc: pd.Timestamp,
        end_utc: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Fetch raw forecast vintages for timestamps in (start_utc, end_utc].
        Returns rows: datetime, predicted_eprice, forecast_generation_time
        """
        if start_utc.tz is None:
            start_utc = start_utc.tz_localize("UTC")
        if end_utc.tz is None:
            end_utc = end_utc.tz_localize("UTC")

        start_str = start_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_str = end_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(
                    """
                    SELECT
                        forecasted_for_timestamp AS datetime,
                        predicted_eprice,
                        forecast_generation_time
                    FROM eprice_forecasts
                    WHERE forecasted_for_timestamp > ?
                    AND forecast_generation_time IS NOT NULL
                    AND forecasted_for_timestamp <= ?
                    AND predicted_eprice IS NOT NULL
                    ORDER BY forecasted_for_timestamp ASC
                    """,
                    conn,
                    params=(start_str, end_str),
                )

            if df.empty:
                return df

            df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
            df["forecast_generation_time"] = pd.to_datetime(df["forecast_generation_time"], utc=True, errors="coerce")
            df = df.dropna(subset=["datetime", "predicted_eprice"])
            df["predicted_eprice"] = pd.to_numeric(df["predicted_eprice"], errors="coerce")
            df = df.dropna(subset=["predicted_eprice"])
            return df

        except Exception as e:
            self._log_operation(f"⚠️ Failed to fetch forecast vintages from eprice_forecasts: {e}")
            return pd.DataFrame(columns=["datetime", "predicted_eprice", "forecast_generation_time"])


    def _build_weighted_forecast_extension(
        self,
        start_utc: pd.Timestamp,
        end_utc: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        For each timestamp in (start_utc, end_utc], compute:
        - median_pred: median of all available vintages
        - n_versions: number of vintages
        - weight: min(1.0, n_versions/3.0)

        Returns rows: datetime, median_pred, n_versions, weight
        """
        vintages = self._fetch_forecast_vintages(start_utc, end_utc)
        if vintages.empty:
            return pd.DataFrame(columns=["datetime", "median_pred", "n_versions", "weight"])

        g = vintages.groupby("datetime")["predicted_eprice"]
        med = g.median()
        out = pd.DataFrame({
            "datetime": med.index,
            "median_pred": med.values,
            "n_versions": g.size().values,
        })
        # n_versions(min/max)=5/5 --> 5 forecast vintages per timestamp
        # fg. 1.-> 0.75^2=0.5625, 2.-> 0.75^3=0.421875, 3.-> 0.75^4=0.31640625, 4.-> 0.75^5=0.2373046875
        # avg_weight: 1−0.2373046875=0.7626953125
        out["weight"] = (1.0 - (0.75 ** out["n_versions"])).clip(0.0, 1.0)
        out = out.sort_values("datetime")
        return out


    def _weighted_mean(self, values: pd.Series, weights: pd.Series) -> float | None:
        v = pd.to_numeric(values, errors="coerce")
        w = pd.to_numeric(weights, errors="coerce")
        mask = v.notna() & w.notna() & (w > 0)
        if not mask.any():
            return None
        return float((v[mask] * w[mask]).sum() / w[mask].sum())

        
    # Rate-of-change safety check for temp_tank_upper (temp3)
    def _check_and_apply_roc_safety(self, live_temps: Dict[str, float], signals: Dict[str, bool]) -> bool:
        """
        Monitors the rate of change of temp_tank_upper. If it falls faster than the
        configured threshold, force gshp + heater_element + circ_pump ON until
        the rate recovers (>= threshold). Returns True if safety forcing is active.
        """
        # Read config (with safe defaults if missing)
        window_sec   = getattr(config, "TANK_ROC_WINDOW_SECONDS", 300)
        thr_c_per_s  = getattr(config, "TANK_ROC_THRESHOLD_C_PER_SEC", -0.003)
        min_delta_c  = getattr(config, "TANK_ROC_MIN_DELTA_C", 0.2)

        t2 = pd.Timestamp.now(tz="UTC")
        temp2 = live_temps.get("temp_tank_upper")

        # If no measurement, nothing to do
        if temp2 is None:
            return self._roc_state["engaged"]

        last_temp = self._roc_state["last_temp"]
        last_ts   = self._roc_state["last_ts"]

        # Only evaluate when a prior sample spaced at least window_sec is available
        if last_temp is not None and last_ts is not None:
            dt = (t2 - last_ts).total_seconds()
            if dt >= max(1.0, float(window_sec)):  # avoid div-by-zero, enforce window
                dT = float(temp2) - float(last_temp)

                # Ignore sensor noise
                if abs(dT) >= float(min_delta_c):
                    rate = dT / dt  # °C/s
                    self._log_operation(
                        f"[ROC] temp_tank_upper dT={dT:+.3f} °C over dt={dt:.1f}s -> rate={rate:+.5f} °C/s (thr={thr_c_per_s:+.5f})"
                    )

                    # Trip if cooling faster (more negative) than threshold
                    if rate < float(thr_c_per_s):
                        if not self._roc_state["engaged"]:
                            self._log_operation("[ROC] ⚠️ Fast drop detected. Engaging safety: force GSHP + Heater + Circ Pump ON.")
                        self._roc_state["engaged"] = True
                    else:
                        # If previously engaged and rate has recovered, disengage and return to normal logic
                        if self._roc_state["engaged"]:
                            self._log_operation("[ROC] ✅ Rate recovered. Disengaging safety, returning to normal control.")
                        self._roc_state["engaged"] = False

                # Move window forward once evaluated
                self._roc_state["last_temp"] = float(temp2)
                self._roc_state["last_ts"]   = t2

            else:
                # Not enough time elapsed yet; keep prior engaged status
                pass
        else:
            # First sample: initialize window
            self._roc_state["last_temp"] = float(temp2)
            self._roc_state["last_ts"]   = t2

        # If engaged, force outputs regardless of normal logic
        if self._roc_state["engaged"]:
            signals["gshp"] = True
            signals["heater_element"] = True
            # Keep/force circ pump ON while engaged (always ON during safety)
            signals["circ_pump"] = True
            return True

        return False
    

    def _check_and_apply_room_roc_comfort(
        self,
        live_temps: Dict[str, float],
        control_params: Dict,
        signals: Dict[str, bool],
    ) -> bool:
        """
        ROC-enhanced comfort guard for room temperature.

        If the room temperature is falling faster than ROOM_ROC_COOL_THRESHOLD_C_PER_SEC
        over ROOM_ROC_WINDOW_SECONDS (and by at least ROOM_ROC_MIN_DELTA_C),
        we force heater_element + circ_pump ON (comfort override), ignoring price logic.

        This is still bypassed by:
          - Price ceiling (handled earlier)
          - Max tank temp safety (handled earlier)
        """
        # Get room temperature from either current or future key
        temp_room = None
        if isinstance(live_temps, dict):
            temp_room = live_temps.get("temp_room_main")
            if temp_room is None:
                temp_room = live_temps.get("temp_room")  # planned temp5

        if temp_room is None:
            return self._room_roc_state["engaged"]

        # Config with safe defaults
        window_sec  = float(getattr(config, "ROOM_ROC_WINDOW_SECONDS", 600.0))   # 10 min
        thr_c_per_s = float(getattr(config, "ROOM_ROC_COOL_THRESHOLD_C_PER_SEC", -0.0005))  # ~-0.03°C/min
        min_delta_c = float(getattr(config, "ROOM_ROC_MIN_DELTA_C", 0.1))

        t2 = pd.Timestamp.now(tz="UTC")
        try:
            temp2 = float(temp_room)
        except Exception:
            return self._room_roc_state["engaged"]

        last_temp = self._room_roc_state["last_temp"]
        last_ts   = self._room_roc_state["last_ts"]

        if last_temp is not None and last_ts is not None:
            dt = (t2 - last_ts).total_seconds()
            if dt >= max(1.0, window_sec):
                dT = temp2 - float(last_temp)

                if abs(dT) >= min_delta_c:
                    rate = dT / dt  # °C/s
                    self._log_operation(
                        f"[ROC-ROOM] dT={dT:+.3f} °C over dt={dt:.1f}s -> rate={rate:+.5f} °C/s "
                        f"(thr={thr_c_per_s:+.5f})"
                    )

                    if rate < thr_c_per_s:
                        if not self._room_roc_state["engaged"]:
                            self._log_operation(
                                "[ROC-ROOM] ⚠️ Fast room cooling detected. "
                                "Engaging comfort guard: force Heater + Circ Pump ON."
                            )
                        self._room_roc_state["engaged"] = True
                    else:
                        if self._room_roc_state["engaged"]:
                            self._log_operation(
                                "[ROC-ROOM] ✅ Room cooling rate recovered. "
                                "Disengaging comfort guard."
                            )
                        self._room_roc_state["engaged"] = False

                # advance window
                self._room_roc_state["last_temp"] = temp2
                self._room_roc_state["last_ts"]   = t2
        else:
            # first sample
            self._room_roc_state["last_temp"] = temp2
            self._room_roc_state["last_ts"]   = t2

        if self._room_roc_state["engaged"]:
            # Comfort override: ensure heating path is active
            signals["heater_element"] = True
            signals["circ_pump"] = True
            return True
        
    def _check_and_apply_outdoor_roc_preheat(
        self,
        live_temps: Dict[str, float],
        control_params: Dict,
        signals: Dict[str, bool],
    ) -> bool:
        """
        ROC-based preheat guard for outdoor temperature.

        If temp_outdoor is falling faster than OUTDOOR_ROC_COOL_THRESHOLD_C_PER_SEC
        over OUTDOOR_ROC_WINDOW_SECONDS (and by at least OUTDOOR_ROC_MIN_DELTA_C),
        we proactively force heater_element + circ_pump ON to avoid the tank
        dropping too low later.

        Still subordinate to:
          - Price ceiling
          - Max tank temperature
        """
        temp_outdoor = None
        if isinstance(live_temps, dict):
            temp_outdoor = live_temps.get("temp_outdoor")

        if temp_outdoor is None:
            return self._outdoor_roc_state["engaged"]

        # Config with safe defaults
        window_sec  = float(getattr(config, "OUTDOOR_ROC_WINDOW_SECONDS", 900.0))   # 15 min
        thr_c_per_s = float(getattr(config, "OUTDOOR_ROC_COOL_THRESHOLD_C_PER_SEC", -0.001))  # ~-0.06°C/min
        min_delta_c = float(getattr(config, "OUTDOOR_ROC_MIN_DELTA_C", 0.2))

        t2 = pd.Timestamp.now(tz="UTC")
        try:
            temp2 = float(temp_outdoor)
        except Exception:
            return self._outdoor_roc_state["engaged"]

        last_temp = self._outdoor_roc_state["last_temp"]
        last_ts   = self._outdoor_roc_state["last_ts"]

        if last_temp is not None and last_ts is not None:
            dt = (t2 - last_ts).total_seconds()
            if dt >= max(1.0, window_sec):
                dT = temp2 - float(last_temp)

                if abs(dT) >= min_delta_c:
                    rate = dT / dt  # °C/s
                    self._log_operation(
                        f"[ROC-OUTDOOR] dT={dT:+.3f} °C over dt={dt:.1f}s -> rate={rate:+.5f} °C/s "
                        f"(thr={thr_c_per_s:+.5f})"
                    )

                    if rate < thr_c_per_s:
                        if not self._outdoor_roc_state["engaged"]:
                            self._log_operation(
                                "[ROC-OUTDOOR] ⚠️ Rapid outdoor cooling detected. "
                                "Engaging preheat: force Heater + Circ Pump ON."
                            )
                        self._outdoor_roc_state["engaged"] = True
                    else:
                        if self._outdoor_roc_state["engaged"]:
                            self._log_operation(
                                "[ROC-OUTDOOR] ✅ Outdoor cooling rate recovered. "
                                "Disengaging preheat guard."
                            )
                        self._outdoor_roc_state["engaged"] = False

                # advance window
                self._outdoor_roc_state["last_temp"] = temp2
                self._outdoor_roc_state["last_ts"]   = t2
        else:
            # first sample
            self._outdoor_roc_state["last_temp"] = temp2
            self._outdoor_roc_state["last_ts"]   = t2

        if self._outdoor_roc_state["engaged"]:
            signals["heater_element"] = True
            signals["circ_pump"] = True
            return True

        return False


        
    def _room_needs_heat(self, live_temps: Dict, control_params: Dict) -> bool:
        """
        Returns True if current room temperature is below ROOM_TARGET_TEMP.
        Works with both current ('temp_room_main') and future ('temp_room') keys.
        If no room sensor is available, returns False (conservative).
        """
        # Support both current and future key names
        temp_room = None
        if isinstance(live_temps, dict):
            temp_room = live_temps.get("temp_room_main")
            if temp_room is None:
                temp_room = live_temps.get("temp_room")  # planned 'temp5'

        target = control_params.get("ROOM_TARGET_TEMP")
        if temp_room is None or target is None:
            return False  # no sensor or no target configured → don't force circulation

        try:
            return float(temp_room) < float(target)
        except Exception:
            return False
        
    def _enforce_max_temperature_limit(
        self,
        temp_tank_upper: float,
        live_temps: Dict,
        control_params: Dict,
        signals: Dict[str, bool],
    ) -> bool:
        """
        If upper tank temperature >= TANK_TEMP_UPPER_MAX:
          - Force all heating OFF (gshp, heater_element)
          - Keep circulation ON only if the room hasn't reached target.
          - Return True to indicate override took effect.
        """
        limit = getattr(config, "TANK_TEMP_UPPER_MAX", None)
        if limit is None or temp_tank_upper is None:
            return False

        try:
            over_limit = float(temp_tank_upper) >= float(limit)
        except Exception:
            return False

        if over_limit:
            # Hard stop heating
            signals["gshp"] = False
            signals["heater_element"] = False

            # Circulation only if room needs heat
            needs_heat = self._room_needs_heat(live_temps, control_params)
            signals["circ_pump"] = bool(needs_heat)

            self._log_operation(
                f"🚫 Maximum tank temperature reached: {float(temp_tank_upper):.2f} °C "
                f"(limit {float(limit):.2f}). Heating disabled."
            )
            # Optional info message requested
            self._log_operation(f"The maximum temperature has been reached {config.TANK_TEMP_UPPER_MAX}.")
            return True

        return False

    def get_current_eprice_now(self) -> float | None:
        """Returns the current interval's eprice (snt/kWh) based on ground_truth_table,
        using the same logic as run_control_logic()."""
        df = self._get_eprice_data_from_ground_truth()
        if df.empty:
            return None

        now_utc = pd.Timestamp.now(tz='UTC')
        floored_now_utc = now_utc.floor('15min')

        # Exact interval first
        exact = df[df[config.TIME_COLUMN] == floored_now_utc]
        if not exact.empty:
            return float(exact[config.TARGET_COLUMN].iloc[0])

        # Fallback: latest past price
        past = df[df[config.TIME_COLUMN] < now_utc]
        if not past.empty:
            return float(past[config.TARGET_COLUMN].iloc[-1])

        return None
    
    def set_spike_guard_enabled(self, enabled: bool):
        self.spike_guard_enabled = bool(enabled)
        self._log_operation(f"[SpikeGuard] spike_guard_enabled set to {self.spike_guard_enabled}")

    def set_spike_manual_override(self, override: bool):
        self.spike_manual_override = bool(override)
        self._log_operation(f"[SpikeGuard] spike_manual_override set to {self.spike_manual_override}")

    
    def run_control_logic(self, live_temps: Dict, control_params: Dict) -> Dict[str, bool]:
        self._log_operation("Running actuator control logic...")

        if self.mode != "auto":
            self._log_operation("Manual mode active. Skipping automatic control.")
            return self.latest_states.copy()

        full_eprice_df = self._get_eprice_data_from_ground_truth()

        if full_eprice_df.empty:
            self._log_operation("❌ Cannot run control logic: No valid eprice_15min data available from ground_truth_table.")
            return self.latest_states.copy()

        now_utc = pd.Timestamp.now(tz='UTC')

        # Floor the current time to the nearest past 15-minute interval
        floored_now_utc = now_utc.floor('15min')

        # Try to find the price for this exact 15-minute interval
        current_price_entry = full_eprice_df[full_eprice_df[config.TIME_COLUMN] == floored_now_utc]

        if not current_price_entry.empty:
            eprice_now = current_price_entry[config.TARGET_COLUMN].iloc[0]
            self._log_operation(
                f"Determined eprice_now from ground_truth_table for current interval "
                f"{floored_now_utc.isoformat()}: {eprice_now:.2f} snt/kWh"
            )
        else:
            # Fallback: if no exact match for the current interval, take the latest available price *before* now_utc
            past_prices = full_eprice_df[full_eprice_df[config.TIME_COLUMN] < now_utc]
            if not past_prices.empty:
                eprice_now = past_prices[config.TARGET_COLUMN].iloc[-1]
                self._log_operation(
                    f"⚠️ No exact price for current interval. Using latest past price from "
                    f"{past_prices[config.TIME_COLUMN].iloc[-1].isoformat()}: {eprice_now:.2f} snt/kWh (fallback)."
                )
            else:
                self._log_operation(
                    "⚠️ No past or current eprice_15min found in ground_truth_table for 'eprice_now'. "
                    "Defaulting to 15.0."
                )
                eprice_now = 15.0  # Default if no suitable price found

        # 2. Determine the *end* of the 'future' window for average calculation
        # If 'now' < 13:00 UTC, the window ends at 22:45 today.
        # If 'now' >= 13:00 UTC, the window ends at 22:45 tomorrow.
        if now_utc.hour < 13:
            # End of today 22:45 UTC
            avg_price_window_end = now_utc.replace(hour=22, minute=45, second=0, microsecond=0)
        else:
            # End of tomorrow 22:45 UTC
            avg_price_window_end = (now_utc + pd.Timedelta(days=1)).replace(
                hour=22, minute=45, second=0, microsecond=0
            )

       
        # Avg Future Price = known actuals + forecast extension (weighted)
        # 2.1 Determine last timestamp where day-ahead (real) prices are known (22:45 rule)
        known_end_utc = self._get_last_known_day_ahead_end_utc(now_utc)

        # Average forward: known day-ahead end (22:45) + forecast extension horizon
        forecast_horizon_steps = int(getattr(config, "FORECAST_HORIZON_STEPS", 24))  # your model: 24
        forecast_step_minutes = int(getattr(config, "FORECAST_STEP_MINUTES", 15))    # 15-min
        avg_end_utc = known_end_utc + pd.Timedelta(minutes=forecast_horizon_steps * forecast_step_minutes)


        # Segment A: actual known future prices in (now_utc, min(avg_end_utc, known_end_utc)]
        actual_end = min(avg_end_utc, known_end_utc)
        actual_future = full_eprice_df[
            (full_eprice_df[config.TIME_COLUMN] > now_utc)
            & (full_eprice_df[config.TIME_COLUMN] <= actual_end)
        ].copy()

        # Prepare weighted average components
        # - Actual prices have weight = 1.0
        combined_vals = []
        combined_wts = []

        if not actual_future.empty:
            combined_vals.append(actual_future[config.TARGET_COLUMN].astype(float))
            combined_wts.append(pd.Series([1.0] * len(actual_future), index=actual_future.index))

        # Segment B: forecast extension in (known_end_utc, avg_end_utc] only if avg window exceeds known prices
        forecast_ext = pd.DataFrame()
        if avg_end_utc > known_end_utc:
            forecast_ext = self._build_weighted_forecast_extension(known_end_utc, avg_end_utc)

            if not forecast_ext.empty:
                combined_vals.append(forecast_ext["median_pred"].astype(float))
                combined_wts.append(forecast_ext["weight"].astype(float))

        # Compute weighted mean if we have anything; else fallback to current price
        avg_future_price = float(eprice_now)  # fallback
        if combined_vals:
            all_vals = pd.concat(combined_vals, ignore_index=True)
            all_wts = pd.concat(combined_wts, ignore_index=True)
            wmean = self._weighted_mean(all_vals, all_wts)
            if wmean is not None:
                avg_future_price = wmean

        # Logging (diagnostic)
        n_actual = 0 if actual_future.empty else len(actual_future)
        n_fc = 0 if forecast_ext.empty else len(forecast_ext)

        if n_actual > 0 or n_fc > 0:
            self._log_operation(
                "Calculated Avg Future Price using actuals + forecast-extension "
                f"(actual_points={n_actual}, forecast_points={n_fc}) "
                f"from now={now_utc.isoformat()} to avg_end={avg_end_utc.isoformat()} "
                f"(known_end={known_end_utc.isoformat()}): {avg_future_price:.2f} snt/kWh"
            )
            if n_fc > 0:
                # brief summary of forecast weights
                min_n = int(forecast_ext["n_versions"].min())
                max_n = int(forecast_ext["n_versions"].max())
                avg_w = float(forecast_ext["weight"].mean())
                self._log_operation(
                    f"[ForecastExtension] points={n_fc}, n_versions(min/max)={min_n}/{max_n}, avg_weight={avg_w:.2f}"
                )
        else:
            self._log_operation(
                "Warning: No actual future prices and no forecast-extension points available for average calculation "
                f"from {now_utc.isoformat()} to {avg_end_utc.isoformat()}. Using current price as avg_future_price."
            )

        # PRICE SPIKE DETECTION FOR CONTROL LOGIC
        # Pull thresholds from control_params (driven by UI), with config fallbacks
        
        from utils.price_spike_detector import PriceSpikeDetector, SpikeDetectorConfig

        # Window days from config.WINDOW, e.g. "3D" -> 3.0
        default_window_days = 3.0
        try:
            if isinstance(config.WINDOW, str) and config.WINDOW.endswith("D"):
                default_window_days = float(config.WINDOW[:-1])
        except Exception:
            default_window_days = 3.0

        spike_window_days = float(
            control_params.get("SPIKE_WINDOW_DAYS", default_window_days)
        )
        spike_z_thr = float(
            control_params.get("SPIKE_Z_THRESHOLD", getattr(config, "Z_THRESHOLD", 1.0))
        )
        spike_pct_thr = float(
            control_params.get("SPIKE_PCT_THRESHOLD", getattr(config, "PCT_THRESHOLD", 0.006))
        )
        spike_abs_min = float(
            control_params.get("SPIKE_ABS_MIN_PRICE", getattr(config, "ABS_MIN_PRICE", 5.0))
        )
        min_pts = int(
            control_params.get("SPIKE_MIN_WINDOW_POINTS", getattr(config, "MIN_WINDOW_POINTS", 40))
        )

        
        # Convert days → rolling window string like "3D"
        spike_window_str = f"{max(1, int(spike_window_days))}D"

        history_for_spikes = full_eprice_df.copy()

        detector = PriceSpikeDetector(
            SpikeDetectorConfig(
                time_col=config.TIME_COLUMN,
                price_col=config.TARGET_COLUMN,
                window=spike_window_str,
                z_threshold=spike_z_thr,
                pct_threshold=spike_pct_thr,
                abs_min_price=spike_abs_min,
                min_window_points=min_pts,
            ),
            logger=self._log_operation,
        )

        spike_df = detector.detect_spikes(history_for_spikes)

        # Last point = "now" (or last known price)
        last_row = spike_df.iloc[-1]
        is_price_spike_now = bool(last_row["is_spike"])
        severity = str(last_row.get("severity", "none"))

        self.last_spike_info = {
            "datetime": last_row[config.TIME_COLUMN],
            "eprice_15min": float(last_row[config.TARGET_COLUMN]),
            "robust_z": float(last_row.get("robust_z", float("nan"))),
            "pct_jump": float(last_row.get("pct_jump", float("nan"))),
            "is_spike": is_price_spike_now,
            "severity": severity,
        }

        if is_price_spike_now:
            self._log_operation(
                f"[SpikeGuard] ⚠️ {severity.upper()} price spike at {self.last_spike_info['datetime']}: "
                f"{self.last_spike_info['eprice_15min']:.3f} snt/kWh "
                f"(z={self.last_spike_info['robust_z']:.2f}, "
                f"jump={self.last_spike_info['pct_jump']:.2%})"
            )
        else:
            self._log_operation("[SpikeGuard] No spike at current interval.")

        # Spike guard only for STRONG spikes (mild = just informational)
        apply_spike_guard = (
            self.spike_guard_enabled
            and not self.spike_manual_override
            and is_price_spike_now
            and severity == "strong"
        )

        temp_solar = live_temps.get("temp_outdoor")
        temp_tank_lower = live_temps.get("temp_tank_lower")
        temp_tank_upper = live_temps.get("temp_tank_upper")
        temp_room = live_temps.get("temp_room_main")


        signals = {
            "solar_circulation": False,
            "gshp": False,
            "heater_element": False,
            "circ_pump": False,
        }

        # PRICE CEILING OVERRIDE — keep circulation ON, turn others OFF if current price >= upper limit
        upper_limit = control_params.get(
            "EPRICE_UPPER_LIMIT", getattr(config, "DEFAULT_EPRICE_UPPER_LIMIT", None)
        )
        try:
            upper_limit = float(upper_limit) if upper_limit is not None else None
        except Exception:
            upper_limit = None

        if (upper_limit is not None) and (float(eprice_now) >= upper_limit):
            self._log_operation(
                f"💸 Price ceiling active: eprice_now={float(eprice_now):.2f} ≥ limit {upper_limit:.2f}. "
                "Forcing heating OFF but keeping circulation ON."
            )
            # Turn heating OFF
            signals["gshp"] = False
            signals["heater_element"] = False
            # Keep main room circulation ON
            signals["circ_pump"] = True
            # Preserve current solar circulation state to avoid unnecessary toggling
            signals["solar_circulation"] = self.latest_states.get("solar_circulation", False)

            for actuator, state in signals.items():
                self._publish_command(actuator, "ON" if state else "OFF")

            self.latest_states = signals.copy()
            return signals

        # HARD SAFETY: Max tank temperature override (circulation only if room needs heat)
        if self._enforce_max_temperature_limit(
            temp_tank_upper=temp_tank_upper,
            live_temps=live_temps,
            control_params=control_params,
            signals=signals,
        ):
            # Publish and exit early: this overrides ROC & normal logic
            for actuator, state in signals.items():
                self._publish_command(actuator, "ON" if state else "OFF")
            self.latest_states = signals.copy()
            return signals

        # Apply ROC safety BEFORE normal logic; short-circuit if engaged
        if self._check_and_apply_roc_safety(live_temps, signals):
            self._log_operation("[ROC] Safety forcing is active. Skipping normal control for this cycle.")
            for actuator, state in signals.items():
                self._publish_command(actuator, "ON" if state else "OFF")
            self.latest_states = signals.copy()
            return signals

        # Room ROC comfort guard — overrides price logic (but NOT max tank / price ceiling)
        if self._check_and_apply_room_roc_comfort(live_temps, control_params, signals):
            self._log_operation("[ROC-ROOM] Comfort guard active. Skipping price-based control for this cycle.")
            for actuator, state in signals.items():
                self._publish_command(actuator, "ON" if state else "OFF")
            self.latest_states = signals.copy()
            return signals

        # Outdoor ROC preheat guard — overrides price logic if cold front incoming
        if self._check_and_apply_outdoor_roc_preheat(live_temps, control_params, signals):
            self._log_operation("[ROC-OUTDOOR] Preheat guard active. Skipping price-based control for this cycle.")
            for actuator, state in signals.items():
                self._publish_command(actuator, "ON" if state else "OFF")
            self.latest_states = signals.copy()
            return signals

        now = pd.Timestamp.now(tz='UTC')
        self._log_operation("\n=== ACTUATOR DECISION DEBUG LOG ===")
        self._log_operation(f"Timestamp: {now}")
        self._log_operation("\n--- Sensor Inputs ---")
        self._log_operation(f"temp_outdoor: {temp_solar} °C")
        self._log_operation(f"temp_tank_lower: {temp_tank_lower} °C")
        self._log_operation(f"temp_tank_upper: {temp_tank_upper} °C")
        self._log_operation(f"temp_room_main: {temp_room} °C")

        self._log_operation("\n--- Control Thresholds ---")
        for k, v in control_params.items():
            self._log_operation(f"{k}: {v}")

        self._log_operation("\n--- Price Info ---")
        self._log_operation(f"Current Price: {eprice_now:.2f} snt/kWh")
        self._log_operation(f"Avg Future Price (actual + forecast extension): {avg_future_price:.2f} snt/kWh")

        if temp_tank_upper is not None:
            tank_heat_needed = temp_tank_upper < (
                control_params["TANK_TARGET_TEMP_UPPER"] - control_params["TANK_HYSTERESIS"]
            )
            tank_critically_low = temp_tank_upper < control_params["TANK_CRITICAL_TEMP_UPPER"]

            self._log_operation("\n--- Decision Conditions ---")
            self._log_operation(
                "Tank heat needed? "
                f"{'YES' if tank_heat_needed else 'NO'} "
                f"(Upper={temp_tank_upper} < Target-Hyst="
                f"{control_params['TANK_TARGET_TEMP_UPPER'] - control_params['TANK_HYSTERESIS']})"
            )
            self._log_operation(
                "Tank critically low? "
                f"{'YES' if tank_critically_low else 'NO'} "
                f"(Upper={temp_tank_upper} < Critical={control_params['TANK_CRITICAL_TEMP_UPPER']})"
            )

            if tank_heat_needed and (eprice_now < avg_future_price):
                self._log_operation("Decision: Heat needed and current price is good. Turning on GSHP.")
                signals["gshp"] = True

            if tank_critically_low or eprice_now < control_params["EPRICE_VERY_LOW_THRESHOLD"]:
                self._log_operation(
                    "Decision: Tank is critically low or electricity is very cheap. Turning on Heater Element."
                )
                signals["heater_element"] = True

            if temp_solar is not None and temp_tank_lower is not None:
                delta = temp_solar - temp_tank_lower
                self._log_operation(
                    f"Solar delta: {delta:.2f} °C → "
                    f"{'ON' if delta >= control_params['SOLAR_DELTA_T_ON'] else 'OFF'} "
                    f"(On>{control_params['SOLAR_DELTA_T_ON']})"
                )
                if delta >= control_params["SOLAR_DELTA_T_ON"]:
                    signals["solar_circulation"] = True
                elif delta <= control_params["SOLAR_DELTA_T_OFF"]:
                    signals["solar_circulation"] = False
                else:
                    signals["solar_circulation"] = self.latest_states.get("solar_circulation", False)

        if temp_room is not None:
            if temp_room < (control_params["ROOM_TARGET_TEMP"] - control_params["ROOM_HYSTERESIS"]):
                signals["circ_pump"] = True

        # PRICE SPIKE GUARD: kill heating but keep circulation logic
        if apply_spike_guard:
            self._log_operation(
                "[SpikeGuard] Spike guard ACTIVE (strong spike) → forcing GSHP + Heater OFF, "
                "keeping circulation/solar as decided by normal logic."
            )
            signals["gshp"] = False
            signals["heater_element"] = False
            

        # Log ON→OFF actuator cycles for analytics
        try:
            self.cycle_logger.update_cycles(
                now_utc=pd.Timestamp.now(tz="UTC"),
                live_temps=live_temps,
                new_signals=signals,
            )
        except Exception as e:
            self._log_operation(f"[CycleLogger] Error updating heating cycles: {e}")

        for actuator, state in signals.items():
            self._publish_command(actuator, "ON" if state else "OFF")

        self.latest_states = signals.copy()
        return signals
    
