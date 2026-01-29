# backend/sensor_monitor_agent.py
import os
import json
import time
import threading
import sqlite3
import paho.mqtt.client as mqtt
from typing import Optional, Dict

from . import config

# Reading schedule for temperature sensors (min)
SENSOR_SCHEDULE_MINUTES = {
    "temp1": 60,
    "temp2": 5,
    "temp3": 1,
    "temp4": 5,
}

class SensorMonitorAgent:
    def __init__(self, db_path: str, logger = None):
        self.logger = logger if logger else print
        self.logger("SensorMonitorAgent initialized.")
        self.db_path = db_path
        self.latest_sensor_data = {}
        self.on_demand_requests = {}  # { sensor_id: {event, result} }
        self.lock = threading.Lock()
        
        self._init_db()
        self.client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION1, protocol=mqtt.MQTTv311)
        # Set the wrappers as callbacks 
        self.client.on_connect = self._on_connect_wrapper
        self.client.on_message = self._on_message_wrapper
        self.client.user_data_set(self) # Ensure 'self' is passed as userdata
        
    
    def _init_db(self):
        db_dir = os.path.dirname(self.db_path)
        os.makedirs(db_dir, exist_ok=True)
        self.logger(f"SensorMonitor: Initializing database at {self.db_path}")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sensor_readings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, sensor_key TEXT,
                    sensor_id TEXT, timestamp REAL, value REAL)""")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, sensor_key TEXT,
                    timestamp REAL, value REAL, message TEXT)""")
        

    def _log_to_db(self, sensor_key, sensor_id, timestamp, value):
        self.logger(f"💾 Logging {sensor_key} = {value} @ {timestamp}")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT INTO sensor_readings (sensor_key, sensor_id, timestamp, value) VALUES (?, ?, ?, ?)",
                         (sensor_key, sensor_id, timestamp, value))
            conn.commit()

    @staticmethod
    def _on_connect_static(client, userdata, flags, rc):
        # userdata is the SensorMonitorAgent instance because of client.user_data_set(self)
        userdata.logger(f"✅ SensorMonitor: Connected to MQTT broker with result code {rc}")
        client.subscribe("sensors/temperature/#")

    # Wrapper for static _on_connect to allow passing self.logger
    def _on_connect_wrapper(self, client, userdata, flags, rc):
        # Receives the arguments from paho-mqtt
        # -> calls the static method, passing the userdata (which is 'self' for SensorMonitorAgent)
        SensorMonitorAgent._on_connect_static(client, userdata, flags, rc)

    @staticmethod
    def _on_message_static(client, userdata, msg):
        agent_instance = userdata
        agent_instance.logger(f"💬 MQTT message received on topic {msg.topic}: {msg.payload.decode()}")

        if msg.topic.endswith("/read"):
            return

        try:
            payload = json.loads(msg.payload.decode())
            sensor_id = msg.topic.split("/")[-1]
            sensor_key = config.SENSOR_KEY_MAP.get(sensor_id)
            if not sensor_key:
                return

            timestamp = payload.get("timestamp", time.time())
            value = payload["temperature"]

            with agent_instance.lock:
                agent_instance.latest_sensor_data[sensor_key] = {"value": value, "timestamp": timestamp}

                if sensor_id in agent_instance.on_demand_requests:
                    request_info = agent_instance.on_demand_requests[sensor_id]
                    request_info["result"] = value
                    request_info["event"].set()

            agent_instance._log_to_db(sensor_key, sensor_id, timestamp, value)

        except Exception as e:
            agent_instance.logger(f"⚠️ SensorMonitor: Failed to process MQTT message on topic {msg.topic}: {e}")

    # Wrapper for on_message
    def _on_message_wrapper(self, client, userdata, msg):
        # This method receives the arguments from paho-mqtt
        # -> calls the static method, passing the userdata (which is 'self' for SensorMonitorAgent)
        SensorMonitorAgent._on_message_static(client, userdata, msg) 

    def _polling_thread_target(self, sensor_id: str, interval_minutes: int):
        topic = f"sensors/temperature/{sensor_id}/read"
        interval_seconds = interval_minutes * 60
        while True:
            self.logger(f"⏰ SensorMonitor: Scheduled request for {sensor_id}...")
            self.client.publish(topic, payload="")
            time.sleep(interval_seconds)

    def start(self):
        self.client.connect(config.MQTT_BROKER_IP, config.MQTT_BROKER_PORT, 60)
        self.client.loop_start() # Starts a new thread that calls loop_forever()
        self.logger(f"SensorMonitor: MQTT listener started, connecting to {config.MQTT_BROKER_IP}.")
        time.sleep(2)

        for sensor_id, interval in SENSOR_SCHEDULE_MINUTES.items():
            if sensor_id in config.SENSOR_KEY_MAP:
                polling_thread = threading.Thread(
                    target=self._polling_thread_target,
                    args=(sensor_id, interval),
                    daemon=True
                )
                polling_thread.start()
                self.logger(f"SensorMonitor: Scheduler started for {sensor_id} (interval: {interval} min).")

    def get_on_demand_reading(self, sensor_key: str, timeout: int = 15) -> Optional[float]:
        sensor_id = None
        for sid, skey in config.SENSOR_KEY_MAP.items():
            if skey == sensor_key:
                sensor_id = sid
                break

        if not sensor_id:
            self.logger(f"SensorMonitor: Error - Unknown sensor key '{sensor_key}' for on-demand read.")
            return None

        request_topic = f"sensors/temperature/{sensor_id}/read"
        event = threading.Event()
        request_info = {"event": event, "result": None}

        with self.lock:
            self.on_demand_requests[sensor_id] = request_info

        self.logger(f"📤 Orchestrator -> SensorMonitor: Publishing ON-DEMAND read for {sensor_key} ({sensor_id}) to topic {request_topic}")
        self.client.publish(request_topic, payload="")

        # Slight pause to ensure response setup
        time.sleep(0.25)
        event_was_set = event.wait(timeout=timeout)

        result = None
        with self.lock:
            if event_was_set:
                result = request_info.get("result")
                self.logger(f"✅ SensorMonitor -> Orchestrator: Received on-demand result for {sensor_key}: {result}")
            else:
                self.logger(f"⏱️ SensorMonitor -> Orchestrator: On-demand request for {sensor_key} TIMED OUT after {timeout} seconds.")

            del self.on_demand_requests[sensor_id]

        return result