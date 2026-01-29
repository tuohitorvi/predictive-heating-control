# MQTT Topic Map

This document summarizes MQTT topics by **component role** and **message direction**.  
Broker Environment: Native Windows (Mosquitto), `MQTT_BROKER_PORT=1884`.

## 1. Broker Settings

Updated for the transition from WSL to Windows:
- Broker IP (Host/Python): localhost (or 127.0.0.1)[1]
- Broker IP (ESP8266/Edge): <YOUR_WINDOWS_LAN_IP> 
- Default Port: 1884 
Crucial Note: Ensure mosquitto.conf contains the following lines to allow the ESP8266 to connect:

Conf
  listener 1884 0.0.0.0
  allow_anonymous true

Remember to allow the port 1884 through the Firewall


## 2. Components

### Host (Python runtime)
- `SensorMonitorAgent` (in `backend/sensor_monitor_agent.py`)
- `ActuatorControlAgent` (in `backend/actuator_control_agent.py`)
- `OrchestratingAgent` coordinates cycles (in `backend/orchestrating_agent.py`)

### Edge (ESP8266 D1 Mini)
- UART gateway for STM32 Nucleo sensor reads
- MQTT bridge for sensor telemetry
- Actuator controller for `out1..out4`
- Implements manual/auto mode behavior (`src/main.cpp`, ESP8266 section)


## 3. Sensor topics

### 3.1 Read request topics (Host → D1 Mini)
**Purpose:** Trigger a sensor read on demand or via schedule.

- **Topic pattern:**  
  `sensors/temperature/{temp1|temp2|temp3|temp4}/read`
- **Published by:** Host (`SensorMonitorAgent`)
- **Subscribed by:** D1 Mini
- **Payload:** Empty string ("")

**Example:**
- `sensors/temperature/temp3/read`

**Notes:**
- Host sends these periodically based on `SENSOR_SCHEDULE_MINUTES`.
- D1 Mini maps `{tempX}` to an index and forwards a UART request: `READ:i`.

---

### 3.2 Sensor telemetry topics (D1 Mini → Host)
**Purpose:** Deliver measured temperatures to the host for logging and control.

- **Topic pattern:**  
  `sensors/temperature/{temp1|temp2|temp3|temp4}`
- **Published by:** D1 Mini
- **Subscribed by:** Host (`SensorMonitorAgent`)
- **Payload (JSON):**
  ```json
  {"temperature": 21.5, "timestamp": 1730000000}

## 4. Actuator & Control Topics

### 4.1 Actuator Command (Host → D1 Mini)
**Purpose:** Set the state of pins out1 through out4.
- **Topic pattern:**
  actuators/output/{out1|out2|out3|out4}/set
- **Published by:** Host (`ActuatorControlAgent`)
- **Payload:** "ON" or "OFF"

### 4.2 Mode Control (Host → D1 Mini)
**Purpose:** Switch the ESP8266 between Manual and Auto logic.
- **Topic:** system/mode/set
- **Payload:** "MANUAL" or "AUTO"

**Notes:**
- In AUTO, the ESP8266 may trigger actuators based on local thresholds if UART communication fails.