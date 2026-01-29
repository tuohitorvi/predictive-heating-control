#include <Arduino.h>

// ========================================================================
// CODE FOR D1_MINI (ESP8266)
// ========================================================================
#if defined(ARDUINO_ARCH_ESP8266)

// --- D1mini libraries ---
#include <ESP8266WiFi.h>
#include <PubSubClient.h>
#include <SoftwareSerial.h>
#include <ArduinoJson.h>
#include <time.h>
#include "private_credentials.h"
#include <ESP8266Ping.h> 


// --- D1mini definitions and globals ---
const char* MQTT_READ_REQUEST_TOPIC = "sensors/temperature/+/read";
// const int   MQTT_PORT = 1884;
// Base topic for publishing sensor data
const char* MQTT_DATA_TOPIC_BASE = "sensors/temperature/";

// Topics for actuator commands 
const char* MQTT_COMMAND_TOPIC = "actuators/+/set";
const char* MQTT_MODE_TOPIC = "actuators/mode";
const char* MQTT_ACK_TOPIC_BASE = "actuators/";
const char* MQTT_STATUS_TOPIC = "actuators/status";

const char* SENSOR_IDS[] = {"temp1", "temp2", "temp3", "temp4"};
const int SENSOR_COUNT = 4;

// SoftwareSerial pins
#define STM_RX_PIN D7
#define STM_TX_PIN D8

// 2. GLOBAL OBJECTS
WiFiClient espClient;
PubSubClient client(espClient);
SoftwareSerial stmSerial(STM_RX_PIN, STM_TX_PIN);

String control_mode = "auto";


struct ActuatorState {
String id;
int pin;
bool state;
};


ActuatorState actuators[] = {
{"out1", D3, false},
{"out2", D4, false},
{"out3", D5, false},
{"out4", D6, false}
};
const int ACTUATOR_COUNT = sizeof(actuators) / sizeof(ActuatorState);


void publish_actuator_status(const String& id, bool state) {
  char topic[64];
  snprintf(topic, sizeof(topic), "%s%s/status", MQTT_ACK_TOPIC_BASE, id.c_str());
  StaticJsonDocument<128> doc;
  doc["state"] = state ? "on" : "off";
  doc["mode"] = control_mode;
  char buffer[128];
  serializeJson(doc, buffer);
  client.publish(topic, buffer, true); // Retained message
}


void set_actuator(const String& id, const String& stateStr) {
  bool state = (stateStr == "on");
  for (int i = 0; i < ACTUATOR_COUNT; i++) {
    if (actuators[i].id == id) {
      pinMode(actuators[i].pin, OUTPUT);
      digitalWrite(actuators[i].pin, state ? HIGH : LOW); // Active-high
      actuators[i].state = state;
      publish_actuator_status(id, state);
      Serial.printf("STM32: Set %s to %s (mode: %s)\n", id.c_str(), stateStr.c_str(), control_mode.c_str());
    }
  }
}


void sync_all_actuators() {
  for (int i = 0; i < ACTUATOR_COUNT; i++) {
    pinMode(actuators[i].pin, OUTPUT);
    digitalWrite(actuators[i].pin, actuators[i].state ? HIGH : LOW);
    publish_actuator_status(actuators[i].id, actuators[i].state);
  }
}

// The callback function that handles read requests
void callback(char* topic, byte* payload, unsigned int length) {
  String topicStr = String(topic);
  payload[length] = '\0';
  String message = String((char*)payload);

  Serial.printf("Message arrived [%s]: %s\n", topic, message.c_str());
  

  // Check if this is a sensor read request
  if (topicStr.endsWith("/read")) {
  topicStr.replace("sensors/temperature/", "");
  topicStr.replace("/read", "");
  String sensorId = topicStr;
  for (int i = 0; i < SENSOR_COUNT; i++) {
    if (sensorId.equals(SENSOR_IDS[i])) {
      Serial.printf("⚙️ Sending to STM32: READ:%d\n", i);
      stmSerial.printf("READ:%d\n", i);
      break;
    }
  }
} else if (topicStr == MQTT_MODE_TOPIC) {
  if (message == "manual" || message == "auto") {
    control_mode = message;
    Serial.printf("Control mode changed to: %s\n", control_mode.c_str());
    sync_all_actuators();
  }
} else if (topicStr.startsWith("actuators/") && topicStr.endsWith("/set")) {
  if (control_mode == "manual") {
    String id = topicStr.substring(10, topicStr.length() - 4); // Extract "out1" from "actuators/out1/set"
    set_actuator(id, message);
  } else {
    Serial.println("Ignoring manual command: System in AUTO mode.");
  }
}
}

void reconnect() {
  while (!client.connected()) {
    Serial.printf("Attempting MQTT connection... WiFi status=%d, IP=%s, broker=%s:%d\n",
                  WiFi.status(),
                  WiFi.localIP().toString().c_str(),
                  MQTT_SERVER_IP,
                  MQTT_PORT);

    if (client.connect("d1_mini_Temp_Gateway")) {
      Serial.println("connected");
      client.subscribe(MQTT_READ_REQUEST_TOPIC);
      client.subscribe(MQTT_COMMAND_TOPIC);
      client.subscribe(MQTT_MODE_TOPIC);
    } else {
      Serial.printf("failed, rc=%d. Retrying in 5s...\n", client.state());
      delay(5000);
    }
  }
}

void setup_wifi() {
  delay(10);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
}

void setup() {
  Serial.begin(9600);// PC monitor
  stmSerial.begin(9600);//UART (STM32 Serial1)


  setup_wifi();


  IPAddress brokerIp;
  brokerIp.fromString(MQTT_SERVER_IP);
  if (Ping.ping(brokerIp)) Serial.println("✅ Broker reachable.");
  else Serial.println("❌ Broker unreachable!");


  configTime(0, 0, "pool.ntp.org", "time.nist.gov");
  while (time(nullptr) < 1510644967) delay(500);

  Serial.printf("Broker: %s:%d\n", MQTT_SERVER_IP, MQTT_PORT);
  Serial.println();
  Serial.println("=== MQTT DEBUG ===");
  Serial.printf("WiFi SSID: %s\n", WIFI_SSID);
  Serial.printf("WiFi status: %d (3=CONNECTED)\n", WiFi.status());
  Serial.printf("ESP IP: %s\n", WiFi.localIP().toString().c_str());
  Serial.printf("ESP GW: %s\n", WiFi.gatewayIP().toString().c_str());
  Serial.printf("Broker IP: %s\n", MQTT_SERVER_IP);
  Serial.printf("Broker port: %d\n", MQTT_PORT);
  Serial.println("==================");
  client.setServer(MQTT_SERVER_IP, MQTT_PORT);
  client.setCallback(callback);

  // MQTT robustness
  client.setBufferSize(512);
  client.setSocketTimeout(5);
}

void loop() {
  delay(1000);
  if (!client.connected()) reconnect();
  client.loop();


  if (stmSerial.available()) {
    String line = stmSerial.readStringUntil('\n');
    line.trim();
    int colonIndex = line.indexOf(':');
    if (colonIndex != -1) {
      int sensorIndex = line.substring(0, colonIndex).toInt();
      String tempValueStr = line.substring(colonIndex + 1);


      if (tempValueStr != "error" && sensorIndex >= 0 && sensorIndex < SENSOR_COUNT) {
        float temperature = tempValueStr.toFloat();
        char dataTopic[50];
        snprintf(dataTopic, 50, "sensors/temperature/%s", SENSOR_IDS[sensorIndex]);


        StaticJsonDocument<128> doc;
        doc["temperature"] = temperature;
        doc["timestamp"] = time(nullptr);
        char jsonBuffer[128];
        serializeJson(doc, jsonBuffer);
        client.publish(dataTopic, jsonBuffer);
      }
    }
  }
}

#endif // ARDUINO_ARCH_ESP8266

// ========================================================================
// CODE FOR NUCLEO (STM32)
// ========================================================================

#if defined(ARDUINO_ARCH_STM32)

// --- Nucleo libraries ---
#include <OneWire.h>
#include <DallasTemperature.h>

// --- Nucleo definitions and globals ---
// 1. PIN CONFIGURATION
#define ONE_WIRE_BUS PA8 // Pin D7 on Nucleo board for sensors
#define COMM_SERIAL Serial1 // Using Serial1 on pins D8 (TX) and D2 (RX)

// 2. Globals and constants
// --- Sensor + filtering configuration ---
static const uint8_t NUM_SENSORS        = 4;      // using 4 DS18B20 temperature sensors
static const uint8_t MEDIAN_WINDOW     = 5;      // size of median window
static const float   EMA_ALPHA         = 0.3f;   // smoothing factor
static const unsigned long SAMPLE_MS   = 1000;   // ~1 second

// History buffers for median filter
float tempHistory[NUM_SENSORS][MEDIAN_WINDOW];
uint8_t histCount[NUM_SENSORS]  = {0};
uint8_t histIndex[NUM_SENSORS]  = {0};

// EMA state
float emaValue[NUM_SENSORS]         = {0.0f};
bool  emaInitialized[NUM_SENSORS]   = {false};

// Last filtered value that will be returned to ESP on READ:x
float lastFilteredValue[NUM_SENSORS];
bool  hasFilteredValue[NUM_SENSORS] = {false};

unsigned long lastSampleMillis = 0;

// --- Helper function forward declarations ---
bool  isValidRawTemp(float t);
float computeMedianForSensor(uint8_t sensorIdx);
float filterSample(uint8_t sensorIdx, float raw);


// 3. GLOBAL OBJECTS
OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature sensors(&oneWire);
DeviceAddress sensorAddresses[NUM_SENSORS];

// 4. SETUP FUNCTION
void setup() {
    Serial.begin(115200);// PC debug console
    Serial.println("--- Nucleo Booted: Serial OK ---");
    // Start the serial port for communication with D1mini
    COMM_SERIAL.begin(9600); //UART
    Serial.println("--- Nucleo Booted ---");
    Serial.println("Initializing sensors...");
    sensors.begin();
    if (sensors.getDeviceCount() < NUM_SENSORS) {
      COMM_SERIAL.println("STM32: ERROR! Not all sensors found. Halting.");
      while (true);
    }
    for (int i = 0; i < NUM_SENSORS; i++) {
      if (!sensors.getAddress(sensorAddresses[i], i)) {
        Serial.print("STM32: ERROR! Sensor address missing for index ");
        Serial.println(i);
      }
    }
    Serial.println("--- Setup Complete ---");
}

// 5. LOOP FUNCTION
void loop() {
  // 1) Periodic background sampling of all sensors
  unsigned long now = millis();
  if (now - lastSampleMillis >= SAMPLE_MS) {
    lastSampleMillis = now;

    // Single conversion for all sensors once per SAMPLE_MS
    sensors.requestTemperatures();

    for (uint8_t i = 0; i < NUM_SENSORS; ++i) {
      float rawC = sensors.getTempC(sensorAddresses[i]);

      // Run through 3-layer filter: reject → median → EMA
      float filtered = filterSample(i, rawC);

      // If filter produced a usable value, store/update it
      if (!isnan(filtered)) {
        lastFilteredValue[i]   = filtered;
        hasFilteredValue[i]    = true;   // redundant with filterSample, but safe
      }
    }
  }

  // 2) Handle serial READ commands from ESP (instant response)
  if (COMM_SERIAL.available()) {
    String command = COMM_SERIAL.readStringUntil('\n');
    command.trim();

    if (command.startsWith("READ:")) {
      int sensorIndex = command.substring(5).toInt();

      if (sensorIndex >= 0 && sensorIndex < NUM_SENSORS) {
        String payload;

        if (hasFilteredValue[sensorIndex]) {
          float tempC = lastFilteredValue[sensorIndex];

          if (!isnan(tempC)) {
            payload = String(sensorIndex) + ":" + String(tempC);
          } else {
            payload = String(sensorIndex) + ":error";
          }
        } else {
          // No valid filtered data yet for this sensor
          payload = String(sensorIndex) + ":error";
        }

        COMM_SERIAL.println(payload);
      }
    }
  }
}
// 6. FILTERING HELPER FUNCTIONS
// Layer 1: reject obvious garbage / out-of-range
bool isValidRawTemp(float t)
{
    // DS18B20 error codes:
    if (t == -127.0f || t == 85.0f) return false;

    // basic physical sanity range
    if (t < -40.0f || t > 125.0f) return false;

    return true;
}

// Compute median of the N most recent values in history
float computeMedianForSensor(uint8_t sensorIdx)
{
    uint8_t n = histCount[sensorIdx];
    if (n == 0) return NAN;

    float tmp[MEDIAN_WINDOW];

    for (uint8_t i = 0; i < n; ++i) {
        tmp[i] = tempHistory[sensorIdx][i];
    }

    // Simple insertion sort (n <= 5)
    for (uint8_t i = 1; i < n; ++i) {
        float key = tmp[i];
        int8_t j = i - 1;
        while (j >= 0 && tmp[j] > key) {
            tmp[j + 1] = tmp[j];
            j--;
        }
        tmp[j + 1] = key;
    }

    // median
    return tmp[n / 2];
}

// Single-sample filter pipeline: Reject → Median → EMA
float filterSample(uint8_t sensorIdx, float raw)
{
    // Layer 1: reject invalid
    if (!isValidRawTemp(raw)) {
        // if we already have an EMA, keep it; else this sample is unusable
        if (hasFilteredValue[sensorIdx]) {
            return emaValue[sensorIdx];
        } else {
            return NAN;
        }
    }

    // Layer 2: update ring buffer and compute median
    tempHistory[sensorIdx][histIndex[sensorIdx]] = raw;

    if (histCount[sensorIdx] < MEDIAN_WINDOW) {
        histCount[sensorIdx]++;
    }

    histIndex[sensorIdx] = (histIndex[sensorIdx] + 1) % MEDIAN_WINDOW;

    float median = computeMedianForSensor(sensorIdx);
    if (isnan(median)) {
        // fallback to raw if something weird happens
        median = raw;
    }

    // Layer 3: EMA smoothing
    if (!emaInitialized[sensorIdx]) {
        emaValue[sensorIdx]        = median;
        emaInitialized[sensorIdx]  = true;
    } else {
        emaValue[sensorIdx] = EMA_ALPHA * median + (1.0f - EMA_ALPHA) * emaValue[sensorIdx];
    }

    hasFilteredValue[sensorIdx] = true;
    return emaValue[sensorIdx];
}

#endif // ARDUINO_ARCH_STM32