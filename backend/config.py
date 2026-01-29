import os
from datetime import datetime
import tensorflow as tf


# --- Data Configuration ---
DATA_DIR = "data"
FORECASTING_DB_PATH = os.path.join(DATA_DIR, "forecasting_data.db")
DB_PATH = FORECASTING_DB_PATH 
SENSOR_DB_PATH = os.path.join(DATA_DIR, "sensor_log.db")
HEATING_CYCLES_DB_PATH = FORECASTING_DB_PATH
SAVE_FORECASTING_PLOT_DIR = 'data/saved_forecasting_plots'
GROUND_TRUTH_TABLE = "ground_truth_table"
ALV = 0.255
LOCAL_TIME_ZONE = 'Europe/Helsinki'

# --- MQTT BROKER Configuration ---
MQTT_BROKER_IP = "localhost" 
MQTT_BROKER_PORT = 1884 

# --- SENSOR SYSTEM Configuration ---
SENSOR_KEY_MAP = {
    "temp1": "temp_outdoor",
    "temp2": "temp_tank_lower",
    "temp3": "temp_tank_upper",
    "temp4": "temp_supply",
}


# --- Feature & Target Configuration ---
# List of all feature columns expected by the LSTM model in order
LSTM_FEATURE_COLUMNS = [
    'hour_extracted',
    'minute_extracted',
    'temp',
    'windEnergy',
    'eprice_lag1',
    'is_weekend',
    'weekday'
]

TARGET_COLUMN = 'eprice_15min'
TIME_COLUMN = 'datetime'
TEMPERATURE_COL_NAME = "temp"
FINGRID_COL_NAME = "windEnergy"


# --- MODEL AND SCALER Configuration ---

# Base Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_BASE_DIR = os.path.join(BASE_DIR, "models_artifacts")
SCALER_BASE_DIR = os.path.join(BASE_DIR, "scaler_artifacts")
PLOTS_BASE_DIR = os.path.join(BASE_DIR, "plots")

# Versioned Subdirectories
MODEL_VERSION_DIR = os.path.join(MODEL_BASE_DIR, "versions")
BASE_SCALER_DIR = os.path.join(SCALER_BASE_DIR, "versions")
FORECAST_PLOTS_DIR = os.path.join(PLOTS_BASE_DIR, "forecasts")


# Specific scaler names to be used when to store them independently
MM_SCALER_NAME = "mm_scaler.pkl"
R_SCALER_NAME = "r_scaler.pkl"
TARGET_SCALER_NAME = "target_scaler.pkl"

# Mapping of columns to which scaler applies
MM_SCALED_COLUMNS = ['hour_extracted', 'minute_extracted', 'temp', 'is_weekend', 'weekday']
R_SCALED_COLUMNS = ['windEnergy', 'eprice_lag1']

# --- Model & Training Hyperparameters ---

# Initial date for "train from scratch" 
INITIAL_TRAINING_START_DATE = "2025-10-01T00:00:00Z" 
HISTORY_DAYS_FOR_FORECAST = 7
SEQUENCE_LENGTH = 672         
FORECAST_HORIZON = 24 # for training
BATCH_SIZE = 32 
EPOCHS = 100
SEED = 42
SPLIT_RATIO = 0.8
EARLY_STOPPING_PATIENCE = 10
HOURS_FOR_INTERPRETATION_HISTORY = 24


# --- ACTUATOR CONTROL Configuration ---

CONTROL_PARAMETERS = {
    # Solar Circulation Logic
    "SOLAR_DELTA_T_ON": 5.0,  # Turn on if solar is 5°C hotter than tank bottom
    "SOLAR_DELTA_T_OFF": 2.0, # Turn off if difference drops to 2°C

    # Main Tank Heating Logic
    "TANK_TARGET_TEMP_UPPER": 65.0, # Target temperature for the top of the tank
    "TANK_HYSTERESIS": 5.0,         # Allow temp to drop 5°C below target before reheating

    # GSHP (Ground Source Heat Pump) Logic
    "EPRICE_LOW_THRESHOLD": 5.0, 

    # Heater Element (Backup/Opportunistic) Logic
    "TANK_CRITICAL_TEMP_UPPER": 30.0, # Turn on backup element if temp drops this low
    "EPRICE_VERY_LOW_THRESHOLD": 2.0, # Use cheap electricity even if tank is hot
    "TANK_TEMP_UPPER_MAX": 90, # Stop heating if the temp is reached

    # Room Circulation Pump Logic
    "ROOM_TARGET_TEMP": 21.0,
    "ROOM_HYSTERESIS": 0.5,
}

# Actuator Control Cycle Settings
ACTUATOR_CONTROL_INTERVAL_SECONDS = 300  # e.g., every 5 minutes (300 seconds)
DEFAULT_EPRICE_UPPER_LIMIT = 10.0        # Default upper price limit in snt/kWh

# Tank temperature ROC safety (example values) ---
TANK_ROC_WINDOW_SECONDS = 300         # how far apart samples must be to evaluate (e.g., 5 min)
TANK_ROC_THRESHOLD_C_PER_SEC = -0.003 # trip when dT/dt is lower (more negative) than this (e.g., <-0.003 °C/s = -0.18 °C/min)
TANK_ROC_MIN_DELTA_C = 0.2            # ignore noise; require at least this absolute delta °C before considering

# For room comfort ROC
ROOM_ROC_WINDOW_SECONDS = 600        # 10 minutes
ROOM_ROC_COOL_THRESHOLD_C_PER_SEC = -0.0005  # ≈ -0.03 °C/min
ROOM_ROC_MIN_DELTA_C = 0.1

# For outdoor preheat ROC
OUTDOOR_ROC_WINDOW_SECONDS = 900     # 15 minutes
OUTDOOR_ROC_COOL_THRESHOLD_C_PER_SEC = -0.001  # ≈ -0.06 °C/min
OUTDOOR_ROC_MIN_DELTA_C = 0.2


# --- SPIKE DETECTOR variables ---
WINDOW = "3D"                       # rolling window for baseline (e.g. "1D", "2D", "48H"), 3D
MIN_WINDOW_POINTS = 40              # minimum number of points required in window, 40
Z_THRESHOLD = 3.0                   # Require larger deviation from median baseline (large deviations → catches only large anomalies):, 1.0
PCT_THRESHOLD = 0.20                 # required relative jump vs previous point (20%), 0.006
ABS_MIN_PRICE = 5.0                 # ignore spikes when absolute price is tiny (< 5 snt), 5.0

# --- UTILITIES FOR PATHS AND VERSIONING ---

def get_version_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def get_versioned_model_path(version_id):
    _version_id = version_id if version_id else get_version_id()
    return os.path.join(MODEL_VERSION_DIR, f"model_{_version_id}.keras")

def get_versioned_scaler_dir(version_id):
    return os.path.join(BASE_SCALER_DIR, version_id)

def get_versioned_scaler_path(version_id, scaler_name):
    """Returns the full path to a specific scaler file for a given version."""
    return os.path.join(get_versioned_scaler_dir(version_id), scaler_name)

def get_forecast_plot_path(version_id):
    return os.path.join(FORECAST_PLOTS_DIR, f"forecast_plot_{version_id}.png")

# Ensure all necessary directories exist 
os.makedirs(MODEL_VERSION_DIR, exist_ok=True)
os.makedirs(BASE_SCALER_DIR, exist_ok=True)
os.makedirs(FORECAST_PLOTS_DIR, exist_ok=True)

SCALER_VERSION_DIR = BASE_SCALER_DIR