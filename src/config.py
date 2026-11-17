# src/config.py
import os

# we use the same seed everywhere so we get the same results each time
SEED = 42

# --- columns of the dataset ---
DATE_COL = 'date'
STATION_COL = 'sensor_name'
TARGET_COL = 'pm25'

FEATURE_COLS = [
    'NO2',
    'CO',
    'O3',
    'AOD',
    'temperature_celsius',
    'pressure_mb',
    'wind_u',
    'wind_v'
]

# satellite columns (Sentinel-5P) that have missing values
S5P_COLS = ['NO2', 'CO', 'O3']

# 80% of the dates for training/CV, the last 20% are kept for the final test
TRAIN_RATIO = 0.8
N_FOLDS = 5

# --- paths ---
# we build the paths from the project folder, so it also works when we run the notebooks
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'data.csv')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')   # csv tables
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')   # png plots
FOLDS_DIR = os.path.join(BASE_DIR, 'folds')       # json files with the fold indices
