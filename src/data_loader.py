# src/data_loader.py
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

def load_and_clean_data():
    """Loads the raw csv, prints some info about the dataset and cleans it.

    Returns the clean dataframe and a dict with how many rows we removed
    (we put these numbers in Exploratory_Data.csv).
    """
    print("1. Loading data...")

    # if the file is not there we stop here with a clear message
    # instead of a long pandas error
    if not os.path.exists(config.RAW_DATA_PATH):
        print(f"ERROR: I can't find {config.RAW_DATA_PATH}")
        sys.exit(1)   # exit code 1 so make/the terminal see that it failed

    df = pd.read_csv(config.RAW_DATA_PATH)
    print(f"   Original shape: {df.shape}")

    # quick look at the dataset (section 3.1): size, dates, stations, coordinates
    print(f"   Date range: {df[config.DATE_COL].min()} -> {df[config.DATE_COL].max()}")
    print(f"   Stations: {df[config.STATION_COL].nunique()} names, {df['id'].nunique()} ids")
    print(f"   Latitude: {df['latitude'].min():.2f} to {df['latitude'].max():.2f}, "
          f"Longitude: {df['longitude'].min():.2f} to {df['longitude'].max():.2f}")
    log = {'raw_rows': len(df), 'raw_columns': df.shape[1]}

    # we remove duplicate rows, otherwise the same day could be counted twice
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        df = df.drop_duplicates()
    log['full_row_duplicates_removed'] = int(duplicates)

    # one station should have only one row per day. Some stations got a new id in 2024
    # so a few days appear twice, we keep the first one
    key_dups = df.duplicated(subset=[config.DATE_COL, config.STATION_COL]).sum()
    df = df.drop_duplicates(subset=[config.DATE_COL, config.STATION_COL], keep='first')
    log['date_station_duplicates_removed'] = int(key_dups)
    log['date_lat_lon_duplicates'] = int(df.duplicated(subset=[config.DATE_COL, 'latitude', 'longitude']).sum())
    log['date_id_duplicates'] = int(df.duplicated(subset=[config.DATE_COL, 'id']).sum())
    print(f"   Duplicates removed: {duplicates} full rows, {key_dups} on (date, station)")

    # we remove the rows where PM2.5 is missing, <= 0 or >= 500 (asked in the project,
    # these values are sensor errors). This is done before the EDA and the split.
    initial_len = len(df)
    log['pm25_missing_removed'] = int(df[config.TARGET_COL].isna().sum())
    log['pm25_le_0_removed'] = int((df[config.TARGET_COL] <= 0).sum())
    log['pm25_ge_500_removed'] = int((df[config.TARGET_COL] >= 500).sum())
    df = df.dropna(subset=[config.TARGET_COL])
    df = df[(df[config.TARGET_COL] > 0) & (df[config.TARGET_COL] < 500)]
    print(f"   Removed bad pm25 rows: {initial_len - len(df)} "
          f"(missing: {log['pm25_missing_removed']}, <=0: {log['pm25_le_0_removed']}, "
          f">=500: {log['pm25_ge_500_removed']})")

    # convert the date column from text to real dates
    df[config.DATE_COL] = pd.to_datetime(df[config.DATE_COL])

    # sort by station and date, the lags need this order to take the right day
    df = df.sort_values(by=[config.STATION_COL, config.DATE_COL]).reset_index(drop=True)
    log['clean_rows'] = len(df)

    print(f"   Cleaning done. New shape: {df.shape}\n")
    return df, log
