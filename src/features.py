# src/features.py
import pandas as pd
import numpy as np
import sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

# the new columns we create in this file (we need the list for the pipeline)
NEW_FEATURES = [
    f'{config.TARGET_COL}_Lag1', f'{config.TARGET_COL}_RollMean7',
    f'{config.TARGET_COL}_RollStd7', f'{config.TARGET_COL}_RollMax7',
    'AOD_Lag1', 'AOD_Change', 'Sin_Day', 'Cos_Day'
]

def past_rolling_7d(df, col, func):
    """Mean/std/max of a column over the last 7 days, for each station.

    closed='left' means today is not in the window, only the days before.
    The window is in days and not in rows, so if some days are missing
    the window just has fewer values.
    """
    out = (df.groupby(config.STATION_COL)
             .rolling('7D', on=config.DATE_COL, closed='left')[col]
             .agg(func))
    # df is sorted by station and date so the result is in the same order as df
    return out.to_numpy()

def engineer_temporal_features(df):
    """Creates the lags, the 7-day rolling stats, the AOD change and the day-of-year features.

    Our setup: to predict PM2.5 of day t we use the satellite and weather values of day t
    and PM2.5 until day t-1. We never use the PM2.5 of day t as an input.
    """
    print("3. Engineering temporal features (Lags, Rolling Stats, Rates of Change)...")
    df = df.sort_values(by=[config.STATION_COL, config.DATE_COL]).reset_index(drop=True)
    g = df.groupby(config.STATION_COL)

    # True if the previous row of the station is really yesterday.
    # Without this, after a gap the "lag 1" could be 5 days old (section 5.3.4)
    prev_is_yesterday = (df[config.DATE_COL] - g[config.DATE_COL].shift(1)) == pd.Timedelta(days=1)

    # 1. lags: PM2.5 and AOD of yesterday (NaN if yesterday is missing)
    df[f'{config.TARGET_COL}_Lag1'] = g[config.TARGET_COL].shift(1).where(prev_is_yesterday)
    df['AOD_Lag1'] = g['AOD'].shift(1).where(prev_is_yesterday)

    # 2. mean and std of the last 7 days (today is not included, so no leakage)
    df[f'{config.TARGET_COL}_RollMean7'] = past_rolling_7d(df, config.TARGET_COL, 'mean')
    df[f'{config.TARGET_COL}_RollStd7'] = past_rolling_7d(df, config.TARGET_COL, 'std')

    # 3. max of the last 7 days
    df[f'{config.TARGET_COL}_RollMax7'] = past_rolling_7d(df, config.TARGET_COL, 'max')

    # 4. AOD change since yesterday (only if yesterday exists)
    df['AOD_Change'] = g['AOD'].diff().where(prev_is_yesterday)

    # 5. day of the year as sin/cos, so 31 Dec and 1 Jan are close
    df['DayOfYear'] = df[config.DATE_COL].dt.dayofyear
    df['Sin_Day'] = np.sin(2 * np.pi * df['DayOfYear'] / 365.25)
    df['Cos_Day'] = np.cos(2 * np.pi * df['DayOfYear'] / 365.25)

    # we drop the rows with no PM2.5 at all in the last 7 days (first days of a station
    # or after a long gap). The other NaN are kept, the pipeline will fill them.
    initial_len = len(df)
    df = df.dropna(subset=[f'{config.TARGET_COL}_RollMean7']).reset_index(drop=True)

    print(f"   Rows dropped (no PM2.5 in the past 7 days): {initial_len - len(df)}")
    print(f"   Lag1 missing because of gaps: {df[f'{config.TARGET_COL}_Lag1'].isna().sum()}")
    print("   Temporal features engineered.\n")
    return df
