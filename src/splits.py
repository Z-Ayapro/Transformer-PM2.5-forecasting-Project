# src/splits.py
import pandas as pd
import numpy as np
import json
import os
import sys
from sklearn.model_selection import TimeSeriesSplit, GroupKFold, KFold

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

def temporal_train_test_split():
    """Splits the data by date: the first 80% of the dates for training, the last 20% for the test.

    Returns the train part, we build the folds on it.
    """
    print("6. Creating Temporal Train/Test Split (Preventing Data Leakage)...")

    # load the data with the features
    df = pd.read_csv(os.path.join(config.PROCESSED_DIR, 'processed_data.csv'))
    df[config.DATE_COL] = pd.to_datetime(df[config.DATE_COL])

    # we split by date and don't shuffle, the test must be in the future
    unique_dates = df[config.DATE_COL].sort_values().unique()
    split_index = int(len(unique_dates) * config.TRAIN_RATIO)
    split_date = unique_dates[split_index]

    print(f"   Training data: Dates before {split_date}")
    print(f"   Testing data: Dates from {split_date} onwards")

    # we sort the train part by date so the fold indices follow the time
    train_df = df[df[config.DATE_COL] < split_date].sort_values(
        [config.DATE_COL, config.STATION_COL]).reset_index(drop=True)
    test_df = df[df[config.DATE_COL] >= split_date].reset_index(drop=True)

    # save both files
    train_df.to_csv(os.path.join(config.PROCESSED_DIR, 'train_data.csv'), index=False)
    test_df.to_csv(os.path.join(config.PROCESSED_DIR, 'test_data.csv'), index=False)

    print(f"   Train shape: {train_df.shape}")
    print(f"   Test shape: {test_df.shape}\n")
    return train_df


def save_folds(folds, df, name, description, seed=None):
    """Saves the fold indices in folds/<name>.json, so all our models use the same folds.

    The indices are row numbers of data/processed/train_data.csv.
    seed is None for the folds that don't use any randomness (temporal, spatial).
    """
    out = {'scheme': name, 'description': description, 'seed': seed,
           'n_rows': len(df), 'folds': []}
    for k, (tr, va) in enumerate(folds, start=1):
        out['folds'].append({
            'fold': k,
            'n_train': len(tr), 'n_val': len(va),
            'train_dates': [str(df[config.DATE_COL].iloc[tr].min().date()), str(df[config.DATE_COL].iloc[tr].max().date())],
            'val_dates': [str(df[config.DATE_COL].iloc[va].min().date()), str(df[config.DATE_COL].iloc[va].max().date())],
            'train_idx': [int(i) for i in tr],
            'val_idx': [int(i) for i in va],
        })
        print(f"   {name} fold {k}: train={len(tr)}, val={len(va)}")
    path = os.path.join(config.FOLDS_DIR, f'{name}.json')
    with open(path, 'w') as f:
        json.dump(out, f)


def make_folds(train_df):
    """Makes the 3 types of folds from section 5.4, only on the train part.

    So the test period is never in a fold.
    """
    print("7. Creating the cross-validation folds (saved in folds/)...")
    os.makedirs(config.FOLDS_DIR, exist_ok=True)
    df = train_df

    # 1. temporal CV: TimeSeriesSplit with a gap of 1 day.
    # We split the list of dates and not the rows, because many stations have the
    # same date, so a gap of 1 row would not be a gap of 1 day.
    dates = np.sort(df[config.DATE_COL].unique())
    tss = TimeSeriesSplit(n_splits=config.N_FOLDS, gap=1)
    temporal = []
    for tr_d, va_d in tss.split(dates):
        tr = np.where(df[config.DATE_COL].isin(dates[tr_d]))[0]
        va = np.where(df[config.DATE_COL].isin(dates[va_d]))[0]
        temporal.append((tr, va))
    save_folds(temporal, df, 'folds_temporal', 'TimeSeriesSplit, 5 folds, 1-day gap (on dates)')

    # 2. spatial CV: GroupKFold by station, so the validation stations are never in training.
    # The station is only used for the groups, it is not a feature.
    gkf = GroupKFold(n_splits=config.N_FOLDS)
    spatial = list(gkf.split(df, groups=df[config.STATION_COL]))
    save_folds(spatial, df, 'folds_spatial', 'GroupKFold, 5 folds, grouped by station')

    # 3. random CV: shuffled KFold with seed 42. Only to compare, to see if random
    # splits give too good results. We won't use it to choose the model.
    kf = KFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)
    standard = list(kf.split(df))
    save_folds(standard, df, 'folds_standard', 'Shuffled KFold, 5 folds, seed 42 (diagnostic only)',
               seed=config.SEED)
    print()


if __name__ == "__main__":
    make_folds(temporal_train_test_split())
