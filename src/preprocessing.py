# src/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
import sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

def build_preprocessor(feature_cols=None):
    """Builds the pipeline that fills missing values and scales the features.
    We don't fit it here. In milestone 2 we will fit it on each training fold only,
    so the median and the scaler don't see the validation or test data.
    """
    print("4. Building Preprocessing Pipeline...")
    if feature_cols is None:
        feature_cols = config.FEATURE_COLS

    # median to fill missing values, RobustScaler because PM2.5 has big outliers.
    # add_indicator=True adds a 0/1 column "this value was missing" (asked in section 5.3.1)
    # for every column that has missing values when the pipeline is fitted: NO2, CO, O3
    # and also pm25_Lag1, pm25_RollStd7, AOD_Lag1, AOD_Change (empty after a gap) -> 7 columns
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
        ('scaler', RobustScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, feature_cols)
        ])
    print(f"   Pipeline built for {len(feature_cols)} features.\n")
    return preprocessor
