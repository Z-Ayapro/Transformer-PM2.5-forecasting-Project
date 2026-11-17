# scripts/run_eda.py
# runs all of milestone 1: cleaning -> EDA -> features -> split -> folds
# run it from the project folder: python scripts/run_eda.py
import sys, os

# add the scripts folder to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# add the src folder so we can import config, eda, etc.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import config
from data_loader import load_and_clean_data
from preprocessing import build_preprocessor
from features import engineer_temporal_features, NEW_FEATURES
from eda import run_eda
from splits import temporal_train_test_split, make_folds

def main():
    print("=== STARTING MILESTONE 1 PIPELINE ===")

    # 1. load and clean (bad PM2.5 rows are removed before the EDA and the split)
    df, log = load_and_clean_data()

    # 2. EDA: plots in figures/ and the 2 csv files in results/
    run_eda(df, log)

    # 3. features (lags, rolling stats...)
    df_feat = engineer_temporal_features(df)

    # 4. build the pipeline (we will fit it in milestone 2, on each training fold)
    preprocessor = build_preprocessor(config.FEATURE_COLS + NEW_FEATURES)

    # 5. save the data with the features
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    output_path = os.path.join(config.PROCESSED_DIR, 'processed_data.csv')
    df_feat.to_csv(output_path, index=False)
    print(f"5. Processed data saved to {output_path}\n")

    # 6. train/test split by date
    train_df = temporal_train_test_split()

    # 7. save the temporal, spatial and random folds (train part only)
    make_folds(train_df)

    print("=== PIPELINE COMPLETE! ===")

    print("\nCheck the 'figures/' folder for:")
    print(" - 17 Diagnostic Plots (.png)")

    print("\nCheck the 'results/' folder for:")
    print(" - Exploratory_Data.csv & Exploratory_temporal.csv")

    print("\nCheck the 'data/processed/' folder for:")
    print(" - processed_data.csv, train_data.csv & test_data.csv")

    print("\nCheck the 'folds/' folder for:")
    print(" - folds_temporal.json, folds_spatial.json, folds_standard.json")

if __name__ == "__main__":
    main()
