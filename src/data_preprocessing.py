import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
try:
    from paths import FEATURES_CSV
except ImportError:
    FEATURES_CSV = "data/processed/battery_features.csv"

# These columns are EXCLUDED from model input — each has a documented reason
LEAKAGE_COLUMNS = [
    "battery_id",           # categorical label, not a generalizable feature
    "discharge_cycle",      # RUL = EOL_cycle - discharge_cycle → direct leakage
    "capacity_ah",          # RUL defined by capacity threshold → target restatement
    "discharge_duration_s", # measures same endpoint as capacity (corr 0.89) → proxy leakage
]

TARGET_COLUMN       = "rul_cycles"
DEFAULT_TEST_BATTERY = "B0006"


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df.columns = [c.strip() for c in df.columns]
    return df


def split_features_target(df: pd.DataFrame):
    drop_cols = LEAKAGE_COLUMNS + [TARGET_COLUMN]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[TARGET_COLUMN]
    return X, y


def prepare_train_test(filepath: str = None,
                        test_battery: str = DEFAULT_TEST_BATTERY):
    """
    Battery-held-out split: all rows from test_battery → test set,
    all rows from remaining batteries → train set.
    """
    if filepath is None:
        filepath = str(FEATURES_CSV)

    df = load_data(filepath)

    if test_battery not in df["battery_id"].unique():
        raise ValueError(
            f"test_battery={test_battery!r} not found. "
            f"Available: {sorted(df['battery_id'].unique())}"
        )

    train_df = df[df["battery_id"] != test_battery].reset_index(drop=True)
    test_df  = df[df["battery_id"] == test_battery].reset_index(drop=True)

    X_train, y_train = split_features_target(train_df)
    X_test,  y_test  = split_features_target(test_df)

    # Drop constant columns (ambient_temperature = 24C across entire batch)
    constant_cols = [c for c in X_train.columns if X_train[c].nunique() <= 1]
    if constant_cols:
        print(f"  Dropping constant columns (zero variance): {constant_cols}")
        X_train = X_train.drop(columns=constant_cols)
        X_test  = X_test.drop(columns=constant_cols)

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),      # .transform only — no leakage from test stats
        columns=X_test.columns,  index=X_test.index
    )

    return {
        "X_train":        X_train,
        "X_test":         X_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled":  X_test_scaled,
        "y_train":        y_train,
        "y_test":         y_test,
        "scaler":         scaler,
        "feature_names":  list(X_train.columns),
        "test_battery":   test_battery,
        "train_batteries": sorted(train_df["battery_id"].unique()),
    }


if __name__ == "__main__":
    data = prepare_train_test()
    print(f"Train batteries : {data['train_batteries']}  ({len(data['X_train'])} rows)")
    print(f"Test battery    : {data['test_battery']}      ({len(data['X_test'])} rows)")
    print(f"Features ({len(data['feature_names'])}): {data['feature_names']}")
