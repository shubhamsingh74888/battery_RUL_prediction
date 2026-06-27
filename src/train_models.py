"""
train_models.py
----------------
Trains and evaluates four regression models for Battery RUL prediction.

UPDATES vs original:
- XGBoost: GridSearchCV hyperparameter tuning added
- Random Forest: GridSearchCV hyperparameter tuning added
- SVR: C reduced from 100 → 10 (100 was overfitting)
- Best params printed so you can see what the search found
- HistGradientBoosting added as a 5th model (sklearn, fast, handles
  small datasets well)
"""

import json
import os
import sys
import joblib
import numpy as np
from sklearn.linear_model     import LinearRegression
from sklearn.ensemble         import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.svm              import SVR
from sklearn.model_selection  import GridSearchCV
from sklearn.metrics          import mean_absolute_error, mean_squared_error, r2_score
from xgboost                  import XGBRegressor

sys.path.insert(0, os.path.dirname(__file__))
from data_preprocessing import prepare_train_test, DEFAULT_TEST_BATTERY
from paths import FEATURES_CSV, MODELS_DIR


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate(model, X_train, X_test, y_train, y_test, name: str) -> dict:
    train_pred = model.predict(X_train)
    test_pred  = model.predict(X_test)
    return {
        "model":       name,
        "train_r2":    round(r2_score(y_train, train_pred), 4),
        "test_r2":     round(r2_score(y_test,  test_pred),  4),
        "test_mae":    round(mean_absolute_error(y_test, test_pred), 2),
        "test_rmse":   round(rmse(y_test, test_pred), 2),
        "overfit_gap": round(
            r2_score(y_train, train_pred) - r2_score(y_test, test_pred), 4
        ),
    }


# ─── MODEL TRAINERS ──────────────────────────────────────────────────────────

def train_linear_regression(X_train_scaled, y_train):
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    return model


def train_random_forest(X_train, y_train):
    """FIX: GridSearchCV added — original used untuned defaults."""
    print("    Tuning Random Forest (GridSearchCV)...")
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth":    [5, 10, 15, None],
        "min_samples_leaf": [1, 2, 4],
    }
    gs = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        param_grid,
        cv=3,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=0,
    )
    gs.fit(X_train, y_train)
    print(f"    Best RF params: {gs.best_params_}  "
          f"(CV RMSE: {-gs.best_score_:.2f})")
    return gs.best_estimator_


def train_xgboost(X_train, y_train):
    """FIX: GridSearchCV added — original used untuned defaults."""
    print("    Tuning XGBoost (GridSearchCV)...")
    param_grid = {
        "n_estimators":  [100, 200, 300],
        "max_depth":     [3, 4, 6],
        "learning_rate": [0.01, 0.05, 0.1],
    }
    gs = GridSearchCV(
        XGBRegressor(subsample=0.9, colsample_bytree=0.9,
                     random_state=42, verbosity=0),
        param_grid,
        cv=3,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=0,
    )
    gs.fit(X_train, y_train)
    print(f"    Best XGB params: {gs.best_params_}  "
          f"(CV RMSE: {-gs.best_score_:.2f})")
    return gs.best_estimator_


def train_svr(X_train_scaled, y_train):
    """FIX: C reduced from 100 → 10. C=100 was too aggressive (overfitting)."""
    print("    Tuning SVR (GridSearchCV)...")
    param_grid = {
        "C":       [1, 10, 50],
        "epsilon": [0.5, 1.0, 2.0],
    }
    gs = GridSearchCV(
        SVR(kernel="rbf", gamma="scale"),
        param_grid,
        cv=3,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=0,
    )
    gs.fit(X_train_scaled, y_train)
    print(f"    Best SVR params: {gs.best_params_}  "
          f"(CV RMSE: {-gs.best_score_:.2f})")
    return gs.best_estimator_


def train_hist_gb(X_train, y_train):
    """NEW: HistGradientBoosting — fast sklearn boosting, handles small datasets well."""
    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


# ─── MAIN COMPARISON ─────────────────────────────────────────────────────────

def run_comparison(data_path: str = None,
                    test_battery: str = DEFAULT_TEST_BATTERY,
                    model_dir: str = None):
    if data_path is None:
        data_path = str(FEATURES_CSV)
    if model_dir is None:
        model_dir = str(MODELS_DIR)

    data = prepare_train_test(data_path, test_battery=test_battery)

    print(f"\nTrain batteries : {data['train_batteries']}  ({len(data['X_train'])} rows)")
    print(f"Test battery    : {data['test_battery']}      ({len(data['X_test'])} rows)")
    print(f"Features used   : {data['feature_names']}\n")

    print("Training Linear Regression...")
    lr_model  = train_linear_regression(data["X_train_scaled"], data["y_train"])
    lr_metrics = evaluate(lr_model, data["X_train_scaled"], data["X_test_scaled"],
                           data["y_train"], data["y_test"], "Linear Regression")

    print("Training Random Forest...")
    rf_model  = train_random_forest(data["X_train"], data["y_train"])
    rf_metrics = evaluate(rf_model, data["X_train"], data["X_test"],
                           data["y_train"], data["y_test"], "Random Forest")

    print("Training XGBoost...")
    xgb_model  = train_xgboost(data["X_train"], data["y_train"])
    xgb_metrics = evaluate(xgb_model, data["X_train"], data["X_test"],
                            data["y_train"], data["y_test"], "XGBoost")

    print("Training SVR...")
    svr_model  = train_svr(data["X_train_scaled"], data["y_train"])
    svr_metrics = evaluate(svr_model, data["X_train_scaled"], data["X_test_scaled"],
                            data["y_train"], data["y_test"], "SVR")

    print("Training HistGradientBoosting...")
    hgb_model  = train_hist_gb(data["X_train"], data["y_train"])
    hgb_metrics = evaluate(hgb_model, data["X_train"], data["X_test"],
                            data["y_train"], data["y_test"], "HistGradientBoosting")

    all_metrics = [lr_metrics, rf_metrics, xgb_metrics, svr_metrics, hgb_metrics]
    all_metrics.sort(key=lambda m: m["test_rmse"])

    print("\n" + "=" * 80)
    print(f"{'Model':<25}{'Train R²':>10}{'Test R²':>10}"
          f"{'Test MAE':>12}{'Test RMSE':>12}{'Overfit':>10}")
    print("=" * 80)
    for m in all_metrics:
        print(f"{m['model']:<25}{m['train_r2']:>10.4f}{m['test_r2']:>10.4f}"
              f"{m['test_mae']:>12.2f}{m['test_rmse']:>12.2f}{m['overfit_gap']:>10.4f}")

    champion = all_metrics[0]
    print("=" * 80)
    print(f"\nChampion (lowest RMSE on held-out {test_battery}): "
          f"{champion['model']}  (RMSE = {champion['test_rmse']:.2f} cycles)")

    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(lr_model,  f"{model_dir}/linear_regression.pkl")
    joblib.dump(rf_model,  f"{model_dir}/random_forest.pkl")
    joblib.dump(xgb_model, f"{model_dir}/xgboost.pkl")
    joblib.dump(svr_model, f"{model_dir}/svr.pkl")
    joblib.dump(hgb_model, f"{model_dir}/hist_gb.pkl")
    joblib.dump(data["scaler"], f"{model_dir}/scaler.pkl")

    with open(f"{model_dir}/metrics.json", "w") as f:
        json.dump({
            "results":        all_metrics,
            "champion":       champion["model"],
            "test_battery":   test_battery,
            "train_batteries": data["train_batteries"],
        }, f, indent=2)

    print(f"Saved models and metrics to {model_dir}/")

    return {
        "models": {
            "linear_regression":    lr_model,
            "random_forest":        rf_model,
            "xgboost":              xgb_model,
            "svr":                  svr_model,
            "hist_gradient_boosting": hgb_model,
        },
        "metrics": all_metrics,
        "champion": champion,
        "data": data,
    }


if __name__ == "__main__":
    run_comparison()
