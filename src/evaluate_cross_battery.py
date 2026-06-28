import json
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from data_preprocessing import prepare_train_test
from train_models import (
    train_linear_regression,
    train_random_forest,
    train_xgboost,
    train_svr,
    train_hist_gb,
    evaluate,
)
from paths import FEATURES_CSV, MODELS_DIR

ALL_BATTERIES = ["B0005", "B0006", "B0007", "B0018"]


def run_cross_battery_evaluation(data_path: str = None,
                                  output_dir: str = None):
    if data_path is None:
        data_path = str(FEATURES_CSV)
    if output_dir is None:
        output_dir = str(MODELS_DIR)

    fold_results = []

    for held_out in ALL_BATTERIES:
        print(f"\n{'='*60}")
        print(f"  FOLD: held-out battery = {held_out}")
        print(f"{'='*60}")
        data = prepare_train_test(data_path, test_battery=held_out)
        print(f"  Train: {data['train_batteries']}  ({len(data['X_train'])} rows)")
        print(f"  Test : {held_out}                 ({len(data['X_test'])} rows)")

        lr  = train_linear_regression(data["X_train_scaled"], data["y_train"])
        rf  = train_random_forest(data["X_train"], data["y_train"])
        xgb = train_xgboost(data["X_train"], data["y_train"])
        svr = train_svr(data["X_train_scaled"], data["y_train"])
        hgb = train_hist_gb(data["X_train"], data["y_train"])

        fold_results.append({
            "held_out_battery":   held_out,
            "Linear Regression":  evaluate(lr,  data["X_train_scaled"], data["X_test_scaled"],
                                            data["y_train"], data["y_test"], "Linear Regression"),
            "Random Forest":      evaluate(rf,  data["X_train"], data["X_test"],
                                            data["y_train"], data["y_test"], "Random Forest"),
            "XGBoost":            evaluate(xgb, data["X_train"], data["X_test"],
                                            data["y_train"], data["y_test"], "XGBoost"),
            "SVR":                evaluate(svr, data["X_train_scaled"], data["X_test_scaled"],
                                            data["y_train"], data["y_test"], "SVR"),
            "HistGradientBoosting": evaluate(hgb, data["X_train"], data["X_test"],
                                              data["y_train"], data["y_test"], "HistGradientBoosting"),
        })

    model_names = ["Linear Regression", "Random Forest",
                   "XGBoost", "SVR", "HistGradientBoosting"]

    summary_rows = []
    for name in model_names:
        rmses = [fold[name]["test_rmse"] for fold in fold_results]
        r2s   = [fold[name]["test_r2"]   for fold in fold_results]
        maes  = [fold[name]["test_mae"]  for fold in fold_results]
        summary_rows.append({
            "model":          name,
            "mean_test_rmse": float(np.mean(rmses)),
            "std_test_rmse":  float(np.std(rmses)),
            "mean_test_r2":   float(np.mean(r2s)),
            "mean_test_mae":  float(np.mean(maes)),
            "rmse_per_fold":  {fold_results[i]["held_out_battery"]: rmses[i]
                               for i in range(len(ALL_BATTERIES))},
        })

    summary_rows.sort(key=lambda r: r["mean_test_rmse"])
    champion = summary_rows[0]["model"]
    runner_up = summary_rows[1]["model"]
    gap = summary_rows[1]["mean_test_rmse"] - summary_rows[0]["mean_test_rmse"]

    print("\n" + "=" * 90)
    print("LEAVE-ONE-BATTERY-OUT CROSS-VALIDATION SUMMARY (averaged across all folds)")
    print("=" * 90)
    print(f"{'Model':<25}{'Mean RMSE':>12}{'Std RMSE':>12}"
          f"{'Mean R²':>10}{'Mean MAE':>12}")
    print("-" * 90)
    for row in summary_rows:
        print(f"{row['model']:<25}{row['mean_test_rmse']:>12.2f}"
              f"{row['std_test_rmse']:>12.2f}"
              f"{row['mean_test_r2']:>10.4f}{row['mean_test_mae']:>12.2f}")
    print("=" * 90)

    if gap < 1.0:
        print(f"\n⚠  Champion: {champion} — but effectively tied with {runner_up} "
              f"(gap = {gap:.2f} cycles < 1 cycle, within noise for 4-fold CV)")
    else:
        print(f"\nChampion: {champion}  (mean RMSE = {summary_rows[0]['mean_test_rmse']:.2f})")

    print("\nPer-fold RMSE breakdown:")
    print(f"{'Model':<25}", end="")
    for b in ALL_BATTERIES:
        print(f"  {b:>8}", end="")
    print()
    for row in summary_rows:
        print(f"{row['model']:<25}", end="")
        for b in ALL_BATTERIES:
            print(f"  {row['rmse_per_fold'][b]:>8.2f}", end="")
        print()

    os.makedirs(output_dir, exist_ok=True)
    out_path = f"{output_dir}/cross_battery_summary.json"
    with open(out_path, "w") as f:
        json.dump({
            "summary":      summary_rows,
            "champion":     champion,
            "fold_details": fold_results,
        }, f, indent=2)
    print(f"\nSaved full CV results to {out_path}")

    return summary_rows, champion, fold_results


if __name__ == "__main__":
    run_cross_battery_evaluation()
