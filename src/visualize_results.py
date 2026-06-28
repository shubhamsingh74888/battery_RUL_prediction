import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sys.path.insert(0, os.path.dirname(__file__))
from paths import FEATURES_CSV, MODELS_DIR, REPORTS_DIR
from data_preprocessing import prepare_train_test
from train_models import (
    train_linear_regression, train_random_forest,
    train_xgboost, train_svr, train_hist_gb
)
from evaluate_cross_battery import run_cross_battery_evaluation, ALL_BATTERIES

PALETTE = ["#2563EB", "#16A34A", "#DC2626", "#D97706", "#7C3AED"]
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "font.size":        11,
})


# ─── 1. CAPACITY DEGRADATION CURVES ──────────────────────────────────────────

def plot_capacity_degradation(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (bid, group) in enumerate(df.groupby("battery_id")):
        group = group.sort_values("discharge_cycle")
        ax.plot(group["discharge_cycle"], group["capacity_ah"],
                label=bid, color=PALETTE[i], linewidth=1.8)

    ax.axhline(1.4, color="red", linestyle="--", linewidth=1.5, label="EOL threshold (1.4 Ah)")
    ax.set_xlabel("Discharge Cycle")
    ax.set_ylabel("Capacity (Ah)")
    ax.set_title("Battery Capacity Degradation — All 4 Cells")
    ax.legend()
    fig.tight_layout()
    out = str(REPORTS_DIR / "capacity_degradation.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ─── 2. FEATURE CORRELATIONS ─────────────────────────────────────────────────

def plot_feature_correlations(df: pd.DataFrame):
    exclude = ["battery_id", "discharge_cycle", "capacity_ah",
               "discharge_duration_s", "ambient_temperature"]
    num_cols = [c for c in df.select_dtypes(include=np.number).columns
                if c not in exclude]

    corr = df[num_cols].corr()[["rul_cycles"]].drop("rul_cycles").sort_values("rul_cycles")

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#DC2626" if v < 0 else "#2563EB" for v in corr["rul_cycles"]]
    ax.barh(corr.index, corr["rul_cycles"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Pearson Correlation with RUL")
    ax.set_title("Feature Correlations with RUL (rul_cycles)")
    fig.tight_layout()
    out = str(REPORTS_DIR / "feature_correlations.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ─── 3. RMSE BY FOLD ─────────────────────────────────────────────────────────

def plot_rmse_by_fold(summary_rows: list):
    model_names = [r["model"] for r in summary_rows]
    x = np.arange(len(ALL_BATTERIES))
    width = 0.15

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (row, color) in enumerate(zip(summary_rows, PALETTE)):
        rmses = [row["rmse_per_fold"][b] for b in ALL_BATTERIES]
        ax.bar(x + i * width, rmses, width, label=row["model"], color=color, alpha=0.85)

    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels([f"{b} (held out)" for b in ALL_BATTERIES])
    ax.set_ylabel("Test RMSE (cycles)")
    ax.set_title("RMSE by Held-Out Battery — Leave-One-Battery-Out CV")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = str(REPORTS_DIR / "rmse_by_fold.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ─── 4. ACTUAL vs PREDICTED ──────────────────────────────────────────────────

def plot_actual_vs_predicted(df: pd.DataFrame):
    """Trains champion model on B0005/7/18, predicts on B0006, plots scatter."""
    data = prepare_train_test(str(FEATURES_CSV), test_battery="B0006")
    model = train_xgboost(data["X_train"], data["y_train"])
    preds = model.predict(data["X_test"])
    actual = data["y_test"].values

    residuals = preds - actual
    rmse_val = np.sqrt(np.mean(residuals**2))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter
    ax = axes[0]
    ax.scatter(actual, preds, alpha=0.5, color="#2563EB", s=20)
    lim = [min(actual.min(), preds.min()) - 5, max(actual.max(), preds.max()) + 5]
    ax.plot(lim, lim, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Actual RUL (cycles)")
    ax.set_ylabel("Predicted RUL (cycles)")
    ax.set_title(f"Actual vs Predicted RUL (XGBoost on B0006)\nRMSE = {rmse_val:.2f} cycles")
    ax.legend()

    # Residuals
    ax = axes[1]
    ax.scatter(actual, residuals, alpha=0.5, color="#DC2626", s=20)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Actual RUL (cycles)")
    ax.set_ylabel("Residual (Predicted − Actual)")
    ax.set_title("Residuals — XGBoost on B0006")

    fig.tight_layout()
    out = str(REPORTS_DIR / "actual_vs_predicted.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ─── 5. MODEL COMPARISON BAR ─────────────────────────────────────────────────

def plot_model_comparison(summary_rows: list):
    names  = [r["model"].replace("HistGradientBoosting", "HistGB") for r in summary_rows]
    means  = [r["mean_test_rmse"] for r in summary_rows]
    stds   = [r["std_test_rmse"]  for r in summary_rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(names, means, yerr=stds, capsize=5,
                  color=PALETTE[:len(names)], alpha=0.85)
    ax.set_ylabel("Mean Test RMSE (cycles)")
    ax.set_title("Model Comparison — Mean RMSE across 4-fold Leave-One-Battery-Out CV\n"
                 "(error bars = ±1 std across folds)")
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    out = str(REPORTS_DIR / "model_comparison.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    print("Loading feature CSV...")
    df = pd.read_csv(str(FEATURES_CSV))
    print(f"  {len(df)} rows, {df['battery_id'].nunique()} batteries\n")

    print("Plot 1/5: Capacity degradation curves")
    plot_capacity_degradation(df)

    print("Plot 2/5: Feature correlations")
    plot_feature_correlations(df)

    print("Plot 3/5 + 5/5: Running cross-battery evaluation (this takes a minute)...")
    summary_rows, champion, _ = run_cross_battery_evaluation()

    print("\nPlot 3/5: RMSE by fold")
    plot_rmse_by_fold(summary_rows)

    print("Plot 4/5: Actual vs Predicted (XGBoost on B0006)")
    plot_actual_vs_predicted(df)

    print("Plot 5/5: Model comparison")
    plot_model_comparison(summary_rows)

    print(f"\nAll plots saved to {REPORTS_DIR}/")


if __name__ == "__main__":
    main()
