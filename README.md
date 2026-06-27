# Battery Remaining Useful Life (RUL) Prediction

Predicts remaining discharge cycles before a lithium-ion battery reaches
end-of-life, using NASA PCoE degradation data. Built as an extension of
research conducted during a technical internship at CSIR-IIP (Council of
Scientific & Industrial Research, Govt. of India).

## Project Structure

```
battery-rul/
├── src/
│   ├── paths.py                   # Central path resolver
│   ├── parse_nasa_battery.py      # .mat → feature CSV + RUL labels
│   ├── data_preprocessing.py      # Leakage removal + train/test split
│   ├── train_models.py            # 5 models + GridSearchCV tuning
│   ├── evaluate_cross_battery.py  # Leave-one-battery-out CV
│   └── visualize_results.py       # Generates all plots
├── data/
│   ├── raw_mat/                   # Place .mat files here (not included)
│   └── processed/                 # Auto-generated CSV
├── models/                        # Saved models + metrics JSON
├── reports/                       # Generated plots
├── requirements.txt
└── README.md
```

## Dataset

NASA PCoE Li-ion Battery Aging Dataset (Saha & Goebel, 2007)
- 4 batteries: B0005, B0006, B0007, B0018
- 628 discharge cycles total after feature engineering
- Download from: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
- Place B0005.mat, B0006.mat, B0007.mat, B0018.mat into data/raw_mat/

## Setup

```bash
pip install -r requirements.txt
```

## Run Pipeline (in order)

```bash
python src/parse_nasa_battery.py        # Step 1: Parse .mat → CSV
python src/train_models.py              # Step 2: Train + tune models
python src/evaluate_cross_battery.py    # Step 3: Cross-validation
python src/visualize_results.py         # Step 4: Generate all plots
```

All scripts resolve paths automatically from any working directory.

## Results

**Leave-one-battery-out cross-validation (4 folds):**

| Model                 | Mean RMSE | Mean R² |
|-----------------------|-----------|---------|
| HistGradientBoosting  | 23.32     | 0.483   |
| XGBoost               | 23.55     | 0.513   |
| Random Forest         | 23.65     | 0.477   |
| SVR                   | 32.72     | -0.026  |
| Linear Regression     | 42.51     | -1.065  |

XGBoost and HistGradientBoosting are effectively tied (gap < 1 cycle).

## Key Design Decisions

- **Leakage removal**: `discharge_cycle`, `capacity_ah`, `discharge_duration_s`
  excluded — these directly or near-directly restate the target (RUL)
- **Rolling features**: `capacity_3cy_slope` captures degradation trend,
  not just current state — the biggest accuracy improvement
- **EOL threshold**: 72% of rated capacity (1.44 Ah) used instead of
  standard 70% — B0007 never crosses 1.4 Ah so 70% triggers the fallback
- **Split strategy**: Leave-one-battery-out CV — rows from the same battery
  are highly correlated, so random row splits leak information
