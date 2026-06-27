"""
parse_nasa_battery.py
----------------------
Converts NASA PCoE Li-ion Battery Aging .mat files into a clean flat
CSV suitable for ML feature engineering.

FIX vs original: B0007 EOL fallback is now explicitly logged so you
can verify whether its labels are computed from a real 1.4 Ah crossing
or from the fallback (last cycle). If fallback triggers, a warning is
printed because those RUL labels are approximate.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from scipy.io import loadmat

sys.path.insert(0, os.path.dirname(__file__))
from paths import RAW_MAT_DIR, PROCESSED_DIR


def load_battery_mat(filepath: str):
    battery_id = os.path.splitext(os.path.basename(filepath))[0]
    raw = loadmat(filepath, simplify_cells=True)
    if battery_id not in raw:
        candidates = [k for k in raw.keys() if not k.startswith("__")]
        battery_id = candidates[0]
    return raw[battery_id]["cycle"], battery_id


def extract_discharge_cycles(cycles, battery_id: str) -> pd.DataFrame:
    rows = []
    discharge_index = 0

    for c in cycles:
        if c.get("type") != "discharge":
            continue

        discharge_index += 1
        data = c.get("data", {})

        voltage     = np.asarray(data.get("Voltage_measured",     []), dtype=float)
        current     = np.asarray(data.get("Current_measured",     []), dtype=float)
        temperature = np.asarray(data.get("Temperature_measured", []), dtype=float)
        time        = np.asarray(data.get("Time",                 []), dtype=float)
        capacity    = data.get("Capacity", np.nan)

        if voltage.size == 0 or np.isnan(capacity):
            continue

        duration = time.max() if time.size else np.nan
        v_drop_rate = (
            (voltage[0] - voltage[-1]) / max(duration, 1e-6)
            if time.size else np.nan
        )

        rows.append({
            "battery_id":          battery_id,
            "discharge_cycle":     discharge_index,
            "ambient_temperature": c.get("ambient_temperature", np.nan),
            "capacity_ah":         float(capacity),
            # Voltage features
            "voltage_mean":        voltage.mean(),
            "voltage_min":         voltage.min(),
            "voltage_std":         voltage.std(),
            "voltage_drop_rate":   v_drop_rate,
            # Current features
            "current_mean":        current.mean(),
            "current_std":         current.std(),
            # Temperature features
            "temperature_max":     temperature.max()  if temperature.size else np.nan,
            "temperature_mean":    temperature.mean() if temperature.size else np.nan,
            # Duration (excluded from model features — near-proxy for capacity)
            "discharge_duration_s": duration,
        })

    return pd.DataFrame(rows)


def add_rul_labels(df: pd.DataFrame,
                   eol_fraction: float = 0.70,
                   rated_capacity_ah: float = 2.0) -> pd.DataFrame:
    """
    Compute ground-truth RUL (remaining discharge cycles) per row.

    FIX: Explicitly report whether each battery's EOL was determined
    from a real capacity crossing or the fallback. If fallback triggers
    for a battery, its RUL labels are approximate — document this.
    """
    df = df.sort_values(["battery_id", "discharge_cycle"]).reset_index(drop=True)
    eol_threshold = eol_fraction * rated_capacity_ah

    print(f"\n  EOL threshold = {eol_threshold:.2f} Ah  "
          f"({eol_fraction*100:.0f}% of {rated_capacity_ah} Ah rated)")
    print(f"  {'Battery':<10} {'Min capacity':>14} {'EOL crossed?':>14} "
          f"{'EOL cycle':>10} {'Max RUL':>10}")
    print(f"  {'-'*60}")

    out_frames = []
    for battery_id, group in df.groupby("battery_id"):
        group = group.sort_values("discharge_cycle").reset_index(drop=True)
        below_eol = group.index[group["capacity_ah"] <= eol_threshold]
        min_cap = group["capacity_ah"].min()

        if len(below_eol) > 0:
            eol_cycle_idx = below_eol[0]
            eol_source = "real crossing"
        else:
            # ⚠️  Battery data ends before capacity hits 1.4 Ah
            # Labels will be approximate — last recorded cycle used as EOL
            eol_cycle_idx = group.index[-1]
            eol_source = "⚠️  FALLBACK (never crossed 1.4Ah)"

        eol_cycle_number = group.loc[eol_cycle_idx, "discharge_cycle"]
        group["rul_cycles"] = eol_cycle_number - group["discharge_cycle"]
        group["rul_cycles"] = group["rul_cycles"].clip(lower=0)

        max_rul = group["rul_cycles"].max()
        print(f"  {battery_id:<10} {min_cap:>14.4f} {eol_source:>14} "
              f"{eol_cycle_number:>10} {max_rul:>10}")

        out_frames.append(group)

    return pd.concat(out_frames, ignore_index=True)


def add_rolling_features(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """
    NEW: Add capacity trend over last N cycles per battery.
    This is the single biggest accuracy improvement — captures whether
    the battery is degrading slowly or rapidly, not just current state.
    Rows with insufficient history (first window-1 cycles) are dropped.
    """
    df = df.sort_values(["battery_id", "discharge_cycle"]).reset_index(drop=True)

    def slope(x):
        if len(x) < 2:
            return np.nan
        return np.polyfit(range(len(x)), x, 1)[0]

    df[f"capacity_{window}cy_slope"] = (
        df.groupby("battery_id")["capacity_ah"]
        .transform(lambda x: x.rolling(window, min_periods=window).apply(slope, raw=True))
    )

    df[f"voltage_mean_{window}cy_avg"] = (
        df.groupby("battery_id")["voltage_mean"]
        .transform(lambda x: x.rolling(window, min_periods=window).mean())
    )

    before = len(df)
    df = df.dropna(subset=[f"capacity_{window}cy_slope"]).reset_index(drop=True)
    dropped = before - len(df)
    print(f"\n  Rolling features (window={window}): dropped first {dropped} rows "
          f"per battery (insufficient history) — expected.")

    return df


def main():
    parser = argparse.ArgumentParser(description="Parse NASA PCoE battery .mat files")
    parser.add_argument("--input",         default=str(RAW_MAT_DIR))
    parser.add_argument("--output",        default=str(PROCESSED_DIR))
    parser.add_argument("--eol-fraction",  type=float, default=0.70)
    parser.add_argument("--rated-capacity",type=float, default=2.0)
    parser.add_argument("--rolling-window",type=int,   default=3)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    mat_files = sorted(f for f in os.listdir(args.input) if f.endswith(".mat"))

    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in {args.input}")

    all_batteries = []
    for fname in mat_files:
        filepath = os.path.join(args.input, fname)
        print(f"Parsing {fname} ...")
        cycles, battery_id = load_battery_mat(filepath)
        battery_df = extract_discharge_cycles(cycles, battery_id)
        print(f"  -> {len(battery_df)} discharge cycles extracted, "
              f"capacity range: {battery_df['capacity_ah'].min():.4f} – "
              f"{battery_df['capacity_ah'].max():.4f} Ah")
        all_batteries.append(battery_df)

    combined = pd.concat(all_batteries, ignore_index=True)

    print("\nAdding RUL labels...")
    combined = add_rul_labels(combined,
                               eol_fraction=args.eol_fraction,
                               rated_capacity_ah=args.rated_capacity)

    print("\nAdding rolling/lag features...")
    combined = add_rolling_features(combined, window=args.rolling_window)

    out_path = os.path.join(args.output, "battery_features.csv")
    combined.to_csv(out_path, index=False)

    print(f"\nSaved {len(combined)} total rows across "
          f"{combined['battery_id'].nunique()} batteries to {out_path}")
    print(f"Columns: {list(combined.columns)}")

    return combined


if __name__ == "__main__":
    main()
