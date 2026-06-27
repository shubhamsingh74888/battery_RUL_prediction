#!/usr/bin/env bash
# run_pipeline.sh
# ---------------------------------------------------------------
# Runs the full Battery RUL pipeline in correct order.
# Usage:
#   ./run_pipeline.sh              # full run
#   ./run_pipeline.sh --skip-parse # skip .mat parsing if CSV exists
# ---------------------------------------------------------------

set -euo pipefail
# set -e  → exit immediately if any command fails
# set -u  → treat unset variables as errors
# set -o pipefail → catch errors inside pipes too

# ── Resolve project root from this script's location ──────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/pipeline_$(date +%Y%m%d_%H%M%S).log"

# ── Logging function ───────────────────────────────────────────
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# ── Parse flags ────────────────────────────────────────────────
SKIP_PARSE=false
for arg in "$@"; do
    [[ "$arg" == "--skip-parse" ]] && SKIP_PARSE=true
done

# ── Check Python is available ──────────────────────────────────
if ! command -v python &>/dev/null && ! command -v python3 &>/dev/null; then
    echo "ERROR: python not found. Install Python 3.8+ first."
    exit 1
fi
PYTHON=$(command -v python3 || command -v python)

# ── Check requirements are installed ──────────────────────────
log "Checking dependencies..."
$PYTHON -c "import pandas, numpy, scipy, sklearn, xgboost, joblib, matplotlib, seaborn" \
    2>/dev/null || {
    log "Missing packages. Running: pip install -r requirements.txt"
    pip install -r "${SCRIPT_DIR}/requirements.txt" | tee -a "$LOG_FILE"
}

log "========================================"
log "  Battery RUL Pipeline — Start"
log "  Log file: $LOG_FILE"
log "========================================"

# ── Step 1: Parse .mat files ───────────────────────────────────
if [ "$SKIP_PARSE" = false ]; then
    log "Step 1/4: Parsing raw .mat files..."
    START=$SECONDS
    $PYTHON "${SCRIPT_DIR}/src/parse_nasa_battery.py" 2>&1 | tee -a "$LOG_FILE"
    log "Step 1 done in $((SECONDS - START))s"
else
    log "Step 1/4: Skipped (--skip-parse flag set)"
fi

# ── Step 2: Train models ───────────────────────────────────────
log "Step 2/4: Training models with hyperparameter tuning..."
START=$SECONDS
$PYTHON "${SCRIPT_DIR}/src/train_models.py" 2>&1 | tee -a "$LOG_FILE"
log "Step 2 done in $((SECONDS - START))s"

# ── Step 3: Cross-validation ───────────────────────────────────
log "Step 3/4: Running leave-one-battery-out cross-validation..."
START=$SECONDS
$PYTHON "${SCRIPT_DIR}/src/evaluate_cross_battery.py" 2>&1 | tee -a "$LOG_FILE"
log "Step 3 done in $((SECONDS - START))s"

# ── Step 4: Visualizations ─────────────────────────────────────
log "Step 4/4: Generating plots..."
START=$SECONDS
$PYTHON "${SCRIPT_DIR}/src/visualize_results.py" 2>&1 | tee -a "$LOG_FILE"
log "Step 4 done in $((SECONDS - START))s"

log "========================================"
log "  Pipeline finished successfully"
log "  Reports saved to: ${SCRIPT_DIR}/reports/"
log "  Models saved to:  ${SCRIPT_DIR}/models/"
log "========================================"
