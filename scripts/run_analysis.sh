#!/bin/bash
# =============================================================================
# Run all analysis scripts for Chapter 4 Results
# =============================================================================

set -e  # Exit on error

PROJECT_ROOT="/home/me/workspace/bspline_ndvi"
MODEL_PATH="${PROJECT_ROOT}/checkpoints/final_model.keras"
RESULTS_DIR="${PROJECT_ROOT}/images/results"
ERROR_ANALYSIS_DIR="${RESULTS_DIR}/error_analysis"
INTERPRETABILITY_DIR="${RESULTS_DIR}/interpretability"
SPLIT="val_chopped"

# =============================================================================
# Toggle which steps to run (comment out lines to skip)
# =============================================================================
RUN_STEPS=(
#    1  # Benchmark Evaluation (Table 4.1)
    2  # Error Analysis (Vegetation Score comparison)
    3  # Export Paper Assets (Figures 4.1, 4.2, etc.)
    4  # Interpretability Analysis (Table 4.2)
    5  # Parameter Map Visualization (Figure 4.4)
)

cd "$PROJECT_ROOT"

# Check for model
if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    echo "Please train the model first."
    exit 1
fi

echo "=============================================="
echo "Analysis Scripts for Chapter 4 Results"
echo "Active steps: ${RUN_STEPS[*]}"
echo "=============================================="
echo ""

# Create output directories
mkdir -p "$ERROR_ANALYSIS_DIR"
mkdir -p "$INTERPRETABILITY_DIR"
mkdir -p "${PROJECT_ROOT}/results"

# -----------------------------------------------------------------------------
# 1. Benchmark Evaluation (Table 4.1)
# -----------------------------------------------------------------------------
if [[ " ${RUN_STEPS[*]} " == *" 1 "* ]]; then
    echo "[1/5] Running benchmark evaluation..."
    python src/evaluate.py \
        --model "$MODEL_PATH" \
        --split "$SPLIT" \
        --metrics nse vegetation_score \
        --output results/benchmark_comparison.json

    echo "      Output: results/benchmark_comparison.json"
    echo ""
else
    echo "[1/5] Skipped benchmark evaluation"
fi

# -----------------------------------------------------------------------------
# 2. Error Analysis (Vegetation Score comparison)
# -----------------------------------------------------------------------------
if [[ " ${RUN_STEPS[*]} " == *" 2 "* ]]; then
    echo "[2/5] Running error analysis..."
    python src/error_analysis.py \
        --model "$MODEL_PATH" \
        --split "$SPLIT" \
        --output "$ERROR_ANALYSIS_DIR"

    echo "      Output: ${ERROR_ANALYSIS_DIR}/"
    echo ""
else
    echo "[2/5] Skipped error analysis"
fi

# -----------------------------------------------------------------------------
# 3. Export Paper Assets (Figures 4.1, 4.2, and more)
# -----------------------------------------------------------------------------
if [[ " ${RUN_STEPS[*]} " == *" 3 "* ]]; then
    echo "[3/5] Exporting paper assets..."
    python src/export_paper_assets.py \
        --split "$SPLIT"

    echo "      Output: ${RESULTS_DIR}/"
    echo ""
else
    echo "[3/5] Skipped paper assets export"
fi

# -----------------------------------------------------------------------------
# 4. Interpretability Analysis (Table 4.2)
# -----------------------------------------------------------------------------
if [[ " ${RUN_STEPS[*]} " == *" 4 "* ]]; then
    echo "[4/5] Running interpretability analysis..."
    python src/interpretability.py \
        --samples 20 \
        --split "$SPLIT" \
        --output "${INTERPRETABILITY_DIR}/parameter_stats.csv"

    echo "      Output: ${INTERPRETABILITY_DIR}/parameter_stats.csv"
    echo ""
else
    echo "[4/5] Skipped interpretability analysis"
fi

# -----------------------------------------------------------------------------
# 5. Parameter Map Visualization (Figure 4.4)
# -----------------------------------------------------------------------------
if [[ " ${RUN_STEPS[*]} " == *" 5 "* ]]; then
    echo "[5/5] Generating parameter maps..."
    python src/visualize.py \
        --show-params \
        --sample 0 \
        --split "$SPLIT" \
        --output "${INTERPRETABILITY_DIR}/parameter_maps.png"

    echo "      Output: ${INTERPRETABILITY_DIR}/parameter_maps.png"
    echo ""
else
    echo "[5/5] Skipped parameter maps"
fi

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo "=============================================="
echo "COMPLETE"
echo "=============================================="
echo ""
echo "Results saved to:"
echo "  - results/benchmark_comparison.json    (Table 4.1)"
echo "  - ${ERROR_ANALYSIS_DIR}/               (Error analysis)"
echo "  - ${INTERPRETABILITY_DIR}/             (Table 4.2, Figure 4.4)"
echo "  - ${RESULTS_DIR}/                      (All figures)"
