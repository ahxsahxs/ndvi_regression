#!/usr/bin/env python3
"""
Python Analysis Runner — replaces run_analysis.sh.

Loads the model and datasets ONCE, then runs all analysis steps with
shared resources. Steps run sequentially because TensorFlow models are
not thread-safe for concurrent inference.

Usage:
    python scripts/run_analysis.py [--split val_chopped] [--max_samples N]
    python scripts/run_analysis.py --steps 2 3 5        # run only specific steps
"""

import os
import sys
import time
import argparse
import traceback

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

sys.path.append(SRC_PATH)

# GPU configuration is handled by gpu_config.configure_gpu() in main()
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

MODEL_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "final_model.keras")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "images", "results")
ERROR_ANALYSIS_DIR = os.path.join(RESULTS_DIR, "error_analysis")
INTERPRETABILITY_DIR = os.path.join(RESULTS_DIR, "interpretability")


# ---------------------------------------------------------------------------
# Step definitions
# ---------------------------------------------------------------------------

def step_1_visualize(model, val_generator, split, **_):
    """[1/2] Prediction Visualizations."""
    from visualize import visualize
    visualize(
        split=split,
        model=model,
        generator=val_generator,
    )
    print("[1/4] Saved prediction visualizations.")


def step_2_earthnet(model, val_generator, split, max_samples, **_):
    """[2/4] EarthNet Comparison (Vegetation Score via earthnet toolkit)."""
    from score_earthnet import score_predictions
    score_predictions(
        split=split
    )
    print(f"[2/4] Generated comparison/")


def step_3_diagnose(model, val_generator, split, max_samples, **_):
    """[3/4] Signal Diagnostics (error vs time, amplitude, coefficients, scatter)."""
    from diagnose_signal import diagnose
    diagnose(
        model=model,
        generator=val_generator,
        split=split,
        max_samples=max_samples or 30,
    )
    print("[3/4] Saved signal diagnostics.")


def step_4_parameter_analysis(model, val_generator, split, max_samples, **_):
    """[4/4] Parameter Analysis (gradient norms, harmonic coefficients, diversity)."""
    from parameter_analysis import analyze_parameters
    analyze_parameters(
        model=model,
        generator=val_generator,
        split=split,
        max_samples=max_samples or 20,
    )
    print("[4/4] Saved parameter analysis.")


# Ordered step registry
ALL_STEPS = {
    1: ("Prediction Visualization", step_1_visualize),
    2: ("EarthNet Comparison",      step_2_earthnet),
    3: ("Signal Diagnostics",       step_3_diagnose),
    4: ("Parameter Analysis",       step_4_parameter_analysis),
}


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run all analysis scripts with shared model/data loading."
    )
    parser.add_argument(
        "--split", type=str, default="val_chopped",
        choices=["train", "val_chopped", "val"],
        help="Dataset split (default: val_chopped)",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Maximum samples per step (None = all)",
    )
    parser.add_argument(
        "--steps", nargs="+", type=int, default=None,
        help="Steps to run (e.g. --steps 1 2). Default: all.",
    )
    args = parser.parse_args()

    active_steps = args.steps or list(ALL_STEPS.keys())

    print("=" * 60)
    print("  Analysis Runner (Python)")
    print(f"  Split: {args.split}")
    print(f"  Max samples: {args.max_samples or 'all'}")
    print(f"  Active steps: {active_steps}")
    print("=" * 60)
    print()

    # ------------------------------------------------------------------
    # Phase 0 — Load model & datasets ONCE
    # ------------------------------------------------------------------
    t0 = time.time()

    if not os.path.isfile(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        sys.exit(1)

    print("[0] Loading model …")
    from build_model import load_model
    model = load_model(MODEL_PATH, compile=False)
    print(f"    Model params: {model.count_params():,}")

    print("[0] Loading datasets …")
    from dataset import DatasetGenerator
    from config import DATASET_PATH, VALIDATION_PATH

    if args.split in ("val_chopped", "val"):
        val_path = VALIDATION_PATH
    else:
        val_path = DATASET_PATH

    val_generator = DatasetGenerator(val_path)
    train_generator = DatasetGenerator(DATASET_PATH)

    print(f"    Train files: {len(train_generator.files)}")
    print(f"    Val files:   {len(val_generator.files)}")
    print(f"    Load time:   {time.time() - t0:.1f}s")
    print()

    # ------------------------------------------------------------------
    # Phase 1 — Run analysis steps in parallel
    # ------------------------------------------------------------------
    # Build kwargs dict that every step receives
    ctx = dict(
        model=model,
        val_generator=val_generator,
        train_generator=train_generator,
        split=args.split,
        max_samples=args.max_samples,
    )

    results = {}
    t1 = time.time()

    for step_num in active_steps:
        if step_num not in ALL_STEPS:
            print(f"WARNING: Unknown step {step_num}, skipping.")
            continue
        name, fn = ALL_STEPS[step_num]
        print(f"\n{'─' * 60}")
        print(f"  [{step_num}/{len(ALL_STEPS)}] {name}")
        print(f"{'─' * 60}")
        step_t = time.time()
        try:
            fn(**ctx)
            results[step_num] = "OK"
            print(f"  ✓ Step {step_num} ({name}) completed in {time.time() - step_t:.1f}s.")
        except Exception as exc:
            results[step_num] = f"FAILED: {exc}"
            print(f"  ✗ Step {step_num} ({name}) FAILED:")
            traceback.print_exception(type(exc), exc, exc.__traceback__)

    elapsed = time.time() - t1

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("  COMPLETE")
    print(f"  Total analysis time: {elapsed:.1f}s")
    print("=" * 60)
    for step_num in sorted(results):
        name = ALL_STEPS[step_num][0]
        status = results[step_num]
        marker = "✓" if status == "OK" else "✗"
        print(f"  {marker} [{step_num}] {name}: {status}")
    print()

    # Exit with error if any step failed
    if any(v != "OK" for v in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
