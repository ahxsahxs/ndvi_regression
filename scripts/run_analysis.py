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
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)

# Force CPU to avoid XLA/CUDA issues across all steps
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

MODEL_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "final_model.keras")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "images", "results")
ERROR_ANALYSIS_DIR = os.path.join(RESULTS_DIR, "error_analysis")
INTERPRETABILITY_DIR = os.path.join(RESULTS_DIR, "interpretability")


# ---------------------------------------------------------------------------
# Step definitions
# ---------------------------------------------------------------------------

def step_1_benchmark(model, val_generator, split, max_samples, **_):
    """[1/7] Benchmark Evaluation (Table 4.1)."""
    from evaluate import evaluate
    output_path = os.path.join(PROJECT_ROOT, "results", "benchmark_comparison.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    results = evaluate(
        model_path=MODEL_PATH,
        split=split,
        max_samples=max_samples,
        metrics=["nse", "vegetation_score"],
        model=model,
        generator=val_generator,
    )

    import json
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[1/7] Saved: {output_path}")


def step_2_error_analysis(model, val_generator, split, max_samples, **_):
    """[2/7] Error Analysis (Vegetation Score comparison)."""
    from error_analysis import run_analysis_refined
    os.makedirs(ERROR_ANALYSIS_DIR, exist_ok=True)

    # Temporarily set the module-level OUTPUT_DIR
    import error_analysis as ea_mod
    ea_mod.OUTPUT_DIR = ERROR_ANALYSIS_DIR

    run_analysis_refined(
        model_path=MODEL_PATH,
        max_samples=max_samples,
        split=split,
        time_steps=10,
        model=model,
        generator=val_generator,
    )
    print(f"[2/7] Saved: {ERROR_ANALYSIS_DIR}/")


def step_3_paper_assets(model, val_generator, train_generator, split, max_samples, **_):
    """[3/7] Export Paper Assets (Figures 4.1, 4.2, etc.)."""
    from export_paper_assets import main as export_main
    export_main(
        max_samples=max_samples or 1500,
        split=split,
        model=model,
        train_generator=train_generator,
        val_generator=val_generator,
    )
    print(f"[3/7] Saved: {RESULTS_DIR}/")


def step_4_interpretability(model, val_generator, split, **_):
    """[4/7] Interpretability Analysis (Table 4.2)."""
    from interpretability import run_interpretability
    output_csv = os.path.join(INTERPRETABILITY_DIR, "parameter_stats.csv")
    os.makedirs(INTERPRETABILITY_DIR, exist_ok=True)

    run_interpretability(
        n_samples=20,
        output_path=output_csv,
        split=split,
        model=model,
        generator=val_generator,
    )
    print(f"[4/7] Saved: {output_csv}")


def step_5_param_maps(model, val_generator, split, **_):
    """[5/7] Parameter Map Visualization (Figure 4.4)."""
    from visualize import visualize_params
    output_png = os.path.join(INTERPRETABILITY_DIR, "parameter_maps.png")
    os.makedirs(INTERPRETABILITY_DIR, exist_ok=True)

    visualize_params(
        output_path=output_png,
        sample_idx=0,
        split=split,
        model=model,
        generator=val_generator,
    )
    print(f"[5/7] Saved: {output_png}")


def step_6_visualize(model, val_generator, split, **_):
    """[6/7] Prediction Visualizations."""
    from visualize import visualize
    visualize(
        split=split,
        model=model,
        generator=val_generator,
    )
    print("[6/7] Saved prediction visualizations.")


def step_7_earthnet(model, val_generator, split, max_samples, **_):
    """[7/7] EarthNet Comparison (Vegetation Score via earthnet toolkit)."""
    from compare_earthnet import run_comparison
    output_dir = os.path.join(PROJECT_ROOT, "predictions")
    run_comparison(
        model_path=MODEL_PATH,
        split=split,
        output_dir=output_dir,
        max_samples=max_samples,
        model=model,
        generator=val_generator,
    )
    print(f"[7/7] Saved: {output_dir}/")


# Ordered step registry
ALL_STEPS = {
    1: ("Benchmark Evaluation",     step_1_benchmark),
    2: ("Error Analysis",           step_2_error_analysis),
    3: ("Export Paper Assets",       step_3_paper_assets),
    4: ("Interpretability",         step_4_interpretability),
    5: ("Parameter Maps",           step_5_param_maps),
    6: ("Prediction Visualization", step_6_visualize),
    7: ("EarthNet Comparison",      step_7_earthnet),
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
        help="Steps to run (e.g. --steps 2 3 5). Default: all.",
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
    print("Results saved to:")
    print(f"  - {os.path.join(PROJECT_ROOT, 'results')}/")
    print(f"  - {ERROR_ANALYSIS_DIR}/")
    print(f"  - {INTERPRETABILITY_DIR}/")
    print(f"  - {RESULTS_DIR}/")

    # Exit with error if any step failed
    if any(v != "OK" for v in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
