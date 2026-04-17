"""
run_full_pipeline.py
--------------------
Runs the complete anime dataset pipeline in order:
  1. clean_anime_dataset.py --enrich
  2. add_images.py
  3. data_prep_worker.py

Make sure your MAL Client ID is already pasted into add_images.py.
"""

import subprocess
import sys
import os
import time

SCRIPTS = [
    ["clean_anime_dataset_resumable.py"],  # no --enrich needed; it always enriches
    ["add_images.py"],
    ["data_prep_worker.py"],
    ["clustering.py"],
]


def run_step(script_args):
    """Run a script with arguments. Returns True if successful."""
    cmd = [sys.executable] + script_args
    print("\n" + "=" * 70)
    print(f"▶ Running: {' '.join(cmd)}")
    print("=" * 70)
    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    start_time = time.time()

    # Check for script existence
    for script_parts in SCRIPTS:
        script_name = script_parts[0]
        if not os.path.exists(script_name):
            print(f"[ERROR] Required script not found: {script_name}")
            sys.exit(1)

    print("🚀 Starting full anime dataset pipeline...")

    for i, script_args in enumerate(SCRIPTS, 1):
        print(f"\n📦 STEP {i}/{len(SCRIPTS)}")
        if not run_step(script_args):
            print(f"\n❌ Pipeline stopped – step {i} failed.")
            sys.exit(1)
        print(f"✅ Step {i} completed successfully.")

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"🎉 All steps completed in {elapsed / 60:.1f} minutes.")
    print("Final outputs:")
    print("  - anime_clean.csv (cleaned + images)")
    print("  - anime_features_scaled.csv (feature matrix)")
    print("  - anime_clean_BACKUP.csv (safety backup)")
    print("=" * 70)


if __name__ == "__main__":
    main()
