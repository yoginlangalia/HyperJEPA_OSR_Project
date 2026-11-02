#!/usr/bin/env python3
"""
DMVL Cross-Scene - Complete Pipeline Runner with Open Set Recognition (OSR)

Pipeline Steps:
1. Data Generation (H13 pretrain, H18 80/20 train/test split)
2. Pretraining on H13 + 128-dim Feature Extraction
3. Closed-Set Few-Shot Evaluation (Zero-shot H13->H18, Adaptation H18->H18)
4. Open-Set Recognition Evaluation (MSP Novelty Scoring, Macro AUROC across splits/seeds)

Usage:
    cd DMVL-Open-Set/cross-scene
    python run.py
"""

import subprocess
import sys
import time
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def run_step(step_name, script_name):
    """Run a step and print status"""
    print("\n" + "=" * 70)
    print(f"  STEP: {step_name}")
    print("=" * 70 + "\n")
    
    start_time = time.time()
    
    script_path = os.path.join(SCRIPT_DIR, script_name)
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=False,
        text=True,
        cwd=SCRIPT_DIR
    )
    
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"\n❌ ERROR: {step_name} failed!")
        print(f"Return code: {result.returncode}")
        sys.exit(1)
    else:
        print(f"\n✅ {step_name} completed in {elapsed:.2f} seconds")
    
    return elapsed

def main():
    print("\n" + "=" * 70)
    print("  DMVL Cross-Scene - Complete Pipeline (Closed-Set + Open-Set)")
    print("  Houston-13 (Source) → Houston-18 (Target)")
    print("=" * 70)
    
    total_start = time.time()
    times = {}
    
    # Check if features already exist to avoid re-training 15 mins
    feat_files = ['Houston13_feature.h5', 'Houston18_feature_train.h5', 'Houston18_feature_test.h5']
    features_exist = all(os.path.isfile(os.path.join(SCRIPT_DIR, 'data', f)) for f in feat_files)

    if '--force' in sys.argv or not features_exist:
        # Step 1: Generate Data
        times['data_gen'] = run_step(
            "Data Generation (H13 pretrain, H18 80/20 split)",
            "step1-generate-data-cross-scene.py"
        )
        
        # Step 2: Pretrain + Feature Extraction
        times['pretrain'] = run_step(
            "Pretraining on H13 + Feature Extraction",
            "step2-train-cross-scene.py"
        )
    else:
        print("\n" + "=" * 70)
        print("  ⚡ Pretrained features already exist on disk! Skipping Steps 1 & 2.")
        print("  (Run 'python run.py --force' if you ever want to re-train from scratch)")
        print("=" * 70)
        times['data_gen'] = 0.0
        times['pretrain'] = 0.0
    
    # Step 3: Closed-set Experiments
    times['closed_set'] = run_step(
        "Closed-Set Evaluation (Exp1 Zero-shot + Exp2 Few-shot)",
        "step3-test-cross-scene.py"
    )

    # Step 4: Open Set Recognition (OSR)
    times['open_set'] = run_step(
        "Open Set Recognition (OSR Evaluation)",
        "run_osr.py"
    )
    
    # Summary
    total_time = time.time() - total_start
    
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\n⏱️  Time Summary:")
    print(f"   - Data Generation:    {times['data_gen']:.2f}s")
    print(f"   - Pretraining:        {times['pretrain']:.2f}s")
    print(f"   - Closed-Set Eval:    {times['closed_set']:.2f}s")
    print(f"   - Open-Set (OSR):     {times['open_set']:.2f}s")
    print(f"   - TOTAL:              {total_time:.2f}s ({total_time/60:.2f} min)")
    
    print(f"\n📁 Output Files:")
    print(f"   - data/Houston13_feature.h5           (H13 features - Exp1)")
    print(f"   - data/Houston18_feature_train.h5      (H18 train features - Exp2)")
    print(f"   - data/Houston18_feature_test.h5       (H18 test features)")
    print(f"   - results/*_cross_scene_model.pth      (Pretrained DMVL model)")
    print(f"   - CrossScene/CrossScene_results.txt    (Closed-set OA/AA/Kappa)")
    print(f"   - CrossScene/CrossScene_OSR_results.txt (Open-set AUROC)")
    
    print("\n" + "=" * 70)
    print("  Check CrossScene/CrossScene_OSR_results.txt for Open Set metrics!")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
