#!/usr/bin/env python3
"""
DMVL Trento - Complete Pipeline Runner

This script runs the complete DMVL pipeline for Trento dataset:
1. Data Generation (80/20 split, patches from TR and TE)
2. Pretraining (contrastive learning on TR)
3. Feature Extraction (for both TR and TE)
4. Few-shot SVM Evaluation (1, 5, 10 shots)

Usage:
    python run.py

Make sure trento.mat is placed in ./data/trento.mat
"""

import subprocess
import sys
import time

def run_step(step_name, script_name):
    """Run a step and print status"""
    print("\n" + "=" * 70)
    print(f"  STEP: {step_name}")
    print("=" * 70 + "\n")
    
    start_time = time.time()
    
    result = subprocess.run(
        [sys.executable, script_name],
        capture_output=False,
        text=True
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
    print("  DMVL Trento - Complete Pipeline")
    print("  Deep Multiview Learning for Hyperspectral Image Classification")
    print("=" * 70)
    
    total_start = time.time()
    times = {}
    
    # Step 1: Generate Data
    times['data_gen'] = run_step(
        "Data Generation (80/20 split, extracting patches)",
        "step1-generate-data-trento.py"
    )
    
    # Step 2: Pretrain + Feature Extraction
    times['pretrain'] = run_step(
        "Pretraining (Contrastive Learning) + Feature Extraction",
        "step2-train-trento.py"
    )
    
    # Step 3: Few-shot Evaluation
    times['eval'] = run_step(
        "Few-shot SVM Evaluation (1, 5, 10 shots)",
        "step3-test-trento.py"
    )
    
    # Summary
    total_time = time.time() - total_start
    
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\n⏱️  Time Summary:")
    print(f"   - Data Generation:    {times['data_gen']:.2f}s")
    print(f"   - Pretraining:        {times['pretrain']:.2f}s")
    print(f"   - Evaluation:         {times['eval']:.2f}s")
    print(f"   - TOTAL:              {total_time:.2f}s ({total_time/60:.2f} min)")
    
    print(f"\n📁 Output Files:")
    print(f"   - data/Trento28-28-6_train.h5  (Training patches)")
    print(f"   - data/Trento28-28-6_test.h5   (Test patches)")
    print(f"   - data/Trento_feature_train.h5 (Training features)")
    print(f"   - data/Trento_feature_test.h5  (Test features)")
    print(f"   - results/*_trento_model.pth   (Trained model)")
    print(f"   - Trento/Trento_results.txt   (Results)")
    
    print("\n" + "=" * 70)
    print("  Check Trento/Trento_results.txt for final metrics!")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
