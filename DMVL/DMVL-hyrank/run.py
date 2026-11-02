#!/usr/bin/env python3
"""
DMVL HyRank - Complete Pipeline Runner

Experiments:
1. Zero-shot: Pre-train on Dioni, few-shot from Dioni, test on Loukia test
2. Few-shot: Pre-train on Dioni, few-shot from Loukia train, test on Loukia test

Usage:
    python run.py

Make sure HyRank .mat files are in ./data/
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
    print("  DMVL HyRank - Complete Pipeline")
    print("  Dioni (Source) → Loukia (Target)")
    print("=" * 70)
    
    total_start = time.time()
    times = {}
    
    # Step 1: Generate Data
    times['data_gen'] = run_step(
        "Data Generation (Dioni pretrain, Loukia 80/20 split)",
        "step1-generate-data-hyrank.py"
    )
    
    # Step 2: Pretrain + Feature Extraction
    times['pretrain'] = run_step(
        "Pretraining on Dioni + Feature Extraction",
        "step2-train-hyrank.py"
    )
    
    # Step 3: Both Experiments
    times['eval'] = run_step(
        "Evaluation (Exp1 Zero-shot + Exp2 Few-shot)",
        "step3-test-hyrank.py"
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
    print(f"   - data/Dioni_pretrain.h5        (Dioni patches for pretrain)")
    print(f"   - data/Loukia_train.h5          (Loukia train patches)")
    print(f"   - data/Loukia_test.h5           (Loukia test patches)")
    print(f"   - data/Dioni_feature.h5         (Dioni features - Exp1)")
    print(f"   - data/Loukia_feature_train.h5  (Loukia train features - Exp2)")
    print(f"   - data/Loukia_feature_test.h5   (Loukia test features)")
    print(f"   - results/*_hyrank_model.pth    (Trained model)")
    print(f"   - HyRank/HyRank_results.txt     (Results)")
    
    print("\n" + "=" * 70)
    print("  Check HyRank/HyRank_results.txt for final metrics!")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
