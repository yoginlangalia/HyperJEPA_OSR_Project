#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SpectralDINO Houston Cross-Scene — Complete Pipeline Runner

Pipeline Steps:
1. Feature Extraction (SpectralDINO Pretraining on H13 + 384-dim Feature Extraction)
2. Closed-Set Few-Shot Evaluation (Zero-shot H13->H18, Adaptation H18->H18)
3. Open Set Recognition Evaluation (MSP Novelty Scoring, Macro AUROC + Known Accuracy)

Usage:
    cd SpectralDINO-Open-Set/cross-scene
    python run.py
"""

import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_step(step_name, script_name, args=None):
    print("\n" + "=" * 70)
    print(f"  STEP: {step_name}")
    print("=" * 70 + "\n")
    
    start_time = time.time()
    cmd = [sys.executable, os.path.join(SCRIPT_DIR, script_name)]
    if args:
        cmd.extend(args)
        
    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"\n❌ ERROR: {step_name} failed with code {result.returncode}!")
        sys.exit(1)
    else:
        print(f"\n✅ {step_name} completed in {elapsed:.2f}s ({elapsed/60:.2f} min)")
    return elapsed


def main():
    print("\n" + "=" * 70)
    print("  SpectralDINO Cross-Scene Pipeline (Closed-Set + Open-Set OSR)")
    print("  Houston-13 (Source) -> Houston-18 (Target)")
    print("=" * 70)

    total_start = time.time()
    times = {}

    feat_files = ['Houston13_feature.h5', 'Houston18_feature_train.h5', 'Houston18_feature_test.h5']
    features_exist = all(os.path.isfile(os.path.join(SCRIPT_DIR, 'data', f)) for f in feat_files)

    extra_args = []
    if '--force' in sys.argv:
        extra_args.append('--force')
    for i, arg in enumerate(sys.argv):
        if arg == '--data_dir' and i + 1 < len(sys.argv):
            extra_args.extend(['--data_dir', sys.argv[i + 1]])
        elif arg.startswith('--data_dir='):
            extra_args.append(arg)
        elif arg == '--pretrain_epochs' and i + 1 < len(sys.argv):
            extra_args.extend(['--pretrain_epochs', sys.argv[i + 1]])
        elif arg.startswith('--pretrain_epochs='):
            extra_args.append(arg)

    # Step 1: Pretraining & Feature Extraction
    if '--force' in sys.argv or not features_exist:
        times['extract'] = run_step(
            "Pretrain SpectralDINO + 384-dim Feature Extraction",
            "step1_extract_features.py",
            extra_args
        )
    else:
        print("\n" + "=" * 70)
        print("  Pretrained features already exist in data/! Skipping extraction.")
        print("  (Pass --force to re-generate from scratch)")
        print("=" * 70)
        times['extract'] = 0.0

    # Step 2: Closed-set Evaluation
    times['closed_set'] = run_step(
        "Closed-Set Few-Shot Evaluation (Zero-shot & Adaptation)",
        "step2_closed_set.py"
    )

    # Step 3: Open-Set Recognition (OSR)
    times['open_set'] = run_step(
        "Open Set Recognition (OSR Evaluation)",
        "run_osr.py"
    )

    total_time = time.time() - total_start
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\n⏱️  Timing:")
    print(f"   - Feature Extraction:  {times['extract']:.2f}s")
    print(f"   - Closed-Set Eval:     {times['closed_set']:.2f}s")
    print(f"   - Open-Set (OSR):      {times['open_set']:.2f}s")
    print(f"   - TOTAL:               {total_time:.2f}s ({total_time/60:.2f} min)")

    print("\n📁 Output Files:")
    print("   - data/Houston13_feature.h5")
    print("   - data/Houston18_feature_train.h5")
    print("   - data/Houston18_feature_test.h5")
    print("   - CrossScene/CrossScene_results.txt     (Closed-set OA/AA/Kappa)")
    print("   - CrossScene/CrossScene_OSR_results.txt (Open-set AUROC + Known Acc)")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
