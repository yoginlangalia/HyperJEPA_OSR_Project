#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SpectralDINO Open Set Recognition — Top-level Pipeline Runner
Runs Open Set Recognition (OSR) evaluation for both:
  1. Houston Cross-Scene (Houston-13 -> Houston-18)
  2. HyRank Cross-Scene (Dioni -> Loukia)

Usage:
    cd SpectralDINO-Open-Set
    python run_all.py
"""

import subprocess
import sys
import time
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_task(dataset_name, script_relpath):
    print("\n" + "#" * 75)
    print(f"  RUNNING OSR EVALUATION: {dataset_name}")
    print("#" * 75 + "\n")
    
    script_path = os.path.join(SCRIPT_DIR, script_relpath)
    cwd = os.path.dirname(script_path)
    start_time = time.time()
    
    result = subprocess.run(
        [sys.executable, os.path.basename(script_path)],
        capture_output=False,
        text=True,
        cwd=cwd
    )
    
    elapsed = time.time() - start_time
    if result.returncode != 0:
        print(f"\n❌ ERROR in {dataset_name}! (Return code: {result.returncode})")
        return elapsed, False
    else:
        print(f"\n✅ {dataset_name} completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
        return elapsed, True


def main():
    print("=" * 75)
    print("  SpectralDINO OPEN SET RECOGNITION (OSR) BENCHMARK SUITE")
    print("  Cross-Scene Datasets: Houston (2013 -> 2018) & HyRank (Dioni -> Loukia)")
    print("=" * 75)
    
    total_start = time.time()
    
    t_houston, ok1 = run_task("Houston Cross-Scene", "cross-scene/run_osr.py")
    t_hyrank, ok2  = run_task("HyRank Cross-Scene", "hyrank/run_osr.py")
    
    total_time = time.time() - total_start
    print("\n" + "=" * 75)
    print("  ALL OSR EVALUATIONS FINISHED")
    print("=" * 75)
    print(f"  Houston Cross-Scene: {t_houston:.1f}s [{'SUCCESS' if ok1 else 'FAILED'}]")
    print(f"  HyRank Cross-Scene:  {t_hyrank:.1f}s [{'SUCCESS' if ok2 else 'FAILED'}]")
    print(f"  Total Duration:      {total_time:.1f}s ({total_time/60:.1f} min)")
    print("=" * 75)


if __name__ == '__main__':
    main()
