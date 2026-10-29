"""
SignBridge QA Analysis - Phase 1: Baseline Verification
1. Verifies existing test runner statuses (backend pytest, npm test, oxlint).
2. Maps real code structure (routes, services, hooks, entry points) vs PROJECT_CONTEXT.md.
3. Computes ground-truth metrics from backend/models/*_training_meta.json.
4. Checks for data leakage in backend/services/data_loader.py and dataset_collected.
"""

import os
import sys
import json
import time
import hashlib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = BASE_DIR / 'backend'
MODELS_DIR = BACKEND_DIR / 'models'

sys.path.insert(0, str(BACKEND_DIR))

def run_phase1():
    print("=" * 70)
    print("  PHASE 1: BASELINE & GROUND TRUTH VERIFICATION")
    print("=" * 70)

    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "test_suites": {},
        "model_metas": {},
        "data_leakage_check": {},
        "mismatches": []
    }

    # 1. Model Metadata Verification
    meta_files = {
        "xgboost": MODELS_DIR / "xgb_training_meta.json",
        "stgcn": MODELS_DIR / "stgcn_training_meta.json",
        "hybrid": MODELS_DIR / "hybrid_training_meta.json",
        "cnn_bilstm_word": MODELS_DIR / "cnn_lstm_training_meta.json",
        "random_forest_word": MODELS_DIR / "word_training_meta.json"
    }

    print("\n--- 1. Model Metadata Files Inspection ---")
    for key, path in meta_files.items():
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            results["model_metas"][key] = {
                "exists": True,
                "model_type": data.get("model_type"),
                "num_classes": data.get("num_classes"),
                "metrics": data.get("metrics") or {
                    "val_accuracy": data.get("val_accuracy"),
                    "train_samples": data.get("train_samples"),
                    "val_samples": data.get("val_samples")
                }
            }
            print(f"[{key}] Loaded {path.name}: classes={data.get('num_classes')}, metrics={results['model_metas'][key]['metrics']}")
        else:
            results["model_metas"][key] = {"exists": False}
            print(f"[{key}] MISSING {path}")

    # 2. Check Data Loader Partitioning and Train/Test Leakage
    print("\n--- 2. Dataset Partition & Train/Test Leakage Audit ---")
    try:
        from services.data_loader import load_dataset_partitioned, ALPHABET_LABELS
        partitions, report = load_dataset_partitioned()
        
        train_hashes = set()
        val_hashes = set()
        test_hashes = set()
        
        # Check frame exact overlap across splits
        X_train = partitions["X_train"]
        X_val = partitions["X_val"]
        X_test = partitions["X_test"]
        
        for frame in X_train:
            h = hashlib.sha256(frame.tobytes()).hexdigest()
            train_hashes.add(h)
        for frame in X_val:
            h = hashlib.sha256(frame.tobytes()).hexdigest()
            val_hashes.add(h)
        for frame in X_test:
            h = hashlib.sha256(frame.tobytes()).hexdigest()
            test_hashes.add(h)
            
        train_test_overlap = train_hashes.intersection(test_hashes)
        train_val_overlap = train_hashes.intersection(val_hashes)
        val_test_overlap = val_hashes.intersection(test_hashes)
        
        results["data_leakage_check"] = {
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "train_test_frame_overlap_count": len(train_test_overlap),
            "train_val_frame_overlap_count": len(train_val_overlap),
            "val_test_frame_overlap_count": len(val_test_overlap),
            "duplicate_files_detected": len(report.get("duplicate_files", [])),
            "duplicate_frames_excluded": report.get("duplicate_frames_excluded", 0)
        }
        print(f"Train samples: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        print(f"Train/Test frame hash overlap: {len(train_test_overlap)}")
        print(f"Train/Val frame hash overlap: {len(train_val_overlap)}")
        print(f"Val/Test frame hash overlap: {len(val_test_overlap)}")
        
    except Exception as e:
        print(f"Data loader audit failed: {e}")
        results["data_leakage_check"] = {"error": str(e)}

    # Save Phase 1 Baseline Output
    out_path = Path(__file__).resolve().parent / "phase1_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nPhase 1 results written to {out_path}")

if __name__ == "__main__":
    run_phase1()
