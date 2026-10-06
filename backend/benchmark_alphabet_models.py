"""
SignBridge — Benchmark Standalone XGBoost, Standalone ST-GCN, and Calibrated Hybrid.
Evaluates all three models identically on the held-out test split, prints comparison,
and saves measured hybrid metrics to backend/models/hybrid_training_meta.json.
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from services.data_loader import load_dataset_partitioned, ALPHABET_LABELS
from services.translator_model import TranslatorModel, HYBRID_META_PATH
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report


def run_benchmark():
    print("=" * 70)
    print("  SignBridge Alphabet Models: Rigorous Held-Out Benchmark")
    print("=" * 70)

    print("\n1. Loading strictly partitioned held-out test dataset...")
    partitions, report = load_dataset_partitioned()
    X_test = partitions['X_test']
    y_test = partitions['y_test']
    labels = ALPHABET_LABELS

    print(f"   Held-out test samples: {len(X_test)}")
    print(f"   Classes: {len(labels)} (A-Z)")

    translator = TranslatorModel()
    print(f"   Active TranslatorModel initialized in mode: {translator.mode}")

    models_to_test = [
        ("XGBoost (Standalone)", translator._predict_xgb),
        ("ST-GCN (Standalone)", translator._predict_stgcn),
        ("Hybrid Ensemble (Blended)", translator._predict_hybrid_ensemble),
    ]

    benchmark_results = {}

    for name, predict_fn in models_to_test:
        print(f"\n2. Evaluating: {name}...")
        y_true = []
        y_pred = []
        latencies = []

        for i in range(len(X_test)):
            sample = X_test[i]
            true_letter = labels[y_test[i]]

            t0 = time.perf_counter()
            res = predict_fn(sample)
            t1 = time.perf_counter()

            latencies.append((t1 - t0) * 1000.0)
            pred_letter = res.get('letter', '?')
            y_true.append(true_letter)
            y_pred.append(pred_letter)

        # Compute metrics
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average='macro', zero_division=0
        )
        avg_lat = float(np.median(latencies))

        report_raw = classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        )
        report_dict: dict = report_raw if isinstance(report_raw, dict) else {}

        benchmark_results[name] = {
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1),
            'latency_ms': round(avg_lat, 2),
            'per_class': {
                lbl: {
                    'precision': round(report_dict[lbl]['precision'], 4),
                    'recall': round(report_dict[lbl]['recall'], 4),
                    'f1-score': round(report_dict[lbl]['f1-score'], 4),
                    'support': int(report_dict[lbl]['support']),
                }
                for lbl in labels if isinstance(report_dict.get(lbl), dict)
            }
        }

        print(f"   Accuracy: {acc * 100:.2f}% | Macro F1: {f1 * 100:.2f}% | Latency: {avg_lat:.2f} ms")

    # Print summary comparative table
    print("\n" + "=" * 70)
    print(f"{'Model Architecture':<30} | {'Accuracy':<10} | {'Macro F1':<10} | {'Latency':<10}")
    print("-" * 70)
    for name, res in benchmark_results.items():
        print(f"{name:<30} | {res['accuracy'] * 100:>8.2f}% | {res['f1'] * 100:>8.2f}% | {res['latency_ms']:>6.2f} ms")
    print("=" * 70)

    # Save true measured hybrid metadata
    hybrid_res = benchmark_results["Hybrid Ensemble (Blended)"]
    hybrid_meta = {
        'model_type': 'hybrid_ensemble',
        'composition': '0.65 * XGBoost + 0.35 * ST-GCN (T=1.20)',
        'raw_input_landmarks': 126,
        'estimator_features': 208,
        'feature_name': 'geometric_invariants_208d',
        'num_classes': 26,
        'labels': labels,
        'test_samples': len(X_test),
        'metrics': {
            'test_accuracy': round(hybrid_res['accuracy'], 4),
            'test_macro_f1': round(hybrid_res['f1'], 4),
            'test_precision': round(hybrid_res['precision'], 4),
            'test_recall': round(hybrid_res['recall'], 4),
            'latency_ms': hybrid_res['latency_ms']
        },
        'per_class': hybrid_res['per_class'],
        'evaluated_at': time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    with open(HYBRID_META_PATH, 'w', encoding='utf-8') as f:
        json.dump(hybrid_meta, f, indent=2)
    print(f"\n[PASS] Saved measured hybrid metadata to: {HYBRID_META_PATH}")


if __name__ == '__main__':
    run_benchmark()
