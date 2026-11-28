"""
SignBridge — Benchmark Standalone XGBoost, Standalone ST-GCN, and Calibrated Hybrid.
Evaluates all three models identically on the held-out test split, prints comparison,
and saves measured hybrid metrics to backend/models/hybrid_training_meta.json.
"""

import sys
import json
import time
from pathlib import Path
import numpy as np

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from services.data_loader import load_dataset_partitioned, ALPHABET_LABELS
from services.translator_model import TranslatorModel, HYBRID_META_PATH
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, f1_score


def _tune_hybrid_on_val(translator, X_val, y_val):
    """Tune ensemble blend weight and temperature strictly on the validation split."""
    import torch
    from services.translator_model import normalize_landmarks
    from services.feature_extractor import extract_features, extract_palm_aligned_features
    from services.gesture_config import CONFIDENCE_THRESHOLD, MARGIN_THRESHOLD

    valid_rows, valid_y = [], []
    for row, lbl in zip(X_val, y_val):
        norm = normalize_landmarks(row)
        if norm is not None and np.any(norm != 0):
            valid_rows.append(norm)
            valid_y.append(lbl)
    if not valid_rows:
        return None

    X_norm = np.asarray(valid_rows, dtype=np.float32)
    y_arr = np.asarray(valid_y, dtype=np.int64)
    swapped = np.zeros_like(X_norm)
    swapped[:, :63] = X_norm[:, 63:]
    swapped[:, 63:] = X_norm[:, :63]
    is_single = np.any(X_norm[:, :63] != 0, axis=1) ^ np.any(X_norm[:, 63:] != 0, axis=1)

    feat_fn = extract_palm_aligned_features if getattr(translator, 'xgb_feature_mode', '') == 'palm_aligned_208d' else extract_features
    probs_xgb_p = translator.xgb_model.predict_proba(feat_fn(X_norm))
    probs_xgb_s = translator.xgb_model.predict_proba(feat_fn(swapped))

    with torch.no_grad():
        logits_p = translator.stgcn_model(torch.tensor(X_norm, dtype=torch.float32)).numpy()
        logits_s = translator.stgcn_model(torch.tensor(swapped, dtype=torch.float32)).numpy()

    def _softmax(logits, temp):
        scaled = logits / temp
        z = scaled - np.max(scaled, axis=1, keepdims=True)
        e = np.exp(z)
        return e / np.sum(e, axis=1, keepdims=True)

    best = None
    cls_labels = list(range(len(ALPHABET_LABELS)))
    for w_xgb in (0.35, 0.50, 0.65, 0.75, 0.85, 0.90):
        w_stgcn = round(1.0 - w_xgb, 2)
        for temp in (0.45, 0.60, 0.75, 0.90, 1.00, 1.20):
            p_p = w_xgb * probs_xgb_p + w_stgcn * _softmax(logits_p, temp)
            p_s = w_xgb * probs_xgb_s + w_stgcn * _softmax(logits_s, temp)
            conf_p = np.max(p_p, axis=1)
            conf_s = np.max(p_s, axis=1)
            use_swap = ((is_single & (conf_p < 0.65)) | (conf_p < 0.55)) & (conf_s > conf_p)
            probs = np.where(use_swap[:, None], p_s, p_p)
            sorted_p = np.sort(probs, axis=1)
            conf = sorted_p[:, -1]
            margin = sorted_p[:, -1] - sorted_p[:, -2]
            preds = np.where((conf >= CONFIDENCE_THRESHOLD) & (margin >= MARGIN_THRESHOLD), np.argmax(probs, axis=1), -1)
            acc = float(accuracy_score(y_arr, preds))
            macro_f1 = float(f1_score(y_arr, preds, labels=cls_labels, average='macro', zero_division=0))
            score = 0.5 * acc + 0.5 * macro_f1
            if best is None or score > best['score']:
                best = {
                    'xgb': w_xgb,
                    'stgcn': w_stgcn,
                    'temperature': temp,
                    'val_accuracy': round(acc, 4),
                    'val_macro_f1': round(macro_f1, 4),
                    'score': score,
                }
    if best is not None:
        best.pop('score', None)
    return best


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

    # Safety guard: ensure full hybrid components are loaded before benchmarking
    if translator.mode not in ('hybrid', 'hybrid_ensemble') or translator.xgb_model is None or translator.stgcn_model is None:
        print(f"\n[WARNING] TranslatorModel is in '{translator.mode}' fallback mode or missing sub-model weights.")
        print("          Benchmark will execute for diagnostics, but hybrid metadata will NOT be overwritten.")
        can_save_metadata = False
    else:
        can_save_metadata = True
        if len(partitions.get('X_val', [])) > 0:
            tuned = _tune_hybrid_on_val(translator, partitions['X_val'], partitions['y_val'])
            if tuned:
                translator.hybrid_xgb_weight = tuned['xgb']
                translator.hybrid_stgcn_weight = tuned['stgcn']
                translator.hybrid_temperature = tuned['temperature']
                print(
                    f"   Val-tuned hybrid weights: {tuned['xgb']:.2f}*XGB + {tuned['stgcn']:.2f}*ST-GCN "
                    f"(T={tuned['temperature']:.2f}, Val={tuned['val_accuracy']*100:.2f}%, ValF1={tuned['val_macro_f1']:.4f})"
                )
        else:
            tuned = None

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
        median_lat = float(np.median(latencies))
        mean_lat = float(np.mean(latencies))
        abstains = sum(1 for p in y_pred if p == '?')
        abstain_rate = float(abstains / len(y_pred)) if y_pred else 0.0

        report_raw = classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        )
        report_dict: dict = report_raw if isinstance(report_raw, dict) else {}

        benchmark_results[name] = {
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1),
            'latency_ms': round(median_lat, 2),
            'mean_latency_ms': round(mean_lat, 2),
            'abstain_rate': round(abstain_rate, 4),
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

        print(f"   Accuracy: {acc * 100:.2f}% | Macro F1: {f1 * 100:.2f}% | Median Latency: {median_lat:.2f} ms | Abstain: {abstain_rate * 100:.1f}%")

    # Print summary comparative table
    print("\n" + "=" * 70)
    print(f"{'Model Architecture':<30} | {'Accuracy':<10} | {'Macro F1':<10} | {'Latency':<10}")
    print("-" * 70)
    for name, res in benchmark_results.items():
        print(f"{name:<30} | {res['accuracy'] * 100:>8.2f}% | {res['f1'] * 100:>8.2f}% | {res['latency_ms']:>6.2f} ms")
    print("=" * 70)

    # Save true measured hybrid metadata only when both models are active
    if can_save_metadata:
        hybrid_res = benchmark_results["Hybrid Ensemble (Blended)"]
        xgb_w = getattr(translator, 'hybrid_xgb_weight', translator.HYBRID_XGB_WEIGHT)
        stgcn_w = getattr(translator, 'hybrid_stgcn_weight', translator.HYBRID_STGCN_WEIGHT)
        temp_val = getattr(translator, 'hybrid_temperature', translator.HYBRID_TEMPERATURE)
        composition_str = f"{xgb_w:.2f} * XGBoost + {stgcn_w:.2f} * ST-GCN (T={temp_val:.2f})"

        hybrid_meta = {
            'model_type': 'hybrid_ensemble',
            'composition': composition_str,
            'weights': tuned if tuned else {
                'xgb': xgb_w,
                'stgcn': stgcn_w,
                'temperature': temp_val,
            },
            'raw_input_landmarks': 126,
            'estimator_features': 208,
            'feature_name': getattr(translator, 'xgb_feature_mode', 'geometric_invariants_208d'),
            'num_classes': 26,
            'labels': labels,
            'test_samples': len(X_test),
            'metrics': {
                'test_accuracy': round(hybrid_res['accuracy'], 4),
                'test_macro_f1': round(hybrid_res['f1'], 4),
                'test_precision': round(hybrid_res['precision'], 4),
                'test_recall': round(hybrid_res['recall'], 4),
                'latency_ms': hybrid_res['latency_ms'],
                'mean_latency_ms': hybrid_res['mean_latency_ms'],
                'abstain_rate': hybrid_res['abstain_rate'],
            },
            'per_class': hybrid_res['per_class'],
            'evaluated_at': time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        with open(HYBRID_META_PATH, 'w', encoding='utf-8') as f:
            json.dump(hybrid_meta, f, indent=2)
        print(f"\n[PASS] Saved verified hybrid metadata to: {HYBRID_META_PATH}")
    else:
        print("\n[SKIP] Preserved existing hybrid metadata (did not overwrite due to fallback mode).")


if __name__ == '__main__':
    run_benchmark()
