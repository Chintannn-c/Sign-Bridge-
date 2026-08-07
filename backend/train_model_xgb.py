"""
Train an XGBoost ISL alphabet classifier from captured MediaPipe landmarks.

This is an alternative to the Keras Dense NN in train_model.py.
XGBoost is expected to achieve significantly higher accuracy on the
126-feature landmark data due to its gradient-boosted tree ensemble
architecture which handles non-linear finger joint relationships better.

Usage:
    cd backend
    python train_model_xgb.py

Output:
    backend/models/isl_xgboost_model.pkl
    backend/models/xgb_training_meta.json
"""

import json
import logging
import os
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

# Reuse existing data loading and preprocessing utilities
from services.translator_model import ISL_LABELS, normalize_landmarks
from train_model import load_captured_data, split_by_session, augment_landmarks

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR / 'models'
MODEL_PATH = MODEL_DIR / 'isl_xgboost_model.pkl'
META_PATH = MODEL_DIR / 'xgb_training_meta.json'
NUM_FEATURES = 126


def train():
    """Train an XGBoost multi-class classifier on ISL landmark data."""
    from xgboost import XGBClassifier
    from sklearn.metrics import classification_report

    # ── Load & Validate Data ──────────────────────────────────────────────
    X, y, sessions = load_captured_data()
    present = Counter(y)
    missing = [label for index, label in enumerate(ISL_LABELS) if index not in present]
    if missing:
        raise RuntimeError(f'Missing captured classes: {", ".join(missing)}.')

    # ── Session-Based Train/Val Split ─────────────────────────────────────
    X_train, X_val, y_train, y_val = split_by_session(X, y, sessions)

    # ── Data Augmentation ─────────────────────────────────────────────────
    X_train_aug = augment_landmarks(X_train, copies=2)
    y_train_aug = np.tile(y_train, 3)  # 3x because augment_landmarks adds 2 copies
    logger.info('Training samples (augmented): %d, Validation samples: %d', len(X_train_aug), len(X_val))

    # ── Train XGBoost ─────────────────────────────────────────────────────
    model = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective='multi:softprob',
        num_class=len(ISL_LABELS),
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
        verbosity=1,
    )

    model.fit(
        X_train_aug, y_train_aug,
        eval_set=[(X_val, y_val)],
        verbose=True,
    )

    # ── Evaluate ──────────────────────────────────────────────────────────
    probabilities = model.predict_proba(X_val)
    predictions = probabilities.argmax(axis=1)
    accuracy = float((predictions == y_val).mean())

    report = classification_report(
        y_val, predictions,
        labels=range(len(ISL_LABELS)),
        target_names=ISL_LABELS,
        output_dict=True,
        zero_division=0,
    )

    # ── Save Model ────────────────────────────────────────────────────────
    MODEL_DIR.mkdir(exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

    META_PATH.write_text(json.dumps({
        'model_type': 'xgboost',
        'train_samples': int(len(X_train_aug)),
        'val_samples': int(len(X_val)),
        'val_accuracy': accuracy,
        'num_classes': len(ISL_LABELS),
        'labels': ISL_LABELS,
        'input_features': NUM_FEATURES,
        'n_estimators': 300,
        'max_depth': 8,
        'learning_rate': 0.05,
        'data_source': 'backend/dataset_collected',
        'preprocessing': 'wrist_center_scale_v1',
        'per_class': {
            label: {
                'precision': report[label]['precision'],
                'recall': report[label]['recall'],
                'f1-score': report[label]['f1-score'],
            }
            for label in ISL_LABELS
        },
    }, indent=2), encoding='utf-8')

    # ── Print Results ─────────────────────────────────────────────────────
    logger.info('=' * 60)
    logger.info('XGBoost Validation Accuracy: %.4f', accuracy)
    logger.info('=' * 60)
    for label in ISL_LABELS:
        logger.info(
            '%s  precision=%.3f  recall=%.3f  f1=%.3f',
            label,
            report[label]['precision'],
            report[label]['recall'],
            report[label]['f1-score'],
        )
    logger.info('Model saved to: %s', MODEL_PATH)
    logger.info('Metadata saved to: %s', META_PATH)


if __name__ == '__main__':
    train()
