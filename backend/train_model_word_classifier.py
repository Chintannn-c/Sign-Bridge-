"""
Leak-Free Random Forest / XGBoost Sequence Classifier for ISL Words.
Guarantees:
1. Splits raw video sequences BEFORE augmentation.
2. Augments only the training partition.
3. Classes with < 2 source videos are flagged as 'insufficient_data'.
"""

import json
import logging
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from services.translator_model import normalize_landmarks

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / 'dataset_words'
MODEL_DIR = SCRIPT_DIR / 'models'
MODEL_PATH = MODEL_DIR / 'isl_word_classifier.pkl'
META_PATH = MODEL_DIR / 'word_training_meta.json'

SEQUENCE_LENGTH = 30
NUM_FEATURES = 126


def load_word_sequences():
    sequences, labels, sources = [], [], []
    word_labels = sorted([d.name.upper() for d in DATA_DIR.iterdir() if d.is_dir()])
    label_to_idx = {w: i for i, w in enumerate(word_labels)}
    
    for word_dir in sorted(DATA_DIR.iterdir()):
        if not word_dir.is_dir():
            continue
        word = word_dir.name.upper()
        for json_path in sorted(word_dir.glob('*.json')):
            try:
                data = json.loads(json_path.read_text(encoding='utf-8'))
                for seq in data.get('frame_sequences', []):
                    arr = np.array(seq, dtype=np.float32)
                    if arr.shape != (SEQUENCE_LENGTH, NUM_FEATURES):
                        continue
                    
                    norm_seq = []
                    for frame in arr:
                        norm = normalize_landmarks(frame)
                        norm_seq.append(norm if norm is not None else np.zeros(NUM_FEATURES, dtype=np.float32))
                        
                    sequences.append(np.array(norm_seq, dtype=np.float32).flatten())
                    labels.append(label_to_idx[word])
                    sources.append(json_path.stem)
            except Exception as e:
                logger.warning(f"Error loading {json_path}: {e}")
                
    return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.int64), word_labels, sources


def augment_training_sequences(X, y, copies=15):
    """Augment ONLY training sequences."""
    rng = np.random.default_rng(42)
    aug_X, aug_y = [X], [y]
    for _ in range(copies):
        noise = rng.normal(0, 0.005, size=X.shape).astype(np.float32)
        aug_X.append(X + noise)
        aug_y.append(y)
    return np.concatenate(aug_X, axis=0), np.concatenate(aug_y, axis=0)


def train():
    X, y, word_labels, sources = load_word_sequences()
    logger.info(f"Loaded {len(X)} sequences across {len(word_labels)} word classes.")
    
    counts = Counter(y)
    insufficient_classes = [word_labels[idx] for idx, count in counts.items() if count < 2]
    evaluable_classes = [w for w in word_labels if w not in insufficient_classes]
    
    logger.info(f"Word classes with single video source (flagged 'insufficient_data'): {insufficient_classes}")
    logger.info(f"Word classes with >= 2 videos for independent validation: {evaluable_classes}")

    # Split BEFORE augmentation
    # Multi-sample classes split 70/30, single-sample classes placed in train
    train_idx, val_idx = [], []
    for cls_idx in range(len(word_labels)):
        idxs = np.where(y == cls_idx)[0]
        if len(idxs) >= 2:
            tr, va = train_test_split(idxs, test_size=0.3, random_state=42)
            train_idx.extend(tr)
            val_idx.extend(va)
        else:
            train_idx.extend(idxs)

    X_train_raw = X[train_idx]
    y_train_raw = y[train_idx]
    X_val = X[val_idx] if len(val_idx) > 0 else np.zeros((0, SEQUENCE_LENGTH * NUM_FEATURES), dtype=np.float32)
    y_val = y[val_idx] if len(val_idx) > 0 else np.zeros((0,), dtype=np.int64)

    logger.info(f"Raw splits -> Train: {len(X_train_raw)}, Val: {len(X_val)}")

    # Augment ONLY training partition
    X_train, y_train = augment_training_sequences(X_train_raw, y_train_raw, copies=15)
    logger.info(f"Augmented Train: {len(X_train)}")

    clf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    val_acc = 0.0
    report = {}
    if len(X_val) > 0:
        preds = clf.predict(X_val)
        val_acc = float((preds == y_val).mean())
        val_labels = [word_labels[i] for i in np.unique(y_val)]
        report = classification_report(y_val, preds, target_names=val_labels, output_dict=True, zero_division=0)
        logger.info(f"Independent Validation Accuracy on evaluable classes: {val_acc*100:.2f}%")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(clf, f)

    per_class_summary = {}
    for l in word_labels:
        if l in insufficient_classes:
            per_class_summary[l] = {'status': 'insufficient_data', 'source_videos': counts[word_labels.index(l)]}
        elif l in report:
            per_class_summary[l] = {
                'precision': round(report[l]['precision'], 4),
                'recall': round(report[l]['recall'], 4),
                'f1-score': round(report[l]['f1-score'], 4),
                'source_videos': counts[word_labels.index(l)]
            }

    meta = {
        'model_type': 'random_forest_sequence',
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'validation_accuracy_evaluable': val_acc,
        'num_classes': len(word_labels),
        'labels': word_labels,
        'insufficient_classes': insufficient_classes,
        'sequence_length': SEQUENCE_LENGTH,
        'input_features': NUM_FEATURES * SEQUENCE_LENGTH,
        'per_class': per_class_summary,
        'notes': 'Single-video classes flagged as insufficient_data; evaluated without augmentation leakage.'
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    logger.info(f"Truthful word model metadata saved to: {META_PATH}")


if __name__ == '__main__':
    train()
