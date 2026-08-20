"""Train the ISL landmark classifier from captured MediaPipe landmarks."""

import json
import logging
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from services.translator_model import ISL_LABELS, normalize_landmarks

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / 'dataset_collected'
ROOT_DATASET_DIR = SCRIPT_DIR.parent / 'dataset' / 'ISL_Landmarks'
MODEL_DIR = SCRIPT_DIR / 'models'
MODEL_PATH = MODEL_DIR / 'isl_gesture_model.h5'
META_PATH = MODEL_DIR / 'training_meta.json'
NUM_FEATURES = 126


def load_captured_data(data_dirs=None):
    if data_dirs is None:
        data_dirs = [DATA_DIR, ROOT_DATASET_DIR]
    elif isinstance(data_dirs, (str, Path)):
        data_dirs = [Path(data_dirs)]

    samples, labels, sessions = [], [], []
    loaded_paths = set()

    for d in data_dirs:
        d_path = Path(d)
        if not d_path.exists():
            continue
        for path in sorted(d_path.glob('*/*.json')):
            if path in loaded_paths:
                continue
            loaded_paths.add(path)
            try:
                payload = json.loads(path.read_text(encoding='utf-8'))
                label = str(payload.get('letter', '')).upper()
                if label not in ISL_LABELS:
                    continue
                session = str(payload.get('session_id') or path.stem)
                for frame in payload.get('frames', []):
                    arr = np.asarray(frame, dtype=np.float32)
                    if arr.size != NUM_FEATURES or not np.isfinite(arr).all():
                        continue
                    normalized = normalize_landmarks(arr)
                    if normalized is not None:
                        samples.append(normalized)
                        labels.append(ISL_LABELS.index(label))
                        sessions.append(f'{label}:{session}')
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                logger.warning('Skipping %s: %s', path, exc)

    if not samples:
        raise RuntimeError(f'No valid captured 126-value landmark frames found in {data_dirs}.')
    return np.asarray(samples, dtype=np.float32), np.asarray(labels, dtype=np.int32), np.asarray(sessions)


def split_by_session(X, y, sessions, validation_fraction=0.2, seed=42):
    rng = np.random.default_rng(seed)
    train_idx, validation_idx = [], []
    for label in range(len(ISL_LABELS)):
        indices = np.flatnonzero(y == label)
        if len(indices) == 0:
            continue
        groups = np.unique(sessions[indices])
        if len(groups) >= 2:
            rng.shuffle(groups)
            validation_groups = set(groups[:max(1, round(len(groups) * validation_fraction))])
            validation_idx.extend(i for i in indices if sessions[i] in validation_groups)
            train_idx.extend(i for i in indices if sessions[i] not in validation_groups)
        else:
            # Fallback to random sample split for single session classes
            shuffled = indices.copy()
            rng.shuffle(shuffled)
            n_val = max(1, round(len(shuffled) * validation_fraction))
            validation_idx.extend(shuffled[:n_val])
            train_idx.extend(shuffled[n_val:])

    if not train_idx or not validation_idx:
        raise RuntimeError('Could not create non-empty train and validation sets.')
    return X[train_idx], X[validation_idx], y[train_idx], y[validation_idx]


def augment_landmarks(X, copies=2, seed=42):
    rng = np.random.default_rng(seed)
    augmented = [X]
    for _ in range(copies):
        points = X.reshape(-1, 42, 3).copy()
        angles = rng.uniform(-0.18, 0.18, size=len(points))
        cos, sin = np.cos(angles), np.sin(angles)
        x, y = points[:, :, 0].copy(), points[:, :, 1].copy()
        points[:, :, 0] = x * cos[:, None] - y * sin[:, None]
        points[:, :, 1] = x * sin[:, None] + y * cos[:, None]
        points += rng.normal(0, 0.008, size=points.shape).astype(np.float32)
        augmented.append(points.reshape(-1, NUM_FEATURES))
    return np.concatenate(augmented, axis=0)


def build_model(input_dim=NUM_FEATURES, num_classes=len(ISL_LABELS)):
    import tensorflow as tf

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax'),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def confidence_temperature(probabilities, labels):
    temperatures = np.arange(0.5, 3.05, 0.05)
    logits = np.log(np.clip(probabilities, 1e-7, 1.0))
    losses = []
    for temperature in temperatures:
        scaled = logits / temperature
        scaled -= scaled.max(axis=1, keepdims=True)
        calibrated = np.exp(scaled)
        calibrated /= calibrated.sum(axis=1, keepdims=True)
        losses.append(-np.log(calibrated[np.arange(len(labels)), labels] + 1e-7).mean())
    return float(temperatures[int(np.argmin(losses))])


def train():
    """Train the Keras ISL gesture classifier."""
    import tensorflow as tf
    from sklearn.metrics import classification_report

    logger.info("Loading captured MediaPipe landmarks...")
    X, y, sessions = load_captured_data()
    present = Counter(y)
    logger.info(f"Loaded {len(X)} landmark samples across {len(present)} classes.")

    present_classes = sorted(list(set(y)))
    present_labels = [ISL_LABELS[i] for i in present_classes]
    num_classes = len(ISL_LABELS)

    # Train / val split
    X_train, X_val, y_train, y_val = split_by_session(X, y, sessions, validation_fraction=0.2)

    # Augmentation
    X_train_aug = augment_landmarks(X_train, copies=2)
    y_train_aug = np.tile(y_train, 3)
    logger.info(f"Training samples (augmented): {len(X_train_aug)}, Validation samples: {len(X_val)}")

    # Build model
    model = build_model(input_dim=NUM_FEATURES, num_classes=num_classes)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5),
    ]

    model.fit(
        X_train_aug, y_train_aug,
        validation_data=(X_val, y_val),
        epochs=80,
        batch_size=64,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluation
    probabilities = model.predict(X_val)
    predictions = probabilities.argmax(axis=1)
    accuracy = float((predictions == y_val).mean())
    temp = confidence_temperature(probabilities, y_val)

    report = classification_report(
        y_val, predictions,
        labels=present_classes,
        target_names=present_labels,
        output_dict=True,
        zero_division=0,
    )

    # Save model and metadata
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_PATH))

    meta = {
        'model_type': 'keras_dense',
        'train_samples': int(len(X_train_aug)),
        'val_samples': int(len(X_val)),
        'val_accuracy': accuracy,
        'temperature': temp,
        'num_classes': num_classes,
        'labels': ISL_LABELS,
        'input_features': NUM_FEATURES,
        'per_class': {
            label: {
                'precision': report[label]['precision'],
                'recall': report[label]['recall'],
                'f1-score': report[label]['f1-score'],
            }
            for label in present_labels if label in report
        },
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding='utf-8')

    logger.info('=' * 60)
    logger.info(f'Keras Validation Accuracy: {accuracy*100:.2f}% | Temperature: {temp:.2f}')
    logger.info('=' * 60)
    for label in present_labels:
        if label in report:
            logger.info(
                f"{label}  precision={report[label]['precision']:.3f}  recall={report[label]['recall']:.3f}  f1={report[label]['f1-score']:.3f}"
            )
    logger.info(f'Model saved to: {MODEL_PATH}')
    logger.info(f'Metadata saved to: {META_PATH}')


if __name__ == '__main__':
    train()
