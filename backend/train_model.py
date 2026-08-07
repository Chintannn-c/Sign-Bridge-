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
ROOT_DATASET_DIR = SCRIPT_DIR.parent / 'dataset' / 'ISL_Landmarks'
DATA_DIR = SCRIPT_DIR / 'dataset_collected'
MODEL_DIR = SCRIPT_DIR / 'models'
MODEL_PATH = MODEL_DIR / 'isl_gesture_model.h5'
META_PATH = MODEL_DIR / 'training_meta.json'
NUM_FEATURES = 126


def load_captured_data(data_dir=None):
    if data_dir is None:
        data_dir = ROOT_DATASET_DIR if ROOT_DATASET_DIR.exists() else DATA_DIR
    samples, labels, sessions = [], [], []
    for path in sorted(Path(data_dir).glob('*/*.json')):
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
        raise RuntimeError(f'No valid captured 126-value landmark frames found in {data_dir}.')
    return np.asarray(samples, dtype=np.float32), np.asarray(labels, dtype=np.int32), np.asarray(sessions)


def split_by_session(X, y, sessions, validation_fraction=0.2, seed=42):
    rng = np.random.default_rng(seed)
    train_idx, validation_idx = [], []
    for label in range(len(ISL_LABELS)):
        groups = np.unique(sessions[y == label])
        if len(groups) < 2:
            raise RuntimeError(f'{ISL_LABELS[label]} needs at least two recording sessions for real validation.')
        rng.shuffle(groups)
        validation_groups = set(groups[:max(1, round(len(groups) * validation_fraction))])
        indices = np.flatnonzero(y == label)
        validation_idx.extend(i for i in indices if sessions[i] in validation_groups)
        train_idx.extend(i for i in indices if sessions[i] not in validation_groups)

    if not train_idx or not validation_idx:
        raise RuntimeError('Could not create non-empty train and validation sets from recording sessions.')
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
    import tensorflow as tf
    from sklearn.metrics import classification_report

    tf.keras.utils.set_random_seed(42)
    X, y, sessions = load_captured_data()
    present = Counter(y)
    missing = [label for index, label in enumerate(ISL_LABELS) if index not in present]
    if missing:
        raise RuntimeError(f'Missing captured classes: {", ".join(missing)}.')

    X_train, X_val, y_train, y_val = split_by_session(X, y, sessions)
    X_train = augment_landmarks(X_train)
    y_train = np.tile(y_train, 3)
    logger.info('Real captured frames: train=%d validation=%d', len(X_train), len(X_val))

    model = build_model()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=80,
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4),
        ],
        verbose=1,
    )

    probabilities = model.predict(X_val, verbose=0)
    predictions = probabilities.argmax(axis=1)
    report = classification_report(y_val, predictions, labels=range(len(ISL_LABELS)), target_names=ISL_LABELS, output_dict=True, zero_division=0)
    accuracy = float((predictions == y_val).mean())
    temperature = confidence_temperature(probabilities, y_val)

    MODEL_DIR.mkdir(exist_ok=True)
    model.save(MODEL_PATH)
    META_PATH.write_text(json.dumps({
        'train_samples': int(len(X_train)),
        'val_samples': int(len(X_val)),
        'val_accuracy': accuracy,
        'num_classes': len(ISL_LABELS),
        'labels': ISL_LABELS,
        'input_features': NUM_FEATURES,
        'epochs_trained': len(history.history['loss']),
        'data_source': 'backend/dataset_collected',
        'preprocessing': 'wrist_center_scale_v1',
        'confidence_temperature': temperature,
        'per_class': {label: {'precision': report[label]['precision'], 'recall': report[label]['recall']} for label in ISL_LABELS},
    }, indent=2), encoding='utf-8')
    logger.info('Validation accuracy: %.4f; temperature: %.2f', accuracy, temperature)
    for label in ISL_LABELS:
        logger.info('%s precision=%.3f recall=%.3f', label, report[label]['precision'], report[label]['recall'])


if __name__ == '__main__':
    train()
