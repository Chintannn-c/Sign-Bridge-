"""
Train a Bidirectional LSTM ISL word classifier from multi-frame landmark sequences.

This classifier processes temporal sequences of 30 MediaPipe landmark frames
to recognize whole ISL words and gestures that involve hand motion over time
(e.g., HELLO, NAMASTE, THANK_YOU, HELP, WASHROOM).

Unlike the single-frame alphabet classifiers (XGBoost/Keras), this model
captures the spatial-temporal trajectory of hand movements.

Input shape:  (batch_size, 30, 126)  — 30 frames x 126 landmark features
Output shape: (batch_size, num_word_classes)

Usage:
    cd backend
    python train_model_lstm.py

Prerequisites:
    - Word-level dataset in backend/dataset_words/<WORD_LABEL>/*.json
    - Each JSON file: { "word": "HELLO", "session_id": "...", "frame_sequences": [ [[126 floats], ...30 frames], ... ] }

Output:
    backend/models/isl_lstm_word_model.h5
    backend/models/lstm_training_meta.json
"""

import json
import logging
import os
from collections import Counter
from pathlib import Path

import numpy as np

from services.translator_model import normalize_landmarks

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
WORD_DATASET_DIR = SCRIPT_DIR / 'dataset_words'
MODEL_DIR = SCRIPT_DIR / 'models'
MODEL_PATH = MODEL_DIR / 'isl_lstm_word_model.h5'
META_PATH = MODEL_DIR / 'lstm_training_meta.json'

NUM_FEATURES = 126
SEQUENCE_LENGTH = 30  # 30 frames = ~1 second at 30 FPS


def load_word_data(data_dir=None):
    """Load multi-frame word sequence data from dataset_words directory.

    Expected directory structure:
        dataset_words/
            HELLO/
                session1.json
                session2.json
            NAMASTE/
                session1.json
            ...

    Each JSON file format:
    {
        "word": "HELLO",
        "session_id": "unique_session_id",
        "frame_sequences": [
            [ [126 floats], [126 floats], ... (30 frames) ],
            ...
        ]
    }
    """
    if data_dir is None:
        data_dir = WORD_DATASET_DIR

    if not Path(data_dir).exists():
        raise RuntimeError(
            f'Word dataset directory not found: {data_dir}\n'
            f'Create it and add word-level recordings first.\n'
            f'Expected structure: dataset_words/<WORD_LABEL>/*.json'
        )

    sequences = []
    labels = []
    sessions = []
    word_labels = sorted([
        d.name for d in Path(data_dir).iterdir()
        if d.is_dir() and not d.name.startswith('.')
    ])

    if not word_labels:
        raise RuntimeError(f'No word label directories found in {data_dir}')

    logger.info(f'Found word classes: {word_labels}')
    label_to_idx = {label: i for i, label in enumerate(word_labels)}

    for word_dir in sorted(Path(data_dir).iterdir()):
        if not word_dir.is_dir() or word_dir.name.startswith('.'):
            continue

        word = word_dir.name.upper()
        if word not in label_to_idx:
            continue

        for path in sorted(word_dir.glob('*.json')):
            try:
                payload = json.loads(path.read_text(encoding='utf-8'))
                session = str(payload.get('session_id') or path.stem)

                for seq in payload.get('frame_sequences', []):
                    frames = np.asarray(seq, dtype=np.float32)

                    # Pad or trim to SEQUENCE_LENGTH
                    if frames.ndim != 2 or frames.shape[1] != NUM_FEATURES:
                        continue

                    if len(frames) < SEQUENCE_LENGTH:
                        # Pad with zeros at the end
                        pad = np.zeros((SEQUENCE_LENGTH - len(frames), NUM_FEATURES), dtype=np.float32)
                        frames = np.concatenate([frames, pad], axis=0)
                    elif len(frames) > SEQUENCE_LENGTH:
                        # Subsample uniformly to SEQUENCE_LENGTH
                        indices = np.linspace(0, len(frames) - 1, SEQUENCE_LENGTH, dtype=int)
                        frames = frames[indices]

                    # Normalize each frame
                    normalized_frames = []
                    for frame in frames:
                        normalized = normalize_landmarks(frame)
                        if normalized is not None:
                            normalized_frames.append(normalized)
                        else:
                            normalized_frames.append(np.zeros(NUM_FEATURES, dtype=np.float32))

                    sequences.append(np.array(normalized_frames))
                    labels.append(label_to_idx[word])
                    sessions.append(f'{word}:{session}')

            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                logger.warning('Skipping %s: %s', path, exc)

    if not sequences:
        raise RuntimeError(f'No valid word sequences found in {data_dir}.')

    return (
        np.asarray(sequences, dtype=np.float32),
        np.asarray(labels, dtype=np.int32),
        np.asarray(sessions),
        word_labels,
    )


def split_by_session(X, y, sessions, word_labels, validation_fraction=0.2, seed=42):
    """Split data by recording session to prevent data leakage."""
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []

    for label_idx in range(len(word_labels)):
        mask = y == label_idx
        groups = np.unique(sessions[mask])
        if len(groups) < 2:
            logger.warning(
                '%s has only %d session(s) — all assigned to train.',
                word_labels[label_idx], len(groups),
            )
            train_idx.extend(np.flatnonzero(mask).tolist())
            continue

        rng.shuffle(groups)
        n_val = max(1, round(len(groups) * validation_fraction))
        val_groups = set(groups[:n_val])
        indices = np.flatnonzero(mask)
        val_idx.extend(i for i in indices if sessions[i] in val_groups)
        train_idx.extend(i for i in indices if sessions[i] not in val_groups)

    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]


def build_lstm_model(sequence_length=SEQUENCE_LENGTH, input_dim=NUM_FEATURES, num_classes=10):
    """Build a 2-layer Bidirectional LSTM classifier.

    Architecture:
        Input (30, 126) → Bi-LSTM(128) → Dropout → Bi-LSTM(64) → Dense(64) → Dense(num_classes)
    """
    import tensorflow as tf

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(sequence_length, input_dim)),

        # First Bi-LSTM layer — returns full sequence
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)
        ),
        tf.keras.layers.BatchNormalization(),

        # Second Bi-LSTM layer — returns final hidden state
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, return_sequences=False, dropout=0.3, recurrent_dropout=0.2)
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        # Classification head
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax'),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def train():
    """Train the Bi-LSTM word classifier."""
    import tensorflow as tf
    from sklearn.metrics import classification_report

    tf.keras.utils.set_random_seed(42)

    # ── Load Data ─────────────────────────────────────────────────────────
    X, y, sessions, word_labels = load_word_data()
    logger.info('Loaded %d sequences across %d word classes.', len(X), len(word_labels))

    present = Counter(y)
    for idx, label in enumerate(word_labels):
        logger.info('  %s: %d samples', label, present.get(idx, 0))

    # ── Split ─────────────────────────────────────────────────────────────
    X_train, X_val, y_train, y_val = split_by_session(X, y, sessions, word_labels)
    logger.info('Train: %d, Validation: %d', len(X_train), len(X_val))

    # ── Build & Train ─────────────────────────────────────────────────────
    model = build_lstm_model(num_classes=len(word_labels))
    model.summary()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy', patience=15, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5
            ),
        ],
        verbose=1,
    )

    # ── Evaluate ──────────────────────────────────────────────────────────
    predictions = model.predict(X_val, verbose=0).argmax(axis=1)
    accuracy = float((predictions == y_val).mean())
    report = classification_report(
        y_val, predictions,
        labels=range(len(word_labels)),
        target_names=word_labels,
        output_dict=True,
        zero_division=0,
    )

    # ── Save ──────────────────────────────────────────────────────────────
    MODEL_DIR.mkdir(exist_ok=True)
    model.save(MODEL_PATH)
    META_PATH.write_text(json.dumps({
        'model_type': 'bi_lstm',
        'train_samples': int(len(X_train)),
        'val_samples': int(len(X_val)),
        'val_accuracy': accuracy,
        'num_classes': len(word_labels),
        'labels': word_labels,
        'sequence_length': SEQUENCE_LENGTH,
        'input_features': NUM_FEATURES,
        'epochs_trained': len(history.history['loss']),
        'data_source': 'backend/dataset_words',
        'preprocessing': 'wrist_center_scale_v1',
        'per_class': {
            label: {
                'precision': report[label]['precision'],
                'recall': report[label]['recall'],
                'f1-score': report[label]['f1-score'],
            }
            for label in word_labels
        },
    }, indent=2), encoding='utf-8')

    logger.info('=' * 60)
    logger.info('Bi-LSTM Word Validation Accuracy: %.4f', accuracy)
    logger.info('=' * 60)
    for label in word_labels:
        logger.info(
            '%s  precision=%.3f  recall=%.3f  f1=%.3f',
            label, report[label]['precision'], report[label]['recall'], report[label]['f1-score'],
        )
    logger.info('Model saved to: %s', MODEL_PATH)


if __name__ == '__main__':
    train()
