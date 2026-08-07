"""
Sign-Bridge Flask API — ISL Word Recognition Service

Loads a trained Bi-LSTM model that classifies temporal sequences of
30 MediaPipe landmark frames into whole ISL word labels (e.g., HELLO,
NAMASTE, THANK_YOU).

This service is separate from the alphabet TranslatorModel to maintain
clean separation of concerns:
  - TranslatorModel: single-frame → letter (A-Z)
  - WordRecognizer:  30-frame sequence → word (HELLO, NAMASTE, etc.)

If no trained LSTM model file exists, the service gracefully
returns None (no impact on existing alphabet recognition flow).
"""

import os
import json
import logging
import numpy as np

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, 'isl_lstm_word_model.h5')
LSTM_META_PATH = os.path.join(MODEL_DIR, 'lstm_training_meta.json')

SEQUENCE_LENGTH = 30
NUM_FEATURES = 126


class WordRecognizer:
    """
    ISL whole-word gesture recognizer using temporal landmark sequences.

    Accepts a sliding window of 30 frames (each 126 landmark features)
    and returns a word prediction with confidence score.

    If no trained model exists, all methods return None gracefully.
    """

    def __init__(self):
        self.model = None
        self.metadata = {}
        self.labels = []
        self.is_available = False
        self._load()

    def _load(self):
        """Attempt to load the trained Bi-LSTM word model."""
        if not os.path.exists(LSTM_MODEL_PATH):
            logger.info(
                "LSTM word model not found at %s. "
                "Word recognition is disabled (alphabet-only mode).",
                LSTM_MODEL_PATH,
            )
            return

        try:
            import tensorflow as tf

            if os.path.exists(LSTM_META_PATH):
                with open(LSTM_META_PATH, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)

            self.model = tf.keras.models.load_model(LSTM_MODEL_PATH)
            self.labels = self.metadata.get('labels', [])
            self.is_available = True
            logger.info(
                "Loaded Bi-LSTM word model from %s (%d word classes: %s)",
                LSTM_MODEL_PATH,
                len(self.labels),
                ', '.join(self.labels[:10]),
            )
        except Exception as e:
            logger.warning("Failed to load LSTM word model: %s", e)
            self.model = None
            self.is_available = False

    def predict(self, frame_sequence):
        """
        Predict a word from a sequence of landmark frames.

        Args:
            frame_sequence: list of 30 frames, each frame is a list/array
                            of 126 floats (42 landmarks x 3 coordinates).

        Returns:
            dict or None:
                {
                    'word': str,           # Predicted word label
                    'confidence': float,   # 0.0 to 1.0
                    'all_scores': dict,    # Word -> score mapping (top 5)
                }
                Returns None if model is not available or input is invalid.
        """
        if not self.is_available or self.model is None:
            return None

        try:
            from services.translator_model import normalize_landmarks

            arr = np.asarray(frame_sequence, dtype=np.float32)

            # Handle different input shapes
            if arr.ndim == 2:
                if arr.shape != (SEQUENCE_LENGTH, NUM_FEATURES):
                    # Try to pad/trim
                    if arr.shape[1] != NUM_FEATURES:
                        return None
                    arr = self._pad_or_trim(arr)
            elif arr.ndim == 3:
                arr = arr.reshape(-1, NUM_FEATURES)
                arr = self._pad_or_trim(arr)
            else:
                return None

            # Normalize each frame
            normalized = []
            for frame in arr:
                norm = normalize_landmarks(frame)
                if norm is not None:
                    normalized.append(norm)
                else:
                    normalized.append(np.zeros(NUM_FEATURES, dtype=np.float32))

            input_data = np.asarray(normalized, dtype=np.float32)
            input_data = input_data.reshape(1, SEQUENCE_LENGTH, NUM_FEATURES)

            # Predict
            probs = self.model.predict(input_data, verbose=0)[0]
            top_idx = int(np.argmax(probs))
            confidence = float(probs[top_idx])
            word = self.labels[top_idx] if top_idx < len(self.labels) else '?'

            # Top 5 scores
            sorted_indices = np.argsort(probs)[::-1][:5]
            top_scores = {
                self.labels[i]: round(float(probs[i]), 4)
                for i in sorted_indices if i < len(self.labels)
            }

            return {
                'word': word,
                'confidence': round(confidence, 4),
                'all_scores': top_scores,
            }

        except Exception as e:
            logger.error("Word recognition prediction failed: %s", e)
            return None

    def _pad_or_trim(self, frames):
        """Ensure frames array is exactly SEQUENCE_LENGTH frames long."""
        if len(frames) < SEQUENCE_LENGTH:
            pad = np.zeros(
                (SEQUENCE_LENGTH - len(frames), NUM_FEATURES),
                dtype=np.float32,
            )
            return np.concatenate([frames, pad], axis=0)
        elif len(frames) > SEQUENCE_LENGTH:
            indices = np.linspace(0, len(frames) - 1, SEQUENCE_LENGTH, dtype=int)
            return frames[indices]
        return frames

    def get_info(self):
        """Return word recognizer metadata."""
        return {
            'available': self.is_available,
            'model_path': LSTM_MODEL_PATH if self.is_available else None,
            'model_type': 'bi_lstm',
            'labels': self.labels,
            'num_classes': len(self.labels),
            'sequence_length': SEQUENCE_LENGTH,
            'input_features': NUM_FEATURES,
            'validation_accuracy': self.metadata.get('val_accuracy'),
        }
