"""
Sign-Bridge Flask API — ISL Word Recognition Service

Loads a trained model that classifies temporal sequences of
30 MediaPipe landmark frames into whole ISL word labels (e.g., HELLO,
NAMASTE, THANK_YOU).

Supports PyTorch (.pt), Scikit-Learn/XGBoost (.pkl), and Keras (.h5) model artifacts.
"""

import os
import json
import logging
import pickle
import numpy as np

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
CNN_LSTM_MODEL_PATH = os.path.join(MODEL_DIR, 'isl_cnn_lstm_word_model.pt')
CNN_LSTM_META_PATH = os.path.join(MODEL_DIR, 'cnn_lstm_training_meta.json')
PKL_MODEL_PATH = os.path.join(MODEL_DIR, 'isl_word_classifier.pkl')
PKL_META_PATH = os.path.join(MODEL_DIR, 'word_training_meta.json')
PT_MODEL_PATH = os.path.join(MODEL_DIR, 'isl_lstm_word_model.pt')
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, 'isl_lstm_word_model.h5')
LSTM_META_PATH = os.path.join(MODEL_DIR, 'lstm_training_meta.json')

SEQUENCE_LENGTH = 30
NUM_FEATURES = 126


class WordRecognizer:
    """
    ISL whole-word gesture recognizer using temporal landmark sequences.

    Accepts a sliding window of 30 frames (each 126 landmark features)
    and returns a word prediction with confidence score.
    """

    def __init__(self):
        self.model = None
        self.mode = None  # 'sklearn' | 'pytorch' | 'keras'
        self.metadata = {}
        self.labels = []
        self.is_available = False
        self._load()

    def _load(self):
        """Attempt to load the trained word model.

        Loading priority:
          1. CNN-BiLSTM hybrid (.pt) — best accuracy for dynamic words
          2. Scikit-Learn/XGBoost sequence model (.pkl)
          3. Plain Bi-LSTM PyTorch (.pt)
          4. Keras Bi-LSTM (.h5)
        """
        # 1. Try CNN-BiLSTM hybrid model (.pt) — highest priority
        if os.path.exists(CNN_LSTM_MODEL_PATH):
            try:
                import torch
                import torch.nn as nn

                checkpoint = torch.load(CNN_LSTM_MODEL_PATH, map_location='cpu')

                if checkpoint.get('model_type') == 'cnn_bilstm':
                    if os.path.exists(CNN_LSTM_META_PATH):
                        with open(CNN_LSTM_META_PATH, 'r', encoding='utf-8') as f:
                            self.metadata = json.load(f)

                    self.labels = checkpoint.get('word_labels') or self.metadata.get('labels', [])
                    num_classes = len(self.labels)
                    cnn_channels = checkpoint.get('cnn_channels', 128)
                    hidden_dim = checkpoint.get('hidden_dim', 128)

                    class CNNBiLSTMWordClassifier(nn.Module):
                        def __init__(self, input_dim=126, cnn_ch=128, hid=128, nc=17, nl=2, dp=0.3):
                            super().__init__()
                            self.cnn = nn.Sequential(
                                nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
                                nn.BatchNorm1d(64),
                                nn.ReLU(),
                                nn.Conv1d(64, cnn_ch, kernel_size=3, padding=1),
                                nn.BatchNorm1d(cnn_ch),
                                nn.ReLU(),
                                nn.Dropout(0.2),
                            )
                            self.lstm = nn.LSTM(
                                input_size=cnn_ch, hidden_size=hid, num_layers=nl,
                                batch_first=True, bidirectional=True,
                                dropout=dp if nl > 1 else 0.0
                            )
                            self.classifier = nn.Sequential(
                                nn.BatchNorm1d(hid * 2),
                                nn.Linear(hid * 2, 64), nn.ReLU(), nn.Dropout(dp),
                                nn.Linear(64, nc),
                            )

                        def forward(self, x):
                            x_cnn = self.cnn(x.permute(0, 2, 1))
                            lstm_out, _ = self.lstm(x_cnn.permute(0, 2, 1))
                            pooled = torch.mean(lstm_out, dim=1)
                            return self.classifier(pooled)

                    model = CNNBiLSTMWordClassifier(
                        cnn_ch=cnn_channels, hid=hidden_dim, nc=num_classes
                    )
                    model.load_state_dict(checkpoint['model_state_dict'])
                    model.eval()
                    self.model = model
                    self.mode = 'cnn_lstm'
                    self.is_available = True
                    logger.info(f"Loaded CNN-BiLSTM word model from {CNN_LSTM_MODEL_PATH} ({num_classes} classes)")
                    return
            except Exception as e:
                logger.warning(f"Failed to load CNN-BiLSTM word model: {e}")

        # 2. Try Scikit-Learn/XGBoost sequence model (.pkl)
        if os.path.exists(PKL_MODEL_PATH):
            try:
                with open(PKL_MODEL_PATH, 'rb') as f:
                    self.model = pickle.load(f)
                if os.path.exists(PKL_META_PATH):
                    with open(PKL_META_PATH, 'r', encoding='utf-8') as f:
                        self.metadata = json.load(f)
                self.labels = self.metadata.get('labels', [])
                self.mode = 'sklearn'
                self.is_available = True
                logger.info(f"Loaded Sequence Word Classifier from {PKL_MODEL_PATH} ({len(self.labels)} classes: {', '.join(self.labels[:10])})")
                return
            except Exception as e:
                logger.warning(f"Failed to load PKL word model: {e}")

        # 3. Try PyTorch model (.pt)
        if os.path.exists(PT_MODEL_PATH):
            try:
                import torch
                import torch.nn as nn

                class BiLSTMWordClassifier(nn.Module):
                    def __init__(self, input_dim=126, hidden_dim=128, num_classes=17, num_layers=2, dropout=0.3):
                        super().__init__()
                        self.lstm = nn.LSTM(
                            input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0
                        )
                        self.batch_norm = nn.BatchNorm1d(hidden_dim * 2)
                        self.fc1 = nn.Linear(hidden_dim * 2, 64)
                        self.relu = nn.ReLU()
                        self.dropout = nn.Dropout(dropout)
                        self.fc2 = nn.Linear(64, num_classes)
                        
                    def forward(self, x):
                        lstm_out, _ = self.lstm(x)
                        pooled = torch.mean(lstm_out, dim=1)
                        normed = self.batch_norm(pooled)
                        out = self.fc1(normed)
                        out = self.relu(out)
                        out = self.dropout(out)
                        return self.fc2(out)

                if os.path.exists(LSTM_META_PATH):
                    with open(LSTM_META_PATH, 'r', encoding='utf-8') as f:
                        self.metadata = json.load(f)

                checkpoint = torch.load(PT_MODEL_PATH, map_location='cpu')
                self.labels = checkpoint.get('word_labels') or self.metadata.get('labels', [])
                num_classes = len(self.labels)

                model = BiLSTMWordClassifier(num_classes=num_classes)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                self.model = model
                self.mode = 'pytorch'
                self.is_available = True
                logger.info(f"Loaded PyTorch Bi-LSTM word model from {PT_MODEL_PATH}")
                return
            except Exception as e:
                logger.warning(f"Failed to load PyTorch LSTM word model: {e}")

        # 3. Try Keras model fallback (.h5)
        if os.path.exists(LSTM_MODEL_PATH):
            try:
                import tensorflow as tf
                if os.path.exists(LSTM_META_PATH):
                    with open(LSTM_META_PATH, 'r', encoding='utf-8') as f:
                        self.metadata = json.load(f)

                self.model = tf.keras.models.load_model(LSTM_MODEL_PATH)
                self.labels = self.metadata.get('labels', [])
                self.mode = 'keras'
                self.is_available = True
                logger.info(f"Loaded Keras Bi-LSTM word model from {LSTM_MODEL_PATH}")
                return
            except Exception as e:
                logger.warning(f"Failed to load Keras LSTM word model: {e}")

        logger.info("Word model not found. Word recognition is disabled.")

    def predict(self, frame_sequence):
        """
        Predict a word from a sequence of landmark frames.

        Args:
            frame_sequence: list of 30 frames, each frame is a list/array
                            of 126 floats (42 landmarks x 3 coordinates).
        """
        if not self.is_available or self.model is None:
            return None

        try:
            from services.translator_model import normalize_landmarks

            arr = np.asarray(frame_sequence, dtype=np.float32)

            if arr.ndim == 2:
                if arr.shape != (SEQUENCE_LENGTH, NUM_FEATURES):
                    if arr.shape[1] != NUM_FEATURES:
                        return None
                    arr = self._pad_or_trim(arr)
            elif arr.ndim == 3:
                arr = arr.reshape(-1, NUM_FEATURES)
                arr = self._pad_or_trim(arr)
            else:
                return None

            # Guard: If no valid hand coordinates exist across the sequence, return no prediction
            if not np.any(arr != 0):
                return {
                    'word': '?',
                    'confidence': 0.0,
                    'mode': self.mode,
                    'all_scores': {},
                }

            normalized = []
            has_valid_hand = False
            for frame in arr:
                norm = normalize_landmarks(frame)
                if norm is not None:
                    has_valid_hand = True
                    normalized.append(norm)
                else:
                    normalized.append(np.zeros(NUM_FEATURES, dtype=np.float32))

            if not has_valid_hand:
                return {
                    'word': '?',
                    'confidence': 0.0,
                    'mode': self.mode,
                    'all_scores': {},
                }

            norm_arr = np.asarray(normalized, dtype=np.float32)
            model = self.model

            if self.mode == 'sklearn':
                flat_in = norm_arr.flatten().reshape(1, -1)
                probs = model.predict_proba(flat_in)[0]
            elif self.mode in ('pytorch', 'cnn_lstm'):
                import torch
                input_data = norm_arr.reshape(1, SEQUENCE_LENGTH, NUM_FEATURES)
                tensor_in = torch.tensor(input_data, dtype=torch.float32)
                with torch.no_grad():
                    logits = model(tensor_in)
                    probs = torch.softmax(logits, dim=1).numpy()[0]
            else:
                input_data = norm_arr.reshape(1, SEQUENCE_LENGTH, NUM_FEATURES)
                probs = model.predict(input_data, verbose=0)[0]

            top_idx = int(np.argmax(probs))
            confidence = float(probs[top_idx])
            word = self.labels[top_idx] if top_idx < len(self.labels) else '?'

            sorted_indices = np.argsort(probs)[::-1][:5]
            top_scores = {
                self.labels[i]: round(float(probs[i]), 4)
                for i in sorted_indices if i < len(self.labels)
            }

            return {
                'word': word,
                'confidence': round(confidence, 4),
                'mode': self.mode,
                'all_scores': top_scores,
            }

        except Exception as e:
            logger.error(f"Word recognition prediction failed: {e}")
            return None

    def _pad_or_trim(self, frames):
        """Ensure frames array is exactly SEQUENCE_LENGTH frames long."""
        if len(frames) < SEQUENCE_LENGTH:
            pad = np.zeros((SEQUENCE_LENGTH - len(frames), NUM_FEATURES), dtype=np.float32)
            return np.concatenate([frames, pad], axis=0)
        elif len(frames) > SEQUENCE_LENGTH:
            indices = np.linspace(0, len(frames) - 1, SEQUENCE_LENGTH, dtype=int)
            return frames[indices]
        return frames

    def get_info(self):
        """Return word recognizer metadata."""
        return {
            'available': self.is_available,
            'model_mode': self.mode,
            'labels': self.labels,
            'num_classes': len(self.labels),
            'sequence_length': SEQUENCE_LENGTH,
            'input_features': NUM_FEATURES,
            'validation_accuracy': self.metadata.get('val_accuracy'),
        }
