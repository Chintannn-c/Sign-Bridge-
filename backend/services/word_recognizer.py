"""
Sign-Bridge Flask API — ISL Word Recognition Service

Loads a trained model that classifies temporal sequences of
30 MediaPipe landmark frames into whole ISL word labels (e.g., HELLO,
NAMASTE, THANK_YOU).

Supports PyTorch (.pt) CNN-BiLSTM / ST-GCN and Scikit-Learn/XGBoost (.pkl) artifacts.
"""

import os
import json
import logging
import pickle
import threading
import numpy as np

try:
    from .gesture_config import CONFIDENCE_THRESHOLD_WORD, MARGIN_THRESHOLD_WORD
except ImportError:
    from services.gesture_config import CONFIDENCE_THRESHOLD_WORD, MARGIN_THRESHOLD_WORD

try:
    from .translator_model import normalize_landmarks_batch
except ImportError:
    from services.translator_model import normalize_landmarks_batch

try:
    from .feature_extractor import transform_to_kinetic_invariants
except ImportError:
    from services.feature_extractor import transform_to_kinetic_invariants

try:
    from ..models.st_gcn import STGCN_BiLSTM_WordClassifier
    from ..models.cnn_bilstm import CNNBiLSTMWordClassifier
except (ImportError, ValueError):
    try:
        from models.st_gcn import STGCN_BiLSTM_WordClassifier
        from models.cnn_bilstm import CNNBiLSTMWordClassifier
    except (ImportError, ValueError):
        STGCN_BiLSTM_WordClassifier = None
        CNNBiLSTMWordClassifier = None

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
CNN_LSTM_MODEL_PATH = os.path.join(MODEL_DIR, 'isl_cnn_lstm_word_model.pt')
CNN_LSTM_META_PATH = os.path.join(MODEL_DIR, 'cnn_lstm_training_meta.json')
PKL_MODEL_PATH = os.path.join(MODEL_DIR, 'isl_word_classifier.pkl')
PKL_META_PATH = os.path.join(MODEL_DIR, 'word_training_meta.json')

SEQUENCE_LENGTH = 30
NUM_FEATURES = 126

# Gating & Velocity Defaults
MIN_MOTION_VELOCITY = 0.008
MAX_SINGLE_FRAME_JUMP = 0.15
MAX_WORD_ENTROPY = 2.75
ENTROPY_MAX_RATIO = 0.85
STATIC_POSE_WORDS = frozenset({'NAMASTE', 'ME', 'INDIA'})


def compute_wrist_relative_motion(arr: np.ndarray, max_jump: float = MAX_SINGLE_FRAME_JUMP) -> float:
    """
    Computes motion velocity across sequence using wrist-relative per-hand deltas.
    Clips single-frame outlier jumps (> max_jump) so tracking glitches, slot-swaps,
    or sudden hand entries/exits cannot artificially trip the motion gate.
    Vectorized with NumPy.
    """
    pts = np.asarray(arr, dtype=np.float32).reshape(-1, 2, 21, 3)
    T = pts.shape[0]
    if T < 2:
        return 0.0

    present = np.any(pts != 0, axis=(2, 3))  # shape: (T, 2)
    both_present = present[:-1] & present[1:]  # shape: (T - 1, 2)

    if not np.any(both_present):
        return 0.0

    # Wrist motion between consecutive frames
    w_t = pts[:-1, :, 0, :]   # shape: (T - 1, 2, 3)
    w_t1 = pts[1:, :, 0, :]   # shape: (T - 1, 2, 3)
    d_w = np.linalg.norm(w_t1 - w_t, axis=-1)  # shape: (T - 1, 2)
    d_w_clipped = np.clip(d_w, 0.0, max_jump)

    # Relative coordinates (relative to wrist)
    rel = pts - pts[:, :, 0:1, :]  # shape: (T, 2, 21, 3)
    rel_diff = rel[1:] - rel[:-1]   # shape: (T - 1, 2, 21, 3)
    d_rel = np.linalg.norm(rel_diff, axis=-1)  # shape: (T - 1, 2, 21)
    d_rel_clipped = np.clip(d_rel, 0.0, max_jump)
    mean_d_rel = np.mean(d_rel_clipped, axis=-1)  # shape: (T - 1, 2)

    # Combined wrist + joint displacement
    deltas = mean_d_rel + d_w_clipped  # shape: (T - 1, 2)
    valid_deltas = deltas[both_present]

    return float(np.mean(valid_deltas)) if len(valid_deltas) > 0 else 0.0


class WordRecognizer:
    """
    ISL whole-word gesture recognizer using temporal landmark sequences.

    Accepts a sliding window of MediaPipe landmark frames (126 features)
    and returns a word prediction with confidence, margin, and entropy metrics.
    Supports PyTorch CNN-BiLSTM / ST-GCN and Scikit-Learn models.
    """

    def __init__(self, min_motion_velocity: float = MIN_MOTION_VELOCITY):
        self._lock = threading.Lock()
        self.min_motion_velocity = min_motion_velocity
        self.model = None
        self.mode = None  # 'sklearn' | 'cnn_lstm' | 'stgcn_bilstm'
        self.metadata = {}
        self.labels = []
        self.input_dim = NUM_FEATURES
        self.is_available = False
        self._load()

    def _make_rejected_result(
        self,
        reason: str,
        motion_velocity: float = 0.0,
        confidence: float = 0.0,
        margin: float = 0.0,
        entropy: float = 0.0,
        all_scores: dict | None = None
    ) -> dict:
        return {
            'word': '?',
            'confidence': round(float(confidence), 4),
            'margin': round(float(margin), 4),
            'entropy': round(float(entropy), 4),
            'mode': self.mode,
            'rejected': True,
            'rejection_reason': reason,
            'motion_velocity': round(float(motion_velocity), 5),
            'all_scores': all_scores or {},
        }

    def _load(self):
        """Attempt to load the trained word model."""
        with self._lock:
            loaded_model = None
            loaded_mode = None
            loaded_metadata = {}
            loaded_labels = []
            loaded_input_dim = NUM_FEATURES
            loaded_available = False

            # 1. Try PyTorch hybrid model (.pt) — highest priority
            if os.path.exists(CNN_LSTM_MODEL_PATH):
                try:
                    import torch
                    checkpoint = torch.load(CNN_LSTM_MODEL_PATH, map_location='cpu', weights_only=True)
                    m_type = checkpoint.get('model_type', 'cnn_bilstm')

                    if m_type in ('stgcn_bilstm', 'cnn_bilstm'):
                        if os.path.exists(CNN_LSTM_META_PATH):
                            with open(CNN_LSTM_META_PATH, 'r', encoding='utf-8') as f:
                                loaded_metadata = json.load(f)

                        loaded_labels = checkpoint.get('word_labels') or loaded_metadata.get('labels', [])
                        state_dict = checkpoint.get('model_state_dict', {})
                        input_dim = checkpoint.get('input_dim', NUM_FEATURES)

                        num_classes = len(loaded_labels)
                        if num_classes == 0 and state_dict:
                            for k in reversed(list(state_dict.keys())):
                                if k.startswith('classifier.') and k.endswith('.weight'):
                                    num_classes = state_dict[k].shape[0]
                                    break
                            if num_classes > 0 and not loaded_labels:
                                loaded_labels = [f"CLASS_{i}" for i in range(num_classes)]

                        nl = checkpoint.get('num_layers', 2)
                        dp = checkpoint.get('dropout', 0.3)

                        if m_type == 'stgcn_bilstm':
                            if STGCN_BiLSTM_WordClassifier is None:
                                raise ImportError("STGCN_BiLSTM_WordClassifier could not be imported")
                            gcn_ch = checkpoint.get('gcn_channels', 64)
                            hid = checkpoint.get('hidden_dim', 128)
                            model = STGCN_BiLSTM_WordClassifier(
                                in_dim=input_dim,
                                num_classes=num_classes,
                                gcn_channels=gcn_ch,
                                lstm_hidden=hid,
                                num_layers=nl,
                                dropout=dp,
                            )
                            model.load_state_dict(state_dict)
                            model.eval()
                            loaded_model = model
                            loaded_mode = 'stgcn_bilstm'
                            loaded_input_dim = input_dim
                            loaded_available = True
                            logger.info("Loaded ST-GCN + BiLSTM word model from %s (%d classes, in_dim=%d)",
                                        CNN_LSTM_MODEL_PATH, num_classes, input_dim)

                        elif m_type == 'cnn_bilstm':
                            if CNNBiLSTMWordClassifier is None:
                                raise ImportError("CNNBiLSTMWordClassifier could not be imported")
                            cnn_channels = checkpoint.get('cnn_channels', 128)
                            hidden_dim = checkpoint.get('hidden_dim', 128)
                            use_attn = checkpoint.get('has_attention', False) or ('attn.0.weight' in state_dict)

                            model = CNNBiLSTMWordClassifier(
                                in_dim=input_dim,
                                cnn_ch=cnn_channels,
                                hid=hidden_dim,
                                nc=num_classes,
                                nl=nl,
                                dp=dp,
                                attn_enabled=use_attn,
                            )
                            model.load_state_dict(state_dict)
                            model.eval()
                            loaded_model = model
                            loaded_mode = 'cnn_lstm'
                            loaded_input_dim = input_dim
                            loaded_available = True
                            logger.info("Loaded CNN-BiLSTM word model from %s (%d classes, in_dim=%d, attn=%s)",
                                        CNN_LSTM_MODEL_PATH, num_classes, input_dim, use_attn)
                    else:
                        logger.warning("Unrecognised model_type in checkpoint %s: %s", CNN_LSTM_MODEL_PATH, m_type)

                except Exception as e:
                    logger.warning("Failed to load CNN-BiLSTM word model: %s", e, exc_info=True)

            # 2. Try Scikit-Learn/XGBoost sequence model (.pkl)
            if not loaded_available and os.path.exists(PKL_MODEL_PATH):
                try:
                    with open(PKL_MODEL_PATH, 'rb') as f:
                        loaded_model = pickle.load(f)
                    if os.path.exists(PKL_META_PATH):
                        with open(PKL_META_PATH, 'r', encoding='utf-8') as f:
                            loaded_metadata = json.load(f)
                    loaded_labels = loaded_metadata.get('labels', [])
                    loaded_mode = 'sklearn'
                    loaded_available = True
                    logger.info("Loaded Sequence Word Classifier from %s (%d classes)",
                                PKL_MODEL_PATH, len(loaded_labels))
                except Exception as e:
                    logger.warning("Failed to load PKL word model: %s", e, exc_info=True)

            if loaded_available:
                self.model = loaded_model
                self.mode = loaded_mode
                self.metadata = loaded_metadata
                self.labels = loaded_labels
                self.input_dim = loaded_input_dim
                self.is_available = True
            else:
                self.model = None
                self.mode = None
                self.metadata = {}
                self.labels = []
                self.input_dim = NUM_FEATURES
                self.is_available = False
                logger.info("Word model not found. Word recognition is disabled.")

    def predict(self, frame_sequence, body_anchors=None, min_motion_velocity=None):
        """
        Predict a word from a sequence of landmark frames with optional MediaPipe Holistic body anchors.

        Args:
            frame_sequence: list or array of landmark frames (126 features each).
            body_anchors: optional dict of upper-body relative anchors from MediaPipe Holistic.
            min_motion_velocity: optional override for motion velocity threshold.

        Returns:
            Dict containing word, confidence, margin, entropy, mode, rejected,
            rejection_reason, motion_velocity, and all_scores.
        """
        with self._lock:
            if not self.is_available or self.model is None:
                return self._make_rejected_result("model_not_available")

        try:
            arr = np.asarray(frame_sequence, dtype=np.float32)

            # Check and sanitize non-finite values
            if not np.all(np.isfinite(arr)):
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

            if arr.ndim == 2:
                if arr.shape[1] != NUM_FEATURES:
                    return self._make_rejected_result("invalid_feature_dim")
            elif arr.ndim == 3:
                arr = arr.reshape(-1, NUM_FEATURES)
            else:
                return self._make_rejected_result("invalid_input_shape")

            # Guard: If no valid hand coordinates exist across the sequence, return rejected prediction
            if not np.any(arr != 0):
                return self._make_rejected_result("no_hands_detected")

            # Compute motion gate on pre-resample sequence
            motion_velocity = compute_wrist_relative_motion(arr, MAX_SINGLE_FRAME_JUMP)

            # Apply consistent _pad_or_trim across all sequence lengths (including 30 frames)
            arr = self._pad_or_trim(arr)

            norm_arr = normalize_landmarks_batch(arr)
            if norm_arr is None:
                return self._make_rejected_result("normalization_failed", motion_velocity=motion_velocity)

            with self._lock:
                model = self.model
                mode = self.mode
                labels = list(self.labels)
                input_dim = getattr(self, 'input_dim', NUM_FEATURES)

            if mode == 'sklearn':
                flat_in = norm_arr.flatten().reshape(1, -1)
                raw_model_probs = model.predict_proba(flat_in)[0]

                if hasattr(model, 'classes_'):
                    classes = list(model.classes_)
                    if len(classes) == len(labels) and classes == labels:
                        raw_probs = raw_model_probs
                    elif set(classes).issubset(set(labels)):
                        label_to_prob = {cls: prob for cls, prob in zip(classes, raw_model_probs)}
                        raw_probs = np.array([label_to_prob.get(lbl, 0.0) for lbl in labels], dtype=np.float32)
                    else:
                        logger.error("Sklearn model.classes_ does not match configured labels")
                        return self._make_rejected_result("label_alignment_error", motion_velocity=motion_velocity)
                else:
                    raw_probs = raw_model_probs

                if len(raw_probs) != len(labels):
                    logger.error("Probability length %d does not match label count %d", len(raw_probs), len(labels))
                    return self._make_rejected_result("label_length_mismatch", motion_velocity=motion_velocity)

            elif mode in ('cnn_lstm', 'stgcn_bilstm'):
                import torch
                if input_dim == 252:
                    kinetic_seq = transform_to_kinetic_invariants(norm_arr)
                    input_data = kinetic_seq.reshape(1, SEQUENCE_LENGTH, 252)
                else:
                    input_data = norm_arr.reshape(1, SEQUENCE_LENGTH, NUM_FEATURES)

                tensor_in = torch.tensor(input_data, dtype=torch.float32)
                with self._lock:
                    with torch.no_grad():
                        logits = model(tensor_in)
                        raw_probs = torch.softmax(logits, dim=1).numpy()[0]
            else:
                logger.error("Unsupported model mode: %s", mode)
                return self._make_rejected_result("unsupported_model_mode", motion_velocity=motion_velocity)

            num_classes = len(labels)
            max_entropy = (
                float(ENTROPY_MAX_RATIO * np.log(num_classes))
                if num_classes > 1 else MAX_WORD_ENTROPY
            )

            # Compute confidence, margin, and entropy on raw model probabilities
            raw_top_idx = int(np.argmax(raw_probs))
            raw_confidence = float(raw_probs[raw_top_idx])
            raw_word = labels[raw_top_idx] if raw_top_idx < len(labels) else '?'

            entropy = -float(np.sum(raw_probs * np.log(raw_probs + 1e-9)))
            sorted_raw = np.sort(raw_probs)[::-1]
            margin = float(sorted_raw[0] - sorted_raw[1]) if len(sorted_raw) > 1 else float(sorted_raw[0])

            effective_min_velocity = (
                self.min_motion_velocity if min_motion_velocity is None else min_motion_velocity
            )

            # Skip motion gate for confident static-pose words (e.g. NAMASTE, ME, INDIA)
            is_static_pose = (
                raw_word in STATIC_POSE_WORDS and
                raw_confidence >= CONFIDENCE_THRESHOLD_WORD and
                margin >= MARGIN_THRESHOLD_WORD
            )

            if motion_velocity < effective_min_velocity and not is_static_pose:
                return self._make_rejected_result(
                    "static_hand_detected",
                    motion_velocity=motion_velocity,
                    confidence=raw_confidence,
                    margin=margin,
                    entropy=entropy,
                    all_scores={
                        labels[i]: round(float(raw_probs[i]), 4)
                        for i in np.argsort(raw_probs)[::-1][:5]
                        if i < len(labels)
                    }
                )

            is_rejected = (
                raw_word in ('NONE', 'IDLE', '?') or
                raw_confidence < CONFIDENCE_THRESHOLD_WORD or
                margin < MARGIN_THRESHOLD_WORD or
                entropy > max_entropy
            )

            rejection_reason = None
            if is_rejected:
                if raw_word in ('NONE', 'IDLE'):
                    rejection_reason = 'idle_detected'
                elif raw_confidence < CONFIDENCE_THRESHOLD_WORD:
                    rejection_reason = 'low_confidence'
                elif margin < MARGIN_THRESHOLD_WORD:
                    rejection_reason = 'low_margin'
                elif entropy > max_entropy:
                    rejection_reason = 'high_entropy'
                else:
                    rejection_reason = 'unknown_gesture'

            # Apply body anchor priors ONLY as a tie-breaker for choosing the label
            final_word = raw_word
            final_probs = raw_probs

            if body_anchors and isinstance(body_anchors, dict) and len(raw_probs) == len(labels):
                def _safe_float(val):
                    if val is None:
                        return None
                    try:
                        f = float(val)
                        return f if np.isfinite(f) else None
                    except (ValueError, TypeError):
                        return None

                l_mouth = _safe_float(body_anchors.get('l_hand_to_mouth'))
                r_mouth = _safe_float(body_anchors.get('r_hand_to_mouth'))
                valid_mouth = [v for v in (l_mouth, r_mouth) if v is not None]
                min_mouth = min(valid_mouth) if valid_mouth else None

                l_chest = _safe_float(body_anchors.get('l_hand_to_chest'))
                r_chest = _safe_float(body_anchors.get('r_hand_to_chest'))
                valid_chest = [v for v in (l_chest, r_chest) if v is not None]
                min_chest = min(valid_chest) if valid_chest else None

                l_elev = _safe_float(body_anchors.get('l_elevation'))
                r_elev = _safe_float(body_anchors.get('r_elevation'))
                valid_elev = [v for v in (l_elev, r_elev) if v is not None]
                max_elev = max(valid_elev) if valid_elev else None
                min_elev = min(valid_elev) if valid_elev else None

                priors = np.ones(len(labels), dtype=np.float32)

                for idx, lbl in enumerate(labels):
                    # Facial / Mouth signs: THANK_YOU, WATER, FOOD, DOCTOR
                    if lbl in ('THANK_YOU', 'WATER', 'FOOD', 'DOCTOR'):
                        if min_mouth is not None:
                            if min_mouth < 0.40:
                                priors[idx] *= 1.45
                            elif min_mouth > 0.85:
                                priors[idx] *= 0.50

                    # Chest signs: NAMASTE, PLEASE, SORRY, ME, HEARING
                    elif lbl in ('NAMASTE', 'PLEASE', 'SORRY', 'ME', 'HEARING'):
                        if min_chest is not None:
                            if min_chest < 0.35 and (max_elev is None or max_elev > -0.15):
                                priors[idx] *= 1.15
                            elif min_chest > 0.80:
                                priors[idx] *= 0.55

                    # High signs (above shoulder level): HELLO, BYE_BYE, INDIA
                    elif lbl in ('HELLO', 'BYE_BYE', 'INDIA'):
                        if max_elev is not None:
                            if max_elev > 0.05:
                                priors[idx] *= 1.35
                            elif max_elev < -0.25:
                                priors[idx] *= 0.60

                    # Lower body / waist signs: WASHROOM
                    elif lbl == 'WASHROOM':
                        if min_elev is not None and min_elev < -0.30:
                            priors[idx] *= 1.40
                        elif max_elev is not None and max_elev > 0.10:
                            priors[idx] *= 0.40

                weighted = raw_probs * priors
                sum_p = np.sum(weighted)
                if sum_p > 1e-6:
                    final_probs = weighted / sum_p
                    top_reweighted_idx = int(np.argmax(final_probs))
                    if top_reweighted_idx < len(labels):
                        final_word = labels[top_reweighted_idx]

            sorted_indices = np.argsort(final_probs)[::-1][:5]
            top_scores = {
                labels[i]: round(float(final_probs[i]), 4)
                for i in sorted_indices if i < len(labels)
            }

            return {
                'word': '?' if is_rejected else final_word,
                'confidence': round(raw_confidence, 4),
                'margin': round(margin, 4),
                'entropy': round(entropy, 4),
                'mode': mode,
                'rejected': bool(is_rejected),
                'rejection_reason': rejection_reason,
                'motion_velocity': round(motion_velocity, 5),
                'all_scores': top_scores,
            }

        except Exception as e:
            logger.exception("Word recognition prediction failed: %s", e)
            return None

    def _pad_or_trim(self, frames):
        """
        Extract active gesture stroke (onset -> apex -> offset) and resample to SEQUENCE_LENGTH frames.
        Filters out leading/trailing idle frames so recognition is robust to signer timing and pause.
        """
        arr = np.asarray(frames, dtype=np.float32)
        if not np.all(np.isfinite(arr)):
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        n = len(arr)
        if n == 0:
            return np.zeros((SEQUENCE_LENGTH, NUM_FEATURES), dtype=np.float32)

        if n >= 10:
            h1_present = np.any(arr[:, :63] != 0, axis=-1)
            h2_present = np.any(arr[:, 63:] != 0, axis=-1)
            h1_both = h1_present[:-1] & h1_present[1:]
            h2_both = h2_present[:-1] & h2_present[1:]

            d1 = np.linalg.norm(arr[1:, :63] - arr[:-1, :63], axis=-1)
            d2 = np.linalg.norm(arr[1:, 63:] - arr[:-1, 63:], axis=-1)

            # Ignore diffs where either frame has an absent hand
            d1 = np.where(h1_both, d1, 0.0)
            d2 = np.where(h2_both, d2, 0.0)
            diffs = d1 + d2

            peak_energy = float(np.percentile(diffs, 95)) if len(diffs) > 0 else 0.0
            if peak_energy > 0.02:
                active = np.where(diffs > 0.15 * peak_energy)[0]
                if len(active) > 0:
                    start_idx = max(0, int(active[0]) - 2)
                    end_idx = min(n - 1, int(active[-1]) + 3)
                    if (end_idx - start_idx + 1) >= 8:
                        arr = arr[start_idx:end_idx + 1]

        if len(arr) < SEQUENCE_LENGTH:
            last_frame = arr[-1:]
            pad = np.repeat(last_frame, SEQUENCE_LENGTH - len(arr), axis=0)
            return np.concatenate([arr, pad], axis=0)
        elif len(arr) > SEQUENCE_LENGTH:
            indices = np.linspace(0, len(arr) - 1, SEQUENCE_LENGTH, dtype=int)
            return arr[indices]
        return arr

    def get_info(self):
        """Return word recognizer metadata."""
        with self._lock:
            return {
                'available': self.is_available,
                'model_mode': self.mode,
                'labels': list(self.labels),
                'num_classes': len(self.labels),
                'sequence_length': SEQUENCE_LENGTH,
                'input_features': getattr(self, 'input_dim', NUM_FEATURES),
                'input_dim': getattr(self, 'input_dim', NUM_FEATURES),
                'validation_accuracy': self.metadata.get('val_accuracy'),
            }
