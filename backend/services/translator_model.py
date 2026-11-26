"""
Sign-Bridge Flask API — ISL Gesture Translation Model Service

Loads trained classifiers that map MediaPipe hand landmark coordinate
sequences to ISL alphabet letters (A-Z).

Model loading priority:
  1. XGBoost (.pkl)         — highest accuracy, fastest CPU inference
  2. TensorFlow/Keras (.h5) — deep learning fallback
  3. Heuristic               — rule-based geometric matching

The model accepts a flat array of 126 landmark values:
  - Left hand:  21 landmarks x (x, y, z) -> flattened to indices 0-62
  - Right hand: 21 landmarks x (x, y, z) -> flattened to indices 63-125
  Total input: 126 features per frame

When no trained model file exists, falls back to a rule-based
heuristic classifier using the Mendeley ISL landmark descriptions.
"""
import os
import csv
import json
import logging
import pickle
import numpy as np

try:
    from .feature_extractor import extract_features, NUM_EXTRACTED_FEATURES, NUM_RAW_FEATURES
    from .gesture_config import CONFIDENCE_THRESHOLD, MARGIN_THRESHOLD
    from .gesture_pipeline import CentroidValidator
except ImportError:
    from services.feature_extractor import extract_features, NUM_EXTRACTED_FEATURES, NUM_RAW_FEATURES
    from services.gesture_config import CONFIDENCE_THRESHOLD, MARGIN_THRESHOLD
    from services.gesture_pipeline import CentroidValidator

logger = logging.getLogger(__name__)

# Path to the trained model weights files
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

# ST-GCN Kinematic Graph model (.pt) — Priority 1 (Highest geometric fidelity)
STGCN_MODEL_PATH = os.path.join(MODEL_DIR, 'isl_stgcn_model.pt')
STGCN_META_PATH = os.path.join(MODEL_DIR, 'stgcn_training_meta.json')

# XGBoost model (created by train_model_xgb.py) — Priority 2
XGB_MODEL_PATH = os.path.join(MODEL_DIR, 'isl_xgboost_model.pkl')
XGB_META_PATH = os.path.join(MODEL_DIR, 'xgb_training_meta.json')

# Calibrated Hybrid Ensemble metadata
HYBRID_META_PATH = os.path.join(MODEL_DIR, 'hybrid_training_meta.json')

# Keras model (created by train_model.py) — Priority 3
MODEL_PATH = os.path.join(MODEL_DIR, 'isl_gesture_model.h5')
META_PATH = os.path.join(MODEL_DIR, 'training_meta.json')

# Path to the Mendeley reference CSV
MENDELEY_CSV = os.path.join(
    os.path.dirname(__file__), '..', '..', 'dataset',
    'Mendeley_ISL', 'extracted', 'ISL_Mendeley_Alphabets.csv'
)

# ISL alphabet + digit labels (36 classes: A-Z then 0-9)
ISL_LABELS: list[str] = [str(c) for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789']

# Below this, a same-hand thumb-to-index distance counts as "touching"
# (in the same normalized units used by normalize_landmarks()).
THUMB_INDEX_TOUCH_THRESHOLD = 0.12

EXPECTED_INPUT_LEN = 126
POINTS_PER_HAND = 21


def normalize_landmarks(landmarks):
    points = np.asarray(landmarks, dtype=np.float32)
    if points.size != 126 or not np.isfinite(points).all():
        return None
    points = points.reshape(2, 21, 3)
    present = np.any(points != 0, axis=(1, 2))
    if not present.any():
        return None
    wrists = points[present, 0]
    anchor = wrists.mean(axis=0)
    hand_sizes = np.linalg.norm(points[present, 12] - points[present, 0], axis=1)
    scale = float(hand_sizes.mean())
    if not np.isfinite(scale) or scale < 1e-4:
        return None
    normalized = (points - anchor) / scale
    normalized[~present] = 0
    return normalized.reshape(-1)


def normalize_landmarks_batch(batch):
    """Vectorized normalize_landmarks for a (N, 126) float32 array.

    Mirrors normalize_landmarks() exactly but operates on all N frames
    in one numpy call — no Python loop, no per-frame overhead.

    Returns a (N, 126) float32 array.  Frames where both hands are absent
    are left as zeros (same contract as the scalar version).
    Returns None if no frame contains any hand data at all.
    """
    pts = np.asarray(batch, dtype=np.float32).reshape(-1, 2, 21, 3)   # (N, 2, 21, 3)
    N = pts.shape[0]

    # Which frames have at least one active hand
    present = np.any(pts != 0, axis=(2, 3))                            # (N, 2) bool

    if not present.any():
        return None

    # Anchor: mean wrist position across present hands per frame
    # Mask absent hands to 0 before summing so they don't skew the mean
    wrist_masked = pts[:, :, 0, :] * present[:, :, None]              # (N, 2, 3)
    n_present = present.sum(axis=1, keepdims=True).clip(min=1)         # (N, 1)
    anchor = wrist_masked.sum(axis=1) / n_present                      # (N, 3)

    # Scale: mean mid-finger-to-wrist distance across present hands
    hand_sizes = np.linalg.norm(
        pts[:, :, 12, :] - pts[:, :, 0, :], axis=-1                   # (N, 2)
    )
    scale = np.where(present, hand_sizes, 0).sum(axis=1) / n_present[:, 0]  # (N,)
    scale = scale.clip(min=1e-4)                                       # guard zero

    # Normalise: subtract anchor, divide by scale
    normed = (pts - anchor[:, None, None, :]) / scale[:, None, None, None]  # (N, 2, 21, 3)

    # Zero-out absent hand slots (keep same contract as scalar version)
    normed[~present] = 0

    return normed.reshape(N, 126).astype(np.float32)


class LandmarkValidationError(ValueError):
    """Raised when incoming landmark data doesn't match the expected shape."""


def validate_landmark_array(landmarks):
    """Coerce landmarks to a (126,) or (N, 126) float array, or raise."""
    if isinstance(landmarks, list) and len(landmarks) == 42 and len(landmarks) > 0 and isinstance(landmarks[0], dict):
        flat = []
        for pt in landmarks:
            flat.extend([float(pt.get('x', 0.0)), float(pt.get('y', 0.0)), float(pt.get('z', 0.0))])
        landmarks = flat

    try:
        arr = np.asarray(landmarks, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise LandmarkValidationError(f"landmarks must be numeric: {exc}") from exc

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] != EXPECTED_INPUT_LEN:
        raise LandmarkValidationError(
            f"expected {EXPECTED_INPUT_LEN} values per frame "
            f"(21 landmarks x 3 coords x 2 hands), got shape {arr.shape}"
        )
    if not np.isfinite(arr).all():
        raise LandmarkValidationError("landmarks contain NaN or infinite values")
    return arr


class TranslatorModel:
    """
    ISL Gesture-to-Text translation model.

    Supports four ML modes:
      1. Hybrid Ensemble mode — Combines ST-GCN graph logits + XGBoost trees (calibrated)
      2. ST-GCN mode          — loads PyTorch kinematic graph model (.pt)
      3. XGBoost mode         — loads trained XGBoost tree classifier (.pkl)
      4. Deep Learning mode   — loads a trained Keras .h5 model
      5. Heuristic mode       — rule-based matching using landmark geometry
    """

    HYBRID_XGB_WEIGHT = 0.65
    HYBRID_STGCN_WEIGHT = 0.35
    HYBRID_TEMPERATURE = 1.20

    def __init__(self):
        self.stgcn_model = None
        self.xgb_model = None
        self.model = None
        self.mode = 'heuristic'  # 'hybrid_ensemble' | 'stgcn' | 'xgboost' | 'deep_learning' | 'heuristic'
        self.mendeley_ref = {}
        self.metadata = {}
        self.centroid_validator = CentroidValidator()
        self._load()

    def _load(self):
        """Attempt to load trained models; fall back to heuristic.

        Loading priority:
          1. Hybrid Ensemble (ST-GCN + XGBoost) — Best calibration & geometric fidelity
          2. ST-GCN standalone (.pt)
          3. XGBoost standalone (.pkl)
          4. Keras (.h5)
          5. Heuristic — rule-based geometric matching
        """
        # Reset state on load/reload (ponytail: prevent stale state leakage)
        self.stgcn_model = None
        self.xgb_model = None
        self.model = None
        self.mode = 'heuristic'
        self.metadata = {}
        self.mendeley_ref = {}

        # Load ST-GCN if available
        if os.path.exists(STGCN_MODEL_PATH):
            try:
                import torch
                from models.st_gcn import STGCNHandClassifier

                if os.path.exists(STGCN_META_PATH):
                    with open(STGCN_META_PATH, 'r', encoding='utf-8') as f:
                        self.metadata = json.load(f)

                labels = self.metadata.get('labels', [c for c in ISL_LABELS if c.isalpha()])
                st_model = STGCNHandClassifier(num_classes=len(labels))
                checkpoint = torch.load(STGCN_MODEL_PATH, map_location='cpu', weights_only=True)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    st_model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    st_model.load_state_dict(checkpoint)
                st_model.eval()
                self.stgcn_model = st_model
                logger.info(f"Loaded ST-GCN Kinematic Graph model from {STGCN_MODEL_PATH}")
            except Exception as e:
                logger.warning(f"Failed to load ST-GCN model: {e}")

        # Load XGBoost if available
        if os.path.exists(XGB_MODEL_PATH):
            try:
                with open(XGB_MODEL_PATH, 'rb') as f:
                    self.xgb_model = pickle.load(f)
                if not self.metadata and os.path.exists(XGB_META_PATH):
                    with open(XGB_META_PATH, 'r', encoding='utf-8') as f:
                        self.metadata = json.load(f)
                logger.info(f"Loaded XGBoost model from {XGB_MODEL_PATH}")
            except Exception as e:
                logger.warning(f"Failed to load XGBoost model: {e}")

        # Determine best operating mode and bind truthful metadata
        if self.stgcn_model is not None and self.xgb_model is not None:
            self.mode = 'hybrid_ensemble'
            self.model = self.stgcn_model
            if os.path.exists(HYBRID_META_PATH):
                with open(HYBRID_META_PATH, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            elif os.path.exists(XGB_META_PATH):
                with open(XGB_META_PATH, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            logger.info("Sign-Bridge Alphabet Engine initialized in CALIBRATED HYBRID ENSEMBLE mode (ST-GCN + XGBoost).")
            return
        elif self.stgcn_model is not None:
            self.mode = 'stgcn'
            self.model = self.stgcn_model
            if os.path.exists(STGCN_META_PATH):
                with open(STGCN_META_PATH, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            logger.info("Sign-Bridge Alphabet Engine initialized in ST-GCN mode.")
            return
        elif self.xgb_model is not None:
            self.mode = 'xgboost'
            self.model = self.xgb_model
            if os.path.exists(XGB_META_PATH):
                with open(XGB_META_PATH, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            logger.info("Sign-Bridge Alphabet Engine initialized in XGBoost mode.")
            return

        logger.warning("No trained alphabet models found. Alphabet translation disabled.")
        self.mode = 'disabled'

    def predict(self, landmarks, body_anchors=None):
        """
        Predict the ISL letter from a set of hand landmarks with optional Holistic body anchors.
        """
        if self.mode == 'hybrid_ensemble':
            res = self._predict_hybrid_ensemble(landmarks)
        elif self.mode == 'stgcn' and self.model is not None:
            res = self._predict_stgcn(landmarks)
        elif self.mode == 'xgboost' and self.model is not None:
            res = self._predict_xgb(landmarks)
        else:
            res = self._invalid_input_result(self.mode or 'disabled')

        if body_anchors and isinstance(res, dict):
            res['holistic_anchored'] = True
        return res

    def _predict_hybrid_ensemble(self, landmarks):
        """Run inference through the calibrated ST-GCN + XGBoost Hybrid Ensemble."""
        if self.stgcn_model is None or self.xgb_model is None:
            return self._predict_stgcn(landmarks) if self.stgcn_model else self._predict_xgb(landmarks)

        try:
            import torch
            arr = validate_landmark_array(landmarks)
            if not np.any(arr != 0):
                return self._invalid_input_result('hybrid_ensemble')

            normalized = [normalize_landmarks(row) for row in arr]
            if any(row is None for row in normalized):
                return self._invalid_input_result('hybrid_ensemble')
            arr_norm = np.asarray(normalized, dtype=np.float32)

            labels = self.metadata.get('labels', [c for c in ISL_LABELS if c.isalpha()])

            # Check single-hand status
            left_active = np.any(arr_norm[:, :63] != 0)
            right_active = np.any(arr_norm[:, 63:] != 0)
            is_single_hand = (left_active and not right_active) or (right_active and not left_active)

            # 1. Primary orientation pass (both models evaluated on original input)
            feat_primary = extract_features(arr_norm)
            probs_xgb = self.xgb_model.predict_proba(feat_primary)[0]

            tensor_in = torch.tensor(arr_norm, dtype=torch.float32)
            with torch.no_grad():
                logits = self.stgcn_model(tensor_in)
                probs_stgcn = torch.softmax(logits / self.HYBRID_TEMPERATURE, dim=1).numpy()[0]

            probs_primary = (self.HYBRID_XGB_WEIGHT * probs_xgb) + (self.HYBRID_STGCN_WEIGHT * probs_stgcn)
            conf_primary = float(np.max(probs_primary))

            # 2. Only evaluate swapped pass if primary confidence is uncertain (< 0.55, or < 0.65 for single-hand)
            # This cuts backend latency in half for clear signs and prevents redundant inference
            if (is_single_hand and conf_primary < 0.65) or conf_primary < 0.55:
                swapped = np.zeros_like(arr_norm)
                swapped[:, :63] = arr_norm[:, 63:]
                swapped[:, 63:] = arr_norm[:, :63]
                feat_swapped = extract_features(swapped)
                probs_xgb_swapped = self.xgb_model.predict_proba(feat_swapped)[0]

                tensor_swapped = torch.tensor(swapped, dtype=torch.float32)
                with torch.no_grad():
                    logits_swapped = self.stgcn_model(tensor_swapped)
                    probs_stgcn_swapped = torch.softmax(logits_swapped / self.HYBRID_TEMPERATURE, dim=1).numpy()[0]

                probs_swapped = (self.HYBRID_XGB_WEIGHT * probs_xgb_swapped) + (self.HYBRID_STGCN_WEIGHT * probs_stgcn_swapped)
                conf_swapped = float(np.max(probs_swapped))

                # Choose ONE consistent orientation for the entire ensemble
                if conf_swapped > conf_primary:
                    probs = probs_swapped
                else:
                    probs = probs_primary
            else:
                probs = probs_primary

            sorted_indices = np.argsort(probs)[::-1]
            top_idx = int(sorted_indices[0])
            confidence = float(probs[top_idx])
            second_conf = float(probs[sorted_indices[1]]) if len(sorted_indices) > 1 else 0.0
            margin = float(confidence - second_conf)
            letter = labels[top_idx] if top_idx < len(labels) else '?'

            # Top 5 scores
            top_scores = {
                labels[i]: round(float(probs[i]), 4)
                for i in sorted_indices[:5] if i < len(labels)
            }

            # Out of dataset / centroid distance check
            norm_first = arr_norm[0] if arr_norm.ndim > 1 else arr_norm
            val_ok, dist, dist_thresh, r_reason = self.centroid_validator.validate(norm_first, letter)

            is_rejected = bool(confidence < CONFIDENCE_THRESHOLD or margin < MARGIN_THRESHOLD or not val_ok or letter == '?')
            rejection_reason = None
            if is_rejected:
                if confidence < CONFIDENCE_THRESHOLD:
                    rejection_reason = 'low_confidence'
                elif margin < MARGIN_THRESHOLD:
                    rejection_reason = 'low_margin'
                elif not val_ok:
                    rejection_reason = r_reason
                else:
                    rejection_reason = 'unknown_gesture'

            return {
                'letter': '?' if is_rejected else letter,
                'confidence': round(confidence, 4),
                'margin': round(margin, 4),
                'centroid_distance': round(dist, 4),
                'mode': 'hybrid_ensemble',
                'rejected': is_rejected,
                'rejection_reason': rejection_reason,
                'all_scores': top_scores
            }
        except LandmarkValidationError as e:
            logger.warning(f"Ensemble prediction skipped — bad input: {e}")
            return self._invalid_input_result('hybrid_ensemble')
        except Exception as e:
            logger.error(f"Ensemble prediction failed: {e}")
            return self._predict_xgb(landmarks)

    def _predict_stgcn(self, landmarks):
        """Run inference through the trained PyTorch ST-GCN model."""
        model = self.stgcn_model or self.model
        if model is None or not hasattr(model, '__call__'):
            return self._predict_xgb(landmarks)

        try:
            import torch
            arr = validate_landmark_array(landmarks)

            if not np.any(arr != 0):
                return self._invalid_input_result('stgcn')

            # Normalization
            normalized = [normalize_landmarks(row) for row in arr]
            if any(row is None for row in normalized):
                return self._invalid_input_result('stgcn')
            arr = np.asarray(normalized, dtype=np.float32)

            labels = self.metadata.get('labels', [c for c in ISL_LABELS if c.isalpha()])

            # 1. Primary orientation pass
            tensor_in = torch.tensor(arr, dtype=torch.float32)
            with torch.no_grad():
                logits = model(tensor_in)
                probs = torch.softmax(logits, dim=1).numpy()[0]

            left_active = np.any(arr[:, :63] != 0)
            right_active = np.any(arr[:, 63:] != 0)
            is_single_hand = (left_active and not right_active) or (right_active and not left_active)

            # 2. Hand-swapped orientation pass only when single-hand or confidence is low (< 0.72)
            if is_single_hand or float(np.max(probs)) < 0.72:
                swapped = np.zeros_like(arr)
                swapped[:, :63] = arr[:, 63:]
                swapped[:, 63:] = arr[:, :63]
                tensor_swapped = torch.tensor(swapped, dtype=torch.float32)
                with torch.no_grad():
                    logits_swapped = model(tensor_swapped)
                    probs_swapped = torch.softmax(logits_swapped, dim=1).numpy()[0]

                if np.max(probs_swapped) > np.max(probs):
                    probs = probs_swapped

            sorted_indices = np.argsort(probs)[::-1]
            top_idx = int(sorted_indices[0])
            confidence = float(probs[top_idx])
            second_conf = float(probs[sorted_indices[1]]) if len(sorted_indices) > 1 else 0.0
            margin = float(confidence - second_conf)
            letter = labels[top_idx] if top_idx < len(labels) else '?'

            # Top 5 scores
            top_scores = {
                labels[i]: round(float(probs[i]), 4)
                for i in sorted_indices[:5] if i < len(labels)
            }

            # Out of dataset / centroid distance check
            norm_first = arr[0] if arr.ndim > 1 else arr
            val_ok, dist, dist_thresh, r_reason = self.centroid_validator.validate(norm_first, letter)

            is_rejected = bool(confidence < CONFIDENCE_THRESHOLD or margin < MARGIN_THRESHOLD or not val_ok or letter == '?')
            rejection_reason = None
            if is_rejected:
                if confidence < CONFIDENCE_THRESHOLD:
                    rejection_reason = 'low_confidence'
                elif margin < MARGIN_THRESHOLD:
                    rejection_reason = 'low_margin'
                elif not val_ok:
                    rejection_reason = r_reason
                else:
                    rejection_reason = 'unknown_gesture'

            return {
                'letter': '?' if is_rejected else letter,
                'confidence': round(confidence, 4),
                'margin': round(margin, 4),
                'centroid_distance': round(dist, 4),
                'mode': 'stgcn',
                'rejected': is_rejected,
                'rejection_reason': rejection_reason,
                'all_scores': top_scores
            }
        except LandmarkValidationError as e:
            logger.warning(f"ST-GCN prediction skipped — bad input: {e}")
            return self._invalid_input_result('stgcn')
        except Exception as e:
            logger.error(f"ST-GCN prediction failed: {e}")
            return self._predict_xgb(landmarks)

    def _predict_xgb(self, landmarks):
        """Run inference through the trained XGBoost model."""
        model = self.xgb_model or self.model
        if model is None or not hasattr(model, 'predict_proba'):
            return self._invalid_input_result('stgcn')

        try:
            arr = validate_landmark_array(landmarks)

            # Fast check: if all zeros
            if not np.any(arr != 0):
                return self._invalid_input_result('xgboost')

            # Apply same normalization as training pipeline unconditionally
            normalized = [normalize_landmarks(row) for row in arr]
            if any(row is None for row in normalized):
                return self._invalid_input_result('xgboost')
            arr = np.asarray(normalized, dtype=np.float32)

            labels = self.metadata.get('labels', ISL_LABELS)

            # Check if this is a single-hand gesture (only one hand slot is non-zero)
            left_active = np.any(arr[:, :63] != 0)
            right_active = np.any(arr[:, 63:] != 0)
            is_single_hand = (left_active and not right_active) or (right_active and not left_active)

            # Predict probabilities for primary placement using 176-D geometric features
            feat_primary = extract_features(arr)
            probs = model.predict_proba(feat_primary)[0]

            conf_primary = float(np.max(probs))

            # Only evaluate alternate hand slot arrangement if primary placement is uncertain
            if (is_single_hand and conf_primary < 0.65) or conf_primary < 0.55:
                swapped = np.zeros_like(arr)
                swapped[:, :63] = arr[:, 63:]
                swapped[:, 63:] = arr[:, :63]
                feat_swapped = extract_features(swapped)
                probs_swapped = model.predict_proba(feat_swapped)[0]
                if np.max(probs_swapped) > np.max(probs):
                    probs = probs_swapped

            # For contact gestures where hands overlap, only evaluate dominant sub-hands if confidence remains low
            if left_active and right_active and np.max(probs) < 0.50:
                h_left_single = np.zeros_like(arr)
                h_left_single[:, :63] = arr[:, :63]
                f_l = extract_features(h_left_single)
                p_l = model.predict_proba(f_l)[0]
                if np.max(p_l) > np.max(probs):
                    probs = p_l

                h_right_single = np.zeros_like(arr)
                h_right_single[:, 63:] = arr[:, 63:]
                f_r = extract_features(h_right_single)
                p_r = model.predict_proba(f_r)[0]
                if np.max(p_r) > np.max(probs):
                    probs = p_r

            sorted_indices = np.argsort(probs)[::-1]
            top_idx = int(sorted_indices[0])
            confidence = float(probs[top_idx])
            second_conf = float(probs[sorted_indices[1]]) if len(sorted_indices) > 1 else 0.0
            margin = float(confidence - second_conf)
            letter = labels[top_idx] if top_idx < len(labels) else '?'

            # Top 5 scores
            top_scores = {
                labels[i]: round(float(probs[i]), 4)
                for i in sorted_indices[:5] if i < len(labels)
            }

            # Out of dataset / centroid distance check
            norm_first = arr[0] if arr.ndim > 1 else arr
            val_ok, dist, dist_thresh, r_reason = self.centroid_validator.validate(norm_first, letter)

            is_rejected = bool(confidence < CONFIDENCE_THRESHOLD or margin < MARGIN_THRESHOLD or not val_ok or letter == '?')
            rejection_reason = None
            if is_rejected:
                if confidence < CONFIDENCE_THRESHOLD:
                    rejection_reason = 'low_confidence'
                elif margin < MARGIN_THRESHOLD:
                    rejection_reason = 'low_margin'
                elif not val_ok:
                    rejection_reason = r_reason
                else:
                    rejection_reason = 'unknown_gesture'

            return {
                'letter': '?' if is_rejected else letter,
                'confidence': round(confidence, 4),
                'margin': round(margin, 4),
                'centroid_distance': round(dist, 4),
                'mode': 'xgboost',
                'rejected': is_rejected,
                'rejection_reason': rejection_reason,
                'all_scores': top_scores
            }
        except LandmarkValidationError as e:
            logger.warning(f"XGBoost prediction skipped — bad input: {e}")
            return self._invalid_input_result('xgboost')
        except Exception as e:
            logger.error(f"XGBoost prediction failed: {e}")
            return self._invalid_input_result('stgcn')

    @staticmethod
    def _invalid_input_result(mode):
        return {
            'letter': '?',
            'confidence': 0.0,
            'margin': 0.0,
            'centroid_distance': 999.0,
            'mode': mode,
            'rejected': True,
            'rejection_reason': 'invalid_input',
            'all_scores': {}
        }

    def get_info(self):
        """Return truthful model metadata derived directly from active configuration."""
        if self.mode == 'hybrid_ensemble':
            model_path = f"{XGB_MODEL_PATH} + {STGCN_MODEL_PATH}"
        elif self.mode == 'xgboost':
            model_path = XGB_MODEL_PATH
        elif self.mode == 'stgcn':
            model_path = STGCN_MODEL_PATH
        else:
            model_path = MODEL_PATH

        labels = [c for c in ISL_LABELS if c.isalpha()] if self.mode in ('xgboost', 'hybrid_ensemble', 'stgcn') else ISL_LABELS
        metrics = self.metadata.get('metrics', {})

        return {
            'mode': self.mode,
            'model_path': model_path,
            'labels': labels,
            'num_classes': len(labels),
            'input_features': NUM_RAW_FEATURES,
            'estimator_features': NUM_EXTRACTED_FEATURES if self.mode in ('xgboost', 'hybrid_ensemble') else NUM_RAW_FEATURES,
            'feature_name': f'geometric_invariants_{NUM_EXTRACTED_FEATURES}d' if self.mode in ('xgboost', 'hybrid_ensemble') else 'wrist_center_scale_v1',
            'validation_accuracy': metrics.get('val_accuracy', self.metadata.get('val_accuracy')),
            'test_accuracy': metrics.get('test_accuracy'),
            'test_macro_f1': metrics.get('test_macro_f1'),
            'model_type': self.metadata.get('model_type', self.mode),
            'mendeley_letters': list(self.mendeley_ref.keys()),
        }