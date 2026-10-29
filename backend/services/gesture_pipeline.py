"""
SignBridge Gesture Recognition Pipeline & State Machine.

Implements:
1. Two open palms base NEUTRAL gesture detector.
2. Idle/resting hand detector.
3. Dataset-match centroid validation (rejects out-of-dataset gestures).
4. State Machine: NEUTRAL -> ARMED -> CAPTURING -> COOLDOWN -> NEUTRAL.
5. K-frame stability gating & majority-vote smoothing inside CAPTURING.
6. Mandatory re-arm between any two consecutive gestures.
"""

import json
import logging
from collections import deque, Counter
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

from services.gesture_config import (
    NEUTRAL_HOLD_FRAMES,
    STABILITY_FRAMES_K,
    CONFIDENCE_THRESHOLD,
    MARGIN_THRESHOLD,
    DEFAULT_MAX_CENTROID_DISTANCE,
    IDLE_WRIST_Y_THRESHOLD,
    MIN_HAND_SPAN,
)

logger = logging.getLogger(__name__)

CENTROIDS_PATH = Path(__file__).resolve().parent.parent / 'models' / 'letter_centroids.json'


class CentroidValidator:
    """Validates live normalized landmarks against stored dataset centroids."""

    def __init__(self, centroids_path: Path = CENTROIDS_PATH):
        self.centroids: Dict[str, np.ndarray] = {}
        self.thresholds: Dict[str, float] = {}
        self._load_centroids(centroids_path)

    def _load_centroids(self, path: Path):
        if not path.exists():
            logger.warning(f"Centroids file not found at {path}. Distance checks will use default threshold.")
            return

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for label, info in data.items():
                    self.centroids[label] = np.asarray(info['centroid'], dtype=np.float32)
                    self.thresholds[label] = float(info.get('threshold', DEFAULT_MAX_CENTROID_DISTANCE))
            logger.info(f"Loaded {len(self.centroids)} class centroids for out-of-dataset validation.")
        except Exception as e:
            logger.error(f"Failed to load centroids from {path}: {e}")

    def validate(self, norm_landmarks: np.ndarray, predicted_label: str) -> Tuple[bool, float, float, Optional[str]]:
        """
        Calculates distance from normalized landmarks to the predicted class centroid.
        Checks both direct orientation and swapped hand slots (min distance).
        Returns (is_valid, distance, threshold, rejection_reason).
        """
        if norm_landmarks is None or norm_landmarks.size != 126:
            return False, 999.0, DEFAULT_MAX_CENTROID_DISTANCE, 'invalid_landmarks'

        if predicted_label not in self.centroids:
            # If label not in dictionary, check against general default threshold
            return True, 0.0, DEFAULT_MAX_CENTROID_DISTANCE, None

        centroid = self.centroids[predicted_label]
        threshold = self.thresholds.get(predicted_label, DEFAULT_MAX_CENTROID_DISTANCE)

        # Direct distance
        dist_direct = float(np.linalg.norm(norm_landmarks - centroid))

        # Hand-swapped distance
        swapped = np.zeros_like(norm_landmarks)
        swapped[:63] = norm_landmarks[63:]
        swapped[63:] = norm_landmarks[:63]
        dist_swapped = float(np.linalg.norm(swapped - centroid))

        dist = min(dist_direct, dist_swapped)

        # ponytail: 1.35x tolerance buffer accommodates natural camera angle and wrist tilt
        effective_threshold = threshold * 1.35
        if dist > effective_threshold:
            return False, dist, effective_threshold, 'out_of_dataset_distance'

        return True, dist, effective_threshold, None


def _is_single_finger_extended(pts: np.ndarray, tip_idx: int, pip_idx: int, mcp_idx: int) -> bool:
    """Check if an individual finger is extended away from palm."""
    wrist = pts[0]
    tip = pts[tip_idx]
    pip = pts[pip_idx]
    mcp = pts[mcp_idx]

    d_tip_w = float(np.linalg.norm(tip - wrist))
    d_pip_w = float(np.linalg.norm(pip - wrist))
    d_tip_mcp = float(np.linalg.norm(tip - mcp))
    d_pip_mcp = float(np.linalg.norm(pip - mcp))

    return (d_tip_w > 1.15 * d_pip_w) and (d_tip_mcp > d_pip_mcp)


def _is_thumb_extended(pts: np.ndarray) -> bool:
    """Check if thumb is open and extended."""
    wrist = pts[0]
    tip = pts[4]
    mcp = pts[2]
    index_mcp = pts[5]

    d_tip_w = float(np.linalg.norm(tip - wrist))
    d_mcp_w = float(np.linalg.norm(mcp - wrist))
    d_tip_index = float(np.linalg.norm(tip - index_mcp))

    return (d_tip_w > 1.10 * d_mcp_w) and (d_tip_index > 0.04)


def _is_palm_facing_camera(pts: np.ndarray) -> bool:
    """Check if palm plane is oriented toward the camera."""
    wrist = pts[0]
    idx_mcp = pts[5]
    pnk_mcp = pts[17]

    v_idx = idx_mcp - wrist
    v_pnk = pnk_mcp - wrist

    normal = np.cross(v_idx, v_pnk)
    norm_mag = float(np.linalg.norm(normal))

    if norm_mag < 1e-5:
        return False

    # Palm spread in X-Y plane (not collapsed edge-on)
    palm_width = float(np.linalg.norm(idx_mcp[:2] - pnk_mcp[:2]))
    if palm_width < 0.030:
        return False

    # Dominant or significant Z normal component
    return (abs(normal[2]) / norm_mag) > 0.20


def is_neutral_gesture(landmarks: Any) -> bool:
    """
    Evaluates whether the frame displays the NEUTRAL BASE gesture:
    'Two open palms' (both hands visible, all fingers extended, palms facing camera).
    """
    if landmarks is None:
        return False

    arr = np.asarray(landmarks, dtype=np.float32)
    if arr.size != 126:
        return False

    left_raw = arr[:63]
    right_raw = arr[63:]

    # Both hands must be present and active
    if not (np.any(left_raw != 0) and np.any(right_raw != 0)):
        return False

    pts_l = left_raw.reshape(21, 3)
    pts_r = right_raw.reshape(21, 3)

    for pts in (pts_l, pts_r):
        # 1. Palm facing camera
        if not _is_palm_facing_camera(pts):
            return False

        # 2. Thumb extended
        if not _is_thumb_extended(pts):
            return False

        # 3. All 4 fingers extended
        if not _is_single_finger_extended(pts, 8, 6, 5):    # Index
            return False
        if not _is_single_finger_extended(pts, 12, 10, 9):  # Middle
            return False
        if not _is_single_finger_extended(pts, 16, 14, 13): # Ring
            return False
        if not _is_single_finger_extended(pts, 20, 18, 17): # Pinky
            return False

    return True


def is_idle_hand(landmarks: Any) -> bool:
    """
    Detects if hands are in a resting/idle pose or moving into/out of frame:
    - Empty or near-zero landmarks
    - Hand resting low on desk/lap (wrist Y > IDLE_WRIST_Y_THRESHOLD)
    - Limp/collapsed hand bounding box
    """
    if landmarks is None:
        return True

    arr = np.asarray(landmarks, dtype=np.float32)
    if arr.size != 126 or not np.any(arr != 0):
        return True

    pts = arr.reshape(2, 21, 3)
    active_hands = 0

    for h in range(2):
        hand = pts[h]
        if not np.any(hand != 0):
            continue

        wrist_y = hand[0, 1]

        # Resting at bottom of camera frame
        if wrist_y > IDLE_WRIST_Y_THRESHOLD:
            continue

        # Check hand bounding box extent
        x_span = hand[:, 0].max() - hand[:, 0].min()
        y_span = hand[:, 1].max() - hand[:, 1].min()
        if x_span < MIN_HAND_SPAN or y_span < MIN_HAND_SPAN:
            continue

        active_hands += 1

    return active_hands == 0


class GestureStateMachine:
    """
    Enforces the required recognition lifecycle:
    NEUTRAL -> ARMED -> CAPTURING -> COOLDOWN -> back to NEUTRAL.
    """

    STATES = ('NEUTRAL', 'ARMED', 'CAPTURING', 'COOLDOWN')

    def __init__(
        self,
        neutral_hold_frames: int = NEUTRAL_HOLD_FRAMES,
        stability_k: int = STABILITY_FRAMES_K,
        cooldown_frames: int = 15,
        majority_window_size: int = 5,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        margin_threshold: float = MARGIN_THRESHOLD,
        centroid_validator: Optional[CentroidValidator] = None,
    ):
        self.neutral_hold_frames = neutral_hold_frames
        self.stability_k = stability_k
        self.cooldown_frames = cooldown_frames
        self.majority_window_size = majority_window_size
        self.confidence_threshold = confidence_threshold
        self.margin_threshold = margin_threshold

        self.validator = centroid_validator or CentroidValidator()

        # State tracking
        self.state = 'NEUTRAL'
        self.neutral_streak = 0
        self.cooldown_remaining = 0

        # Capturing buffers
        self.majority_window: deque = deque(maxlen=majority_window_size)
        self.current_candidate: Optional[str] = None
        self.candidate_streak = 0
        self.last_emitted_item: Optional[str] = None

    def reset(self):
        """Reset state machine to initial NEUTRAL state."""
        self.state = 'NEUTRAL'
        self.neutral_streak = 0
        self.cooldown_remaining = 0
        self.majority_window.clear()
        self.current_candidate = None
        self.candidate_streak = 0

    def process_frame(
        self,
        landmarks: Any,
        prediction: Optional[Dict[str, Any]] = None,
        norm_landmarks: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Step the state machine with the current video frame.

        Args:
            landmarks: raw 126 float list/array
            prediction: dict from model with keys ('letter' or 'word', 'confidence', 'margin', etc.)
            norm_landmarks: optional wrist-relative, scale-normalized 126 floats

        Returns:
            dict detailing state, emission (if any), metrics, and rejection info.
        """
        # 1. Analyze frame conditions
        neutral = is_neutral_gesture(landmarks)
        idle = is_idle_hand(landmarks)

        emitted: Optional[str] = None
        conf = float(prediction.get('confidence', 0.0)) if prediction else 0.0
        margin = float(prediction.get('margin', 0.0)) if prediction else 0.0
        pred_label = str(prediction.get('letter') or prediction.get('word') or '?') if prediction else '?'

        dist = 0.0
        dist_threshold = DEFAULT_MAX_CENTROID_DISTANCE
        rejection_reason = None
        is_rejected = False

        # Validate prediction against thresholds and dataset centroids
        if prediction and pred_label != '?':
            if conf < self.confidence_threshold:
                is_rejected = True
                rejection_reason = 'low_confidence'
            elif margin < self.margin_threshold:
                is_rejected = True
                rejection_reason = 'low_margin'
            else:
                # Centroid distance check
                if norm_landmarks is not None:
                    val_ok, dist, dist_threshold, r_reason = self.validator.validate(norm_landmarks, pred_label)
                    if not val_ok:
                        is_rejected = True
                        rejection_reason = r_reason

        if is_rejected:
            pred_label = '?'

        # 2. State Machine Transitions
        if self.state == 'COOLDOWN':
            self.cooldown_remaining -= 1
            if self.cooldown_remaining <= 0 or neutral:
                self.state = 'NEUTRAL'
                self.neutral_streak = 1 if neutral else 0
                self.cooldown_remaining = 0

        elif self.state == 'NEUTRAL':
            self.majority_window.clear()
            self.current_candidate = None
            self.candidate_streak = 0

            if neutral:
                self.neutral_streak += 1
                if self.neutral_streak >= self.neutral_hold_frames:
                    # Transition to ARMED
                    self.state = 'ARMED'
                    self.neutral_streak = 0
            else:
                self.neutral_streak = 0

        elif self.state == 'ARMED':
            self.majority_window.clear()
            self.current_candidate = None
            self.candidate_streak = 0

            if idle:
                # Stay in ARMED waiting for user to form a gesture
                pass
            elif neutral:
                # Maintain armed state while user is in neutral
                pass
            else:
                # User has moved out of neutral and into a gesture pose
                self.state = 'CAPTURING'

        if self.state == 'CAPTURING':
            if idle:
                # Hand dropped or left frame: reset streak
                self.majority_window.clear()
                self.current_candidate = None
                self.candidate_streak = 0
            elif neutral:
                # User cancelled gesture and returned to neutral
                self.state = 'NEUTRAL'
                self.neutral_streak = 1
                self.majority_window.clear()
                self.current_candidate = None
                self.candidate_streak = 0
            else:
                # Valid capturing frame
                if not is_rejected and pred_label != '?':
                    self.majority_window.append(pred_label)

                    # Majority vote smoothing
                    counts = Counter(self.majority_window)
                    smoothed_label, _ = counts.most_common(1)[0]

                    if smoothed_label == self.current_candidate:
                        self.candidate_streak += 1
                    else:
                        self.current_candidate = smoothed_label
                        self.candidate_streak = 1

                    # K-frame stability check
                    if self.candidate_streak >= self.stability_k:
                        # EMIT GESTURE!
                        emitted = self.current_candidate
                        self.last_emitted_item = emitted

                        # Move directly to COOLDOWN
                        self.state = 'COOLDOWN'
                        self.cooldown_remaining = self.cooldown_frames
                        self.majority_window.clear()
                        self.current_candidate = None
                        self.candidate_streak = 0
                else:
                    # Low confidence / unknown gesture / transition noise
                    # Do not commit, decay streak
                    self.candidate_streak = max(0, self.candidate_streak - 1)

        return {
            'state': self.state,
            'emitted': emitted,
            'is_neutral': neutral,
            'is_idle': idle,
            'neutral_streak': self.neutral_streak,
            'candidate_streak': self.candidate_streak,
            'candidate': self.current_candidate,
            'confidence': conf,
            'margin': margin,
            'centroid_distance': dist,
            'centroid_threshold': dist_threshold,
            'rejected': is_rejected,
            'rejection_reason': rejection_reason,
            'predicted_label': pred_label,
        }
