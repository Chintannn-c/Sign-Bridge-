"""
SignBridge Ultra-Fast "Lock, then Hold Steady" Gesture State Machine.

Single source of truth for the recognition lifecycle:
NEUTRAL -> ARMED -> CAPTURING -> COOLDOWN (locked, hold steady) -> ARMED / NEUTRAL.

Speed comes from removing waiting, not from loosening accuracy checks:
- Instant commit: 1 single frame >= INSTANT_COMMIT_CONFIDENCE (0.92) commits immediately.
- Quick commit: STABILITY_FRAMES_K (3) consecutive identical valid frames (~100ms) commits.
- Jitter tolerance: MAX_MISS_FRAMES (1) dropped frame tolerated mid-streak.
- Dynamic re-arm: MIN_LOCK_MS (250ms), then hand steady for STEADY_MS (350ms) re-arms.
- REQUIRE_NEUTRAL_BETWEEN_SIGNS (False default) allows seamless continuous signing.
"""

from typing import Dict, Any, Optional, Tuple, Callable, List
import math

from services.gesture_config import (
    NEUTRAL,
    ARMED,
    CAPTURING,
    COOLDOWN,
    STATUS_LABELS,
    NEUTRAL_HOLD_FRAMES,
    STABILITY_FRAMES_K,
    INSTANT_COMMIT_CONFIDENCE,
    MAX_MISS_FRAMES,
    MIN_LOCK_MS,
    STEADY_MS,
    STEADY_MOTION_THRESHOLD,
    MAX_COOLDOWN_MS,
    REQUIRE_NEUTRAL_BETWEEN_SIGNS,
    HAND_LOST_FRAMES,
    RELEASE_FRAMES,
    CONFIDENCE_THRESHOLD,
    MARGIN_THRESHOLD,
    DEFAULT_MAX_CENTROID_DISTANCE,
)


def _compute_motion(curr_landmarks: Any, prev_landmarks: Any) -> float:
    """Calculates mean Euclidean movement per landmark between consecutive frames."""
    if not curr_landmarks or not prev_landmarks:
        return 0.0

    sum_d = 0.0
    count = 0

    # Handle (N, 2) or (N, 3) arrays, lists of dicts, or tuples
    for i in range(min(len(curr_landmarks), len(prev_landmarks))):
        p1 = curr_landmarks[i]
        p2 = prev_landmarks[i]

        if hasattr(p1, 'x') and hasattr(p2, 'x'):
            x1, y1 = p1.x, p1.y
            x2, y2 = p2.x, p2.y
        elif isinstance(p1, (list, tuple)) and isinstance(p2, (list, tuple)):
            x1, y1 = p1[0], p1[1]
            x2, y2 = p2[0], p2[1]
        elif isinstance(p1, dict) and isinstance(p2, dict):
            x1, y1 = p1.get('x', 0.0), p1.get('y', 0.0)
            x2, y2 = p2.get('x', 0.0), p2.get('y', 0.0)
        else:
            continue

        if (x1 != 0 or y1 != 0) and (x2 != 0 or y2 != 0):
            sum_d += math.hypot(x1 - x2, y1 - y2)
            count += 1

    return (sum_d / count) if count > 0 else 0.0


class GestureStateMachine:
    """Fast, robust gesture recognition state machine with dynamic hold-steady re-arming."""

    def __init__(
        self,
        on_commit: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        require_neutral_between_signs: bool = REQUIRE_NEUTRAL_BETWEEN_SIGNS,
    ):
        self.on_commit = on_commit
        self.require_neutral_between_signs = require_neutral_between_signs

        # Lifecycle State
        self.state: str = ARMED if not self.require_neutral_between_signs else NEUTRAL
        self.neutral_streak: int = 0
        self.hand_absent_streak: int = 0
        self.release_frames_counter: int = RELEASE_FRAMES

        # Capturing Streak
        self.current_candidate: Optional[str] = None
        self.candidate_streak: int = 0
        self.miss_frames: int = 0
        self.candidate_first_seen_ms: int = 0

        # Locked / Steady State
        self.locked_sign: Optional[str] = None
        self.last_committed_sign: Optional[str] = None
        self.locked_at_ms: int = 0
        self.steady_duration_ms: float = 0.0
        self.steady_progress: float = 0.0

        # Motion tracking
        self.prev_landmarks: Any = None
        self.last_update_ms: int = 0

    def reset(self):
        """Resets machine to initial state."""
        self.state = ARMED if not self.require_neutral_between_signs else NEUTRAL
        self.neutral_streak = 0
        self.hand_absent_streak = 0
        self.release_frames_counter = RELEASE_FRAMES
        self.current_candidate = None
        self.candidate_streak = 0
        self.miss_frames = 0
        self.candidate_first_seen_ms = 0
        self.locked_sign = None
        self.last_committed_sign = None
        self.locked_at_ms = 0
        self.steady_duration_ms = 0.0
        self.steady_progress = 0.0
        self.prev_landmarks = None
        self.last_update_ms = 0

    def _is_valid(self, prediction: Optional[Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
        """
        Validates prediction against confidence, margin, and centroid distance limits.
        Keeps project accuracy limits intact.
        """
        if not prediction:
            return False, 'no_prediction'

        label = str(prediction.get('label') or prediction.get('letter') or prediction.get('word') or '?')
        if label == '?' or not label.strip():
            return False, 'unknown_label'

        conf = float(prediction.get('confidence', 0.0))
        if conf < CONFIDENCE_THRESHOLD:
            return False, 'low_confidence'

        margin = float(prediction.get('margin', 1.0))
        if margin < MARGIN_THRESHOLD:
            return False, 'low_margin'

        centroid_dist = float(prediction.get('centroid_distance', 0.0))
        max_dist = float(prediction.get('max_centroid_distance', DEFAULT_MAX_CENTROID_DISTANCE))
        if centroid_dist > max_dist:
            return False, 'out_of_dataset_distance'

        if prediction.get('rejected', False):
            return False, prediction.get('rejection_reason', 'rejected')

        return True, None

    def update(
        self,
        now_ms: int,
        landmarks: Any,
        prediction: Optional[Dict[str, Any]] = None,
        is_neutral: bool = False,
    ) -> Dict[str, Any]:
        """
        Single per-frame update driving the state machine.

        Args:
            now_ms: current timestamp in milliseconds
            landmarks: 21 hand landmarks or None
            prediction: classifier output dict
            is_neutral: True if two open palms detected
        """
        emitted_sign: Optional[str] = None
        has_hand = landmarks is not None and len(landmarks) > 0

        # 1. Update hand presence and motion
        if not has_hand:
            self.hand_absent_streak += 1
            motion = 0.0
        else:
            self.hand_absent_streak = 0
            motion = _compute_motion(landmarks, self.prev_landmarks)
            self.prev_landmarks = landmarks

        dt = max(1, now_ms - self.last_update_ms) if self.last_update_ms > 0 else 33
        self.last_update_ms = now_ms

        # 2. State Machine Transitions

        # --- A. COOLDOWN (Locked, Hold Steady) -------------------------------
        if self.state == COOLDOWN:
            elapsed = now_ms - self.locked_at_ms

            # Failsafe: never stay locked longer than MAX_COOLDOWN_MS
            if elapsed >= MAX_COOLDOWN_MS:
                self.state = ARMED if not self.require_neutral_between_signs else NEUTRAL
                self.locked_sign = None
                self.steady_duration_ms = 0.0
                self.steady_progress = 0.0
            # Hand left the frame: immediate re-arm and release counted
            elif self.hand_absent_streak >= HAND_LOST_FRAMES:
                self.state = ARMED if not self.require_neutral_between_signs else NEUTRAL
                self.locked_sign = None
                self.steady_duration_ms = 0.0
                self.steady_progress = 0.0
                self.release_frames_counter = RELEASE_FRAMES
            else:
                # Minimum lock time guaranteed
                if elapsed >= MIN_LOCK_MS:
                    if motion <= STEADY_MOTION_THRESHOLD:
                        self.steady_duration_ms += dt
                    else:
                        # Hand moved significantly -> advance release counter
                        self.release_frames_counter += 1
                        self.steady_duration_ms = max(0.0, self.steady_duration_ms - 20)

                    self.steady_progress = min(1.0, self.steady_duration_ms / STEADY_MS)

                    if self.steady_progress >= 1.0:
                        # Hand held steady -> Re-arm for next detection!
                        self.state = ARMED if not self.require_neutral_between_signs else NEUTRAL
                        self.locked_sign = None
                        self.steady_duration_ms = 0.0
                        self.steady_progress = 0.0
                else:
                    self.steady_progress = 0.0

        # --- B. NEUTRAL (Holding two open palms to arm) ----------------------
        elif self.state == NEUTRAL:
            self.current_candidate = None
            self.candidate_streak = 0
            self.miss_frames = 0
            self.locked_sign = None
            self.steady_progress = 0.0

            if is_neutral:
                self.neutral_streak += 1
                if self.neutral_streak >= NEUTRAL_HOLD_FRAMES:
                    self.state = ARMED
                    self.neutral_streak = 0
            else:
                self.neutral_streak = 0

        # --- C. ARMED (Ready to detect next sign) ----------------------------
        elif self.state == ARMED:
            self.current_candidate = None
            self.candidate_streak = 0
            self.miss_frames = 0
            self.locked_sign = None
            self.steady_progress = 0.0

            if has_hand and not is_neutral:
                # Hand in frame and forming sign -> start CAPTURING
                self.state = CAPTURING

        # --- D. CAPTURING (Active recognition & fast-accept) -----------------
        if self.state == CAPTURING:
            if self.require_neutral_between_signs and is_neutral:
                self.state = NEUTRAL
                self.neutral_streak = 1
                self.current_candidate = None
                self.candidate_streak = 0
            elif self.hand_absent_streak >= HAND_LOST_FRAMES:
                self.state = ARMED if not self.require_neutral_between_signs else NEUTRAL
                self.current_candidate = None
                self.candidate_streak = 0
            else:
                is_valid, _ = self._is_valid(prediction)

                if is_valid and prediction:
                    label = str(prediction.get('label') or prediction.get('letter') or prediction.get('word') or '')
                    conf = float(prediction.get('confidence', 0.0))

                    # Prevent accidental immediate repeat of same sign without release
                    is_same_as_last = (label == self.last_committed_sign)
                    can_repeat = (not is_same_as_last) or (self.release_frames_counter >= RELEASE_FRAMES)

                    if can_repeat:
                        should_commit = False

                        # Rule 1: Instant accept on single high-confidence frame
                        if conf >= INSTANT_COMMIT_CONFIDENCE:
                            should_commit = True
                        else:
                            # Rule 2: Quick accept on STABILITY_FRAMES_K consecutive frames
                            if label == self.current_candidate:
                                self.candidate_streak += 1
                                self.miss_frames = 0
                            else:
                                self.current_candidate = label
                                self.candidate_streak = 1
                                self.miss_frames = 0
                                self.candidate_first_seen_ms = now_ms

                            if self.candidate_streak >= STABILITY_FRAMES_K:
                                should_commit = True

                        if should_commit:
                            # LOCK SIGN!
                            emitted_sign = label
                            self.locked_sign = label
                            self.last_committed_sign = label
                            self.release_frames_counter = 0
                            self.locked_at_ms = now_ms
                            self.steady_duration_ms = 0.0
                            self.steady_progress = 0.0
                            self.state = COOLDOWN
                            self.current_candidate = None
                            self.candidate_streak = 0
                            self.miss_frames = 0

                            if self.on_commit:
                                self.on_commit(label, prediction)
                    else:
                        # Holding same sign before release: ignore
                        pass
                else:
                    # Dropped / invalid frame: tolerate up to MAX_MISS_FRAMES
                    self.release_frames_counter += 1
                    self.miss_frames += 1
                    if self.miss_frames > MAX_MISS_FRAMES:
                        self.candidate_streak = 0
                        self.current_candidate = None

        return {
            'state': self.state,
            'status_label': STATUS_LABELS.get(self.state, "Ready"),
            'locked_sign': self.locked_sign,
            'steady_progress': self.steady_progress,
            'emitted': emitted_sign,
            'candidate': self.current_candidate,
            'candidate_streak': self.candidate_streak,
        }
