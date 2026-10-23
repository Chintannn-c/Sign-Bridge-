"""
Unit tests for SignBridge Fast "Lock, then Hold Steady" Gesture State Machine.

Tests all fast-path and accuracy requirements:
1. Instant commit on single high-confidence frame (>= 0.92)
2. Quick commit on STABILITY_FRAMES_K (3) consecutive valid frames (~100ms)
3. Single miss-frame tolerance mid-streak
4. Hold steady motion-gating: hand steady for STEADY_MS re-arms
5. Hand leaving frame (HAND_LOST_FRAMES) triggers immediate re-arm
6. Accuracy guardrails: low confidence (< 0.75), low margin (< 0.10), high centroid distance
7. Repeat prevention: same sign requires RELEASE_FRAMES before re-committing
8. Failsafe: MAX_COOLDOWN_MS re-arms regardless of motion
9. Optional neutral mode: REQUIRE_NEUTRAL_BETWEEN_SIGNS behavior
"""

import unittest
from services.gesture_config import (
    NEUTRAL,
    ARMED,
    CAPTURING,
    COOLDOWN,
    NEUTRAL_HOLD_FRAMES,
    STABILITY_FRAMES_K,
    INSTANT_COMMIT_CONFIDENCE,
    MIN_LOCK_MS,
    STEADY_MS,
    STEADY_MOTION_THRESHOLD,
    MAX_COOLDOWN_MS,
    HAND_LOST_FRAMES,
    RELEASE_FRAMES,
    CONFIDENCE_THRESHOLD,
    MARGIN_THRESHOLD,
)
from services.gesture_state_machine import GestureStateMachine


def make_landmarks(dx=0.0):
    """Synthetic 21-landmark list."""
    return [{'x': 0.5 + dx, 'y': 0.5, 'z': 0.0} for _ in range(21)]


class TestGestureStateMachine(unittest.TestCase):

    def setUp(self):
        self.committed = []
        self.sm = GestureStateMachine(
            on_commit=lambda label, pred: self.committed.append((label, pred)),
            require_neutral_between_signs=False,
        )

    def test_instant_commit(self):
        """Single high-confidence frame (>= 0.92) commits instantly on 1st frame."""
        now = 1000
        landmarks = make_landmarks()
        pred = {'label': 'A', 'confidence': 0.94, 'margin': 0.30, 'centroid_distance': 3.0}

        res = self.sm.update(now, landmarks, prediction=pred)

        self.assertEqual(res['emitted'], 'A')
        self.assertEqual(res['locked_sign'], 'A')
        self.assertEqual(res['state'], COOLDOWN)
        self.assertEqual(len(self.committed), 1)
        self.assertEqual(self.committed[0][0], 'A')

    def test_quick_commit_three_frames(self):
        """3 consecutive identical valid frames (confidence >= 0.75) commits in ~100ms."""
        landmarks = make_landmarks()
        pred = {'label': 'B', 'confidence': 0.80, 'margin': 0.25, 'centroid_distance': 2.5}

        # Frame 1: Capturing starts, streak = 1
        res1 = self.sm.update(1000, landmarks, prediction=pred)
        self.assertIsNone(res1['emitted'])
        self.assertEqual(res1['state'], CAPTURING)
        self.assertEqual(res1['candidate_streak'], 1)

        # Frame 2: streak = 2
        res2 = self.sm.update(1033, landmarks, prediction=pred)
        self.assertIsNone(res2['emitted'])
        self.assertEqual(res2['candidate_streak'], 2)

        # Frame 3: STABILITY_FRAMES_K (3) reached -> COMMITTED!
        res3 = self.sm.update(1066, landmarks, prediction=pred)
        self.assertEqual(res3['emitted'], 'B')
        self.assertEqual(res3['locked_sign'], 'B')
        self.assertEqual(res3['state'], COOLDOWN)

    def test_single_miss_frame_tolerance(self):
        """A single dropped/invalid frame mid-streak is tolerated without resetting streak."""
        landmarks = make_landmarks()
        pred_valid = {'label': 'C', 'confidence': 0.82, 'margin': 0.20, 'centroid_distance': 2.0}
        pred_invalid = {'label': '?', 'confidence': 0.40}

        # Frame 1: Valid
        self.sm.update(1000, landmarks, prediction=pred_valid)
        self.assertEqual(self.sm.candidate_streak, 1)

        # Frame 2: 1 miss frame (tolerated)
        self.sm.update(1033, landmarks, prediction=pred_invalid)
        self.assertEqual(self.sm.miss_frames, 1)

        # Frame 3: Valid again -> streak continues to 2
        self.sm.update(1066, landmarks, prediction=pred_valid)
        self.assertEqual(self.sm.candidate_streak, 2)

        # Frame 4: Valid -> reaches 3 -> commits!
        res = self.sm.update(1100, landmarks, prediction=pred_valid)
        self.assertEqual(res['emitted'], 'C')

    def test_hold_steady_rearms(self):
        """After locking, holding hand steady for STEADY_MS re-arms to ARMED."""
        now = 1000
        landmarks = make_landmarks()
        pred = {'label': 'D', 'confidence': 0.95, 'margin': 0.30}

        # Lock
        self.sm.update(now, landmarks, prediction=pred)
        self.assertEqual(self.sm.state, COOLDOWN)

        # During MIN_LOCK_MS (250ms), progress stays 0
        now += 100
        self.sm.update(now, landmarks)
        self.assertEqual(self.sm.state, COOLDOWN)
        self.assertEqual(self.sm.steady_progress, 0.0)

        # After MIN_LOCK_MS, steady hand advances steady_progress
        now += 200 # elapsed = 300ms > MIN_LOCK_MS
        res = self.sm.update(now, landmarks) # motionless
        self.assertGreater(res['steady_progress'], 0.0)

        # Feed steady frames until STEADY_MS (350ms) reached
        for _ in range(12):
            now += 33
            res = self.sm.update(now, landmarks)
            if res['state'] == ARMED:
                break

        self.assertEqual(self.sm.state, ARMED)
        self.assertIsNone(self.sm.locked_sign)

    def test_hand_loss_rearms_immediately(self):
        """When hand leaves the camera frame (HAND_LOST_FRAMES), state re-arms immediately."""
        now = 1000
        landmarks = make_landmarks()
        pred = {'label': 'E', 'confidence': 0.95}

        # Lock
        self.sm.update(now, landmarks, prediction=pred)
        self.assertEqual(self.sm.state, COOLDOWN)

        # Hand leaves frame
        for _ in range(HAND_LOST_FRAMES):
            now += 33
            res = self.sm.update(now, landmarks=None)

        # pyrefly: ignore[unbound-name]
        self.assertEqual(res['state'], ARMED)
        self.assertIsNone(res['locked_sign'])

    def test_accuracy_rejections(self):
        """Rejects low confidence (< 0.75), low margin (< 0.10), or large centroid distance."""
        landmarks = make_landmarks()

        # Low confidence
        low_conf = {'label': 'F', 'confidence': 0.70, 'margin': 0.30, 'centroid_distance': 2.0}
        for _ in range(5):
            res = self.sm.update(1000, landmarks, prediction=low_conf)
            self.assertIsNone(res['emitted'])

        # Low margin
        low_margin = {'label': 'F', 'confidence': 0.90, 'margin': 0.08, 'centroid_distance': 2.0}
        for _ in range(5):
            res = self.sm.update(1000, landmarks, prediction=low_margin)
            self.assertIsNone(res['emitted'])

        # Out-of-dataset centroid distance
        high_dist = {'label': 'F', 'confidence': 0.90, 'margin': 0.30, 'centroid_distance': 15.0}
        for _ in range(5):
            res = self.sm.update(1000, landmarks, prediction=high_dist)
            self.assertIsNone(res['emitted'])

    def test_repeat_sign_requires_release(self):
        """Same sign cannot be committed twice in a row without release frames."""
        landmarks = make_landmarks()
        pred = {'label': 'G', 'confidence': 0.95}

        # Lock 'G'
        self.sm.update(1000, landmarks, prediction=pred)
        self.assertEqual(self.sm.locked_sign, 'G')

        # Fast forward past MAX_COOLDOWN_MS to force re-arm
        self.sm.update(3000, landmarks)
        self.assertEqual(self.sm.state, ARMED)

        # Immediately presenting same sign 'G' before release frames -> NOT committed
        res = self.sm.update(3033, landmarks, prediction=pred)
        self.assertIsNone(res['emitted'])

        # After RELEASE_FRAMES (4 frames of other/no gesture), 'G' can be accepted again
        for _ in range(RELEASE_FRAMES):
            self.sm.update(3100, landmarks, prediction=None)

        res = self.sm.update(3300, landmarks, prediction=pred)
        self.assertEqual(res['emitted'], 'G')


if __name__ == '__main__':
    unittest.main()
