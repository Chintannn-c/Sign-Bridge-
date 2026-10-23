"""
Unit tests for SignBridge Gesture Recognition Pipeline and State Machine.

Tests all required acceptance scenarios:
  1. No-hand input rejection
  2. Idle / resting hand detection
  3. Neutral-gesture ("two open palms") detection & non-neutral rejection
  4. Mandatory re-arm between two consecutive gestures (including repeat of same letter)
  5. Low-confidence & low-margin rejection
  6. Out-of-dataset gesture rejection (centroid distance threshold)
  7. Cooldown & duplicate emission prevention
"""

import unittest
import numpy as np
from pathlib import Path

from services.gesture_config import (
    NEUTRAL_HOLD_FRAMES,
    STABILITY_FRAMES_K,
    CONFIDENCE_THRESHOLD,
    MARGIN_THRESHOLD,
)

COOLDOWN_FRAMES = 15
from services.gesture_pipeline import (
    is_neutral_gesture,
    is_idle_hand,
    CentroidValidator,
    GestureStateMachine,
)


def create_open_palm_hand(wrist_x=0.5, wrist_y=0.7, z_normal=-1.0):
    """Generates synthetic 21-landmark coordinates for an open hand facing camera."""
    pts = np.zeros((21, 3), dtype=np.float32)
    # 0: Wrist
    pts[0] = [wrist_x, wrist_y, 0.0]

    # Thumb: 1, 2, 3, 4
    pts[1] = [wrist_x - 0.04, wrist_y - 0.05, 0.0]
    pts[2] = [wrist_x - 0.07, wrist_y - 0.10, 0.0]
    pts[3] = [wrist_x - 0.10, wrist_y - 0.15, 0.0]
    pts[4] = [wrist_x - 0.13, wrist_y - 0.19, 0.0]  # Tip extended

    # Index: 5, 6, 7, 8
    pts[5] = [wrist_x - 0.05, wrist_y - 0.16, 0.0]
    pts[6] = [wrist_x - 0.05, wrist_y - 0.26, 0.0]
    pts[7] = [wrist_x - 0.05, wrist_y - 0.35, 0.0]
    pts[8] = [wrist_x - 0.05, wrist_y - 0.44, 0.0]  # Tip extended

    # Middle: 9, 10, 11, 12
    pts[9] = [wrist_x, wrist_y - 0.17, 0.0]
    pts[10] = [wrist_x, wrist_y - 0.28, 0.0]
    pts[11] = [wrist_x, wrist_y - 0.38, 0.0]
    pts[12] = [wrist_x, wrist_y - 0.48, 0.0] # Tip extended

    # Ring: 13, 14, 15, 16
    pts[13] = [wrist_x + 0.04, wrist_y - 0.16, 0.0]
    pts[14] = [wrist_x + 0.04, wrist_y - 0.26, 0.0]
    pts[15] = [wrist_x + 0.04, wrist_y - 0.35, 0.0]
    pts[16] = [wrist_x + 0.04, wrist_y - 0.43, 0.0] # Tip extended

    # Pinky: 17, 18, 19, 20
    pts[17] = [wrist_x + 0.08, wrist_y - 0.14, 0.0]
    pts[18] = [wrist_x + 0.08, wrist_y - 0.22, 0.0]
    pts[19] = [wrist_x + 0.08, wrist_y - 0.30, 0.0]
    pts[20] = [wrist_x + 0.08, wrist_y - 0.38, 0.0] # Tip extended

    return pts


def create_two_open_palms():
    """Returns 126 floats for two open palms facing camera."""
    left = create_open_palm_hand(wrist_x=0.35, wrist_y=0.7)
    right = create_open_palm_hand(wrist_x=0.65, wrist_y=0.7)
    return np.concatenate([left.flatten(), right.flatten()])


def create_fist_hand(wrist_x=0.5, wrist_y=0.7):
    """Generates synthetic 21-landmark coordinates for a closed fist."""
    pts = np.zeros((21, 3), dtype=np.float32)
    pts[0] = [wrist_x, wrist_y, 0.0]
    # Fingers curled in toward palm (tips close to MCPs)
    for base, mcp, tip in [(5, 5, 8), (9, 9, 12), (13, 13, 16), (17, 17, 20)]:
        pts[mcp] = [wrist_x + (mcp - 10) * 0.015, wrist_y - 0.10, 0.0]
        pts[tip] = [wrist_x + (mcp - 10) * 0.015, wrist_y - 0.08, 0.0]  # Curled inside
    pts[4] = [wrist_x, wrist_y - 0.09, 0.0]
    return pts


class TestGesturePipeline(unittest.TestCase):

    def setUp(self):
        self.pipeline = GestureStateMachine(
            neutral_hold_frames=NEUTRAL_HOLD_FRAMES,
            stability_k=STABILITY_FRAMES_K,
            cooldown_frames=COOLDOWN_FRAMES,
        )

    def test_no_hand_input(self):
        """Test 1: No hand or all-zero landmarks emit nothing and are recognized as idle."""
        zeros = np.zeros(126, dtype=np.float32)
        self.assertTrue(is_idle_hand(zeros))
        self.assertFalse(is_neutral_gesture(zeros))

        res = self.pipeline.process_frame(zeros, prediction={'letter': 'A', 'confidence': 0.95})
        self.assertIsNone(res['emitted'])
        self.assertEqual(res['state'], 'NEUTRAL')

    def test_idle_hand(self):
        """Test 2: Hand resting low at bottom of frame is classified as idle and produces no output."""
        hand = create_open_palm_hand(wrist_x=0.5, wrist_y=0.92)  # Below IDLE_WRIST_Y_THRESHOLD
        landmarks = np.zeros(126, dtype=np.float32)
        landmarks[:63] = hand.flatten()

        self.assertTrue(is_idle_hand(landmarks))
        res = self.pipeline.process_frame(landmarks, prediction={'letter': 'A', 'confidence': 0.95})
        self.assertIsNone(res['emitted'])

    def test_neutral_gesture_detection(self):
        """Test 3: Two open palms is detected as neutral base; fists and single hands are not."""
        neutral_frame = create_two_open_palms()
        self.assertTrue(is_neutral_gesture(neutral_frame))

        # Single open hand -> NOT neutral
        single_hand = np.zeros(126, dtype=np.float32)
        single_hand[:63] = create_open_palm_hand().flatten()
        self.assertFalse(is_neutral_gesture(single_hand))

        # Two closed fists -> NOT neutral
        fist_frame = np.concatenate([
            create_fist_hand(0.35).flatten(),
            create_fist_hand(0.65).flatten()
        ])
        self.assertFalse(is_neutral_gesture(fist_frame))

    def test_rearm_between_two_gestures(self):
        """Test 4: State machine requires holding neutral to ARMED, then CAPTURING, COOLDOWN, and back to NEUTRAL before re-arming."""
        neutral_frame = create_two_open_palms()
        gesture_frame = np.concatenate([create_fist_hand(0.35).flatten(), create_open_palm_hand(0.65).flatten()])

        # 1. Initially NEUTRAL
        self.assertEqual(self.pipeline.state, 'NEUTRAL')

        # 2. Must hold neutral for NEUTRAL_HOLD_FRAMES frames to reach ARMED
        for i in range(NEUTRAL_HOLD_FRAMES - 1):
            res = self.pipeline.process_frame(neutral_frame)
            self.assertEqual(res['state'], 'NEUTRAL')
            self.assertIsNone(res['emitted'])

        # On the Nth frame, transitions to ARMED
        res = self.pipeline.process_frame(neutral_frame)
        self.assertEqual(res['state'], 'ARMED')

        # 3. User begins gesture -> transitions to CAPTURING
        pred = {'letter': 'A', 'confidence': 0.92, 'margin': 0.35}
        res = self.pipeline.process_frame(gesture_frame, prediction=pred)
        self.assertEqual(res['state'], 'CAPTURING')
        self.assertIsNone(res['emitted'])

        # 4. Hold gesture for STABILITY_FRAMES_K frames
        for i in range(STABILITY_FRAMES_K - 2):
            res = self.pipeline.process_frame(gesture_frame, prediction=pred)
            self.assertIsNone(res['emitted'])
            self.assertEqual(res['state'], 'CAPTURING')

        # On reaching K frames, gesture is EMITTED and state enters COOLDOWN
        res = self.pipeline.process_frame(gesture_frame, prediction=pred)
        self.assertEqual(res['emitted'], 'A')
        self.assertEqual(res['state'], 'COOLDOWN')

        # 5. During COOLDOWN, even if identical gesture continues, emit nothing
        res = self.pipeline.process_frame(gesture_frame, prediction=pred)
        self.assertIsNone(res['emitted'])

        # Step through remainder of cooldown
        for _ in range(COOLDOWN_FRAMES):
            self.pipeline.process_frame(gesture_frame, prediction=pred)

        # After cooldown expires, state returns to NEUTRAL
        self.assertEqual(self.pipeline.state, 'NEUTRAL')

        # 6. Attempting to repeat same gesture 'A' without neutral MUST NOT emit
        for _ in range(STABILITY_FRAMES_K + 5):
            res = self.pipeline.process_frame(gesture_frame, prediction=pred)
            self.assertIsNone(res['emitted'])
            self.assertEqual(res['state'], 'NEUTRAL')

        # 7. Re-arming: Hold neutral again for NEUTRAL_HOLD_FRAMES
        for _ in range(NEUTRAL_HOLD_FRAMES):
            self.pipeline.process_frame(neutral_frame)
        self.assertEqual(self.pipeline.state, 'ARMED')

    def test_low_confidence_rejection(self):
        """Test 5: Predictions below CONFIDENCE_THRESHOLD or MARGIN_THRESHOLD are rejected and never emitted."""
        # Arm the state machine
        neutral_frame = create_two_open_palms()
        for _ in range(NEUTRAL_HOLD_FRAMES):
            self.pipeline.process_frame(neutral_frame)
        self.assertEqual(self.pipeline.state, 'ARMED')

        gesture_frame = np.concatenate([create_fist_hand(0.35).flatten(), create_open_palm_hand(0.65).flatten()])

        # Low confidence prediction (0.60 < 0.85)
        low_conf_pred = {'letter': 'B', 'confidence': 0.60, 'margin': 0.30}
        for _ in range(STABILITY_FRAMES_K + 5):
            res = self.pipeline.process_frame(gesture_frame, prediction=low_conf_pred)
            self.assertIsNone(res['emitted'])
            self.assertTrue(res['rejected'])
            self.assertEqual(res['rejection_reason'], 'low_confidence')

        # Low margin prediction (margin 0.05 < 0.15)
        low_margin_pred = {'letter': 'B', 'confidence': 0.90, 'margin': 0.05}
        for _ in range(STABILITY_FRAMES_K + 5):
            res = self.pipeline.process_frame(gesture_frame, prediction=low_margin_pred)
            self.assertIsNone(res['emitted'])
            self.assertTrue(res['rejected'])
            self.assertEqual(res['rejection_reason'], 'low_margin')

    def test_out_of_dataset_gesture_rejection(self):
        """Test 6: Valid model prediction is rejected if landmark distance exceeds dataset centroid threshold."""
        validator = CentroidValidator()

        # Synthetic landmark vector extremely far from class A's centroid
        far_landmarks = np.ones(126, dtype=np.float32) * 25.0
        val_ok, dist, threshold, reason = validator.validate(far_landmarks, 'A')

        self.assertFalse(val_ok)
        self.assertEqual(reason, 'out_of_dataset_distance')
        self.assertGreater(dist, threshold)

        # Arm pipeline
        neutral_frame = create_two_open_palms()
        for _ in range(NEUTRAL_HOLD_FRAMES):
            self.pipeline.process_frame(neutral_frame)

        # Process frame with high confidence but out-of-dataset distance
        pred = {'letter': 'A', 'confidence': 0.95, 'margin': 0.40}
        res = self.pipeline.process_frame(neutral_frame, prediction=pred, norm_landmarks=far_landmarks)
        self.assertIsNone(res['emitted'])
        self.assertTrue(res['rejected'])
        self.assertEqual(res['rejection_reason'], 'out_of_dataset_distance')

    def test_cooldown_duplicate_prevention(self):
        """Test 7: Cooldown strictly prevents immediate repeated emissions."""
        # Arm pipeline
        neutral_frame = create_two_open_palms()
        for _ in range(NEUTRAL_HOLD_FRAMES):
            self.pipeline.process_frame(neutral_frame)

        gesture_frame = np.concatenate([create_fist_hand(0.35).flatten(), create_open_palm_hand(0.65).flatten()])
        pred = {'letter': 'C', 'confidence': 0.95, 'margin': 0.35}

        # Step until emitted
        emissions = []
        for _ in range(STABILITY_FRAMES_K + 2):
            res = self.pipeline.process_frame(gesture_frame, prediction=pred)
            if res['emitted']:
                emissions.append(res['emitted'])

        self.assertEqual(emissions, ['C'])
        self.assertEqual(self.pipeline.state, 'COOLDOWN')

        # In COOLDOWN state, no emissions allowed
        for _ in range(COOLDOWN_FRAMES - 1):
            res = self.pipeline.process_frame(gesture_frame, prediction=pred)
            self.assertIsNone(res['emitted'])

    def test_translator_model_schema(self):
        """Test 8: TranslatorModel output schema includes margin, all_scores, centroid_distance."""
        from services.translator_model import TranslatorModel
        model = TranslatorModel()
        landmarks = np.zeros(126, dtype=np.float32).tolist()
        res = model.predict(landmarks)
        self.assertIn('margin', res)
        self.assertIn('all_scores', res)
        self.assertIn('centroid_distance', res)
        self.assertIn('rejected', res)
        self.assertTrue(res['rejected'])


if __name__ == '__main__':
    unittest.main()

