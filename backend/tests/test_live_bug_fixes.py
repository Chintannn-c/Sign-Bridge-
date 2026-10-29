"""
Regression and verification test suite for the two live-use bugs:
BUG 1: Letter 'H' dual-hand slot invariance and canonical ordering.
BUG 2: Word 'NAMASTE' idle hallucination prevention and wrist-relative motion gating.
"""

import json
from pathlib import Path
import numpy as np
import pytest

from services.landmark_utils import assign_canonical_hand_slots
from services.word_recognizer import WordRecognizer, compute_wrist_relative_motion


@pytest.fixture(scope="module")
def word_recognizer():
    rec = WordRecognizer()
    return rec


def test_synthetic_idle_static_rejected(word_recognizer):
    """Two completely static open hands must be rejected by motion gate with static_hand_detected."""
    static_hands = np.zeros((30, 2, 21, 3), dtype=np.float32)
    static_hands[:, 0, 0] = [0.4, 0.65, 0.0]
    static_hands[:, 1, 0] = [0.6, 0.65, 0.0]
    for j in range(1, 21):
        static_hands[:, 0, j] = static_hands[:, 0, 0] + np.array([-0.03 * (j % 5), -0.04 * (j // 5), 0.0])
        static_hands[:, 1, j] = static_hands[:, 1, 0] + np.array([0.03 * (j % 5), -0.04 * (j // 5), 0.0])

    res = word_recognizer.predict(static_hands.reshape(30, 126))
    assert res is not None
    assert res['rejected'] is True
    assert res['rejection_reason'] == 'static_hand_detected'
    assert res['word'] == '?'


def test_synthetic_idle_jitter_never_namaste(word_recognizer):
    """Two static open hands with natural camera jitter must NOT be predicted as NAMASTE."""
    rng = np.random.default_rng(42)
    idle_hands = np.zeros((30, 2, 21, 3), dtype=np.float32)
    idle_hands[:, 0, 0] = [0.4, 0.65, 0.0]
    idle_hands[:, 1, 0] = [0.6, 0.65, 0.0]
    for j in range(1, 21):
        idle_hands[:, 0, j] = idle_hands[:, 0, 0] + np.array([-0.03 * (j % 5), -0.04 * (j // 5), 0.0])
        idle_hands[:, 1, j] = idle_hands[:, 1, 0] + np.array([0.03 * (j % 5), -0.04 * (j // 5), 0.0])
    idle_hands += rng.normal(0, 0.002, idle_hands.shape).astype(np.float32)

    res = word_recognizer.predict(idle_hands.reshape(30, 126))
    assert res is not None
    # Must either be rejected (static_hand_detected, low_confidence, idle_detected) or NOT be NAMASTE
    assert res['word'] != 'NAMASTE'


def test_hands_entering_frame_glitch_suppression(word_recognizer):
    """Hands entering frame (from 0 to coordinate) must not hallucinate NAMASTE."""
    seq = np.zeros((30, 2, 21, 3), dtype=np.float32)
    for t in range(15, 30):
        y_pos = 0.85 - (t - 15) * 0.005
        seq[t, 0, 0] = [0.4, y_pos, 0.0]
        seq[t, 1, 0] = [0.6, y_pos, 0.0]
        for j in range(1, 21):
            seq[t, 0, j] = seq[t, 0, 0] + np.array([-0.02 * (j % 5), -0.03 * (j // 5), 0.0])
            seq[t, 1, j] = seq[t, 1, 0] + np.array([0.02 * (j % 5), -0.03 * (j // 5), 0.0])

    res = word_recognizer.predict(seq.reshape(30, 126))
    assert res is not None
    # Must never hallucinate NAMASTE on entry transition
    assert res['word'] != 'NAMASTE'


def test_real_namaste_passes(word_recognizer):
    """Real NAMASTE video sequence from dataset must pass motion gate and yield word prediction."""
    namaste_files = list(Path(__file__).resolve().parent.parent.glob('dataset_words/NAMASTE/*.json'))
    if not namaste_files:
        pytest.skip("No NAMASTE dataset files found")

    data = json.loads(namaste_files[0].read_text(encoding='utf-8'))
    frames = data.get('frames', [])
    if len(frames) < 30:
        pytest.skip("NAMASTE file has fewer than 30 frames")

    arr = np.array(frames[:30], dtype=np.float32)
    vel = compute_wrist_relative_motion(arr)
    # Real gesture motion must easily pass the 0.025 gate
    assert vel >= 0.025

    res = word_recognizer.predict(arr)
    assert res is not None
    assert res['rejection_reason'] != 'static_hand_detected'


def test_h_canonical_slot_invariance():
    """Verifies that whether hands are presented in original or swapped order, canonical slots match."""
    hand_left = np.full((21, 3), 0.85, dtype=np.float32)
    hand_right = np.full((21, 3), 0.21, dtype=np.float32)

    # Orientation A: Hand 0 is left (0.85), Hand 1 is right (0.21)
    s0_a, s1_a = assign_canonical_hand_slots([
        {'coords': hand_left, 'label': 'Left', 'score': 0.9},
        {'coords': hand_right, 'label': 'Right', 'score': 0.9},
    ])

    # Orientation B: Hand 0 is right (0.21), Hand 1 is left (0.85)
    s0_b, s1_b = assign_canonical_hand_slots([
        {'coords': hand_right, 'label': 'Right', 'score': 0.9},
        {'coords': hand_left, 'label': 'Left', 'score': 0.9},
    ])

    # Both orientations must assign Hand Left (0.85) to Slot 0 and Hand Right (0.21) to Slot 1
    assert np.allclose(s0_a, s0_b)
    assert np.allclose(s1_a, s1_b)
    assert np.allclose(s0_a, hand_left)
    assert np.allclose(s1_a, hand_right)
