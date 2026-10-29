"""
Unit tests for deterministic canonical hand slot ordering.
Verifies that MediaPipe label flips, colliding labels ('Left'/'Left' or 'Right'/'Right'),
and horizontal coordinate ordering map consistently to canonical 126-D feature slots:
  - Slot 0 (floats 0..62): subject's left hand (screen-right, x >= other_x in RealSign)
  - Slot 1 (floats 63..125): subject's right hand (screen-left, x < other_x in RealSign)
"""

import numpy as np
import pytest
from services.landmark_utils import assign_canonical_hand_slots, get_landmark_vector_from_result


def _make_dummy_hand(wrist_x: float, wrist_y: float = 0.5) -> np.ndarray:
    """Creates a synthetic 21x3 hand landmark array with given wrist coordinate."""
    hand = np.zeros((21, 3), dtype=np.float32)
    hand[0] = [wrist_x, wrist_y, 0.0]
    for i in range(1, 21):
        hand[i] = [wrist_x + 0.01 * (i % 5), wrist_y - 0.02 * (i // 5), 0.0]
    return hand


def test_distinct_valid_labels():
    """Distinct Left and Right labels must map to Slot 0 and Slot 1 respectively."""
    h_left = _make_dummy_hand(0.85)
    h_right = _make_dummy_hand(0.21)

    detected = [
        {'coords': h_left, 'label': 'Left', 'score': 0.95},
        {'coords': h_right, 'label': 'Right', 'score': 0.92},
    ]
    s0, s1 = assign_canonical_hand_slots(detected)
    assert np.allclose(s0, h_left)
    assert np.allclose(s1, h_right)


def test_swapped_input_order_with_labels():
    """Even if Right hand is detected first in list, Slot 0 must receive Left."""
    h_left = _make_dummy_hand(0.85)
    h_right = _make_dummy_hand(0.21)

    detected = [
        {'coords': h_right, 'label': 'Right', 'score': 0.95},
        {'coords': h_left, 'label': 'Left', 'score': 0.92},
    ]
    s0, s1 = assign_canonical_hand_slots(detected)
    assert np.allclose(s0, h_left)
    assert np.allclose(s1, h_right)


def test_colliding_labels_both_right():
    """When MediaPipe labels both hands as Right (common contact glitch), spatial fallback holds."""
    h_left = _make_dummy_hand(0.85)
    h_right = _make_dummy_hand(0.21)

    detected = [
        {'coords': h_left, 'label': 'Right', 'score': 0.88},
        {'coords': h_right, 'label': 'Right', 'score': 0.87},
    ]
    s0, s1 = assign_canonical_hand_slots(detected)
    # Larger X (0.85) must be assigned to Slot 0 (left_hand)
    assert np.allclose(s0, h_left)
    assert np.allclose(s1, h_right)


def test_colliding_labels_both_left():
    """When MediaPipe labels both hands as Left, spatial fallback holds."""
    h_left = _make_dummy_hand(0.85)
    h_right = _make_dummy_hand(0.21)

    detected = [
        {'coords': h_right, 'label': 'Left', 'score': 0.90},
        {'coords': h_left, 'label': 'Left', 'score': 0.90},
    ]
    s0, s1 = assign_canonical_hand_slots(detected)
    assert np.allclose(s0, h_left)
    assert np.allclose(s1, h_right)


def test_unlabelled_two_hands():
    """When handedness metadata is missing completely, spatial ordering assigns consistently."""
    h_left = _make_dummy_hand(0.80)
    h_right = _make_dummy_hand(0.25)

    detected = [
        {'coords': h_left, 'label': None, 'score': 0.85},
        {'coords': h_right, 'label': None, 'score': 0.85},
    ]
    s0, s1 = assign_canonical_hand_slots(detected)
    assert np.allclose(s0, h_left)
    assert np.allclose(s1, h_right)


def test_single_hand_assignment():
    """Single hand with Left label or x >= 0.5 maps to Slot 0; Right or x < 0.5 to Slot 1."""
    h_l = _make_dummy_hand(0.70)
    h_r = _make_dummy_hand(0.30)

    s0_l, s1_l = assign_canonical_hand_slots([{'coords': h_l, 'label': 'Left', 'score': 0.9}])
    assert s0_l is not None and s1_l is None

    s0_r, s1_r = assign_canonical_hand_slots([{'coords': h_r, 'label': 'Right', 'score': 0.9}])
    assert s0_r is None and s1_r is not None
