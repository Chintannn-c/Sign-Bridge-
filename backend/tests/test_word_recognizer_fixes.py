"""
Unit tests for WordRecognizer fixes in SignBridge.
Verifies:
  1. Fast-wave motion (> 0.15 jump) is clipped rather than dropped, not rejected as static.
  2. Missing or bad body_anchors leave model probabilities and predictions unchanged.
  3. Exactly-30-frame input undergoes the same stroke trimming and padding.
  4. Non-finite / NaN inputs are sanitized and do not crash or produce NaNs.
  5. Consistent 9-key return schema across every early-return and valid inference path.
  6. Configurable motion gate and static-pose word exemption (e.g. NAMASTE, ME).
  7. get_info reports input_dim and metadata safely without stale state.
"""

import numpy as np
import pytest
import torch

from models.st_gcn import STGCN_BiLSTM_WordClassifier
from services.word_recognizer import (
    WordRecognizer,
    compute_wrist_relative_motion,
    MIN_MOTION_VELOCITY,
    MAX_SINGLE_FRAME_JUMP,
)

REQUIRED_SCHEMA_KEYS = {
    'word',
    'confidence',
    'margin',
    'entropy',
    'mode',
    'rejected',
    'rejection_reason',
    'motion_velocity',
    'all_scores',
}


def _create_mock_recognizer(num_classes=3, labels=None, input_dim=126):
    """Helper to construct a controlled WordRecognizer with ST-GCN classifier."""
    labels = labels or ['HELLO', 'THANK_YOU', 'NAMASTE']
    model = STGCN_BiLSTM_WordClassifier(in_dim=input_dim, num_classes=num_classes)
    model.eval()

    wr = WordRecognizer()
    wr.model = model
    wr.mode = 'stgcn_bilstm'
    wr.input_dim = input_dim
    wr.labels = labels
    wr.is_available = True
    return wr


def test_fast_wave_motion_not_rejected():
    """Fast-wave motion exceeding max_jump (0.15) is clamped, not dropped, and preserves velocity."""
    T = 30
    seq = np.zeros((T, 126), dtype=np.float32)

    # Simulate a fast wave: wrist alternates with jumps of 0.25 per frame (> 0.15)
    for t in range(T):
        seq[t, 0] = 0.3 if t % 2 == 0 else 0.55  # Wrist X jumps by 0.25
        seq[t, 1] = 0.5
        seq[t, 3:15] = 0.2  # Hand landmarks present

    vel = compute_wrist_relative_motion(seq, max_jump=MAX_SINGLE_FRAME_JUMP)
    # Velocity must be clamped to at least max_jump and well above MIN_MOTION_VELOCITY
    assert vel >= MAX_SINGLE_FRAME_JUMP
    assert vel > MIN_MOTION_VELOCITY

    wr = _create_mock_recognizer()
    res = wr.predict(seq)
    assert res is not None
    # Must NOT be rejected as static_hand_detected
    assert res.get('rejection_reason') != 'static_hand_detected'
    assert res['motion_velocity'] >= MAX_SINGLE_FRAME_JUMP


def test_missing_body_anchors_leaving_probs_unchanged():
    """Missing, None, non-numeric, or absent-hand body anchors do not alter predictions."""
    wr = _create_mock_recognizer(num_classes=4, labels=['THANK_YOU', 'NAMASTE', 'HELLO', 'WATER'])

    # Bias class 0 ('THANK_YOU')
    wr.model.classifier[-1].bias.data[0] = 15.0

    seq = []
    for t in range(30):
        frame = np.zeros(126, dtype=np.float32)
        frame[0] = 0.5 + t * 0.01
        frame[1] = 0.5 + t * 0.005
        frame[3:15] = 0.2
        seq.append(frame)

    baseline = wr.predict(seq, body_anchors=None)
    assert baseline['word'] == 'THANK_YOU'
    assert not baseline['rejected']

    # 1. Empty body_anchors dict
    empty_res = wr.predict(seq, body_anchors={})
    assert empty_res['word'] == baseline['word']
    assert empty_res['confidence'] == baseline['confidence']
    assert empty_res['margin'] == baseline['margin']

    # 2. Non-numeric or None anchor values
    corrupt_anchors = {
        'l_hand_to_mouth': 'corrupt_string',
        'r_hand_to_mouth': None,
        'l_hand_to_chest': float('nan'),
        'r_hand_to_chest': float('inf'),
        'l_elevation': 'invalid',
    }
    corrupt_res = wr.predict(seq, body_anchors=corrupt_anchors)
    assert corrupt_res is not None
    assert corrupt_res['word'] == baseline['word']
    assert corrupt_res['confidence'] == baseline['confidence']

    # 3. Absent hand keys (should NOT apply 0.50x penalty)
    partial_anchors = {'some_other_key': 123.4}
    partial_res = wr.predict(seq, body_anchors=partial_anchors)
    assert partial_res['word'] == baseline['word']
    assert partial_res['confidence'] == baseline['confidence']


def test_exactly_30_frame_input_same_trimming():
    """A 30-frame sequence goes through the exact same stroke trimming and padding logic."""
    wr = _create_mock_recognizer()

    # 30 frames: 6 stationary + 18 moving + 6 stationary
    frames = []
    for _ in range(6):
        f = np.zeros(126, dtype=np.float32)
        f[0], f[1] = 0.3, 0.4
        f[3:15] = 0.1
        frames.append(f)
    for t in range(18):
        f = np.zeros(126, dtype=np.float32)
        f[0] = 0.3 + t * 0.025
        f[1] = 0.4 + t * 0.02
        f[3:15] = 0.1 + t * 0.005
        frames.append(f)
    for _ in range(6):
        f = np.zeros(126, dtype=np.float32)
        f[0], f[1] = 0.8, 0.8
        f[3:15] = 0.2
        frames.append(f)

    # Must pad or trim cleanly to exactly (30, 126)
    trimmed = wr._pad_or_trim(frames)
    assert trimmed.shape == (30, 126)
    assert not np.isnan(trimmed).any()

    # Inference on 30 frames must succeed without dimension error
    result = wr.predict(frames)
    assert result is not None
    assert REQUIRED_SCHEMA_KEYS.issubset(result.keys())


def test_nan_input_handling():
    """NaN, Inf, and non-finite inputs are sanitized to 0.0 and do not crash."""
    wr = _create_mock_recognizer()

    # Sequence filled entirely with NaNs
    nan_seq = np.full((30, 126), np.nan, dtype=np.float32)
    res_nan = wr.predict(nan_seq)
    assert res_nan is not None
    assert res_nan['rejected'] is True
    assert res_nan['word'] == '?'
    assert res_nan['confidence'] == 0.0
    assert not np.isnan(res_nan['motion_velocity'])
    assert REQUIRED_SCHEMA_KEYS.issubset(res_nan.keys())

    # Sequence with isolated NaNs in otherwise valid motion
    seq = np.zeros((30, 126), dtype=np.float32)
    for t in range(30):
        seq[t, 0] = 0.4 + t * 0.01
        seq[t, 1] = 0.5
        seq[t, 3:15] = 0.2
    seq[5, 10] = np.nan
    seq[12, 20] = np.inf

    res_partial = wr.predict(seq)
    assert res_partial is not None
    assert REQUIRED_SCHEMA_KEYS.issubset(res_partial.keys())
    assert not np.isnan(res_partial['confidence'])
    assert not np.isnan(res_partial['margin'])
    assert not np.isnan(res_partial['entropy'])


def test_consistent_return_schema_on_every_path():
    """Every early return and completion path returns the full 9-key dict schema."""
    wr = _create_mock_recognizer()

    # Path 1: Empty hand sequence (all 0s)
    empty_seq = np.zeros((30, 126), dtype=np.float32)
    res_empty = wr.predict(empty_seq)
    assert res_empty is not None
    assert REQUIRED_SCHEMA_KEYS.issubset(res_empty.keys())
    assert res_empty['rejected'] is True
    assert res_empty['rejection_reason'] == 'no_hands_detected'

    # Path 2: Invalid feature dimension (e.g. 50 features instead of 126)
    invalid_dim_seq = np.ones((30, 50), dtype=np.float32)
    res_dim = wr.predict(invalid_dim_seq)
    assert res_dim is not None
    assert REQUIRED_SCHEMA_KEYS.issubset(res_dim.keys())
    assert res_dim['rejected'] is True
    assert res_dim['rejection_reason'] == 'invalid_feature_dim'

    # Path 3: Static hand detected
    static_frame = np.zeros(126, dtype=np.float32)
    static_frame[0], static_frame[1] = 0.5, 0.5
    static_frame[3:15] = 0.2
    static_seq = [static_frame.copy() for _ in range(30)]
    res_static = wr.predict(static_seq)
    assert res_static is not None
    assert REQUIRED_SCHEMA_KEYS.issubset(res_static.keys())
    assert res_static['rejected'] is True
    assert res_static['rejection_reason'] == 'static_hand_detected'

    # Path 4: Model not available
    wr_unavailable = WordRecognizer()
    wr_unavailable.is_available = False
    wr_unavailable.model = None
    res_unavail = wr_unavailable.predict(static_seq)
    assert res_unavail is not None
    assert REQUIRED_SCHEMA_KEYS.issubset(res_unavail.keys())
    assert res_unavail['rejected'] is True
    assert res_unavail['rejection_reason'] == 'model_not_available'

    # Path 5: Caller can access any key without KeyError
    for r in (res_empty, res_dim, res_static, res_unavail):
        assert isinstance(r['rejected'], bool)
        assert isinstance(r['confidence'], float)
        assert isinstance(r['margin'], float)
        assert isinstance(r['entropy'], float)
        assert isinstance(r['motion_velocity'], float)
        assert isinstance(r['all_scores'], dict)


def test_static_pose_words_not_rejected_by_motion_gate():
    """Held signs like NAMASTE are exempted from the motion velocity pre-gate when confident."""
    wr = _create_mock_recognizer(num_classes=3, labels=['HELLO', 'THANK_YOU', 'NAMASTE'])
    # Strong bias for NAMASTE (index 2)
    wr.model.classifier[-1].bias.data[2] = 20.0

    # Completely still sequence (motion velocity = 0.0)
    static_frame = np.zeros(126, dtype=np.float32)
    static_frame[0], static_frame[1] = 0.5, 0.5
    static_frame[3:15] = 0.2
    static_seq = [static_frame.copy() for _ in range(30)]

    res = wr.predict(static_seq)
    assert res is not None
    # Because NAMASTE is a known static sign and prediction is confident, it must NOT be rejected
    assert res['word'] == 'NAMASTE'
    assert res['rejected'] is False
    assert res['rejection_reason'] is None


def test_get_info_reports_input_dim():
    """get_info returns accurate input_dim without stale state."""
    wr = _create_mock_recognizer(input_dim=252)
    info = wr.get_info()
    assert info['available'] is True
    assert info['input_dim'] == 252
    assert info['input_features'] == 252
    assert info['num_classes'] == 3

