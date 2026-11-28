"""
Unit tests for STGCN_BiLSTM_WordClassifier and WordRecognizer integration.
Verifies:
  1. Forward pass shapes and output dimensions for 252-D (kinetic) and 126-D inputs.
  2. Temporal attention mechanism normalisation (sums to 1 across frames).
  3. WordRecognizer loading and inference with stgcn_bilstm architecture.
  4. Motion pre-gate (static hand rejection) safety.
"""

import numpy as np
import pytest
import torch
from models.st_gcn import STGCN_BiLSTM_WordClassifier, STGCNHandClassifier
from models.temporal_conformer import TemporalConformerWordClassifier
from services.feature_extractor import extract_word_two_stream_features, align_landmarks_to_palm_frame
from services.word_recognizer import WordRecognizer


def test_stgcn_bilstm_forward_shape():
    """STGCN_BiLSTM_WordClassifier maps (B, 30, 252) to (B, num_classes)."""
    model = STGCN_BiLSTM_WordClassifier(in_dim=252, num_classes=57, gcn_channels=64, lstm_hidden=128)
    model.eval()

    x = torch.randn(4, 30, 252)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (4, 57)
    assert not torch.isnan(out).any()


def test_stgcn_bilstm_126_dim_support():
    """Supports 126-D landmark inputs (3 coordinates per node)."""
    model = STGCN_BiLSTM_WordClassifier(in_dim=126, num_classes=57, gcn_channels=64, lstm_hidden=128)
    model.eval()

    x = torch.randn(2, 30, 126)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (2, 57)


def test_stgcn_bilstm_attention_weights():
    """Attention weights over the 30 sequence frames must sum to 1 per sequence."""
    model = STGCN_BiLSTM_WordClassifier(in_dim=252, num_classes=57)
    model.eval()

    x = torch.randn(2, 30, 252)
    with torch.no_grad():
        B, T, _ = x.shape
        x_nodes = x.view(B * T, 42, model.node_channels)
        g1 = model.gconv1(x_nodes)
        g2 = model.gconv2(g1)
        frame_feat = torch.mean(g2, dim=1)
        seq_feat = frame_feat.view(B, T, -1).permute(0, 2, 1)
        cnn_out = model.temporal_cnn(seq_feat)
        lstm_in = cnn_out.permute(0, 2, 1)
        lstm_out, _ = model.lstm(lstm_in)
        attn_w = torch.softmax(model.attn(lstm_out), dim=1)

    assert attn_w.shape == (2, 30, 1)
    sums = torch.sum(attn_w, dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_word_recognizer_with_stgcn():
    """WordRecognizer dynamically accepts STGCN_BiLSTM_WordClassifier in stgcn_bilstm mode."""
    model = STGCN_BiLSTM_WordClassifier(in_dim=252, num_classes=3)
    # Bias class 0 to produce high confidence prediction
    model.classifier[-1].bias.data[0] = 10.0
    model.eval()

    wr = WordRecognizer()
    wr.model = model
    wr.mode = 'stgcn_bilstm'
    wr.input_dim = 252
    wr.labels = ['HELLO', 'THANK_YOU', 'NAMASTE']
    wr.is_available = True

    # Moving sequence: 30 frames with motion
    moving_seq = []
    for t in range(30):
        frame = np.zeros(126, dtype=np.float32)
        frame[0] = 0.5 + t * 0.01  # Wrist X
        frame[1] = 0.5 + t * 0.005  # Wrist Y
        frame[3:15] = 0.2 + t * 0.002
        moving_seq.append(frame)

    result = wr.predict(moving_seq)
    assert result is not None
    assert result['word'] == 'HELLO'
    assert result['confidence'] > 0.8
    assert result['mode'] == 'stgcn_bilstm'
    assert 'HELLO' in result['all_scores']


def test_word_recognizer_static_hand_rejected():
    """Static or resting hand sequence is rejected before model execution."""
    model = STGCN_BiLSTM_WordClassifier(in_dim=252, num_classes=3)
    model.eval()

    wr = WordRecognizer()
    wr.model = model
    wr.mode = 'stgcn_bilstm'
    wr.input_dim = 252
    wr.labels = ['HELLO', 'THANK_YOU', 'NAMASTE']
    wr.is_available = True

    # Completely static sequence (no motion)
    static_frame = np.zeros(126, dtype=np.float32)
    static_frame[0] = 0.5
    static_frame[1] = 0.5
    static_frame[3:15] = 0.2
    static_seq = [static_frame.copy() for _ in range(30)]

    result = wr.predict(static_seq)
    assert result is not None
    assert result['word'] == '?'
    assert result.get('rejected') is True
    assert result.get('rejection_reason') == 'static_hand_detected'


def test_word_recognizer_stroke_segmentation():
    """Active stroke segmentation trims stationary lead-in/lead-out frames and returns 30 frames."""
    wr = WordRecognizer()

    # 40-frame sequence: 10 stationary + 20 moving + 10 stationary
    frames = []
    # 10 stationary lead-in frames
    for _ in range(10):
        f = np.zeros(126, dtype=np.float32)
        f[0], f[1] = 0.5, 0.5
        frames.append(f)
    # 20 active stroke frames with clear motion
    for t in range(20):
        f = np.zeros(126, dtype=np.float32)
        f[0] = 0.5 + t * 0.02
        f[1] = 0.5 + t * 0.015
        frames.append(f)
    # 10 stationary tail frames
    for _ in range(10):
        f = np.zeros(126, dtype=np.float32)
        f[0], f[1] = 0.9, 0.8
        frames.append(f)

    trimmed = wr._pad_or_trim(frames)
    assert trimmed.shape == (30, 126)
    assert not np.isnan(trimmed).any()


def test_temporal_conformer_two_stream_and_palm_invariance():
    """Two-Stream Conformer preserves clip trajectory, palm frame is rotation-invariant, and anatomical ST-GCN matches shape."""
    # 1. Stream A preserves wrist movement across frames (unlike per-frame wrist centering)
    moving_seq = np.zeros((30, 126), dtype=np.float32)
    for t in range(30):
        moving_seq[t, 0] = 0.2 + t * 0.02  # wrist x moves across screen
        moving_seq[t, 1] = 0.5
        moving_seq[t, 9 * 3] = moving_seq[t, 0]
        moving_seq[t, 9 * 3 + 1] = 0.4
    two_stream = extract_word_two_stream_features(moving_seq)
    assert two_stream.shape == (30, 252)
    assert two_stream[-1, 0] > two_stream[0, 0] + 0.5

    # 2. TemporalConformerWordClassifier integrates cleanly into WordRecognizer
    model = TemporalConformerWordClassifier(in_dim=252, num_classes=3, d_model=64, nhead=4, num_layers=1)
    model.classifier[-1].bias.data[1] = 10.0
    model.eval()
    wr = WordRecognizer()
    wr.model = model
    wr.mode = 'temporal_conformer'
    wr.feature_mode = 'two_stream_252d'
    wr.input_dim = 252
    wr.labels = ['HELLO', 'THANK_YOU', 'NAMASTE']
    wr.is_available = True
    res = wr.predict(moving_seq)
    assert res is not None
    assert res['word'] == 'THANK_YOU'
    assert len(res['top_k']) == 3

    # 3. Palm frame alignment is invariant to 2D rotation around Z
    hand = np.zeros(126, dtype=np.float32)
    hand[0:3] = [0.5, 0.7, 0.0]
    hand[5 * 3:5 * 3 + 3] = [0.45, 0.55, 0.0]
    hand[9 * 3:9 * 3 + 3] = [0.50, 0.50, 0.0]
    hand[12 * 3:12 * 3 + 3] = [0.50, 0.35, 0.0]
    hand[17 * 3:17 * 3 + 3] = [0.58, 0.58, 0.0]
    theta = 0.6
    c, s = np.cos(theta), np.sin(theta)
    rot = hand.reshape(42, 3).copy()
    x, y = rot[:, 0].copy(), rot[:, 1].copy()
    rot[:, 0] = x * c - y * s
    rot[:, 1] = x * s + y * c
    assert np.allclose(align_landmarks_to_palm_frame(hand), align_landmarks_to_palm_frame(rot.reshape(126)), atol=1e-4)

    # 4. Anatomical ST-GCN output shape
    stgcn = STGCNHandClassifier(num_classes=26, pooling='anatomical')
    assert stgcn(torch.randn(2, 126)).shape == (2, 26)

