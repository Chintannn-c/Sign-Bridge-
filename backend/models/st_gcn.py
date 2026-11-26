"""
SignBridge — Spatial-Temporal Graph Convolutional Network (ST-GCN) for ISL Recognition.

Models the 42 dual-hand MediaPipe landmarks as a physical kinematic graph.
Captures anatomical bone connections, finger curl constraints, and cross-hand relationships.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Anatomical edges for 1 hand (21 joints)
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (0, 9), (9, 10), (10, 11), (11, 12),    # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (5, 9), (9, 13), (13, 17),              # Palm transverse
]

def build_dual_hand_adjacency():
    """
    Builds the 42x42 normalized adjacency matrix with self-loops for dual hands.
    """
    num_nodes = 42
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    # Hand 1 (0..20)
    for i, j in HAND_EDGES:
        A[i, j] = 1.0
        A[j, i] = 1.0

    # Hand 2 (21..41)
    for i, j in HAND_EDGES:
        A[i + 21, j + 21] = 1.0
        A[j + 21, i + 21] = 1.0

    # Inter-hand connection (Wrists)
    A[0, 21] = 1.0
    A[21, 0] = 1.0

    # Add self-loops
    A += np.eye(num_nodes, dtype=np.float32)

    # Degree normalization: D^(-1/2) * A * D^(-1/2)
    degrees = np.sum(A, axis=1)
    d_inv_sqrt = np.power(degrees, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D_mat = np.diag(d_inv_sqrt)
    A_norm = D_mat @ A @ D_mat
    return torch.tensor(A_norm, dtype=torch.float32)


class SpatialGraphConv(nn.Module):
    """
    Spatial Graph Convolution layer.
    Transforms node features using the adjacency matrix A.
    """
    A: torch.Tensor

    def __init__(self, in_channels: int, out_channels: int, A: torch.Tensor):
        super().__init__()
        self.register_buffer('A', A)
        self.fc = nn.Linear(in_channels, out_channels, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, Num_Nodes=42, In_Channels)
        ax = torch.matmul(self.A, x)
        out = self.fc(ax)
        # Permute for BatchNorm1d: (Batch, Out_Channels, Num_Nodes)
        out = out.permute(0, 2, 1)
        out = self.bn(out)
        out = self.relu(out)
        return out.permute(0, 2, 1)


class STGCNHandClassifier(nn.Module):
    """
    ST-GCN Classifier for Static / Frame-Level Dual-Hand ISL Gestures.
    Input: (Batch, 42, 3) or (Batch, 126)
    Output: (Batch, Num_Classes)
    """
    def __init__(self, num_classes=26, in_channels=3, hidden_dim=64):
        super().__init__()
        A = build_dual_hand_adjacency()
        self.gconv1 = SpatialGraphConv(in_channels, hidden_dim, A)
        self.gconv2 = SpatialGraphConv(hidden_dim, hidden_dim * 2, A)
        self.gconv3 = SpatialGraphConv(hidden_dim * 2, hidden_dim * 2, A)
        
        self.fc1 = nn.Linear(hidden_dim * 2, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        if x.ndim == 2:
            x = x.view(-1, 42, 3)
        
        h = self.gconv1(x)
        h = self.gconv2(h)
        h = self.gconv3(h)
        
        # Global graph pooling (mean across 42 joint nodes)
        pooled = torch.mean(h, dim=1)
        out = F.relu(self.fc1(pooled))
        out = self.dropout(out)
        logits = self.fc2(out)
        return logits


class STGCN_BiLSTM_WordClassifier(nn.Module):
    """
    Spatial-Temporal Hybrid Classifier for Whole-Word ISL Gestures:
      1. Spatial Graph Convolutions (ST-GCN) over the 42 physical hand joints per frame.
      2. Temporal 1D Convolution for inter-frame landmark smoothing.
      3. Bidirectional LSTM across the 30-frame temporal trajectory.
      4. Temporal Self-Attention pooling across stroke keyframes.
    Input: (Batch, Sequence_Length=30, In_Dim=252) or (Batch, 30, In_Dim=126)
    Output: (Batch, Num_Classes)
    """
    def __init__(
        self,
        in_dim: int = 252,
        num_classes: int = 57,
        gcn_channels: int = 64,
        lstm_hidden: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        self.in_dim = in_dim
        self.node_channels = in_dim // 42  # 6 if in_dim=252, 3 if in_dim=126
        A = build_dual_hand_adjacency()

        # 1. Spatial Graph Convolutions per frame
        self.gconv1 = SpatialGraphConv(self.node_channels, gcn_channels, A)
        self.gconv2 = SpatialGraphConv(gcn_channels, gcn_channels * 2, A)

        # 2. Temporal 1D Convolution along the sequence
        self.temporal_cnn = nn.Sequential(
            nn.Conv1d(gcn_channels * 2, gcn_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(gcn_channels * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

        # 3. Bidirectional LSTM across sequence frames
        self.lstm = nn.LSTM(
            input_size=gcn_channels * 2,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # 4. Temporal Self-Attention
        self.attn = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # 5. Classification Head
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(lstm_hidden * 2),
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, Sequence_Length=30, 252)
        B, T, _ = x.shape
        x_nodes = x.view(B * T, 42, self.node_channels)

        # Frame-level spatial graph convolution
        g1 = self.gconv1(x_nodes)
        g2 = self.gconv2(g1)
        frame_feat = torch.mean(g2, dim=1)  # (B * T, gcn_channels * 2)

        # Temporal sequence modeling
        seq_feat = frame_feat.view(B, T, -1).permute(0, 2, 1)  # (B, C, T)
        cnn_out = self.temporal_cnn(seq_feat)                  # (B, C, T)
        lstm_in = cnn_out.permute(0, 2, 1)                     # (B, T, C)

        lstm_out, _ = self.lstm(lstm_in)                       # (B, T, 2 * lstm_hidden)
        attn_w = torch.softmax(self.attn(lstm_out), dim=1)     # (B, T, 1)
        pooled = torch.sum(lstm_out * attn_w, dim=1)           # (B, 2 * lstm_hidden)

        return self.classifier(pooled)
