"""
SignBridge — Two-Stream Temporal Conformer for ISL Whole-Word Recognition.

Separates Signing-Space Trajectory (Stream A: clip-anchored coordinates) from
Handshape Posture (Stream B: wrist-centered finger pose), fusing them through a
lightweight depthwise temporal Conv1d + TransformerEncoder + attention pooling.
"""

import torch
import torch.nn as nn


class TemporalConformerWordClassifier(nn.Module):
    """Lightweight Two-Stream Conformer/Transformer word classifier for 30-frame sequences."""

    def __init__(
        self,
        in_dim: int = 252,
        num_classes: int = 57,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.25,
        max_len: int = 30,
    ):
        super().__init__()
        self.in_dim = in_dim
        half_dim = in_dim // 2
        half_model = d_model // 2

        self.traj_proj = nn.Sequential(
            nn.Linear(half_dim, half_model),
            nn.LayerNorm(half_model),
            nn.GELU(),
        )
        self.shape_proj = nn.Sequential(
            nn.Linear(in_dim - half_dim, d_model - half_model),
            nn.LayerNorm(d_model - half_model),
            nn.GELU(),
        )

        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        self.local_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.Dropout(dropout * 0.5),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)

        self.attn = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, T, in_dim)
        half = self.in_dim // 2
        z_traj = self.traj_proj(x[:, :, :half])
        z_shape = self.shape_proj(x[:, :, half:])
        h = torch.cat([z_traj, z_shape], dim=-1)

        T = h.shape[1]
        h = h + self.pos_emb[:, :T, :]
        h = h + self.local_conv(h.permute(0, 2, 1)).permute(0, 2, 1)
        h = self.encoder(h)

        w = torch.softmax(self.attn(h), dim=1)
        pooled = torch.sum(h * w, dim=1)
        return self.classifier(pooled)
