"""
SignBridge — CNN-BiLSTM Temporal Classifier for ISL Whole-Word Recognition.
"""

import torch
import torch.nn as nn


class CNNBiLSTMWordClassifier(nn.Module):
    """
    Hybrid 1D-CNN + Bidirectional LSTM with optional temporal attention
    for ISL whole-word classification from sequence landmarks.
    """
    def __init__(
        self,
        in_dim: int = 126,
        cnn_ch: int = 128,
        hid: int = 128,
        nc: int = 17,
        nl: int = 2,
        dp: float = 0.3,
        attn_enabled: bool = False
    ):
        super().__init__()
        self.attn_enabled = attn_enabled
        self.cnn = nn.Sequential(
            nn.Conv1d(in_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, cnn_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_ch),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.lstm = nn.LSTM(
            input_size=cnn_ch,
            hidden_size=hid,
            num_layers=nl,
            batch_first=True,
            bidirectional=True,
            dropout=dp if nl > 1 else 0.0,
        )
        if self.attn_enabled:
            self.attn = nn.Sequential(
                nn.Linear(hid * 2, 64),
                nn.Tanh(),
                nn.Linear(64, 1),
            )
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(hid * 2),
            nn.Linear(hid * 2, 64),
            nn.ReLU(),
            nn.Dropout(dp),
            nn.Linear(64, nc),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, in_dim)
        x_cnn = self.cnn(x.permute(0, 2, 1))
        lstm_out, _ = self.lstm(x_cnn.permute(0, 2, 1))
        if self.attn_enabled:
            w = torch.softmax(self.attn(lstm_out), dim=1)
            pooled = torch.sum(lstm_out * w, dim=1)
        else:
            pooled = torch.mean(lstm_out, dim=1)
        return self.classifier(pooled)

