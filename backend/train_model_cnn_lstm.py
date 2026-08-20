"""
SignBridge — Hybrid 1D-CNN + Bi-LSTM Temporal Word Classifier Trainer (PyTorch)

Architecture:
  1D-CNN extracts local finger-joint geometry patterns per frame.
  Bi-LSTM tracks temporal motion trajectory across the 30-frame window.

Input:  (batch_size, 30, 126) — 30 frames x 126 landmark features
Output: (batch_size, num_word_classes)

Outputs:
  backend/models/isl_cnn_lstm_word_model.pt
  backend/models/cnn_lstm_training_meta.json
"""

import json
import logging
import os
from pathlib import Path
from collections import Counter
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import classification_report

from services.translator_model import normalize_landmarks

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
WORD_DATASET_DIR = SCRIPT_DIR / 'dataset_words'
MODEL_DIR = SCRIPT_DIR / 'models'
MODEL_PATH = MODEL_DIR / 'isl_cnn_lstm_word_model.pt'
META_PATH = MODEL_DIR / 'cnn_lstm_training_meta.json'

NUM_FEATURES = 126
SEQUENCE_LENGTH = 30


# ═══════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════

class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ═══════════════════════════════════════════════════════════════════════════
# Model Architecture: 1D-CNN + Bi-LSTM Hybrid
# ═══════════════════════════════════════════════════════════════════════════

class CNNBiLSTMWordClassifier(nn.Module):
    """
    Hybrid 1D-CNN + Bidirectional LSTM for ISL word recognition.

    The CNN block processes each frame's 126 landmark features to extract
    local finger-joint geometry patterns (knuckle→tip relationships).
    The BiLSTM block then processes the CNN-extracted features across
    the temporal dimension to capture motion trajectories.
    """

    def __init__(self, input_dim=126, cnn_channels=128, hidden_dim=128,
                 num_classes=17, num_layers=2, dropout=0.3):
        super().__init__()

        # 1D-CNN Block: extracts local spatial features per frame
        # Conv1d expects (batch, channels, seq_len) — we treat features as channels
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Bi-LSTM Block: captures temporal dynamics across frames
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Classifier Head
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x shape: (batch, seq_len=30, features=126)

        # Transpose for Conv1d: (batch, features, seq_len)
        x_cnn = x.permute(0, 2, 1)

        # CNN extracts local geometry: (batch, cnn_channels, seq_len)
        x_cnn = self.cnn(x_cnn)

        # Transpose back for LSTM: (batch, seq_len, cnn_channels)
        x_lstm_in = x_cnn.permute(0, 2, 1)

        # BiLSTM captures temporal trajectory
        lstm_out, _ = self.lstm(x_lstm_in)

        # Temporal mean pooling over all frames
        pooled = torch.mean(lstm_out, dim=1)  # (batch, hidden*2)

        # Classify
        logits = self.classifier(pooled)
        return logits


# ═══════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════

def load_word_sequences(data_dir=WORD_DATASET_DIR):
    """Loads all 30-frame sequence JSONs from dataset_words."""
    if not data_dir.exists():
        raise RuntimeError(f"Word dataset directory {data_dir} does not exist!")

    word_labels = sorted([
        d.name for d in data_dir.iterdir()
        if d.is_dir() and not d.name.startswith('.')
    ])
    if not word_labels:
        raise RuntimeError(f"No word subfolders found in {data_dir}!")

    logger.info(f"Found {len(word_labels)} word classes: {word_labels}")
    label_to_idx = {l: i for i, l in enumerate(word_labels)}

    sequences, labels, sources = [], [], []

    for word_dir in sorted(data_dir.iterdir()):
        if not word_dir.is_dir() or word_dir.name.startswith('.'):
            continue
        word = word_dir.name.upper()

        for json_path in sorted(word_dir.glob('*.json')):
            try:
                data = json.loads(json_path.read_text(encoding='utf-8'))
                for seq in data.get('frame_sequences', []):
                    arr = np.array(seq, dtype=np.float32)
                    if arr.shape != (SEQUENCE_LENGTH, NUM_FEATURES):
                        continue

                    # Normalize each frame
                    norm_seq = []
                    for frame in arr:
                        norm = normalize_landmarks(frame)
                        norm_seq.append(
                            norm if norm is not None
                            else np.zeros(NUM_FEATURES, dtype=np.float32)
                        )

                    sequences.append(np.array(norm_seq, dtype=np.float32))
                    labels.append(label_to_idx[word])
                    sources.append(json_path.stem)
            except Exception as e:
                logger.warning(f"Error loading {json_path}: {e}")

    if not sequences:
        raise RuntimeError(f"No valid sequences found in {data_dir}!")

    return (
        np.array(sequences, dtype=np.float32),
        np.array(labels, dtype=np.int64),
        word_labels,
        sources,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Data Augmentation
# ═══════════════════════════════════════════════════════════════════════════

def augment_sequences(X, y, copies=10):
    """
    Multi-strategy temporal augmentation:
      1. Gaussian noise on landmarks
      2. Random frame dropout (replace random frames with zeros)
      3. Time scale jitter (resample with slight stretch/compress)
    """
    rng = np.random.default_rng(42)
    aug_X, aug_y = [X], [y]

    for c in range(copies):
        batch = X.copy()

        # Strategy 1: Gaussian noise (always applied)
        noise_scale = 0.003 + (c * 0.001)  # Gradually increase noise
        noise = rng.normal(0, noise_scale, size=batch.shape).astype(np.float32)
        batch = batch + noise

        # Strategy 2: Random frame dropout (30% of copies)
        if c % 3 == 0:
            for i in range(len(batch)):
                n_drop = rng.integers(1, 4)  # Drop 1-3 frames
                drop_idx = rng.choice(SEQUENCE_LENGTH, size=n_drop, replace=False)
                batch[i, drop_idx] = 0.0

        # Strategy 3: Slight time warp (shift frames by ±1)
        if c % 4 == 0:
            for i in range(len(batch)):
                shift = rng.integers(-2, 3)  # -2 to +2 frame shift
                if shift != 0:
                    batch[i] = np.roll(batch[i], shift, axis=0)

        aug_X.append(batch)
        aug_y.append(y)

    return np.concatenate(aug_X, axis=0), np.concatenate(aug_y, axis=0)


# ═══════════════════════════════════════════════════════════════════════════
# Source-Disjoint Validation Split
# ═══════════════════════════════════════════════════════════════════════════

def split_by_source(sources, labels, word_labels, val_fraction=0.2, seed=42):
    """
    Split data by source file to avoid data leakage.
    Ensures augmented copies of the same video don't appear in both splits.
    """
    rng = np.random.default_rng(seed)
    unique_sources = list(set(sources))
    rng.shuffle(unique_sources)

    n_val = max(1, round(len(unique_sources) * val_fraction))
    val_sources = set(unique_sources[:n_val])

    train_idx = [i for i, s in enumerate(sources) if s not in val_sources]
    val_idx = [i for i, s in enumerate(sources) if s in val_sources]

    if not train_idx or not val_idx:
        # Fallback to random split
        all_idx = list(range(len(sources)))
        rng.shuffle(all_idx)
        n_val = max(1, round(len(all_idx) * val_fraction))
        val_idx = all_idx[:n_val]
        train_idx = all_idx[n_val:]

    return train_idx, val_idx


# ═══════════════════════════════════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════════════════════════════════

def train():
    # Load data
    X, y, word_labels, sources = load_word_sequences()
    logger.info(f"Loaded {len(X)} raw sequences across {len(word_labels)} classes.")
    logger.info(f"Class distribution: {dict(Counter(y))}")

    # Source-disjoint split BEFORE augmentation
    train_idx, val_idx = split_by_source(sources, y, word_labels)

    X_train_raw, y_train_raw = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # Augment only training data
    X_train, y_train = augment_sequences(X_train_raw, y_train_raw, copies=15)
    logger.info(f"Training: {len(X_train)} (augmented), Validation: {len(X_val)} (raw)")

    # Create data loaders
    train_dataset = SequenceDataset(X_train, y_train)
    val_dataset = SequenceDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Initialize model
    model = CNNBiLSTMWordClassifier(num_classes=len(word_labels))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    # Training loop with early stopping
    best_acc = 0.0
    best_state = None
    patience_counter = 0
    max_patience = 10
    epochs = 60

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * len(batch_y)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_y).sum().item()
            total += len(batch_y)

        train_acc = correct / total

        # Validate
        model.eval()
        val_correct, val_total = 0, 0
        val_preds_all, val_labels_all = [], []
        with torch.no_grad():
            for val_X, val_y in val_loader:
                v_logits = model(val_X)
                v_preds = torch.argmax(v_logits, dim=1)
                val_correct += (v_preds == val_y).sum().item()
                val_total += len(val_y)
                val_preds_all.extend(v_preds.numpy())
                val_labels_all.extend(val_y.numpy())

        val_acc = val_correct / val_total if val_total > 0 else 0.0
        scheduler.step(val_acc)

        # Early stopping check
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == epochs or patience_counter == 0:
            logger.info(
                f"Epoch {epoch:02d}/{epochs} - Loss: {total_loss/total:.4f} | "
                f"Train: {train_acc*100:.1f}% | Val: {val_acc*100:.1f}% | "
                f"Best: {best_acc*100:.1f}% | Patience: {patience_counter}/{max_patience}"
            )

        if patience_counter >= max_patience:
            logger.info(f"Early stopping at epoch {epoch} (no improvement for {max_patience} epochs)")
            break

    # Load best model state
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation
    model.eval()
    val_preds_all, val_labels_all = [], []
    with torch.no_grad():
        for val_X, val_y in val_loader:
            v_logits = model(val_X)
            v_preds = torch.argmax(v_logits, dim=1)
            val_preds_all.extend(v_preds.numpy())
            val_labels_all.extend(val_y.numpy())

    report = classification_report(
        val_labels_all, val_preds_all,
        labels=range(len(word_labels)),
        target_names=word_labels,
        output_dict=True,
        zero_division=0,
    )

    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_type': 'cnn_bilstm',
        'model_state_dict': model.state_dict(),
        'input_dim': NUM_FEATURES,
        'cnn_channels': 128,
        'hidden_dim': 128,
        'sequence_length': SEQUENCE_LENGTH,
        'num_classes': len(word_labels),
        'word_labels': word_labels,
    }, str(MODEL_PATH))

    # Save metadata
    meta = {
        'model_type': 'cnn_bilstm',
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'val_accuracy': best_acc,
        'num_classes': len(word_labels),
        'labels': word_labels,
        'sequence_length': SEQUENCE_LENGTH,
        'input_features': NUM_FEATURES,
        'epochs_trained': epoch,
        'architecture': '1D-CNN(126→64→128) + BiLSTM(128, 2-layers) + FC(256→64→N)',
        'per_class': {
            label: {
                'precision': report[label]['precision'],
                'recall': report[label]['recall'],
                'f1-score': report[label]['f1-score'],
            }
            for label in word_labels
            if label in report
        },
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding='utf-8')

    logger.info('=' * 60)
    logger.info(f'CNN-BiLSTM Validation Accuracy: {best_acc*100:.2f}%')
    logger.info('=' * 60)
    for label in word_labels:
        if label in report:
            logger.info(
                f"  {label:15s}  P={report[label]['precision']:.3f}  "
                f"R={report[label]['recall']:.3f}  F1={report[label]['f1-score']:.3f}"
            )
    logger.info(f'Model saved to: {MODEL_PATH}')
    logger.info(f'Metadata saved to: {META_PATH}')


if __name__ == '__main__':
    train()
