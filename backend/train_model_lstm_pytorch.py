"""
SignBridge — Bi-LSTM Temporal Word Classifier Trainer (PyTorch)
Trains a 2-layer Bidirectional LSTM classifier on 30-frame MediaPipe landmark sequences
extracted from ISL gesture videos.

Input:  (batch_size, 30, 126) — 30 frames x 126 landmark features
Output: (batch_size, num_word_classes)

Outputs:
  backend/models/isl_lstm_word_model.pt
  backend/models/lstm_training_meta.json
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
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report

from services.translator_model import normalize_landmarks

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
WORD_DATASET_DIR = SCRIPT_DIR / 'dataset_words'
MODEL_DIR = SCRIPT_DIR / 'models'
MODEL_PATH = MODEL_DIR / 'isl_lstm_word_model.pt'
META_PATH = MODEL_DIR / 'lstm_training_meta.json'

NUM_FEATURES = 126
SEQUENCE_LENGTH = 30

class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class BiLSTMWordClassifier(nn.Module):
    def __init__(self, input_dim=NUM_FEATURES, hidden_dim=128, num_classes=17, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.batch_norm = nn.BatchNorm1d(hidden_dim * 2)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        lstm_out, (hn, cn) = self.lstm(x)
        # Take mean over time dimension for robust temporal pooling
        pooled = torch.mean(lstm_out, dim=1)
        normed = self.batch_norm(pooled)
        out = self.fc1(normed)
        out = self.relu(out)
        out = self.dropout(out)
        logits = self.fc2(out)
        return logits

def load_word_sequences(data_dir=WORD_DATASET_DIR):
    """Loads all 30-frame sequence JSONs from dataset_words."""
    if not data_dir.exists():
        raise RuntimeError(f"Word dataset directory {data_dir} does not exist!")
        
    word_labels = sorted([d.name for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith('.')])
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
                        norm_seq.append(norm if norm is not None else np.zeros(NUM_FEATURES, dtype=np.float32))
                        
                    sequences.append(np.array(norm_seq, dtype=np.float32))
                    labels.append(label_to_idx[word])
                    sources.append(json_path.stem)
            except Exception as e:
                logger.warning(f"Error loading {json_path}: {e}")
                
    if not sequences:
        raise RuntimeError(f"No valid sequences found in {data_dir}!")
        
    return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.int64), word_labels

def augment_sequences(X, y, copies=10):
    """Adds slight temporal jitter and gaussian noise to augment sequence data."""
    rng = np.random.default_rng(42)
    aug_X, aug_y = [X], [y]
    
    for _ in range(copies):
        noise = rng.normal(0, 0.005, size=X.shape).astype(np.float32)
        # Apply time scale / warp
        noisy_seqs = X + noise
        aug_X.append(noisy_seqs)
        aug_y.append(y)
        
    return np.concatenate(aug_X, axis=0), np.concatenate(aug_y, axis=0)

def train():
    X, y, word_labels = load_word_sequences()
    logger.info(f"Loaded raw sequences: {len(X)} across {len(word_labels)} classes.")
    
    # Augment data for robust LSTM convergence
    X_aug, y_aug = augment_sequences(X, y, copies=15)
    logger.info(f"Augmented sequences for training: {len(X_aug)}")
    
    dataset = SequenceDataset(X_aug, y_aug)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False)
    
    model = BiLSTMWordClassifier(num_classes=len(word_labels))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    best_acc = 0.0
    best_state = None
    epochs = 40
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(batch_y)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_y).sum().item()
            total += len(batch_y)
            
        train_acc = correct / total
        
        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for val_X, val_y in val_loader:
                v_logits = model(val_X)
                v_preds = torch.argmax(v_logits, dim=1)
                val_correct += (v_preds == val_y).sum().item()
                val_total += len(val_y)
                
        val_acc = val_correct / val_total
        scheduler.step(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict().copy()
            
        if epoch % 5 == 0 or epoch == epochs:
            logger.info(f"Epoch {epoch:02d}/{epochs:02d} - Loss: {total_loss/total:.4f} | Train Acc: {train_acc*100:.1f}% | Val Acc: {val_acc*100:.1f}%")
            
    # Save best model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if best_state is not None:
        model.load_state_dict(best_state)
        
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': NUM_FEATURES,
        'sequence_length': SEQUENCE_LENGTH,
        'num_classes': len(word_labels),
        'word_labels': word_labels
    }, str(MODEL_PATH))
    
    # Save metadata
    meta = {
        'model_type': 'bi_lstm_pytorch',
        'train_samples': len(train_set),
        'val_samples': len(val_set),
        'val_accuracy': best_acc,
        'num_classes': len(word_labels),
        'labels': word_labels,
        'sequence_length': SEQUENCE_LENGTH,
        'input_features': NUM_FEATURES,
        'epochs_trained': epochs
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    logger.info(f"=== Model training complete! Best Validation Accuracy: {best_acc*100:.2f}% ===")
    logger.info(f"Model saved -> {MODEL_PATH}")
    logger.info(f"Metadata saved -> {META_PATH}")

if __name__ == '__main__':
    train()
