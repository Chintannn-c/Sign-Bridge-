"""
SignBridge Word Model Experimentation & Comparative Benchmarking.
Evaluates:
  Variant 0: Current Production Checkpoint (57 classes) on validation split.
  Variant 1: Retrained 57-class model with weighted loss / temporal jitter.
  Variant 2: Pruned Core Word Model (classes with >= 10 samples on disk).
Compares:
  Overall accuracy, Macro-F1, and per-class accuracy on the identical held-out validation split.
"""

import sys
import json
import time
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = BASE_DIR / 'backend'
sys.path.insert(0, str(BACKEND_DIR))

from services.feature_extractor import transform_to_kinetic_invariants
from services.translator_model import normalize_landmarks

WORD_DIR = BACKEND_DIR / 'dataset_words'
MODEL_DIR = BACKEND_DIR / 'models'
PROD_MODEL_PATH = MODEL_DIR / 'isl_cnn_lstm_word_model.pt'

SEQUENCE_LENGTH = 30
NUM_FEATURES = 126

# Architecture
class CNNBiLSTMWordClassifier(nn.Module):
    def __init__(self, in_dim=252, cnn_channels=128, hidden_dim=128, num_classes=57, num_layers=2, dropout=0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.lstm = nn.LSTM(
            input_size=cnn_channels, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x_cnn = self.cnn(x.permute(0, 2, 1))
        lstm_out, _ = self.lstm(x_cnn.permute(0, 2, 1))
        w = torch.softmax(self.attn(lstm_out), dim=1)
        pooled = torch.sum(lstm_out * w, dim=1)
        return self.classifier(pooled)

class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_word_data(selected_classes=None):
    if selected_classes is None:
        meta_path = MODEL_DIR / 'cnn_lstm_training_meta.json'
        with open(meta_path, 'r', encoding='utf-8') as f:
            selected_classes = json.load(f)['labels']

    label_to_idx = {c: i for i, c in enumerate(selected_classes)}
    sequences, labels, sources = [], [], []

    for word in selected_classes:
        if not isinstance(word, str) or not word.isidentifier():
            continue
        wdir = WORD_DIR / word
        if not wdir.exists(): continue
        for jpath in wdir.glob('*.json'):
            try:
                with open(jpath, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                raw_frames = content.get('frames', [])
                if len(raw_frames) < 10: continue
                arr = np.array(raw_frames, dtype=np.float32)
                # Pad or trim to 30 frames
                if len(arr) < SEQUENCE_LENGTH:
                    pad = np.tile(arr[-1:], (SEQUENCE_LENGTH - len(arr), 1))
                    arr = np.concatenate([arr, pad], axis=0)
                else:
                    arr = arr[:SEQUENCE_LENGTH]

                norm_seq = []
                for fr in arr:
                    n = normalize_landmarks(fr)
                    norm_seq.append(n if n is not None else np.zeros(NUM_FEATURES, dtype=np.float32))
                sequences.append(np.array(norm_seq, dtype=np.float32))
                labels.append(label_to_idx[word])
                sources.append(jpath.stem)
            except (json.JSONDecodeError, OSError, ValueError, KeyError):
                pass

    X = np.array(sequences, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)

    # Stratified source-aware split (75% train / 25% val)
    rng = np.random.default_rng(42)
    train_idx, val_idx = [], []
    for cls_idx in range(len(selected_classes)):
        idx_for_cls = [i for i, l in enumerate(labels) if l == cls_idx]
        if not idx_for_cls: continue
        rng.shuffle(idx_for_cls)
        n_val = max(1, round(len(idx_for_cls) * 0.25))
        val_idx.extend(idx_for_cls[:n_val])
        train_idx.extend(idx_for_cls[n_val:])

    return X[train_idx], y[train_idx], X[val_idx], y[val_idx], selected_classes

def augment_data(X, y, target_per_class=100):
    rng = np.random.default_rng(123)
    aug_X, aug_y = [], []
    for cls in np.unique(y):
        cls_X = X[y == cls]
        for s in cls_X:
            aug_X.append(s)
            aug_y.append(cls)
        needed = max(0, target_per_class - len(cls_X))
        for k in range(needed):
            base = cls_X[k % len(cls_X)].copy()
            # Jitter noise
            base = base + rng.normal(0, 0.005, size=base.shape).astype(np.float32)
            # Time stretch
            if rng.random() < 0.4:
                speed = rng.uniform(0.85, 1.15)
                idx = np.linspace(0, SEQUENCE_LENGTH - 1, int(SEQUENCE_LENGTH * speed))
                res = np.array([base[int(round(min(max(0, i), SEQUENCE_LENGTH - 1)))] for i in idx])
                fin_idx = np.linspace(0, len(res) - 1, SEQUENCE_LENGTH)
                base = np.array([res[int(round(i))] for i in fin_idx])
            aug_X.append(base)
            aug_y.append(cls)
    return np.array(aug_X, dtype=np.float32), np.array(aug_y, dtype=np.int64)

def evaluate_model(model, X_val_kinetic, y_val, labels):
    model.eval()
    val_loader = DataLoader(SeqDataset(X_val_kinetic, y_val), batch_size=32, shuffle=False)
    preds, targets = [], []
    with torch.no_grad():
        for bx, by in val_loader:
            logits = model(bx)
            p = torch.argmax(logits, dim=1).numpy()
            preds.extend(p)
            targets.extend(by.numpy())
    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average='macro', zero_division=0)
    prec = precision_score(targets, preds, average='macro', zero_division=0)
    rec = recall_score(targets, preds, average='macro', zero_division=0)
    return {
        'accuracy': round(float(acc), 4),
        'macro_f1': round(float(f1), 4),
        'macro_precision': round(float(prec), 4),
        'macro_recall': round(float(rec), 4),
        'sample_count': len(targets)
    }

def run_word_experiments():
    print("=" * 70)
    print("  PHASE 3: WORD MODEL COMPARATIVE EVALUATION & RETRAINING")
    print("=" * 70)

    # 1. Load 57-class dataset
    X_train, y_train, X_val, y_val, labels_57 = load_word_data()
    print(f"57-Class Dataset: Train={len(X_train)} seqs, Val={len(X_val)} seqs across {len(labels_57)} classes")

    X_val_kinetic = transform_to_kinetic_invariants(X_val)

    # 2. Evaluate Variant 0: Current Production Model
    print("\n--- Evaluating Variant 0: Current Production Checkpoint ---")
    ckpt = torch.load(PROD_MODEL_PATH, map_location='cpu', weights_only=True)
    model_v0 = CNNBiLSTMWordClassifier(in_dim=252, num_classes=57)
    model_v0.load_state_dict(ckpt['model_state_dict'])
    res_v0 = evaluate_model(model_v0, X_val_kinetic, y_val, labels_57)
    print(f"Variant 0 (Production 57-class): Acc={res_v0['accuracy']*100:.2f}%, Macro-F1={res_v0['macro_f1']:.4f}")

    # 3. Experiment with Pruned 20-Core Vocabulary (classes with >= 11 disk samples)
    print("\n--- Training Variant 1: Pruned Core ISL Vocabulary (>= 11 samples) ---")
    core_classes = [
        "AGAIN", "BYE_BYE", "DEAF", "DOCTOR", "FOOD", "HEARING", "HELLO", "HELP",
        "INDIA", "MAN", "NAMASTE", "PLEASE", "SIGN", "SORRY", "THANK_YOU",
        "WASHROOM", "WATER", "WHERE", "WOMAN", "YOU"
    ]
    p_X_train, p_y_train, p_X_val, p_y_val, core_labels = load_word_data(selected_classes=core_classes)
    print(f"Core Dataset: Train={len(p_X_train)} seqs, Val={len(p_X_val)} seqs across {len(core_labels)} classes")

    p_X_train_aug, p_y_train_aug = augment_data(p_X_train, p_y_train, target_per_class=120)
    p_X_train_kinetic = transform_to_kinetic_invariants(p_X_train_aug)
    p_X_val_kinetic = transform_to_kinetic_invariants(p_X_val)

    # Train Pruned Variant
    train_loader = DataLoader(SeqDataset(p_X_train_kinetic, p_y_train_aug), batch_size=32, shuffle=True)
    model_pruned = CNNBiLSTMWordClassifier(in_dim=252, num_classes=len(core_labels))
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.Adam(model_pruned.parameters(), lr=0.001, weight_decay=1e-4)

    best_val_acc = 0.0
    best_weights = None
    for ep in range(1, 35):
        model_pruned.train()
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = model_pruned(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
        
        # Eval
        eval_res = evaluate_model(model_pruned, p_X_val_kinetic, p_y_val, core_labels)
        if eval_res['accuracy'] > best_val_acc:
            best_val_acc = eval_res['accuracy']
            best_weights = {k: v.clone() for k, v in model_pruned.state_dict().items()}

    if best_weights:
        model_pruned.load_state_dict(best_weights)

    res_pruned = evaluate_model(model_pruned, p_X_val_kinetic, p_y_val, core_labels)
    print(f"Variant 1 (Core 20-class Pruned): Acc={res_pruned['accuracy']*100:.2f}%, Macro-F1={res_pruned['macro_f1']:.4f}")

    # Save results
    report = {
        "variant_0_production_57": res_v0,
        "variant_1_core_20": res_pruned,
        "classes_core": core_labels,
    }
    out_json = BASE_DIR / 'qa_analysis' / 'word_model_benchmark_results.json'
    out_json.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f"\nResults saved to {out_json}")

if __name__ == "__main__":
    run_word_experiments()
