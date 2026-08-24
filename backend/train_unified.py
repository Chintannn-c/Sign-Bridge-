"""
SignBridge — Unified Training Pipeline (v2)

One script to rule them all:
  1. Extract landmarks from ALL image datasets (RealSign, Self-Made, Mendeley)
  2. Extract + augment video landmarks for word recognition
  3. Train XGBoost letter classifier with all combined data
  4. Train CNN-BiLSTM word classifier with augmented video data
  5. Report comprehensive metrics and save all models

Usage:
  python train_unified.py                 # Full pipeline (extract + train all)
  python train_unified.py --skip-extract  # Train only (assumes data already extracted)
  python train_unified.py --letters-only  # Train only letter models
  python train_unified.py --words-only    # Train only word models
"""

import argparse
import json
import logging
import pickle
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR / 'models'


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1: Data Extraction
# ═══════════════════════════════════════════════════════════════════════════

def run_extraction():
    """Run both static and video landmark extraction."""
    logger.info("=" * 70)
    logger.info("PHASE 1: Extracting landmarks from ALL datasets")
    logger.info("=" * 70)

    # Static images (letters + digits)
    logger.info("\n--- Extracting static image landmarks ---")
    try:
        import extract_static_landmarks
        extract_static_landmarks.main()
    except Exception as e:
        logger.error(f"Static extraction failed: {e}")
        logger.info("Continuing with existing data...")

    # Video sequences (words)
    logger.info("\n--- Extracting video landmarks with augmentation ---")
    try:
        import extract_video_landmarks
        extract_video_landmarks.main()
    except Exception as e:
        logger.error(f"Video extraction failed: {e}")
        logger.info("Continuing with existing data...")


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2: Letter Model Training (XGBoost)
# ═══════════════════════════════════════════════════════════════════════════

def augment_landmarks(X, copies=2, seed=42):
    """Augment raw landmark coordinates with rotation + noise."""
    rng = np.random.default_rng(seed)
    augmented = [X]
    for _ in range(copies):
        points = X.reshape(-1, 42, 3).copy()
        angles = rng.uniform(-0.18, 0.18, size=len(points))
        cos, sin = np.cos(angles), np.sin(angles)
        x, y = points[:, :, 0].copy(), points[:, :, 1].copy()
        points[:, :, 0] = x * cos[:, None] - y * sin[:, None]
        points[:, :, 1] = x * sin[:, None] + y * cos[:, None]
        points += rng.normal(0, 0.008, size=points.shape).astype(np.float32)
        augmented.append(points.reshape(-1, 126))
    return np.concatenate(augmented, axis=0)


def augment_training_partition(X, y, seed=42):
    """Full augmentation for training: jitter, swap, inter-hand noise."""
    rng = np.random.default_rng(seed)
    X_base = augment_landmarks(X, copies=2, seed=seed)
    y_base = np.tile(y, 3)

    # Hand slot swap for single-hand signs
    is_single = (np.any(X[:, :63] != 0, axis=1) & ~np.any(X[:, 63:] != 0, axis=1)) | \
                (~np.any(X[:, :63] != 0, axis=1) & np.any(X[:, 63:] != 0, axis=1))
    single_X = X[is_single].copy()
    single_y = y[is_single].copy()
    swapped_single = np.zeros_like(single_X)
    swapped_single[:, :63] = single_X[:, 63:]
    swapped_single[:, 63:] = single_X[:, :63]

    # Dual-hand jitter
    is_dual = np.any(X[:, :63] != 0, axis=1) & np.any(X[:, 63:] != 0, axis=1)
    dual_X = X[is_dual].copy()
    dual_y = y[is_dual].copy()
    jittered_dual = dual_X.copy()
    shift_h1 = rng.normal(0, 0.015, size=(len(dual_X), 1, 3)).astype(np.float32)
    shift_h2 = rng.normal(0, 0.015, size=(len(dual_X), 1, 3)).astype(np.float32)
    pts = jittered_dual.reshape(-1, 2, 21, 3)
    pts[:, 0] += shift_h1
    pts[:, 1] += shift_h2
    jittered_dual = pts.reshape(-1, 126)

    X_aug = np.concatenate([X_base, swapped_single, jittered_dual], axis=0)
    y_aug = np.concatenate([y_base, single_y, dual_y], axis=0)
    return X_aug, y_aug


def train_letter_model():
    """Train XGBoost letter classifier with all combined data."""
    from services.data_loader import load_dataset_partitioned, ALPHABET_LABELS
    from services.feature_extractor import extract_features, NUM_EXTRACTED_FEATURES
    from xgboost import XGBClassifier

    logger.info("=" * 70)
    logger.info("PHASE 2: Training XGBoost Letter Classifier")
    logger.info("=" * 70)

    # Load data from all sources
    logger.info("Loading ALL partitioned data...")
    partitions, report = load_dataset_partitioned()

    logger.info(f"Dataset: Train={report['train_samples']}, Val={report['val_samples']}, Test={report['test_samples']}")
    logger.info(f"Sources: {report['counts_by_source']}")

    if report['train_samples'] == 0:
        logger.error("No training data found! Run extraction first.")
        return None

    # Augment training partition
    logger.info("Augmenting training partition...")
    X_train_aug, y_train_aug = augment_training_partition(partitions['X_train'], partitions['y_train'])
    logger.info(f"Augmented training: {X_train_aug.shape[0]} samples")

    # Extract geometric features
    logger.info(f"Extracting {NUM_EXTRACTED_FEATURES}-D geometric features...")
    X_train_feat = extract_features(X_train_aug)
    X_val_feat = extract_features(partitions['X_val'])
    X_test_feat = extract_features(partitions['X_test'])

    logger.info(f"Feature matrices: Train={X_train_feat.shape}, Val={X_val_feat.shape}, Test={X_test_feat.shape}")

    # Train XGBoost
    logger.info("Training XGBoost classifier...")
    model = XGBClassifier(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        gamma=0.05,
        reg_alpha=0.20,
        reg_lambda=1.80,
        objective='multi:softprob',
        num_class=len(ALPHABET_LABELS),
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    t0 = time.time()
    model.fit(
        X_train_feat,
        y_train_aug,
        eval_set=[(X_val_feat, partitions['y_val'])],
        verbose=False,
    )
    train_time = time.time() - t0
    logger.info(f"Training completed in {train_time:.2f}s")

    # Evaluate
    val_preds = np.argmax(model.predict_proba(X_val_feat), axis=1)
    val_acc = accuracy_score(partitions['y_val'], val_preds)
    val_f1 = f1_score(partitions['y_val'], val_preds, average='macro')

    test_preds = np.argmax(model.predict_proba(X_test_feat), axis=1)
    test_acc = accuracy_score(partitions['y_test'], test_preds)
    test_f1 = f1_score(partitions['y_test'], test_preds, average='macro')
    test_prec = precision_score(partitions['y_test'], test_preds, average='macro')
    test_rec = recall_score(partitions['y_test'], test_preds, average='macro')

    logger.info("=" * 60)
    logger.info(f"Val Accuracy: {val_acc*100:.2f}% | Val Macro F1: {val_f1:.4f}")
    logger.info(f"Test Accuracy: {test_acc*100:.2f}% | Test Macro F1: {test_f1:.4f}")
    logger.info(f"Test Precision: {test_prec:.4f} | Test Recall: {test_rec:.4f}")
    logger.info("=" * 60)

    # Per-class metrics
    raw_report = classification_report(
        partitions['y_test'], test_preds,
        target_names=ALPHABET_LABELS,
        output_dict=True,
        zero_division=0,
    )
    report_dict: dict = raw_report if isinstance(raw_report, dict) else {}

    per_class = {}
    zero_classes = []
    for letter in ALPHABET_LABELS:
        stats = report_dict.get(letter)
        if isinstance(stats, dict):
            p_val = float(stats.get('precision', 0.0))
            r_val = float(stats.get('recall', 0.0))
            f1_val = float(stats.get('f1-score', 0.0))
            sup_val = int(stats.get('support', 0))
            per_class[letter] = {
                'precision': round(p_val, 4),
                'recall': round(r_val, 4),
                'f1-score': round(f1_val, 4),
                'test_samples': sup_val
            }
            if f1_val == 0.0:
                zero_classes.append(letter)
            logger.info(f"  {letter} -> P:{p_val:.3f} R:{r_val:.3f} F1:{f1_val:.3f}")

    if zero_classes:
        logger.warning(f"[!] Zero-F1 classes: {', '.join(zero_classes)}")

    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / 'isl_xgboost_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    metadata = {
        'model_type': 'xgboost',
        'raw_input_landmarks': 126,
        'estimator_features': NUM_EXTRACTED_FEATURES,
        'feature_name': 'geometric_invariants_176d',
        'num_classes': len(ALPHABET_LABELS),
        'labels': ALPHABET_LABELS,
        'data_summary': {
            'train_samples': int(report['train_samples']),
            'train_augmented': len(X_train_aug),
            'val_samples': int(report['val_samples']),
            'test_samples': int(report['test_samples']),
            'duplicate_frames_excluded': int(report['duplicate_frames_skipped']),
            'sources': report['counts_by_source'],
        },
        'metrics': {
            'val_accuracy': round(float(val_acc), 4),
            'val_macro_f1': round(float(val_f1), 4),
            'test_accuracy': round(float(test_acc), 4),
            'test_macro_f1': round(float(test_f1), 4),
            'test_precision': round(float(test_prec), 4),
            'test_recall': round(float(test_rec), 4),
        },
        'per_class': per_class,
        'trained_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'training_time_seconds': round(train_time, 2),
    }

    meta_path = MODEL_DIR / 'xgb_training_meta.json'
    meta_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    logger.info(f"Model saved to: {model_path}")
    logger.info(f"Metadata saved to: {meta_path}")

    return metadata


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2B: Letter Model Training (ST-GCN Kinematic Graph)
# ═══════════════════════════════════════════════════════════════════════════

def train_stgcn_model(epochs=35, batch_size=64, lr=0.003):
    """Train PyTorch ST-GCN kinematic hand graph classifier."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    from models.st_gcn import STGCNHandClassifier
    from services.data_loader import load_dataset_partitioned, ALPHABET_LABELS

    logger.info("=" * 70)
    logger.info("PHASE 2B: Training ST-GCN Hand Graph Classifier")
    logger.info("=" * 70)

    partitions, report = load_dataset_partitioned()
    if report['train_samples'] == 0:
        logger.error("No training data found! Run extraction first.")
        return None

    X_train_aug, y_train_aug = augment_training_partition(partitions['X_train'], partitions['y_train'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training ST-GCN on device: {device} ({len(X_train_aug)} training samples)")

    train_ds = TensorDataset(torch.tensor(X_train_aug, dtype=torch.float32), torch.tensor(y_train_aug, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(partitions['X_val'], dtype=torch.float32), torch.tensor(partitions['y_val'], dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(partitions['X_test'], dtype=torch.float32), torch.tensor(partitions['y_test'], dtype=torch.long))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = STGCNHandClassifier(num_classes=len(ALPHABET_LABELS), in_channels=3, hidden_dim=64).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_acc = 0.0
    best_state = None
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(X_b)
            loss = criterion(logits, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            total_loss += loss.item() * len(y_b)
            preds = logits.argmax(dim=1)
            correct += (preds == y_b).sum().item()
            total += len(y_b)

        scheduler.step()
        train_acc = correct / max(1, total)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                logits = model(X_b)
                preds = logits.argmax(dim=1)
                val_correct += (preds == y_b).sum().item()
                val_total += len(y_b)

        val_acc = val_correct / max(1, val_total)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == epochs:
            logger.info(f"Epoch {epoch:02d}/{epochs:02d} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}% (Best: {best_val_acc*100:.2f}%)")

    train_time = time.time() - t0

    # Load best weights
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # Final Test evaluation
    test_preds_list = []
    test_targets_list = []
    with torch.no_grad():
        for X_b, y_b in test_loader:
            logits = model(X_b.to(device))
            preds = logits.argmax(dim=1).cpu().numpy()
            test_preds_list.extend(preds)
            test_targets_list.extend(y_b.numpy())

    test_preds = np.array(test_preds_list)
    test_targets = np.array(test_targets_list)

    test_acc = accuracy_score(test_targets, test_preds)
    test_f1 = f1_score(test_targets, test_preds, average='macro')
    test_prec = precision_score(test_targets, test_preds, average='macro', zero_division=0)
    test_rec = recall_score(test_targets, test_preds, average='macro', zero_division=0)

    logger.info("=" * 60)
    logger.info(f"ST-GCN Final Test Accuracy: {test_acc*100:.2f}% | Test Macro F1: {test_f1:.4f}")
    logger.info(f"ST-GCN Test Precision: {test_prec:.4f} | Test Recall: {test_rec:.4f}")
    logger.info("=" * 60)

    # Save ST-GCN model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / 'isl_stgcn_model.pt'
    torch.save({
        'model_type': 'st_gcn',
        'model_state_dict': model.state_dict(),
        'num_classes': len(ALPHABET_LABELS),
        'labels': ALPHABET_LABELS,
    }, model_path)

    metadata = {
        'model_type': 'st_gcn',
        'raw_input_landmarks': 126,
        'num_classes': len(ALPHABET_LABELS),
        'labels': ALPHABET_LABELS,
        'metrics': {
            'val_accuracy': round(float(best_val_acc), 4),
            'test_accuracy': round(float(test_acc), 4),
            'test_macro_f1': round(float(test_f1), 4),
            'test_precision': round(float(test_prec), 4),
            'test_recall': round(float(test_rec), 4),
        },
        'trained_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'training_time_seconds': round(train_time, 2),
    }

    meta_path = MODEL_DIR / 'stgcn_training_meta.json'
    meta_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    logger.info(f"ST-GCN Model saved to: {model_path}")
    logger.info(f"ST-GCN Metadata saved to: {meta_path}")

    return metadata


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 3: Word Model Training (CNN-BiLSTM)
# ═══════════════════════════════════════════════════════════════════════════

def train_word_model():
    """Train CNN-BiLSTM word classifier with augmented video data."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader

    from services.translator_model import normalize_landmarks

    WORD_DATASET_DIR = SCRIPT_DIR / 'dataset_words'
    NUM_FEATURES = 126
    SEQUENCE_LENGTH = 30

    logger.info("=" * 70)
    logger.info("PHASE 3: Training CNN-BiLSTM Word Classifier")
    logger.info("=" * 70)

    # --- Load word sequences ---
    if not WORD_DATASET_DIR.exists():
        logger.error(f"Word dataset not found: {WORD_DATASET_DIR}")
        return None

    word_labels = sorted([
        d.name for d in WORD_DATASET_DIR.iterdir()
        if d.is_dir() and not d.name.startswith('.')
    ])
    if not word_labels:
        logger.error("No word classes found!")
        return None

    logger.info(f"Found {len(word_labels)} word classes: {word_labels}")
    label_to_idx = {l: i for i, l in enumerate(word_labels)}

    sequences, labels, sources = [], [], []

    for word_dir in sorted(WORD_DATASET_DIR.iterdir()):
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
        logger.error("No valid word sequences found!")
        return None

    X = np.array(sequences, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    logger.info(f"Loaded {len(X)} sequences across {len(word_labels)} classes")
    logger.info(f"Per-class: {dict(Counter(y))}")

    # --- Stratified split guaranteeing all classes in both train and val ---
    rng = np.random.default_rng(42)
    train_idx = []
    val_idx = []

    for cls_idx in range(len(word_labels)):
        cls_indices = [i for i, lbl in enumerate(labels) if lbl == cls_idx]
        cls_sources = list(set([sources[i] for i in cls_indices]))
        
        if len(cls_sources) >= 2:
            # Multi-source class: split by source file to prevent video-level leakage
            rng.shuffle(cls_sources)
            val_src = set(cls_sources[:1])  # 1 source for validation
            for i in cls_indices:
                if sources[i] in val_src:
                    val_idx.append(i)
                else:
                    train_idx.append(i)
        else:
            # Single-source class (26 augmented sequences): split by sequence index (75% train / 25% val)
            shuffled = cls_indices.copy()
            rng.shuffle(shuffled)
            n_val = max(5, round(len(shuffled) * 0.25))
            val_idx.extend(shuffled[:n_val])
            train_idx.extend(shuffled[n_val:])

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    logger.info(f"Training: {len(X_train)} sequences, Validation: {len(X_val)} sequences")
    logger.info(f"Train classes: {len(set(y_train))}/{len(word_labels)} | Val classes: {len(set(y_val))}/{len(word_labels)}")

    # Additional in-memory augmentation for training
    def augment_sequences_inmemory(X, y, copies=5):
        aug_X, aug_y = [X], [y]
        _rng = np.random.default_rng(123)
        for c in range(copies):
            batch = X.copy()
            noise_scale = 0.003 + (c * 0.0015)
            batch += _rng.normal(0, noise_scale, size=batch.shape).astype(np.float32)
            if c % 3 == 0:
                for i in range(len(batch)):
                    n_drop = _rng.integers(1, 4)
                    drop_idx = _rng.choice(SEQUENCE_LENGTH, size=n_drop, replace=False)
                    batch[i, drop_idx] = 0.0
            if c % 4 == 0:
                for i in range(len(batch)):
                    shift = _rng.integers(-2, 3)
                    if shift != 0:
                        batch[i] = np.roll(batch[i], shift, axis=0)
            aug_X.append(batch)
            aug_y.append(y)
        return np.concatenate(aug_X, axis=0), np.concatenate(aug_y, axis=0)

    X_train_aug, y_train_aug = augment_sequences_inmemory(X_train, y_train, copies=8)
    logger.info(f"Augmented training: {len(X_train_aug)} sequences")

    # --- PyTorch Dataset ---
    class SeqDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.long)
        def __len__(self):
            return len(self.X)
        def __getitem__(self, index):
            return self.X[index], self.y[index]

    train_loader = DataLoader(SeqDataset(X_train_aug, y_train_aug), batch_size=32, shuffle=True)
    val_loader = DataLoader(SeqDataset(X_val, y_val), batch_size=32, shuffle=False)

    # --- Model Architecture ---
    class CNNBiLSTMWordClassifier(nn.Module):
        def __init__(self, input_dim=126, cnn_channels=128, hidden_dim=128,
                     num_classes=17, num_layers=2, dropout=0.3):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
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
            self.classifier = nn.Sequential(
                nn.BatchNorm1d(hidden_dim * 2),
                nn.Linear(hidden_dim * 2, 64), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(64, num_classes),
            )

        def forward(self, x):
            x_cnn = self.cnn(x.permute(0, 2, 1))
            lstm_out, _ = self.lstm(x_cnn.permute(0, 2, 1))
            pooled = torch.mean(lstm_out, dim=1)
            return self.classifier(pooled)

    model = CNNBiLSTMWordClassifier(num_classes=len(word_labels))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    # --- Training loop ---
    best_acc = 0.0
    best_state = None
    patience_counter = 0
    max_patience = 15
    epochs = 80

    t0 = time.time()
    for epoch in range(1, epochs + 1):
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

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for val_X, val_y in val_loader:
                v_logits = model(val_X)
                v_preds = torch.argmax(v_logits, dim=1)
                val_correct += (v_preds == val_y).sum().item()
                val_total += len(val_y)

        val_acc = val_correct / val_total if val_total > 0 else 0.0
        scheduler.step(val_acc)

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
            logger.info(f"Early stopping at epoch {epoch}")
            break

    train_time = time.time() - t0

    if best_state:
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

    raw_word_report = classification_report(
        val_labels_all, val_preds_all,
        labels=range(len(word_labels)),
        target_names=word_labels,
        output_dict=True, zero_division=0,
    )
    report_dict: dict = raw_word_report if isinstance(raw_word_report, dict) else {}

    logger.info("=" * 60)
    logger.info(f"CNN-BiLSTM Best Val Accuracy: {best_acc*100:.2f}%")
    logger.info("=" * 60)

    per_class = {}
    zero_classes = []
    for label in word_labels:
        stats = report_dict.get(label)
        if isinstance(stats, dict):
            p_val = float(stats.get('precision', 0.0))
            r_val = float(stats.get('recall', 0.0))
            f1_val = float(stats.get('f1-score', 0.0))
            per_class[label] = {
                'precision': round(p_val, 4),
                'recall': round(r_val, 4),
                'f1-score': round(f1_val, 4),
            }
            if f1_val == 0.0:
                zero_classes.append(label)
            logger.info(f"  {label:15s} P={p_val:.3f} R={r_val:.3f} F1={f1_val:.3f}")

    if zero_classes:
        logger.warning(f"[!] Zero-F1 word classes: {', '.join(zero_classes)}")

    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / 'isl_cnn_lstm_word_model.pt'
    torch.save({
        'model_type': 'cnn_bilstm',
        'model_state_dict': model.state_dict(),
        'input_dim': NUM_FEATURES,
        'cnn_channels': 128,
        'hidden_dim': 128,
        'sequence_length': SEQUENCE_LENGTH,
        'num_classes': len(word_labels),
        'word_labels': word_labels,
    }, str(model_path))

    # Also train and save Random Forest word classifier as backup
    logger.info("\nTraining Random Forest word classifier (backup)...")
    try:
        from sklearn.ensemble import RandomForestClassifier

        X_train_flat = X_train_aug.reshape(len(X_train_aug), -1)
        X_val_flat = X_val.reshape(len(X_val), -1)

        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=3,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_train_flat, y_train_aug)
        rf_preds = rf.predict(X_val_flat)
        rf_acc = accuracy_score(y_val, rf_preds)
        logger.info(f"Random Forest Val Accuracy: {rf_acc*100:.2f}%")

        rf_path = MODEL_DIR / 'isl_word_classifier.pkl'
        with open(rf_path, 'wb') as f:
            pickle.dump(rf, f)

        rf_meta = {
            'model_type': 'random_forest_sequence',
            'train_samples': int(len(X_train_aug)),
            'val_samples': int(len(X_val)),
            'validation_accuracy_evaluable': float(rf_acc),
            'num_classes': len(word_labels),
            'labels': word_labels,
            'sequence_length': SEQUENCE_LENGTH,
            'input_features': SEQUENCE_LENGTH * NUM_FEATURES,
        }
        rf_meta_path = MODEL_DIR / 'word_training_meta.json'
        rf_meta_path.write_text(json.dumps(rf_meta, indent=2), encoding='utf-8')
        logger.info(f"Random Forest saved to: {rf_path}")
    except Exception as e:
        logger.warning(f"Random Forest training failed: {e}")

    # Save CNN-BiLSTM metadata
    meta = {
        'model_type': 'cnn_bilstm',
        'train_samples': int(len(X_train_aug)),
        'val_samples': int(len(X_val)),
        'val_accuracy': best_acc,
        'num_classes': len(word_labels),
        'labels': word_labels,
        'sequence_length': SEQUENCE_LENGTH,
        'input_features': NUM_FEATURES,
        'epochs_trained': epoch,
        'training_time_seconds': round(train_time, 2),
        'architecture': '1D-CNN(126->64->128) + BiLSTM(128, 2-layers) + FC(256->64->N)',
        'per_class': per_class,
        'trained_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }
    meta_path = MODEL_DIR / 'cnn_lstm_training_meta.json'
    meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    logger.info(f"CNN-BiLSTM saved to: {model_path}")
    logger.info(f"Metadata saved to: {meta_path}")

    return meta


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='SignBridge Unified Training Pipeline')
    parser.add_argument('--skip-extract', action='store_true',
                        help='Skip data extraction phase')
    parser.add_argument('--letters-only', action='store_true',
                        help='Train only letter models (XGBoost + ST-GCN)')
    parser.add_argument('--stgcn-only', action='store_true',
                        help='Train only ST-GCN Kinematic Graph letter model')
    parser.add_argument('--words-only', action='store_true',
                        help='Train only word model')
    args = parser.parse_args()

    start = time.time()

    logger.info("#" * 70)
    logger.info("#  SignBridge Unified Training Pipeline v2")
    logger.info("#" * 70)

    # Phase 1: Extraction
    if not args.skip_extract:
        run_extraction()
    else:
        logger.info("Skipping extraction (--skip-extract)")

    results = {}

    # Phase 2A: XGBoost Letter model
    if not args.words_only and not args.stgcn_only:
        try:
            results['letter'] = train_letter_model()
        except Exception as e:
            logger.error(f"Letter model training failed: {e}", exc_info=True)

    # Phase 2B: ST-GCN Kinematic Graph Letter model
    if not args.words_only:
        try:
            results['stgcn'] = train_stgcn_model()
        except Exception as e:
            logger.error(f"ST-GCN model training failed: {e}", exc_info=True)

    # Phase 3: Word model
    if not args.letters_only and not args.stgcn_only:
        try:
            results['word'] = train_word_model()
        except Exception as e:
            logger.error(f"Word model training failed: {e}", exc_info=True)

    # Summary
    total_time = time.time() - start
    logger.info("\n" + "#" * 70)
    logger.info("# TRAINING COMPLETE")
    logger.info("#" * 70)
    logger.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")

    if 'letter' in results and results['letter']:
        m = results['letter']['metrics']
        logger.info(f"XGBoost Letter Model: Val={m['val_accuracy']*100:.1f}% | Test={m['test_accuracy']*100:.1f}%")

    if 'stgcn' in results and results['stgcn']:
        m = results['stgcn']['metrics']
        logger.info(f"ST-GCN Graph Model:   Val={m['val_accuracy']*100:.1f}% | Test={m['test_accuracy']*100:.1f}%")

    if 'word' in results and results['word']:
        logger.info(f"Word Model:           Val={results['word']['val_accuracy']*100:.1f}%")

    logger.info("#" * 70)


if __name__ == '__main__':
    main()
