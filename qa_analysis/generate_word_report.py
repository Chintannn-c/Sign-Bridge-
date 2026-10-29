"""
Generate word_class_report.csv
Scans backend/dataset_words and correlates with backend/models/cnn_lstm_training_meta.json.
Exports:
class_name,sample_count_on_disk,precision,recall,f1_score,status
"""

import csv
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
WORD_DIR = BASE_DIR / 'backend' / 'dataset_words'
META_PATH = BASE_DIR / 'backend' / 'models' / 'cnn_lstm_training_meta.json'
OUT_CSV = Path(__file__).resolve().parent / 'word_class_report.csv'

with open(META_PATH, 'r', encoding='utf-8') as f:
    meta = json.load(f)

per_class_meta = meta.get('per_class', {})
labels = meta.get('labels', [])

rows = []
for label in labels:
    class_dir = WORD_DIR / label
    disk_count = len(list(class_dir.glob('*.json'))) if class_dir.exists() else 0
    stats = per_class_meta.get(label, {})
    p = stats.get('precision', 0.0)
    r = stats.get('recall', 0.0)
    f1 = stats.get('f1-score', 0.0)
    
    # Class status
    if r == 0.0:
        status = "ZERO_RECALL_FAIL"
    elif r < 0.5:
        status = "LOW_RECALL"
    elif disk_count < 10:
        status = "EXTREME_IMBALANCE"
    elif r >= 0.85:
        status = "HIGH_ACCURACY"
    else:
        status = "MODERATE"

    rows.append({
        'class_name': label,
        'sample_count_on_disk': disk_count,
        'precision': round(p, 4),
        'recall': round(r, 4),
        'f1_score': round(f1, 4),
        'status': status
    })

# Sort by sample count descending
rows.sort(key=lambda x: (x['sample_count_on_disk'], x['recall']), reverse=True)

with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['class_name', 'sample_count_on_disk', 'precision', 'recall', 'f1_score', 'status'])
    writer.writeheader()
    writer.writerows(rows)

print(f"Generated {OUT_CSV} with {len(rows)} classes.")
