"""
Script to compute per-class centroids and distance thresholds from dataset_collected.
Generates backend/models/letter_centroids.json used for out-of-dataset rejection.
"""

import os
import glob
import json
import logging
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / 'dataset_collected'
OUTPUT_FILE = BASE_DIR / 'models' / 'letter_centroids.json'

def compute_centroids():
    from services.translator_model import normalize_landmarks_batch

    class_dirs = sorted([d for d in DATASET_DIR.iterdir() if d.is_dir()])
    logger.info(f"Found {len(class_dirs)} class folders in {DATASET_DIR}")

    centroids_data = {}

    for c_dir in class_dirs:
        class_name = c_dir.name
        json_files = list(c_dir.glob('*.json'))
        if not json_files:
            continue

        all_frames = []
        for jf in json_files:
            try:
                with open(jf, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                    frames = content.get('frames', [])
                    for frame in frames:
                        if isinstance(frame, list) and len(frame) == 126:
                            all_frames.append(frame)
            except Exception as e:
                logger.warning(f"Error loading {jf}: {e}")

        if not all_frames:
            continue

        norm_frames = normalize_landmarks_batch(all_frames)
        if norm_frames is None or len(norm_frames) == 0:
            continue

        # Compute centroid
        centroid = np.mean(norm_frames, axis=0)

        # Compute Euclidean distances
        dists = np.linalg.norm(norm_frames - centroid, axis=1)

        # 98th percentile with margin
        p98 = float(np.percentile(dists, 98))
        std = float(np.std(dists))
        mean_d = float(np.mean(dists))
        threshold = max(8.5, min(14.0, mean_d + 2.5 * std, p98 * 1.15))

        centroids_data[class_name] = {
            'centroid': [round(float(v), 5) for v in centroid],
            'threshold': round(float(threshold), 3),
            'mean_dist': round(mean_d, 3),
            'samples': len(norm_frames)
        }
        logger.info(f"Class {class_name:3s}: {len(norm_frames):5d} frames | Mean dist: {mean_d:.2f} | Thr: {threshold:.2f}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(centroids_data, f, indent=2)

    logger.info(f"Successfully saved {len(centroids_data)} class centroids to {OUTPUT_FILE}")

if __name__ == '__main__':
    compute_centroids()
