"""
Clean, Leak-Free Data Loader and Partition Manager for SignBridge ISL Alphabet Recognition.
Ensures:
1. One canonical data source: backend/dataset_collected
2. Content-hash deduplication (SHA-256) across files and individual frames
3. Preservation of intended RealSign splits:
   - Training: RealSign Training + User Session 1 & 2
   - Validation: RealSign Validation + User Session 3
   - Testing: RealSign Testing + Canonical Reference Letters (100% held out from training & tuning)
4. Group-based splitting at session/file level (no random frame bleed)
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

from services.translator_model import ISL_LABELS, normalize_landmarks

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent.parent
CANONICAL_DATA_DIR = SCRIPT_DIR / 'dataset_collected'
ALPHABET_LABELS = [c for c in ISL_LABELS if c.isalpha()] # A-Z (26 classes)
NUM_RAW_LANDMARKS = 126


def get_file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def load_dataset_partitioned(
    data_dir: Path = CANONICAL_DATA_DIR,
    labels_list: List[str] = ALPHABET_LABELS
) -> Tuple[Dict[str, np.ndarray], Dict[str, dict]]:
    """
    Load dataset strictly partitioned by origin without frame-level leakage:
    - 'train': RealSign Training + Session 1 & Session 2 captures
    - 'val': RealSign Validation + Session 3 captures
    - 'test': RealSign Testing + Canonical reference letters (untouched)
    
    Returns:
        partitions: {
            'X_train': np.ndarray, 'y_train': np.ndarray, 'meta_train': list,
            'X_val': np.ndarray,   'y_val': np.ndarray,   'meta_val': list,
            'X_test': np.ndarray,  'y_test': np.ndarray,  'meta_test': list
        }
        report: Dict containing duplicate statistics, sample counts, and hash info.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Canonical dataset directory not found: {data_path}")

    seen_file_hashes = {}
    seen_frame_hashes = set()
    duplicate_files = []
    duplicate_frames_count = 0

    partitions_data = {
        'train': {'X': [], 'y': [], 'meta': []},
        'val': {'X': [], 'y': [], 'meta': []},
        'test': {'X': [], 'y': [], 'meta': []}
    }

    counts_by_source = {
        'realsign_training': 0,
        'realsign_validation': 0,
        'realsign_testing': 0,
        'canonical_reference': 0,
        'session_train': 0,
        'session_val': 0
    }

    for json_path in sorted(data_path.glob('*/*.json')):
        file_hash = get_file_sha256(json_path)
        if file_hash in seen_file_hashes:
            duplicate_files.append((str(json_path), seen_file_hashes[file_hash]))
            continue
        seen_file_hashes[file_hash] = str(json_path)

        try:
            payload = json.loads(json_path.read_text(encoding='utf-8'))
        except Exception as e:
            logger.warning(f"Skipping corrupt JSON {json_path}: {e}")
            continue

        letter = str(payload.get('letter', '')).upper()
        if letter not in labels_list:
            continue
        cls_idx = labels_list.index(letter)

        filename = json_path.name.lower()
        
        # Determine partition based on ground-truth source provenance
        if 'realsign_training' in filename or 'training' in filename and 'realsign' in filename:
            target_partition = 'train'
            source_tag = 'realsign_training'
        elif 'realsign_validation' in filename or 'validation' in filename and 'realsign' in filename:
            target_partition = 'val'
            source_tag = 'realsign_validation'
        elif 'realsign_testing' in filename or 'testing' in filename and 'realsign' in filename:
            target_partition = 'test'
            source_tag = 'realsign_testing'
        elif 'canonical' in filename:
            target_partition = 'test'
            source_tag = 'canonical_reference'
        elif 'session_3' in filename:
            # Session 3 reserved for validation/evaluation
            target_partition = 'val'
            source_tag = 'session_val'
        elif 'session_1' in filename or 'session_2' in filename:
            # Sessions 1 & 2 for training
            target_partition = 'train'
            source_tag = 'session_train'
        else:
            # Default any extra captures to train
            target_partition = 'train'
            source_tag = 'other_capture'

        frames = payload.get('frames', [])
        for f_idx, frame in enumerate(frames):
            arr = np.asarray(frame, dtype=np.float32)
            if arr.size != NUM_RAW_LANDMARKS or not np.isfinite(arr).all():
                continue

            # Frame-level content hash
            frame_bytes = arr.tobytes()
            frame_hash = hashlib.sha256(frame_bytes).hexdigest()
            if frame_hash in seen_frame_hashes:
                duplicate_frames_count += 1
                continue
            seen_frame_hashes.add(frame_hash)

            normalized = normalize_landmarks(arr)
            if normalized is not None:
                partitions_data[target_partition]['X'].append(normalized)
                partitions_data[target_partition]['y'].append(cls_idx)
                partitions_data[target_partition]['meta'].append({
                    'source': source_tag,
                    'file': json_path.name,
                    'letter': letter,
                    'frame_idx': f_idx
                })
                counts_by_source[source_tag] = counts_by_source.get(source_tag, 0) + 1

    partitions = {
        'X_train': np.asarray(partitions_data['train']['X'], dtype=np.float32),
        'y_train': np.asarray(partitions_data['train']['y'], dtype=np.int32),
        'meta_train': partitions_data['train']['meta'],

        'X_val': np.asarray(partitions_data['val']['X'], dtype=np.float32),
        'y_val': np.asarray(partitions_data['val']['y'], dtype=np.int32),
        'meta_val': partitions_data['val']['meta'],

        'X_test': np.asarray(partitions_data['test']['X'], dtype=np.float32),
        'y_test': np.asarray(partitions_data['test']['y'], dtype=np.int32),
        'meta_test': partitions_data['test']['meta'],
    }

    report = {
        'unique_files_loaded': len(seen_file_hashes),
        'duplicate_files_skipped': len(duplicate_files),
        'duplicate_frames_skipped': duplicate_frames_count,
        'counts_by_source': counts_by_source,
        'train_samples': len(partitions['X_train']),
        'val_samples': len(partitions['X_val']),
        'test_samples': len(partitions['X_test']),
        'total_unique_samples': len(partitions['X_train']) + len(partitions['X_val']) + len(partitions['X_test'])
    }

    return partitions, report
