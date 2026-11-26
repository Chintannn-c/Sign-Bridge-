"""
SignBridge — INCLUDE-50 Dataset Importer
Imports raw videos or pre-extracted MediaPipe landmarks from the AI4Bharat INCLUDE dataset
into SignBridge's word dataset format (backend/dataset_words/<WORD>/<file>.json).

Usage:
  # Option 1: Import raw downloaded INCLUDE videos
  python backend/scripts/import_include.py --video-dir path/to/include_videos

  # Option 2: Import pre-extracted Kaggle .npy landmarks
  python backend/scripts/import_include.py --npy-dir path/to/include_landmarks
"""

import argparse
import json
import logging
import re
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / 'dataset_words'
SEQUENCE_LENGTH = 30
NUM_FEATURES = 126

# Normalization mapping from INCLUDE names to SignBridge classes
INCLUDE_WORD_MAP = {
    'HELLO': 'HELLO',
    'NAMASTE': 'NAMASTE',
    'THANK YOU': 'THANK_YOU',
    'PLEASE': 'PLEASE',
    'SORRY': 'SORRY',
    'WELCOME': 'WELCOME',
    'HELP': 'HELP',
    'FOOD': 'FOOD',
    'WATER': 'WATER',
    'DOCTOR': 'DOCTOR',
    'TEACHER': 'TEACHER',
    'GOOD': 'GOOD',
    'BAD': 'BAD',
    'EASY': 'EASY',
    'DIFFICULT': 'DIFFICULT',
    'YES': 'YES',
    'NO': 'NO',
    'DEAF': 'DEAF',
    'HEARING': 'HEARING',
    'SIGN': 'SIGN',
    'LANGUAGE': 'LANGUAGE',
    'INDIA': 'INDIA',
    'DAY': 'DAY',
    'MORNING': 'MORNING',
    'AGAIN': 'AGAIN',
    'UNDERSTAND': 'UNDERSTAND',
    'REMEMBER': 'REMEMBER',
    'WASHROOM': 'WASHROOM',
    'WHERE': 'WHERE',
    'MAN': 'MAN',
    'WOMAN': 'WOMAN',
    'BOY': 'BOY',
    'GIRL': 'GIRL',
    'CHILD': 'CHILD',
    'ME': 'ME',
    'YOU': 'YOU',
}


def normalize_label(name: str) -> str:
    """Map arbitrary folder/file label to SignBridge canonical class."""
    clean = re.sub(r'[^a-zA-Z0-9]+', ' ', name).strip().upper()
    if clean in INCLUDE_WORD_MAP:
        return INCLUDE_WORD_MAP[clean]
    # Direct match on underscored names
    u = clean.replace(' ', '_')
    if (OUTPUT_DIR / u).exists():
        return u
    return clean.replace(' ', '_')


def import_npy_landmarks(npy_dir: Path):
    """Import pre-extracted landmark .npy arrays directly into dataset_words."""
    npy_files = list(npy_dir.rglob('*.npy'))
    logger.info(f"Found {len(npy_files)} .npy files in {npy_dir}")

    imported = 0
    for f in npy_files:
        word = normalize_label(f.parent.name)
        target_dir = OUTPUT_DIR / word
        target_dir.mkdir(parents=True, exist_ok=True)

        try:
            arr = np.load(str(f))
            # Expected shape: (T, 126) or (T, 42, 3)
            if arr.ndim == 3:
                arr = arr.reshape(len(arr), -1)
            if arr.shape[1] != NUM_FEATURES:
                continue

            # Resample to SEQUENCE_LENGTH
            indices = np.linspace(0, len(arr) - 1, SEQUENCE_LENGTH, dtype=int)
            seq = arr[indices].astype(np.float32).tolist()

            session_id = f"include_npy_{f.stem.lower()}"
            out_file = target_dir / f"{session_id}.json"
            out_file.write_text(json.dumps({
                "word": word,
                "session_id": session_id,
                "source": "INCLUDE_Kaggle",
                "source_file": f.name,
                "frame_sequences": [seq]
            }, indent=2), encoding='utf-8')
            imported += 1
        except Exception as e:
            logger.warning(f"Failed to process {f.name}: {e}")

    logger.info(f"Successfully imported {imported} sequences from INCLUDE .npy files.")


def import_videos(video_dir: Path):
    """Register video folder and trigger SignBridge extract_video_landmarks."""
    from extract_video_landmarks import VIDEO_DIRS, main as extract_main
    VIDEO_DIRS.append(video_dir)
    logger.info(f"Added {video_dir} to extraction sources. Running extractor...")
    extract_main()


def main():
    parser = argparse.ArgumentParser(description="Import INCLUDE-50 into SignBridge")
    parser.add_argument('--video-dir', type=Path, help="Directory containing raw INCLUDE video clips")
    parser.add_argument('--npy-dir', type=Path, help="Directory containing pre-extracted INCLUDE .npy landmarks")
    args = parser.parse_args()

    if args.npy_dir:
        import_npy_landmarks(args.npy_dir)
    elif args.video_dir:
        import_videos(args.video_dir)
    else:
        logger.info("Specify either --video-dir <path> or --npy-dir <path>.")


if __name__ == '__main__':
    main()

