"""
SignBridge — Batch Video Landmark Extractor (v2) + Heavy Augmentation
Extracts 30-frame temporal landmark sequences from ISL gesture videos (Words and Phrases)
using MediaPipe HandLandmarker Tasks API for Bi-LSTM sequence training.

v2 Enhancements:
  - Heavy augmentation: generates 25+ augmented sequences per original video
  - Augmentation strategies: time warp, speed variation, mirror, noise, frame dropout,
    sub-clip sampling, rotation jitter
  - Produces enough data to train CNN-BiLSTM with non-zero F1 on all classes

Outputs:
  backend/dataset_words/<WORD>/<session_id>.json
"""

import os
import re
import sys
import json
import logging
from pathlib import Path
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
VIDEO_DIRS = [
    PROJECT_ROOT / 'Dataset_Words' / 'Words',
]
OUTPUT_DIR = BASE_DIR / 'dataset_words'
MODEL_TASK_PATH = BASE_DIR / 'models' / 'hand_landmarker.task'

from services.landmark_utils import (
    SEQUENCE_LENGTH,
    clean_word_label as _shared_clean_word_label,
    get_landmark_vector_from_result,
    trim_idle_frames,
    sample_or_interpolate,
    augment_raw_sequence,
)


def clean_word_label(path_or_name):
    """Derives a clean uppercase word class name from video file path or filename."""
    return _shared_clean_word_label(path_or_name, video_dirs=VIDEO_DIRS)


def process_video(video_path, detector, mp):
    """Processes a single video into raw landmark frames, gating low detection and trimming idle lead-in/out."""
    cap = cv2.VideoCapture(str(video_path))
    raw_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = detector.detect(mp_image)
        vec = get_landmark_vector_from_result(results)
        raw_frames.append(vec)

    cap.release()
    return trim_idle_frames(raw_frames)


def main():
    import mediapipe as mp  # type: ignore # pyright: ignore[reportMissingImports]
    from mediapipe.tasks import python  # type: ignore # pyright: ignore[reportMissingImports]
    from mediapipe.tasks.python import vision  # type: ignore # pyright: ignore[reportMissingImports]

    AUGMENTATIONS_PER_VIDEO = 30  # Generate 30 augmented sequences per original

    if not MODEL_TASK_PATH.exists():
        logger.error(f"Task model not found at {MODEL_TASK_PATH}")
        sys.exit(1)

    base_options = python.BaseOptions(model_asset_path=str(MODEL_TASK_PATH))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.25,
        min_hand_presence_confidence=0.25
    )
    detector = vision.HandLandmarker.create_from_options(options)

    videos = []
    video_exts = {'.mp4', '.avi', '.mov', '.webm', '.mkv'}
    for vdir in VIDEO_DIRS:
        if vdir.exists():
            vids = [f for f in vdir.rglob('*') if f.is_file() and f.suffix.lower() in video_exts]
            logger.info(f"Found {len(vids)} gesture videos in {vdir}")
            videos.extend(vids)
    videos = sorted(videos, key=lambda v: (clean_word_label(v), v.name))
    logger.info(f"Total gesture videos to process: {len(videos)}")

    rng = np.random.default_rng(42)
    total_saved = 0
    total_augmented = 0
    words_collected = {}

    for vid in videos:
        word = clean_word_label(vid)
        target_dir = OUTPUT_DIR / word
        target_dir.mkdir(parents=True, exist_ok=True)

        ext_tag = vid.suffix.lower().strip('.')
        session_id = f"vid_{vid.stem.replace(' ', '_').lower()}_{ext_tag}"
        out_file = target_dir / f"{session_id}.json"

        # ponytail: Skip re-extracting videos that already have extracted landmark sequences
        if out_file.exists():
            try:
                data = json.loads(out_file.read_text(encoding='utf-8'))
                seqs = data.get('frame_sequences', [])
                words_collected[word] = words_collected.get(word, 0) + len(seqs)
                total_saved += len(seqs)
                logger.info(f"  [Cached] {vid.name} -> {word}: {len(seqs)} sequences already extracted")
                continue
            except Exception:
                pass

        # Extract raw frames from video
        raw_frames = process_video(vid, detector, mp)
        if not raw_frames or len(raw_frames) < 5:
            logger.warning(f"  Skipping {vid.name} (too few frames: {len(raw_frames) if raw_frames else 0})")
            continue

        # Save original (resampled to 30 frames)
        original_seq = sample_or_interpolate(raw_frames, SEQUENCE_LENGTH)
        all_sequences = [original_seq]

        # Generate augmented sequences
        for aug_i in range(AUGMENTATIONS_PER_VIDEO):
            aug_seq = augment_raw_sequence(raw_frames, rng, aug_i)
            if aug_seq is not None:
                all_sequences.append(aug_seq)
                total_augmented += 1

        # Save all sequences in one file
        out_file = target_dir / f"{session_id}.json"
        out_file.write_text(json.dumps({
            "word": word,
            "session_id": session_id,
            "source_file": vid.name,
            "original_raw_frames": len(raw_frames),
            "frame_sequences": all_sequences
        }, indent=2), encoding='utf-8')

        words_collected[word] = words_collected.get(word, 0) + len(all_sequences)
        total_saved += len(all_sequences)
        logger.info(f"  {vid.name} -> {word}: 1 original + {len(all_sequences)-1} augmented = {len(all_sequences)} sequences")

    detector.close()

    logger.info("=" * 70)
    logger.info(f"=== Video Extraction + Augmentation Complete! ===")
    logger.info(f"  Total videos processed: {len(videos)}")
    logger.info(f"  Total sequences saved:  {total_saved} ({total_augmented} augmented)")
    logger.info(f"  Word classes ({len(words_collected)}):")
    for word, count in sorted(words_collected.items()):
        logger.info(f"    {word:15s}: {count:4d} sequences")
    logger.info("=" * 70)

if __name__ == '__main__':
    main()