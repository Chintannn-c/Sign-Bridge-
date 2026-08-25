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
VIDEO_DIR = PROJECT_ROOT / 'dataset' / 'Words and Phrases'
OUTPUT_DIR = BASE_DIR / 'dataset_words'
MODEL_TASK_PATH = BASE_DIR / 'models' / 'hand_landmarker.task'

SEQUENCE_LENGTH = 30  # Standard frame buffer size for Bi-LSTM


def clean_word_label(filename):
    """Derives a clean uppercase word class name from video filename."""
    stem = Path(filename).stem.strip()
    name = re.sub(r'[\d_]+', ' ', stem).strip().upper()
    name = re.sub(r'\s+', '_', name)
    if 'THANK' in name:
        return 'THANK_YOU'
    if 'BYE' in name:
        return 'BYE_BYE'
    if 'NAMASTE' in name:
        return 'NAMASTE'
    if 'INDIAN' in name or 'INDIA' in name:
        return 'INDIA'
    return name


def get_landmark_vector_from_result(detection_result):
    """Standardizes MediaPipe HandLandmarker result into a 126-float array."""
    left_hand = np.zeros((21, 3), dtype=np.float32)
    right_hand = np.zeros((21, 3), dtype=np.float32)

    if not detection_result.hand_landmarks:
        return np.zeros(126, dtype=np.float32).tolist()

    handedness_list = detection_result.handedness or []
    landmarks_list = detection_result.hand_landmarks

    for i, hand_lms in enumerate(landmarks_list):
        label = None
        if i < len(handedness_list) and handedness_list[i]:
            label = handedness_list[i][0].category_name

        coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms], dtype=np.float32)

        if label == 'Left':
            left_hand = coords
        elif label == 'Right':
            right_hand = coords
        else:
            if coords[0, 0] < 0.5:
                left_hand = coords
            else:
                right_hand = coords

    if len(landmarks_list) == 1 and not np.any(right_hand != 0) and not np.any(left_hand != 0):
        coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks_list[0]], dtype=np.float32)
        right_hand = coords

    combined = np.concatenate([left_hand.reshape(-1), right_hand.reshape(-1)])
    return combined.tolist()


def sample_or_interpolate(sequence, target_length=SEQUENCE_LENGTH):
    """Resamples a list of landmark frames to exactly target_length frames."""
    arr = np.array(sequence, dtype=np.float32)
    current_len = len(arr)
    if current_len == target_length:
        return arr.tolist()
    if current_len == 0:
        return np.zeros((target_length, 126), dtype=np.float32).tolist()

    indices = np.linspace(0, current_len - 1, target_length)
    sampled = []
    for idx in indices:
        sampled.append(arr[int(round(idx))].tolist())
    return sampled


def augment_raw_sequence(raw_frames, rng, aug_id):
    """
    Generate a single augmented 30-frame sequence from raw video frames.
    
    Strategies (combined randomly):
      1. Sub-clip sampling: take different temporal windows from the raw frames
      2. Speed variation: stretch/compress the gesture speed
      3. Gaussian noise: add coordinate jitter
      4. Frame dropout: replace random frames with zeros
      5. Time warp: non-linear time stretching
      6. Hand mirror: swap left/right hand slots
      7. Rotation jitter: small 2D rotations on landmark coordinates
    """
    arr = np.array(raw_frames, dtype=np.float32)
    n_raw = len(arr)
    
    if n_raw < 5:
        return None
    
    # --- Strategy 1: Sub-clip sampling ---
    # Take a random sub-window (60-100% of the video)
    clip_ratio = rng.uniform(0.6, 1.0)
    clip_len = max(5, int(n_raw * clip_ratio))
    max_start = max(0, n_raw - clip_len)
    start = rng.integers(0, max_start + 1)
    sub_clip = arr[start:start + clip_len]
    
    # --- Strategy 2: Speed variation ---
    # Resample to SEQUENCE_LENGTH with slight speed jitter
    speed_factor = rng.uniform(0.8, 1.2)
    target_frames = max(10, int(SEQUENCE_LENGTH * speed_factor))
    indices = np.linspace(0, len(sub_clip) - 1, target_frames)
    resampled = np.array([sub_clip[int(round(idx))] for idx in indices])
    
    # --- Strategy 3: Time warp ---
    if rng.random() < 0.4:
        # Non-linear time warping
        warp_points = np.sort(rng.uniform(0, 1, size=3))
        warp_points = np.concatenate([[0], warp_points, [1]])
        target_points = np.sort(rng.uniform(0, 1, size=3))
        target_points = np.concatenate([[0], target_points, [1]])
        
        orig_indices = np.linspace(0, 1, len(resampled))
        warped_indices = np.interp(orig_indices, warp_points, target_points)
        warped_indices = np.clip(warped_indices * (len(resampled) - 1), 0, len(resampled) - 1)
        resampled = np.array([resampled[int(round(idx))] for idx in warped_indices])
    
    # Final resample to exact SEQUENCE_LENGTH
    if len(resampled) != SEQUENCE_LENGTH:
        final_indices = np.linspace(0, len(resampled) - 1, SEQUENCE_LENGTH)
        resampled = np.array([resampled[(round(idx))] for idx in final_indices])
    
    result = resampled.copy()
    
    # --- Strategy 4: Gaussian noise ---
    noise_scale = rng.uniform(0.002, 0.010)
    result += rng.normal(0, noise_scale, size=result.shape).astype(np.float32)
    
    # --- Strategy 5: Frame dropout ---
    if rng.random() < 0.3:
        n_drop = rng.integers(1, 4)
        drop_idx = rng.choice(SEQUENCE_LENGTH, size=n_drop, replace=False)
        result[drop_idx] = 0.0
    
    # --- Strategy 6: Hand mirror (swap left/right) ---
    if rng.random() < 0.3:
        mirrored = np.zeros_like(result)
        mirrored[:, :63] = result[:, 63:]
        mirrored[:, 63:] = result[:, :63]
        result = mirrored
    
    # --- Strategy 7: Rotation jitter ---
    if rng.random() < 0.4:
        angle = rng.uniform(-0.15, 0.15)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        pts = result.reshape(SEQUENCE_LENGTH, 42, 3).copy()
        x, y = pts[:, :, 0].copy(), pts[:, :, 1].copy()
        pts[:, :, 0] = x * cos_a - y * sin_a
        pts[:, :, 1] = x * sin_a + y * cos_a
        result = pts.reshape(SEQUENCE_LENGTH, 126)
    
    return result.tolist()


def process_video(video_path, detector, mp):
    """Processes a single video into raw landmark frames (variable length)."""
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

    if not raw_frames:
        return None

    return raw_frames


def main():
    import mediapipe as mp  # type: ignore # pyright: ignore[reportMissingImports]
    from mediapipe.tasks import python  # type: ignore # pyright: ignore[reportMissingImports]
    from mediapipe.tasks.python import vision  # type: ignore # pyright: ignore[reportMissingImports]

    AUGMENTATIONS_PER_VIDEO = 25  # Generate 25 augmented sequences per original

    if not MODEL_TASK_PATH.exists():
        logger.error(f"Task model not found at {MODEL_TASK_PATH}")
        sys.exit(1)

    if not VIDEO_DIR.exists():
        logger.error(f"Video directory not found at {VIDEO_DIR}")
        sys.exit(1)

    base_options = python.BaseOptions(model_asset_path=str(MODEL_TASK_PATH))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.25,
        min_hand_presence_confidence=0.25
    )
    detector = vision.HandLandmarker.create_from_options(options)

    videos = sorted(list(VIDEO_DIR.glob('*.mp4')) + list(VIDEO_DIR.glob('*.avi')) + list(VIDEO_DIR.glob('*.mov')))
    logger.info(f"Found {len(videos)} gesture videos in {VIDEO_DIR}")

    rng = np.random.default_rng(42)
    total_saved = 0
    total_augmented = 0
    words_collected = {}

    for vid in videos:
        word = clean_word_label(vid.name)
        target_dir = OUTPUT_DIR / word
        target_dir.mkdir(parents=True, exist_ok=True)

        # Extract raw frames from video
        raw_frames = process_video(vid, detector, mp)
        if not raw_frames or len(raw_frames) < 5:
            logger.warning(f"  Skipping {vid.name} (too few frames: {len(raw_frames) if raw_frames else 0})")
            continue

        # Save original (resampled to 30 frames)
        original_seq = sample_or_interpolate(raw_frames, SEQUENCE_LENGTH)
        session_id = f"vid_{vid.stem.replace(' ', '_').lower()}"
        
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
