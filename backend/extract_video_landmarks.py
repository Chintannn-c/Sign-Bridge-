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

SEQUENCE_LENGTH = 30  # Standard frame buffer size for Bi-LSTM

def _normalize_word_str(s):
    """Normalizes string to canonical uppercase word label."""
    name = re.sub(r'[\d_\-,\.]+', ' ', s).strip().upper()
    name = re.sub(r'\s+', '_', name)
    if 'THANK' in name and 'VERY' in name:
        return 'THANK_YOU_VERY_MUCH'
    if 'THANK' in name:
        return 'THANK_YOU'
    if 'BYE' in name:
        return 'BYE_BYE'
    if 'NAMASTE' in name:
        return 'NAMASTE'
    if 'INDIAN' in name or 'INDIA' in name:
        return 'INDIA'
    if 'DONT_UNDERSTAND' in name or 'I_DONT_UNDERSTAND' in name:
        return 'I_DONT_UNDERSTAND'
    if 'I_AM_FINE' in name or 'IM_FINE' in name:
        return 'IM_FINE'
    if 'MY_NAME' in name:
        return 'MY_NAME_IS'
    if 'GOOD_MORNING' in name:
        return 'GOOD_MORNING'
    if 'GOOD_AFTERNOON' in name:
        return 'GOOD_AFTERNOON'
    if 'GOOD_EVENING' in name:
        return 'GOOD_EVENING'
    if 'GOOD_NIGHT' in name:
        return 'GOOD_NIGHT'
    if 'HOW_ARE_YOU' in name:
        return 'HOW_ARE_YOU'
    if 'NO_FEAR' in name:
        return 'NO_FEAR'
    if name in ('ME', 'I', 'ME_I') or name.startswith('ME_'):
        return 'ME'
    return name


def clean_word_label(path_or_name):
    """Derives a clean uppercase word class name from video file path or filename."""
    p = Path(path_or_name)
    # Check if the video is located in a word subdirectory under VIDEO_DIRS
    for vdir in VIDEO_DIRS:
        try:
            rel = p.resolve().relative_to(vdir.resolve())
            if len(rel.parts) > 1:
                folder_cand = _normalize_word_str(rel.parts[0])
                if folder_cand:
                    return folder_cand
        except (ValueError, RuntimeError):
            pass
    return _normalize_word_str(p.stem)


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

    indices = np.linspace(0, current_len - 1, target_length).round().astype(np.int64)
    return arr[indices].tolist()


def augment_raw_sequence(raw_frames, rng, aug_id):
    """
    Apply heavy augmentation to a raw sequence of frames before resampling.
    Strategies:
      1. Random sub-clip sampling (60-100% of the video)
      2. Speed variation (resample to different length first)
      3. Non-linear time warping (fast start, slow middle, fast end, etc.)
      4. Gaussian landmark coordinate jitter
      5. Frame dropout (zero out random frames)
      6. Hand mirror / horizontal flip (simulate opposite-hand or two-hand variants)
      7. Small 2D rotation jitter around wrist
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
    speed_idx = np.linspace(0, len(sub_clip) - 1, target_frames).round().astype(np.int64)
    resampled = sub_clip[speed_idx]
    
    # --- Strategy 3: Time warp ---
    if rng.random() < 0.4:
        # Non-linear time warping
        warp_points = np.sort(rng.uniform(0, 1, size=3))
        warp_points = np.concatenate([[0], warp_points, [1]])
        target_points = np.sort(rng.uniform(0, 1, size=3))
        target_points = np.concatenate([[0], target_points, [1]])
        
        orig_indices = np.linspace(0, 1, len(resampled))
        warped_indices = np.interp(orig_indices, warp_points, target_points)
        warp_idx = np.clip(warped_indices * (len(resampled) - 1), 0, len(resampled) - 1).round().astype(np.int64)
        resampled = resampled[warp_idx]
    
    # Final resample to exact SEQUENCE_LENGTH
    if len(resampled) != SEQUENCE_LENGTH:
        final_idx = np.linspace(0, len(resampled) - 1, SEQUENCE_LENGTH).round().astype(np.int64)
        resampled = resampled[final_idx]
    
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

        # Extract raw frames from video
        raw_frames = process_video(vid, detector, mp)
        if not raw_frames or len(raw_frames) < 5:
            logger.warning(f"  Skipping {vid.name} (too few frames: {len(raw_frames) if raw_frames else 0})")
            continue

        # Save original (resampled to 30 frames)
        original_seq = sample_or_interpolate(raw_frames, SEQUENCE_LENGTH)
        ext_tag = vid.suffix.lower().strip('.')
        session_id = f"vid_{vid.stem.replace(' ', '_').lower()}_{ext_tag}"
        
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