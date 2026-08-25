"""
SignBridge — High-Speed Hugging Face ISL Zero-Disk Streaming Landmark Extractor
Directly targets specific word subfolders (e.g. N/Namaste.mp4, H/Hello.mp4)
for lightning-fast (<45s) in-memory extraction and heavy augmentation.
"""

import os
import re
import sys
import json
import logging
import argparse
import tempfile
from pathlib import Path
import cv2
import numpy as np
from huggingface_hub import HfFileSystem

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / 'dataset_words'
MODEL_TASK_PATH = BASE_DIR / 'models' / 'hand_landmarker.task'
HF_REPO = "datasets/silentone0725/Indian_Sign_Language_Data.gov_Rencoded"

SEQUENCE_LENGTH = 30  # Standard frame buffer size for Bi-LSTM


def init_mediapipe_detector():
    """Initializes Google MediaPipe HandLandmarker Tasks API with 2-hand detection."""
    import mediapipe as mp
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    if not MODEL_TASK_PATH.exists():
        raise FileNotFoundError(f"MediaPipe task model not found at {MODEL_TASK_PATH}")

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_TASK_PATH)),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.25,
        min_hand_presence_confidence=0.25,
        min_tracking_confidence=0.25
    )
    return HandLandmarker.create_from_options(options), mp


def clean_word_label(raw_name):
    """Derives a standardized uppercase word class name from video filename."""
    if not raw_name:
        return "UNKNOWN"
    stem = Path(raw_name).stem.strip()
    name = re.sub(r'[\(\)\[\]\{\}\-_]+', ' ', stem).strip().upper()
    name = re.sub(r'\s+', '_', name)
    if 'THANK' in name:
        return 'THANK_YOU'
    if 'BYE' in name:
        return 'BYE_BYE'
    if 'NAMASTE' in name:
        return 'NAMASTE'
    if 'INDIAN' in name or name == 'INDIA':
        return 'INDIA'
    if name == 'HI':
        return 'HELLO'
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


def extract_raw_frames_from_video_bytes(video_bytes, detector, mp):
    """
    Decodes video frames directly from memory bytes buffer and extracts 42 3D landmarks per frame.
    Immediately unlinks temporary file from disk upon completion.
    """
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            tmp_file.write(video_bytes)
            tmp_path = tmp_file.name

        cap = cv2.VideoCapture(tmp_path)
        raw_frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_image)
            vec = get_landmark_vector_from_result(result)
            raw_frames.append(vec)

        cap.release()
        return raw_frames
    except Exception as e:
        logger.warning(f"Error decoding video stream: {e}")
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


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


def augment_raw_sequence(raw_frames, rng):
    """Generates an augmented 30-frame sequence (speed warp, jitter, mirror, noise)."""
    arr = np.array(raw_frames, dtype=np.float32)
    n_raw = len(arr)
    if n_raw < 5:
        return None

    clip_ratio = rng.uniform(0.65, 1.0)
    clip_len = max(5, int(n_raw * clip_ratio))
    max_start = max(0, n_raw - clip_len)
    start = rng.integers(0, max_start + 1)
    sub_clip = arr[start:start + clip_len]

    speed_factor = rng.uniform(0.8, 1.25)
    target_frames = max(10, int(SEQUENCE_LENGTH * speed_factor))
    indices = np.linspace(0, len(sub_clip) - 1, target_frames)
    resampled = np.array([sub_clip[int(round(idx))] for idx in indices])

    if len(resampled) != SEQUENCE_LENGTH:
        final_indices = np.linspace(0, len(resampled) - 1, SEQUENCE_LENGTH)
        resampled = np.array([resampled[int(round(idx))] for idx in final_indices])

    result = resampled.copy()
    noise_scale = rng.uniform(0.002, 0.008)
    result += rng.normal(0, noise_scale, size=result.shape).astype(np.float32)

    # Random mirror (swap left/right hand)
    if rng.random() < 0.25:
        mirrored = np.zeros_like(result)
        mirrored[:, :63] = result[:, 63:]
        mirrored[:, 63:] = result[:, :63]
        result = mirrored

    return result.tolist()


def main():
    parser = argparse.ArgumentParser(description="SignBridge Direct-Lookup Zero-Disk Streaming Extractor")
    parser.add_argument("--augment", type=int, default=10, help="Number of synthetic augmentations per video")
    parser.add_argument("--target-words", type=str, default="HELLO,NAMASTE,INDIA,SIGN,PLEASE,THANK_YOU,SORRY,AGAIN,DEAF,HEARING,WELCOME,ME,YOU,MAN,WOMAN,BYE_BYE,WASHROOM,HELP,WATER,WHERE,FOOD,DOCTOR", help="Target words")
    args = parser.parse_args()

    targets = [w.strip().upper().replace(" ", "_") for w in args.target_words.split(",") if w.strip()]

    logger.info("Initializing Google MediaPipe HandLandmarker...")
    detector, mp = init_mediapipe_detector()

    logger.info("Connecting to Hugging Face Hub (HfFileSystem)...")
    fs = HfFileSystem()
    rng = np.random.default_rng(42)

    total_saved = 0
    words_found = 0

    for word in targets:
        # Determine initial letter folder (e.g. 'NAMASTE' -> 'N')
        first_letter = word[0]
        search_pattern = f"{HF_REPO}/{first_letter}/*.mp4"
        
        try:
            folder_files = fs.glob(search_pattern)
        except Exception as e:
            logger.warning(f"Could not scan folder {first_letter}: {e}")
            continue

        # Look for exact or closely matching file
        matched_file = None
        for f in folder_files:
            cleaned = clean_word_label(f)
            if cleaned == word or word in cleaned.split('_'):
                matched_file = f
                break

        if not matched_file:
            logger.info(f"  [Skipped] Word '{word}' not found in folder '{first_letter}/'.")
            continue

        logger.info(f"Streaming [{word}] from: {matched_file}...")
        try:
            with fs.open(matched_file, "rb") as f:
                video_bytes = f.read()
        except Exception as e:
            logger.warning(f"  Failed to stream video {matched_file}: {e}")
            continue

        raw_frames = extract_raw_frames_from_video_bytes(video_bytes, detector, mp)
        if not raw_frames or len(raw_frames) < 5:
            logger.warning(f"  Insufficient landmarks detected for {word}")
            continue

        target_dir = OUTPUT_DIR / word
        target_dir.mkdir(parents=True, exist_ok=True)

        base_seq = sample_or_interpolate(raw_frames, SEQUENCE_LENGTH)

        # 1. Save canonical 30-frame sequence (15 KB)
        out_path = target_dir / f"session_hf_canonical.json"
        out_path.write_text(json.dumps({
            "word": word,
            "session_id": f"hf_{word}_canonical",
            "source": f"ISLRTC_HuggingFace/{matched_file}",
            "sequence_length": SEQUENCE_LENGTH,
            "frames": base_seq
        }, indent=2), encoding='utf-8')
        total_saved += 1

        # 2. Generate heavy augmented variations (10 augmentations)
        for aug_idx in range(args.augment):
            aug_seq = augment_raw_sequence(raw_frames, rng)
            if aug_seq:
                aug_path = target_dir / f"session_hf_aug_{aug_idx:02d}.json"
                aug_path.write_text(json.dumps({
                    "word": word,
                    "session_id": f"hf_{word}_aug_{aug_idx}",
                    "source": "ISLRTC_HuggingFace/Augmented",
                    "sequence_length": SEQUENCE_LENGTH,
                    "frames": aug_seq
                }, indent=2), encoding='utf-8')
                total_saved += 1

        words_found += 1
        logger.info(f"  -> Extracted & generated {args.augment + 1} sequences for [{word}] (Saved in {target_dir.name}/)")

    logger.info(f"\n==========================================")
    logger.info(f"Direct In-Memory Streaming Complete!")
    logger.info(f"Words processed: {words_found}/{len(targets)}")
    logger.info(f"Total landmark sequence JSONs saved: {total_saved}")
    logger.info(f"Zero video files saved on disk.")
    logger.info(f"==========================================")


if __name__ == "__main__":
    main()
