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


from services.landmark_utils import (
    clean_word_label,
    get_landmark_vector_from_result,
    trim_idle_frames,
    sample_or_interpolate,
    augment_raw_sequence,
)


def extract_raw_frames_from_video_bytes(video_bytes, detector, mp):
    """
    Decodes video frames directly from memory bytes buffer and extracts 42 3D landmarks per frame.
    Trims idle lead-in/out frames and unlinks temporary file from disk upon completion.
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
        return trim_idle_frames(raw_frames)
    except Exception as e:
        logger.warning(f"Error decoding video stream: {e}")
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


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

        base_seq = sample_or_interpolate(raw_frames, 30)
        all_sequences = [base_seq]

        for aug_idx in range(args.augment):
            aug_seq = augment_raw_sequence(raw_frames, rng, aug_id=aug_idx)
            if aug_seq:
                all_sequences.append(aug_seq)

        # Save all sequences for this video together (conforms to File 1 schema)
        matched_stem = Path(matched_file).stem.replace(' ', '_').lower()
        session_id = f"hf_{word.lower()}_{matched_stem}"
        out_path = target_dir / f"{session_id}.json"
        out_path.write_text(json.dumps({
            "word": word,
            "session_id": session_id,
            "source_file": matched_file,
            "source": f"ISLRTC_HuggingFace/{matched_file}",
            "original_raw_frames": len(raw_frames),
            "frame_sequences": all_sequences
        }, indent=2), encoding='utf-8')
        total_saved += len(all_sequences)

        words_found += 1
        logger.info(f"  -> Saved {len(all_sequences)} sequences (1 canonical + {len(all_sequences)-1} augmented) for [{word}] in {out_path.name}")

    logger.info(f"\n==========================================")
    logger.info(f"Direct In-Memory Streaming Complete!")
    logger.info(f"Words processed: {words_found}/{len(targets)}")
    logger.info(f"Total landmark sequence JSONs saved: {total_saved}")
    logger.info(f"Zero video files saved on disk.")
    logger.info(f"==========================================")


if __name__ == "__main__":
    main()
