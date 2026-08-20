"""
SignBridge — Batch Video Landmark Extractor
Extracts 30-frame temporal landmark sequences from ISL gesture videos (Words and Phrases)
using MediaPipe HandLandmarker Tasks API for Bi-LSTM sequence training.

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

def process_video(video_path, detector, mp):
    """Processes a single video into a 30-frame landmark sequence."""
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
        
    return sample_or_interpolate(raw_frames, SEQUENCE_LENGTH)

def main():
    import mediapipe as mp  # type: ignore # pyright: ignore[reportMissingImports]
    from mediapipe.tasks import python  # type: ignore # pyright: ignore[reportMissingImports]
    from mediapipe.tasks.python import vision  # type: ignore # pyright: ignore[reportMissingImports]
    
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
    
    total_saved = 0
    words_collected = {}
    
    for vid in videos:
        word = clean_word_label(vid.name)
        target_dir = OUTPUT_DIR / word
        target_dir.mkdir(parents=True, exist_ok=True)
        
        seq = process_video(vid, detector, mp)
        if seq:
            session_id = f"vid_{vid.stem.replace(' ', '_').lower()}"
            out_file = target_dir / f"{session_id}.json"
            out_file.write_text(json.dumps({
                "word": word,
                "session_id": session_id,
                "source_file": vid.name,
                "frame_sequences": [seq]
            }, indent=2), encoding='utf-8')
            
            words_collected[word] = words_collected.get(word, 0) + 1
            total_saved += 1
            logger.info(f"  Processed {vid.name} -> Class: {word} -> {out_file.name}")
            
    detector.close()
    logger.info(f"=== Video Extraction Complete! Saved {total_saved} sequences across {len(words_collected)} word classes ===")
    logger.info(f"Word classes: {sorted(list(words_collected.keys()))}")

if __name__ == '__main__':
    main()
