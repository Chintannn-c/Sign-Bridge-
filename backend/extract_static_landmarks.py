"""
SignBridge — Batch Static Landmark Extractor
Processes ISL image datasets (RealSign, ISL self made dataset) using MediaPipe HandLandmarker
and extracts normalized 126-feature vectors (21 keypoints x 3 coords x 2 hands).

Outputs:
  backend/dataset_collected/<LETTER>/session_<source>_<id>.json
"""

import os
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
DATASET_DIR = PROJECT_ROOT / 'dataset'
OUTPUT_DIR = BASE_DIR / 'dataset_collected'
MODEL_TASK_PATH = BASE_DIR / 'models' / 'hand_landmarker.task'

def get_landmark_vector_from_result(detection_result):
    """
    Standardizes MediaPipe HandLandmarker result into a 126-float array.
    Indices 0..62: Left hand (21 x 3)
    Indices 63..125: Right hand (21 x 3)
    """
    left_hand = np.zeros((21, 3), dtype=np.float32)
    right_hand = np.zeros((21, 3), dtype=np.float32)
    
    if not detection_result.hand_landmarks:
        return None

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

def process_realsign_letters(detector, mp):
    """Processes RealSign 26 canonical reference letters (Letters/A.jpg - Z.jpg)."""
    letters_dir = DATASET_DIR / 'RealSign-Indian-Sign-Language-Dataset-main' / 'Letters'
    if not letters_dir.exists():
        logger.warning(f"RealSign Letters directory not found at {letters_dir}")
        return 0

    total_extracted = 0
    logger.info("Processing RealSign Letters (A-Z canonical reference images)...")
    for img_p in sorted(list(letters_dir.glob('*.jpg')) + list(letters_dir.glob('*.png'))):
        letter = img_p.stem.upper()
        target_dir = OUTPUT_DIR / letter
        target_dir.mkdir(parents=True, exist_ok=True)

        img = cv2.imread(str(img_p))
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = detector.detect(mp_image)

        if results.hand_landmarks:
            vec = get_landmark_vector_from_result(results)
            if vec:
                session_id = "realsign_canonical_letter"
                out_file = target_dir / f"{session_id}.json"
                out_file.write_text(json.dumps({
                    "letter": letter,
                    "session_id": session_id,
                    "source": "RealSign/Letters",
                    "frames": [vec]
                }, indent=2), encoding='utf-8')
                total_extracted += 1
                logger.info(f"  [Canonical] Letter {letter}: Landmark saved -> {out_file.name}")
    return total_extracted

def process_realsign_dataset(detector, mp, max_samples_per_class=350):
    """Processes RealSign Training and Validation datasets."""
    realsign_dir = DATASET_DIR / 'RealSign-Indian-Sign-Language-Dataset-main'
    if not realsign_dir.exists():
        logger.warning(f"RealSign dataset directory not found at {realsign_dir}")
        return 0

    total_extracted = 0
    splits = ['Training', 'Validation', 'Testing']
    
    for split in splits:
        split_dir = realsign_dir / split
        if not split_dir.exists():
            continue
            
        logger.info(f"Processing RealSign split: {split}...")
        for letter_dir in sorted(split_dir.iterdir()):
            if not letter_dir.is_dir() or letter_dir.name.startswith('.'):
                continue
                
            letter = letter_dir.name.upper()
            target_dir = OUTPUT_DIR / letter
            target_dir.mkdir(parents=True, exist_ok=True)
            
            img_paths = sorted(list(letter_dir.glob('*.jpg')) + list(letter_dir.glob('*.png')))
            if max_samples_per_class:
                img_paths = img_paths[:max_samples_per_class]
                
            frames = []
            for img_p in img_paths:
                img = cv2.imread(str(img_p))
                if img is None:
                    continue
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                results = detector.detect(mp_image)
                
                if results.hand_landmarks:
                    vec = get_landmark_vector_from_result(results)
                    if vec:
                        frames.append(vec)
                        
            if frames:
                session_id = f"realsign_{split.lower()}_{len(frames)}"
                out_file = target_dir / f"{session_id}.json"
                out_file.write_text(json.dumps({
                    "letter": letter,
                    "session_id": session_id,
                    "source": f"RealSign/{split}",
                    "frames": frames
                }, indent=2), encoding='utf-8')
                total_extracted += len(frames)
                logger.info(f"  [RealSign {split}] Letter {letter}: {len(frames)}/{len(img_paths)} landmarks saved -> {out_file.name}")
                
    return total_extracted

def process_self_made_dataset(detector, mp, max_samples_per_class=250):
    """Processes ISL self made dataset (0-9 and a-z)."""
    self_made_dir = DATASET_DIR / 'ISL self made dataset'
    if not self_made_dir.exists():
        logger.warning(f"ISL self made dataset not found at {self_made_dir}")
        return 0

    total_extracted = 0
    logger.info("Processing ISL self made dataset...")
    
    for class_dir in sorted(self_made_dir.iterdir()):
        if not class_dir.is_dir() or class_dir.name.startswith('.'):
            continue
            
        label = class_dir.name.upper()
        target_dir = OUTPUT_DIR / label
        target_dir.mkdir(parents=True, exist_ok=True)
        
        img_paths = sorted(list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png')))
        if max_samples_per_class:
            img_paths = img_paths[:max_samples_per_class]
            
        frames = []
        for img_p in img_paths:
            img = cv2.imread(str(img_p))
            if img is None:
                continue
            if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
                rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = detector.detect(mp_image)
            if results.hand_landmarks:
                vec = get_landmark_vector_from_result(results)
                if vec:
                    frames.append(vec)
                    
        if frames:
            session_id = f"selfmade_{len(frames)}"
            out_file = target_dir / f"{session_id}.json"
            out_file.write_text(json.dumps({
                "letter": label,
                "session_id": session_id,
                "source": "ISL_self_made",
                "frames": frames
            }, indent=2), encoding='utf-8')
            total_extracted += len(frames)
            logger.info(f"  [Self-Made] Class {label}: {len(frames)}/{len(img_paths)} landmarks saved -> {out_file.name}")
            
    return total_extracted

def main():
    import mediapipe as mp  # type: ignore # pyright: ignore[reportMissingImports]
    from mediapipe.tasks import python  # type: ignore # pyright: ignore[reportMissingImports]
    from mediapipe.tasks.python import vision  # type: ignore # pyright: ignore[reportMissingImports]
    
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
    
    logger.info("=== Starting MediaPipe Static Landmark Extraction ===")
    l_count = process_realsign_letters(detector, mp)
    r_count = process_realsign_dataset(detector, mp, max_samples_per_class=350)
    s_count = process_self_made_dataset(detector, mp, max_samples_per_class=250)
    
    detector.close()
    logger.info(f"=== Extraction Complete! CanonicalLetters={l_count}, RealSignSplits={r_count}, SelfMade={s_count}, Total={l_count + r_count + s_count} ===")

if __name__ == '__main__':
    main()
