"""
SignBridge — Custom Photo Ingestion Tool
Processes user-provided photos from `backend/custom_photos/<LETTER>/`
and extracts MediaPipe hand landmarks directly into `backend/dataset_collected/<LETTER>/session_custom_photos.json`.

How to Use:
1. Create folders for the letters you want to improve inside `backend/custom_photos/`
   Example:
     backend/custom_photos/S/photo1.jpg
     backend/custom_photos/S/photo2.jpg
     backend/custom_photos/O/my_sign.png
     backend/custom_photos/W/w_angle1.jpg
2. Run this script:
     cd backend
     python add_custom_photos.py
3. Retrain the model:
     python train_unified.py --letters-only --skip-extract
"""

import os
import sys
import json
import logging
from pathlib import Path
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("AddCustomPhotos")

BASE_DIR = Path(__file__).resolve().parent
PHOTOS_DIR = BASE_DIR / 'custom_photos'
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


def main():
    if not PHOTOS_DIR.exists():
        PHOTOS_DIR.mkdir(parents=True, exist_ok=True)
        # Create sample placeholder folders for the weak letters
        for letter in ['S', 'O', 'W', 'T', 'K', 'P']:
            (PHOTOS_DIR / letter).mkdir(exist_ok=True)
        logger.info(f"Created custom photos directory at: {PHOTOS_DIR}")
        logger.info("Please drop your photos into the respective letter subfolders and run this script again.")
        return

    # Import mediapipe tasks
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks import python as mp_python
    except ImportError:
        logger.error("MediaPipe is not installed. Please run: pip install mediapipe opencv-python")
        return

    if not MODEL_TASK_PATH.exists():
        logger.error(f"MediaPipe task model not found at {MODEL_TASK_PATH}")
        return

    base_options = mp_python.BaseOptions(model_asset_path=str(MODEL_TASK_PATH))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.35,
        min_tracking_confidence=0.35
    )
    detector = vision.HandLandmarker.create_from_options(options)

    total_images_processed = 0
    total_landmarks_saved = 0
    letter_counts = {}

    subdirs = [d for d in sorted(PHOTOS_DIR.iterdir()) if d.is_dir() and not d.name.startswith('.')]

    if not subdirs:
        logger.warning(f"No letter folders found inside {PHOTOS_DIR}")
        logger.info("Example structure:")
        logger.info("  backend/custom_photos/S/img1.jpg")
        logger.info("  backend/custom_photos/O/img2.png")
        return

    for letter_dir in subdirs:
        letter = letter_dir.name.upper()
        target_dir = OUTPUT_DIR / letter
        target_dir.mkdir(parents=True, exist_ok=True)

        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        img_paths = [p for p in letter_dir.iterdir() if p.suffix.lower() in valid_extensions]

        if not img_paths:
            continue

        frames = []
        logger.info(f"Processing {len(img_paths)} pictures for Letter '{letter}'...")

        for img_p in img_paths:
            img = cv2.imread(str(img_p))
            if img is None:
                logger.warning(f"Could not read image: {img_p.name}")
                continue

            total_images_processed += 1

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
            else:
                logger.warning(f"  [No Hand Detected in {img_p.name}] Make sure hands are well-lit and clearly visible.")

        if frames:
            session_id = "session_custom_photos"
            out_file = target_dir / f"{session_id}.json"

            # Merge with existing custom photos if present
            existing_frames = []
            if out_file.exists():
                try:
                    data = json.loads(out_file.read_text(encoding='utf-8'))
                    existing_frames = data.get('frames', [])
                except Exception:
                    pass

            all_frames = existing_frames + frames
            out_file.write_text(json.dumps({
                "letter": letter,
                "session_id": session_id,
                "source": "custom_user_photos",
                "frames": all_frames
            }, indent=2), encoding='utf-8')

            total_landmarks_saved += len(frames)
            letter_counts[letter] = len(all_frames)
            logger.info(f"  ✅ Letter '{letter}': Added {len(frames)} new landmark frames (Total custom: {len(all_frames)}) -> {out_file.name}")

    detector.close()

    print("\n" + "=" * 60)
    print("🎉 CUSTOM PHOTO INGESTION COMPLETE")
    print("=" * 60)
    print(f"Total Photos Processed : {total_images_processed}")
    print(f"Total Hand Landmarks Saved: {total_landmarks_saved}")
    if letter_counts:
        print("\nSummary by Letter:")
        for ltr, count in letter_counts.items():
            print(f"  - Letter {ltr}: {count} total frames in dataset")
        print("\nNext Step: Run model retraining:")
        print("  cd backend")
        print("  python train_unified.py --letters-only --skip-extract")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
