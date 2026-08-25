"""
SignBridge — Dataset 2 Batch Ingestion & Landmark Extraction
Processes all 521 photos from C:\\React\\SignBridge\\dataset_2 into canonical landmark JSONs.
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
DATASET_2_DIR = BASE_DIR.parent / 'dataset_2'
OUTPUT_DIR = BASE_DIR / 'dataset_collected'
MODEL_TASK_PATH = BASE_DIR / 'models' / 'hand_landmarker.task'


def get_landmark_vector(detection_result):
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


def augment_vector(vec_126, rng, copies=5):
    """Generates slight rotation, scale, and jitter variations of a landmark vector."""
    arr = np.array(vec_126, dtype=np.float32).reshape(2, 21, 3)
    aug_vectors = []
    
    for _ in range(copies):
        copy_pts = arr.copy()
        
        # 1. Random 2D rotation jitter (-10 to +10 degrees)
        angle = rng.uniform(-0.15, 0.15)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        for h in range(2):
            if np.any(copy_pts[h] != 0):
                # Rotate around hand wrist
                wrist = copy_pts[h, 0].copy()
                centered = copy_pts[h] - wrist
                rx = centered[:, 0] * cos_a - centered[:, 1] * sin_a
                ry = centered[:, 0] * sin_a + centered[:, 1] * cos_a
                copy_pts[h, :, 0] = rx + wrist[0]
                copy_pts[h, :, 1] = ry + wrist[1]
                
        # 2. Scale jitter (0.95 to 1.05)
        scale = rng.uniform(0.95, 1.05)
        for h in range(2):
            if np.any(copy_pts[h] != 0):
                wrist = copy_pts[h, 0].copy()
                copy_pts[h] = (copy_pts[h] - wrist) * scale + wrist
                
        # 3. Gaussian coordinate noise
        noise = rng.normal(0, 0.005, size=copy_pts.shape).astype(np.float32)
        for h in range(2):
            if np.any(copy_pts[h] != 0):
                copy_pts[h] += noise[h]
                
        aug_vectors.append(copy_pts.reshape(-1).tolist())
        
    return aug_vectors


def main():
    if not DATASET_2_DIR.exists():
        logger.error(f"Dataset 2 directory not found at: {DATASET_2_DIR}")
        return

    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    logger.info("Initializing MediaPipe HandLandmarker...")
    base_options = python.BaseOptions(model_asset_path=str(MODEL_TASK_PATH))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.25,
        min_hand_presence_confidence=0.25
    )
    detector = vision.HandLandmarker.create_from_options(options)

    rng = np.random.default_rng(42)
    letter_dirs = sorted([p for p in DATASET_2_DIR.iterdir() if p.is_dir()])
    logger.info(f"Processing {len(letter_dirs)} letter directories in {DATASET_2_DIR}...")

    total_images_processed = 0
    total_landmarks_extracted = 0
    per_letter_counts = {}

    for ldir in letter_dirs:
        letter = ldir.name.upper()
        imgs = sorted(list(ldir.glob('*.jpg')) + list(ldir.glob('*.png')) + list(ldir.glob('*.jpeg')))
        if not imgs:
            continue

        letter_output_dir = OUTPUT_DIR / letter
        letter_output_dir.mkdir(parents=True, exist_ok=True)

        # Save stratified splits (70% train, 15% val, 15% test) by original photo index
        n_imgs = len(imgs)
        val_count = max(1, (round(n_imgs * 0.15)))
        test_count = max(1, (round(n_imgs * 0.15)))
        train_count = n_imgs - val_count - test_count

        shuffled_indices = list(range(n_imgs))
        rng.shuffle(shuffled_indices)

        train_set = set(shuffled_indices[:train_count])
        val_set = set(shuffled_indices[train_count:train_count + val_count])
        test_set = set(shuffled_indices[train_count + val_count:])

        train_frames, val_frames, test_frames = [], [], []

        for idx, img_p in enumerate(imgs):
            total_images_processed += 1
            cv_img = cv2.imread(str(img_p))
            if cv_img is None:
                continue

            # Resize to max 1280px for reliable detection
            h, w = cv_img.shape[:2]
            max_dim = max(h, w)
            if max_dim > 1280:
                scale = 1280.0 / max_dim
                cv_img = cv2.resize(cv_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

            rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_image)

            vec = get_landmark_vector(result)
            if vec is not None:
                if idx in train_set:
                    train_frames.append(vec)
                    # Add 5 augmentations to training
                    train_frames.extend(augment_vector(vec, rng, copies=5))
                elif idx in val_set:
                    val_frames.append(vec)
                    val_frames.extend(augment_vector(vec, rng, copies=2))
                else:
                    test_frames.append(vec)

        # Write split JSONs
        for split_name, split_frames in [('dataset2_train', train_frames), ('dataset2_val', val_frames), ('dataset2_test', test_frames)]:
            if split_frames:
                out_json = letter_output_dir / f'{split_name}.json'
                payload = {
                    'letter': letter,
                    'session_id': f'{split_name}_{letter}',
                    'source': f'dataset_2_{split_name}',
                    'extracted_frames_count': len(split_frames),
                    'frames': split_frames
                }
                out_json.write_text(json.dumps(payload, indent=2), encoding='utf-8')

        # Remove old monolithic json if exists
        old_mono = letter_output_dir / 'custom_dataset2.json'
        if old_mono.exists():
            old_mono.unlink()

        total_extracted = len(train_frames) + len(val_frames) + len(test_frames)
        total_landmarks_extracted += total_extracted
        per_letter_counts[letter] = {
            'train': len(train_frames),
            'val': len(val_frames),
            'test': len(test_frames),
            'total': total_extracted
        }
        logger.info(f"Letter {letter:2s}: {len(imgs)} photos -> Train={len(train_frames)}, Val={len(val_frames)}, Test={len(test_frames)}")

    logger.info("=" * 70)
    logger.info(f"DONE! Processed {total_images_processed} photos across {len(per_letter_counts)} classes.")
    logger.info(f"Total extracted & augmented landmark frames: {total_landmarks_extracted}")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
