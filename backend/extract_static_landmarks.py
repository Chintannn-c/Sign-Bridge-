"""
SignBridge — Unified Batch Static Landmark Extractor (v2)
Processes ALL available ISL image datasets using MediaPipe HandLandmarker:
  1. RealSign (Training, Validation, Testing, Letters) — NO cap
  2. ISL Self-Made (a-z, 0-9) — NO cap
  3. Mendeley ISL (auto-extract from zip if needed)
  4. ISL_Landmarks reference files

Outputs:
  backend/dataset_collected/<LETTER>/session_<source>_<id>.json
"""

import os
import sys
import json
import logging
import zipfile
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


def process_realsign_dataset(detector, mp):
    """Processes ALL RealSign Training, Validation, and Testing images — NO cap."""
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

        logger.info(f"Processing RealSign split: {split} (ALL images, no cap)...")
        for letter_dir in sorted(split_dir.iterdir()):
            if not letter_dir.is_dir() or letter_dir.name.startswith('.'):
                continue

            letter = letter_dir.name.upper()
            target_dir = OUTPUT_DIR / letter
            target_dir.mkdir(parents=True, exist_ok=True)

            # Check if already extracted with sufficient data
            existing_file = target_dir / f"realsign_{split.lower()}.json"
            if existing_file.exists():
                try:
                    existing_data = json.loads(existing_file.read_text(encoding='utf-8'))
                    existing_count = len(existing_data.get('frames', []))
                    img_count = len(list(letter_dir.glob('*.jpg')) + list(letter_dir.glob('*.png')))
                    if existing_count >= img_count * 0.9:  # Already have 90%+ extracted
                        logger.info(f"  [RealSign {split}] {letter}: Already extracted ({existing_count} frames), skipping")
                        total_extracted += existing_count
                        continue
                except Exception:
                    pass

            img_paths = sorted(list(letter_dir.glob('*.jpg')) + list(letter_dir.glob('*.png')))

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
                session_id = f"realsign_{split.lower()}"
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


def process_self_made_dataset(detector, mp):
    """Processes ALL ISL self-made dataset images (0-9 and a-z) — NO cap."""
    self_made_dir = DATASET_DIR / 'ISL self made dataset'
    if not self_made_dir.exists():
        logger.warning(f"ISL self made dataset not found at {self_made_dir}")
        return 0

    total_extracted = 0
    logger.info("Processing ISL self made dataset (ALL images, no cap)...")

    for class_dir in sorted(self_made_dir.iterdir()):
        if not class_dir.is_dir() or class_dir.name.startswith('.'):
            continue

        label = class_dir.name.upper()
        target_dir = OUTPUT_DIR / label
        target_dir.mkdir(parents=True, exist_ok=True)

        # Check if already extracted
        existing_file = target_dir / "selfmade.json"
        if existing_file.exists():
            try:
                existing_data = json.loads(existing_file.read_text(encoding='utf-8'))
                existing_count = len(existing_data.get('frames', []))
                img_count = len(list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png')))
                if existing_count >= img_count * 0.9:
                    logger.info(f"  [Self-Made] {label}: Already extracted ({existing_count} frames), skipping")
                    total_extracted += existing_count
                    continue
            except Exception:
                pass

        img_paths = sorted(list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png')))

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
            session_id = "selfmade"
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


def extract_mendeley_zip():
    """Extract Mendeley ISL zip if not already done."""
    mendeley_dir = DATASET_DIR / 'Mendeley_ISL'
    zip_path = mendeley_dir / 'ISL_Dataset.zip'
    extract_dir = mendeley_dir / 'extracted'

    if not zip_path.exists():
        logger.warning(f"Mendeley zip not found at {zip_path}")
        return False

    # Check if zip is valid and has content
    try:
        with zipfile.ZipFile(str(zip_path), 'r') as zf:
            file_list = zf.namelist()
            if not file_list:
                logger.warning("Mendeley zip is empty")
                return False
            
            # Check if already extracted
            if extract_dir.exists() and any(extract_dir.iterdir()):
                logger.info("Mendeley already extracted, skipping")
                return True

            logger.info(f"Extracting Mendeley ISL dataset ({len(file_list)} files)...")
            extract_dir.mkdir(parents=True, exist_ok=True)
            zf.extractall(str(extract_dir))
            logger.info(f"Mendeley extracted to {extract_dir}")
            return True
    except zipfile.BadZipFile:
        logger.warning(f"Mendeley zip is corrupt/invalid: {zip_path}")
        return False


def process_mendeley_dataset(detector, mp):
    """Process extracted Mendeley ISL images."""
    mendeley_extracted = DATASET_DIR / 'Mendeley_ISL' / 'extracted'

    if not mendeley_extracted.exists():
        logger.info("Mendeley not extracted, attempting extraction...")
        if not extract_mendeley_zip():
            logger.warning("Mendeley extraction failed, skipping")
            return 0

    # Scan for image directories
    total_extracted = 0
    image_dirs = []

    # Look for class directories (letter names or digit names)
    for item in sorted(mendeley_extracted.rglob('*')):
        if item.is_dir():
            imgs = list(item.glob('*.jpg')) + list(item.glob('*.png')) + list(item.glob('*.jpeg'))
            if imgs:
                image_dirs.append(item)

    if not image_dirs:
        # Maybe flat structure - look for images directly
        imgs = list(mendeley_extracted.glob('*.jpg')) + list(mendeley_extracted.glob('*.png'))
        if imgs:
            logger.info(f"Mendeley has flat structure with {len(imgs)} images")
        else:
            logger.warning("No images found in Mendeley extracted directory")
        return 0

    logger.info(f"Processing Mendeley ISL dataset ({len(image_dirs)} class directories)...")

    for class_dir in image_dirs:
        label = class_dir.name.upper().strip()
        # Normalize label
        if len(label) == 1 and (label.isalpha() or label.isdigit()):
            pass  # Good label
        elif label.lower() in [str(i) for i in range(10)]:
            label = label
        else:
            logger.info(f"  [Mendeley] Skipping non-standard label: {label}")
            continue

        target_dir = OUTPUT_DIR / label
        target_dir.mkdir(parents=True, exist_ok=True)

        # Check if already extracted
        existing_file = target_dir / "mendeley.json"
        if existing_file.exists():
            try:
                existing_data = json.loads(existing_file.read_text(encoding='utf-8'))
                if len(existing_data.get('frames', [])) > 0:
                    logger.info(f"  [Mendeley] {label}: Already extracted, skipping")
                    total_extracted += len(existing_data['frames'])
                    continue
            except Exception:
                pass

        img_paths = sorted(list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpeg')))

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
            out_file = target_dir / "mendeley.json"
            out_file.write_text(json.dumps({
                "letter": label,
                "session_id": "mendeley",
                "source": "Mendeley_ISL",
                "frames": frames
            }, indent=2), encoding='utf-8')
            total_extracted += len(frames)
            logger.info(f"  [Mendeley] Class {label}: {len(frames)}/{len(img_paths)} landmarks -> mendeley.json")

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

    logger.info("=" * 70)
    logger.info("=== SignBridge Unified Static Landmark Extraction (v2) ===")
    logger.info("=== Processing ALL images from ALL datasets, NO caps ===")
    logger.info("=" * 70)

    # 1. Canonical reference letters
    l_count = process_realsign_letters(detector, mp)

    # 2. RealSign Training/Validation/Testing — ALL images
    r_count = process_realsign_dataset(detector, mp)

    # 3. Self-made dataset — ALL images
    s_count = process_self_made_dataset(detector, mp)

    # 4. Mendeley dataset (extract zip + process)
    m_count = process_mendeley_dataset(detector, mp)

    detector.close()

    logger.info("=" * 70)
    logger.info(f"=== Extraction Complete! ===")
    logger.info(f"  Canonical Letters: {l_count}")
    logger.info(f"  RealSign Splits:   {r_count}")
    logger.info(f"  Self-Made:         {s_count}")
    logger.info(f"  Mendeley:          {m_count}")
    logger.info(f"  TOTAL:             {l_count + r_count + s_count + m_count}")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
