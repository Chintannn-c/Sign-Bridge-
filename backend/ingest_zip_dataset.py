"""
SignBridge — Custom Dataset.zip Ingestion & Feature Extraction
Reads C:\\Users\\sharm\\Downloads\\Dataset.zip, extracts 42 MediaPipe HandLandmarks
for all 305 photos across 26 letters (A-Z), and saves normalized landmark JSONs
into backend/dataset_collected/<LETTER>/.
"""

import os
import io
import json
import logging
import zipfile
from pathlib import Path
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / 'dataset_collected'
MODEL_TASK_PATH = BASE_DIR / 'models' / 'hand_landmarker.task'
ZIP_PATH = Path(r'C:\Users\sharm\Downloads\Dataset.zip')


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


def augment_landmark_vector(vec, rng):
    """Generate subtle kinematic rotations and noise for a static landmark vector."""
    arr = np.array(vec, dtype=np.float32).reshape(2, 21, 3)
    aug = arr.copy()
    
    # 2D in-plane rotation
    angle = rng.uniform(-0.12, 0.12)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    for h in range(2):
        if np.any(aug[h] != 0):
            # Center around wrist (landmark 0)
            wrist = aug[h, 0, :2].copy()
            coords = aug[h, :, :2] - wrist
            rot_x = coords[:, 0] * cos_a - coords[:, 1] * sin_a
            rot_y = coords[:, 0] * sin_a + coords[:, 1] * cos_a
            aug[h, :, 0] = rot_x + wrist[0]
            aug[h, :, 1] = rot_y + wrist[1]
            
    # Gaussian coordinate jitter
    noise = rng.normal(0, 0.005, size=aug.shape).astype(np.float32)
    aug += noise
    return aug.reshape(-1).tolist()


def ingest_zip():
    if not ZIP_PATH.exists():
        logger.error(f"Dataset.zip not found at: {ZIP_PATH}")
        return 0

    logger.info(f"Loading MediaPipe HandLandmarker from {MODEL_TASK_PATH}...")
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    base_options = python.BaseOptions(model_asset_path=str(MODEL_TASK_PATH))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.25,
        min_hand_presence_confidence=0.25
    )
    detector = vision.HandLandmarker.create_from_options(options)

    rng = np.random.default_rng(42)
    total_images_processed = 0
    total_landmarks_saved = 0

    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        img_names = [n for n in z.namelist() if n.lower().endswith(('.jpg', '.jpeg', '.png'))]
        logger.info(f"Found {len(img_names)} images in {ZIP_PATH.name}")

        # Group by letter folder
        letters_map = {}
        for name in img_names:
            parts = name.strip('/').split('/')
            if len(parts) >= 2:
                folder_name = parts[-2].upper()
                if len(folder_name) == 1 and folder_name.isalpha():
                    letters_map.setdefault(folder_name, []).append(name)

        logger.info(f"Found {len(letters_map)} alphabet classes: {sorted(letters_map.keys())}")

        for letter, file_list in sorted(letters_map.items()):
            target_dir = OUTPUT_DIR / letter
            target_dir.mkdir(parents=True, exist_ok=True)
            
            letter_samples = []

            for f_name in file_list:
                try:
                    img_bytes = z.read(f_name)
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if cv_img is None:
                        continue

                    # Resize to 1280px max dimension for fast, accurate detection
                    h, w = cv_img.shape[:2]
                    scale = 1280.0 / max(h, w)
                    resized = cv2.resize(cv_img, (int(w * scale), int(h * scale)))
                    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                    result = detector.detect(mp_image)
                    if result.hand_landmarks:
                        vec = get_landmark_vector_from_result(result)
                        if vec is not None:
                            total_images_processed += 1
                            letter_samples.append(vec)
                            # Generate 5 augmented variations
                            for _ in range(5):
                                aug_vec = augment_landmark_vector(vec, rng)
                                letter_samples.append(aug_vec)
                except Exception as e:
                    logger.warning(f"Failed to process {f_name}: {e}")

            if letter_samples:
                session_id = "session_custom_photos_zip"
                out_path = target_dir / f"{session_id}.json"
                out_path.write_text(json.dumps({
                    "letter": letter,
                    "session_id": session_id,
                    "source": "Custom_Dataset_Zip",
                    "count": len(letter_samples),
                    "landmarks": letter_samples
                }, indent=2), encoding='utf-8')
                total_landmarks_saved += len(letter_samples)
                logger.info(f"  Letter '{letter}': {len(file_list)} images -> {len(letter_samples)} landmark samples saved to {out_path.name}")

    logger.info("=" * 60)
    logger.info(f"Ingestion complete! Processed {total_images_processed} images -> {total_landmarks_saved} landmark samples created.")
    logger.info("=" * 60)
    return total_landmarks_saved


if __name__ == '__main__':
    ingest_zip()
