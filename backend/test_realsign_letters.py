"""
Verification Script: Test ISL Gesture Recognition Model on RealSign Canonical Letters (A-Z).
Processes each of the 26 letter images from dataset/RealSign-Indian-Sign-Language-Dataset-main/Letters,
extracts MediaPipe landmarks, feeds them to TranslatorModel, and reports the predicted letter and confidence.
"""

import os
import sys
import json
import logging
from pathlib import Path
import cv2
import numpy as np

# Ensure backend root is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from services.translator_model import TranslatorModel, ISL_LABELS
from extract_static_landmarks import get_landmark_vector_from_result

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

LETTERS_DIR = SCRIPT_DIR.parent / 'dataset' / 'RealSign-Indian-Sign-Language-Dataset-main' / 'Letters'
MODEL_TASK_PATH = SCRIPT_DIR / 'models' / 'hand_landmarker.task'

def test_canonical_letters():
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    if not LETTERS_DIR.exists():
        logger.error(f"Letters directory not found at: {LETTERS_DIR}")
        return

    # Initialize HandLandmarker
    base_options = python.BaseOptions(model_asset_path=str(MODEL_TASK_PATH))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.2,
        min_hand_presence_confidence=0.2
    )
    detector = vision.HandLandmarker.create_from_options(options)

    # Initialize Translator Model
    translator = TranslatorModel()
    logger.info(f"Loaded Translator Model in mode: {translator.mode}")

    correct = 0
    total = 0
    results = []

    print("\n" + "=" * 70)
    print(f"{'Letter':<8} | {'Status':<10} | {'Predicted':<10} | {'Confidence':<12} | {'Top 3 Predictions'}")
    print("=" * 70)

    for img_path in sorted(LETTERS_DIR.glob('*.jpg')):
        expected_letter = img_path.stem.upper()
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"{expected_letter:<8} | {'ERROR':<10} | {'-':<10} | {'-':<12} | Could not read image")
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detection = detector.detect(mp_image)

        if not detection.hand_landmarks:
            print(f"{expected_letter:<8} | {'NO HAND':<10} | {'-':<10} | {'-':<12} | No hands detected by MediaPipe")
            total += 1
            results.append({'letter': expected_letter, 'status': 'no_hands', 'predicted': None, 'confidence': 0.0})
            continue

        landmarks = get_landmark_vector_from_result(detection)
        if not landmarks:
            print(f"{expected_letter:<8} | {'NO VECTOR':<10} | {'-':<10} | {'-':<12} | Could not format landmarks")
            total += 1
            results.append({'letter': expected_letter, 'status': 'no_vector', 'predicted': None, 'confidence': 0.0})
            continue

        # Predict
        prediction = translator.predict(landmarks)
        pred_letter = prediction.get('letter')
        confidence = prediction.get('confidence', 0.0)
        all_scores = prediction.get('all_scores', {})

        top3 = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        top3_str = ", ".join([f"{k}: {v*100:.1f}%" for k, v in top3])

        is_correct = (pred_letter == expected_letter)
        if is_correct:
            correct += 1
            status = "MATCH"
        else:
            status = "MISMATCH"

        total += 1
        results.append({
            'letter': expected_letter,
            'status': status,
            'predicted': pred_letter,
            'confidence': confidence,
            'top3': top3_str
        })

        print(f"{expected_letter:<8} | {status:<10} | {str(pred_letter):<10} | {confidence*100:>8.2f}%    | {top3_str}")

    detector.close()

    accuracy = (correct / total * 100) if total > 0 else 0
    print("=" * 70)
    print(f"Final RealSign Canonical Letters Accuracy: {correct}/{total} ({accuracy:.2f}%)\n")

if __name__ == '__main__':
    test_canonical_letters()
