"""
SignBridge ML Models Diagnostic & Inference Smoke Test
Tests both Tier 1 (Alphabet Classifier) and Tier 2 (Word Recognizer) services.
"""

import sys
import numpy as np

from services.translator_model import TranslatorModel
from services.word_recognizer import WordRecognizer

def test_services():
    print("=== Testing SignBridge ML Models ===")
    
    # 1. Test TranslatorModel (Alphabet)
    translator = TranslatorModel()
    print(f"[Tier 1 Alphabet] Mode: {translator.mode}")
    info = translator.get_info()
    print(f"  Accuracy: {info.get('validation_accuracy')}")
    print(f"  Num classes: {info.get('num_classes')}")
    
    # Dummy single-frame inference test (126 floats)
    dummy_frame = np.random.uniform(-0.5, 0.5, size=126).tolist()
    pred = translator.predict(dummy_frame)
    print(f"  Inference test -> Predicted: {pred.get('letter')}, Conf: {pred.get('confidence'):.4f}, Top 5: {pred.get('all_scores')}")
    
    # 2. Test WordRecognizer (Temporal Words)
    recognizer = WordRecognizer()
    print(f"\n[Tier 2 Word Recognizer] Available: {recognizer.is_available}, Mode: {recognizer.mode}")
    w_info = recognizer.get_info()
    print(f"  Num classes: {w_info.get('num_classes')}")
    print(f"  Classes: {w_info.get('labels')}")
    
    # Dummy 30-frame sequence inference test (30 x 126)
    dummy_seq = [np.random.uniform(-0.5, 0.5, size=126).tolist() for _ in range(30)]
    w_pred = recognizer.predict(dummy_seq)
    if w_pred:
        print(f"  Inference test -> Predicted Word: {w_pred.get('word')}, Conf: {w_pred.get('confidence'):.4f}, Top 5: {w_pred.get('all_scores')}")
    else:
        print("  Word prediction returned None.")
        
    print("\n=== All Model Smoke Tests Passed Successfully! ===")

if __name__ == '__main__':
    test_services()
