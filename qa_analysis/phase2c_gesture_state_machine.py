"""
SignBridge QA Analysis - Phase 2C: Gesture State Machine Verification
Evaluates:
1. Lifecycle transitions: NEUTRAL -> ARMED -> CAPTURING -> COOLDOWN -> ARMED.
2. Fast commit mechanics: Instant commit (>= 0.92), 3-frame rapid commit.
3. Jitter & dropouts: 1-frame jitter tolerance, 2-frame dropout reset.
4. Cooldown & Re-arming: 250ms lock, 350ms steady hand motion (< 0.015), MAX_COOLDOWN failsafe (1500ms).
5. Double-commit prevention & same-sign repeat: RELEASE_FRAMES (4 frames) lockout.
6. Fingerspelling simulation: 'H-E-L-L-O' with consecutive repeated letters ('L').
"""

import sys
import json
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = BASE_DIR / 'backend'
sys.path.insert(0, str(BACKEND_DIR))

from services.gesture_state_machine import GestureStateMachine
from services.gesture_config import (
    NEUTRAL, ARMED, CAPTURING, COOLDOWN,
    INSTANT_COMMIT_CONFIDENCE, STABILITY_FRAMES_K,
    MIN_LOCK_MS, STEADY_MS, STEADY_MOTION_THRESHOLD,
    MAX_COOLDOWN_MS, RELEASE_FRAMES
)

def make_dummy_landmarks(wrist_x=0.5, wrist_y=0.5):
    return [{'x': wrist_x, 'y': wrist_y, 'z': 0.0} for _ in range(42)]

def run_phase2c():
    print("=" * 70)
    print("  PHASE 2C: GESTURE STATE MACHINE DEEP VERIFICATION")
    print("=" * 70)

    results = {}

    # 1. Instant Commit at >= 0.92
    print("\n1. Testing Instant Commit (>= 0.92 confidence)...")
    committed = []
    def on_commit(sign, meta):
        committed.append((sign, meta))

    sm = GestureStateMachine(on_commit=on_commit, require_neutral_between_signs=False)
    dummy_lm = make_dummy_landmarks()
    
    # Send 1 frame with confidence 0.95 (>= 0.92)
    pred_instant = {
        'letter': 'A',
        'confidence': 0.95,
        'margin': 0.30,
        'centroid_distance': 2.0,
        'max_centroid_distance': 11.5,
        'rejected': False
    }
    out = sm.update(now_ms=1000, landmarks=dummy_lm, prediction=pred_instant)
    state = out.get('state')
    lock = out.get('locked_sign')
    
    results["instant_commit"] = {
        "expected_state": COOLDOWN,
        "actual_state": state,
        "locked_sign": lock,
        "committed_count": len(committed),
        "committed_sign": committed[0][0] if committed else None,
        "pass": (state == COOLDOWN and len(committed) == 1 and committed[0][0] == "A")
    }
    print(f"Instant commit result: State={state}, Committed={len(committed)} (Pass: {results['instant_commit']['pass']})")

    # 2. 3-Frame Rapid Commit (STABILITY_FRAMES_K = 3) at moderate confidence (0.80)
    print("\n2. Testing 3-Frame Rapid Commit...")
    committed.clear()
    sm2 = GestureStateMachine(on_commit=on_commit, require_neutral_between_signs=False)
    pred_mod = {
        'letter': 'B',
        'confidence': 0.80,
        'margin': 0.20,
        'centroid_distance': 3.0,
        'max_centroid_distance': 11.5,
        'rejected': False
    }
    
    o1 = sm2.update(now_ms=1000, landmarks=dummy_lm, prediction=pred_mod)
    o2 = sm2.update(now_ms=1033, landmarks=dummy_lm, prediction=pred_mod)
    o3 = sm2.update(now_ms=1066, landmarks=dummy_lm, prediction=pred_mod)
    
    results["three_frame_commit"] = {
        "frame_1_state": o1['state'],
        "frame_2_state": o2['state'],
        "frame_3_state": o3['state'],
        "locked_sign": o3['locked_sign'],
        "committed_count": len(committed),
        "pass": (o1['state'] == CAPTURING and o2['state'] == CAPTURING and o3['state'] == COOLDOWN and len(committed) == 1)
    }
    print(f"3-Frame commit: s1={o1['state']}, s2={o2['state']}, s3={o3['state']} -> Committed: {len(committed)} (Pass: {results['three_frame_commit']['pass']})")

    # 3. 1-Frame Jitter Tolerance vs 2-Frame Dropout Reset
    print("\n3. Testing Jitter Tolerance (1 dropped frame allowed) vs Dropout Reset...")
    committed.clear()
    sm3 = GestureStateMachine(on_commit=on_commit, require_neutral_between_signs=False)
    pred_c = {'letter': 'C', 'confidence': 0.80, 'margin': 0.2, 'centroid_distance': 2.0, 'max_centroid_distance': 11.5, 'rejected': False}
    pred_glitch = {'letter': '?', 'confidence': 0.0, 'rejected': True}
    
    sm3.update(now_ms=1000, landmarks=dummy_lm, prediction=pred_c)
    sm3.update(now_ms=1033, landmarks=dummy_lm, prediction=pred_c)
    sm3.update(now_ms=1066, landmarks=dummy_lm, prediction=pred_glitch)
    o_jitter = sm3.update(now_ms=1100, landmarks=dummy_lm, prediction=pred_c)
    
    # 2 consecutive dropouts
    committed.clear()
    sm4 = GestureStateMachine(on_commit=on_commit, require_neutral_between_signs=False)
    pred_d = {'letter': 'D', 'confidence': 0.80, 'margin': 0.2, 'centroid_distance': 2.0, 'max_centroid_distance': 11.5, 'rejected': False}
    sm4.update(now_ms=1000, landmarks=dummy_lm, prediction=pred_d)
    sm4.update(now_ms=1033, landmarks=dummy_lm, prediction=pred_d)
    sm4.update(now_ms=1066, landmarks=dummy_lm, prediction=pred_glitch)
    sm4.update(now_ms=1100, landmarks=dummy_lm, prediction=pred_glitch)
    o_reset = sm4.update(now_ms=1133, landmarks=dummy_lm, prediction=pred_d)
    
    results["jitter_and_dropout"] = {
        "one_frame_glitch_tolerated": (o_jitter['state'] == COOLDOWN and len(committed) == 1),
        "two_frame_dropout_reset_streak": (o_reset['state'] == CAPTURING),
        "pass": ((o_jitter['state'] == COOLDOWN) and (o_reset['state'] == CAPTURING))
    }
    print(f"Jitter tolerated: {results['jitter_and_dropout']['one_frame_glitch_tolerated']} | 2-Frame reset: {results['jitter_and_dropout']['two_frame_dropout_reset_streak']}")

    # 4. Hold Steady Dynamic Re-arming (350ms steady motion < 0.015)
    print("\n4. Testing Dynamic Re-arming (350ms steady)...")
    committed.clear()
    sm5 = GestureStateMachine(on_commit=on_commit, require_neutral_between_signs=False)
    pred_e = {'letter': 'E', 'confidence': 0.95, 'margin': 0.3, 'centroid_distance': 2.0, 'max_centroid_distance': 11.5, 'rejected': False}
    
    sm5.update(now_ms=1000, landmarks=dummy_lm, prediction=pred_e)
    # t=1100ms (< 250ms lock)
    o_early = sm5.update(now_ms=1100, landmarks=dummy_lm, prediction=pred_e)
    # t=1300ms (> 250ms, steady for 200ms)
    o_mid = sm5.update(now_ms=1300, landmarks=dummy_lm, prediction=pred_e)
    # t=1450ms (steady for 450ms >= 350ms) -> Re-arms to ARMED!
    o_rearm = sm5.update(now_ms=1450, landmarks=dummy_lm, prediction=pred_e)
    
    results["rearming"] = {
        "early_lock_state": o_early['state'],
        "rearmed_state": o_rearm['state'],
        "pass": (o_early['state'] == COOLDOWN and o_rearm['state'] == ARMED)
    }
    print(f"Early lock at 1100ms: {o_early['state']} -> Re-armed at 1450ms: {o_rearm['state']} (Pass: {results['rearming']['pass']})")

    # 5. Whole Word Fingerspelling Simulation: 'H-E-L-L-O' (Consecutive repeated 'L')
    print("\n5. Simulating Whole Word Fingerspelling ('H-E-L-L-O' with repeated 'L')...")
    committed.clear()
    sm_spell = GestureStateMachine(on_commit=on_commit, require_neutral_between_signs=False)
    
    t = 1000
    word_letters = ['H', 'E', 'L', 'L', 'O']
    for letter in word_letters:
        pred_let = {'letter': letter, 'confidence': 0.95, 'margin': 0.3, 'centroid_distance': 2.0, 'max_centroid_distance': 11.5, 'rejected': False}
        sm_spell.update(now_ms=t, landmarks=dummy_lm, prediction=pred_let)
        t += 300  # Minimum lock time
        
        # Advance release counter for repeated letter handling
        # Moving hands simulates 4 release frames
        for f in range(RELEASE_FRAMES + 1):
            t += 33
            moving_lm = make_dummy_landmarks(wrist_x=0.5 + 0.05 * (f + 1))
            sm_spell.update(now_ms=t, landmarks=moving_lm, prediction=None)
            
        # Hold steady 350ms
        t += 350
        steady_lm = make_dummy_landmarks(wrist_x=0.7)
        sm_spell.update(now_ms=t, landmarks=steady_lm, prediction=None)
        
    spelled = "".join([c[0] for c in committed])
    results["fingerspelling_hello"] = {
        "target": "HELLO",
        "spelled": spelled,
        "committed_sequence": [c[0] for c in committed],
        "pass": (spelled == "HELLO")
    }
    print(f"Spelled result: '{spelled}' (Target: 'HELLO') -> Pass: {results['fingerspelling_hello']['pass']}")

    out_file = Path(__file__).resolve().parent / "phase2c_results.json"
    out_file.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(f"\nPhase 2C results written to {out_file}")

if __name__ == "__main__":
    run_phase2c()
