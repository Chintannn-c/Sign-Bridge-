"""
Unit test for realistic human fingerspelling at ~30 FPS (~33ms/frame).

Validates:
1. Held sign commits once; holding longer does NOT double-commit.
2. Released sign allows consecutive duplicate letter commit (e.g. LL in HELLO, PP in APPLE).
3. Words with repeats: HELLO, APPLE, GOOD, BALL.
4. Words without repeats: WORLD.
5. Supported release mechanisms: transition gap (brief low-confidence movement), hand drop, and neutral palms.
"""

import unittest
from services.gesture_state_machine import GestureStateMachine


def make_landmarks(x=0.5, y=0.5):
    return [{'x': x, 'y': y, 'z': 0.0} for _ in range(21)]


def simulate_word(word: str, release_mode: str = "transition_gap"):
    committed = []
    sm = GestureStateMachine(
        on_commit=lambda letter, meta: committed.append(letter),
        require_neutral_between_signs=False,
    )
    t = 1000
    dt = 33

    for target_char in word:
        # Sign Hold Phase: 450ms (~14 frames)
        for f in range(14):
            t += dt
            x_pos = 0.5 + (0.02 if f == 0 else 0.0)
            lm = make_landmarks(x=x_pos, y=0.5)
            pred = {
                'letter': target_char,
                'confidence': 0.94,
                'margin': 0.30,
                'centroid_distance': 2.0,
                'max_centroid_distance': 11.5,
                'rejected': False,
            }
            sm.update(now_ms=t, landmarks=lm, prediction=pred, is_neutral=False)

        # Release / Transition Phase (200ms ~ 6 frames)
        if release_mode == "transition_gap":
            for f in range(6):
                t += dt
                lm = make_landmarks(x=0.5 + 0.03 * ((f % 2) * 2 - 1), y=0.5)
                pred_low = {
                    'letter': '?',
                    'confidence': 0.35,
                    'rejected': True,
                    'rejection_reason': 'low_confidence',
                }
                sm.update(now_ms=t, landmarks=lm, prediction=pred_low, is_neutral=False)
        elif release_mode == "hand_drop":
            for _ in range(6):
                t += dt
                sm.update(now_ms=t, landmarks=None, prediction=None, is_neutral=False)
        elif release_mode == "neutral_gesture":
            for _ in range(6):
                t += dt
                lm = make_landmarks(x=0.5, y=0.5)
                sm.update(now_ms=t, landmarks=lm, prediction=None, is_neutral=True)

    return "".join(committed)


class TestRealisticFingerspelling(unittest.TestCase):

    def test_fingerspelling_hello_transition_gap(self):
        result = simulate_word("HELLO", release_mode="transition_gap")
        self.assertEqual(result, "HELLO")

    def test_fingerspelling_apple_transition_gap(self):
        result = simulate_word("APPLE", release_mode="transition_gap")
        self.assertEqual(result, "APPLE")

    def test_fingerspelling_good_transition_gap(self):
        result = simulate_word("GOOD", release_mode="transition_gap")
        self.assertEqual(result, "GOOD")

    def test_fingerspelling_ball_transition_gap(self):
        result = simulate_word("BALL", release_mode="transition_gap")
        self.assertEqual(result, "BALL")

    def test_fingerspelling_world_transition_gap(self):
        result = simulate_word("WORLD", release_mode="transition_gap")
        self.assertEqual(result, "WORLD")

    def test_fingerspelling_hello_hand_drop(self):
        result = simulate_word("HELLO", release_mode="hand_drop")
        self.assertEqual(result, "HELLO")

    def test_fingerspelling_hello_neutral_gesture(self):
        result = simulate_word("HELLO", release_mode="neutral_gesture")
        self.assertEqual(result, "HELLO")


if __name__ == '__main__':
    unittest.main()
