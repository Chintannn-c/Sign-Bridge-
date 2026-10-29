"""
Generates synthetic negative / idle sequences for SignBridge word recognizer.
Creates balanced samples for the 'NONE' class:
  - 10 static two open hands with realistic camera noise
  - 10 static single open hand
  - 10 hands entering the frame from below
  - 10 hands leaving the frame
  - 10 resting hands on desk / lap
"""

import json
from pathlib import Path
import numpy as np

def generate_none_dataset():
    none_dir = Path(__file__).resolve().parent / 'dataset_words' / 'NONE'
    none_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    count = 0

    # 1. Two static open hands
    for i in range(10):
        seq = np.zeros((30, 2, 21, 3), dtype=np.float32)
        seq[:, 0, 0] = [0.40, 0.65, 0.0]
        seq[:, 1, 0] = [0.60, 0.65, 0.0]
        for j in range(1, 21):
            seq[:, 0, j] = seq[:, 0, 0] + np.array([-0.03 * (j % 5), -0.04 * (j // 5), 0.0])
            seq[:, 1, j] = seq[:, 1, 0] + np.array([0.03 * (j % 5), -0.04 * (j // 5), 0.0])
        seq += rng.normal(0, 0.002, seq.shape).astype(np.float32)
        payload = {
            'word': 'NONE',
            'session_id': f'idle_static_both_{i}',
            'source': 'synthetic_idle_generator',
            'sequence_length': 30,
            'frames': seq.reshape(30, 126).tolist()
        }
        (none_dir / f'idle_static_both_{i:02d}.json').write_text(json.dumps(payload, indent=2))
        count += 1

    # 2. Single open hand
    for i in range(10):
        seq = np.zeros((30, 2, 21, 3), dtype=np.float32)
        h_idx = i % 2
        seq[:, h_idx, 0] = [0.50, 0.65, 0.0]
        for j in range(1, 21):
            seq[:, h_idx, j] = seq[:, h_idx, 0] + np.array([0.03 * (j % 5), -0.04 * (j // 5), 0.0])
        seq += rng.normal(0, 0.002, seq.shape).astype(np.float32)
        payload = {
            'word': 'NONE',
            'session_id': f'idle_single_{i}',
            'source': 'synthetic_idle_generator',
            'sequence_length': 30,
            'frames': seq.reshape(30, 126).tolist()
        }
        (none_dir / f'idle_single_{i:02d}.json').write_text(json.dumps(payload, indent=2))
        count += 1

    # 3. Hands entering frame
    for i in range(10):
        seq = np.zeros((30, 2, 21, 3), dtype=np.float32)
        for t in range(10, 30):
            y_pos = 1.0 - (t - 10) * 0.02
            seq[t, 0, 0] = [0.40, y_pos, 0.0]
            seq[t, 1, 0] = [0.60, y_pos, 0.0]
            for j in range(1, 21):
                seq[t, 0, j] = seq[t, 0, 0] + np.array([-0.02 * (j % 5), -0.03 * (j // 5), 0.0])
                seq[t, 1, j] = seq[t, 1, 0] + np.array([0.02 * (j % 5), -0.03 * (j // 5), 0.0])
        seq += rng.normal(0, 0.002, seq.shape).astype(np.float32)
        seq[:10] = 0.0
        payload = {
            'word': 'NONE',
            'session_id': f'idle_entering_{i}',
            'source': 'synthetic_idle_generator',
            'sequence_length': 30,
            'frames': seq.reshape(30, 126).tolist()
        }
        (none_dir / f'idle_entering_{i:02d}.json').write_text(json.dumps(payload, indent=2))
        count += 1

    # 4. Hands leaving frame
    for i in range(10):
        seq = np.zeros((30, 2, 21, 3), dtype=np.float32)
        for t in range(0, 20):
            y_pos = 0.65 + t * 0.02
            seq[t, 0, 0] = [0.40, y_pos, 0.0]
            seq[t, 1, 0] = [0.60, y_pos, 0.0]
            for j in range(1, 21):
                seq[t, 0, j] = seq[t, 0, 0] + np.array([-0.02 * (j % 5), -0.03 * (j // 5), 0.0])
                seq[t, 1, j] = seq[t, 1, 0] + np.array([0.02 * (j % 5), -0.03 * (j // 5), 0.0])
        seq += rng.normal(0, 0.002, seq.shape).astype(np.float32)
        seq[20:] = 0.0
        payload = {
            'word': 'NONE',
            'session_id': f'idle_leaving_{i}',
            'source': 'synthetic_idle_generator',
            'sequence_length': 30,
            'frames': seq.reshape(30, 126).tolist()
        }
        (none_dir / f'idle_leaving_{i:02d}.json').write_text(json.dumps(payload, indent=2))
        count += 1

    # 5. Resting at desk / lap (low y)
    for i in range(10):
        seq = np.zeros((30, 2, 21, 3), dtype=np.float32)
        seq[:, 0, 0] = [0.35, 0.90, 0.0]
        seq[:, 1, 0] = [0.65, 0.90, 0.0]
        for j in range(1, 21):
            seq[:, 0, j] = seq[:, 0, 0] + np.array([-0.02 * (j % 5), -0.02 * (j // 5), 0.0])
            seq[:, 1, j] = seq[:, 1, 0] + np.array([0.02 * (j % 5), -0.02 * (j // 5), 0.0])
        seq += rng.normal(0, 0.002, seq.shape).astype(np.float32)
        payload = {
            'word': 'NONE',
            'session_id': f'idle_lap_{i}',
            'source': 'synthetic_idle_generator',
            'sequence_length': 30,
            'frames': seq.reshape(30, 126).tolist()
        }
        (none_dir / f'idle_lap_{i:02d}.json').write_text(json.dumps(payload, indent=2))
        count += 1

    print(f"Generated {count} NONE/idle sequences in {none_dir}")

if __name__ == '__main__':
    generate_none_dataset()