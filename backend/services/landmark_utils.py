"""
SignBridge — Centralized Landmark Extraction, Preprocessing & Augmentation Utilities.

Provides shared, bug-free implementations of:
1. `clean_word_label`: Single source of truth for ISL gloss normalization.
2. `get_landmark_vector_from_result`: Standardizes MediaPipe HandLandmarker results to 126 floats (42 points x 3)
   with proper dual-hand preservation without overwriting.
3. `trim_idle_frames`: Trims non-active lead-in/lead-out frames and gates videos without hand detection.
4. `sample_or_interpolate`: Resamples landmark frame sequences to exact target length.
5. `augment_raw_sequence`: Clean, artifact-free augmentation (masked noise, coordinate-flipped mirror, wrist-centered rotation).
"""

import re
from pathlib import Path
from typing import List, Optional, Union
import numpy as np

SEQUENCE_LENGTH = 30


def _normalize_word_str(s: str) -> str:
    """Normalizes string to canonical uppercase word label."""
    if not s:
        return "UNKNOWN"
    name = re.sub(r'[\d_\-,\.]+', ' ', str(s)).strip().upper()
    name = re.sub(r'\s+', '_', name)
    if 'THANK' in name and 'VERY' in name:
        return 'THANK_YOU_VERY_MUCH'
    if 'THANK' in name:
        return 'THANK_YOU'
    if 'BYE' in name:
        return 'BYE_BYE'
    if 'NAMASTE' in name:
        return 'NAMASTE'
    if 'INDIAN' in name or 'INDIA' in name:
        return 'INDIA'
    if 'DONT_UNDERSTAND' in name or 'I_DONT_UNDERSTAND' in name:
        return 'I_DONT_UNDERSTAND'
    if 'I_AM_FINE' in name or 'IM_FINE' in name:
        return 'IM_FINE'
    if 'MY_NAME' in name:
        return 'MY_NAME_IS'
    if 'GOOD_MORNING' in name:
        return 'GOOD_MORNING'
    if 'GOOD_AFTERNOON' in name:
        return 'GOOD_AFTERNOON'
    if 'GOOD_EVENING' in name:
        return 'GOOD_EVENING'
    if 'GOOD_NIGHT' in name:
        return 'GOOD_NIGHT'
    if 'HOW_ARE_YOU' in name:
        return 'HOW_ARE_YOU'
    if 'NO_FEAR' in name:
        return 'NO_FEAR'
    if name in ('ME', 'I', 'ME_I') or name.startswith('ME_'):
        return 'ME'
    if name == 'HI':
        return 'HELLO'
    return name


def clean_word_label(path_or_name: Union[str, Path], video_dirs: Optional[List[Path]] = None) -> str:
    """Derives a clean uppercase word class name from video file path or filename."""
    p = Path(path_or_name)
    if video_dirs:
        for vdir in video_dirs:
            try:
                rel = p.resolve().relative_to(vdir.resolve())
                if len(rel.parts) > 1:
                    folder_cand = _normalize_word_str(rel.parts[0])
                    if folder_cand and folder_cand != "UNKNOWN":
                        return folder_cand
            except (ValueError, RuntimeError):
                pass
    return _normalize_word_str(p.stem)


def get_landmark_vector_from_result(detection_result) -> List[float]:
    """
    Standardizes MediaPipe HandLandmarker result into a 126-float array.
    Preserves both hands even when labels collide, and avoids dead-code overwrite bugs.
    """
    if not detection_result or not getattr(detection_result, 'hand_landmarks', None):
        return [0.0] * 126

    landmarks_list = detection_result.hand_landmarks
    if not landmarks_list:
        return [0.0] * 126

    handedness_list = getattr(detection_result, 'handedness', []) or []

    detected_hands = []
    for i, hand_lms in enumerate(landmarks_list):
        label = None
        score = 0.5
        if i < len(handedness_list) and handedness_list[i]:
            cat = handedness_list[i][0]
            label = getattr(cat, 'category_name', None) or getattr(cat, 'display_name', None)
            score = getattr(cat, 'score', 0.5)

        coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms], dtype=np.float32)
        detected_hands.append({'coords': coords, 'label': label, 'score': score})

    left_hand = None
    right_hand = None

    if len(detected_hands) == 1:
        h = detected_hands[0]
        if h['label'] == 'Left':
            left_hand = h['coords']
        elif h['label'] == 'Right':
            right_hand = h['coords']
        else:
            if h['coords'][0, 0] < 0.5:
                left_hand = h['coords']
            else:
                right_hand = h['coords']
    elif len(detected_hands) >= 2:
        # Sort by confidence score descending to keep top 2
        detected_hands.sort(key=lambda x: x['score'], reverse=True)
        h0, h1 = detected_hands[0], detected_hands[1]

        # If labels are distinct, assign by label
        if h0['label'] and h1['label'] and h0['label'] != h1['label']:
            left_hand = h0['coords'] if h0['label'] == 'Left' else h1['coords']
            right_hand = h0['coords'] if h0['label'] == 'Right' else h1['coords']
        else:
            # Fallback for identical/ambiguous labels: assign by x position to avoid overwriting
            if h0['coords'][0, 0] < h1['coords'][0, 0]:
                left_hand, right_hand = h0['coords'], h1['coords']
            else:
                left_hand, right_hand = h1['coords'], h0['coords']

    l_arr = left_hand if left_hand is not None else np.zeros((21, 3), dtype=np.float32)
    r_arr = right_hand if right_hand is not None else np.zeros((21, 3), dtype=np.float32)

    combined = np.concatenate([l_arr.reshape(-1), r_arr.reshape(-1)])
    return combined.tolist()


def trim_idle_frames(raw_frames: List[List[float]], min_active_ratio: float = 0.4, margin: int = 2) -> Optional[List[List[float]]]:
    """
    Trims non-active lead-in and lead-out frames from video.
    Returns None if the video has insufficient detected hand activity (< min_active_ratio).
    """
    if not raw_frames:
        return None

    active_indices = [
        i for i, f in enumerate(raw_frames)
        if any(v != 0.0 for v in f)
    ]

    if not active_indices or (len(active_indices) / len(raw_frames)) < min_active_ratio:
        return None

    start_idx = max(0, active_indices[0] - margin)
    end_idx = min(len(raw_frames), active_indices[-1] + margin + 1)
    trimmed = raw_frames[start_idx:end_idx]

    return trimmed if len(trimmed) >= 5 else None


def sample_or_interpolate(sequence: List[List[float]], target_length: int = SEQUENCE_LENGTH) -> List[List[float]]:
    """Resamples a list of landmark frames to exactly target_length frames."""
    arr = np.array(sequence, dtype=np.float32)
    current_len = len(arr)
    if current_len == target_length:
        return arr.tolist()
    if current_len == 0:
        return np.zeros((target_length, 126), dtype=np.float32).tolist()

    indices = np.linspace(0, current_len - 1, target_length).round().astype(np.int64)
    return arr[indices].tolist()


def augment_raw_sequence(
    raw_frames: List[List[float]],
    rng: np.random.Generator,
    aug_id: int = 0,
    target_length: int = SEQUENCE_LENGTH,
    allow_mirror: bool = True
) -> Optional[List[List[float]]]:
    """
    Apply robust, artifact-free augmentation to a sequence of raw landmark frames.
    Fixes:
      1. Noise is masked so missing hands stay exact zeros (no ghost hands).
      2. Hand mirror swaps hand channels AND flips x-coordinates (1.0 - x).
      3. Rotation rotates each detected hand around its wrist, leaving zeros untouched.
      4. True speed variation by temporal stretching/compressing before padding/cropping.
    """
    arr = np.array(raw_frames, dtype=np.float32)
    n_raw = len(arr)
    if n_raw < 5:
        return None

    # 1. Random sub-clip sampling (70-100% of video)
    clip_ratio = rng.uniform(0.70, 1.0)
    clip_len = max(5, int(n_raw * clip_ratio))
    max_start = max(0, n_raw - clip_len)
    start = rng.integers(0, max_start + 1)
    sub_clip = arr[start:start + clip_len]

    # 2. True speed variation: resample sub_clip to (target_length * speed_factor), then crop/pad
    speed_factor = rng.uniform(0.8, 1.25)
    speed_len = max(10, int(target_length * speed_factor))
    speed_idx = np.linspace(0, len(sub_clip) - 1, speed_len).round().astype(np.int64)
    speed_seq = sub_clip[speed_idx]

    if len(speed_seq) > target_length:
        # Faster motion: crop random sub-window to target_length
        crop_start = rng.integers(0, len(speed_seq) - target_length + 1)
        resampled = speed_seq[crop_start:crop_start + target_length]
    elif len(speed_seq) < target_length:
        # Slower motion: pad by repeating final pose
        pad_len = target_length - len(speed_seq)
        last_frame = np.repeat(speed_seq[-1:], pad_len, axis=0)
        resampled = np.concatenate([speed_seq, last_frame], axis=0)
    else:
        resampled = speed_seq

    result = resampled.copy()

    # 3. Gaussian coordinate jitter (applied ONLY to detected points, NEVER to zeros)
    noise_scale = rng.uniform(0.002, 0.008)
    present_mask = (result != 0.0).astype(np.float32)
    result += rng.normal(0, noise_scale, size=result.shape).astype(np.float32) * present_mask

    # 4. Hand mirror (horizontal flip: swap Left/Right channels AND reflect x -> 1 - x)
    if allow_mirror and rng.random() < 0.25:
        pts = result.reshape(target_length, 2, 21, 3).copy()
        # Reflect x for all present landmarks
        x_present = (pts[..., 0] != 0.0)
        pts[..., 0] = np.where(x_present, 1.0 - pts[..., 0], 0.0)
        # Swap Left hand (idx 0) and Right hand (idx 1)
        mirrored = pts[:, ::-1].reshape(target_length, 126)
        result = mirrored

    # 5. Rotation jitter: rotate around each present hand's wrist (point 0)
    if rng.random() < 0.35:
        angle = rng.uniform(-0.12, 0.12)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        pts = result.reshape(target_length, 2, 21, 3).copy()

        for h_idx in (0, 1):
            hand_pts = pts[:, h_idx]  # shape: (target_length, 21, 3)
            # Check if this hand is present
            if np.any(hand_pts != 0.0):
                # Wrist is keypoint 0
                wrist_x = hand_pts[:, 0:1, 0]  # shape: (target_length, 1)
                wrist_y = hand_pts[:, 0:1, 1]

                dx = hand_pts[:, :, 0] - wrist_x
                dy = hand_pts[:, :, 1] - wrist_y

                rot_x = dx * cos_a - dy * sin_a + wrist_x
                rot_y = dx * sin_a + dy * cos_a + wrist_y

                # Only update non-zero points
                mask = (hand_pts[:, :, 0] != 0.0)
                hand_pts[:, :, 0] = np.where(mask, rot_x, 0.0)
                hand_pts[:, :, 1] = np.where(mask, rot_y, 0.0)

        result = pts.reshape(target_length, 126)

    # 6. Random frame dropout (zero out 1-2 random frames to simulate camera flicker)
    if rng.random() < 0.20:
        n_drop = rng.integers(1, 3)
        drop_idx = rng.choice(target_length, size=n_drop, replace=False)
        result[drop_idx] = 0.0

    return result.tolist()
