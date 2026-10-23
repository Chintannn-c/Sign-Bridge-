"""
SignBridge Gesture Recognition Configuration and Thresholds.
Centralizes all state machine and validation parameters for easy tuning.
"""

# State Machine Frames & Timings (based on ~30 FPS camera feed)
NEUTRAL_HOLD_FRAMES = 10       # ~0.33s hold of two open palms required to re-arm
STABILITY_FRAMES_K = 8         # 8 consecutive identical predictions required to accept gesture
COOLDOWN_FRAMES = 15           # ~0.50s cooldown frames between gestures
COOLDOWN_MS = 1000             # 1.0s timestamp-based cooldown to prevent duplicate emissions
MAJORITY_VOTE_WINDOW = 5       # Size of majority-vote smoothing window inside CAPTURING

# Model & Dataset Validation Thresholds
CONFIDENCE_THRESHOLD = 0.85    # Top-class probability threshold (reject unknowns)
MARGIN_THRESHOLD = 0.15        # Margin between top-1 and top-2 probabilities
DEFAULT_MAX_CENTROID_DISTANCE = 11.5  # Fallback maximum Euclidean distance to class centroid

# Hand Tracking & Idle Thresholds
MIN_HAND_SPAN = 0.04           # Minimum bounding box span for active hand
IDLE_WRIST_Y_THRESHOLD = 0.88  # Wrist below 88% of frame is treated as resting/idle hand
TRACKING_CONFIDENCE_MIN = 0.35 # Minimum confidence for landmark validity

# Debug
DEBUG_OVERLAY_ENABLED = False
