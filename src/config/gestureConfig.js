/**
 * SignBridge Gesture Recognition Configuration & Thresholds.
 * Centralizes all state machine timings, confidence limits, and stability parameters.
 */

export const GESTURE_CONFIG = {
  // 1. State Machine & Timing (calibrated for 30 FPS camera feed)
  NEUTRAL_HOLD_FRAMES: 10,       // Consecutive frames (~0.33s) holding 'two open palms' required to arm
  STABILITY_FRAMES_K: 8,         // Consecutive identical predictions (8-12) required to commit sign
  COOLDOWN_MS: 1000,             // Cooldown (1.0s) after emitting a sign before allowing re-arm
  MAJORITY_VOTE_WINDOW: 5,       // Sliding window for majority-vote smoothing in CAPTURING

  // 2. Confidence & Ambiguity Thresholds
  CONFIDENCE_THRESHOLD_LETTER: 0.85, // Top-class probability threshold for static letters
  CONFIDENCE_THRESHOLD_WORD: 0.80,   // Probability threshold for whole-word sequences
  MARGIN_THRESHOLD: 0.15,            // Minimum gap between top-1 and top-2 probabilities
  MAX_CENTROID_DISTANCE: 11.5,       // Maximum Euclidean distance to class centroid

  // 3. Resting / Idle Hand Detection
  IDLE_WRIST_Y_THRESHOLD: 0.88,      // Wrist below 88% of camera frame treated as resting on desk
  MIN_HAND_SPAN: 0.035,              // Minimum bounding box dimension for an intentional gesture

  // 4. Debug & Developer Tools
  DEBUG_OVERLAY_ENABLED: false,      // Display live debug metrics HUD overlay on camera
};

export const GESTURE_STATES = {
  NEUTRAL: 'neutral',
  ARMED: 'armed',
  CAPTURING: 'capturing',
  COOLDOWN: 'cooldown',
};

export const GESTURE_STATUS_LABELS = {
  neutral: 'Show neutral to continue',
  armed: 'Ready',
  capturing: 'Recognising...',
  cooldown: 'Sign locked',
};
