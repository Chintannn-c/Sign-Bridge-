import { useState, useRef, useCallback, useEffect } from 'react';

/**
 * Gesture Recognition State Machine with Temporal Smoothing.
 *
 * States: idle → searching → detected → tracking → recognising → stable
 *
 * Temporal smoothing requires N consecutive identical predictions above
 * a confidence threshold before committing a letter to the sentence buffer.
 * This prevents flickering from unstable single-frame predictions.
 */

const API_BASE = '/api';

const STATES = {
  IDLE: 'idle',
  SEARCHING: 'searching',
  DETECTED: 'detected',
  TRACKING: 'tracking',
  RECOGNISING: 'recognising',
  STABLE: 'stable',
};

const STATUS_LABELS = {
  idle: 'Ready',
  searching: 'Searching...',
  detected: 'Hand Detected',
  tracking: 'Tracking...',
  recognising: 'Recognising...',
  stable: 'Gesture Stable',
};

// Minimum consecutive identical predictions needed to commit a letter
const STABILITY_THRESHOLD = 3;
// Minimum confidence to consider a prediction valid
const CONFIDENCE_THRESHOLD = 0.65;

export function useGestureRecognition({ enabled = false } = {}) {
  const [status, setStatus] = useState(STATES.IDLE);
  const [detectedLetter, setDetectedLetter] = useState(null);
  const [confidence, setConfidence] = useState(0);
  const [sentenceBuffer, setSentenceBuffer] = useState('');
  const [allScores, setAllScores] = useState({});
  const [handInfo, setHandInfo] = useState(null); // { label, confidence }
  const [guidance, setGuidance] = useState(null);

  // Temporal smoothing refs
  const consecutiveRef = useRef({ letter: null, count: 0 });
  const lastCommitRef = useRef(0);
  const lastCommittedLetterRef = useRef(null);

  // Reset when disabled
  useEffect(() => {
    if (!enabled) {
      setStatus(STATES.IDLE);
      setDetectedLetter(null);
      setConfidence(0);
      consecutiveRef.current = { letter: null, count: 0 };
      lastCommittedLetterRef.current = null;
    } else {
      setStatus(STATES.SEARCHING);
    }
  }, [enabled]);

  /**
   * Process a frame of 42 hand landmarks from MediaPipe.
   * Called by the HandTrackingOverlay or LiveDetectionTab on each detection cycle.
   */
  const processLandmarks = useCallback(async (landmarks, handCount = 0, handedness = null) => {
    if (!enabled) return null;

    // Update hand info
    if (handCount > 0) {
      setHandInfo({
        count: handCount,
        label: handedness || (handCount >= 2 ? 'Both Hands' : 'Single Hand'),
      });
      setStatus(STATES.DETECTED);
    } else {
      setHandInfo(null);
      setStatus(STATES.SEARCHING);
      setGuidance('Show your hand inside the detection area.');
      consecutiveRef.current = { letter: null, count: 0 };
      lastCommittedLetterRef.current = null;
      return null;
    }

    // Check landmark quality and generate guidance
    if (handCount < 2) {
      setGuidance('ISL is a 2-Handed sign system. Please raise both hands into frame.');
    } else if (landmarks && landmarks.length >= 126) {
      const guidanceMsg = analyzeQuality(landmarks);
      setGuidance(guidanceMsg);
    }

    // Send to Flask API
    setStatus(STATES.TRACKING);
    try {
      const res = await fetch(`${API_BASE}/translate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ landmarks }),
      });

      if (!res.ok) return null;

      const prediction = await res.json();
      setStatus(STATES.RECOGNISING);
      setDetectedLetter(prediction.letter);
      setConfidence(prediction.confidence);
      setAllScores(prediction.all_scores || {});

      // Temporal smoothing — require N consecutive identical predictions
      if (prediction.confidence >= CONFIDENCE_THRESHOLD) {
        const prev = consecutiveRef.current;
        if (prev.letter === prediction.letter) {
          prev.count += 1;
        } else {
          consecutiveRef.current = { letter: prediction.letter, count: 1 };
        }

        // Commit letter after reaching stability threshold (only if different from last committed letter)
        if (consecutiveRef.current.count >= STABILITY_THRESHOLD) {
          const now = Date.now();
          if (now - lastCommitRef.current > 400) {
            if (lastCommittedLetterRef.current !== prediction.letter) {
              setStatus(STATES.STABLE);
              setSentenceBuffer(prev => prev + prediction.letter);
              lastCommitRef.current = now;
              lastCommittedLetterRef.current = prediction.letter;
              consecutiveRef.current = { letter: null, count: 0 };
            }
          }
        }
      } else {
        consecutiveRef.current = { letter: null, count: 0 };
      }

      return prediction;
    } catch (e) {
      console.warn('Gesture recognition API error:', e);
      return null;
    }
  }, [enabled]);

  // Sentence buffer actions
  const undoLetter = useCallback(() => {
    setSentenceBuffer(prev => prev.slice(0, -1));
  }, []);

  const deleteLetter = useCallback(() => {
    setSentenceBuffer(prev => prev.slice(0, -1));
  }, []);

  const clearBuffer = useCallback(() => {
    setSentenceBuffer('');
    consecutiveRef.current = { letter: null, count: 0 };
    lastCommittedLetterRef.current = null;
  }, []);

  const addSpace = useCallback(() => {
    setSentenceBuffer(prev => prev + ' ');
  }, []);

  const updateSentence = useCallback((newText) => {
    setSentenceBuffer(newText);
  }, []);

  const refineSentence = useCallback(async () => {
    if (!sentenceBuffer.trim()) return null;
    try {
      const res = await fetch(`${API_BASE}/llm/refine`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: sentenceBuffer.trim() })
      });
      if (res.ok) {
        const data = await res.json();
        if (data.refined_sentence) {
          setSentenceBuffer(data.refined_sentence);
          return data;
        }
      }
    } catch (e) {
      console.warn('Refine LLM error:', e);
    }
    return null;
  }, [sentenceBuffer]);

  const commitSentence = useCallback(() => {
    const text = sentenceBuffer.trim();
    setSentenceBuffer('');
    return text;
  }, [sentenceBuffer]);

  return {
    // State
    status,
    statusLabel: STATUS_LABELS[status] || status,
    detectedLetter,
    confidence,
    allScores,
    handInfo,
    guidance,
    sentenceBuffer,

    // Actions
    processLandmarks,
    undoLetter,
    deleteLetter,
    clearBuffer,
    addSpace,
    updateSentence,
    refineSentence,
    commitSentence,
  };
}

/**
 * Analyze landmark quality and return guidance message if needed.
 */
function analyzeQuality(landmarks) {
  if (!landmarks || landmarks.length < 126) {
    return 'Hand not fully visible. Move closer to camera.';
  }

  // Check if landmarks are too clustered (hand too far away)
  let xRange = 0, yRange = 0;
  let xMin = 1, xMax = 0, yMin = 1, yMax = 0;

  for (let i = 0; i < 42; i++) {
    const x = landmarks[i * 3];
    const y = landmarks[i * 3 + 1];
    if (x > 0 || y > 0) {
      xMin = Math.min(xMin, x);
      xMax = Math.max(xMax, x);
      yMin = Math.min(yMin, y);
      yMax = Math.max(yMax, y);
    }
  }

  xRange = xMax - xMin;
  yRange = yMax - yMin;

  if (xRange < 0.05 && yRange < 0.05) {
    return 'Move hand closer to camera.';
  }

  // Check if hand is near edges (partially out of frame)
  if (xMin < 0.02 || xMax > 0.98 || yMin < 0.02 || yMax > 0.98) {
    return 'Hand partially outside frame. Center your hand.';
  }

  // Check if fingers are spread enough for recognition
  if (xRange < 0.1 && yRange < 0.1) {
    return 'Spread fingers slightly for better recognition.';
  }

  return null; // No issues
}
