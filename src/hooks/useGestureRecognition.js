import { useState, useRef, useCallback, useEffect } from 'react';

/**
 * Gesture Recognition State Machine with Temporal Smoothing & Dual Mode (Letter & Word).
 *
 * Modes:
 *   1. 'letter' — Frame-by-frame static alphabet recognition (A-Z) via /api/translate
 *   2. 'word'   — Sliding 30-frame temporal sequence recognition (ISL words) via /api/translate/word
 *
 * States: idle → searching → detected → tracking → recognising → stable
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
  searching: 'Searching for hands...',
  detected: 'Hands Detected',
  tracking: 'Tracking Keypoints...',
  recognising: 'Recognising Sign...',
  stable: 'Sign Locked',
};

// Minimum consecutive identical predictions needed to commit a letter
const STABILITY_THRESHOLD = 3;
// Minimum confidence to consider a prediction valid
const CONFIDENCE_THRESHOLD_LETTER = 0.52;
const CONFIDENCE_THRESHOLD_WORD = 0.65;
const WORD_SEQUENCE_LENGTH = 30;

function isValidHandGeometry(landmarks) {
  if (!landmarks || landmarks.length < 126) return false;
  let activePoints = 0;
  let xMin = 1, xMax = 0, yMin = 1, yMax = 0;
  for (let i = 0; i < 42; i++) {
    const x = landmarks[i * 3];
    const y = landmarks[i * 3 + 1];
    if (x !== 0 || y !== 0) {
      activePoints++;
      xMin = Math.min(xMin, x);
      xMax = Math.max(xMax, x);
      yMin = Math.min(yMin, y);
      yMax = Math.max(yMax, y);
    }
  }
  if (activePoints < 21) return false;
  if ((xMax - xMin) < 0.03 || (yMax - yMin) < 0.03) return false;
  return true;
}

function captureVideoSnapshot(videoEl) {
  if (!videoEl || videoEl.readyState < 2) return null;
  try {
    const canvas = document.createElement('canvas');
    canvas.width = 224;
    canvas.height = 224;
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;
    ctx.drawImage(videoEl, 0, 0, 224, 224);
    return canvas.toDataURL('image/jpeg', 0.8);
  } catch {
    return null;
  }
}

export function useGestureRecognition({ enabled = false, initialMode = 'letter', videoElement = null, onSendMessage = null } = {}) {
  const [recognitionMode, setRecognitionMode] = useState(initialMode); // 'letter' | 'word'
  const [status, setStatus] = useState(STATES.IDLE);
  const [detectedLetter, setDetectedLetter] = useState(null);
  const [detectedWord, setDetectedWord] = useState(null);
  const [confidence, setConfidence] = useState(0);
  const [sentenceBuffer, setSentenceBuffer] = useState('');
  const [allScores, setAllScores] = useState({});
  const [handInfo, setHandInfo] = useState(null); // { count, label }
  const [guidance, setGuidance] = useState(null);
  const [wordBufferCount, setWordBufferCount] = useState(0);
  const [availableWords, setAvailableWords] = useState([
    'AGAIN', 'BYE_BYE', 'DEAF', 'HEARING', 'HELLO', 'INDIA', 'LANGUAGE', 'MAN', 'ME', 'NAMASTE'
  ]);

  // Inactivity / Hand-Drop Auto-Send states
  const [autoSendEnabled, setAutoSendEnabled] = useState(true);
  const [autoSendTimeoutMs, setAutoSendTimeoutMs] = useState(1800);
  const [inactivityCountdown, setInactivityCountdown] = useState(null);
  const [lastAutoSpoken, setLastAutoSpoken] = useState(null);

  // Temporal smoothing & buffering refs
  const consecutiveLetterRef = useRef({ letter: null, count: 0 });
  const consecutiveWordRef = useRef({ word: null, count: 0 });
  const lastCommitRef = useRef(0);
  const lastCommittedItemRef = useRef(null);
  const frameBufferRef = useRef([]);
  const wordInferenceCooldownRef = useRef(0);
  const isRequestPendingRef = useRef(false);

  // Inactivity auto-send refs
  const sentenceBufferRef = useRef('');
  sentenceBufferRef.current = sentenceBuffer;
  const autoSendTimerRef = useRef(null);
  const countdownIntervalRef = useRef(null);
  const isAutoSendingRef = useRef(false);

  const cancelInactivityCountdown = useCallback(() => {
    if (autoSendTimerRef.current) {
      clearTimeout(autoSendTimerRef.current);
      autoSendTimerRef.current = null;
    }
    if (countdownIntervalRef.current) {
      clearInterval(countdownIntervalRef.current);
      countdownIntervalRef.current = null;
    }
    setInactivityCountdown(null);
  }, []);

  const triggerAutoSend = useCallback(async () => {
    const rawText = sentenceBufferRef.current.trim();
    if (!rawText || isAutoSendingRef.current) {
      cancelInactivityCountdown();
      return;
    }

    isAutoSendingRef.current = true;
    cancelInactivityCountdown();

    try {
      let speechText = rawText;
      // 1. Refine with LLM (Groq / Gemini)
      try {
        const res = await fetch(`${API_BASE}/llm/refine`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: rawText })
        });
        if (res.ok) {
          const data = await res.json();
          if (data.refined_sentence && data.refined_sentence.trim()) {
            speechText = data.refined_sentence.trim();
          }
        }
      } catch (refineErr) {
        console.debug('Auto-send LLM refine notice:', refineErr);
      }

      // Sanitize text from any residual thinking tokens
      speechText = speechText
        .replace(/<think>[\s\S]*?(?:<\/think>|$)/gi, '')
        .replace(/^(?:Here'?s (?:a )?thinking process:?|Thinking Process:?)[\s\S]*?(?=\n\n|\n|$)/gi, '')
        .trim() || rawText;

      // 2. Speak aloud via Web Speech TTS
      if (speechText && 'speechSynthesis' in window) {
        window.speechSynthesis.cancel(); // Stop any previous speech
        const utterance = new SpeechSynthesisUtterance(speechText);
        utterance.rate = 0.95;
        window.speechSynthesis.speak(utterance);
      }

      // 3. Send message to the Dialogue Chat Thread
      if (onSendMessage) {
        onSendMessage(speechText, 'human');
      }

      // 4. Update feedback and clear buffer
      setLastAutoSpoken(speechText);
      setSentenceBuffer('');
    } catch (e) {
      console.warn('Auto-send execution error:', e);
    } finally {
      isAutoSendingRef.current = false;
      cancelInactivityCountdown();
    }
  }, [cancelInactivityCountdown, onSendMessage]);

  const handleHandsDropped = useCallback(() => {
    if (!autoSendEnabled || !sentenceBufferRef.current.trim() || isAutoSendingRef.current) {
      return;
    }

    // Start countdown if not already started
    if (!autoSendTimerRef.current) {
      const startTime = Date.now();
      const targetTime = startTime + autoSendTimeoutMs;

      setInactivityCountdown((autoSendTimeoutMs / 1000).toFixed(1));

      countdownIntervalRef.current = setInterval(() => {
        const remainingMs = targetTime - Date.now();
        if (remainingMs <= 0) {
          if (countdownIntervalRef.current) clearInterval(countdownIntervalRef.current);
          countdownIntervalRef.current = null;
        } else {
          setInactivityCountdown((remainingMs / 1000).toFixed(1));
        }
      }, 100);

      autoSendTimerRef.current = setTimeout(() => {
        autoSendTimerRef.current = null;
        triggerAutoSend();
      }, autoSendTimeoutMs);
    }
  }, [autoSendEnabled, autoSendTimeoutMs, triggerAutoSend]);

  // Fetch model information & available word classes
  useEffect(() => {
    async function fetchModelInfo() {
      try {
        const res = await fetch(`${API_BASE}/health`);
        if (res.ok) {
          const data = await res.json();
          if (data.word_recognizer_available) {
            const infoRes = await fetch(`${API_BASE}/words/info`);
            if (infoRes.ok) {
              const infoData = await infoRes.json();
              if (infoData.labels && infoData.labels.length > 0) {
                setAvailableWords(infoData.labels);
              }
            }
          }
        }
      } catch (e) {
        console.warn('Could not fetch backend model info:', e);
      }
    }
    fetchModelInfo();
  }, []);

  // Reset state when disabled or when mode switches
  useEffect(() => {
    if (!enabled) {
      setStatus(STATES.IDLE);
      setDetectedLetter(null);
      setDetectedWord(null);
      setConfidence(0);
      consecutiveLetterRef.current = { letter: null, count: 0 };
      consecutiveWordRef.current = { word: null, count: 0 };
      lastCommittedItemRef.current = null;
      frameBufferRef.current = [];
      setWordBufferCount(0);
      cancelInactivityCountdown();
    } else {
      setStatus(STATES.SEARCHING);
      frameBufferRef.current = [];
      setWordBufferCount(0);
    }
  }, [enabled, recognitionMode, cancelInactivityCountdown]);

  /**
   * Process a frame of 126 landmark floats from MediaPipe.
   */
  const processLandmarks = useCallback(async (landmarks, handCount = 0, handedness = null) => {
    if (!enabled) return null;

    const hasActiveLandmarks = Boolean(handCount > 0 && isValidHandGeometry(landmarks));

    // Update hand status
    if (handCount > 0 && hasActiveLandmarks) {
      cancelInactivityCountdown(); // Active signing: cancel any pending hand-drop auto-send
      setHandInfo({
        count: handCount,
        label: handedness || (handCount >= 2 ? 'Both Hands (ISL)' : 'Single Hand'),
      });
      setStatus(STATES.DETECTED);
    } else {
      setHandInfo(null);
      setStatus(STATES.SEARCHING);
      setDetectedLetter(null);
      setDetectedWord(null);
      setConfidence(0);
      setAllScores({});
      setGuidance('Show your hands inside the camera frame.');
      consecutiveLetterRef.current = { letter: null, count: 0 };
      consecutiveWordRef.current = { word: null, count: 0 };
      lastCommittedItemRef.current = null;
      frameBufferRef.current = [];
      setWordBufferCount(0);

      // Trigger hand-drop countdown if sentence buffer has words
      handleHandsDropped();
      return null;
    }

    // Adaptive Guidance for Single-Hand vs Dual-Hand Gestures
    if (handCount === 1) {
      setGuidance('Single hand detected (e.g. C, L, O, V, numbers 0–9)');
    } else if (handCount >= 2) {
      const guidanceMsg = landmarks && landmarks.length >= 126 ? analyzeQuality(landmarks) : null;
      setGuidance(guidanceMsg || 'Dual-hand ISL tracking active');
    }

    setStatus(STATES.TRACKING);

    // ─────────────────────────────────────────────────────────────────────────
    // MODE 1: WORD SEQUENCE RECOGNITION (Sliding 30-Frame Window)
    // ─────────────────────────────────────────────────────────────────────────
    if (recognitionMode === 'word') {
      // Append current frame to rolling sequence buffer
      frameBufferRef.current.push(landmarks);
      if (frameBufferRef.current.length > WORD_SEQUENCE_LENGTH) {
        frameBufferRef.current.shift();
      }
      setWordBufferCount(frameBufferRef.current.length);

      // Only perform word inference when buffer is full and throttled (every ~100ms)
      const now = Date.now();
      if (frameBufferRef.current.length >= WORD_SEQUENCE_LENGTH && (now - wordInferenceCooldownRef.current >= 100)) {
        wordInferenceCooldownRef.current = now;

        try {
          const res = await fetch(`${API_BASE}/translate/word`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ frames: frameBufferRef.current }),
          });

          if (!res.ok) return null;

          const prediction = await res.json();

          // Handle confidence rejection from backend
          if (prediction.rejected) {
            setStatus(STATES.TRACKING);
            setDetectedWord(null);
            setDetectedLetter(null);
            setConfidence(prediction.confidence || 0);
            setGuidance('Low confidence — adjust your hand position.');
            consecutiveWordRef.current = { word: null, count: 0 };
            return prediction;
          }

          if (!prediction || !prediction.word || prediction.word === '?') return null;

          setStatus(STATES.RECOGNISING);
          setDetectedWord(prediction.word);
          setDetectedLetter(null);
          setConfidence(prediction.confidence);
          setAllScores(prediction.all_scores || {});

          // Temporal smoothing for whole words
          if (prediction.confidence >= CONFIDENCE_THRESHOLD_WORD) {
            const prev = consecutiveWordRef.current;
            if (prev.word === prediction.word) {
              prev.count += 1;
            } else {
              consecutiveWordRef.current = { word: prediction.word, count: 1 };
            }

            // Commit word after 2 consecutive frames or high confidence (>0.85)
            if (consecutiveWordRef.current.count >= 2 || prediction.confidence >= 0.85) {
              if (now - lastCommitRef.current > 1200) { // 1.2s cooldown between words
                if (lastCommittedItemRef.current !== prediction.word) {
                  setStatus(STATES.STABLE);
                  setSentenceBuffer(prev => {
                    const clean = prev.trim();
                    return clean.length > 0 ? `${clean} ${prediction.word}` : prediction.word;
                  });
                  lastCommitRef.current = now;
                  lastCommittedItemRef.current = prediction.word;
                  consecutiveWordRef.current = { word: null, count: 0 };
                  frameBufferRef.current = []; // Clear buffer after successful word lock
                  setWordBufferCount(0);
                }
              }
            }
          } else {
            consecutiveWordRef.current = { word: null, count: 0 };
          }

          return prediction;
        } catch (e) {
          console.warn('Word recognition error:', e);
          return null;
        }
      }
      return null;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // MODE 2: STATIC LETTER RECOGNITION (A-Z)
    // ─────────────────────────────────────────────────────────────────────────
    if (isRequestPendingRef.current) return null;
    isRequestPendingRef.current = true;

    try {
      const res = await fetch(`${API_BASE}/translate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ landmarks }),
      });

      if (!res.ok) return null;

      let prediction = await res.json();

      // On-demand visual silhouette fallback for borderline contact signs
      if (videoElement && (prediction.rejected || prediction.confidence < 0.65)) {
        const snapB64 = captureVideoSnapshot(videoElement);
        if (snapB64) {
          try {
            const snapRes = await fetch(`${API_BASE}/translate/snapshot`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ landmarks, image: snapB64 }),
            });
            if (snapRes.ok) {
              const snapData = await snapRes.json();
              if (snapData && snapData.letter && snapData.letter !== '?') {
                prediction = snapData;
              }
            }
          } catch (snapErr) {
            console.debug('Snapshot fallback bypassed:', snapErr);
          }
        }
      }

      // Handle confidence rejection or low confidence from backend
      if (prediction.rejected || !prediction.letter || prediction.letter === '?' || prediction.confidence < CONFIDENCE_THRESHOLD_LETTER) {
        setStatus(STATES.TRACKING);
        setDetectedLetter(null);
        setDetectedWord(null);
        setConfidence(prediction.confidence || 0);
        setAllScores(prediction.all_scores || {});
        setGuidance(prediction.confidence > 0 ? 'Hold sign steady...' : 'Show hands clearly');
        consecutiveLetterRef.current = { letter: null, count: 0 };
        return prediction;
      }

      // Temporal smoothing & stability check
      const prev = consecutiveLetterRef.current;
      if (prev.letter === prediction.letter) {
        prev.count += 1;
      } else {
        // Sign changed — immediately reset streak for new letter
        consecutiveLetterRef.current = { letter: prediction.letter, count: 1 };
      }

      // Display detected letter with high responsiveness
      if (consecutiveLetterRef.current.count >= 1) {
        setStatus(consecutiveLetterRef.current.count >= 2 ? STATES.RECOGNISING : STATES.TRACKING);
        setDetectedLetter(prediction.letter);
        setDetectedWord(null);
        setConfidence(prediction.confidence);
        setAllScores(prediction.all_scores || {});
      }

      // Commit letter after reaching stability threshold
      if (consecutiveLetterRef.current.count >= STABILITY_THRESHOLD) {
        const now = Date.now();
        if (now - lastCommitRef.current > 450) {
          if (lastCommittedItemRef.current !== prediction.letter) {
            setStatus(STATES.STABLE);
            setSentenceBuffer(prevBuf => prevBuf + prediction.letter);
            lastCommitRef.current = now;
            lastCommittedItemRef.current = prediction.letter;
            consecutiveLetterRef.current = { letter: null, count: 0 };
          }
        }
      }

      return prediction;
    } catch (e) {
      console.warn('Letter recognition API error:', e);
      return null;
    } finally {
      isRequestPendingRef.current = false;
    }
  }, [enabled, recognitionMode]);

  // Sentence buffer actions
  const undoLetter = useCallback(() => {
    setSentenceBuffer(prev => {
      const trimmed = prev.trimEnd();
      const lastSpaceIdx = trimmed.lastIndexOf(' ');
      if (lastSpaceIdx !== -1) {
        return trimmed.substring(0, lastSpaceIdx + 1);
      }
      return prev.slice(0, -1);
    });
  }, []);

  const deleteLetter = useCallback(() => {
    setSentenceBuffer(prev => prev.slice(0, -1));
  }, []);

  const clearBuffer = useCallback(() => {
    setSentenceBuffer('');
    consecutiveLetterRef.current = { letter: null, count: 0 };
    consecutiveWordRef.current = { word: null, count: 0 };
    lastCommittedItemRef.current = null;
    frameBufferRef.current = [];
    setWordBufferCount(0);
  }, []);

  const addSpace = useCallback(() => {
    setSentenceBuffer(prev => (prev.endsWith(' ') ? prev : prev + ' '));
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

  const sendSentence = useCallback(async () => {
    const rawText = sentenceBufferRef.current.trim();
    if (!rawText) return;
    cancelInactivityCountdown();
    let textToSend = rawText;
    try {
      const res = await fetch(`${API_BASE}/llm/refine`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: rawText })
      });
      if (res.ok) {
        const data = await res.json();
        if (data.refined_sentence && data.refined_sentence.trim()) {
          textToSend = data.refined_sentence.trim();
        }
      }
    } catch (e) {
      console.warn('Manual send refine notice:', e);
    }
    textToSend = textToSend.replace(/<think>[\s\S]*?(?:<\/think>|$)/gi, '').trim() || rawText;
    if (onSendMessage) {
      onSendMessage(textToSend, 'human');
    }
    setLastAutoSpoken(textToSend);
    setSentenceBuffer('');
  }, [cancelInactivityCountdown, onSendMessage]);

  return {
    // Mode
    recognitionMode,
    setRecognitionMode,

    // State
    status,
    statusLabel: STATUS_LABELS[status] || status,
    detectedLetter,
    detectedWord,
    detectedSign: detectedWord || detectedLetter,
    confidence,
    allScores,
    handInfo,
    guidance,
    sentenceBuffer,
    wordBufferCount,
    wordBufferMax: WORD_SEQUENCE_LENGTH,
    // Auto-Send / Inactivity states
    autoSendEnabled,
    setAutoSendEnabled,
    autoSendTimeoutMs,
    setAutoSendTimeoutMs,
    inactivityCountdown,
    lastAutoSpoken,

    // Actions
    processLandmarks,
    undoLetter,
    deleteLetter,
    clearBuffer,
    addSpace,
    updateSentence,
    refineSentence,
    commitSentence,
    sendSentence,
    triggerAutoSend,
    cancelInactivityCountdown,
  };
}

/**
 * Analyze landmark quality and return guidance message if needed.
 */
function analyzeQuality(landmarks) {
  if (!landmarks || landmarks.length < 126) {
    return 'Hand not fully visible. Move closer to camera.';
  }

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

  const xRange = xMax - xMin;
  const yRange = yMax - yMin;

  if (xRange < 0.05 && yRange < 0.05) {
    return 'Move hands closer to camera.';
  }

  if (xMin < 0.02 || xMax > 0.98 || yMin < 0.02 || yMax > 0.98) {
    return 'Hands partially outside frame. Center both hands.';
  }

  if (xRange < 0.1 && yRange < 0.1) {
    return 'Spread fingers clearly for better recognition.';
  }

  return null;
}
