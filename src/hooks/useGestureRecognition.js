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

/**
 * Fixed-capacity ring buffer — O(1) push, zero shifting, zero allocation after init.
 * Replaces frameBufferRef's plain array + Array.shift() pattern.
 */
class RingBuffer {
  constructor(capacity) {
    this._cap = capacity;
    this._buf = new Array(capacity);
    this._head = 0; // next write slot
    this._len  = 0;
  }
  push(item) {
    this._buf[this._head] = item;
    this._head = (this._head + 1) % this._cap;
    if (this._len < this._cap) this._len++;
  }
  get length() { return this._len; }
  /** Most-recently pushed item. */
  last()  { return this._len === 0 ? undefined : this._buf[(this._head - 1 + this._cap) % this._cap]; }
  /** Second-to-last item. */
  last2() { return this._len < 2  ? undefined : this._buf[(this._head - 2 + this._cap) % this._cap]; }
  /** Return items in insertion order (oldest → newest). */
  toArray() {
    if (this._len === 0) return [];
    const start = (this._head - this._len + this._cap) % this._cap;
    if (start + this._len <= this._cap) {
      return this._buf.slice(start, start + this._len);
    }
    return this._buf.slice(start).concat(this._buf.slice(0, this._head));
  }
  clear() { this._head = 0; this._len = 0; }
}

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
const STABILITY_THRESHOLD = 2; // Immediate responsiveness without freezing
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

function formatWordForSentence(rawWord) {
  if (!rawWord) return '';
  // Convert compound names with underscores (e.g. 'GOOD_MORNING' -> 'GOOD MORNING') into natural words
  return rawWord.replace(/_/g, ' ').trim();
}

function isHandActive(frame, offset) {
  return Boolean(frame && (frame[offset] !== 0 || frame[offset + 1] !== 0));
}

function getFrameHandCentroid(frame, offset) {
  const indices = [0, 5, 9, 13, 17];
  let x = 0;
  let y = 0;
  for (const idx of indices) {
    x += frame[offset + idx * 3];
    y += frame[offset + idx * 3 + 1];
  }
  return { x: x / 5, y: y / 5 };
}

function swapHandsInFrame(frame) {
  const swapped = new Array(126);
  for (let i = 0; i < 63; i++) {
    swapped[i] = frame[63 + i];
    swapped[63 + i] = frame[i];
  }
  return swapped;
}

export function useGestureRecognition({ enabled = false, initialMode = 'letter', _videoElement = null, onSendMessage = null } = {}) {
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
    'AGAIN', 'BAD', 'BOY', 'BYE_BYE', 'CHILD', 'CORRECT', 'DAY', 'DEAF', 'DIFFICULT', 'DOCTOR',
    'EASY', 'FEAR', 'FOOD', 'GIRL', 'GOOD', 'GOOD_AFTERNOON', 'GOOD_EVENING', 'GOOD_MORNING', 'GOOD_NIGHT',
    'HE', 'HEARING', 'HELLO', 'HELP', 'HOW_ARE_YOU', 'IM_FINE', 'INDIA', 'I_DONT_UNDERSTAND',
    'LANGUAGE', 'MAN', 'ME', 'MORNING', 'MY_NAME_IS', 'NAMASTE', 'NO', 'NO_FEAR', 'PEACE',
    'PLEASE', 'PRACTICE', 'REMEMBER', 'SHE', 'SIGN', 'SORRY', 'STRONG', 'TEACHER', 'THANK_YOU',
    'THANK_YOU_VERY_MUCH', 'THIN', 'UNDERSTAND', 'WASHROOM', 'WATER', 'WEAK', 'WELCOME', 'WHERE',
    'WOMAN', 'WRONG', 'YES', 'YOU'
  ]);

  // Inactivity / Hand-Drop Auto-Send states (5-second countdown timer)
  const [autoSendEnabled, setAutoSendEnabled] = useState(true);
  const [autoSendTimeoutMs, setAutoSendTimeoutMs] = useState(5000);
  const [inactivityCountdown, setInactivityCountdown] = useState(null);
  const [isAutoSending, setIsAutoSending] = useState(false);

  // Synchronous Sentence Buffer & Ref
  const sentenceBufferRef = useRef('');
  sentenceBufferRef.current = sentenceBuffer;
  const setSentenceBufferSync = useCallback((updater) => {
    setSentenceBuffer(prev => {
      const next = typeof updater === 'function' ? updater(prev) : updater;
      sentenceBufferRef.current = next;
      return next;
    });
  }, []);

  // Auto-Space Between Words states & stable refs
  const [autoSpaceWords, setAutoSpaceWords] = useState(true);
  const [autoWordSpaceDelayMs, setAutoWordSpaceDelayMs] = useState(900);
  const autoSpaceTimerRef = useRef(null);
  const autoSpaceWordsRef = useRef(autoSpaceWords);
  const autoWordSpaceDelayMsRef = useRef(autoWordSpaceDelayMs);

  useEffect(() => {
    autoSpaceWordsRef.current = autoSpaceWords;
  }, [autoSpaceWords]);

  useEffect(() => {
    autoWordSpaceDelayMsRef.current = autoWordSpaceDelayMs;
  }, [autoWordSpaceDelayMs]);

  const clearAutoSpaceTimer = useCallback(() => {
    if (autoSpaceTimerRef.current) {
      clearTimeout(autoSpaceTimerRef.current);
      autoSpaceTimerRef.current = null;
    }
  }, []);

  // Temporal smoothing & buffering refs
  const consecutiveLetterRef = useRef({ letter: null, count: 0 });
  const consecutiveWordRef = useRef({ word: null, count: 0 });
  const lastCommitRef = useRef(0);
  const lastCommittedItemRef = useRef(null);
  const frameBufferRef = useRef(new RingBuffer(WORD_SEQUENCE_LENGTH));
  const wordInferenceCooldownRef = useRef(0);
  const letterInferenceCooldownRef = useRef(0);
  const isRequestPendingRef = useRef(false);
  const isWordRequestPendingRef = useRef(false);
  const bodyAnchorsRef = useRef(null);
  const handsMissingCountRef = useRef(0);

  const wsRef = useRef(null);
  const pendingWsRequestsRef = useRef(new Map());
  const wsReqIdRef = useRef(0);

  // Stabilize detected sign and onSendMessage across renders
  const detectedRef = useRef(null);
  useEffect(() => {
    detectedRef.current = detectedWord || detectedLetter;
  }, [detectedWord, detectedLetter]);

  const onSendMessageRef = useRef(onSendMessage);
  useEffect(() => {
    onSendMessageRef.current = onSendMessage;
  }, [onSendMessage]);

  // Inactivity auto-send timers & refs
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
    setIsAutoSending(true);
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
      if (onSendMessageRef.current) {
        onSendMessageRef.current(speechText, 'human');
      }

      // 4. Clear draft buffer
      setSentenceBufferSync('');
    } catch (e) {
      console.warn('Auto-send execution error:', e);
    } finally {
      isAutoSendingRef.current = false;
      setIsAutoSending(false);
      cancelInactivityCountdown();
    }
  }, [cancelInactivityCountdown, setSentenceBufferSync]);

  /**
   * Start or maintain the 5-second countdown timer when no gesture is detected.
   * If forceRestart is true, restarts the timer at 5.0s.
   * If already running and forceRestart is false, lets the timer continue counting down.
   */
  const startOrResetInactivityTimer = useCallback((forceRestart = false) => {
    const rawText = sentenceBufferRef.current.trim();
    if (!autoSendEnabled || !rawText || isAutoSendingRef.current) {
      return;
    }

    // If timer is already running and forceRestart is false, do not disrupt the countdown
    if (autoSendTimerRef.current && !forceRestart) {
      return;
    }

    // Reset any existing timer & interval
    if (autoSendTimerRef.current) {
      clearTimeout(autoSendTimerRef.current);
      autoSendTimerRef.current = null;
    }
    if (countdownIntervalRef.current) {
      clearInterval(countdownIntervalRef.current);
      countdownIntervalRef.current = null;
    }

    const startTime = Date.now();
    const targetTime = startTime + autoSendTimeoutMs;

    setInactivityCountdown((autoSendTimeoutMs / 1000).toFixed(1));

    countdownIntervalRef.current = setInterval(() => {
      const remainingMs = targetTime - Date.now();
      if (remainingMs <= 0) {
        if (countdownIntervalRef.current) {
          clearInterval(countdownIntervalRef.current);
          countdownIntervalRef.current = null;
        }
        setInactivityCountdown('0.0');
      } else {
        setInactivityCountdown((remainingMs / 1000).toFixed(1));
      }
    }, 100);

    autoSendTimerRef.current = setTimeout(() => {
      autoSendTimerRef.current = null;
      triggerAutoSend();
    }, autoSendTimeoutMs);
  }, [autoSendEnabled, autoSendTimeoutMs, triggerAutoSend]);

  const handleHandsDropped = useCallback(() => {
    // If auto-space is enabled, automatically insert space after the last completed word
    if (autoSpaceWordsRef.current) {
      clearAutoSpaceTimer();
      setSentenceBufferSync(prev => {
        const trimmed = prev.trimEnd();
        if (trimmed.length > 0 && !prev.endsWith(' ')) {
          return trimmed + ' ';
        }
        return prev;
      });
      lastCommittedItemRef.current = null;
    }

    // No gesture detected (hands dropped): start or continue 5-second countdown
    startOrResetInactivityTimer(false);
  }, [startOrResetInactivityTimer, clearAutoSpaceTimer, setSentenceBufferSync]);

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
      frameBufferRef.current.clear();
      setWordBufferCount(0);
      cancelInactivityCountdown();
    } else {
      setStatus(STATES.SEARCHING);
      frameBufferRef.current.clear();
      setWordBufferCount(0);
    }
  }, [enabled, recognitionMode, cancelInactivityCountdown]);

  // Persistent WebSocket connection for sub-2ms low-latency inference
  useEffect(() => {
    let ws = null;
    let reconnectTimeout = null;
    let isCancelled = false;

    function connect() {
      if (isCancelled) return;
      try {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        // Connect via Vite proxy /ws in dev or direct to port 5001
        const wsUrl = window.location.port === '5173'
          ? `${protocol}//${window.location.host}/ws`
          : `${protocol}//${window.location.hostname}:5001`;

        ws = new WebSocket(wsUrl);
        wsRef.current = ws;

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            const reqId = data.req_id;
            if (reqId !== undefined && pendingWsRequestsRef.current.has(reqId)) {
              const resolver = pendingWsRequestsRef.current.get(reqId);
              pendingWsRequestsRef.current.delete(reqId);
              resolver(data);
            }
          } catch (e) {
            console.debug('WS parse notice:', e);
          }
        };

        ws.onclose = () => {
          wsRef.current = null;
          if (!isCancelled) {
            reconnectTimeout = setTimeout(connect, 2000);
          }
        };

        ws.onerror = () => {
          try { ws.close(); } catch {}
        };
      } catch {
        if (!isCancelled) {
          reconnectTimeout = setTimeout(connect, 3000);
        }
      }
    }

    connect();

    return () => {
      isCancelled = true;
      if (reconnectTimeout) clearTimeout(reconnectTimeout);
      if (ws) {
        try { ws.close(); } catch {}
      }
      wsRef.current = null;
      pendingWsRequestsRef.current.clear();
    };
  }, []);

  /**
   * Send inference request over WebSocket pipe (1-3ms) with seamless HTTP fallback.
   */
  const predictViaSocketOrHttp = useCallback(async (payload, httpEndpoint) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      return new Promise((resolve) => {
        const reqId = ++wsReqIdRef.current;
        payload.req_id = reqId;

        const timeout = setTimeout(async () => {
          pendingWsRequestsRef.current.delete(reqId);
          // Fallback to HTTP if socket request times out
          try {
            const res = await fetch(`${API_BASE}${httpEndpoint}`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(payload),
            });
            resolve(res.ok ? await res.json() : null);
          } catch {
            resolve(null);
          }
        }, 300);

        pendingWsRequestsRef.current.set(reqId, (data) => {
          clearTimeout(timeout);
          resolve(data);
        });

        ws.send(JSON.stringify(payload));
      });
    }

    // Default HTTP fallback
    try {
      const res = await fetch(`${API_BASE}${httpEndpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      return res.ok ? await res.json() : null;
    } catch {
      return null;
    }
  }, []);

  /**
   * Process a frame of 126 landmark floats from MediaPipe Holistic (with upper-body anchors).
   */
  const processLandmarks = useCallback(async (landmarks, handCount = 0, handedness = null, holisticData = null) => {
    if (!enabled) return null;

    if (holisticData && holisticData.bodyAnchors) {
      bodyAnchorsRef.current = holisticData.bodyAnchors;
    }

    const hasActiveLandmarks = Boolean(handCount > 0 && isValidHandGeometry(landmarks));

    // Update hand status with tracking grace period
    if (handCount > 0 && hasActiveLandmarks) {
      handsMissingCountRef.current = 0;
      const label = handedness || (handCount >= 2 ? 'Both Hands (ISL)' : 'Single Hand');
      setHandInfo(p => (p && p.count === handCount && p.label === label ? p : { count: handCount, label }));
      setStatus(prev => (prev === STATES.DETECTED ? prev : STATES.DETECTED));
    } else {
      handsMissingCountRef.current += 1;
      // Grace period: allow up to 6 dropped tracking frames (~250ms) without clearing the word buffer
      if (handsMissingCountRef.current < 6 && frameBufferRef.current.length > 0) {
        return null; // Retain buffer during momentary tracking flicker
      }
      setHandInfo(null);
      setStatus(prev => (prev === STATES.SEARCHING ? prev : STATES.SEARCHING));
      setDetectedLetter(null);
      setDetectedWord(null);
      setConfidence(0);
      setAllScores({});
      setGuidance(prev => (prev === 'Show your hands inside the camera frame.' ? prev : 'Show your hands inside the camera frame.'));
      consecutiveLetterRef.current = { letter: null, count: 0 };
      consecutiveWordRef.current = { word: null, count: 0 };
      lastCommittedItemRef.current = null;
      frameBufferRef.current.clear();
      setWordBufferCount(0);

      // Trigger hand-drop countdown if sentence buffer has words
      handleHandsDropped();
      return null;
    }

    // Adaptive Guidance for Single-Hand vs Dual-Hand Gestures
    if (handCount === 1) {
      const g1 = 'Single hand detected (e.g. C, L, O, V, numbers 0–9)';
      setGuidance(prev => (prev === g1 ? prev : g1));
    } else if (handCount >= 2) {
      const guidanceMsg = (landmarks && landmarks.length >= 126 ? analyzeQuality(landmarks) : null) || 'Dual-hand ISL tracking active';
      setGuidance(prev => (prev === guidanceMsg ? prev : guidanceMsg));
    }

    setStatus(prev => (prev === STATES.TRACKING ? prev : STATES.TRACKING));

    // ─────────────────────────────────────────────────────────────────────────
    // MODE 1: WORD SEQUENCE RECOGNITION (Sliding 30-Frame Window)
    // ─────────────────────────────────────────────────────────────────────────
    if (recognitionMode === 'word') {
      // Maintain hand slot continuity across momentary occlusion or crossing
      let frameToPush = landmarks;
      if (frameBufferRef.current.length > 0 && landmarks && landmarks.length >= 126) {
        const prevFrame = frameBufferRef.current.last();
        const prevHasL = isHandActive(prevFrame, 0);
        const prevHasR = isHandActive(prevFrame, 63);
        const curHasL = isHandActive(landmarks, 0);
        const curHasR = isHandActive(landmarks, 63);

        if (prevHasL && prevHasR && curHasL && curHasR) {
          const prevL = getFrameHandCentroid(prevFrame, 0);
          const prevR = getFrameHandCentroid(prevFrame, 63);
          const curL = getFrameHandCentroid(landmarks, 0);
          const curR = getFrameHandCentroid(landmarks, 63);

          const distDirect = Math.hypot(prevL.x - curL.x, prevL.y - curL.y) + Math.hypot(prevR.x - curR.x, prevR.y - curR.y);
          const distCross = Math.hypot(prevL.x - curR.x, prevL.y - curR.y) + Math.hypot(prevR.x - curL.x, prevR.y - curL.y);

          if (distCross + 0.05 < distDirect && distDirect > 0.15) {
            frameToPush = swapHandsInFrame(landmarks);
          }
        } else if (!prevHasL && prevHasR && curHasL && !curHasR) {
          const prevR = getFrameHandCentroid(prevFrame, 63);
          const curL = getFrameHandCentroid(landmarks, 0);
          if (Math.hypot(prevR.x - curL.x, prevR.y - curL.y) < 0.20) {
            frameToPush = swapHandsInFrame(landmarks);
          }
        } else if (prevHasL && !prevHasR && !curHasL && curHasR) {
          const prevL = getFrameHandCentroid(prevFrame, 0);
          const curR = getFrameHandCentroid(landmarks, 63);
          if (Math.hypot(prevL.x - curR.x, prevL.y - curR.y) < 0.20) {
            frameToPush = swapHandsInFrame(landmarks);
          }
        }
      }

      // Append current frame to rolling sequence buffer
      frameBufferRef.current.push(frameToPush); // O(1) — ring buffer, no shift
      const currentBufLen = frameBufferRef.current.length;
      if (currentBufLen % 3 === 0 || currentBufLen >= WORD_SEQUENCE_LENGTH) {
        setWordBufferCount(currentBufLen);
      }

      // Start inference as soon as minimum viable sequence (>= 10 frames, ~0.33s) is reached
      const now = Date.now();
      const hasViableSequence = frameBufferRef.current.length >= 10;
      const isCooldownOver = (now - wordInferenceCooldownRef.current >= 200); // ponytail: 200ms halves round-trips; lower when model inference < 50ms

      if (hasViableSequence && isCooldownOver && !isWordRequestPendingRef.current) {
        wordInferenceCooldownRef.current = now;
        isWordRequestPendingRef.current = true;

        // If sequence has 15-29 frames, linearly resample to exactly 30 frames for the model
        let framesToSend = frameBufferRef.current.toArray();
        if (framesToSend.length < WORD_SEQUENCE_LENGTH) {
          const N = framesToSend.length;
          const indices = Array.from({ length: WORD_SEQUENCE_LENGTH }, (_, i) =>
            Math.min(N - 1, Math.round(i * (N - 1) / (WORD_SEQUENCE_LENGTH - 1)))
          );
          framesToSend = indices.map(idx => framesToSend[idx]);
        }

        try {
          const prediction = await predictViaSocketOrHttp({
            type: 'word',
            frames: framesToSend,
            body_anchors: bodyAnchorsRef.current,
          }, '/translate/word');

          if (!prediction) {
            startOrResetInactivityTimer(false);
            return null;
          }

          // Handle confidence rejection from backend
          if (prediction.rejected || !prediction.word || prediction.word === '?' || prediction.confidence < CONFIDENCE_THRESHOLD_WORD) {
            setStatus(prev => (prev === STATES.TRACKING ? prev : STATES.TRACKING));
            setDetectedWord(null);
            setDetectedLetter(null);
            setConfidence(prev => (Math.abs(prev - (prediction.confidence || 0)) < 0.02 ? prev : (prediction.confidence || 0)));
            setGuidance(prev => (prev === 'Low confidence — adjust your hand position.' ? prev : 'Low confidence — adjust your hand position.'));
            consecutiveWordRef.current = { word: null, count: 0 };
            // No gesture detected: start or maintain 5-second countdown
            startOrResetInactivityTimer(false);
            return prediction;
          }

          // Active gesture detected: pause/cancel countdown
          cancelInactivityCountdown();
          setStatus(prev => (prev === STATES.RECOGNISING ? prev : STATES.RECOGNISING));
          setDetectedWord(prediction.word);
          setDetectedLetter(null);
          setConfidence(prediction.confidence);
          setAllScores(prediction.all_scores || {});

          // Temporal smoothing & kinetic stroke completion gating for whole words
          if (prediction.confidence >= CONFIDENCE_THRESHOLD_WORD) {
            const prev = consecutiveWordRef.current;
            if (prev.word === prediction.word) {
              prev.count += 1;
            } else {
              consecutiveWordRef.current = { word: prediction.word, count: 1 };
            }

            // Measure recent kinetic velocity to detect gesture stroke completion / deceleration
            let strokeEnergy = 0;
            if (frameBufferRef.current.length >= 2) {
              const currF = frameBufferRef.current.last();
              const prevF = frameBufferRef.current.last2();
              let diffSum = 0;
              let validPts = 0;
              for (let p = 0; p < Math.min(currF.length, prevF.length); p += 3) {
                if (currF[p] !== 0 || prevF[p] !== 0) {
                  const dx = currF[p] - prevF[p];
                  const dy = currF[p + 1] - prevF[p + 1];
                  const dz = currF[p + 2] - prevF[p + 2];
                  diffSum += Math.sqrt(dx * dx + dy * dy + dz * dz);
                  validPts++;
                }
              }
              strokeEnergy = validPts > 0 ? diffSum / validPts : 0;
            }

            // Commit word when gesture decelerates (stroke completion) or has strong confidence / stability
            const isStrokeComplete = strokeEnergy < 0.080 || prediction.confidence >= 0.78 || consecutiveWordRef.current.count >= 2;

            if (isStrokeComplete && (consecutiveWordRef.current.count >= 2 || prediction.confidence >= 0.85)) {
              const formattedWord = formatWordForSentence(prediction.word);
              const isDifferentWord = (lastCommittedItemRef.current !== prediction.word);
              const canRepeatSameWord = (now - lastCommitRef.current > 1800);

              if (now - lastCommitRef.current > 1100 && (isDifferentWord || canRepeatSameWord)) {
                setStatus(STATES.STABLE);
                clearAutoSpaceTimer();
                setSentenceBufferSync(prev => {
                  const clean = prev.trim();
                  // Automatically place space between words to form a word-level sentence
                  return clean.length > 0 ? `${clean} ${formattedWord}` : formattedWord;
                });
                lastCommitRef.current = now;
                lastCommittedItemRef.current = prediction.word;
                consecutiveWordRef.current = { word: null, count: 0 };
                frameBufferRef.current.clear(); // Clear buffer after successful word lock
                setWordBufferCount(0);
                // Word locked: start 5-second countdown to automatically send draft message
                startOrResetInactivityTimer(true);
              }
            }
          } else {
            consecutiveWordRef.current = { word: null, count: 0 };
          }

          return prediction;
        } catch (e) {
          console.warn('Word recognition error:', e);
          startOrResetInactivityTimer(false);
          return null;
        } finally {
          isWordRequestPendingRef.current = false;
        }
      } else {
        // Viable sequence not reached or cooldown active: check if waiting
        if (!detectedRef.current) {
          startOrResetInactivityTimer(false);
        }
      }
      return null;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // MODE 2: STATIC LETTER RECOGNITION (A-Z)
    // ─────────────────────────────────────────────────────────────────────────


    const now = Date.now();
    if (now - letterInferenceCooldownRef.current < 35 || isRequestPendingRef.current) return null;
    letterInferenceCooldownRef.current = now;
    isRequestPendingRef.current = true;

    try {
      const r = v => Math.round(v * 1e4) / 1e4;
      const roundedLandmarks = landmarks.map(r);
      const prediction = await predictViaSocketOrHttp({
        type: 'letter',
        landmarks: roundedLandmarks,
        body_anchors: bodyAnchorsRef.current,
      }, '/translate');

      if (!prediction) {
        startOrResetInactivityTimer(false);
        return null;
      }

      // Handle confidence rejection or low confidence from backend
      if (prediction.rejected || !prediction.letter || prediction.letter === '?' || prediction.confidence < CONFIDENCE_THRESHOLD_LETTER) {
        setStatus(prev => (prev === STATES.TRACKING ? prev : STATES.TRACKING));
        setDetectedLetter(null);
        setDetectedWord(null);
        setConfidence(prev => (Math.abs(prev - (prediction.confidence || 0)) < 0.02 ? prev : (prediction.confidence || 0)));
        setAllScores(prediction.all_scores || {});
        consecutiveLetterRef.current = { letter: null, count: 0 };
        // No gesture detected: start or maintain 5-second countdown
        startOrResetInactivityTimer(false);
        return prediction;
      }

      // Valid gesture detected: pause/cancel countdown
      cancelInactivityCountdown();

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
        const nextStatus = consecutiveLetterRef.current.count >= 2 ? STATES.RECOGNISING : STATES.TRACKING;
        setStatus(prev => (prev === nextStatus ? prev : nextStatus));
        setDetectedLetter(prediction.letter);
        setDetectedWord(null);
        setConfidence(prediction.confidence);
        setAllScores(prediction.all_scores || {});
      }

      // Commit letter after reaching stability threshold
      if (consecutiveLetterRef.current.count >= STABILITY_THRESHOLD) {
        const now = Date.now();
        if (now - lastCommitRef.current > 300) {
          if (lastCommittedItemRef.current !== prediction.letter) {
            setStatus(STATES.STABLE);
            clearAutoSpaceTimer();
            setSentenceBufferSync(prevBuf => prevBuf + prediction.letter);
            lastCommitRef.current = now;
            lastCommittedItemRef.current = prediction.letter;
            consecutiveLetterRef.current = { letter: null, count: 0 };

            // Letter locked: start 5-second countdown to automatically send draft message
            startOrResetInactivityTimer(true);

            // Automatic word-spacing: when user pauses between words, automatically append space
            if (autoSpaceWordsRef.current) {
              autoSpaceTimerRef.current = setTimeout(() => {
                setSentenceBufferSync(prev => {
                  const trimmed = prev.trimEnd();
                  if (trimmed.length > 0 && !prev.endsWith(' ')) {
                    return trimmed + ' ';
                  }
                  return prev;
                });
                lastCommittedItemRef.current = null;
              }, autoWordSpaceDelayMsRef.current);
            }
          }
        }
      }

      return prediction;
    } catch (e) {
      console.warn('Letter recognition API error:', e);
      startOrResetInactivityTimer(false);
      return null;
    } finally {
      isRequestPendingRef.current = false;
    }
  }, [enabled, recognitionMode, cancelInactivityCountdown, startOrResetInactivityTimer, handleHandsDropped, clearAutoSpaceTimer, setSentenceBufferSync]);

  // Sentence buffer actions
  const undoLetter = useCallback(() => {
    setSentenceBufferSync(prev => {
      const trimmed = prev.trimEnd();
      const lastSpaceIdx = trimmed.lastIndexOf(' ');
      const next = lastSpaceIdx !== -1 ? trimmed.substring(0, lastSpaceIdx + 1) : prev.slice(0, -1);
      if (!next.trim()) {
        cancelInactivityCountdown();
      } else {
        startOrResetInactivityTimer(true);
      }
      return next;
    });
  }, [setSentenceBufferSync, cancelInactivityCountdown, startOrResetInactivityTimer]);

  const deleteLetter = useCallback(() => {
    setSentenceBufferSync(prev => {
      const next = prev.slice(0, -1);
      if (!next.trim()) {
        cancelInactivityCountdown();
      } else {
        startOrResetInactivityTimer(true);
      }
      return next;
    });
  }, [setSentenceBufferSync, cancelInactivityCountdown, startOrResetInactivityTimer]);

  const clearBuffer = useCallback(() => {
    clearAutoSpaceTimer();
    cancelInactivityCountdown();
    setSentenceBufferSync('');
    consecutiveLetterRef.current = { letter: null, count: 0 };
    consecutiveWordRef.current = { word: null, count: 0 };
    lastCommittedItemRef.current = null;
    frameBufferRef.current.clear();
    setWordBufferCount(0);
  }, [clearAutoSpaceTimer, cancelInactivityCountdown, setSentenceBufferSync]);

  const addSpace = useCallback(() => {
    setSentenceBufferSync(prev => (prev.endsWith(' ') ? prev : prev + ' '));
  }, [setSentenceBufferSync]);

  const updateSentence = useCallback((newText) => {
    setSentenceBufferSync(newText);
    if (!newText.trim()) {
      cancelInactivityCountdown();
    } else {
      startOrResetInactivityTimer(true);
    }
  }, [setSentenceBufferSync, cancelInactivityCountdown, startOrResetInactivityTimer]);

  const refineSentence = useCallback(async () => {
    const rawText = sentenceBufferRef.current.trim();
    if (!rawText) return null;
    try {
      const res = await fetch(`${API_BASE}/llm/refine`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: rawText })
      });
      if (res.ok) {
        const data = await res.json();
        if (data.refined_sentence) {
          setSentenceBufferSync(data.refined_sentence);
          startOrResetInactivityTimer(true);
          return data;
        }
      }
    } catch (e) {
      console.warn('Refine LLM error:', e);
    }
    return null;
  }, [setSentenceBufferSync, startOrResetInactivityTimer]);

  const commitSentence = useCallback(() => {
    const text = sentenceBufferRef.current.trim();
    cancelInactivityCountdown();
    setSentenceBufferSync('');
    return text;
  }, [cancelInactivityCountdown, setSentenceBufferSync]);

  const sendSentence = useCallback(async () => {
    const rawText = sentenceBufferRef.current.trim();
    if (!rawText) return;
    cancelInactivityCountdown();
    setIsAutoSending(true);
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
    } finally {
      setIsAutoSending(false);
    }
    textToSend = textToSend.replace(/<think>[\s\S]*?(?:<\/think>|$)/gi, '').trim() || rawText;
    if (onSendMessage) {
      onSendMessage(textToSend, 'human');
    }
    setSentenceBufferSync('');
  }, [cancelInactivityCountdown, onSendMessage, setSentenceBufferSync]);

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
    availableWords,
    // Auto-Space Between Words states
    autoSpaceWords,
    setAutoSpaceWords,
    autoWordSpaceDelayMs,
    setAutoWordSpaceDelayMs,
    // Auto-Send / Inactivity states
    autoSendEnabled,
    setAutoSendEnabled,
    autoSendTimeoutMs,
    setAutoSendTimeoutMs,
    inactivityCountdown,
    lastAutoSpoken: null,
    isAutoSending,

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