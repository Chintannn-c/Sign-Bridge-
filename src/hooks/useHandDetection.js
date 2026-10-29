import { useState, useEffect, useRef, useCallback } from 'react';
import { LandmarkSmoother } from '../utils/oneEuroFilter';

/**
 * Custom React Hook for MediaPipe Holistic integration.
 *
 * Extracts:
 *   - Dual-hand landmarks: 42 points (21 per hand x 3 coordinates = 126 floats)
 *   - Upper-body pose landmarks: 33 body points (shoulders, elbows, wrists, hips, head)
 *   - Kinematic body anchors: Hand-to-mouth, hand-to-chest, and shoulder-relative elevations.
 *
 * MediaPipe Holistic runs directly in the browser via CDN scripts,
 * with automatic fallback to MediaPipe Hands if Holistic CDN is unavailable.
 */

const MEDIAPIPE_HOLISTIC_CDN = 'https://cdn.jsdelivr.net/npm/@mediapipe/holistic@0.5.1675471629/holistic.min.js';
const MEDIAPIPE_HANDS_CDN = 'https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/hands.min.js';
const MEDIAPIPE_CAMERA = 'https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils@0.3.1675466862/camera_utils.min.js';

const HANDS_OPTIONS = {
  maxNumHands: 2,
  modelComplexity: 0, // Lite model for real-time high-speed motion tracking
  minDetectionConfidence: 0.25, // Matches training confidence for unbroken tracking when hands touch
  minTrackingConfidence: 0.25,
};

const HOLISTIC_OPTIONS = {
  modelComplexity: 0,
  smoothLandmarks: false,
  enableSegmentation: false,
  smoothSegmentation: false,
  refineFaceLandmarks: false,
  minDetectionConfidence: 0.25,
  minTrackingConfidence: 0.25,
};

function getHandCentroid(pts) {
  if (!pts || !pts[0]) return { x: 0, y: 0 };
  const indices = [0, 5, 9, 13, 17];
  let x = 0;
  let y = 0;
  for (const idx of indices) {
    x += pts[idx].x;
    y += pts[idx].y;
  }
  return { x: x / 5, y: y / 5 };
}

function distSq(a, b) {
  if (!a || !b) return 999;
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return dx * dx + dy * dy;
}

function trackAndSeparateHands(landmarksList, handednessList, prevHands) {
  let h0 = landmarksList[0];
  let h1 = landmarksList[1];
  let l0 = handednessList?.[0]?.label;
  let l1 = handednessList?.[1]?.label;

  const num = h1 ? 2 : (h0 ? 1 : 0);
  if (num === 0) {
    if (prevHands.left) prevHands.lostL = (prevHands.lostL || 0) + 1;
    if (prevHands.right) prevHands.lostR = (prevHands.lostR || 0) + 1;
    if (prevHands.lostL > 8) prevHands.left = null;
    if (prevHands.lostR > 8) prevHands.right = null;
    return { leftHand: null, rightHand: null, count: 0 };
  }

  let leftHand = null;
  let rightHand = null;

  if (num === 2) {
    const c0 = getHandCentroid(h0);
    const c1 = getHandCentroid(h1);

    const prevL = prevHands.left;
    const prevR = prevHands.right;
    const hasHistory = Boolean(prevL && prevR && (prevHands.lostL || 0) < 6 && (prevHands.lostR || 0) < 6);

    if (hasHistory) {
      const costA = distSq(c0, prevL.centroid) + distSq(c1, prevR.centroid);
      const costB = distSq(c1, prevL.centroid) + distSq(c0, prevR.centroid);

      const biasA = (l0 === 'Left' ? -0.015 : 0) + (l1 === 'Right' ? -0.015 : 0);
      const biasB = (l1 === 'Left' ? -0.015 : 0) + (l0 === 'Right' ? -0.015 : 0);

      if (costA + biasA <= costB + biasB) {
        leftHand = h0;
        rightHand = h1;
      } else {
        leftHand = h1;
        rightHand = h0;
      }
    } else {
      if (l0 && l1 && l0 !== l1) {
        leftHand = (l0 === 'Left') ? h0 : h1;
        rightHand = (l0 === 'Left') ? h1 : h0;
      } else {
        leftHand = (c0.x >= c1.x) ? h0 : h1;
        rightHand = (c0.x >= c1.x) ? h1 : h0;
      }
    }
  } else {
    // num === 1
    const c0 = getHandCentroid(h0);
    const prevL = prevHands.left;
    const prevR = prevHands.right;

    if (prevL && prevR && (prevHands.lostL || 0) < 5 && (prevHands.lostR || 0) < 5) {
      const dL = distSq(c0, prevL.centroid);
      const dR = distSq(c0, prevR.centroid);
      if (dL < dR * 0.7) {
        leftHand = h0;
      } else if (dR < dL * 0.7) {
        rightHand = h0;
      } else {
        if (l0 === 'Left') leftHand = h0;
        else rightHand = h0;
      }
    } else if (prevL && (prevHands.lostL || 0) < 5 && (!prevR || (prevHands.lostR || 0) >= 5)) {
      leftHand = h0;
    } else if (prevR && (prevHands.lostR || 0) < 5 && (!prevL || (prevHands.lostL || 0) >= 5)) {
      rightHand = h0;
    } else {
      if (l0 === 'Left') leftHand = h0;
      else rightHand = h0;
    }
  }

  // Update temporal tracking state
  if (leftHand) {
    prevHands.left = { centroid: getHandCentroid(leftHand), wrist: leftHand[0] };
    prevHands.lostL = 0;
  } else {
    prevHands.lostL = (prevHands.lostL || 0) + 1;
    if (prevHands.lostL > 8) prevHands.left = null;
  }

  if (rightHand) {
    prevHands.right = { centroid: getHandCentroid(rightHand), wrist: rightHand[0] };
    prevHands.lostR = 0;
  } else {
    prevHands.lostR = (prevHands.lostR || 0) + 1;
    if (prevHands.lostR > 8) prevHands.right = null;
  }

  return { leftHand, rightHand, count: num };
}

function flattenHands(leftHand, rightHand) {
  const rawFlat = new Array(126);
  for (let i = 0; i < 21; i++) {
    const idx = i * 3;
    if (leftHand && leftHand[i]) {
      rawFlat[idx] = leftHand[i].x;
      rawFlat[idx + 1] = leftHand[i].y;
      rawFlat[idx + 2] = leftHand[i].z || 0;
    } else {
      rawFlat[idx] = 0;
      rawFlat[idx + 1] = 0;
      rawFlat[idx + 2] = 0;
    }
  }
  for (let i = 0; i < 21; i++) {
    const idx = 63 + i * 3;
    if (rightHand && rightHand[i]) {
      rawFlat[idx] = rightHand[i].x;
      rawFlat[idx + 1] = rightHand[i].y;
      rawFlat[idx + 2] = rightHand[i].z || 0;
    } else {
      rawFlat[idx] = 0;
      rawFlat[idx + 1] = 0;
      rawFlat[idx + 2] = 0;
    }
  }
  return rawFlat;
}

export function useHandDetection({
  videoElement,
  enabled = false,
  onLandmarks = null,
  throttleMs = 35,
  preferredPipeline = 'hands',
}) {
  const [pipelineMode, setPipelineMode] = useState(preferredPipeline); // 'hands' | 'holistic'
  const [isLoaded, setIsLoaded] = useState(false);
  const [isDetecting, setIsDetecting] = useState(false);
  const [landmarkData, setLandmarkData] = useState(null);
  const [poseData, setPoseData] = useState(null);
  const [bodyAnchors, setBodyAnchors] = useState(null);
  const [handCount, setHandCount] = useState(0);
  const [error, setError] = useState(null);
  const [fps, setFps] = useState(0);
  const fpsFrameCountRef = useRef(0);
  const lastFpsTimeRef = useRef(performance.now());

  const trackerRef = useRef(null);
  const lastCallRef = useRef(0);
  const lastUiRef = useRef(0);
  const isProcessingRef = useRef(false);
  const smootherRef = useRef(new LandmarkSmoother(42, 1.0, 0.07));
  const prevHandsRef = useRef({ left: null, right: null, lostL: 0, lostR: 0 });
  // Raw landmarks ref: written every MediaPipe frame, read by overlay rAF loop (zero-lag display)
  const rawLandmarksRef = useRef({ landmarks: null, poseLandmarks: null, bodyAnchors: null });
  const frameCaptureTimeRef = useRef(0);

  const togglePipelineMode = useCallback(() => {
    setPipelineMode(prev => (prev === 'hands' ? 'holistic' : 'hands'));
  }, []);

  const latestProps = useRef({ enabled, onLandmarks, throttleMs });
  useEffect(() => {
    latestProps.current = { enabled, onLandmarks, throttleMs };
  }, [enabled, onLandmarks, throttleMs]);

  // Load MediaPipe Scripts via CDN (guards against duplicate tags & StrictMode remount)
  const loadScript = useCallback((src) => {
    return new Promise((resolve, reject) => {
      const existing = document.querySelector(`script[src="${src}"]`);
      if (existing) {
        if (existing.dataset.loaded === 'true') {
          resolve();
          return;
        }
        existing.addEventListener('load', () => resolve(), { once: true });
        existing.addEventListener('error', (err) => reject(err), { once: true });
        return;
      }
      const script = document.createElement('script');
      script.src = src;
      script.async = true;
      script.onload = () => {
        script.dataset.loaded = 'true';
        resolve();
      };
      script.onerror = (err) => reject(err);
      document.head.appendChild(script);
    });
  }, []);

  // Initialize MediaPipe Tracker
  useEffect(() => {
    if (!enabled) return;

    let cancelled = false;

    async function init() {
      try {
        await loadScript(MEDIAPIPE_CAMERA);

        if (pipelineMode === 'holistic') {
          await loadScript(MEDIAPIPE_HOLISTIC_CDN);
          if (cancelled) return;

          if (typeof window.Holistic !== 'undefined') {
            const holistic = new window.Holistic({
              locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic@0.5.1675471629/${file}`,
            });
            holistic.setOptions(HOLISTIC_OPTIONS);

            holistic.onResults((results) => {
              const props = latestProps.current;
              if (!props.enabled) {
                setIsDetecting(false);
                setLandmarkData(null);
                setPoseData(null);
                setBodyAnchors(null);
                return;
              }

              let leftHand = results.leftHandLandmarks || null;
              let rightHand = results.rightHandLandmarks || null;
              const pose = results.poseLandmarks || null;

              const rawHands = [];
              const rawHandedness = [];
              if (leftHand) {
                rawHands.push(leftHand);
                rawHandedness.push({ label: 'Left', score: 0.9 });
              }
              if (rightHand) {
                rawHands.push(rightHand);
                rawHandedness.push({ label: 'Right', score: 0.9 });
              }

              const separation = trackAndSeparateHands(rawHands, rawHandedness, prevHandsRef.current);
              const nHands = separation.count;
              const hasDetection = Boolean(nHands > 0 || (pose && pose.length >= 25));

              const rawFlat = flattenHands(separation.leftHand, separation.rightHand);

              // Extract Upper-Body Anchors
              let anchors = null;
              if (pose && pose.length >= 25) {
                const nose = pose[0];
                const mouth = {
                  x: ((pose[9]?.x || 0) + (pose[10]?.x || 0)) / 2,
                  y: ((pose[9]?.y || 0) + (pose[10]?.y || 0)) / 2,
                  z: ((pose[9]?.z || 0) + (pose[10]?.z || 0)) / 2,
                };
                const leftShoulder = pose[11];
                const rightShoulder = pose[12];
                const chest = {
                  x: (leftShoulder.x + rightShoulder.x) / 2,
                  y: (leftShoulder.y + rightShoulder.y) / 2,
                  z: ((leftShoulder.z || 0) + (rightShoulder.z || 0)) / 2,
                };
                const leftElbow = pose[13];
                const rightElbow = pose[14];
                const leftWrist = pose[15];
                const rightWrist = pose[16];
                const leftHip = pose[23];
                const rightHip = pose[24];
                const shoulderDist = Math.hypot(leftShoulder.x - rightShoulder.x, leftShoulder.y - rightShoulder.y) || 1.0;

                const getHandDist = (handPts, target) => {
                  if (!handPts || !handPts[0]) return 999.0;
                  const wrist = handPts[0];
                  const tip = handPts[8] || wrist;
                  const dWrist = Math.hypot(wrist.x - target.x, wrist.y - target.y);
                  const dTip = Math.hypot(tip.x - target.x, tip.y - target.y);
                  return Math.min(dWrist, dTip) / shoulderDist;
                };

                anchors = {
                  nose, mouth, chest, leftShoulder, rightShoulder,
                  leftElbow, rightElbow, leftWrist, rightWrist, leftHip, rightHip,
                  shoulderDist,
                  l_hand_to_mouth: getHandDist(separation.leftHand, mouth),
                  r_hand_to_mouth: getHandDist(separation.rightHand, mouth),
                  l_hand_to_chest: getHandDist(separation.leftHand, chest),
                  r_hand_to_chest: getHandDist(separation.rightHand, chest),
                  l_elevation: separation.leftHand ? (leftShoulder.y - separation.leftHand[0].y) / shoulderDist : 0,
                  r_elevation: separation.rightHand ? (rightShoulder.y - separation.rightHand[0].y) / shoulderDist : 0,
                };
              }

              // Write raw landmarks to ref every frame for zero-lag overlay
              rawLandmarksRef.current = { landmarks: rawFlat, poseLandmarks: pose, bodyAnchors: anchors };

              const t = performance.now();
              const shouldUpdateUi = (t - lastUiRef.current > 40);
              if (shouldUpdateUi) {
                lastUiRef.current = t;
                setHandCount(nHands);
                setIsDetecting(hasDetection);
                setPoseData(pose);
                setBodyAnchors(anchors);
              }

              if (nHands > 0) {
                const smoothedLandmarks = smootherRef.current.smooth(rawFlat);
                if (shouldUpdateUi) {
                  setLandmarkData(smoothedLandmarks);
                }
                const now = Date.now();
                if (props.onLandmarks && (now - lastCallRef.current) >= props.throttleMs) {
                  lastCallRef.current = now;
                  const handednessStr = nHands >= 2 ? 'Both Hands' : (separation.leftHand ? 'Left' : 'Right');
                  try {
                    props.onLandmarks(smoothedLandmarks, nHands, handednessStr, {
                      poseLandmarks: pose,
                      bodyAnchors: anchors,
                      captureTime: frameCaptureTimeRef.current,
                    });
                  } catch (cbErr) {
                    console.error('Holistic onLandmarks callback error:', cbErr);
                  }
                }
              } else {
                if (shouldUpdateUi) {
                  setLandmarkData(null);
                }
                smootherRef.current.reset();
                if (props.onLandmarks && (!pose || pose.length === 0)) {
                  try {
                    props.onLandmarks(null, 0, null, null);
                  } catch (cbErr) {
                    console.error('Holistic empty onLandmarks callback error:', cbErr);
                  }
                }
              }
            });

            trackerRef.current = holistic;
            setIsLoaded(true);
            return;
          }
        }

        // Default & High-Performance: MediaPipe Hands
        await loadScript(MEDIAPIPE_HANDS_CDN);
        if (cancelled) return;

        if (typeof window.Hands !== 'undefined') {
          const hands = new window.Hands({
            locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/${file}`,
          });
          hands.setOptions(HANDS_OPTIONS);

          hands.onResults((results) => {
            const props = latestProps.current;
            if (!props.enabled) {
              setIsDetecting(false);
              setLandmarkData(null);
              setPoseData(null);
              setBodyAnchors(null);
              return;
            }

            const rawHands = results.multiHandLandmarks || [];
            const rawHandedness = results.multiHandedness || [];

            const separation = trackAndSeparateHands(rawHands, rawHandedness, prevHandsRef.current);
            const nHands = separation.count;
            const hasDetection = nHands > 0;

            const t = performance.now();
            const shouldUpdateUi = (t - lastUiRef.current > 33);
            if (shouldUpdateUi) {
              lastUiRef.current = t;
              setHandCount(nHands);
              setIsDetecting(hasDetection);
              setPoseData(null);
              setBodyAnchors(null);
            }

            if (nHands > 0) {
              const rawFlat = flattenHands(separation.leftHand, separation.rightHand);
              // Write raw landmarks to ref every frame for zero-lag overlay
              rawLandmarksRef.current = { landmarks: rawFlat, poseLandmarks: null, bodyAnchors: null };
              const smoothedLandmarks = smootherRef.current.smooth(rawFlat);

              if (shouldUpdateUi) {
                setLandmarkData(smoothedLandmarks);
              }

              const now = Date.now();
              if (props.onLandmarks && (now - lastCallRef.current) >= props.throttleMs) {
                lastCallRef.current = now;
                const handednessStr = nHands >= 2 ? 'Both Hands' : (separation.leftHand ? 'Left' : 'Right');
                try {
                  props.onLandmarks(smoothedLandmarks, nHands, handednessStr, {
                    poseLandmarks: null,
                    bodyAnchors: null,
                    captureTime: frameCaptureTimeRef.current,
                  });
                } catch (cbErr) {
                  console.error('Hands onLandmarks callback error:', cbErr);
                }
              }
            } else {
              if (shouldUpdateUi) {
                setLandmarkData(null);
              }
              smootherRef.current.reset();
              if (props.onLandmarks) {
                try {
                  props.onLandmarks(null, 0, null, null);
                } catch (cbErr) {
                  console.error('Hands empty onLandmarks callback error:', cbErr);
                }
              }
            }
          });

          trackerRef.current = hands;
          setIsLoaded(true);
        } else {
          setError('Model failed to load');
          console.error('MediaPipe Hands script loaded but window.Hands is undefined');
        }
      } catch (err) {
        setError('Model failed to load');
        console.error('Failed to initialize MediaPipe tracking pipeline:', err);
      }
    }

    init();

    return () => {
      cancelled = true;
      if (trackerRef.current && typeof trackerRef.current.close === 'function') {
        try {
          trackerRef.current.close();
        } catch {
          // ignore cleanup errors on unmount
        }
        trackerRef.current = null;
      }
      setIsLoaded(false);
      prevHandsRef.current = { left: null, right: null, lostL: 0, lostR: 0 };
    };
  }, [enabled, loadScript]);

  // Camera Frame Loop: requestVideoFrameCallback (zero-latency hardware camera sync) with rAF fallback
  useEffect(() => {
    if (!enabled || !isLoaded || !videoElement || !trackerRef.current) return;

    let cancelled = false;
    let rVFCId = null;
    let rAFId = null;
    let lastVideoTime = -1;

    const hasRVFC = typeof HTMLVideoElement !== 'undefined' && 'requestVideoFrameCallback' in HTMLVideoElement.prototype;

    const onFrame = async () => {
      if (cancelled) return;

      if (trackerRef.current && videoElement.readyState >= 2 && !isProcessingRef.current) {
        if (!hasRVFC || videoElement.currentTime !== lastVideoTime) {
          lastVideoTime = videoElement.currentTime;
          isProcessingRef.current = true;
          try {
            frameCaptureTimeRef.current = performance.now();
            await trackerRef.current.send({ image: videoElement });
            fpsFrameCountRef.current++;
            const nowT = performance.now();
            if (nowT - lastFpsTimeRef.current >= 1000) {
              setFps(Math.round((fpsFrameCountRef.current * 1000) / (nowT - lastFpsTimeRef.current)));
              fpsFrameCountRef.current = 0;
              lastFpsTimeRef.current = nowT;
            }
          } catch (sendErr) {
            console.warn('Frame processing dropped:', sendErr);
          } finally {
            isProcessingRef.current = false;
          }
        }
      }

      if (!cancelled) {
        if (hasRVFC && typeof videoElement.requestVideoFrameCallback === 'function') {
          rVFCId = videoElement.requestVideoFrameCallback(onFrame);
        } else {
          rAFId = requestAnimationFrame(onFrame);
        }
      }
    };

    if (hasRVFC && typeof videoElement.requestVideoFrameCallback === 'function') {
      rVFCId = videoElement.requestVideoFrameCallback(onFrame);
    } else {
      rAFId = requestAnimationFrame(onFrame);
    }

    return () => {
      cancelled = true;
      if (hasRVFC && rVFCId !== null && typeof videoElement.cancelVideoFrameCallback === 'function') {
        videoElement.cancelVideoFrameCallback(rVFCId);
      }
      if (rAFId) cancelAnimationFrame(rAFId);
    };
  }, [enabled, isLoaded, videoElement]);

  return {
    isLoaded,
    isDetecting,
    landmarkData,
    poseData,
    bodyAnchors,
    handCount,
    pipelineMode,
    togglePipelineMode,
    rawLandmarksRef,
    error,
    fps,
  };
}