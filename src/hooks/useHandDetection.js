import { useState, useEffect, useRef, useCallback } from 'react';
import { LandmarkSmoother } from '../utils/oneEuroFilter';

/**
 * Custom React Hook for MediaPipe Hands integration.
 *
 * Extracts 42 hand landmarks (21 per hand x 2 hands) from a live video feed.
 * Landmark data is passed to the onLandmarks callback in real-time for
 * translation by the Flask API.
 *
 * MediaPipe Hands runs directly in the browser via CDN scripts,
 * so no npm package installation is needed.
 *
 * Usage:
 *   const { isLoaded, isDetecting, landmarkData } = useHandDetection({
 *     videoRef: myVideoElement,
 *     enabled: true,
 *     onLandmarks: (landmarks) => translateLandmarks(landmarks)
 *   });
 */

const MEDIAPIPE_CDN = 'https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/hands.min.js';
const MEDIAPIPE_CAMERA = 'https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils@0.3.1675466862/camera_utils.min.js';

export function useHandDetection({ videoElement, enabled = false, onLandmarks = null, throttleMs = 120 }) {
  const [isLoaded, setIsLoaded] = useState(false);
  const [isDetecting, setIsDetecting] = useState(false);
  const [landmarkData, setLandmarkData] = useState(null);
  const [handCount, setHandCount] = useState(0);

  const handsRef = useRef(null);
  const lastCallRef = useRef(0);
  const scriptLoadedRef = useRef(false);
  const smootherRef = useRef(new LandmarkSmoother(42, 1.2, 0.005));

  const latestProps = useRef({ enabled, onLandmarks, throttleMs });
  useEffect(() => {
    latestProps.current = { enabled, onLandmarks, throttleMs };
  }, [enabled, onLandmarks, throttleMs]);

  // ─── Load MediaPipe Scripts via CDN ───────────────────────────────────
  const loadScript = useCallback((src) => {
    return new Promise((resolve, reject) => {
      // Check if already loaded
      const existing = document.querySelector(`script[src="${src}"]`);
      if (existing) {
        resolve();
        return;
      }

      const script = document.createElement('script');
      script.src = src;
      script.async = true;
      script.onload = resolve;
      script.onerror = reject;
      document.head.appendChild(script);
    });
  }, []);

  // ─── Initialize MediaPipe Hands ───────────────────────────────────────
  useEffect(() => {
    if (!enabled || scriptLoadedRef.current) return;

    let cancelled = false;

    async function init() {
      try {
        // Load MediaPipe CDN scripts
        await loadScript(MEDIAPIPE_CDN);
        await loadScript(MEDIAPIPE_CAMERA);

        if (cancelled) return;

        // Verify global Hands constructor is available
        if (typeof window.Hands === 'undefined') {
          console.warn('MediaPipe Hands not available after script load.');
          return;
        }

        // Create Hands instance
        const hands = new window.Hands({
          locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/${file}`
        });

        hands.setOptions({
          maxNumHands: 2,           // ISL requires both hands
          modelComplexity: 1,       // 1=Full model (higher accuracy for dual hands & overlapping fingers)
          minDetectionConfidence: 0.50, // Optimal threshold for robust hand detection without phantom noise
          minTrackingConfidence: 0.50,  // Optimal threshold for stable inter-frame finger tracking
        });

        hands.onResults((results) => {
          const props = latestProps.current;
          if (!props.enabled) {
            setIsDetecting(false);
            setLandmarkData(null);
            return;
          }

          const numHands = results.multiHandLandmarks ? results.multiHandLandmarks.length : 0;
          setHandCount(numHands);

          if (numHands >= 1) {
            setIsDetecting(true);

            // Determine left and right hand robustly
            let leftHand = null;
            let rightHand = null;

            if (numHands >= 2) {
              const h0 = results.multiHandLandmarks[0];
              const h1 = results.multiHandLandmarks[1];
              const label0 = results.multiHandedness?.[0]?.label;
              const label1 = results.multiHandedness?.[1]?.label;

              // If MediaPipe provides distinct handedness labels
              if (label0 && label1 && label0 !== label1) {
                if (label0 === 'Left') {
                  leftHand = h0;
                  rightHand = h1;
                } else {
                  rightHand = h0;
                  leftHand = h1;
                }
              } else {
                // Spatial sorting fallback: sort by wrist X position if labels conflict/overlap
                if (h0[0].x < h1[0].x) {
                  leftHand = h0;
                  rightHand = h1;
                } else {
                  leftHand = h1;
                  rightHand = h0;
                }
              }
            } else {
              const h0 = results.multiHandLandmarks[0];
              const label0 = results.multiHandedness?.[0]?.label;
              if (label0 === 'Left') {
                leftHand = h0;
              } else {
                rightHand = h0;
              }
            }

            // Extract 126-feature landmarks (21x3 Left + 21x3 Right)
            const rawFlat = [];
            for (let i = 0; i < 21; i++) {
              if (leftHand && leftHand[i]) {
                rawFlat.push(leftHand[i].x, leftHand[i].y, leftHand[i].z);
              } else {
                rawFlat.push(0, 0, 0);
              }
            }
            for (let i = 0; i < 21; i++) {
              if (rightHand && rightHand[i]) {
                rawFlat.push(rightHand[i].x, rightHand[i].y, rightHand[i].z);
              } else {
                rawFlat.push(0, 0, 0);
              }
            }

            // One-Euro adaptive filter smoothing to eliminate coordinate jitter
            const smoothedLandmarks = smootherRef.current.smooth(rawFlat);
            setLandmarkData(smoothedLandmarks);

            // Throttled callback to avoid flooding the API
            const now = Date.now();
            if (props.onLandmarks && (now - lastCallRef.current) >= props.throttleMs) {
              lastCallRef.current = now;
              
              // Determine handedness string
              let handednessStr = null;
              if (numHands === 1) {
                handednessStr = leftHand ? 'Left' : 'Right';
              } else if (numHands >= 2) {
                handednessStr = 'Both Hands';
              }

              props.onLandmarks(smoothedLandmarks, numHands, handednessStr);
            }
          } else {
            setIsDetecting(false);
            setLandmarkData(null);
            smootherRef.current.reset();
            if (props.onLandmarks) {
              props.onLandmarks(null, 0, null);
            }
          }
        });

        handsRef.current = hands;
        scriptLoadedRef.current = true;
        setIsLoaded(true);
        console.log('MediaPipe Hands initialized successfully.');
      } catch (err) {
        console.error('Failed to load MediaPipe Hands:', err);
      }
    }

    init();

    return () => {
      cancelled = true;
    };
  }, [enabled, loadScript, onLandmarks, throttleMs]);

  // ─── Start/Stop Camera Processing ─────────────────────────────────────
  useEffect(() => {
    if (!enabled || !isLoaded || !videoElement || !handsRef.current) return;

    let cancelled = false;

    // Process frames manually using requestAnimationFrame
    // We avoid window.Camera because it tries to call getUserMedia and overrides the useWebcam stream.
    let animId;
    let lastVideoTime = -1;

    const processFrame = async () => {
      if (cancelled) return;
      if (handsRef.current && videoElement.readyState >= 2) {
        // Only process if the video has a new frame
        if (videoElement.currentTime !== lastVideoTime) {
          lastVideoTime = videoElement.currentTime;
          try {
            await handsRef.current.send({ image: videoElement });
          } catch (e) {
            console.warn('MediaPipe send error:', e);
          }
        }
      }
      animId = requestAnimationFrame(processFrame);
    };
    processFrame();

    return () => {
      cancelled = true;
      if (animId) cancelAnimationFrame(animId);
    };
  }, [enabled, isLoaded, videoElement]);

  return {
    isLoaded,
    isDetecting,
    landmarkData,
    handCount,
  };
}
