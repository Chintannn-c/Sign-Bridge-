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

export function useHandDetection({ videoElement, enabled = false, onLandmarks = null, throttleMs = 100 }) {
  const [isLoaded, setIsLoaded] = useState(false);
  const [isDetecting, setIsDetecting] = useState(false);
  const [landmarkData, setLandmarkData] = useState(null);
  const [poseData, setPoseData] = useState(null);
  const [bodyAnchors, setBodyAnchors] = useState(null);
  const [handCount, setHandCount] = useState(0);
  const [pipelineMode, setPipelineMode] = useState('initializing'); // 'holistic' | 'hands'

  const trackerRef = useRef(null);
  const lastCallRef = useRef(0);
  const scriptLoadedRef = useRef(false);
  const isProcessingRef = useRef(false);
  const smootherRef = useRef(new LandmarkSmoother(42, 1.2, 0.005));

  const latestProps = useRef({ enabled, onLandmarks, throttleMs });
  useEffect(() => {
    latestProps.current = { enabled, onLandmarks, throttleMs };
  }, [enabled, onLandmarks, throttleMs]);

  // ─── Load MediaPipe Scripts via CDN ───────────────────────────────────
  const loadScript = useCallback((src) => {
    return new Promise((resolve, reject) => {
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

  // ─── Initialize MediaPipe Holistic (with Hands fallback) ─────────────
  useEffect(() => {
    if (!enabled || scriptLoadedRef.current) return;

    let cancelled = false;

    async function init() {
      try {
        await loadScript(MEDIAPIPE_CAMERA);

        let isHolistic = false;
        try {
          await loadScript(MEDIAPIPE_HOLISTIC_CDN);
          if (typeof window.Holistic !== 'undefined') {
            isHolistic = true;
          }
        } catch (holisticErr) {
          console.warn('Holistic CDN load failed, falling back to Hands CDN:', holisticErr);
        }

        if (!isHolistic) {
          await loadScript(MEDIAPIPE_HANDS_CDN);
        }

        if (cancelled) return;

        if (isHolistic && typeof window.Holistic !== 'undefined') {
          // ─── 1. Holistic Pipeline (Pose + Dual Hands) ───
          const holistic = new window.Holistic({
            locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic@0.5.1675471629/${file}`,
          });

          holistic.setOptions({
            modelComplexity: 1,
            smoothLandmarks: true,
            enableSegmentation: false,
            smoothSegmentation: false,
            refineFaceLandmarks: false,
            minDetectionConfidence: 0.45,
            minTrackingConfidence: 0.45,
          });

          holistic.onResults((results) => {
            const props = latestProps.current;
            if (!props.enabled) {
              setIsDetecting(false);
              setLandmarkData(null);
              setPoseData(null);
              setBodyAnchors(null);
              return;
            }

            const leftHand = results.leftHandLandmarks || null;
            const rightHand = results.rightHandLandmarks || null;
            const pose = results.poseLandmarks || null;

            let nHands = 0;
            if (leftHand) nHands++;
            if (rightHand) nHands++;
            setHandCount(nHands);

            const hasDetection = Boolean(nHands > 0 || (pose && pose.length >= 25));
            setIsDetecting(hasDetection);

            // Extract standardized 126-float array [Left (63), Right (63)]
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

            // Extract Upper-Body Anchors (relative to head, shoulders, chest)
            let anchors = null;
            if (pose && pose.length >= 25) {
              const nose = pose[0];
              const mouth = {
                x: ((pose[9]?.x || 0) + (pose[10]?.x || 0)) / 2,
                y: ((pose[9]?.y || 0) + (pose[10]?.y || 0)) / 2,
                z: (((pose[9]?.z || 0) + (pose[10]?.z || 0)) / 2),
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
                nose,
                mouth,
                chest,
                leftShoulder,
                rightShoulder,
                leftElbow,
                rightElbow,
                leftWrist,
                rightWrist,
                leftHip,
                rightHip,
                shoulderDist,
                l_hand_to_mouth: getHandDist(leftHand, mouth),
                r_hand_to_mouth: getHandDist(rightHand, mouth),
                l_hand_to_chest: getHandDist(leftHand, chest),
                r_hand_to_chest: getHandDist(rightHand, chest),
                l_elevation: leftHand ? (leftShoulder.y - leftHand[0].y) / shoulderDist : 0,
                r_elevation: rightHand ? (rightShoulder.y - rightHand[0].y) / shoulderDist : 0,
              };
            }

            setPoseData(pose);
            setBodyAnchors(anchors);

            if (nHands > 0) {
              const smoothedLandmarks = smootherRef.current.smooth(rawFlat);
              setLandmarkData(smoothedLandmarks);

              const now = Date.now();
              if (props.onLandmarks && (now - lastCallRef.current) >= props.throttleMs) {
                lastCallRef.current = now;
                const handednessStr = nHands >= 2 ? 'Both Hands (Holistic)' : (leftHand ? 'Left' : 'Right');
                props.onLandmarks(smoothedLandmarks, nHands, handednessStr, {
                  poseLandmarks: pose,
                  bodyAnchors: anchors,
                });
              }
            } else {
              setLandmarkData(null);
              smootherRef.current.reset();
              if (props.onLandmarks && (!pose || pose.length === 0)) {
                props.onLandmarks(null, 0, null, null);
              }
            }
          });

          trackerRef.current = holistic;
          setPipelineMode('holistic');
          console.log('MediaPipe Holistic pipeline initialized successfully.');
        } else if (typeof window.Hands !== 'undefined') {
          // ─── 2. Fallback Hands-Only Pipeline ───
          const hands = new window.Hands({
            locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/${file}`,
          });

          hands.setOptions({
            maxNumHands: 2,
            modelComplexity: 1,
            minDetectionConfidence: 0.45,
            minTrackingConfidence: 0.45,
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
              let leftHand = null;
              let rightHand = null;

              if (numHands >= 2) {
                const h0 = results.multiHandLandmarks[0];
                const h1 = results.multiHandLandmarks[1];
                const label0 = results.multiHandedness?.[0]?.label;
                const label1 = results.multiHandedness?.[1]?.label;

                if (label0 && label1 && label0 !== label1) {
                  leftHand = label0 === 'Left' ? h0 : h1;
                  rightHand = label0 === 'Left' ? h1 : h0;
                } else {
                  leftHand = h0[0].x < h1[0].x ? h0 : h1;
                  rightHand = h0[0].x < h1[0].x ? h1 : h0;
                }
              } else {
                const h0 = results.multiHandLandmarks[0];
                const label0 = results.multiHandedness?.[0]?.label;
                if (label0 === 'Left') leftHand = h0;
                else rightHand = h0;
              }

              const rawFlat = [];
              for (let i = 0; i < 21; i++) {
                if (leftHand && leftHand[i]) rawFlat.push(leftHand[i].x, leftHand[i].y, leftHand[i].z);
                else rawFlat.push(0, 0, 0);
              }
              for (let i = 0; i < 21; i++) {
                if (rightHand && rightHand[i]) rawFlat.push(rightHand[i].x, rightHand[i].y, rightHand[i].z);
                else rawFlat.push(0, 0, 0);
              }

              const smoothedLandmarks = smootherRef.current.smooth(rawFlat);
              setLandmarkData(smoothedLandmarks);

              const now = Date.now();
              if (props.onLandmarks && (now - lastCallRef.current) >= props.throttleMs) {
                lastCallRef.current = now;
                const handednessStr = numHands >= 2 ? 'Both Hands' : (leftHand ? 'Left' : 'Right');
                props.onLandmarks(smoothedLandmarks, numHands, handednessStr, null);
              }
            } else {
              setIsDetecting(false);
              setLandmarkData(null);
              smootherRef.current.reset();
              if (props.onLandmarks) {
                props.onLandmarks(null, 0, null, null);
              }
            }
          });

          trackerRef.current = hands;
          setPipelineMode('hands');
          console.log('MediaPipe Hands fallback initialized.');
        }

        scriptLoadedRef.current = true;
        setIsLoaded(true);
      } catch (err) {
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
        scriptLoadedRef.current = false;
      }
    };
  }, [enabled, loadScript, onLandmarks, throttleMs]);

  // ─── Camera Frame Loop ────────────────────────────────────────────────
  useEffect(() => {
    if (!enabled || !isLoaded || !videoElement || !trackerRef.current) return;

    let cancelled = false;
    let animId;
    let lastVideoTime = -1;

    const processFrame = async () => {
      if (cancelled) return;
      if (trackerRef.current && videoElement.readyState >= 2 && !isProcessingRef.current) {
        if (videoElement.currentTime !== lastVideoTime) {
          lastVideoTime = videoElement.currentTime;
          isProcessingRef.current = true;
          try {
            await trackerRef.current.send({ image: videoElement });
          } catch (e) {
            console.warn('MediaPipe send error:', e);
          } finally {
            isProcessingRef.current = false;
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
    poseData,
    bodyAnchors,
    handCount,
    pipelineMode,
  };
}
