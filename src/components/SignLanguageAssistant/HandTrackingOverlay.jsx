import React, { useRef, useEffect, useCallback } from 'react';
import {
  HAND_CONNECTIONS,
  POSE_CONNECTIONS,
  UPPER_BODY_JOINTS,
  FINGERTIPS,
  PALM_CENTER_INDICES,
  isPresent,
  calculateBackingStore,
  resolveActiveLandmarks,
  getHandColorInfo,
  updateFingertipTrails,
  computeRippleStep,
} from './overlayEngine.js';

/**
 * HandTrackingOverlay — Canvas overlay for rendering MediaPipe Holistic landmarks
 * on top of the camera feed.
 *
 * Renders:
 *   1. Full upper-body skeleton (shoulders, torso, elbows, wrists, and head)
 *   2. Dual-hand skeletons with glowing fingertips and palm center
 *   3. Kinematic arm-to-wrist continuity lines and motion trails
 *
 * Hand index mapping:
 *   hand === 0 (landmarks 0..62): Subject's Left Hand (vivid emerald green)
 *   hand === 1 (landmarks 63..125): Subject's Right Hand (vivid golden amber)
 *   Matches backend canonical Slot 0 (Left) and Slot 1 (Right).
 *
 * Expected bodyAnchors shape for overlay:
 *   {
 *     chest: { x: number, y: number, z?: number },
 *     nose:  { x: number, y: number, z?: number }
 *   }
 * Also tolerates backend's scalar-key shape ({ l_hand_to_mouth, r_hand_to_chest, ... })
 * without errors by verifying chest/nose coordinates existence and finiteness.
 */

const HandTrackingOverlayComponent = ({
  landmarks,
  poseLandmarks = null,
  bodyAnchors = null,
  rawLandmarksRef = null,
  width = 640,
  height = 480,
  isMirrored = true,
  activeSign = null,
}) => {
  const canvasRef = useRef(null);
  const ripplesRef = useRef([]);
  const lastPalmsRef = useRef([]);
  const trailsRef = useRef([{}, {}]);
  const wasActiveRef = useRef(false);

  // Synchronize props directly during render without 1-frame effect lag
  const propsRef = useRef({
    landmarks,
    poseLandmarks,
    bodyAnchors,
    width,
    height,
    activeSign,
  });
  propsRef.current = {
    landmarks,
    poseLandmarks,
    bodyAnchors,
    width,
    height,
    activeSign,
  };

  // Trigger electric shockwave ripple whenever a sign is committed/active
  useEffect(() => {
    if (activeSign && activeSign !== '?' && lastPalmsRef.current.length > 0) {
      const now = performance.now();
      for (const palm of lastPalmsRef.current) {
        ripplesRef.current.push({
          x: palm.x,
          y: palm.y,
          radius: 8,
          maxRadius: 60,
          duration: 400,
          color: palm.color || 'rgba(16, 185, 129, 0.95)',
          baseColor: palm.baseColor || (palm.color?.includes('251') ? '251, 191, 36' : '16, 185, 129'),
          birth: now,
        });
      }
    }
  }, [activeSign]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    const { width, height } = propsRef.current;

    // Zero-lag read: use raw ref as-is when present without fallback to lagging React state
    const resolved = resolveActiveLandmarks(rawLandmarksRef, propsRef.current);
    const { landmarks, poseLandmarks, bodyAnchors } = resolved;

    const hasHands = Boolean(landmarks && landmarks.length >= 126);
    const hasPose = Boolean(poseLandmarks && poseLandmarks.length >= 25);
    const hasRipples = ripplesRef.current.length > 0;
    const isActive = hasHands || hasPose || hasRipples;

    // Reset last palms array every frame before drawing so no hands = empty palms
    lastPalmsRef.current = [];

    // HiDPI backing store scaling
    const dpr = typeof window !== 'undefined' ? (window.devicePixelRatio || 1) : 1;
    const { pixelWidth, pixelHeight } = calculateBackingStore(width, height, dpr);

    if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
      canvas.width = pixelWidth;
      canvas.height = pixelHeight;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // Skip drawing if idle and already cleared
    if (!isActive) {
      if (wasActiveRef.current) {
        ctx.clearRect(0, 0, width, height);
        wasActiveRef.current = false;
      }
      trailsRef.current = [{}, {}];
      return;
    }
    wasActiveRef.current = true;
    ctx.clearRect(0, 0, width, height);

    const now = performance.now();

    // ─────────────────────────────────────────────────────────────────────────
    // 1. RENDER UPPER-BODY SKELETON (MediaPipe Pose)
    // ─────────────────────────────────────────────────────────────────────────
    if (hasPose) {
      const getPosePoint = (idx) => {
        const p = poseLandmarks[idx];
        if (!p) return null;
        const vis = p.visibility !== undefined ? p.visibility : 1.0;
        if (vis < 0.4) return null;
        const x = (p.x || 0) * width;
        const y = (p.y || 0) * height;
        if (!isPresent(x, y)) return null;
        return { x, y };
      };

      const posePulseAlpha = Math.sin(now / 350) * 0.15 + 0.65;
      ctx.strokeStyle = `rgba(56, 189, 248, ${posePulseAlpha})`;
      ctx.lineWidth = 2.0;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';

      for (const [a, b] of POSE_CONNECTIONS) {
        const pA = getPosePoint(a);
        const pB = getPosePoint(b);
        if (!pA || !pB) continue;

        ctx.beginPath();
        ctx.moveTo(pA.x, pA.y);
        ctx.lineTo(pB.x, pB.y);
        ctx.stroke();
      }

      // Draw Chest Centroid & Sternum Link when finite numeric x/y exist
      if (
        bodyAnchors &&
        typeof bodyAnchors === 'object' &&
        bodyAnchors.chest &&
        bodyAnchors.nose &&
        isPresent(bodyAnchors.chest.x, bodyAnchors.chest.y) &&
        isPresent(bodyAnchors.nose.x, bodyAnchors.nose.y)
      ) {
        const cX = bodyAnchors.chest.x * width;
        const cY = bodyAnchors.chest.y * height;
        const nX = bodyAnchors.nose.x * width;
        const nY = bodyAnchors.nose.y * height;

        ctx.strokeStyle = 'rgba(147, 197, 253, 0.45)';
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(nX, nY);
        ctx.lineTo(cX, cY);
        ctx.stroke();
        ctx.setLineDash([]);

        const chestPulse = Math.sin(now / 200) * 2 + 7;
        ctx.beginPath();
        ctx.arc(cX, cY, chestPulse, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(56, 189, 248, 0.85)';
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }

      // Draw Upper Body Keypoints
      for (const jIdx of UPPER_BODY_JOINTS) {
        const p = getPosePoint(jIdx);
        if (!p) continue;

        ctx.beginPath();
        ctx.arc(p.x, p.y, 8, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(56, 189, 248, 0.22)';
        ctx.fill();

        ctx.beginPath();
        ctx.arc(p.x, p.y, 3.5, 0, Math.PI * 2);
        ctx.fillStyle = '#ffffff';
        ctx.fill();
        ctx.strokeStyle = 'rgba(56, 189, 248, 0.9)';
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 2. RENDER DUAL HANDS (MediaPipe Hands / Holistic Hands)
    // ─────────────────────────────────────────────────────────────────────────
    const currentPalms = [];
    if (hasHands) {
      const fingertipPulse = Math.sin(now / 160) * 3 + 12;

      for (let hand = 0; hand < 2; hand++) {
        const offset = hand * 21;
        const points = [];
        let hasData = false;

        for (let i = 0; i < 21; i++) {
          const idx = (offset + i) * 3;
          const rawX = landmarks[idx];
          const rawY = landmarks[idx + 1];
          const z = landmarks[idx + 2];
          const x = rawX * width;
          const y = rawY * height;
          points.push({ x, y, z });
          if (isPresent(rawX, rawY)) hasData = true;
        }

        if (!hasData) {
          // Clear trails for this absent hand
          trailsRef.current[hand] = {};
          continue;
        }

        const { color, baseColor } = getHandColorInfo(hand);

        ctx.strokeStyle = color;
        ctx.lineWidth = 2.5;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        for (const [a, b] of HAND_CONNECTIONS) {
          const pA = points[a];
          const pB = points[b];
          if (!isPresent(pA.x, pA.y) || !isPresent(pB.x, pB.y)) continue;

          ctx.beginPath();
          ctx.moveTo(pA.x, pA.y);
          ctx.lineTo(pB.x, pB.y);
          ctx.stroke();
        }

        // Motion trails limited by elapsed age (~90ms)
        const handTrails = trailsRef.current[hand] || {};
        trailsRef.current[hand] = handTrails;

        updateFingertipTrails(handTrails, points, FINGERTIPS, now, 90);

        for (const tip of FINGERTIPS) {
          const history = handTrails[tip];
          if (history && history.length > 1) {
            ctx.beginPath();
            ctx.moveTo(history[0].x, history[0].y);
            for (let h = 1; h < history.length; h++) {
              ctx.lineTo(history[h].x, history[h].y);
            }
            ctx.strokeStyle = `rgba(${baseColor}, 0.28)`;
            ctx.lineWidth = 2.0;
            ctx.lineCap = 'round';
            ctx.lineJoin = 'round';
            ctx.stroke();

            const tail = history[0];
            ctx.beginPath();
            ctx.arc(tail.x, tail.y, 2.2, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(${baseColor}, 0.18)`;
            ctx.fill();
          }
        }

        // Draw joint points with fingertip living halos
        for (let i = 0; i < 21; i++) {
          const p = points[i];
          if (!isPresent(p.x, p.y)) continue;

          const isFingertip = FINGERTIPS.has(i);
          const radius = isFingertip ? 5.5 : 3.5;

          if (isFingertip) {
            ctx.fillStyle = `rgba(${baseColor}, 0.25)`;
            ctx.beginPath();
            ctx.arc(p.x, p.y, fingertipPulse, 0, Math.PI * 2);
            ctx.fill();
            ctx.fillStyle = `rgba(${baseColor}, 0.45)`;
            ctx.beginPath();
            ctx.arc(p.x, p.y, fingertipPulse * 0.55, 0, Math.PI * 2);
            ctx.fill();
          }

          ctx.beginPath();
          ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
          ctx.fillStyle = isFingertip ? '#ffffff' : color;
          ctx.fill();
          ctx.strokeStyle = color;
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }

        // Palm center target reticle
        const validPalmPts = PALM_CENTER_INDICES.map(idx => points[idx]).filter(pt => isPresent(pt.x, pt.y));
        if (validPalmPts.length === PALM_CENTER_INDICES.length) {
          const palmX = validPalmPts.reduce((s, pt) => s + pt.x, 0) / validPalmPts.length;
          const palmY = validPalmPts.reduce((s, pt) => s + pt.y, 0) / validPalmPts.length;

          currentPalms.push({ x: palmX, y: palmY, color, baseColor });

          const angle = (now / 1100) % (Math.PI * 2);
          ctx.save();
          ctx.translate(palmX, palmY);
          ctx.rotate(angle);

          for (let a = 0; a < 4; a++) {
            ctx.rotate(Math.PI / 2);
            ctx.beginPath();
            ctx.arc(0, 0, 11, -0.3, 0.3);
            ctx.strokeStyle = color;
            ctx.lineWidth = 1.8;
            ctx.stroke();
          }
          ctx.restore();

          ctx.beginPath();
          ctx.arc(palmX, palmY, 3.5, 0, Math.PI * 2);
          ctx.fillStyle = '#ffffff';
          ctx.fill();
          ctx.strokeStyle = color;
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }
      }
      lastPalmsRef.current = currentPalms;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 3. RENDER EXPANDING SHOCKWAVE RIPPLES (Refresh-rate independent)
    // ─────────────────────────────────────────────────────────────────────────
    if (ripplesRef.current.length > 0) {
      const activeRipples = [];
      for (const rip of ripplesRef.current) {
        const step = computeRippleStep(rip, now);
        if (step) {
          ctx.save();
          ctx.beginPath();
          ctx.arc(step.x, step.y, step.radius, 0, Math.PI * 2);
          ctx.strokeStyle = step.mainStroke;
          ctx.lineWidth = 2.5 * step.alpha;
          ctx.stroke();

          if (step.hasEcho) {
            ctx.beginPath();
            ctx.arc(step.x, step.y, step.echoRadius, 0, Math.PI * 2);
            ctx.strokeStyle = step.echoStroke;
            ctx.lineWidth = 1.5 * step.alpha;
            ctx.stroke();
          }
          ctx.restore();
          activeRipples.push(rip);
        }
      }
      ripplesRef.current = activeRipples;
    }
  }, [rawLandmarksRef]);

  // RequestAnimationFrame loop — paused when document is hidden
  useEffect(() => {
    let animId;
    const render = () => {
      if (typeof document !== 'undefined' && document.hidden) {
        animId = requestAnimationFrame(render);
        return;
      }
      draw();
      animId = requestAnimationFrame(render);
    };
    animId = requestAnimationFrame(render);
    return () => cancelAnimationFrame(animId);
  }, [draw]);

  return (
    <canvas
      ref={canvasRef}
      className="sla-camera-canvas"
      style={{
        transform: isMirrored ? 'scaleX(-1)' : 'none',
        pointerEvents: 'none',
      }}
      aria-hidden="true"
    />
  );
};

export const HandTrackingOverlay = React.memo(
  HandTrackingOverlayComponent,
  (prevProps, nextProps) => {
    // If rawLandmarksRef is provided and active, landmark prop churn does not trigger re-render
    if (nextProps.rawLandmarksRef && nextProps.rawLandmarksRef.current) {
      return (
        prevProps.width === nextProps.width &&
        prevProps.height === nextProps.height &&
        prevProps.isMirrored === nextProps.isMirrored &&
        prevProps.activeSign === nextProps.activeSign &&
        prevProps.rawLandmarksRef === nextProps.rawLandmarksRef
      );
    }
    return (
      prevProps.landmarks === nextProps.landmarks &&
      prevProps.poseLandmarks === nextProps.poseLandmarks &&
      prevProps.bodyAnchors === nextProps.bodyAnchors &&
      prevProps.width === nextProps.width &&
      prevProps.height === nextProps.height &&
      prevProps.isMirrored === nextProps.isMirrored &&
      prevProps.activeSign === nextProps.activeSign
    );
  }
);

HandTrackingOverlay.displayName = 'HandTrackingOverlay';
