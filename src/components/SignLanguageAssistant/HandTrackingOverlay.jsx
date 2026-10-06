import React, { useRef, useEffect, useCallback } from 'react';

/**
 * HandTrackingOverlay — Canvas overlay for rendering MediaPipe Holistic landmarks
 * on top of the camera feed.
 *
 * Renders:
 *   1. Full upper-body skeleton (shoulders, torso, elbows, wrists, and head)
 *   2. Dual-hand skeletons with glowing fingertips and palm center
 *   3. Kinematic arm-to-wrist continuity lines
 */

// MediaPipe Hands skeleton connections (21 joints per hand)
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],       // Thumb
  [0, 5], [5, 6], [6, 7], [7, 8],       // Index
  [0, 9], [9, 10], [10, 11], [11, 12],  // Middle
  [0, 13], [13, 14], [14, 15], [15, 16],// Ring
  [0, 17], [17, 18], [18, 19], [19, 20],// Pinky
  [5, 9], [9, 13], [13, 17],          // Palm
];

// Upper-Body Pose connections (33-point MediaPipe Pose topology)
const POSE_CONNECTIONS = [
  [11, 12],          // Shoulder girdle
  [11, 23], [12, 24],// Torso lateral borders
  [23, 24],          // Pelvic line
  [11, 13], [13, 15],// Left arm (Shoulder -> Elbow -> Wrist)
  [12, 14], [14, 16],// Right arm (Shoulder -> Elbow -> Wrist)
  [9, 10],           // Mouth
  [1, 2], [2, 3],    // Left eye
  [4, 5], [5, 6],    // Right eye
];

const UPPER_BODY_JOINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24];
const FINGERTIPS = [4, 8, 12, 16, 20];
const PALM_CENTER_INDICES = [0, 5, 9, 13, 17];

export const HandTrackingOverlay = ({
  landmarks,
  poseLandmarks = null,
  bodyAnchors = null,
  width = 640,
  height = 480,
  isMirrored = true,
  activeSign = null,
}) => {
  const canvasRef = useRef(null);
  const ripplesRef = useRef([]);
  const lastPalmsRef = useRef([]);

  // Trigger electric shockwave ripple whenever a sign is committed/active
  useEffect(() => {
    if (activeSign && activeSign !== '?' && lastPalmsRef.current.length > 0) {
      for (const palm of lastPalmsRef.current) {
        ripplesRef.current.push({
          x: palm.x,
          y: palm.y,
          radius: 8,
          maxRadius: 60,
          color: palm.color || 'rgba(16, 185, 129, 0.95)',
          birth: performance.now(),
        });
      }
    }
  }, [activeSign]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    ctx.clearRect(0, 0, width, height);

    const now = performance.now();

    // ─────────────────────────────────────────────────────────────────────────
    // 1. RENDER UPPER-BODY SKELETON (MediaPipe Pose)
    // ─────────────────────────────────────────────────────────────────────────
    if (poseLandmarks && poseLandmarks.length >= 25) {
      const posePoints = poseLandmarks.map((p) => ({
        x: (p.x || 0) * width,
        y: (p.y || 0) * height,
        visibility: p.visibility !== undefined ? p.visibility : 1.0,
      }));

      // Pose lines style: subtle electric cyan with pulse
      const posePulseAlpha = Math.sin(now / 350) * 0.15 + 0.65;
      ctx.strokeStyle = `rgba(56, 189, 248, ${posePulseAlpha})`;
      ctx.lineWidth = 2.0;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';

      // Draw Pose bone connections
      for (const [a, b] of POSE_CONNECTIONS) {
        const pA = posePoints[a];
        const pB = posePoints[b];
        if (!pA || !pB) continue;
        if (pA.visibility < 0.4 || pB.visibility < 0.4) continue;
        if (pA.x === 0 && pA.y === 0) continue;
        if (pB.x === 0 && pB.y === 0) continue;

        ctx.beginPath();
        ctx.moveTo(pA.x, pA.y);
        ctx.lineTo(pB.x, pB.y);
        ctx.stroke();
      }

      // Draw Chest Centroid & Sternum Neck Link if available
      if (bodyAnchors && bodyAnchors.chest && bodyAnchors.nose) {
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

        // Chest anchor pulsing ring
        const chestPulse = Math.sin(now / 200) * 2 + 7;
        ctx.beginPath();
        ctx.arc(cX, cY, chestPulse, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(56, 189, 248, 0.85)';
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }

      // Draw Upper Body Keypoints (Shoulders, Elbows, Wrists, Hips, Head)
      for (const jIdx of UPPER_BODY_JOINTS) {
        const p = posePoints[jIdx];
        if (!p || p.visibility < 0.4 || (p.x === 0 && p.y === 0)) continue;

        // Outer glow
        ctx.beginPath();
        ctx.arc(p.x, p.y, 8, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(56, 189, 248, 0.22)';
        ctx.fill();

        // Core dot
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
    if (landmarks && landmarks.length >= 126) {
      // Dynamic pulsating metrics for living electric halos
      const fingertipPulse = Math.sin(now / 160) * 3 + 12; // 9px to 15px pulse

      for (let hand = 0; hand < 2; hand++) {
        const offset = hand * 21;
        const points = [];
        let hasData = false;

        for (let i = 0; i < 21; i++) {
          const idx = (offset + i) * 3;
          const x = landmarks[idx] * width;
          const y = landmarks[idx + 1] * height;
          const z = landmarks[idx + 2];
          points.push({ x, y, z });
          if (landmarks[idx] > 0 || landmarks[idx + 1] > 0) hasData = true;
        }

        if (!hasData) continue;

        const color = hand === 0
          ? 'rgba(52, 211, 153, 0.95)' // Left hand — vivid emerald green
          : 'rgba(251, 191, 36, 0.95)'; // Right hand — vivid golden amber

        const baseColor = hand === 0 ? '52, 211, 153' : '251, 191, 36';

        // Draw skeleton connections
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.5;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        for (const [a, b] of HAND_CONNECTIONS) {
          const pA = points[a];
          const pB = points[b];
          if (pA.x === 0 && pA.y === 0) continue;
          if (pB.x === 0 && pB.y === 0) continue;

          ctx.beginPath();
          ctx.moveTo(pA.x, pA.y);
          ctx.lineTo(pB.x, pB.y);
          ctx.stroke();
        }

        // Draw joint points with living halo animation
        for (let i = 0; i < 21; i++) {
          const p = points[i];
          if (p.x === 0 && p.y === 0) continue;

          const isFingertip = FINGERTIPS.includes(i);
          const radius = isFingertip ? 5.5 : 3.5;

          // Living glowing halo for fingertips
          if (isFingertip) {
            const grad = ctx.createRadialGradient(p.x, p.y, 2, p.x, p.y, fingertipPulse);
            grad.addColorStop(0, `rgba(${baseColor}, 0.85)`);
            grad.addColorStop(0.5, `rgba(${baseColor}, 0.35)`);
            grad.addColorStop(1, 'rgba(0, 0, 0, 0)');
            ctx.fillStyle = grad;
            ctx.beginPath();
            ctx.arc(p.x, p.y, fingertipPulse, 0, Math.PI * 2);
            ctx.fill();
          }

          // Joint core dot
          ctx.beginPath();
          ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
          ctx.fillStyle = isFingertip ? '#ffffff' : color;
          ctx.fill();
          ctx.strokeStyle = color;
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }

        // Draw animated rotating palm center target reticle
        const palmX = PALM_CENTER_INDICES.reduce((s, i) => s + points[i].x, 0) / PALM_CENTER_INDICES.length;
        const palmY = PALM_CENTER_INDICES.reduce((s, i) => s + points[i].y, 0) / PALM_CENTER_INDICES.length;

        if (palmX > 0 && palmY > 0) {
          currentPalms.push({ x: palmX, y: palmY, color });

          // Rotating sci-fi orbital brackets
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

          // Palm core dot
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
    // 3. RENDER EXPANDING SHOCKWAVE RIPPLES (Gesture Locks / Commits)
    // ─────────────────────────────────────────────────────────────────────────
    if (ripplesRef.current.length > 0) {
      const activeRipples = [];
      for (const rip of ripplesRef.current) {
        rip.radius += 2.4;
        const alpha = Math.max(0, 1 - (rip.radius / rip.maxRadius));

        if (alpha > 0.02 && rip.radius < rip.maxRadius) {
          ctx.save();
          ctx.beginPath();
          ctx.arc(rip.x, rip.y, rip.radius, 0, Math.PI * 2);
          ctx.strokeStyle = `rgba(16, 185, 129, ${alpha * 0.9})`;
          ctx.lineWidth = 2.5 * alpha;
          ctx.stroke();

          // Secondary subtle trailing echo ring
          if (rip.radius > 12) {
            ctx.beginPath();
            ctx.arc(rip.x, rip.y, rip.radius * 0.7, 0, Math.PI * 2);
            ctx.strokeStyle = `rgba(52, 211, 153, ${alpha * 0.45})`;
            ctx.lineWidth = 1.5 * alpha;
            ctx.stroke();
          }
          ctx.restore();
          activeRipples.push(rip);
        }
      }
      ripplesRef.current = activeRipples;
    }
  }, [landmarks, poseLandmarks, bodyAnchors, width, height]);

  // Silky 60fps continuous requestAnimationFrame loop for breathing micro-animations
  useEffect(() => {
    let animId;
    const render = () => {
      draw();
      animId = requestAnimationFrame(render);
    };
    animId = requestAnimationFrame(render);
    return () => cancelAnimationFrame(animId);
  }, [draw]);

  return (
    <canvas
      ref={canvasRef}
      className={`sla-camera-canvas ${isMirrored ? 'is-mirrored' : ''}`}
      style={{ transform: isMirrored ? 'scaleX(-1)' : 'none' }}
      width={width}
      height={height}
    />
  );
};
