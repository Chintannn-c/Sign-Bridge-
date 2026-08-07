import React, { useRef, useEffect, useCallback } from 'react';

/**
 * HandTrackingOverlay — Canvas overlay for rendering MediaPipe landmarks
 * on top of the camera feed. Draws skeleton connections, joints, fingertips,
 * and palm center with transparent overlay.
 */

// MediaPipe Hands skeleton connections
const CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],       // Thumb
  [0,5],[5,6],[6,7],[7,8],       // Index
  [0,9],[9,10],[10,11],[11,12],  // Middle
  [0,13],[13,14],[14,15],[15,16],// Ring
  [0,17],[17,18],[18,19],[19,20],// Pinky
  [5,9],[9,13],[13,17],          // Palm
];

const FINGERTIPS = [4, 8, 12, 16, 20];
const PALM_CENTER_INDICES = [0, 5, 9, 13, 17];

export const HandTrackingOverlay = ({ landmarks, width = 640, height = 480, isMirrored = true }) => {
  const canvasRef = useRef(null);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    canvas.width = width;
    canvas.height = height;
    ctx.clearRect(0, 0, width, height);

    if (!landmarks || landmarks.length < 126) return;

    // Draw both hands
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
        ? 'rgba(110, 127, 107, 0.9)'  // Left hand — sage green
        : 'rgba(200, 173, 147, 0.9)'; // Right hand — camel

      const glowColor = hand === 0
        ? 'rgba(110, 127, 107, 0.3)'
        : 'rgba(200, 173, 147, 0.3)';

      // Draw skeleton connections
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';

      for (const [a, b] of CONNECTIONS) {
        const pA = points[a];
        const pB = points[b];
        if (pA.x === 0 && pA.y === 0) continue;
        if (pB.x === 0 && pB.y === 0) continue;

        ctx.beginPath();
        ctx.moveTo(pA.x, pA.y);
        ctx.lineTo(pB.x, pB.y);
        ctx.stroke();
      }

      // Draw joint points
      for (let i = 0; i < 21; i++) {
        const p = points[i];
        if (p.x === 0 && p.y === 0) continue;

        const isFingertip = FINGERTIPS.includes(i);
        const radius = isFingertip ? 6 : 3.5;

        // Glow effect for fingertips
        if (isFingertip) {
          ctx.beginPath();
          ctx.arc(p.x, p.y, 12, 0, Math.PI * 2);
          ctx.fillStyle = glowColor;
          ctx.fill();
        }

        // Joint dot
        ctx.beginPath();
        ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
        ctx.fillStyle = isFingertip ? '#fff' : color;
        ctx.fill();
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }

      // Draw palm center
      const palmX = PALM_CENTER_INDICES.reduce((s, i) => s + points[i].x, 0) / PALM_CENTER_INDICES.length;
      const palmY = PALM_CENTER_INDICES.reduce((s, i) => s + points[i].y, 0) / PALM_CENTER_INDICES.length;

      if (palmX > 0 && palmY > 0) {
        // Palm center ring
        ctx.beginPath();
        ctx.arc(palmX, palmY, 8, 0, Math.PI * 2);
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.stroke();

        // Palm center dot
        ctx.beginPath();
        ctx.arc(palmX, palmY, 3, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
      }
    }
  }, [landmarks, width, height]);

  useEffect(() => {
    draw();
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
