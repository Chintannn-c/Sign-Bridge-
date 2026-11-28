/**
 * SignBridge — HandTrackingOverlay Pure Math & State Engine.
 * Isolates rendering math, trail physics, ripple kinematics, and presence checks
 * for zero-overhead unit testing and fast rAF execution.
 */

// MediaPipe Hands skeleton connections (21 joints per hand)
export const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],        // Thumb
  [0, 5], [5, 6], [6, 7], [7, 8],        // Index
  [0, 9], [9, 10], [10, 11], [11, 12],   // Middle
  [0, 13], [13, 14], [14, 15], [15, 16], // Ring
  [0, 17], [17, 18], [18, 19], [19, 20], // Pinky
  [5, 9], [9, 13], [13, 17],              // Palm
];

// Upper-Body Pose connections (33-point MediaPipe Pose topology)
export const POSE_CONNECTIONS = [
  [11, 12],           // Shoulder girdle
  [11, 23], [12, 24], // Torso lateral borders
  [23, 24],           // Pelvic line
  [11, 13], [13, 15], // Left arm (Shoulder -> Elbow -> Wrist)
  [12, 14], [14, 16], // Right arm (Shoulder -> Elbow -> Wrist)
  [9, 10],            // Mouth
  [1, 2], [2, 3],     // Left eye
  [4, 5], [5, 6],     // Right eye
];

export const UPPER_BODY_JOINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24];
export const FINGERTIPS = new Set([4, 8, 12, 16, 20]);
export const PALM_CENTER_INDICES = [0, 5, 9, 13, 17];

/**
 * Returns false only for all-zero sentinel or non-finite coordinates.
 */
export function isPresent(x, y) {
  if (typeof x !== 'number' || typeof y !== 'number') return false;
  if (!Number.isFinite(x) || !Number.isFinite(y)) return false;
  return !(x === 0 && y === 0);
}

/**
 * Computes HiDPI canvas backing store dimensions.
 */
export function calculateBackingStore(width, height, dpr = 1) {
  const safeDpr = Number.isFinite(dpr) && dpr > 0 ? dpr : 1;
  return {
    pixelWidth: Math.round(width * safeDpr),
    pixelHeight: Math.round(height * safeDpr),
    cssWidth: `${width}px`,
    cssHeight: `${height}px`,
    dpr: safeDpr,
  };
}

/**
 * Resolves active landmarks without falling back to stale React state
 * when rawLandmarksRef is provided.
 */
export function resolveActiveLandmarks(rawLandmarksRef, fallbackProps) {
  const hasRawRef = Boolean(rawLandmarksRef && rawLandmarksRef.current);
  if (hasRawRef) {
    const raw = rawLandmarksRef.current;
    return {
      landmarks: raw.landmarks ?? null,
      poseLandmarks: raw.poseLandmarks ?? null,
      bodyAnchors: raw.bodyAnchors ?? null,
      fromRawRef: true,
    };
  }
  return {
    landmarks: fallbackProps?.landmarks ?? null,
    poseLandmarks: fallbackProps?.poseLandmarks ?? null,
    bodyAnchors: fallbackProps?.bodyAnchors ?? null,
    fromRawRef: false,
  };
}

/**
 * Canonical hand color & label mapping matching backend Slot 0 (Left) and Slot 1 (Right).
 */
export function getHandColorInfo(handIndex) {
  if (handIndex === 0) {
    // Subject's Left Hand (Canonical Slot 0)
    return {
      handLabel: 'Left',
      baseColor: '52, 211, 153',
      color: 'rgba(52, 211, 153, 0.95)',
    };
  }
  // Subject's Right Hand (Canonical Slot 1)
  return {
    handLabel: 'Right',
    baseColor: '251, 191, 36',
    color: 'rgba(251, 191, 36, 0.95)',
  };
}

/**
 * Updates trail points for fingertips, clearing absent points and limiting by age (~90ms).
 */
export function updateFingertipTrails(handTrails, points, fingertips, now, maxAgeMs = 90) {
  for (const tip of fingertips) {
    const p = points[tip];
    if (!p || !isPresent(p.x, p.y)) {
      delete handTrails[tip];
      continue;
    }

    if (!handTrails[tip]) handTrails[tip] = [];
    const history = handTrails[tip];

    const last = history[history.length - 1];
    const dx = last ? p.x - last.x : 1;
    const dy = last ? p.y - last.y : 1;
    // Add point only if moved > 0.5px
    if (!last || (dx * dx + dy * dy) > 0.25) {
      history.push({ x: p.x, y: p.y, t: now });
    }

    const cutoff = now - maxAgeMs;
    while (history.length > 0 && history[0].t < cutoff) {
      history.shift();
    }
  }
}

/**
 * Computes ripple expansion state using elapsed time, ensuring identical
 * duration and radius progression across all monitor refresh rates (60Hz, 120Hz).
 */
export function computeRippleStep(rip, now) {
  const elapsed = now - rip.birth;
  const progress = elapsed / rip.duration;

  if (progress < 0 || progress >= 1) {
    return null;
  }

  const radius = rip.radius + progress * (rip.maxRadius - rip.radius);
  const alpha = Math.max(0, 1 - progress);
  const base = rip.baseColor || '16, 185, 129';

  return {
    x: rip.x,
    y: rip.y,
    radius,
    alpha,
    progress,
    mainStroke: `rgba(${base}, ${alpha * 0.9})`,
    echoStroke: `rgba(${base}, ${alpha * 0.45})`,
    hasEcho: radius > 12,
    echoRadius: radius * 0.7,
  };
}

function getHandBBox(landmarks, offset) {
  let xMin = Infinity, yMin = Infinity, xMax = -Infinity, yMax = -Infinity;
  let valid = 0;
  for (let i = 0; i < 21; i++) {
    const x = landmarks[offset + i * 3];
    const y = landmarks[offset + i * 3 + 1];
    if (isPresent(x, y)) {
      if (x < xMin) xMin = x;
      if (y < yMin) yMin = y;
      if (x > xMax) xMax = x;
      if (y > yMax) yMax = y;
      valid++;
    }
  }
  return valid > 0 ? { xMin, yMin, xMax, yMax } : null;
}

/**
 * Computes per-hand bounding boxes and overlap IoU from 126-float landmarks
 * to detect interlocking/occluding hands for confidence-gated disambiguation.
 */
export function computeHandOverlap(landmarks, iouThreshold = 0.15) {
  if (!landmarks || landmarks.length < 126) {
    return { hasLeft: false, hasRight: false, iou: 0, isOverlapping: false, box: null };
  }
  const leftBox = getHandBBox(landmarks, 0);
  const rightBox = getHandBBox(landmarks, 63);
  const hasLeft = Boolean(leftBox);
  const hasRight = Boolean(rightBox);

  let unionBox = leftBox || rightBox;
  let iou = 0;

  if (leftBox && rightBox) {
    const ixMin = Math.max(leftBox.xMin, rightBox.xMin);
    const iyMin = Math.max(leftBox.yMin, rightBox.yMin);
    const ixMax = Math.min(leftBox.xMax, rightBox.xMax);
    const iyMax = Math.min(leftBox.yMax, rightBox.yMax);
    const interArea = Math.max(0, ixMax - ixMin) * Math.max(0, iyMax - iyMin);
    const areaL = Math.max(0, leftBox.xMax - leftBox.xMin) * Math.max(0, leftBox.yMax - leftBox.yMin);
    const areaR = Math.max(0, rightBox.xMax - rightBox.xMin) * Math.max(0, rightBox.yMax - rightBox.yMin);
    const unionArea = areaL + areaR - interArea;
    iou = unionArea > 1e-6 ? interArea / unionArea : 0;
    unionBox = {
      xMin: Math.min(leftBox.xMin, rightBox.xMin),
      yMin: Math.min(leftBox.yMin, rightBox.yMin),
      xMax: Math.max(leftBox.xMax, rightBox.xMax),
      yMax: Math.max(leftBox.yMax, rightBox.yMax),
    };
  }

  return {
    hasLeft,
    hasRight,
    iou,
    isOverlapping: Boolean(hasLeft && hasRight && iou > iouThreshold),
    box: unionBox,
  };
}

