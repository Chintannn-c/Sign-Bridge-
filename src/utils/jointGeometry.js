/**
 * Geometric Joint Utilities for Hand Landmark Analysis
 *
 * Provides scale-invariant and rotation-invariant joint metrics:
 * 1. 3D Finger Flexion Angles (DIP, PIP, MCP joint bend angles)
 * 2. Palm-Normalized Pinch Distances (Thumb-Index, Thumb-Middle, etc.)
 * 3. Dual-Hand Inter-Joint Contact Metrics (for ISL two-handed signs)
 */

/**
 * Compute the 3D angle in degrees between three landmark points A-B-C (joint at B)
 */
export function calculateAngle(a, b, c) {
  if (!a || !b || !c) return 0;
  const ba = [a.x - b.x, a.y - b.y, (a.z || 0) - (b.z || 0)];
  const bc = [c.x - b.x, c.y - b.y, (c.z || 0) - (b.z || 0)];

  const dot = ba[0] * bc[0] + ba[1] * bc[1] + ba[2] * bc[2];
  const magBA = Math.sqrt(ba[0] ** 2 + ba[1] ** 2 + ba[2] ** 2);
  const magBC = Math.sqrt(bc[0] ** 2 + bc[1] ** 2 + bc[2] ** 2);

  if (magBA * magBC === 0) return 0;
  const cosAngle = Math.max(-1.0, Math.min(1.0, dot / (magBA * magBC)));
  return (Math.acos(cosAngle) * 180) / Math.PI;
}

/**
 * Compute Euclidean distance between two 3D points
 */
export function calculateDistance(p1, p2) {
  if (!p1 || !p2) return 0;
  const dx = p1.x - p2.x;
  const dy = p1.y - p2.y;
  const dz = (p1.z || 0) - (p2.z || 0);
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

/**
 * Compute 3D flexion (bend) angles for all 5 fingers
 * Returns angles in degrees (180° = fully straight, <90° = curled)
 */
export function calculateFingerFlexions(landmarks) {
  if (!landmarks || landmarks.length < 21) {
    return { thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0 };
  }

  return {
    thumb: calculateAngle(landmarks[1], landmarks[2], landmarks[4]),
    index: calculateAngle(landmarks[5], landmarks[6], landmarks[8]),
    middle: calculateAngle(landmarks[9], landmarks[10], landmarks[12]),
    ring: calculateAngle(landmarks[13], landmarks[14], landmarks[16]),
    pinky: calculateAngle(landmarks[17], landmarks[18], landmarks[20]),
  };
}

/**
 * Compute palm-normalized pinch distances for key finger combinations
 */
export function calculatePinchDistances(landmarks) {
  if (!landmarks || landmarks.length < 21) {
    return { thumbIndex: 0, thumbMiddle: 0, palmScale: 1 };
  }

  // Palm size reference scale (Wrist to Middle MCP joint)
  const palmScale = Math.max(calculateDistance(landmarks[0], landmarks[9]), 0.001);

  return {
    thumbIndex: calculateDistance(landmarks[4], landmarks[8]) / palmScale,
    thumbMiddle: calculateDistance(landmarks[4], landmarks[12]) / palmScale,
    indexMiddle: calculateDistance(landmarks[8], landmarks[12]) / palmScale,
    palmScale,
  };
}

/**
 * Analyze dual-hand contact between Left Hand and Right Hand landmarks
 * Crucial for two-handed Indian Sign Language (ISL) signs
 */
export function calculateInterHandContact(leftHand, rightHand, threshold = 0.15) {
  if (!leftHand || !rightHand || leftHand.length < 21 || rightHand.length < 21) {
    return { isContacting: false, minDistance: 999, contactPairs: [] };
  }

  const palmScaleLeft = Math.max(calculateDistance(leftHand[0], leftHand[9]), 0.001);
  const keyJoints = [4, 8, 12, 16, 20, 0]; // Fingertips + Wrist
  let minDistance = 999;
  const contactPairs = [];

  for (const lIdx of keyJoints) {
    for (const rIdx of keyJoints) {
      const rawDist = calculateDistance(leftHand[lIdx], rightHand[rIdx]);
      const normalizedDist = rawDist / palmScaleLeft;
      if (normalizedDist < minDistance) {
        minDistance = normalizedDist;
      }
      if (normalizedDist <= threshold) {
        contactPairs.push({ leftJoint: lIdx, rightJoint: rIdx, distance: normalizedDist });
      }
    }
  }

  return {
    isContacting: contactPairs.length > 0,
    minDistance,
    contactPairs,
  };
}
