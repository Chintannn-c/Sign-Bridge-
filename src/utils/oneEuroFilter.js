/**
 * One-Euro Filter Implementation in JavaScript
 *
 * An adaptive low-pass filter for noisy human motion tracking.
 * - At low speed: High filtering (smooth, eliminates jitter/shaking)
 * - At high speed: Low filtering (zero latency/lag)
 *
 * Reference: Casiez, N., Roussel, N. and Vogel, D. (2012)
 * "1€ filter: a simple speed-based low-pass filter for noisy input in interactive systems"
 */

class LowPassFilter {
  constructor(alpha = 1.0, initValue = 0) {
    this.alpha = alpha;
    this.y = initValue;
    this.initialized = false;
  }

  filter(value, alpha) {
    if (alpha !== undefined) this.alpha = alpha;
    if (!this.initialized) {
      this.y = value;
      this.initialized = true;
      return value;
    }
    this.y = this.alpha * value + (1.0 - this.alpha) * this.y;
    return this.y;
  }

  reset() {
    this.initialized = false;
    this.y = 0;
  }
}

export class OneEuroFilter {
  constructor(minCutoff = 1.0, beta = 0.007, dCutoff = 1.0) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;

    this.xFilter = new LowPassFilter();
    this.dxFilter = new LowPassFilter();
    this.lastTime = null;
  }

  computeAlpha(rate, cutoff) {
    const tau = 1.0 / (2 * Math.PI * cutoff);
    const te = 1.0 / rate;
    return 1.0 / (1.0 + tau / te);
  }

  filter(value, timestamp = Date.now()) {
    if (this.lastTime === null) {
      this.lastTime = timestamp;
      return this.xFilter.filter(value);
    }

    const dt = Math.max((timestamp - this.lastTime) / 1000.0, 0.001);
    this.lastTime = timestamp;
    const rate = 1.0 / dt;

    // Estimate derivative (speed of movement)
    const dx = (value - this.xFilter.y) / dt;
    const edx = this.dxFilter.filter(dx, this.computeAlpha(rate, this.dCutoff));

    // Dynamic cutoff frequency based on speed
    const cutoff = this.minCutoff + this.beta * Math.abs(edx);
    return this.xFilter.filter(value, this.computeAlpha(rate, cutoff));
  }

  reset() {
    this.xFilter.reset();
    this.dxFilter.reset();
    this.lastTime = null;
  }
}

/**
 * Smoothing filter array for 42 hand landmarks (126 floats: 2 hands x 21 joints x 3 coordinates)
 */
export class LandmarkSmoother {
  constructor(numPoints = 42, minCutoff = 1.2, beta = 0.005) {
    this.filters = [];
    for (let i = 0; i < numPoints * 3; i++) {
      this.filters.push(new OneEuroFilter(minCutoff, beta));
    }
  }

  /**
   * Smooth flat landmark array of shape [x1, y1, z1, x2, y2, z2, ...]
   * Points at (0, 0, 0) immediately reset their filters so that when the hand
   * reappears or moves, it snaps to true position without dragging across the screen.
   */
  smooth(landmarks, timestamp = Date.now()) {
    if (!landmarks || landmarks.length === 0) return landmarks;

    // Detect slot swap / large sudden jump per hand (e.g. wrist jumping > 0.18 across the screen)
    // and reset that hand's filters so it snaps immediately without dragging ghost points across the frame.
    for (let h = 0; h < 2; h++) {
      const wIdx = h * 63;
      if (wIdx < landmarks.length && (landmarks[wIdx] !== 0 || landmarks[wIdx + 1] !== 0)) {
        const fX = this.filters[wIdx]?.xFilter;
        const fY = this.filters[wIdx + 1]?.xFilter;
        if (fX && fY && fX.initialized && fY.initialized) {
          const dx = landmarks[wIdx] - fX.y;
          const dy = landmarks[wIdx + 1] - fY.y;
          if (dx * dx + dy * dy > 0.035) {
            this.resetHand(h);
          }
        }
      }
    }

    const result = new Array(landmarks.length);
    for (let i = 0; i < landmarks.length; i += 3) {
      const isMissing = landmarks[i] === 0 && landmarks[i + 1] === 0 && landmarks[i + 2] === 0;
      if (isMissing) {
        result[i] = 0;
        result[i + 1] = 0;
        result[i + 2] = 0;
        if (i + 2 < this.filters.length) {
          this.filters[i].reset();
          this.filters[i + 1].reset();
          this.filters[i + 2].reset();
        }
        continue;
      }

      for (let k = 0; k < 3; k++) {
        const idx = i + k;
        if (idx < this.filters.length) {
          result[idx] = this.filters[idx].filter(landmarks[idx], timestamp);
        } else {
          result[idx] = landmarks[idx];
        }
      }
    }
    return result;
  }

  resetHand(handIdx = 0) {
    const start = handIdx * 63;
    const end = start + 63;
    for (let i = start; i < end && i < this.filters.length; i++) {
      this.filters[i].reset();
    }
  }

  reset() {
    for (const filter of this.filters) {
      filter.reset();
    }
  }
}
