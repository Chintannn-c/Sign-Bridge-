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
   */
  smooth(landmarks, timestamp = Date.now()) {
    if (!landmarks || landmarks.length === 0) return landmarks;
    const result = new Array(landmarks.length);
    for (let i = 0; i < landmarks.length; i++) {
      // If point is missing (0,0,0), pass through without filtering
      if (landmarks[i] === 0 && (i % 3 === 0)) {
        result[i] = 0;
        if (i + 1 < landmarks.length) result[i + 1] = 0;
        if (i + 2 < landmarks.length) result[i + 2] = 0;
        i += 2;
        continue;
      }

      if (i < this.filters.length) {
        result[i] = this.filters[i].filter(landmarks[i], timestamp);
      } else {
        result[i] = landmarks[i];
      }
    }
    return result;
  }

  reset() {
    for (const filter of this.filters) {
      filter.reset();
    }
  }
}
