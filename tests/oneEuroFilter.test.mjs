import test from 'node:test';
import assert from 'node:assert/strict';
import { LandmarkSmoother } from '../src/utils/oneEuroFilter.js';

function makeHandFrame(x = 0.5, y = 0.5) {
  const frame = new Array(126).fill(0);
  for (let i = 0; i < 21; i++) {
    frame[i * 3] = x + i * 0.002;
    frame[i * 3 + 1] = y + i * 0.002;
    frame[i * 3 + 2] = 0;
  }
  return frame;
}

test('LandmarkSmoother smooths jitter and preserves filter state across brief 1-4 frame drops', () => {
  const smoother = new LandmarkSmoother(42, 1.2, 0.2, 4);
  const empty = new Array(126).fill(0);

  // Warm up at steady position (0.5, 0.5)
  smoother.smooth(makeHandFrame(0.5, 0.5), 1000);
  smoother.smooth(makeHandFrame(0.5, 0.5), 1033);

  // Brief 3-frame drop (< resetAfterFrames = 4) must NOT reset filter history
  for (let k = 0; k < 3; k++) {
    const outEmpty = smoother.smooth(empty, 1066 + k * 33);
    assert.equal(outEmpty[0], 0);
  }
  assert.equal(smoother.filters[0].xFilter.initialized, true);

  // Slight jitter on reappear should still be smoothed toward ~0.5
  const reappeared = smoother.smooth(makeHandFrame(0.52, 0.52), 1165);
  assert.ok(reappeared[0] > 0.5 && reappeared[0] < 0.52, `Expected smoothed x in (0.5, 0.52), got ${reappeared[0]}`);

  // Sustained drop (> 4 frames) must reset filters so next frame snaps cleanly
  for (let k = 0; k < 5; k++) {
    smoother.smooth(empty, 1200 + k * 33);
  }
  assert.equal(smoother.filters[0].xFilter.initialized, false);

  const snapped = smoother.smooth(makeHandFrame(0.2, 0.2), 1400);
  assert.equal(snapped[0], 0.2);
});
