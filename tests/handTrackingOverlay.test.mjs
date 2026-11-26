import test from 'node:test';
import assert from 'node:assert/strict';
import {
  isPresent,
  calculateBackingStore,
  resolveActiveLandmarks,
  getHandColorInfo,
  updateFingertipTrails,
  computeRippleStep,
} from '../src/components/SignLanguageAssistant/overlayEngine.js';

test('Presence checks reject only all-zero sentinel or non-finite coordinates', () => {
  // All zero sentinel -> absent
  assert.equal(isPresent(0, 0), false);

  // Non-finite coordinates -> absent
  assert.equal(isPresent(NaN, 0.5), false);
  assert.equal(isPresent(0.5, Infinity), false);
  assert.equal(isPresent(undefined, null), false);

  // Valid coordinates, including zero on one axis or slight negatives
  assert.equal(isPresent(0, 0.5), true);
  assert.equal(isPresent(-0.02, 0.4), true);
  assert.equal(isPresent(0.5, 0.5), true);
});

test('HiDPI canvas backing store scales accurately with devicePixelRatio', () => {
  const width = 640;
  const height = 480;

  // At devicePixelRatio = 2 (Retina display)
  const dpr2 = calculateBackingStore(width, height, 2);
  assert.equal(dpr2.pixelWidth, 1280);
  assert.equal(dpr2.pixelHeight, 960);
  assert.equal(dpr2.cssWidth, '640px');
  assert.equal(dpr2.cssHeight, '480px');
  assert.equal(dpr2.dpr, 2);

  // At devicePixelRatio = 1 (Standard display)
  const dpr1 = calculateBackingStore(width, height, 1);
  assert.equal(dpr1.pixelWidth, 640);
  assert.equal(dpr1.pixelHeight, 480);
});

test('Hands disappearing: rawLandmarksRef takes strict precedence without stale React state fallback', () => {
  const staleReactProps = {
    landmarks: new Array(126).fill(0.5),
    poseLandmarks: [{ x: 0.5, y: 0.5 }],
    bodyAnchors: { chest: { x: 0.5, y: 0.5 } },
  };

  // When rawLandmarksRef exists and reports null (hands left frame), it must return null as-is
  const rawRefEmpty = { current: { landmarks: null, poseLandmarks: null, bodyAnchors: null } };
  const resolved = resolveActiveLandmarks(rawRefEmpty, staleReactProps);

  assert.equal(resolved.landmarks, null, 'Must NOT fall back to stale React state landmarks');
  assert.equal(resolved.poseLandmarks, null, 'Must NOT fall back to stale React state pose');
  assert.equal(resolved.bodyAnchors, null, 'Must NOT fall back to stale React state anchors');
  assert.equal(resolved.fromRawRef, true);

  // Only when rawLandmarksRef is not provided does it use props
  const resolvedFallback = resolveActiveLandmarks(null, staleReactProps);
  assert.notEqual(resolvedFallback.landmarks, null);
  assert.equal(resolvedFallback.fromRawRef, false);
});

test('Trails: cleared when point or hand is absent, pruned by age (~90ms), and ignores micro-jitter (< 0.5px)', () => {
  const handTrails = {};
  const t0 = 1000;

  // 1. Initial point at tip 8
  const points = {
    8: { x: 100, y: 100 },
  };
  updateFingertipTrails(handTrails, points, [8], t0, 90);
  assert.equal(handTrails[8].length, 1);
  assert.equal(handTrails[8][0].x, 100);

  // 2. Micro-jitter (< 0.5px movement): should NOT append duplicate history point
  const jitterPoints = {
    8: { x: 100.2, y: 100.2 },
  };
  updateFingertipTrails(handTrails, jitterPoints, [8], t0 + 16, 90);
  assert.equal(handTrails[8].length, 1, 'Micro-jitter under 0.5px should not create new trail entries');

  // 3. Significant movement (> 0.5px) within window
  const movedPoints = {
    8: { x: 110, y: 110 },
  };
  updateFingertipTrails(handTrails, movedPoints, [8], t0 + 32, 90);
  assert.equal(handTrails[8].length, 2);

  // 4. Age expiration (> 90ms): older points must be pruned
  const laterPoints = {
    8: { x: 120, y: 120 },
  };
  updateFingertipTrails(handTrails, laterPoints, [8], t0 + 150, 90);
  // Point at t0 (1000) is now 150ms old and must be purged; points at t0+32 (1032) and t0+150 (1150) remain
  assert.ok(handTrails[8].every(pt => pt.t >= t0 + 60));

  // 5. Point becomes absent (0, 0): trail must be immediately cleared
  const absentPoints = {
    8: { x: 0, y: 0 },
  };
  updateFingertipTrails(handTrails, absentPoints, [8], t0 + 166, 90);
  assert.equal(handTrails[8], undefined, 'Absent point must clear trail history');
});

test('Ripples: color matches the hand (amber for right hand, emerald for left hand)', () => {
  const leftHand = getHandColorInfo(0);
  const rightHand = getHandColorInfo(1);

  assert.equal(leftHand.handLabel, 'Left');
  assert.equal(leftHand.baseColor, '52, 211, 153');
  assert.ok(leftHand.color.includes('52, 211, 153'));

  assert.equal(rightHand.handLabel, 'Right');
  assert.equal(rightHand.baseColor, '251, 191, 36');
  assert.ok(rightHand.color.includes('251, 191, 36'));

  // Ripple step for Right Hand (amber) must produce amber stroke
  const amberRipple = {
    x: 200,
    y: 200,
    radius: 8,
    maxRadius: 60,
    duration: 400,
    baseColor: rightHand.baseColor,
    birth: 1000,
  };

  const stepAmber = computeRippleStep(amberRipple, 1200); // 50% through duration
  assert.ok(stepAmber !== null);
  assert.ok(stepAmber.mainStroke.includes('251, 191, 36'), 'Amber hand must produce amber ripple');
  assert.ok(stepAmber.echoStroke.includes('251, 191, 36'), 'Echo ring must derive from amber baseColor');
});

test('Ripples: expansion duration and radii are identical at 60 fps and 120 fps', () => {
  const duration = 400;
  const birth = 1000;
  const initialRadius = 8;
  const maxRadius = 60;

  const rip = {
    x: 100,
    y: 100,
    radius: initialRadius,
    maxRadius,
    duration,
    baseColor: '52, 211, 153',
    birth,
  };

  // Simulate 60 fps timeline (dt = 16.666ms)
  const frames60 = [];
  for (let t = birth; t <= birth + duration; t += 1000 / 60) {
    frames60.push({ t, step: computeRippleStep(rip, t) });
  }

  // Simulate 120 fps timeline (dt = 8.333ms)
  const frames120 = [];
  for (let t = birth; t <= birth + duration; t += 1000 / 120) {
    frames120.push({ t, step: computeRippleStep(rip, t) });
  }

  // At identical wall-clock time t = birth + 200ms (50% progress), radius and alpha must match exactly
  const mid60 = computeRippleStep(rip, birth + 200);
  const mid120 = computeRippleStep(rip, birth + 200);

  assert.equal(mid60.radius, 34);
  assert.equal(mid120.radius, 34);
  assert.equal(mid60.alpha, 0.5);
  assert.equal(mid120.alpha, 0.5);

  // At t >= birth + duration, both must naturally expire (return null)
  assert.equal(computeRippleStep(rip, birth + duration + 1), null);
});
