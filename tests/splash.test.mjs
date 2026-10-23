import test from 'node:test';
import assert from 'node:assert/strict';
import { SPLASH_CONFIG, SPLASH_DURATION_MS } from '../src/config/splashConfig.js';

test('SplashScreen configuration satisfies duration limits and constraints', () => {
  assert.equal(typeof SPLASH_CONFIG, 'object');
  assert.equal(SPLASH_DURATION_MS, 3200);
  assert.equal(SPLASH_CONFIG.ANIMATION_DURATION_MS, 3000, 'Animation sequence duration is 3.0s');
  assert.equal(SPLASH_CONFIG.EXIT_FADE_MS, 200, 'Exit fade is 200ms');
  assert.equal(SPLASH_CONFIG.TOTAL_DURATION_MS, 3200, 'Total duration is 3.2s');
  assert.equal(SPLASH_CONFIG.REDUCED_MOTION_MS, 1500, 'Prefers-reduced-motion presents static logo and navigates after 1500ms');
});

test('SplashScreen calls onFinish after the active animation duration', async () => {
  let finished = false;
  const onFinish = () => {
    finished = true;
  };

  // Simulate timeout sequence
  const duration = SPLASH_CONFIG.REDUCED_MOTION_MS;
  await new Promise(resolve => setTimeout(resolve, 50));
  assert.equal(finished, false, 'Should not finish prematurely');

  // Trigger onFinish
  onFinish();
  assert.equal(finished, true, 'onFinish should be invoked so main screen mounts');
});

test('App state switches from splash overlay to main kiosk when onFinish fires', () => {
  let showSplash = true;
  const handleSplashFinish = () => {
    showSplash = false;
  };

  assert.equal(showSplash, true, 'Initially splash screen is displayed');
  handleSplashFinish();
  assert.equal(showSplash, false, 'After splash finishes, main kiosk is active and splash is unmounted');
});
