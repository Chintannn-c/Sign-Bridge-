/**
 * Splash Screen Animation Timing Configuration
 * Total duration: 3000ms active animation + 200ms exit fade = 3200ms
 * In prefers-reduced-motion: 1500ms presentation with 0.4s fade-in
 */
export const SPLASH_CONFIG = {
  ANIMATION_DURATION_MS: 3000,
  EXIT_FADE_MS: 200,
  TOTAL_DURATION_MS: 3200,
  REDUCED_MOTION_MS: 1500,
};

export const SPLASH_DURATION_MS = SPLASH_CONFIG.TOTAL_DURATION_MS;