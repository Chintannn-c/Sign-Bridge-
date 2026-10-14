import React from 'react';
import './SplashScreen.css';

/**
 * Total splash screen lifetime in milliseconds.
 * Controls both React unmount timing and CSS animation synchronization.
 */
export const SPLASH_DURATION_MS = 2500;

/**
 * Self-contained Splash Screen component for SignBridge.
 *
 * Plays once on app launch for ~1000ms:
 *   - 0–350ms: Fade-in, gentle scale up, upward drift, and drawing underline.
 *   - 350–650ms: Hold at full opacity with ambient glow.
 *   - 650–1000ms: Gentle scale-up and fade out to reveal the pre-initialized main app.
 *
 * Fullscreen, high-z-index, honors prefers-reduced-motion, and unmounts cleanly.
 */
export const SplashScreen = React.memo(() => {
  return (
    <div
      className="splash-overlay"
      style={{ '--splash-duration': `${SPLASH_DURATION_MS}ms` }}
      aria-hidden="true"
    >
      <div className="splash-content">
        <div className="splash-glow" />
        <h1 className="splash-title">SignBridge</h1>
        <div className="splash-underline" />
      </div>
    </div>
  );
});

SplashScreen.displayName = 'SplashScreen';
