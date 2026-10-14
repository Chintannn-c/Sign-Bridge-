import React from 'react';
import './SplashScreen.css';

/**
 * Total splash screen lifetime in milliseconds.
 * Controls both React unmount timing and CSS animation synchronization.
 */
export const SPLASH_DURATION_MS = 1000;

/**
 * SignBridge Launch Splash Screen.
 *
 * Features:
 *   - Bold display wordmark with split "Sign" / "Bridge" slide-in & shimmer.
 *   - Two animated line-art SVG hands performing a conversational wave and meeting at the bridge arc.
 *   - Proportionately scaled bridge arc with pulsing contact nodes and ripple.
 *   - Fullscreen camel background (#C8AD93) with subtle ambient orbs.
 *   - Auto unmounts at SPLASH_DURATION_MS (1000ms) with zero delay to background app startup.
 *   - Honors prefers-reduced-motion.
 */
export const SplashScreen = React.memo(() => {
  return (
    <div
      className="splash-overlay"
      style={{ '--splash-duration': `${SPLASH_DURATION_MS}ms` }}
      aria-hidden="true"
    >
      {/* Ambient background glow orbs */}
      <div className="splash-ambient-orb orb-1" />
      <div className="splash-ambient-orb orb-2" />

      <div className="splash-content">
        <div className="splash-main-row">
          {/* Left Hand SVG (Waving & Connecting) */}
          <div className="splash-hand-wrap splash-hand-left">
            <svg
              viewBox="0 0 54 64"
              fill="rgba(255, 255, 255, 0.05)"
              stroke="currentColor"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
              className="splash-hand-svg"
            >
              <path d="M18 60 C18 52 14 46 12 40 C10 34 6 32 4 35 C2 38 4 43 8 48" />
              <path d="M16 38 L16 16 C16 12 21 12 21 16 L21 34" />
              <path d="M21 34 L21 10 C21 6 27 6 27 10 L27 34" />
              <path d="M27 34 L27 14 C27 10 33 10 33 14 L33 36" />
              <path d="M33 36 L33 22 C33 18 38 18 38 22 L38 40 C38 52 32 60 22 60 Z" />
            </svg>
          </div>

          {/* Centered Wordmark with "Sign" / "Bridge" split and Bridge Arc */}
          <div className="splash-wordmark-wrap">
            <h1 className="splash-title">
              <span className="splash-word-part splash-sign">Sign</span>
              <span className="splash-word-part splash-bridge">Bridge</span>
            </h1>

            {/* Connecting Bridge Arc */}
            <div className="splash-bridge-wrap">
              <svg
                viewBox="0 0 320 28"
                fill="none"
                className="splash-bridge-arc"
                preserveAspectRatio="none"
              >
                <path
                  d="M 12 24 Q 160 2, 308 24"
                  stroke="currentColor"
                  strokeWidth="2.5"
                  strokeLinecap="round"
                  className="splash-arc-path"
                />
                <circle cx="12" cy="24" r="3.5" fill="currentColor" className="splash-node-dot dot-left" />
                <circle cx="308" cy="24" r="3.5" fill="currentColor" className="splash-node-dot dot-right" />
                <circle cx="160" cy="13" r="1.5" stroke="currentColor" strokeWidth="1.5" className="splash-ripple-ring" />
              </svg>
            </div>
          </div>

          {/* Right Hand SVG (Mirrored Conversational Wave & Connecting) */}
          <div className="splash-hand-wrap splash-hand-right">
            <svg
              viewBox="0 0 54 64"
              fill="rgba(255, 255, 255, 0.05)"
              stroke="currentColor"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
              className="splash-hand-svg"
            >
              <path d="M18 60 C18 52 14 46 12 40 C10 34 6 32 4 35 C2 38 4 43 8 48" />
              <path d="M16 38 L16 16 C16 12 21 12 21 16 L21 34" />
              <path d="M21 34 L21 10 C21 6 27 6 27 10 L27 34" />
              <path d="M27 34 L27 14 C27 10 33 10 33 14 L33 36" />
              <path d="M33 36 L33 22 C33 18 38 18 38 22 L38 40 C38 52 32 60 22 60 Z" />
            </svg>
          </div>
        </div>
      </div>
    </div>
  );
});

SplashScreen.displayName = 'SplashScreen';
