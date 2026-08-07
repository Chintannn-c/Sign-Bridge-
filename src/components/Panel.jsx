import React from 'react';
import { CameraFeed } from './CameraFeed';

/**
 * Clove Dental Light Theme Panel Component for Sign-Bridge
 * Supports Camera feed on the LEFT side of the Human panel content.
 */
export const Panel = ({
  label,
  fullText,
  streamingText,
  isActive,
  isStreaming,
  side,
  onClick
}) => {
  const isHuman = side === 'left';
  const isRobot = side === 'right';
  const displayText = streamingText || fullText || '';

  // Extract individual words and current forming word for Robot feed
  const words = fullText ? fullText.split(' ') : [];
  const currentWordIndex = Math.min(
    Math.floor((streamingText.length / (fullText.length || 1)) * words.length),
    words.length - 1
  );
  const currentWord = words[currentWordIndex] || words[0] || '';

  return (
    <div
      className={`panel-card ${side}-card ${isActive ? 'active-panel' : ''}`}
      onClick={onClick}
    >
      <div className="click-layer" title="Click to trigger live input simulation" />

      {/* Card Header Strip */}
      <div className="card-header">
        <div className={`card-label ${isRobot ? 'robot-label' : ''}`}>
          <span>{label}</span>
          {isHuman && isActive && <span className="listening-dot" />}
        </div>
      </div>

      {/* Panel Body Content */}
      {isHuman ? (
        <div className="left-card-content">
          {/* Camera Feed on the LEFT side */}
          <CameraFeed isActive={isActive} />

          {/* Text Feed on the RIGHT side of the Human panel */}
          <div className="card-text-wrapper">
            <h2 className="card-text">
              {displayText}
              {isStreaming && <span className="streaming-cursor-teal" />}
            </h2>
          </div>
        </div>
      ) : (
        /* Robot Card Content */
        <div className="card-text-wrapper">
          <h2 className="card-text">
            {displayText}
            {isStreaming && <span className="streaming-cursor-teal" />}
          </h2>

          {/* Soft Letter-Tile Row for Robot Fingerspelling Feed */}
          {currentWord && (
            <div className="letter-tiles-row">
              {currentWord.split('').map((char, idx) => {
                const isCurrentTile =
                  isStreaming && idx === streamingText.length % (currentWord.length || 1);
                return (
                  <div
                    key={`${char}-${idx}`}
                    className={`letter-tile ${isCurrentTile ? 'active-letter' : ''}`}
                  >
                    {char.toUpperCase()}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}
    </div>
  );
};
