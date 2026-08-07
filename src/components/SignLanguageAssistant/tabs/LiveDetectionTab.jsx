import React, { useState } from 'react';
import {CameraOff, AlertCircle } from 'lucide-react';
import { useHandDetection } from '../../../hooks/useHandDetection';
import { useGestureRecognition } from '../../../hooks/useGestureRecognition';
import { HandTrackingOverlay } from '../HandTrackingOverlay';
import { SentenceBuilder } from '../SentenceBuilder';

export const LiveDetectionTab = ({ isActive, onSendToChat }) => {
  const [videoElement, setVideoElement] = useState(null);

  // Recognition state machine with temporal smoothing
  const recognition = useGestureRecognition({ enabled: isActive });

  // MediaPipe hands integration
  const detection = useHandDetection({
    videoElement: videoElement,
    enabled: isActive,
    onLandmarks: recognition.processLandmarks,
    throttleMs: 100 // Process at 10 FPS to save CPU
  });

  const handleSend = () => {
    const text = recognition.commitSentence();
    if (text && onSendToChat) {
      onSendToChat(text);
    }
  };

  return (
    <div className="sla-tab-pane">
      {/* Status Bar */}
      <div className="sla-status-bar">
        <div className={`sla-status-dot ${recognition.status}`} />
        <div className="sla-status-label">{recognition.statusLabel}</div>
        {recognition.confidence > 0 && (
          <div className="sla-status-confidence">
            Conf: {Math.round(recognition.confidence * 100)}%
          </div>
        )}
      </div>

      {/* Camera & Tracking Overlay */}
      <div className="sla-camera-container">
        {isActive ? (
          <video
            ref={setVideoElement}
            className="sla-camera-video mirrored"
            autoPlay
            playsInline
            muted
          />
        ) : (
          <div className="sla-camera-placeholder">
            <CameraOff size={24} />
            <span>Camera inactive</span>
          </div>
        )}

        {/* MediaPipe tracking visualization */}
        {isActive && detection.isDetecting && detection.landmarkData && (
          <HandTrackingOverlay 
            landmarks={detection.landmarkData}
            width={videoElement?.videoWidth || 640}
            height={videoElement?.videoHeight || 480}
          />
        )}

        {/* Overlays */}
        {isActive && (
          <>
            <div className="sla-camera-badge">
              <div className="sla-live-dot" /> LIVE
            </div>

            {/* Hand detected info */}
            {recognition.handInfo && (
              <div className="sla-hand-info">
                <div className="sla-hand-label">{recognition.handInfo.label}</div>
              </div>
            )}

            {/* Detected gesture popup */}
            {recognition.detectedLetter && (
              <div className="sla-detection-overlay">
                <div className="sla-detected-letter">{recognition.detectedLetter}</div>
                <div className="sla-detected-conf">{Math.round(recognition.confidence * 100)}%</div>
              </div>
            )}
          </>
        )}
      </div>

      {/* AI Guidance Feedback */}
      {recognition.guidance && isActive && (
        <div className="sla-guidance">
          <AlertCircle size={14} className="sla-guidance-icon" />
          <span>{recognition.guidance}</span>
        </div>
      )}

      {/* Sentence Builder */}
      <SentenceBuilder 
        sentence={recognition.sentenceBuffer}
        onUndo={recognition.undoLetter}
        onDelete={recognition.deleteLetter}
        onClear={recognition.clearBuffer}
        onAddSpace={recognition.addSpace}
        onSend={handleSend}
      />
    </div>
  );
};
