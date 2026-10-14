import React, { useEffect, useRef, useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useWebcam } from '../hooks/useWebcam';
import { useHandDetection } from '../hooks/useHandDetection';
import { useGestureRecognition } from '../hooks/useGestureRecognition';
import { HandTrackingOverlay } from './SignLanguageAssistant/HandTrackingOverlay';
import { Video, VideoOff, CameraOff, Camera, FlipHorizontal, Layers, Maximize2, Minimize2, Send, Trash2 } from 'lucide-react';
import './SignLanguageAssistant/assistant.css';

/**
 * Pure React Camera View Component
 * Positioned on the left side of the Human panel card.
 * Supports dual-mode recognition: Static Letters (A-Z) & Whole-Word Sequence Gestures.
 * Fullscreen mode allows dedicating 100% of the screen exclusively to the camera feed and live tracking.
 */
export const CameraView = ({ isActive, onRecognitionUpdate, onSendMessage }) => {
  const {
    videoRef: webcamVideoRef,
    isLive,
    isCameraOn,
    isMirrored,
    devices,
    selectedDeviceId,
    selectCamera,
    toggleCamera,
    toggleMirror,
    turnOnCamera
  } = useWebcam();
  
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const [videoElement, setVideoElement] = useState(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [isCameraSelectOpen, setIsCameraSelectOpen] = useState(false);

  const toggleFullscreen = useCallback(() => {
    setIsFullscreen(prev => {
      const next = !prev;
      if (next) {
        if (containerRef.current && containerRef.current.requestFullscreen) {
          containerRef.current.requestFullscreen().catch(() => {});
        }
      } else {
        if (document.fullscreenElement && document.exitFullscreen) {
          document.exitFullscreen().catch(() => {});
        }
      }
      return next;
    });
  }, []);

  useEffect(() => {
    const handleFullscreenChange = () => {
      const isDocFull = Boolean(document.fullscreenElement && document.fullscreenElement === containerRef.current);
      if (!isDocFull && isFullscreen) {
        setIsFullscreen(false);
      }
    };
    const handleKeyDown = (e) => {
      if (e.key === 'Escape' && isFullscreen) {
        if (document.fullscreenElement && document.exitFullscreen) {
          document.exitFullscreen().catch(() => {});
        }
        setIsFullscreen(false);
      }
      if ((e.key === 'f' || e.key === 'F') && !['INPUT', 'TEXTAREA'].includes(e.target?.tagName) && isCameraOn) {
        toggleFullscreen();
      }
    };
    document.addEventListener('fullscreenchange', handleFullscreenChange);
    window.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('fullscreenchange', handleFullscreenChange);
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [isFullscreen, isCameraOn, toggleFullscreen]);

  const handleVideoRef = useCallback((node) => {
    webcamVideoRef(node);
    if (node !== videoElement) {
      setVideoElement(node);
    }
  }, [webcamVideoRef, videoElement]);

  // Recognition state machine with dual letter & word modes and automatic snapshot fallback
  const recognition = useGestureRecognition({ enabled: isLive && isCameraOn, videoElement, onSendMessage });

  // MediaPipe hands integration (dedicated 2-hand detection, 0.25 confidence, 0ms throttle)
  const detection = useHandDetection({
    videoElement: videoElement,
    enabled: isLive && isCameraOn && !!videoElement,
    onLandmarks: recognition.processLandmarks,
    throttleMs: 0
  });

  const recognitionRef = useRef(recognition);
  recognitionRef.current = recognition;

  // Notify parent of recognition updates
  useEffect(() => {
    if (onRecognitionUpdate) {
      onRecognitionUpdate({
        ...recognitionRef.current,
        isCameraOn
      });
    }
  }, [
    onRecognitionUpdate,
    isCameraOn,
    recognition.status,
    recognition.detectedSign,
    recognition.detectedLetter,
    recognition.detectedWord,
    recognition.confidence,
    recognition.sentenceBuffer,
    recognition.recognitionMode,
    recognition.guidance,
    recognition.inactivityCountdown,
    recognition.isAutoSending,
    recognition.handInfo,
    recognition.wordBufferCount
  ]);

  // Canvas visualizer animation when camera fallback is active
  useEffect(() => {
    if (!isCameraOn || isLive || !canvasRef.current || !containerRef.current) return;
    const canvas = canvasRef.current;
    const container = containerRef.current;
    const ctx = canvas.getContext('2d');
    let animId;
    let step = 0;

    const handleResize = () => {
      if (container) {
        canvas.width = container.clientWidth || 280;
        canvas.height = container.clientHeight || 320;
      }
    };

    handleResize();
    window.addEventListener('resize', handleResize);

    const render = () => {
      step += 0.05;
      const width = canvas.width;
      const height = canvas.height;
      const centerX = width / 2;
      const centerY = height / 2;

      ctx.clearRect(0, 0, width, height);

      // Pitch black background
      ctx.fillStyle = '#000000';
      ctx.fillRect(0, 0, width, height);

      // Tracking points mesh
      ctx.strokeStyle = isActive ? 'rgba(13, 148, 136, 0.5)' : 'rgba(148, 163, 184, 0.25)';
      ctx.lineWidth = 1.5;

      for (let i = -3; i <= 3; i++) {
        ctx.beginPath();
        const offset = Math.sin(step + i) * 12;
        ctx.arc(centerX + i * 26 + offset, centerY + Math.cos(step + i) * 14, 6, 0, Math.PI * 2);
        ctx.fillStyle = isActive ? 'rgba(13, 148, 136, 0.85)' : 'rgba(148, 163, 184, 0.5)';
        ctx.fill();
        ctx.stroke();
      }

      animId = requestAnimationFrame(render);
    };

    render();
    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener('resize', handleResize);
    };
  }, [isLive, isActive, isCameraOn]);

  const isWordMode = recognition.recognitionMode === 'word';
  const activeSign = (recognition.handInfo && recognition.confidence >= 0.60) ? (recognition.detectedWord || recognition.detectedLetter) : null;

  return (
    <div
      className={`camera-feed-box ${isFullscreen ? 'camera-fullscreen-mode' : ''}`}
      ref={containerRef}
      onClick={() => isCameraSelectOpen && setIsCameraSelectOpen(false)}
    >
      {/* Control Column pinned to top-right */}
      <div className="camera-control-column" onClick={e => e.stopPropagation()}>
        {/* 1. Mode toggle: Letters / Words */}
        <button
          type="button"
          className={`camera-icon-btn ${isWordMode ? 'is-active' : ''}`}
          onClick={() => recognition.setRecognitionMode(prev => (prev === 'letter' ? 'word' : 'letter'))}
          title={isWordMode ? 'Mode: Words (click for Letters)' : 'Mode: Letters (click for Words)'}
        >
          <Layers size={17} />
        </button>

        {/* 2. Mirror toggle */}
        <button
          type="button"
          className={`camera-icon-btn ${isMirrored ? 'is-active' : ''}`}
          onClick={toggleMirror}
          title={isMirrored ? 'Mirrored (click for normal)' : 'Normal (click to mirror)'}
        >
          <FlipHorizontal size={17} />
        </button>

        {/* 3. Fullscreen toggle */}
        <button
          type="button"
          className={`camera-icon-btn ${isFullscreen ? 'is-active' : ''}`}
          onClick={toggleFullscreen}
          title={isFullscreen ? 'Exit full screen (Esc or F)' : 'Full screen (F)'}
        >
          {isFullscreen ? <Minimize2 size={17} /> : <Maximize2 size={17} />}
        </button>

        {/* 4. Camera on/off toggle */}
        <button
          type="button"
          className={`camera-icon-btn ${!isCameraOn ? 'is-off' : ''}`}
          onClick={toggleCamera}
          title={isCameraOn ? 'Turn camera off' : 'Turn camera on'}
        >
          {isCameraOn ? <Video size={17} /> : <VideoOff size={17} />}
        </button>

        {/* 5. Camera source selector (only if more than one device) */}
        {devices && devices.length > 1 && (
          <div className="camera-device-select-wrap">
            <button
              type="button"
              className={`camera-icon-btn ${isCameraSelectOpen ? 'is-active' : ''}`}
              onClick={() => setIsCameraSelectOpen(prev => !prev)}
              title="Select camera source"
            >
              <Camera size={17} />
            </button>
            {isCameraSelectOpen && (
              <div className="camera-dropdown-menu">
                {devices.map((device, idx) => (
                  <button
                    key={device.deviceId || idx}
                    type="button"
                    className={`camera-dropdown-item ${selectedDeviceId === device.deviceId ? 'is-active' : ''}`}
                    onClick={(e) => {
                      selectCamera(device.deviceId, e);
                      setIsCameraSelectOpen(false);
                    }}
                  >
                    {device.label || `Camera ${idx + 1}`}
                  </button>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {isCameraOn ? (
        isLive ? (
          <>
            <video
              ref={handleVideoRef}
              autoPlay
              playsInline
              muted
              className={`camera-video ${isMirrored ? 'is-mirrored' : ''}`}
            />
            {/* Real-time Tracking Overlay */}
            {detection.isDetecting && (detection.landmarkData || detection.poseData) && (
              <HandTrackingOverlay 
                landmarks={detection.landmarkData}
                poseLandmarks={detection.poseData}
                bodyAnchors={detection.bodyAnchors}
                rawLandmarksRef={detection.rawLandmarksRef}
                width={videoElement?.videoWidth || 640}
                height={videoElement?.videoHeight || 480}
                isMirrored={isMirrored}
                activeSign={activeSign}
              />
            )}

            {/* Detected Sign Badge: restyled smaller and placed at bottom-center */}
            <AnimatePresence mode="wait">
              {activeSign && (
                <motion.div
                  key={`sign-${activeSign}`}
                  initial={{ scale: 0.8, opacity: 0, y: 10, x: '-50%' }}
                  animate={{ scale: 1, opacity: 1, y: 0, x: '-50%' }}
                  exit={{ scale: 0.8, opacity: 0, y: -6, x: '-50%' }}
                  transition={{ type: 'spring', stiffness: 500, damping: 25 }}
                  className="detected-sign-hud-badge"
                >
                  <div className="detected-sign-label">
                    {activeSign}
                  </div>
                  {recognition.confidence > 0 && (
                    <div className="detected-sign-conf">
                      {Math.round(recognition.confidence * 100)}% {isWordMode ? 'Word' : 'Letter'}
                    </div>
                  )}
                </motion.div>
              )}
            </AnimatePresence>
          </>
        ) : (
          <div className="camera-sim-fallback">
            <canvas ref={canvasRef} className="hand-sim-canvas" />
          </div>
        )
      ) : (
        <div className="camera-off-overlay">
          <div className="camera-off-icon-box">
            <CameraOff size={24} />
          </div>
          <div className="camera-off-title">Camera is Turned Off</div>
          <button className="turn-on-cam-btn" onClick={turnOnCamera}>
            <Video size={14} />
            <span>Turn On Camera</span>
          </button>
        </div>
      )}

      {/* Floating HUD only visible in Fullscreen Mode */}
      {isFullscreen && (
        <div className="fullscreen-live-hud" onClick={e => e.stopPropagation()}>
          <div className="fullscreen-hud-main">
            <div className="fullscreen-hud-tag">
              <span className="fullscreen-hud-dot" />
              <span>{isWordMode ? 'ISL WORDS' : 'A-Z LETTERS'}</span>
            </div>

            <div className="fullscreen-hud-buffer">
              {recognition.sentenceBuffer && recognition.sentenceBuffer.trim().length > 0 ? (
                <div className="fullscreen-hud-text" style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', flexWrap: 'wrap' }}>
                  <span>{recognition.sentenceBuffer}</span>
                  <span className="fullscreen-hud-cursor">|</span>
                  {recognition.inactivityCountdown && (
                    <span style={{
                      fontSize: '0.72rem',
                      color: '#a78bfa',
                      background: 'rgba(167, 139, 250, 0.15)',
                      padding: '2px 8px',
                      borderRadius: '6px',
                      border: '1px solid rgba(167, 139, 250, 0.35)',
                      fontWeight: 600,
                      animation: 'pulse 1.5s infinite'
                    }}>
                      Auto-send in {recognition.inactivityCountdown}s
                    </span>
                  )}
                </div>
              ) : (
                <span className="fullscreen-hud-placeholder">
                  Full screen camera active • Signs recognized in real time
                </span>
              )}
            </div>

            {recognition.sentenceBuffer && recognition.sentenceBuffer.trim().length > 0 && (
              <div className="fullscreen-hud-actions">
                <button
                  type="button"
                  className="fullscreen-action-btn btn-clear"
                  onClick={recognition.clearBuffer}
                  title="Clear recognized text"
                >
                  <Trash2 size={13} />
                  <span>Clear</span>
                </button>
                <button
                  type="button"
                  className="fullscreen-action-btn btn-send"
                  onClick={recognition.sendSentence}
                  disabled={recognition.isAutoSending}
                  title="Send to AI Assistant"
                >
                  <Send size={13} />
                  <span>{recognition.isAutoSending ? 'Sending...' : 'Send'}</span>
                </button>
              </div>
            )}
          </div>

          <button
            type="button"
            className="fullscreen-exit-pill"
            onClick={toggleFullscreen}
            title="Exit Full Screen (Esc or F)"
          >
            <Minimize2 size={13} />
            <span>Exit Full Screen (Esc)</span>
          </button>
        </div>
      )}
    </div>
  );
};