import React, { useEffect, useRef, useState, useCallback } from 'react';
import { useWebcam } from '../hooks/useWebcam';
import { useHandDetection } from '../hooks/useHandDetection';
import { useGestureRecognition } from '../hooks/useGestureRecognition';
import { HandTrackingOverlay } from './SignLanguageAssistant/HandTrackingOverlay';
import { Video, VideoOff, Activity, CameraOff, Camera, FlipHorizontal, AlertCircle, Layers } from 'lucide-react';
import './SignLanguageAssistant/assistant.css';

/**
 * Pure React Camera View Component
 * Positioned on the left side of the Human panel card.
 * Supports dual-mode recognition: Static Letters (A-Z) & Whole-Word Sequence Gestures.
 */
export const CameraView = ({ isActive, onRecognitionUpdate }) => {
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

  const handleVideoRef = useCallback((node) => {
    webcamVideoRef(node);
    if (node !== videoElement) {
      setVideoElement(node);
    }
  }, [webcamVideoRef, videoElement]);

  // Recognition state machine with dual letter & word modes
  const recognition = useGestureRecognition({ enabled: isLive && isCameraOn });

  // MediaPipe hands integration (2-hand detection, 0.25 confidence)
  const detection = useHandDetection({
    videoElement: videoElement,
    enabled: isLive && isCameraOn && !!videoElement,
    onLandmarks: recognition.processLandmarks,
    throttleMs: 80 // 12.5 FPS smooth real-time response without HTTP queue backlog
  });

  // Notify parent of recognition updates
  useEffect(() => {
    if (onRecognitionUpdate) {
      onRecognitionUpdate({
        ...recognition,
        isCameraOn
      });
    }
  }, [
    recognition.sentenceBuffer, 
    recognition.status, 
    recognition.confidence, 
    recognition.guidance, 
    recognition.handInfo, 
    recognition.detectedLetter,
    recognition.detectedWord,
    recognition.recognitionMode,
    isCameraOn
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
    <div className="camera-feed-box" ref={containerRef}>
      {/* Top-Left Camera Source Selector Dropdown */}
      {devices && devices.length > 0 && (
        <div className="camera-select-pill" onClick={e => e.stopPropagation()}>
          <Camera size={13} className="text-teal-400" />
          <select
            value={selectedDeviceId}
            onChange={e => selectCamera(e.target.value, e)}
            className="camera-select-dropdown"
            title="Select Camera Source"
          >
            <option value="">Default Camera</option>
            {devices.map((device, idx) => (
              <option key={device.deviceId || idx} value={device.deviceId}>
                {device.label || `Camera ${idx + 1}`}
              </option>
            ))}
          </select>
        </div>
      )}

      {/* Top-Right Action Controls Group */}
      <div className="camera-top-actions" onClick={e => e.stopPropagation()}>
        {/* Mode Switcher: Letters (A-Z) vs Words (ISL) */}
        {isCameraOn && (
          <button
            className={`camera-toggle-btn ${isWordMode ? 'is-mirrored-active' : ''}`}
            onClick={() => recognition.setRecognitionMode(prev => (prev === 'letter' ? 'word' : 'letter'))}
            title={isWordMode ? 'Click to switch to Letters (A-Z) Mode' : 'Click to switch to ISL Words (HELLO, NAMASTE, etc.) Mode'}
            style={{ fontWeight: 700 }}
          >
            <Layers size={14} />
            <span>{isWordMode ? 'Words Mode' : 'Letters Mode'}</span>
          </button>
        )}

        {/* Mirror Video Flip Toggle Button */}
        {isCameraOn && (
          <button
            className={`camera-toggle-btn ${isMirrored ? 'is-mirrored-active' : ''}`}
            onClick={toggleMirror}
            title={isMirrored ? 'Video is Mirrored' : 'Video is Normal'}
          >
            <FlipHorizontal size={14} />
            <span>{isMirrored ? 'Mirrored' : 'Normal'}</span>
          </button>
        )}

        {/* Camera Quick Toggle Button */}
        <button
          className={`camera-toggle-btn ${!isCameraOn ? 'is-off' : ''}`}
          onClick={toggleCamera}
          title={isCameraOn ? 'Turn Off Camera' : 'Turn On Camera'}
        >
          {isCameraOn ? <Video size={14} /> : <VideoOff size={14} />}
          <span>{isCameraOn ? 'Camera ON' : 'Camera OFF'}</span>
        </button>
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
            {/* Real-time Hand Tracking Overlay */}
            {detection.isDetecting && detection.landmarkData && (
              <HandTrackingOverlay 
                landmarks={detection.landmarkData}
                width={videoElement?.videoWidth || 640}
                height={videoElement?.videoHeight || 480}
                isMirrored={isMirrored}
              />
            )}

            {/* Word Mode Sequence Buffer Progress Bar */}
            {isWordMode && (
              <div style={{
                position: 'absolute',
                top: '3.3rem',
                left: '0.875rem',
                zIndex: 32,
                background: 'rgba(28, 25, 23, 0.85)',
                color: '#fff',
                padding: '0.35rem 0.75rem',
                borderRadius: '8px',
                display: 'flex',
                alignItems: 'center',
                gap: '0.6rem',
                fontSize: '0.72rem',
                backdropFilter: 'blur(6px)',
                border: '1px solid rgba(200, 173, 147, 0.4)',
                boxShadow: '0 2px 8px rgba(0,0,0,0.3)'
              }}>
                <span style={{ fontWeight: 600, color: 'var(--accent-camel)' }}>Sequence:</span>
                <div style={{
                  width: '64px',
                  height: '6px',
                  background: 'rgba(255,255,255,0.2)',
                  borderRadius: '3px',
                  overflow: 'hidden'
                }}>
                  <div style={{
                    width: `${Math.min(100, (recognition.wordBufferCount / recognition.wordBufferMax) * 100)}%`,
                    height: '100%',
                    background: recognition.wordBufferCount >= recognition.wordBufferMax ? '#10b981' : 'var(--accent-sage)',
                    transition: 'width 0.1s ease'
                  }} />
                </div>
                <span style={{ fontFamily: 'monospace', opacity: 0.9 }}>
                  {recognition.wordBufferCount}/{recognition.wordBufferMax}
                </span>
              </div>
            )}
            
            {/* Live Guidance Feedback */}
            {recognition.guidance && (
              <div style={{ position: 'absolute', bottom: '3.2rem', left: '0.875rem', zIndex: 30, background: 'rgba(0,0,0,0.75)', color: '#fff', padding: '0.45rem 0.85rem', borderRadius: '9999px', display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.78rem', backdropFilter: 'blur(8px)', pointerEvents: 'none', maxWidth: 'calc(100% - 2rem)', boxSizing: 'border-box' }}>
                <AlertCircle size={14} className="text-amber-400" style={{ flexShrink: 0 }} />
                <span style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{recognition.guidance}</span>
              </div>
            )}
            
            {/* Detected Sign/Word Popup — or Low Confidence Indicator */}
            {activeSign ? (
              <div style={{
                position: 'absolute',
                bottom: '1rem',
                right: '1rem',
                zIndex: 35,
                background: 'rgba(13,148,136,0.92)',
                color: 'white',
                padding: activeSign.length > 2 ? '8px 14px' : '10px 16px',
                borderRadius: '12px',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                boxShadow: '0 4px 14px rgba(0,0,0,0.35)',
                backdropFilter: 'blur(8px)',
                pointerEvents: 'none',
                maxWidth: '180px'
              }}>
                <div style={{ 
                  fontSize: activeSign.length > 4 ? '1.15rem' : (activeSign.length > 2 ? '1.4rem' : '2.5rem'), 
                  fontWeight: 'bold', 
                  lineHeight: 1.1,
                  textAlign: 'center',
                  letterSpacing: activeSign.length > 2 ? '0.04em' : 'normal'
                }}>
                  {activeSign}
                </div>
                {recognition.confidence > 0 && (
                  <div style={{ fontSize: '0.7rem', opacity: 0.95, marginTop: '3px', fontFamily: 'monospace' }}>
                    {Math.round(recognition.confidence * 100)}% {isWordMode ? 'Word' : 'Letter'}
                  </div>
                )}
              </div>
            ) : (recognition.confidence > 0 && recognition.confidence < 0.55 && recognition.status === 'tracking') ? (
              <div style={{
                position: 'absolute',
                bottom: '1rem',
                right: '1rem',
                zIndex: 35,
                background: 'rgba(217, 119, 6, 0.85)',
                color: 'white',
                padding: '8px 14px',
                borderRadius: '12px',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                boxShadow: '0 4px 14px rgba(0,0,0,0.35)',
                backdropFilter: 'blur(8px)',
                pointerEvents: 'none',
                animation: 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
                maxWidth: '180px'
              }}>
                <AlertCircle size={20} style={{ marginBottom: '4px', opacity: 0.9 }} />
                <div style={{ fontSize: '0.78rem', fontWeight: 600, textAlign: 'center' }}>
                  Low Confidence
                </div>
                <div style={{ fontSize: '0.65rem', opacity: 0.85, marginTop: '2px', fontFamily: 'monospace' }}>
                  {Math.round(recognition.confidence * 100)}% — adjust hand
                </div>
              </div>
            ) : null}
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

      {/* Live / Status Badge */}
      <div className="camera-badge" style={{ zIndex: 40 }}>
        {isCameraOn ? (
          isLive ? (
            <>
              <Video size={14} className="text-teal-400" />
              <span>{isWordMode ? 'WORD DETECTION' : 'LETTER DETECTION'}</span>
            </>
          ) : (
            <>
              <Activity size={14} />
              <span>SIMULATED FEED</span>
            </>
          )
        ) : (
          <>
            <CameraOff size={14} />
            <span>OFFLINE</span>
          </>
        )}
      </div>
    </div>
  );
};