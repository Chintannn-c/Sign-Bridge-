import React, { useEffect, useRef, useState, useCallback } from 'react';
import { useWebcam } from '../hooks/useWebcam';
import { useHandDetection } from '../hooks/useHandDetection';
import { useGestureRecognition } from '../hooks/useGestureRecognition';
import { HandTrackingOverlay } from './SignLanguageAssistant/HandTrackingOverlay';
import { Video, VideoOff, Activity, CameraOff, Camera, FlipHorizontal, AlertCircle } from 'lucide-react';
import './SignLanguageAssistant/assistant.css';

/**
 * Pure React Camera View Component
 * Positioned on the left side of the Human panel card.
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

  // Recognition state machine with temporal smoothing
  const recognition = useGestureRecognition({ enabled: isLive && isCameraOn });

  // MediaPipe hands integration
  const detection = useHandDetection({
    videoElement: videoElement,
    enabled: isLive && isCameraOn && !!videoElement,
    onLandmarks: recognition.processLandmarks,
    throttleMs: 16 // 60 FPS ultra-fast tracking & API response
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

      // Pure pitch black background
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
      <div className="camera-top-actions">
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
            
            {/* Live Guidance Feedback (stacked cleanly above LIVE FEED badge) */}
            {recognition.guidance && (
              <div style={{ position: 'absolute', bottom: '3.2rem', left: '0.875rem', zIndex: 30, background: 'rgba(0,0,0,0.75)', color: '#fff', padding: '0.45rem 0.85rem', borderRadius: '9999px', display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.78rem', backdropFilter: 'blur(8px)', pointerEvents: 'none', maxWidth: 'calc(100% - 2rem)', boxSizing: 'border-box' }}>
                <AlertCircle size={14} className="text-amber-400" style={{ flexShrink: 0 }} />
                <span style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{recognition.guidance}</span>
              </div>
            )}
            
            {/* Detected Gesture Popup */}
            {recognition.detectedLetter && (
              <div style={{
                position: 'absolute',
                bottom: '1rem',
                right: '1rem',
                zIndex: 35,
                background: 'rgba(13,148,136,0.9)',
                color: 'white',
                padding: '10px 16px',
                borderRadius: '12px',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
                backdropFilter: 'blur(8px)',
                pointerEvents: 'none'
              }}>
                <div style={{ fontSize: '2.5rem', fontWeight: 'bold', lineHeight: 1 }}>
                  {recognition.detectedLetter}
                </div>
                {recognition.confidence > 0 && (
                  <div style={{ fontSize: '0.7rem', opacity: 0.9, marginTop: '2px', fontFamily: 'monospace' }}>
                    {Math.round(recognition.confidence * 100)}%
                  </div>
                )}
              </div>
            )}
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
              <span>LIVE FEED</span>
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