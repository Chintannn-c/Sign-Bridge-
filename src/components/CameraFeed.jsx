import React, { useEffect, useRef } from 'react';
import { useWebcam } from '../hooks/useWebcam';
import { Video, VideoOff, Activity, CameraOff, Camera, FlipHorizontal } from 'lucide-react';

/**
 * CameraFeed Component - Live camera view for Human input panel
 * Positioned on the left side of the Human panel card.
 */
export const CameraFeed = ({ isActive }) => {
  const {
    videoRef,
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

  // Canvas visualizer animation when camera fallback is active
  useEffect(() => {
    if (!isCameraOn || isLive || !canvasRef.current || !containerRef.current) return;
    const canvas = canvasRef.current;
    const container = containerRef.current;
    const ctx = canvas.getContext('2d');
    let animId;
    let step = 0;

    const resizeCanvas = () => {
      if (container) {
        canvas.width = container.clientWidth || 300;
        canvas.height = container.clientHeight || 350;
      }
    };

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

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

      // Draw tracking landmarks grid
      ctx.strokeStyle = isActive ? 'rgba(13, 148, 136, 0.45)' : 'rgba(148, 163, 184, 0.25)';
      ctx.lineWidth = 1.5;

      for (let i = -3; i <= 3; i++) {
        ctx.beginPath();
        const offset = Math.sin(step + i) * 12;
        ctx.arc(centerX + i * 28 + offset, centerY + Math.cos(step + i) * 15, 6, 0, Math.PI * 2);
        ctx.fillStyle = isActive ? 'rgba(13, 148, 136, 0.85)' : 'rgba(148, 163, 184, 0.5)';
        ctx.fill();
        ctx.stroke();
      }

      animId = requestAnimationFrame(render);
    };

    render();
    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener('resize', resizeCanvas);
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
            title="Select Camera Source (e.g. Iriun Webcam, Integrated Camera)"
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
            title={isMirrored ? 'Video is Mirrored (Click for Normal)' : 'Video is Normal (Click to Mirror)'}
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
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className={`camera-video ${isMirrored ? 'is-mirrored' : ''}`}
          />
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
      <div className="camera-badge">
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
