import React, { useState, useRef, useCallback } from 'react';
import { CameraOff, Play, Square, Save, Trash2 } from 'lucide-react';
import { useHandDetection } from '../../../hooks/useHandDetection';
import { HandTrackingOverlay } from '../HandTrackingOverlay';

const ALPHABETS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ');
function list(str) {
  return str.split('');
}

export const DataCollectionTab = ({ isActive }) => {
  const videoRef = useRef(null);
  
  const [selectedLetter, setSelectedLetter] = useState('A');
  const [isRecording, setIsRecording] = useState(false);
  const [recordedFrames, setRecordedFrames] = useState([]);
  const [saveStatus, setSaveStatus] = useState('');

  // We only enable hand detection when this tab is active
  const detection = useHandDetection({
    videoElement: videoRef.current,
    enabled: isActive,
    onLandmarks: (landmarks) => handleLandmarks(landmarks),
    throttleMs: 33 // ~30 FPS for recording
  });

  const handleLandmarks = useCallback((landmarks) => {
    if (isRecording && landmarks && landmarks.length === 126) {
      setRecordedFrames(prev => [...prev, landmarks]);
    }
  }, [isRecording]);

  const toggleRecording = () => {
    if (isRecording) {
      setIsRecording(false);
    } else {
      setRecordedFrames([]); // Clear previous
      setSaveStatus('');
      setIsRecording(true);
    }
  };

  const clearRecording = () => {
    setRecordedFrames([]);
    setSaveStatus('');
  };

  const saveRecording = async () => {
    if (recordedFrames.length === 0) return;
    
    setSaveStatus('Saving...');
    try {
      const response = await fetch('http://localhost:5000/api/collect_data', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          letter: selectedLetter,
          session_id: `${Date.now()}`,
          frames: recordedFrames
        })
      });
      
      if (response.ok) {
        setSaveStatus(`Saved ${recordedFrames.length} frames for ${selectedLetter}!`);
        setRecordedFrames([]);
      } else {
        const err = await response.json();
        setSaveStatus(`Error: ${err.error || 'Failed to save'}`);
      }
    } catch (e) {
      setSaveStatus(`Network Error: ${e.message}`);
    }
  };

  return (
    <div className="sla-tab-pane" style={{ display: 'flex', flexDirection: 'column', gap: '1rem', height: '100%', overflowY: 'auto', padding: '1rem' }}>
      <div style={{ textAlign: 'center' }}>
        <h3 style={{ margin: '0 0 0.5rem 0', color: '#0d9488' }}>Dataset Collection</h3>
        <p style={{ fontSize: '0.85rem', color: '#666', margin: 0 }}>Record your hand gestures to train the neural network.</p>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', background: '#f8fafc', padding: '1rem', borderRadius: '12px', border: '1px solid #e2e8f0' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', flex: 1 }}>
          <label style={{ fontSize: '0.8rem', fontWeight: 'bold', color: '#475569' }}>Target Letter</label>
          <select 
            value={selectedLetter}
            onChange={(e) => setSelectedLetter(e.target.value)}
            disabled={isRecording}
            style={{ padding: '0.5rem', borderRadius: '6px', border: '1px solid #cbd5e1', fontSize: '1rem' }}
          >
            {ALPHABETS.map(l => <option key={l} value={l}>{l}</option>)}
          </select>
        </div>
        
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', width: '80px' }}>
          <div style={{ fontSize: '2rem', fontWeight: 'bold', color: '#0f172a' }}>{selectedLetter}</div>
        </div>
      </div>

      {/* Camera & Tracking Overlay */}
      <div className="sla-camera-container" style={{ minHeight: '300px', flexShrink: 0 }}>
        {isActive ? (
          <>
            <video
              ref={videoRef}
              className="sla-camera-video mirrored"
              autoPlay
              playsInline
              muted
            />
            {detection.isDetecting && detection.landmarkData && (
              <HandTrackingOverlay 
                landmarks={detection.landmarkData}
                width={videoRef.current?.videoWidth || 640}
                height={videoRef.current?.videoHeight || 480}
              />
            )}
            
            {!detection.isDetecting && (
              <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(0,0,0,0.5)', color: 'white' }}>
                Searching for hands...
              </div>
            )}
          </>
        ) : (
          <div className="sla-camera-video" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#1e293b' }}>
            <CameraOff size={32} color="#64748b" />
          </div>
        )}
        
        {/* Recording Indicator Overlay */}
        {isRecording && (
          <div style={{ position: 'absolute', top: '10px', left: '10px', background: 'rgba(239,68,68,0.9)', color: 'white', padding: '4px 12px', borderRadius: '20px', fontSize: '0.8rem', fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: '6px', animation: 'pulse 2s infinite' }}>
            <div style={{ width: '8px', height: '8px', background: 'white', borderRadius: '50%' }} />
            REC
          </div>
        )}
      </div>

      {/* Controls */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ fontSize: '0.9rem', color: '#475569' }}>
            Frames: <strong style={{ color: '#0d9488' }}>{recordedFrames.length}</strong>
          </div>
          
          <button 
            onClick={toggleRecording}
            style={{ 
              display: 'flex', alignItems: 'center', gap: '0.5rem', 
              background: isRecording ? '#ef4444' : '#0d9488', 
              color: 'white', border: 'none', padding: '0.6rem 1.2rem', 
              borderRadius: '8px', fontWeight: 'bold', cursor: 'pointer' 
            }}
          >
            {isRecording ? <><Square size={16} /> Stop</> : <><Play size={16} /> Record</>}
          </button>
        </div>

        {recordedFrames.length > 0 && !isRecording && (
          <div style={{ display: 'flex', gap: '0.5rem' }}>
            <button 
              onClick={clearRecording}
              style={{ flex: 1, display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '0.5rem', background: '#f1f5f9', color: '#64748b', border: '1px solid #cbd5e1', padding: '0.6rem', borderRadius: '8px', cursor: 'pointer' }}
            >
              <Trash2 size={16} /> Discard
            </button>
            <button 
              onClick={saveRecording}
              style={{ flex: 2, display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '0.5rem', background: '#10b981', color: 'white', border: 'none', padding: '0.6rem', borderRadius: '8px', fontWeight: 'bold', cursor: 'pointer' }}
            >
              <Save size={16} /> Save to Dataset
            </button>
          </div>
        )}
        
        {saveStatus && (
          <div style={{ padding: '0.8rem', background: saveStatus.includes('Error') ? '#fee2e2' : '#dcfce3', color: saveStatus.includes('Error') ? '#991b1b' : '#166534', borderRadius: '8px', fontSize: '0.85rem', textAlign: 'center' }}>
            {saveStatus}
          </div>
        )}
      </div>

    </div>
  );
};
