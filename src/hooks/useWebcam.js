import { useState, useEffect, useRef, useCallback } from 'react';

/**
 * Custom React Hook for managing Webcam stream with device selection, mirroring, and ON/OFF toggle
 */
export function useWebcam() {
  const [isCameraOn, setIsCameraOn] = useState(false);
  const [isLive, setIsLive] = useState(false);
  const [isMirrored, setIsMirrored] = useState(false);
  const [devices, setDevices] = useState([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState('');
  const [error, setError] = useState(null);

  const videoRef = useRef(null);
  const streamRef = useRef(null);

  const fetchDevices = useCallback(async () => {
    try {
      if (navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) {
        const allDevices = await navigator.mediaDevices.enumerateDevices();
        const videoInputs = allDevices
          .filter(d => d.kind === 'videoinput')
          .map((d, index) => ({
            deviceId: d.deviceId,
            label: d.label || `Camera ${index + 1}`
          }));
        setDevices(videoInputs);
      }
    } catch (err) {
      console.warn('Unable to enumerate video devices:', err);
    }
  }, []);

  useEffect(() => {
    let isMounted = true;

    async function startMedia() {
      if (isCameraOn) {
        try {
          if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            // Stop any existing tracks before starting a new stream
            if (streamRef.current) {
              streamRef.current.getTracks().forEach(track => track.stop());
              streamRef.current = null;
            }

            const videoConstraints = selectedDeviceId
              ? { deviceId: { exact: selectedDeviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
              : { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' };

            const stream = await navigator.mediaDevices.getUserMedia({
              video: videoConstraints,
              audio: false
            });

            if (!isMounted) {
              stream.getTracks().forEach(track => track.stop());
              return;
            }

            streamRef.current = stream;
            setIsLive(true);
            setError(null);

            if (videoRef.current) {
              videoRef.current.srcObject = stream;
            }

            // Fetch device list with populated labels after permission is granted
            fetchDevices();
          }
        } catch (err) {
          if (isMounted) {
            setError(err);
            setIsLive(false);
          }
        }
      } else {
        // Stop camera tracks cleanly
        if (streamRef.current) {
          streamRef.current.getTracks().forEach(track => track.stop());
          streamRef.current = null;
        }
        if (videoRef.current) {
          videoRef.current.srcObject = null;
        }
        setIsLive(false);
      }
    }

    startMedia();

    return () => {
      isMounted = false;
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
        streamRef.current = null;
      }
      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
    };
  }, [isCameraOn, selectedDeviceId, fetchDevices]);

  // Handle camera plug/unplug or connection changes (e.g., connecting Iriun Webcam)
  useEffect(() => {
    if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) {
      const handleDeviceChange = () => fetchDevices();
      navigator.mediaDevices.addEventListener('devicechange', handleDeviceChange);
      return () => {
        navigator.mediaDevices.removeEventListener('devicechange', handleDeviceChange);
      };
    }
  }, [fetchDevices]);

  // Ref callback to bind video element when mounted
  const setVideoRef = useCallback((node) => {
    videoRef.current = node;
    if (node && streamRef.current && isCameraOn) {
      node.srcObject = streamRef.current;
    }
  }, [isCameraOn]);

  const selectCamera = useCallback((deviceId, e) => {
    if (e && e.stopPropagation) {
      e.stopPropagation();
    }
    setSelectedDeviceId(deviceId);
    setIsCameraOn(true);
  }, []);

  const toggleCamera = useCallback((e) => {
    if (e) {
      if (e.stopPropagation) e.stopPropagation();
      if (e.preventDefault) e.preventDefault();
    }
    setIsCameraOn(prev => !prev);
  }, []);

  const toggleMirror = useCallback((e) => {
    if (e) {
      if (e.stopPropagation) e.stopPropagation();
      if (e.preventDefault) e.preventDefault();
    }
    setIsMirrored(prev => !prev);
  }, []);

  const turnOnCamera = useCallback((e) => {
    if (e) {
      if (e.stopPropagation) e.stopPropagation();
      if (e.preventDefault) e.preventDefault();
    }
    setIsCameraOn(true);
  }, []);

  const turnOffCamera = useCallback((e) => {
    if (e) {
      if (e.stopPropagation) e.stopPropagation();
      if (e.preventDefault) e.preventDefault();
    }
    setIsCameraOn(false);
  }, []);

  return {
    videoRef: setVideoRef,
    isLive,
    isCameraOn,
    isMirrored,
    devices,
    selectedDeviceId,
    selectCamera,
    setIsCameraOn,
    toggleCamera,
    toggleMirror,
    turnOnCamera,
    turnOffCamera,
    error
  };
}
