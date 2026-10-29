import React, { useState, useCallback, useEffect } from 'react';
import { SignBridgeKiosk } from './components/SignBridgeKiosk';
import { SplashScreen } from './components/SplashScreen';
import { ErrorBoundary } from './components/ErrorBoundary';

function App() {
  const isPreviewMode = typeof window !== 'undefined' && 
    (window.location.search.includes('preview=splash') || window.location.search.includes('splash=true'));

  const [showSplash, setShowSplash] = useState(true);

  // Expose global preview trigger for easy isolation testing in dev mode only
  useEffect(() => {
    if (import.meta.env.DEV) {
      window.previewSplash = () => setShowSplash(true);
    }
  }, []);

  const handleSplashFinish = useCallback(() => {
    if (!isPreviewMode) {
      setShowSplash(false);
    }
  }, [isPreviewMode]);

  return (
    <ErrorBoundary>
      {/* Main app mounts and begins camera/model loading immediately in background */}
      <SignBridgeKiosk />
      {showSplash && <SplashScreen onFinish={handleSplashFinish} />}
    </ErrorBoundary>
  );
}

export default App;