import React, { useState, useEffect } from 'react';
import { SignBridgeKiosk } from './components/SignBridgeKiosk';
import { SplashScreen, SPLASH_DURATION_MS } from './components/SplashScreen';

function App() {
  const [showSplash, setShowSplash] = useState(true);

  useEffect(() => {
    const timer = setTimeout(() => {
      setShowSplash(false);
    }, SPLASH_DURATION_MS);

    return () => clearTimeout(timer);
  }, []);

  return (
    <>
      {/* Main app mounts and begins camera/model loading immediately in background */}
      <SignBridgeKiosk />
      {showSplash && <SplashScreen />}
    </>
  );
}

export default App;