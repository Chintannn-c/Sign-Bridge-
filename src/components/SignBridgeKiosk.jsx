import React, { useEffect, useState } from 'react';
import { useISLTranslation } from '../hooks/useISLTranslation';
import { HumanPanel } from './HumanPanel';
import { RobotPanel } from './RobotPanel';
import { AssistantBottomSheet } from './SignLanguageAssistant/AssistantBottomSheet';

export const SignBridgeKiosk = () => {
  const [isAssistantOpen, setIsAssistantOpen] = useState(false);

  const {
    humanText,
    robotText,
    humanStreamText,
    robotStreamText,
    activeSide,
    isStreamingHuman,
    isStreamingRobot,
    messages,
    sendMessage,
    setIsAutoDemo,
    triggerNextHuman,
    triggerNextRobot,
    clearAll
  } = useISLTranslation();

  // Keyboard shortcut listener (bypassed when typing in inputs)
  useEffect(() => {
    const handleKeyDown = (e) => {
      const target = e.target;
      if (
        target &&
        (target.tagName === 'INPUT' ||
          target.tagName === 'TEXTAREA' ||
          target.isContentEditable)
      ) {
        return;
      }

       if (e.code === 'Space') {
        e.preventDefault();
        setIsAutoDemo(prev => !prev);
      } else if (e.code === 'Digit1' || e.code === 'KeyH') {
        triggerNextHuman();
      } else if (e.code === 'Digit2' || e.code === 'KeyR') {
        triggerNextRobot();
      } else if (e.code === 'KeyC') {
        clearAll();
      } else if (e.code === 'KeyA') {
        setIsAssistantOpen(prev => !prev);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [setIsAutoDemo, triggerNextHuman, triggerNextRobot, clearAll]);

  return (
    <div className="kiosk-container">
      <main className="main-body-v3">
        <HumanPanel
          fullText={humanText}
          streamingText={humanStreamText}
          isActive={activeSide === 'human'}
          isStreaming={isStreamingHuman}
          onSendMessage={sendMessage}
        />

        <RobotPanel
          messages={messages}
          fullText={robotText}
          streamingText={robotStreamText}
          isActive={activeSide === 'robot'}
          isStreaming={isStreamingRobot}
          onSendMessage={sendMessage}
          onClear={clearAll}
          onOpenAssistant={() => setIsAssistantOpen(true)}
        />
      </main>

      <AssistantBottomSheet
        isOpen={isAssistantOpen}
        onClose={() => setIsAssistantOpen(false)}
        onSendToChat={sendMessage}
      />
    </div>
  );
};
