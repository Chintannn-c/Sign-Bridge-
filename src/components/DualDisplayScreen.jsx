import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Panel } from './Panel';
import { ISL_CONVERSATIONS, HUMAN_INPUT_POOL, ROBOT_INPUT_POOL } from '../data/islConversations';

export const DualDisplayScreen = () => {
  // Current text feeds
  const [humanText, setHumanText] = useState('Hello! Namaste.');
  const [robotText, setRobotText] = useState('Hello! Welcome to SignBridge.');

  // Streaming typed text state
  const [humanStreaming, setHumanStreaming] = useState('');
  const [robotStreaming, setRobotStreaming] = useState('');

  // Active panel status ('human', 'robot', or null)
  const [activePanel, setActivePanel] = useState('human');
  const [isStreamingHuman, setIsStreamingHuman] = useState(false);
  const [isStreamingRobot, setIsStreamingRobot] = useState(false);

  // Auto conversation stream demo state
  const [isPlayingAutoDemo, setIsPlayingAutoDemo] = useState(false);
  const scenarioIndexRef = useRef(0);
  const exchangeIndexRef = useRef(0);
  const streamingTimerRef = useRef(null);

  /**
   * Smooth typing stream engine
   */
  const streamTextToPanel = useCallback((panel, fullText, onComplete) => {
    let index = 0;
    if (streamingTimerRef.current) clearInterval(streamingTimerRef.current);

    if (panel === 'human') {
      setActivePanel('human');
      setIsStreamingHuman(true);
      setHumanText(fullText);
      setHumanStreaming('');

      streamingTimerRef.current = setInterval(() => {
        index += 1;
        setHumanStreaming(fullText.slice(0, index));
        if (index >= fullText.length) {
          clearInterval(streamingTimerRef.current);
          setIsStreamingHuman(false);
          if (onComplete) onComplete();
        }
      }, 45);
    } else {
      setActivePanel('robot');
      setIsStreamingRobot(true);
      setRobotText(fullText);
      setRobotStreaming('');

      streamingTimerRef.current = setInterval(() => {
        index += 1;
        setRobotStreaming(fullText.slice(0, index));
        if (index >= fullText.length) {
          clearInterval(streamingTimerRef.current);
          setIsStreamingRobot(false);
          if (onComplete) onComplete();
        }
      }, 55);
    }
  }, []);

  /**
   * Trigger next Human input
   */
  const triggerHumanInput = useCallback(() => {
    const randomText = HUMAN_INPUT_POOL[Math.floor(Math.random() * HUMAN_INPUT_POOL.length)];
    streamTextToPanel('human', randomText);
  }, [streamTextToPanel]);

  /**
   * Trigger next Robot input
   */
  const triggerRobotInput = useCallback(() => {
    const randomText = ROBOT_INPUT_POOL[Math.floor(Math.random() * ROBOT_INPUT_POOL.length)];
    streamTextToPanel('robot', randomText);
  }, [streamTextToPanel]);

  /**
   * Auto playback loop for ISL dual exchange demo
   */
  useEffect(() => {
    if (!isPlayingAutoDemo) return;

    const currentScenario = ISL_CONVERSATIONS[scenarioIndexRef.current];
    const exchanges = currentScenario.exchanges;
    const currentExchange = exchanges[exchangeIndexRef.current];

    if (!currentExchange) {
      scenarioIndexRef.current = (scenarioIndexRef.current + 1) % ISL_CONVERSATIONS.length;
      exchangeIndexRef.current = 0;
      return;
    }

    streamTextToPanel(currentExchange.speaker, currentExchange.text, () => {
      const timeout = setTimeout(() => {
        exchangeIndexRef.current += 1;
        if (exchangeIndexRef.current >= exchanges.length) {
          scenarioIndexRef.current = (scenarioIndexRef.current + 1) % ISL_CONVERSATIONS.length;
          exchangeIndexRef.current = 0;
        }
      }, 1500);

      return () => clearTimeout(timeout);
    });

    return () => {
      if (streamingTimerRef.current) clearInterval(streamingTimerRef.current);
    };
  }, [isPlayingAutoDemo, streamTextToPanel]);

  /**
   * Keyboard shortcuts for live testing
   */
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.code === 'Space') {
        e.preventDefault();
        setIsPlayingAutoDemo(prev => !prev);
      } else if (e.code === 'Digit1' || e.code === 'KeyH') {
        setIsPlayingAutoDemo(false);
        triggerHumanInput();
      } else if (e.code === 'Digit2' || e.code === 'KeyR') {
        setIsPlayingAutoDemo(false);
        triggerRobotInput();
      } else if (e.code === 'KeyC') {
        setIsPlayingAutoDemo(false);
        setHumanText('');
        setRobotText('');
        setHumanStreaming('');
        setRobotStreaming('');
        setActivePanel(null);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [triggerHumanInput, triggerRobotInput]);

  return (
    <div className="kiosk-container">
      {/* Floating rounded cards split by a soft vertical gap */}
      <main className="main-body-v3">
        <Panel
          label="YOU"
          fullText={humanText}
          streamingText={humanStreaming}
          isActive={activePanel === 'human'}
          isStreaming={isStreamingHuman}
          side="left"
          onClick={() => {
            setIsPlayingAutoDemo(false);
            triggerHumanInput();
          }}
        />

        <Panel
          label="SIGN-BRIDGE"
          fullText={robotText}
          streamingText={robotStreaming}
          isActive={activePanel === 'robot'}
          isStreaming={isStreamingRobot}
          side="right"
          onClick={() => {
            setIsPlayingAutoDemo(false);
            triggerRobotInput();
          }}
        />
      </main>
    </div>
  );
};
