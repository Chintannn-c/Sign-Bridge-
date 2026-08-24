import { useState, useEffect, useRef, useCallback } from 'react';
import { ISL_CONVERSATIONS, HUMAN_INPUT_POOL, ROBOT_INPUT_POOL } from '../data/islConversations';

// ─── Flask API Configuration ────────────────────────────────────────────────
const API_BASE = '/api';  // Proxied by Vite to http://localhost:5000

/**
 * Custom React Hook for managing live ISL translation feeds and Chat UI history.
 *
 * Supports two modes:
 *   1. LIVE mode  — sends MediaPipe landmarks to Flask API for real ML translation
 *   2. DEMO mode  — uses simulated typewriter text pools for UI demonstration
 */
export function useISLTranslation() {
  const [humanText, setHumanText] = useState('');
  const [robotText, setRobotText] = useState('');

  const [humanStreamText, setHumanStreamText] = useState('');
  const [robotStreamText, setRobotStreamText] = useState('');

  const [activeSide, setActiveSide] = useState(null); // 'human' | 'robot' | null
  const [isStreamingHuman, setIsStreamingHuman] = useState(false);
  const [isStreamingRobot, setIsStreamingRobot] = useState(false);

  const [messages, setMessages] = useState([]);

  const [isAutoDemo, setIsAutoDemo] = useState(false);

  // ─── Live Translation State ─────────────────────────────────────────────
  const [isLiveMode, setIsLiveMode] = useState(false);
  const [lastPrediction, setLastPrediction] = useState(null);
  const [apiStatus, setApiStatus] = useState('disconnected'); // 'connected' | 'disconnected' | 'error'
  const [liveBuffer, setLiveBuffer] = useState('');           // Accumulates recognized letters

  const scenarioIdxRef = useRef(0);
  const exchangeIdxRef = useRef(0);
  const timerRef = useRef(null);

  // ─── Check Flask API Health ─────────────────────────────────────────────
  const checkApiHealth = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/health`);
      if (res.ok) {
        const data = await res.json();
        setApiStatus('connected');
        return data;
      }
      setApiStatus('disconnected');
      return null;
    } catch {
      setApiStatus('disconnected');
      return null;
    }
  }, []);

  // Check API status on mount and periodically
  useEffect(() => {
    checkApiHealth();
    const interval = setInterval(checkApiHealth, 30000); // Every 30s
    return () => clearInterval(interval);
  }, [checkApiHealth]);

  // ─── Core Message System ────────────────────────────────────────────────
  const addMessage = useCallback((sender, text) => {
    const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    setMessages(prev => [
      ...prev,
      {
        id: Date.now() + Math.random(),
        sender,
        text,
        timestamp: timeStr
      }
    ]);
  }, []);

  /**
   * Typewriter stream effect
   */
  const streamText = useCallback((targetSide, rawFullText, onDone) => {
    const fullText = (rawFullText != null) ? String(rawFullText) : '';
    let charIdx = 0;
    if (timerRef.current) clearInterval(timerRef.current);

    if (targetSide === 'human') {
      setActiveSide('human');
      setIsStreamingHuman(true);
      setHumanText(fullText);
      setHumanStreamText('');

      timerRef.current = setInterval(() => {
        charIdx += 1;
        setHumanStreamText(fullText.slice(0, charIdx));
        if (charIdx >= fullText.length) {
          clearInterval(timerRef.current);
          setIsStreamingHuman(false);
          addMessage('human', fullText);
          if (onDone) onDone();
        }
      }, 45);
    } else {
      setActiveSide('robot');
      setIsStreamingRobot(true);
      setRobotText(fullText);
      setRobotStreamText('');

      timerRef.current = setInterval(() => {
        charIdx += 1;
        setRobotStreamText(fullText.slice(0, charIdx));
        if (charIdx >= fullText.length) {
          clearInterval(timerRef.current);
          setIsStreamingRobot(false);
          addMessage('robot', fullText);
          if (onDone) onDone();
        }
      }, 55);
    }
  }, [addMessage]);


  // ─── LIVE MODE: Send Landmarks to Flask API ─────────────────────────────
  const translateLandmarks = useCallback(async (landmarks) => {
    if (apiStatus !== 'connected') return null;

    try {
      const res = await fetch(`${API_BASE}/translate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ landmarks })
      });

      if (res.ok) {
        const prediction = await res.json();
        setLastPrediction(prediction);

        // Accumulate high-confidence letters into buffer
        if (prediction.confidence > 0.6) {
          setLiveBuffer(prev => prev + prediction.letter);
          // Update the human display text
          setHumanText(prev => {
            const newText = prev + prediction.letter;
            setHumanStreamText(newText);
            return newText;
          });
        }

        return prediction;
      }
    } catch (e) {
      console.warn('Translation API error:', e);
    }
    return null;
  }, [apiStatus]);

  // ─── Send Text to Arduino Robotic Hands ─────────────────────────────────
  const sendToRobot = useCallback(async (text) => {
    try {
      const res = await fetch(`${API_BASE}/robot/sign`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });

      if (res.ok) {
        const data = await res.json();
        // Display robot response in chat
        const robotMsg = data.arduino_connected
          ? `Signing: "${text}" on robotic hands`
          : `Queued "${text}" for robotic hands (Arduino not connected)`;
        streamText('robot', robotMsg);
        return data;
      }
    } catch (e) {
      console.warn('Robot sign API error:', e);
    }
    return null;
  }, [streamText]);

  // ─── LLM API: Refine Raw ISL Text Buffer ──────────────────────────────
  const refineWithLLM = useCallback(async (rawText) => {
    if (!rawText || !rawText.trim()) return null;
    try {
      const res = await fetch(`${API_BASE}/llm/refine`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: rawText.trim() })
      });
      if (res.ok) {
        const data = await res.json();
        return data; // { raw_text, refined_sentence, llm_provider, fallback_used }
      }
    } catch (e) {
      console.warn('LLM refine API error:', e);
    }
    return null;
  }, []);

  // ─── LLM API: Simplify Speech for Robotic Hands ─────────────────────────
  const simplifyWithLLM = useCallback(async (spokenText) => {
    if (!spokenText || !spokenText.trim()) return null;
    try {
      const res = await fetch(`${API_BASE}/llm/simplify`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: spokenText.trim() })
      });
      if (res.ok) {
        const data = await res.json();
        return data; // { original_speech, robot_keywords, llm_provider, fallback_used }
      }
    } catch (e) {
      console.warn('LLM simplify API error:', e);
    }
    return null;
  }, []);

  // ─── LLM API: Answer Specific User Question ─────────────────────────────
  const fetchAIAnswer = useCallback(async (userText) => {
    if (!userText || !userText.trim()) return null;
    const cleanQuery = userText.trim();
    try {
      const res = await fetch(`${API_BASE}/llm/answer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: cleanQuery })
      });
      if (res.ok) {
        const data = await res.json();
        if (data.answer) return data.answer;
      }
    } catch (e) {
      console.warn('LLM answer API error:', e);
    }

    // Client-side smart fallback if API endpoint is unreachable
    const q = cleanQuery.toLowerCase();
    if (q.includes('washroom') || q.includes('toilet') || q.includes('restroom') || q.includes('bathroom')) {
      return "The washroom is straight ahead to your left.";
    } else if (q.includes('hello') || q.includes('namaste') || q.includes('hi') || q.includes('hey')) {
      return "Namaste! How can I help you?";
    } else if (q.includes('how are you') || q.includes('how do you do') || q.includes('how r u')) {
      return "I am doing well, thank you! How can I help you with Indian Sign Language today?";
    } else if (q.includes('name') || q.includes('who are you')) {
      return "I am SignBridge AI, your dual-communication Indian Sign Language assistant.";
    } else if (q.includes('thank')) {
      return "You are very welcome! Happy to help.";
    } else if (q.includes('help') || q.includes('assist')) {
      return "I am here to assist you! You can sign or type your message.";
    } else if (q.includes('bye') || q.includes('goodbye')) {
      return "Goodbye! Have a wonderful day ahead.";
    } else if (q.includes('nice to meet you')) {
      return "Nice to meet you too! Welcome to SignBridge.";
    }
    return `I received your query: "${cleanQuery}". How can I assist you further?`;
  }, []);

  // ─── LIVE MODE: Commit Buffer as a Message ──────────────────────────────
  const commitLiveBuffer = useCallback(async () => {
    if (liveBuffer.trim()) {
      const rawBuf = liveBuffer.trim();
      setLiveBuffer('');
      setHumanStreamText('');

      // 1. Refine ISL raw letters to fluent sentence
      let queryText = rawBuf;
      const refined = await refineWithLLM(rawBuf);
      if (refined && refined.refined_sentence) {
        queryText = refined.refined_sentence;
      }

      addMessage('human', queryText);

      // 2. Fetch specific AI answer
      const aiAnswer = await fetchAIAnswer(queryText);
      streamText('robot', aiAnswer, () => {
        sendToRobot(aiAnswer);
      });
    }
  }, [liveBuffer, addMessage, refineWithLLM, fetchAIAnswer, streamText, sendToRobot]);


  // ─── DEMO MODE: Simulated Conversation ──────────────────────────────────
  const triggerNextHuman = useCallback(() => {
    setIsAutoDemo(false);
    const text = HUMAN_INPUT_POOL[Math.floor(Math.random() * HUMAN_INPUT_POOL.length)];
    const cleanText = text.replace(/[\u{1F600}-\u{1F64F}\u{1F300}-\u{1F5FF}\u{1F680}-\u{1F6FF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}]/gu, '').trim();

    streamText('human', text, async () => {
      const aiAnswer = await fetchAIAnswer(cleanText);
      streamText('robot', aiAnswer);
    });
  }, [streamText, fetchAIAnswer]);

  const triggerNextRobot = useCallback(() => {
    setIsAutoDemo(false);
    const text = ROBOT_INPUT_POOL[Math.floor(Math.random() * ROBOT_INPUT_POOL.length)];
    streamText('robot', text);
  }, [streamText]);

  const sendMessage = useCallback((customText) => {
    if (!customText || !customText.trim()) return;
    const cleanText = customText.trim();
    setIsAutoDemo(false);
    if (timerRef.current) clearInterval(timerRef.current);

    // 1. Instantly show sent message in the chat thread
    addMessage('human', cleanText);
    setHumanText(cleanText);
    setHumanStreamText('');
    setIsStreamingHuman(false);
    setIsStreamingRobot(false);

    // 2. Fetch AI response and stream it to the robot panel
    (async () => {
      const fetchedAnswer = await fetchAIAnswer(cleanText);
      const aiAnswer = fetchedAnswer || `I received your message: "${cleanText}". How can I help you further with Indian Sign Language?`;
      
      streamText('robot', aiAnswer, async () => {
        if (apiStatus === 'connected') {
          // Extract keywords via LLM simplify before robotic fingerspelling
          const simplified = await simplifyWithLLM(aiAnswer);
          const textToSign = simplified?.robot_keywords || aiAnswer;
          sendToRobot(textToSign);
        }
      });
    })();
  }, [addMessage, fetchAIAnswer, streamText, apiStatus, simplifyWithLLM, sendToRobot]);


  const clearAll = useCallback(() => {
    setIsAutoDemo(false);
    if (timerRef.current) clearInterval(timerRef.current);
    setHumanText('');
    setRobotText('');
    setHumanStreamText('');
    setRobotStreamText('');
    setActiveSide(null);
    setIsStreamingHuman(false);
    setIsStreamingRobot(false);
    setMessages([]);
    setLiveBuffer('');
    setLastPrediction(null);
  }, []);

  // Auto conversation loop (demo mode only)
  useEffect(() => {
    if (!isAutoDemo) return;

    const currentScenario = ISL_CONVERSATIONS[scenarioIdxRef.current];
    const exchanges = currentScenario.exchanges;
    const currentExchange = exchanges[exchangeIdxRef.current];

    if (!currentExchange) {
      scenarioIdxRef.current = (scenarioIdxRef.current + 1) % ISL_CONVERSATIONS.length;
      exchangeIdxRef.current = 0;
      return;
    }

    streamText(currentExchange.speaker, currentExchange.text, () => {
      const delay = setTimeout(() => {
        exchangeIdxRef.current += 1;
        if (exchangeIdxRef.current >= exchanges.length) {
          scenarioIdxRef.current = (scenarioIdxRef.current + 1) % ISL_CONVERSATIONS.length;
          exchangeIdxRef.current = 0;
        }
      }, 1500);

      return () => clearTimeout(delay);
    });

    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [isAutoDemo, streamText]);

  return {
    // Existing display state
    humanText,
    robotText,
    humanStreamText,
    robotStreamText,
    activeSide,
    isStreamingHuman,
    isStreamingRobot,
    messages,
    sendMessage,
    isAutoDemo,
    setIsAutoDemo,
    triggerNextHuman,
    triggerNextRobot,
    clearAll,

    // NEW: Live translation API
    isLiveMode,
    setIsLiveMode,
    translateLandmarks,
    commitLiveBuffer,
    lastPrediction,
    liveBuffer,
    apiStatus,
    checkApiHealth,
    sendToRobot,

    // LLM Fallback Engine API
    refineWithLLM,
    simplifyWithLLM,
  };
}
