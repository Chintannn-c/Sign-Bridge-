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
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 2000);
      const res = await fetch(`${API_BASE}/health`, { signal: controller.signal });
      clearTimeout(timeoutId);
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
    return () => {
      clearInterval(interval);
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [checkApiHealth]);

  // ─── Core Message System ────────────────────────────────────────────────
  const addMessage = useCallback((sender, text) => {
    if (!text || text.includes('Arduino not connected')) return;
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
    if (fullText.includes('Arduino not connected')) {
      if (onDone) onDone();
      return;
    }
    let charIdx = 0;
    if (timerRef.current) clearInterval(timerRef.current);
    setIsStreamingHuman(false);
    setIsStreamingRobot(false);

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

  // ─── Send Text to Arduino Robotic Hands (Hardware actuation only) ───────
  const sendToRobot = useCallback(async (text) => {
    try {
      const res = await fetch(`${API_BASE}/robot/sign`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });

      if (res.ok) {
        return await res.json();
      }
    } catch (e) {
      console.warn('Robot sign API error:', e);
    }
    return null;
  }, []);

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

    // Client-side smart fallback with word boundaries if API endpoint is unreachable
    const q = cleanQuery.toLowerCase();
    const hasWord = (regex) => regex.test(q);

    if (q.length <= 2 && !['hi', 'no', 'ok'].includes(q)) {
      return "Could you sign that again?";
    } else if (hasWord(/\b(washroom|toilet|restroom|bathroom)\b/)) {
      return "Down the hall on your left.";
    } else if (hasWord(/\b(water|drink|thirsty)\b/)) {
      return "Sure thing! Here's some water for you.";
    } else if (hasWord(/\b(food|hungry|eat)\b/)) {
      return "Let's grab a bite! What are you craving?";
    } else if (hasWord(/\b(hello|namaste|hi|hey)\b/)) {
      return "Hey! How's your day going?";
    } else if (hasWord(/\b(how are you|how do you do|how r u)\b/)) {
      return "Doing great, thanks! How about you?";
    } else if (hasWord(/\b(name|who are you)\b/)) {
      return "I'm SignBridge! Nice to meet you.";
    } else if (hasWord(/\b(thank|thanks)\b/)) {
      return "Anytime! Happy to help.";
    } else if (hasWord(/\b(help|assist)\b/)) {
      return "I'm right here! How can I help?";
    } else if (hasWord(/\b(bye|goodbye)\b/)) {
      return "Take care! See you soon.";
    } else if (q.includes('nice to meet you')) {
      return "Nice to meet you too!";
    }
    return "Got it! How can I help with that?";
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

  const sendMessage = useCallback((customText, sender = 'human') => {
    if (!customText || !customText.trim()) return;
    const cleanText = customText.trim();
    setIsAutoDemo(false);
    if (timerRef.current) clearInterval(timerRef.current);

    if (sender === 'robot') {
      // Path 2: Hearing desk staff voice input -> simplify -> robot hands fingerspelling
      addMessage('robot', cleanText);
      setRobotText(cleanText);
      setRobotStreamText('');
      setIsStreamingHuman(false);
      setIsStreamingRobot(false);

      (async () => {
        if (apiStatus === 'connected') {
          const simplified = await simplifyWithLLM(cleanText);
          const rawKeywords = simplified?.robot_keywords || cleanText;
          const textToSign = rawKeywords.toUpperCase().replace(/_/g, ' ').replace(/[^A-Z0-9 ]/g, ' ').replace(/\s+/g, ' ').trim();
          if (textToSign) {
            sendToRobot(textToSign);
          }
        }
      })();
      return;
    }

    // Path 1: Deaf user signed input -> AI response -> robot fingerspelling
    addMessage('human', cleanText);
    setHumanText(cleanText);
    setHumanStreamText('');
    setIsStreamingHuman(false);
    setIsStreamingRobot(false);

    (async () => {
      const fetchedAnswer = await fetchAIAnswer(cleanText);
      const aiAnswer = fetchedAnswer || "Got it! How can I help with that?";
      
      streamText('robot', aiAnswer, async () => {
        if (apiStatus === 'connected') {
          const simplified = await simplifyWithLLM(aiAnswer);
          const rawKeywords = simplified?.robot_keywords || aiAnswer;
          const textToSign = rawKeywords.toUpperCase().replace(/_/g, ' ').replace(/[^A-Z0-9 ]/g, ' ').replace(/\s+/g, ' ').trim();
          if (textToSign) {
            sendToRobot(textToSign);
          }
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

  const [demoStep, setDemoStep] = useState(0);
  const autoDemoTimeoutRef = useRef(null);

  // Auto conversation loop (demo mode only) - uses demoStep state to reliably progress
  useEffect(() => {
    if (!isAutoDemo) {
      if (autoDemoTimeoutRef.current) clearTimeout(autoDemoTimeoutRef.current);
      return;
    }

    const currentScenario = ISL_CONVERSATIONS[scenarioIdxRef.current];
    const exchanges = currentScenario.exchanges;
    const currentExchange = exchanges[exchangeIdxRef.current];

    if (!currentExchange) {
      scenarioIdxRef.current = (scenarioIdxRef.current + 1) % ISL_CONVERSATIONS.length;
      exchangeIdxRef.current = 0;
      setDemoStep(prev => prev + 1);
      return;
    }

    streamText(currentExchange.speaker, currentExchange.text, () => {
      autoDemoTimeoutRef.current = setTimeout(() => {
        exchangeIdxRef.current += 1;
        if (exchangeIdxRef.current >= exchanges.length) {
          scenarioIdxRef.current = (scenarioIdxRef.current + 1) % ISL_CONVERSATIONS.length;
          exchangeIdxRef.current = 0;
        }
        setDemoStep(prev => prev + 1);
      }, 1500);
    });

    return () => {
      if (autoDemoTimeoutRef.current) clearTimeout(autoDemoTimeoutRef.current);
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [isAutoDemo, demoStep, streamText]);

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
