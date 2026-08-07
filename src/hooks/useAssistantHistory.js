import { useState, useCallback } from 'react';

/**
 * Manages localStorage-based history of recognised ISL phrases.
 * Each entry stores the sentence, timestamp, confidence, and gesture sequence.
 */
const STORAGE_KEY = 'signbridge_assistant_history';

function loadHistory() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveHistory(entries) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(entries));
  } catch {
    // Storage full or unavailable
  }
}

export function useAssistantHistory() {
  const [history, setHistory] = useState(() => loadHistory());

  const addEntry = useCallback((sentence, confidence = 0, gestures = []) => {
    const entry = {
      id: Date.now() + Math.random(),
      sentence,
      confidence: Math.round(confidence * 100),
      gestures,
      date: new Date().toLocaleDateString(),
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      timestamp: Date.now(),
    };
    setHistory(prev => {
      const updated = [entry, ...prev].slice(0, 100); // Keep last 100
      saveHistory(updated);
      return updated;
    });
    return entry;
  }, []);

  const deleteEntry = useCallback((id) => {
    setHistory(prev => {
      const updated = prev.filter(e => e.id !== id);
      saveHistory(updated);
      return updated;
    });
  }, []);

  const clearHistory = useCallback(() => {
    setHistory([]);
    saveHistory([]);
  }, []);

  return {
    history,
    addEntry,
    deleteEntry,
    clearHistory,
  };
}
