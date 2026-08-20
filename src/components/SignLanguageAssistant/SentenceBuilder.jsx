import React, { useState } from 'react';
import { Undo2, Delete, Trash2, Copy, Volume2, Sparkles } from 'lucide-react';

export const SentenceBuilder = ({
  sentence,
  onUndo,
  onDelete,
  onClear,
  onAddSpace,
  onAIRefine,
}) => {
  const [isRefining, setIsRefining] = useState(false);

  const handleCopy = () => {
    if (sentence) navigator.clipboard.writeText(sentence);
  };

  const handleSpeak = () => {
    if (sentence && 'speechSynthesis' in window) {
      const utterance = new SpeechSynthesisUtterance(sentence);
      utterance.rate = 0.9;
      window.speechSynthesis.speak(utterance);
    }
  };

  const handleRefineClick = async () => {
    if (!sentence || !onAIRefine || isRefining) return;
    setIsRefining(true);
    try {
      await onAIRefine(sentence);
    } finally {
      setIsRefining(false);
    }
  };

  return (
    <div className="sla-builder-actions">
      <button className="sla-action-btn" onClick={onAddSpace} disabled={!sentence}>
        Space
      </button>
      <button className="sla-action-btn" onClick={onUndo} disabled={!sentence} title="Undo last letter">
        <Undo2 size={12} /> Undo
      </button>
      <button className="sla-action-btn" onClick={onDelete} disabled={!sentence} title="Delete last letter">
        <Delete size={12} /> Delete
      </button>
      <button 
        className="sla-action-btn danger" 
        onClick={onClear} 
        disabled={!sentence} 
        title="Clear entire sentence"
        style={sentence ? { color: '#ef4444', borderColor: 'rgba(239, 68, 68, 0.35)', background: 'rgba(239, 68, 68, 0.06)' } : {}}
      >
        <Trash2 size={12} /> Clear
      </button>
      <button className="sla-action-btn" onClick={handleCopy} disabled={!sentence} title="Copy to clipboard">
        <Copy size={12} /> Copy
      </button>
      <button className="sla-action-btn" onClick={handleSpeak} disabled={!sentence} title="Speak sentence aloud">
        <Volume2 size={12} /> Speak
      </button>
      
      {onAIRefine && (
        <button 
          className="sla-action-btn primary" 
          onClick={handleRefineClick} 
          disabled={!sentence || isRefining}
          title="Refine sentence with AI (Groq / Gemini)"
          style={{ background: 'linear-gradient(135deg, #6366f1, #a855f7)', color: '#fff', border: 'none' }}
        >
          <Sparkles size={12} className={isRefining ? 'animate-spin' : ''} /> {isRefining ? 'Refining...' : 'AI Refine ✨'}
        </button>
      )}
    </div>
  );
};