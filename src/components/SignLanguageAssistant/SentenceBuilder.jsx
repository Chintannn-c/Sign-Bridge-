import React, { useState } from 'react';
import { Undo2, Delete, Trash2, Copy, Volume2, Sparkles, Hand, Clock, CheckCircle2, Send } from 'lucide-react';

export const SentenceBuilder = ({
  sentence,
  onUndo,
  onDelete,
  onClear,
  onAddSpace,
  onAIRefine,
  onSend,
  inactivityCountdown = null,
  autoSendEnabled = true,
  onToggleAutoSend = null,
  lastAutoSpoken = null,
  onCancelCountdown = null,
}) => {
  const [isRefining, setIsRefining] = useState(false);

  const cleanSpoken = (lastAutoSpoken || '')
    .replace(/<think>[\s\S]*?(?:<\/think>|$)/gi, '')
    .replace(/^(?:Here'?s (?:a )?(?:quick )?thinking process:?|Thinking Process:?)[\s\S]*?(?=\n\n|\n|$)/gi, '')
    .trim();

  const handleCopy = () => {
    if (sentence) navigator.clipboard.writeText(sentence);
  };

  const handleSpeak = () => {
    if (sentence && 'speechSynthesis' in window) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(sentence);
      utterance.rate = 0.95;
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
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', width: '100%' }}>
      {/* Inactivity / Hand-Drop Countdown Banner */}
      {inactivityCountdown !== null && sentence && (
        <div 
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.12), rgba(168, 85, 247, 0.12))',
            border: '1px solid rgba(99, 102, 241, 0.35)',
            borderRadius: '10px',
            padding: '0.45rem 0.8rem',
            fontSize: '0.8rem',
            color: 'var(--text-espresso, #2D2A26)',
            animation: 'pulse 1.5s infinite'
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <Clock size={14} className="animate-spin text-indigo-500" />
            <span>
              Hands dropped: Auto-sending in <strong>{inactivityCountdown}s</strong>... (keep signing to continue)
            </span>
          </div>
          {onCancelCountdown && (
            <button
              onClick={onCancelCountdown}
              style={{
                background: 'rgba(45, 42, 38, 0.1)',
                border: 'none',
                borderRadius: '6px',
                padding: '0.2rem 0.5rem',
                color: 'var(--text-espresso, #2D2A26)',
                fontSize: '0.75rem',
                fontWeight: 600,
                cursor: 'pointer'
              }}
            >
              Cancel
            </button>
          )}
        </div>
      )}

      {/* Clean auto-spoken toast with high contrast */}
      {cleanSpoken && !sentence && (
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            background: 'rgba(110, 127, 107, 0.15)',
            border: '1px solid rgba(110, 127, 107, 0.35)',
            borderRadius: '10px',
            padding: '0.45rem 0.85rem',
            fontSize: '0.8rem',
            color: 'var(--text-espresso, #2D2A26)',
            boxShadow: '0 2px 6px rgba(45, 42, 38, 0.04)'
          }}
        >
          <CheckCircle2 size={15} color="#475845" />
          <span>Auto-sent to dialogue: <strong style={{ color: '#2D2A26' }}>"{cleanSpoken}"</strong></span>
        </div>
      )}

      {/* Action Toolbar */}
      <div className="sla-builder-actions" style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem', alignItems: 'center' }}>
        {onSend && (
          <button 
            className="sla-action-btn primary" 
            onClick={() => onSend(sentence)} 
            disabled={!sentence}
            title="Send sentence to conversation dialogue"
            style={{
              background: sentence ? 'linear-gradient(135deg, #818cf8, #6366f1)' : undefined,
              color: sentence ? '#FFFFFF' : undefined,
              border: sentence ? 'none' : undefined,
              fontWeight: 600,
              boxShadow: sentence ? '0 2px 8px rgba(99, 102, 241, 0.25)' : undefined
            }}
          >
            <Send size={12} /> Send
          </button>
        )}

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

        {/* Auto-Send Mode Toggle */}
        {onToggleAutoSend && (
          <button
            onClick={onToggleAutoSend}
            className="sla-action-btn"
            title="Toggle Hands-Free Auto-Send on Hand Drop"
            style={{
              marginLeft: 'auto',
              fontSize: '0.75rem',
              color: autoSendEnabled ? '#6366f1' : '#888',
              borderColor: autoSendEnabled ? 'rgba(99, 102, 241, 0.4)' : 'rgba(200, 173, 147, 0.4)',
              background: autoSendEnabled ? 'rgba(99, 102, 241, 0.1)' : 'transparent',
              fontWeight: 600
            }}
          >
            <Hand size={12} /> Hand-Drop Auto-Send: <strong>{autoSendEnabled ? 'ON' : 'OFF'}</strong>
          </button>
        )}
      </div>
    </div>
  );
};