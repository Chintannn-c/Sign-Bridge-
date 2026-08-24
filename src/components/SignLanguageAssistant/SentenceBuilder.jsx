import React, { useState } from 'react';
import { Undo2, Delete, Trash2, Copy, Volume2, Sparkles, Hand, Clock, CheckCircle2 } from 'lucide-react';

export const SentenceBuilder = ({
  sentence,
  onUndo,
  onDelete,
  onClear,
  onAddSpace,
  onAIRefine,
  inactivityCountdown = null,
  autoSendEnabled = true,
  onToggleAutoSend = null,
  lastAutoSpoken = null,
  onCancelCountdown = null,
}) => {
  const [isRefining, setIsRefining] = useState(false);

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
            background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(168, 85, 247, 0.15))',
            border: '1px solid rgba(99, 102, 241, 0.35)',
            borderRadius: '10px',
            padding: '0.45rem 0.8rem',
            fontSize: '0.8rem',
            color: '#e0e7ff',
            animation: 'pulse 1.5s infinite'
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <Clock size={14} className="animate-spin text-indigo-400" />
            <span>
              Hands dropped: Auto-speaking in <strong>{inactivityCountdown}s</strong>... (keep signing to continue)
            </span>
          </div>
          {onCancelCountdown && (
            <button
              onClick={onCancelCountdown}
              style={{
                background: 'rgba(255, 255, 255, 0.12)',
                border: 'none',
                borderRadius: '6px',
                padding: '0.2rem 0.5rem',
                color: '#fff',
                fontSize: '0.75rem',
                cursor: 'pointer'
              }}
            >
              Cancel
            </button>
          )}
        </div>
      )}

      {/* Last auto-spoken toast */}
      {lastAutoSpoken && !sentence && (
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.4rem',
            background: 'rgba(34, 197, 94, 0.12)',
            border: '1px solid rgba(34, 197, 94, 0.3)',
            borderRadius: '8px',
            padding: '0.35rem 0.75rem',
            fontSize: '0.75rem',
            color: '#86efac'
          }}
        >
          <CheckCircle2 size={13} />
          <span>Auto-spoken aloud: <em>"{lastAutoSpoken}"</em></span>
        </div>
      )}

      {/* Action Toolbar */}
      <div className="sla-builder-actions" style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem', alignItems: 'center' }}>
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
              color: autoSendEnabled ? '#818cf8' : '#94a3b8',
              borderColor: autoSendEnabled ? 'rgba(129, 140, 248, 0.4)' : 'rgba(255,255,255,0.1)',
              background: autoSendEnabled ? 'rgba(99, 102, 241, 0.1)' : 'transparent'
            }}
          >
            <Hand size={12} /> Hand-Drop Auto-Send: <strong>{autoSendEnabled ? 'ON' : 'OFF'}</strong>
          </button>
        )}
      </div>
    </div>
  );
};