import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Delete, Trash2, Volume2, Sparkles, Hand, Clock, Send } from 'lucide-react';

export const SentenceBuilder = React.memo(({
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
  onCancelCountdown = null,
}) => {
  const [isRefining, setIsRefining] = useState(false);

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

  const handleDeleteLast = onDelete || onUndo;

  return (
    <div className="draft-controls-wrap">
      {/* Animated Status Strip: Countdown */}
      <AnimatePresence>
        {inactivityCountdown !== null && sentence ? (
          <motion.div 
            key="countdown-pill"
            initial={{ opacity: 0, y: -6, scale: 0.96 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -4, scale: 0.96 }}
            transition={{ duration: 0.2 }}
            className="draft-status-pill countdown-pill"
          >
            <Clock size={13} className="animate-spin text-indigo-500" />
            <span>Auto-sending in <strong>{inactivityCountdown}s</strong></span>
            {onCancelCountdown && (
              <button
                type="button"
                onClick={onCancelCountdown}
                className="draft-pill-cancel-btn"
                title="Cancel countdown"
              >
                Cancel
              </button>
            )}
          </motion.div>
        ) : null}
      </AnimatePresence>

      {/* Clean, Balanced Action Toolbar */}
      <div className="draft-toolbar">
        {/* Left Action Group: Editing & Sending */}
        <div className="draft-toolbar-group">
          {onSend && (
            <button 
              type="button"
              className="draft-btn btn-send" 
              onClick={() => onSend(sentence)} 
              disabled={!sentence}
              title="Send message (Enter)"
            >
              <Send size={13} />
              <span>Send</span>
            </button>
          )}

          <button 
            type="button"
            className="draft-btn" 
            onClick={onAddSpace} 
            disabled={!sentence}
            title="Add space between words"
          >
            Space
          </button>

          <button 
            type="button"
            className="draft-btn" 
            onClick={handleDeleteLast} 
            disabled={!sentence} 
            title="Delete / Backspace last item"
          >
            <Delete size={13} />
            <span>Delete</span>
          </button>

          <button 
            type="button"
            className="draft-btn btn-danger" 
            onClick={onClear} 
            disabled={!sentence} 
            title="Clear draft"
          >
            <Trash2 size={13} />
            <span>Clear</span>
          </button>
        </div>

        {/* Right Action Group: Intelligence & Voice */}
        <div className="draft-toolbar-group">
          {onAIRefine && (
            <button 
              type="button"
              className="draft-btn btn-refine" 
              onClick={handleRefineClick} 
              disabled={!sentence || isRefining}
              title="Refine sentence with AI"
            >
              <Sparkles size={13} className={isRefining ? 'animate-spin' : ''} />
              <span>{isRefining ? 'Refining...' : 'AI Refine'}</span>
            </button>
          )}

          <button 
            type="button"
            className="draft-btn" 
            onClick={handleSpeak} 
            disabled={!sentence} 
            title="Speak aloud"
          >
            <Volume2 size={13} />
          </button>
          
          {onToggleAutoSend && (
            <button
              type="button"
              onClick={onToggleAutoSend}
              className={`draft-btn btn-toggle ${autoSendEnabled ? 'is-active' : ''}`}
              title="Toggle Hands-Free Auto-Send on Hand Drop"
            >
              <Hand size={12} />
              <span>Auto-Send</span>
              {autoSendEnabled && <span className="draft-toggle-dot" />}
            </button>
          )}
        </div>
      </div>
    </div>
  );
});

SentenceBuilder.displayName = 'SentenceBuilder';