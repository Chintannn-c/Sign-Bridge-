import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Keyboard, Send, X, Trash2 } from 'lucide-react';
import { CameraView } from './CameraView';
import { SentenceBuilder } from './SignLanguageAssistant/SentenceBuilder';

/**
 * HumanPanel Component - Left Panel Card ("YOU")
 * Horizontal widescreen webcam aspect ratio at top, prominent text underneath.
 * Optimized with React.memo for max UI performance.
 */
export const HumanPanel = React.memo(({
  fullText,
  streamingText,
  isActive,
  isStreaming,
  onSendMessage
}) => {
  const [recognitionState, setRecognitionState] = useState(null);
  const [showTextbox, setShowTextbox] = useState(false);
  const [tempInput, setTempInput] = useState('');

  // Camera & recognition state logic
  const isCameraOn = Boolean(recognitionState?.isCameraOn);
  const rawBuffer = recognitionState?.sentenceBuffer || '';
  const sentenceBuffer = isCameraOn ? rawBuffer : (fullText || '');
  const hasLiveSentence = Boolean(sentenceBuffer && sentenceBuffer.trim().length > 0);
  const displayText = isStreaming
    ? streamingText
    : (hasLiveSentence
        ? sentenceBuffer
        : (isCameraOn ? 'Start signing or type below...' : (fullText || 'Start signing or type below...')));

  const handleSubmit = (e) => {
    if (e) {
      e.preventDefault();
      e.stopPropagation();
    }
    const msg = tempInput.trim();
    if (msg) {
      if (onSendMessage) {
        onSendMessage(msg);
      }
      if (recognitionState?.clearBuffer) {
        recognitionState.clearBuffer();
      }
      setTempInput('');
      setShowTextbox(false);
    }
  };

  return (
    <div
      className={`panel-card left-card ${isActive ? 'active-panel' : ''}`}
    >
      {/* Card Header Strip */}
      <div className="card-header">
        <div className="card-label">
          <span>YOU</span>
          {isActive && <span className="listening-dot" />}
        </div>
        <div className="chat-header-actions" onClick={e => e.stopPropagation()}>
          {isCameraOn && hasLiveSentence && (
            <button
              className="chat-action-btn"
              onClick={recognitionState?.clearBuffer}
              title="Clear Live Sentence"
              style={{ color: '#ef4444', borderColor: 'rgba(239, 68, 68, 0.35)', background: 'rgba(239, 68, 68, 0.08)' }}
            >
              <Trash2 size={14} />
            </button>
          )}
          <button
            className={`chat-action-btn ${showTextbox ? 'is-active-toggle' : ''}`}
            onClick={() => setShowTextbox(prev => !prev)}
            title={showTextbox ? 'Hide Input' : 'Type / Edit'}
          >
            <Keyboard size={15} />
          </button>
        </div>
      </div>

      {/* Card Body with Horizontal Widescreen Camera Feed */}
      <div className="human-card-body">
        {/* Horizontal Webcam Feed */}
        <CameraView 
          isActive={isActive} 
          onRecognitionUpdate={setRecognitionState}
        />

        {/* Text Feed below the Horizontal Webcam */}
        <div className="card-text-wrapper human-text-wrapper" style={{ flexDirection: 'column', alignItems: 'flex-start', justifyContent: 'center' }}>
          {!showTextbox && (
            <div style={{ width: '100%', position: 'relative', display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '0.75rem' }}>
              <h2 className={`card-text ${isCameraOn && !hasLiveSentence ? 'text-slate-400 italic' : ''}`} style={{ flex: 1, margin: 0 }}>
                {displayText}
                {isStreaming && !isCameraOn && <span className="streaming-cursor-teal" />}
                {isCameraOn && recognitionState?.status === 'detecting' && <span className="streaming-cursor-teal" />}
              </h2>
              {isCameraOn && hasLiveSentence && (
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    recognitionState?.clearBuffer();
                  }}
                  title="Clear sentence"
                  style={{
                    background: 'rgba(239, 68, 68, 0.1)',
                    color: '#ef4444',
                    border: '1px solid rgba(239, 68, 68, 0.25)',
                    borderRadius: '8px',
                    padding: '0.35rem 0.65rem',
                    fontSize: '0.75rem',
                    fontWeight: 600,
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.35rem',
                    flexShrink: 0,
                    transition: 'all 0.2s',
                    marginTop: '0.2rem'
                  }}
                  onMouseEnter={e => {
                    e.currentTarget.style.background = '#ef4444';
                    e.currentTarget.style.color = '#fff';
                  }}
                  onMouseLeave={e => {
                    e.currentTarget.style.background = 'rgba(239, 68, 68, 0.1)';
                    e.currentTarget.style.color = '#ef4444';
                  }}
                >
                  <Trash2 size={13} />
                  <span>Clear</span>
                </button>
              )}
            </div>
          )}

          {/* Temporary Custom Textbox Input */}
          <AnimatePresence>
            {showTextbox && (
              <motion.div
                initial={{ opacity: 0, y: 8, height: 0 }}
                animate={{ opacity: 1, y: 0, height: 'auto' }}
                exit={{ opacity: 0, y: 8, height: 0 }}
                style={{ width: '100%', marginTop: 0 }}
                onClick={e => e.stopPropagation()}
              >
                <form 
                  onSubmit={handleSubmit}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.5rem',
                    background: 'rgba(255, 255, 255, 0.95)',
                    padding: '0.4rem 0.6rem',
                    borderRadius: '12px',
                    border: '1.5px solid var(--accent-camel)',
                    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.08)'
                  }}
                >
                  <input
                    type="text"
                    placeholder="Type custom text/question..."
                    value={tempInput}
                    onChange={(e) => setTempInput(e.target.value)}
                    autoFocus
                    style={{
                      flex: 1,
                      border: 'none',
                      background: 'transparent',
                      outline: 'none',
                      fontSize: '0.9rem',
                      color: 'var(--text-espresso)',
                      padding: '0.3rem'
                    }}
                  />
                  <button
                    type="submit"
                    title="Send message"
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.3rem',
                      padding: '0.4rem 0.75rem',
                      borderRadius: '8px',
                      background: 'var(--accent-sage)',
                      color: '#fff',
                      border: 'none',
                      fontSize: '0.8rem',
                      fontWeight: 600,
                      cursor: 'pointer'
                    }}
                  >
                    <Send size={13} />
                    <span>Send</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => setShowTextbox(false)}
                    title="Close"
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      padding: '0.4rem',
                      borderRadius: '8px',
                      background: 'transparent',
                      color: 'var(--text-espresso)',
                      border: 'none',
                      cursor: 'pointer'
                    }}
                  >
                    <X size={14} />
                  </button>
                </form>
              </motion.div>
            )}
          </AnimatePresence>
          
          {/* Action controls for the live sentence */}
          {isCameraOn && recognitionState && (
            <div style={{ marginTop: '1rem', width: '100%', opacity: hasLiveSentence ? 1 : 0.5, transition: 'opacity 0.2s' }} onClick={e => e.stopPropagation()}>
              <SentenceBuilder 
                sentence={sentenceBuffer}
                onUndo={recognitionState.undoLetter}
                onDelete={recognitionState.deleteLetter}
                onClear={recognitionState.clearBuffer}
                onAddSpace={recognitionState.addSpace}
                onAIRefine={recognitionState.refineSentence}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
});
