import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Keyboard, Send, X } from 'lucide-react';
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
  onClick,
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
  const displayText = isCameraOn
    ? (hasLiveSentence ? sentenceBuffer : 'Start signing or type below...')
    : (isStreaming ? streamingText : fullText);

  const handleSubmit = (e) => {
    if (e) e.preventDefault();
    if (tempInput.trim() && onSendMessage) {
      onSendMessage(tempInput.trim());
      setTempInput('');
    }
  };

  return (
    <div
      className={`panel-card left-card ${isActive ? 'active-panel' : ''}`}
      onClick={!isCameraOn ? onClick : undefined} // Only allow demo click if camera is off
    >
      {!isCameraOn && <div className="click-layer" title="Click to trigger live input simulation" />}

      {/* Card Header Strip */}
      <div className="card-header">
        <div className="card-label">
          <span>YOU</span>
          {isActive && <span className="listening-dot" />}
        </div>
        <div className="chat-header-actions" onClick={e => e.stopPropagation()}>
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
            <h2 className={`card-text ${isCameraOn && !hasLiveSentence ? 'text-slate-400 italic' : ''}`}>
              {displayText}
              {isStreaming && !isCameraOn && <span className="streaming-cursor-teal" />}
              {isCameraOn && recognitionState?.status === 'detecting' && <span className="streaming-cursor-teal" />}
            </h2>
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
