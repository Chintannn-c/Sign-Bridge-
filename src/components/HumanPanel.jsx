import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Keyboard, Send, X, Trash2, Sparkles } from 'lucide-react';
import { CameraView } from './CameraView';
import { SentenceBuilder } from './SignLanguageAssistant/SentenceBuilder';

/**
 * Dynamically scale font size down as message length grows so full text stays completely visible.
 */
function getDynamicFontSize(textLength) {
  if (textLength > 65) return 'clamp(0.85rem, 1.0vw, 1.05rem)';
  if (textLength > 45) return 'clamp(0.95rem, 1.2vw, 1.2rem)';
  if (textLength > 28) return 'clamp(1.1rem, 1.5vw, 1.4rem)';
  if (textLength > 16) return 'clamp(1.3rem, 1.9vw, 1.75rem)';
  if (textLength > 8) return 'clamp(1.55rem, 2.3vw, 2.1rem)';
  return 'clamp(1.8rem, 2.8vw, 2.5rem)';
}

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
  const textScrollContainerRef = useRef(null);

  // Camera & recognition state logic
  const isCameraOn = Boolean(recognitionState?.isCameraOn);
  const rawBuffer = recognitionState?.sentenceBuffer || '';
  const sentenceBuffer = isCameraOn ? rawBuffer : (fullText || '');
  const hasLiveSentence = Boolean(sentenceBuffer && sentenceBuffer.trim().length > 0);
  const isAutoSending = Boolean(recognitionState?.isAutoSending);
  const inactivityCountdown = recognitionState?.inactivityCountdown;
  const detectedSign = recognitionState?.detectedSign || recognitionState?.detectedLetter || recognitionState?.detectedWord;
  const displayText = isStreaming
    ? streamingText
    : (hasLiveSentence
        ? sentenceBuffer
        : (isCameraOn ? 'Start signing or type below...' : (fullText || 'Start signing or type below...')));

  // Auto-scroll to ensure latest incoming characters and words are always visible
  useEffect(() => {
    if (textScrollContainerRef.current) {
      textScrollContainerRef.current.scrollTop = textScrollContainerRef.current.scrollHeight;
    }
  }, [sentenceBuffer, displayText]);

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
          onSendMessage={onSendMessage}
        />

        {/* Unified Message Draft Composer */}
        <div className="message-draft-composer">
          {/* Draft Text Display Area */}
          <div className="draft-text-area" ref={textScrollContainerRef}>
            <AnimatePresence mode="wait">
              {showTextbox ? (
                <motion.div
                  key="textbox"
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -6 }}
                  transition={{ duration: 0.18 }}
                  style={{ width: '100%' }}
                  onClick={e => e.stopPropagation()}
                >
                  <form onSubmit={handleSubmit} className="draft-input-form">
                    <input
                      type="text"
                      placeholder="Type custom text/question..."
                      value={tempInput}
                      onChange={(e) => setTempInput(e.target.value)}
                      autoFocus
                      className="draft-input-field"
                    />
                    <button
                      type="submit"
                      disabled={!tempInput.trim()}
                      className="draft-input-send-btn"
                      title="Send message"
                    >
                      <Send size={14} />
                    </button>
                    <button
                      type="button"
                      onClick={() => setShowTextbox(false)}
                      className="draft-input-close-btn"
                      title="Close"
                    >
                      <X size={14} />
                    </button>
                  </form>
                </motion.div>
              ) : (
                <motion.div
                  key="draft-display"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.15 }}
                  style={{ width: '100%' }}
                >
                  {hasLiveSentence ? (
                    <div className="draft-sentence-text">
                      <span>{sentenceBuffer.length > 1 ? sentenceBuffer.slice(0, -1) : ''}</span>
                      <span key={`last-${sentenceBuffer.length}`} className="draft-last-char">
                        {sentenceBuffer.slice(-1) === ' ' ? '\u00A0' : sentenceBuffer.slice(-1)}
                      </span>
                      <span className="draft-cursor" />
                    </div>
                  ) : (
                    <div
                      className="draft-placeholder-text"
                      onClick={() => setShowTextbox(true)}
                      title="Click to type with keyboard"
                    >
                      {displayText}
                      {isStreaming && <span className="streaming-cursor-teal" />}
                    </div>
                  )}
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Action Toolbar */}
          <SentenceBuilder 
            sentence={sentenceBuffer}
            onUndo={recognitionState?.undoLetter}
            onDelete={recognitionState?.deleteLetter}
            onClear={recognitionState?.clearBuffer}
            onAddSpace={recognitionState?.addSpace}
            onAIRefine={recognitionState?.refineSentence}
            onSend={() => {
              if (recognitionState?.sendSentence) {
                recognitionState.sendSentence();
              } else if (onSendMessage) {
                onSendMessage(sentenceBuffer, 'human');
              }
            }}
            inactivityCountdown={recognitionState?.inactivityCountdown}
            autoSendEnabled={recognitionState?.autoSendEnabled ?? true}
            onToggleAutoSend={() => recognitionState?.setAutoSendEnabled?.(prev => !prev)}
            onCancelCountdown={recognitionState?.cancelInactivityCountdown}
          />
        </div>
      </div>
    </div>
  );
});
