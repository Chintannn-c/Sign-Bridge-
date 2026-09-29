import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Keyboard, Send, X, Trash2, Sparkles } from 'lucide-react';
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
  const isAutoSending = Boolean(recognitionState?.isAutoSending);
  const inactivityCountdown = recognitionState?.inactivityCountdown;
  const detectedSign = recognitionState?.detectedSign || recognitionState?.detectedLetter || recognitionState?.detectedWord;
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
          onSendMessage={onSendMessage}
        />

        {/* Text Feed below the Horizontal Webcam */}
        <div className="card-text-wrapper human-text-wrapper" style={{ flexDirection: 'column', alignItems: 'flex-start', justifyContent: 'center' }}>
          
          {/* Animated Text Processing Header Bar (appears after letters appear in section) */}
          <AnimatePresence>
            {hasLiveSentence && !showTextbox && (
              <motion.div
                initial={{ opacity: 0, y: -6, height: 0 }}
                animate={{ opacity: 1, y: 0, height: 'auto' }}
                exit={{ opacity: 0, y: -6, height: 0 }}
                transition={{ duration: 0.22 }}
                className="text-processing-header-bar"
              >
                <div className={`live-processing-pill ${isAutoSending ? 'is-refining' : ''}`}>
                  <div className="neural-soundwave">
                    <span className="wave-bar bar-1" />
                    <span className="wave-bar bar-2" />
                    <span className="wave-bar bar-3" />
                    <span className="wave-bar bar-4" />
                  </div>
                  <span className="live-processing-text">
                    {isAutoSending ? (
                      <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem' }}>
                        <Sparkles size={13} className="animate-spin text-purple-600" />
                        AI Refining Spoken Sentence...
                      </span>
                    ) : (
                      inactivityCountdown ? (
                        <span>Auto-sending in <strong>{inactivityCountdown}s</strong>...</span>
                      ) : (
                        <span>Live AI Text Processing...</span>
                      )
                    )}
                  </span>
                </div>

                <div className="live-processing-meta">
                  {detectedSign && (
                    <span className="live-detected-tag">
                      <span className="live-pulse-dot" />
                      Sign: <strong>{detectedSign}</strong>
                    </span>
                  )}
                  <span className="live-char-counter">
                    {sentenceBuffer.length} {sentenceBuffer.length === 1 ? 'char' : 'chars'}
                  </span>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {!showTextbox && (
            <div style={{ width: '100%', position: 'relative', display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '0.75rem' }}>
              <h2 className={`card-text ${isCameraOn && !hasLiveSentence ? 'text-slate-400 italic' : ''}`} style={{ flex: 1, margin: 0 }}>
                {hasLiveSentence ? (
                  <>
                    <AnimatePresence mode="popLayout">
                      {sentenceBuffer.split('').map((char, index) => (
                        <motion.span
                          key={`char-${index}-${char}`}
                          initial={{ opacity: 0, y: 8, scale: 0.8 }}
                          animate={{ opacity: 1, y: 0, scale: 1 }}
                          exit={{ opacity: 0, scale: 0.8 }}
                          transition={{ duration: 0.18, ease: 'easeOut' }}
                          className={`live-letter-char ${index === sentenceBuffer.length - 1 ? 'latest-letter' : ''}`}
                          style={{ display: 'inline-block', whiteSpace: char === ' ' ? 'pre' : 'normal' }}
                        >
                          {char === ' ' ? '\u00A0' : char}
                        </motion.span>
                      ))}
                    </AnimatePresence>
                    <span className="active-live-cursor" />
                  </>
                ) : (
                  <>
                    {displayText}
                    {isStreaming && !isCameraOn && <span className="streaming-cursor-teal" />}
                    {isCameraOn && recognitionState?.status === 'detecting' && <span className="streaming-cursor-teal" />}
                  </>
                )}
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

          {/* Flowing Animated Shimmer Underline (when letters appear in section) */}
          <AnimatePresence>
            {hasLiveSentence && !showTextbox && (
              <motion.div
                initial={{ opacity: 0, scaleX: 0 }}
                animate={{ opacity: 1, scaleX: 1 }}
                exit={{ opacity: 0, scaleX: 0 }}
                transition={{ duration: 0.25 }}
                className="processing-shimmer-track"
              >
                <div className={`processing-shimmer-glow ${isAutoSending ? 'is-refining' : ''}`} />
              </motion.div>
            )}
          </AnimatePresence>

          {/* Live Detected Sign Token Badges Row (when letters appear in section) */}
          <AnimatePresence>
            {hasLiveSentence && !showTextbox && (
              <motion.div
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 4 }}
                transition={{ duration: 0.2 }}
                className="live-sign-tokens-container"
              >
                <div className="live-sign-tokens-list">
                  {sentenceBuffer.split(' ').filter(Boolean).map((word, wIdx, arr) => (
                    <motion.div 
                      key={`token-${wIdx}-${word}`}
                      initial={{ scale: 0.85, opacity: 0 }}
                      animate={{ scale: 1, opacity: 1 }}
                      className={`live-sign-token-chip ${wIdx === arr.length - 1 ? 'is-active-token' : ''}`}
                    >
                      <span className="token-text">{word}</span>
                    </motion.div>
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>

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
                    background: '#FFFFFF',
                    border: '1.5px solid rgba(200, 173, 147, 0.5)',
                    borderRadius: '9999px',
                    padding: '0.35rem 0.45rem 0.35rem 1.25rem',
                    boxShadow: '0 4px 16px -2px rgba(45, 42, 38, 0.08), 0 2px 6px rgba(0, 0, 0, 0.02)',
                    width: '100%'
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
                      fontSize: '0.88rem',
                      color: 'var(--text-espresso)',
                      fontFamily: 'var(--font-family)',
                      fontWeight: 450
                    }}
                  />
                  <button
                    type="submit"
                    title="Send message"
                    disabled={!tempInput.trim()}
                    style={{
                      width: '36px',
                      height: '36px',
                      borderRadius: '50%',
                      background: tempInput.trim() 
                        ? 'linear-gradient(135deg, #818cf8, #6366f1)' 
                        : 'linear-gradient(135deg, #a5b4fc, #818cf8)',
                      color: '#fff',
                      border: 'none',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      cursor: tempInput.trim() ? 'pointer' : 'default',
                      opacity: tempInput.trim() ? 1 : 0.85,
                      boxShadow: '0 4px 12px rgba(99, 102, 241, 0.35)',
                      transition: 'transform 0.15s ease'
                    }}
                    onMouseEnter={e => {
                      if (tempInput.trim()) e.currentTarget.style.transform = 'scale(1.06)';
                    }}
                    onMouseLeave={e => {
                      e.currentTarget.style.transform = 'scale(1)';
                    }}
                  >
                    <Send size={15} style={{ transform: 'translateX(1px)' }} />
                  </button>
                  <button
                    type="button"
                    onClick={() => setShowTextbox(false)}
                    title="Close"
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      width: '28px',
                      height: '28px',
                      borderRadius: '50%',
                      background: 'transparent',
                      color: 'var(--text-muted)',
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
            <div style={{ marginTop: '1rem', width: '100%', opacity: (hasLiveSentence || recognitionState.inactivityCountdown || recognitionState.lastAutoSpoken) ? 1 : 0.5, transition: 'opacity 0.2s' }} onClick={e => e.stopPropagation()}>
              <SentenceBuilder 
                sentence={sentenceBuffer}
                onUndo={recognitionState.undoLetter}
                onDelete={recognitionState.deleteLetter}
                onClear={recognitionState.clearBuffer}
                onAddSpace={recognitionState.addSpace}
                onAIRefine={recognitionState.refineSentence}
                onSend={() => recognitionState.sendSentence ? recognitionState.sendSentence() : (onSendMessage && onSendMessage(sentenceBuffer, 'human'))}
                inactivityCountdown={recognitionState.inactivityCountdown}
                autoSendEnabled={recognitionState.autoSendEnabled}
                onToggleAutoSend={() => recognitionState.setAutoSendEnabled(prev => !prev)}
                lastAutoSpoken={recognitionState.lastAutoSpoken}
                onCancelCountdown={recognitionState.cancelInactivityCountdown}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
});
