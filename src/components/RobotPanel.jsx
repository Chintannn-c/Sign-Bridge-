import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Trash2, Bot, User, ChevronDown, Keyboard, Send, X, BookOpen, Eye, EyeOff } from 'lucide-react';
import { GestureReferenceSheet } from './GestureReferenceSheet';

/**
 * RobotPanel Component - Right Panel Card ("SIGN-BRIDGE")
 * Gesture Detection Chat Thread with embedded in-card Gesture Reference Guide split view.
 */
export const RobotPanel = React.memo(({
  messages = [],
  fullText = '',
  streamingText = '',
  isActive = false,
  isStreaming = false,
  onSendMessage,
  onClear
}) => {
  const [inputVal, setInputVal] = useState('');
  const [showKeyboard, setShowKeyboard] = useState(false);
  const [isSheetOpen, setIsSheetOpen] = useState(false);
  const [showActions, setShowActions] = useState(true);
  const [showScrollBottom, setShowScrollBottom] = useState(false);
  const scrollContainerRef = useRef(null);
  const chatEndRef = useRef(null);

  // Calculate active word being fingerspelled
  const safeFullText = fullText || '';
  const safeStreamingText = streamingText || '';
  const words = safeFullText ? safeFullText.split(' ') : [];
  const currentWordIndex = Math.min(
    Math.floor((safeStreamingText.length / (safeFullText.length || 1)) * words.length),
    words.length - 1
  );
  const currentWord = words[currentWordIndex] || words[0] || '';

  // Auto-scroll chat thread to bottom on new messages
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamingText]);

  // Monitor scroll position to show/hide "Scroll to Bottom" button
  const handleScroll = () => {
    if (!scrollContainerRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = scrollContainerRef.current;
    const isAtBottom = scrollHeight - scrollTop - clientHeight < 50;
    setShowScrollBottom(!isAtBottom);
  };

  const scrollToBottom = (e) => {
    if (e) e.stopPropagation();
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSend = (e) => {
    if (e) e.preventDefault();
    if (inputVal.trim() && onSendMessage) {
      onSendMessage(inputVal.trim());
      setInputVal('');
      setShowKeyboard(false);
    }
  };

  return (
    <div
      className={`panel-card right-card ${isActive ? 'active-panel' : ''} ${isSheetOpen ? 'has-split-sheet' : ''}`}
    >
      {/* Card Header Strip */}
      <div className="card-header">
        <div className="card-label robot-label">
          <Bot size={16} style={{ color: '#6E7F6B' }} />
          <span>SIGN-BRIDGE CHAT UI</span>
          {isActive && <span className="listening-dot" />}
        </div>
        <div className="chat-header-actions" onClick={e => e.stopPropagation()}>
          <AnimatePresence mode="wait">
            {showActions && (
              <motion.div
                key="header-action-buttons"
                initial={{ opacity: 0, x: 10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 10 }}
                transition={{ duration: 0.15 }}
                style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}
              >
                <button
                  className={`chat-action-btn ${isSheetOpen ? 'is-active-toggle' : ''}`}
                  onClick={() => setIsSheetOpen(prev => !prev)}
                  title={isSheetOpen ? 'Hide Gesture Guide' : 'Learn Gestures'}
                >
                  <BookOpen size={15} />
                </button>
                <button
                  className={`chat-action-btn ${showKeyboard ? 'is-active-toggle' : ''}`}
                  onClick={() => setShowKeyboard(prev => !prev)}
                  title={showKeyboard ? 'Hide Input' : 'Type Message'}
                >
                  <Keyboard size={15} />
                </button>
                {onClear && (
                  <button
                    className="chat-action-btn"
                    onClick={onClear}
                    title="Clear Chat History"
                  >
                    <Trash2 size={15} />
                  </button>
                )}
              </motion.div>
            )}
          </AnimatePresence>

          {/* Single Toggle Button to Hide / Unhide All Action Buttons */}
          <button
            className={`chat-action-btn toggle-actions-btn ${!showActions ? 'is-active-toggle' : ''}`}
            onClick={() => setShowActions(prev => !prev)}
            title={showActions ? "Hide action buttons" : "Unhide action buttons"}
          >
            {showActions ? <EyeOff size={15} /> : <Eye size={15} />}
          </button>
        </div>

      </div>


      {/* Chat UI Thread Container */}
      <div className={`chat-thread-container ${isSheetOpen ? 'is-split-thread' : ''}`}>
        {/* Scrollable Conversation Messages List */}
        <div
          className="chat-messages-scroll"
          ref={scrollContainerRef}
          onScroll={handleScroll}
          onClick={e => e.stopPropagation()}
        >
          {messages.length === 0 && !isStreaming && (
            <motion.div 
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.35, ease: 'easeOut' }}
              style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                height: '100%',
                minHeight: '260px',
                padding: '2rem 1.5rem',
                textAlign: 'center',
                userSelect: 'none'
              }}
            >
              {/* Glowing Icon Container */}
              <div 
                style={{
                  width: '58px',
                  height: '58px',
                  borderRadius: '18px',
                  background: 'linear-gradient(135deg, rgba(110, 127, 107, 0.15), rgba(13, 148, 136, 0.15))',
                  border: '1px solid rgba(110, 127, 107, 0.25)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  marginBottom: '1rem',
                  boxShadow: '0 10px 28px -6px rgba(110, 127, 107, 0.25)',
                  color: '#556b52'
                }}
              >
                <Bot size={28} strokeWidth={2.2} />
              </div>

              {/* Main Headline */}
              <h3 
                style={{ 
                  fontSize: '1.2rem', 
                  fontWeight: 700, 
                  color: '#2D2A26', 
                  marginBottom: '0.45rem',
                  letterSpacing: '-0.02em',
                  fontFamily: 'var(--font-family)'
                }}
              >
                Ready for Live Dialogue
              </h3>

              {/* Subtitle with High-Contrast Typography */}
              <p 
                style={{ 
                  fontSize: '0.84rem', 
                  color: '#656059', 
                  lineHeight: 1.55,
                  maxWidth: '310px',
                  marginBottom: '1.4rem',
                  fontWeight: 400
                }}
              >
                Perform <strong style={{ color: '#475845', fontWeight: 600 }}>camera signs</strong> on the left, or pick a sample prompt to start:
              </p>

              {/* Quick-Prompt Suggestion Chips */}
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.45rem', justifyContent: 'center', maxWidth: '340px' }}>
                {[
                  { text: 'Namaste! How can I help you?', label: '🙏 Namaste' },
                  { text: 'How are you?', label: '🤝 How are you?' },
                  { text: 'Where is the washroom?', label: '🚻 Washroom' },
                ].map((chip, idx) => (
                  <button
                    key={idx}
                    onClick={() => onSendMessage && onSendMessage(chip.text, 'human')}
                    style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '0.35rem',
                      padding: '0.45rem 0.85rem',
                      fontSize: '0.78rem',
                      fontWeight: 600,
                      color: '#38332e',
                      background: 'rgba(255, 255, 255, 0.85)',
                      border: '1px solid rgba(200, 173, 147, 0.35)',
                      borderRadius: '20px',
                      cursor: 'pointer',
                      backdropFilter: 'blur(10px)',
                      boxShadow: '0 2px 8px rgba(45, 42, 38, 0.04)',
                      transition: 'all 0.2s cubic-bezier(0.16, 1, 0.3, 1)'
                    }}
                    onMouseEnter={e => {
                      e.currentTarget.style.transform = 'translateY(-2px)';
                      e.currentTarget.style.borderColor = 'rgba(110, 127, 107, 0.6)';
                      e.currentTarget.style.color = '#475845';
                      e.currentTarget.style.boxShadow = '0 6px 14px rgba(110, 127, 107, 0.15)';
                    }}
                    onMouseLeave={e => {
                      e.currentTarget.style.transform = 'translateY(0)';
                      e.currentTarget.style.borderColor = 'rgba(200, 173, 147, 0.35)';
                      e.currentTarget.style.color = '#38332e';
                      e.currentTarget.style.boxShadow = '0 2px 8px rgba(45, 42, 38, 0.04)';
                    }}
                  >
                    {chip.label}
                  </button>
                ))}
              </div>
            </motion.div>
          )}

          {messages.map((msg) => {
            const isHuman = msg.sender === 'human';
            return (
              <div
                key={msg.id}
                className={`chat-bubble-row ${isHuman ? 'row-human' : 'row-robot'}`}
              >
                <div className="chat-avatar">
                  {isHuman ? <User size={14} /> : <Bot size={14} />}
                </div>
                <div className="chat-bubble">
                  <div className="chat-sender-name">
                    {isHuman ? 'YOU (Gesture Detection)' : 'SIGN-BRIDGE AI'}
                  </div>
                  <div className="chat-bubble-text">{msg.text}</div>
                  <div className="chat-timestamp">{msg.timestamp}</div>
                </div>
              </div>
            );
          })}

          {/* Active Live Streaming Typewriter Message */}
          {isStreaming && (
            <div className={`chat-bubble-row ${isActive ? 'row-robot' : 'row-human'} is-streaming-row`}>
              <div className="chat-avatar">
                {isActive ? <Bot size={14} /> : <User size={14} />}
              </div>
              <div className="chat-bubble streaming-bubble">
                <div className="chat-sender-name">
                  {isActive ? 'SIGN-BRIDGE AI' : 'YOU (Gesture Detection)'}
                  <span className="live-typing-pill">Translating...</span>
                </div>
                <div className="chat-bubble-text">
                  {streamingText || fullText}
                  <span className="streaming-cursor-teal" />
                </div>
              </div>
            </div>
          )}

          <div ref={chatEndRef} />
        </div>

        {/* Floating Scroll to Bottom Button */}
        {showScrollBottom && (
          <button
            className="scroll-bottom-btn"
            onClick={scrollToBottom}
            title="Scroll to latest message"
          >
            <ChevronDown size={14} />
            <span>Latest</span>
          </button>
        )}

        {/* Animated Fingerspelling Tile Row for Active Word */}
        {currentWord && isStreaming && (
          <div className="letter-tiles-wrapper" onClick={e => e.stopPropagation()}>
            <div className="tiles-label">FINGERSPELLING LANDMARKS:</div>
            <div className="letter-tiles-row">
              <AnimatePresence mode="wait">
                {currentWord.split('').map((char, idx) => {
                  const isCurrentTile =
                    idx === streamingText.length % (currentWord.length || 1);
                  return (
                    <motion.div
                      key={`${currentWord}-${char}-${idx}`}
                      initial={{ opacity: 0, scale: 0.8 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ duration: 0.15, delay: idx * 0.02 }}
                      className={`letter-tile ${isCurrentTile ? 'active-letter' : ''}`}
                    >
                      {char.toUpperCase()}
                    </motion.div>
                  );
                })}
              </AnimatePresence>
            </div>
          </div>
        )}

        {/* Collapsible Secondary Keyboard Input Bar */}
        <AnimatePresence>
          {showKeyboard && (
            <motion.div
              initial={{ opacity: 0, y: 12, height: 0 }}
              animate={{ opacity: 1, y: 0, height: 'auto' }}
              exit={{ opacity: 0, y: 12, height: 0 }}
              className="secondary-input-wrapper"
              onClick={e => e.stopPropagation()}
            >
              <div className="secondary-input-header">
                <span className="secondary-tag">
                  <Keyboard size={11} /> SECONDARY INPUT
                </span>
                <button
                  className="hide-keyboard-close-btn"
                  onClick={() => setShowKeyboard(false)}
                >
                  <X size={12} />
                  <span>Hide</span>
                </button>
              </div>
              <form className="chat-input-bar secondary-input-bar" onSubmit={handleSend}>
                <input
                  type="text"
                  placeholder="Type message response here..."
                  value={inputVal}
                  onChange={e => setInputVal(e.target.value)}
                  className="chat-input"
                  autoFocus
                />
                <button type="submit" className="chat-send-btn secondary-send-btn" title="Send text message">
                  <Send size={13} />
                  <span>Send</span>
                </button>
              </form>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Embedded In-Card Gesture Reference Panel (Splits Right Panel into top/bottom half) */}
      <GestureReferenceSheet
        isOpen={isSheetOpen}
        onClose={() => setIsSheetOpen(false)}
        onSimulateGesture={onSendMessage}
        isEmbedded={true}
      />
    </div>
  );
});

