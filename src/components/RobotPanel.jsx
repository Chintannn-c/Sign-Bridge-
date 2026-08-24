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
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.25 }}
              style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                height: '100%',
                minHeight: '220px',
                padding: '2rem 1rem',
                textAlign: 'center',
                userSelect: 'none',
                color: 'var(--text-espresso)'
              }}
            >
              {/* Native Rounded Avatar matching Chat UI */}
              <div 
                style={{
                  width: '42px',
                  height: '42px',
                  borderRadius: '50%',
                  background: 'var(--bg-secondary, #FAF8F5)',
                  border: '1.5px solid var(--accent-camel, #C8AD93)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  marginBottom: '0.85rem',
                  color: 'var(--accent-sage, #6E7F6B)',
                  boxShadow: '0 2px 8px rgba(45, 42, 38, 0.04)'
                }}
              >
                <Bot size={22} />
              </div>

              {/* Native Header */}
              <div 
                style={{ 
                  fontSize: '1rem', 
                  fontWeight: 600, 
                  color: 'var(--text-espresso)', 
                  marginBottom: '0.25rem'
                }}
              >
                Ready for Live Dialogue
              </div>

              {/* Subtitle */}
              <div 
                style={{ 
                  fontSize: '0.82rem', 
                  color: 'var(--text-muted, #6A6A67)', 
                  lineHeight: 1.5,
                  maxWidth: '290px',
                  marginBottom: '1.1rem'
                }}
              >
                Sign with your camera or select a sample prompt to start:
              </div>

              {/* Native Suggestion Chips matching app action buttons */}
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem', justifyContent: 'center', maxWidth: '320px' }}>
                {[
                  { text: 'Namaste! How can I help you?', label: 'Namaste' },
                  { text: 'How are you?', label: 'How are you?' },
                  { text: 'Where is the washroom?', label: 'Where is the washroom?' },
                ].map((chip, idx) => (
                  <button
                    key={idx}
                    onClick={() => onSendMessage && onSendMessage(chip.text, 'human')}
                    style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      padding: '0.35rem 0.75rem',
                      fontSize: '0.76rem',
                      fontWeight: 500,
                      color: 'var(--text-espresso)',
                      background: 'var(--ui-white, #FFFFFF)',
                      border: '1px solid var(--accent-camel, #C8AD93)',
                      borderRadius: '9999px',
                      cursor: 'pointer',
                      transition: 'all 0.15s ease',
                      boxShadow: '0 1px 3px rgba(45, 42, 38, 0.03)'
                    }}
                    onMouseEnter={e => {
                      e.currentTarget.style.background = 'var(--accent-sage, #6E7F6B)';
                      e.currentTarget.style.color = 'var(--ui-white, #FFFFFF)';
                      e.currentTarget.style.borderColor = 'var(--accent-sage, #6E7F6B)';
                    }}
                    onMouseLeave={e => {
                      e.currentTarget.style.background = 'var(--ui-white, #FFFFFF)';
                      e.currentTarget.style.color = 'var(--text-espresso)';
                      e.currentTarget.style.borderColor = 'var(--accent-camel, #C8AD93)';
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

