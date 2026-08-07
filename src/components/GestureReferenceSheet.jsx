import React, { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Search,
  X,
  BookOpen,
  Star,
  Play,
  RotateCcw,
  Sparkles,
  Info,
  CheckCircle2,
  Video,
  ExternalLink
} from 'lucide-react';
import {
  ALPHABET_GESTURES,
  WORD_GESTURES,
  SENTENCE_GESTURES,
  ISL_YOUTUBE_PLAYLIST_URL
} from '../data/gestureData';

/**
 * GestureReferenceSheet Component
 * Two-Handed Indian Sign Language (ISL) Reference Sheet & Interactive Guide.
 */
export const GestureReferenceSheet = ({
  isOpen,
  onClose,
  onSimulateGesture,
  isEmbedded = false
}) => {
  const [activeTab, setActiveTab] = useState('alphabet'); // 'alphabet' | 'words' | 'sentences' | 'favorites'
  const [searchQuery, setSearchQuery] = useState('');
  const [favorites, setFavorites] = useState(['alpha-A', 'word-hello', 'sent-howareyou']);
  const [selectedGesture, setSelectedGesture] = useState(null);
  const [isPlayingDemo, setIsPlayingDemo] = useState(false);

  // Toggle favorite helper
  const toggleFavorite = (id, e) => {
    if (e) e.stopPropagation();
    setFavorites(prev =>
      prev.includes(id) ? prev.filter(fId => fId !== id) : [...prev, id]
    );
  };

  // Filtered lists based on search and active tab
  const filteredAlphabet = useMemo(() => {
    return ALPHABET_GESTURES.filter(item =>
      item.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      item.letter.toLowerCase() === searchQuery.trim().toLowerCase()
    );
  }, [searchQuery]);

  const filteredWords = useMemo(() => {
    return WORD_GESTURES.filter(item =>
      item.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      item.category.toLowerCase().includes(searchQuery.toLowerCase())
    );
  }, [searchQuery]);

  const filteredSentences = useMemo(() => {
    return SENTENCE_GESTURES.filter(item =>
      item.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      item.sequence.some(seq => seq.toLowerCase().includes(searchQuery.toLowerCase()))
    );
  }, [searchQuery]);

  const favoriteItems = useMemo(() => {
    const all = [...ALPHABET_GESTURES, ...WORD_GESTURES, ...SENTENCE_GESTURES];
    return all.filter(item => favorites.includes(item.id));
  }, [favorites]);

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <div className={`sheet-backdrop ${isEmbedded ? 'in-panel-backdrop' : ''}`} onClick={onClose}>
        <motion.div
          initial={{ y: '100%' }}
          animate={{ y: 0 }}
          exit={{ y: '100%' }}
          transition={{ type: 'spring', damping: 28, stiffness: 280 }}
          className={`bottom-sheet-container ${isEmbedded ? 'in-panel-sheet' : ''}`}
          onClick={e => e.stopPropagation()}
        >
          {/* Drag Handle Bar */}
          <div className="sheet-drag-bar-wrapper">
            <div className="sheet-drag-handle" />
          </div>

          {/* Sheet Header Strip */}
          <div className="sheet-header">
            <div className="sheet-title-group">
              <div className="sheet-icon-box">
                <BookOpen size={18} style={{ color: '#6E7F6B' }} />
              </div>
              <div>
                <h3 className="sheet-title">Indian Sign Language (ISL) Guide</h3>
                <p className="sheet-subtitle">Authentic 2-Handed ISL Alphabets, Words & Sentences</p>
              </div>
            </div>
            <div className="sheet-header-actions">
              <a
                href={ISL_YOUTUBE_PLAYLIST_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="watch-isl-youtube-btn"
                title="Watch Official ISL YouTube Video Lessons"
              >
                <Video size={14} style={{ color: '#ef4444' }} />
                <span>ISL Videos</span>
                <ExternalLink size={11} />
              </a>
              <button className="sheet-close-btn" onClick={onClose} title="Close Panel">
                <X size={18} />
              </button>
            </div>
          </div>

          {/* Search Bar */}
          <div className="sheet-search-wrapper">
            <Search size={16} className="sheet-search-icon" />
            <input
              type="text"
              placeholder="Search ISL letters A-Z, words, phrases..."
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              className="sheet-search-input"
            />
            {searchQuery && (
              <button className="sheet-clear-search" onClick={() => setSearchQuery('')}>
                <X size={14} />
              </button>
            )}
          </div>

          {/* Navigation Tabs */}
          <div className="sheet-tabs-bar">
            <button
              className={`sheet-tab-btn ${activeTab === 'alphabet' ? 'is-active' : ''}`}
              onClick={() => setActiveTab('alphabet')}
            >
              <span>Alphabet (2-Hand ISL)</span>
              <span className="tab-badge">{ALPHABET_GESTURES.length}</span>
            </button>
            <button
              className={`sheet-tab-btn ${activeTab === 'words' ? 'is-active' : ''}`}
              onClick={() => setActiveTab('words')}
            >
              <span>Words</span>
              <span className="tab-badge">{WORD_GESTURES.length}</span>
            </button>
            <button
              className={`sheet-tab-btn ${activeTab === 'sentences' ? 'is-active' : ''}`}
              onClick={() => setActiveTab('sentences')}
            >
              <span>Sentences</span>
              <span className="tab-badge">{SENTENCE_GESTURES.length}</span>
            </button>
            <button
              className={`sheet-tab-btn ${activeTab === 'favorites' ? 'is-active' : ''}`}
              onClick={() => setActiveTab('favorites')}
            >
              <Star size={13} fill={activeTab === 'favorites' ? '#6E7F6B' : 'transparent'} />
              <span>Favorites</span>
              <span className="tab-badge">{favorites.length}</span>
            </button>
          </div>

          {/* Sheet Body Content Container */}
          <div className="sheet-content-scroll">
            {/* 1. ALPHABET TAB */}
            {activeTab === 'alphabet' && (
              <div className="alphabet-grid">
                {filteredAlphabet.map(item => {
                  const isFav = favorites.includes(item.id);
                  return (
                    <div
                      key={item.id}
                      className="alphabet-card"
                      onClick={() => setSelectedGesture(item)}
                    >
                      <button
                        className={`card-fav-btn ${isFav ? 'is-fav' : ''}`}
                        onClick={e => toggleFavorite(item.id, e)}
                        title="Add to Favorites"
                      >
                        <Star size={13} fill={isFav ? '#C8AD93' : 'transparent'} />
                      </button>
                      <div className="alphabet-char">{item.letter}</div>
                      <div className="alphabet-symbol">{item.symbol}</div>
                      {item.lh && item.rh && (
                        <div className="isl-2hand-breakdown">
                          <span className="hand-tag-chip lh-tag">LH: {item.lh.split(' ')[0]}</span>
                          <span className="hand-tag-chip rh-tag">RH: {item.rh.split(' ')[0]}</span>
                        </div>
                      )}
                      <div className="alphabet-label">{item.title}</div>
                    </div>
                  );
                })}
              </div>
            )}

            {/* 2. WORDS TAB */}
            {activeTab === 'words' && (
              <div className="words-grid">
                {filteredWords.map(item => {
                  const isFav = favorites.includes(item.id);
                  return (
                    <div
                      key={item.id}
                      className="word-card"
                      onClick={() => setSelectedGesture(item)}
                    >
                      <div className="word-card-top">
                        <span className="category-pill">{item.category} • 👐 2-Hand</span>
                        <button
                          className={`card-fav-btn ${isFav ? 'is-fav' : ''}`}
                          onClick={e => toggleFavorite(item.id, e)}
                        >
                          <Star size={14} fill={isFav ? '#C8AD93' : 'transparent'} />
                        </button>
                      </div>
                      <div className="word-card-icon">{item.icon}</div>
                      <h4 className="word-card-title">{item.title}</h4>
                      <p className="word-card-desc">{item.description}</p>
                      {onSimulateGesture && (
                        <button
                          className="simulate-gesture-btn"
                          onClick={(e) => {
                            e.stopPropagation();
                            onSimulateGesture(item.title);
                            onClose();
                          }}
                        >
                          <Sparkles size={13} />
                          <span>Simulate Translation</span>
                        </button>
                      )}
                    </div>
                  );
                })}
              </div>
            )}

            {/* 3. SENTENCES TAB */}
            {activeTab === 'sentences' && (
              <div className="sentences-list">
                {filteredSentences.map(item => {
                  const isFav = favorites.includes(item.id);
                  return (
                    <div
                      key={item.id}
                      className="sentence-card"
                      onClick={() => setSelectedGesture(item)}
                    >
                      <div className="sentence-card-header">
                        <div className="sentence-title-row">
                          <span className="sentence-icon">{item.icon}</span>
                          <h4 className="sentence-title">{item.title}</h4>
                        </div>
                        <button
                          className={`card-fav-btn ${isFav ? 'is-fav' : ''}`}
                          onClick={e => toggleFavorite(item.id, e)}
                        >
                          <Star size={14} fill={isFav ? '#C8AD93' : 'transparent'} />
                        </button>
                      </div>
                      <div className="sequence-chips-row">
                        {item.sequence.map((seq, idx) => (
                          <span key={idx} className="sequence-chip">
                            {seq}
                          </span>
                        ))}
                      </div>
                      <p className="sentence-desc">{item.description}</p>
                    </div>
                  );
                })}
              </div>
            )}

            {/* 4. FAVORITES TAB */}
            {activeTab === 'favorites' && (
              <div className="words-grid">
                {favoriteItems.length === 0 ? (
                  <div className="empty-favorites-box">
                    <Star size={32} style={{ color: '#C8AD93' }} />
                    <h4>No favorites saved yet</h4>
                    <p>Click the star icon on any letter, word, or phrase to save it here for quick access.</p>
                  </div>
                ) : (
                  favoriteItems.map(item => (
                    <div
                      key={item.id}
                      className="word-card"
                      onClick={() => setSelectedGesture(item)}
                    >
                      <div className="word-card-top">
                        <span className="category-pill">{item.type.toUpperCase()} • 👐 2-Hand</span>
                        <button
                          className="card-fav-btn is-fav"
                          onClick={e => toggleFavorite(item.id, e)}
                        >
                          <Star size={14} fill="#C8AD93" />
                        </button>
                      </div>
                      <div className="word-card-icon">{item.symbol || item.icon || '🤲'}</div>
                      <h4 className="word-card-title">{item.title}</h4>
                      <p className="word-card-desc">{item.description}</p>
                    </div>
                  ))
                )}
              </div>
            )}
          </div>
        </motion.div>
      </div>

      {/* ENLARGED GESTURE DETAIL MODAL */}
      {selectedGesture && (
        <div className="gesture-modal-backdrop" onClick={() => setSelectedGesture(null)}>
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            className="gesture-modal-box"
            onClick={e => e.stopPropagation()}
          >
            <div className="modal-header">
              <div className="modal-title-row">
                <span className="modal-icon">{selectedGesture.icon || selectedGesture.symbol || '🤲'}</span>
                <div>
                  <h3>{selectedGesture.title}</h3>
                  <span className="modal-isl-tag">👐 2-Handed Indian Sign Language (ISL)</span>
                </div>
              </div>
              <button className="modal-close-btn" onClick={() => setSelectedGesture(null)}>
                <X size={18} />
              </button>
            </div>

            <div className="modal-visualizer-area">
              <div className={`gesture-visual-display isl-2hand-visual ${isPlayingDemo ? 'is-animating' : ''}`}>
                <div className="visual-char">{selectedGesture.symbol || selectedGesture.letter || selectedGesture.title}</div>
                <div className="visual-hand-landmarks dual-hands">
                  {/* Left Hand Landmark Cluster */}
                  <div className="hand-cluster left-hand">
                    <span className="hand-tag">LH</span>
                    <div className="landmark-node node-1" />
                    <div className="landmark-node node-2" />
                  </div>
                  {/* Interaction Connection Line */}
                  <div className="landmark-line dual-connection" />
                  {/* Right Hand Landmark Cluster */}
                  <div className="hand-cluster right-hand">
                    <span className="hand-tag">RH</span>
                    <div className="landmark-node node-3" />
                    <div className="landmark-node node-4" />
                  </div>
                </div>
              </div>
              <button
                className="play-demo-btn"
                onClick={() => {
                  setIsPlayingDemo(true);
                  setTimeout(() => setIsPlayingDemo(false), 2000);
                }}
              >
                {isPlayingDemo ? <RotateCcw size={14} className="spin" /> : <Play size={14} />}
                <span>{isPlayingDemo ? 'Playing 2-Hand Motion Demo...' : 'Play ISL Motion Animation'}</span>
              </button>
            </div>

            <div className="modal-body">
              {selectedGesture.lh && selectedGesture.rh && (
                <div className="modal-hand-spec-box">
                  <div className="spec-pill lh-spec">
                    <span className="spec-label">LEFT HAND (LH):</span>
                    <span className="spec-val">{selectedGesture.lh}</span>
                  </div>
                  <div className="spec-pill rh-spec">
                    <span className="spec-label">RIGHT HAND (RH):</span>
                    <span className="spec-val">{selectedGesture.rh}</span>
                  </div>
                </div>
              )}

              <h5 className="section-label">HOW TO PERFORM THIS 2-HANDED GESTURE (ISL):</h5>
              <p className="modal-description">{selectedGesture.description}</p>

              {selectedGesture.steps && (
                <div className="modal-steps-list">
                  {selectedGesture.steps.map((step, idx) => (
                    <div key={idx} className="step-item">
                      <CheckCircle2 size={16} className="step-icon" />
                      <span>{step}</span>
                    </div>
                  ))}
                </div>
              )}

              {selectedGesture.breakdown && (
                <div className="breakdown-box">
                  <Info size={14} />
                  <span>{selectedGesture.breakdown}</span>
                </div>
              )}
            </div>

            <div className="modal-footer modal-footer-dual">
              <a
                href={ISL_YOUTUBE_PLAYLIST_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="modal-action-btn youtube-action"
              >
                <Video size={15} style={{ color: '#ef4444' }} />
                <span>Watch ISL Video Tutorial</span>
                <ExternalLink size={12} />
              </a>

              {onSimulateGesture && (
                <button
                  className="modal-action-btn primary-action"
                  onClick={() => {
                    onSimulateGesture(selectedGesture.title);
                    setSelectedGesture(null);
                    onClose();
                  }}
                >
                  <Sparkles size={15} />
                  <span>Simulate in Live Feed</span>
                </button>
              )}
            </div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
};
