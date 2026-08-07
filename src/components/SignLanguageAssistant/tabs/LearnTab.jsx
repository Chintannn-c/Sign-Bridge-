import React, { useState } from 'react';
import { ChevronRight, Play, BookOpen, Star, Sparkles } from 'lucide-react';
import { ALPHABET_GESTURES, WORD_GESTURES } from '../../../data/gestureData';

export const LearnTab = () => {
  const [activeCategory, setActiveCategory] = useState('alphabet');
  const [expandedId, setExpandedId] = useState(null);

  const data = activeCategory === 'alphabet' ? ALPHABET_GESTURES : WORD_GESTURES;

  return (
    <div className="sla-tab-pane">
      {/* Category Toggles */}
      <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1rem' }}>
        <button 
          className={`sla-action-btn ${activeCategory === 'alphabet' ? 'primary' : ''}`}
          onClick={() => { setActiveCategory('alphabet'); setExpandedId(null); }}
        >
          Alphabet (A-Z)
        </button>
        <button 
          className={`sla-action-btn ${activeCategory === 'words' ? 'primary' : ''}`}
          onClick={() => { setActiveCategory('words'); setExpandedId(null); }}
        >
          Common Words
        </button>
      </div>

      <div className="sla-learn-section-title">
        <BookOpen size={14} className="sla-title-icon" />
        {activeCategory === 'alphabet' ? 'Learn ISL Alphabet' : 'Learn ISL Words'}
      </div>

      <div className="sla-learn-grid">
        {data.slice(0, 26).map(item => {
          const isExpanded = expandedId === item.id;
          return (
            <React.Fragment key={item.id}>
              <div 
                className={`sla-learn-card ${isExpanded ? 'active' : ''}`}
                onClick={() => setExpandedId(isExpanded ? null : item.id)}
              >
                <div className="sla-learn-letter">{item.letter || item.title[0]}</div>
                <div className="sla-learn-label">{item.title}</div>
              </div>

              {/* Expanded Detail View */}
              {isExpanded && (
                <div className="sla-learn-detail" style={{ gridColumn: '1 / -1' }}>
                  <div className="sla-learn-detail-header">
                    <div className="sla-learn-detail-letter">{item.letter || item.title[0]}</div>
                    <div className="sla-learn-detail-info">
                      <div className="sla-learn-detail-title">{item.title}</div>
                      <div className="sla-learn-detail-sub">Difficulty: {item.difficulty || 'Easy'}</div>
                    </div>
                  </div>
                  
                  <div className="sla-learn-hand-tags">
                    <span className="sla-hand-tag">{item.hands || 'One-handed'}</span>
                    <span className="sla-hand-tag">Static</span>
                  </div>

                  <div className="sla-learn-detail-desc">{item.description}</div>
                  
                  <div className="sla-learn-steps">
                    {item.steps?.map((step, idx) => (
                      <div key={idx} className="sla-learn-step">
                        <div className="sla-learn-step-num">{idx + 1}</div>
                        <div>{step}</div>
                      </div>
                    ))}
                  </div>

                  <button className="sla-action-btn primary" style={{ width: '100%', marginTop: '1rem', justifyContent: 'center' }}>
                    <Play size={12} /> Practice this gesture
                  </button>
                </div>
              )}
            </React.Fragment>
          );
        })}
      </div>
    </div>
  );
};
