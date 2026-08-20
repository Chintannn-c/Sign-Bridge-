import React, { useState, useEffect } from 'react';
import { Target, CheckCircle2, ArrowRight } from 'lucide-react';
import { ALPHABET_GESTURES } from '../../../data/gestureData';

export const PracticeTab = ({ activeLetter, confidence, isActive }) => {
  const [targetIndex, setTargetIndex] = useState(0);
  const [score, setScore] = useState(0);
  const [attempts, setAttempts] = useState(0);
  const [isSuccess, setIsSuccess] = useState(false);

  const targetGesture = ALPHABET_GESTURES[targetIndex];

  // Check if current detected letter matches target
  useEffect(() => {
    if (!isActive || isSuccess) return;

    if (activeLetter === targetGesture.letter && confidence > 0.6) {
      setIsSuccess(true);
      setScore(prev => prev + 100);
      setAttempts(prev => prev + 1);
    } else if (activeLetter && activeLetter !== targetGesture.letter) {
      // Small penalty or track attempt
    }
  }, [activeLetter, confidence, targetGesture.letter, isActive, isSuccess]);

  const handleNext = () => {
    setTargetIndex(prev => (prev + 1) % ALPHABET_GESTURES.length);
    setIsSuccess(false);
    if (!isSuccess) setAttempts(prev => prev + 1); // Skipped
  };

  return (
    <div className="sla-tab-pane">
      <div className="sla-practice-target">
        <div className="sla-practice-prompt">Perform this gesture</div>
        <div className="sla-practice-letter">{targetGesture.letter}</div>
        <div className="sla-practice-name">{targetGesture.title}</div>
      </div>

      <div className="sla-practice-score">
        <div className="sla-score-card">
          <div className="sla-score-value">{score}</div>
          <div className="sla-score-label">Score</div>
        </div>
        <div className="sla-score-card">
          <div className="sla-score-value">{attempts}</div>
          <div className="sla-score-label">Attempts</div>
        </div>
        <div className="sla-score-card">
          <div className="sla-score-value">
            {attempts > 0 ? Math.round((score / (attempts * 100)) * 100) : 0}%
          </div>
          <div className="sla-score-label">Accuracy</div>
        </div>
      </div>

      {isSuccess ? (
        <div className="sla-practice-result correct">
          <CheckCircle2 size={16} style={{ display: 'inline', verticalAlign: 'middle', marginRight: '4px' }} />
          Excellent! Gesture recognized correctly.
        </div>
      ) : (
        <div className="sla-practice-result">
          <Target size={16} style={{ display: 'inline', verticalAlign: 'middle', marginRight: '4px' }} />
          Show the gesture to the camera...
        </div>
      )}

      <button className="sla-practice-next-btn" onClick={handleNext}>
        {isSuccess ? 'Next Gesture' : 'Skip Gesture'} <ArrowRight size={14} />
      </button>
    </div>
  );
};
