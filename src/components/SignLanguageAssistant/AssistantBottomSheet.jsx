import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence, useAnimation, useDragControls } from 'framer-motion';
import { X, HandMetal } from 'lucide-react';
import { LearnTab } from './tabs/LearnTab';
import { LiveDetectionTab } from './tabs/LiveDetectionTab';
import { PracticeTab } from './tabs/PracticeTab';
import { HistoryTab } from './tabs/HistoryTab';
import { DataCollectionTab } from './tabs/DataCollectionTab';
import { useAssistantHistory } from '../../hooks/useAssistantHistory';
import { useGestureRecognition } from '../../hooks/useGestureRecognition';
import './assistant.css';

export const AssistantBottomSheet = ({ isOpen, onClose, onSendToChat }) => {
  const [activeTab, setActiveTab] = useState('live'); // 'learn' | 'live' | 'practice' | 'history'
  const controls = useAnimation();
  const dragControls = useDragControls();
  const { addEntry } = useAssistantHistory();

  // For PracticeTab integration
  const { detectedLetter, confidence } = useGestureRecognition({ enabled: activeTab === 'practice' && isOpen });

  // Handle open/close animations
  useEffect(() => {
    if (isOpen) {
      controls.start({ y: 0, transition: { type: 'spring', damping: 25, stiffness: 200 } });
    } else {
      controls.start({ y: '100%', transition: { type: 'spring', damping: 25, stiffness: 200 } });
    }
  }, [isOpen, controls]);

  // Handle drag to dismiss
  const handleDragEnd = (event, info) => {
    if (info.offset.y > 100 || info.velocity.y > 500) {
      onClose();
    } else {
      controls.start({ y: 0, transition: { type: 'spring', damping: 25, stiffness: 200 } });
    }
  };

  const handleSend = (text) => {
    // Add to history
    addEntry(text);
    // Send to main chat
    if (onSendToChat) {
      onSendToChat(text);
    }
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          <motion.div 
            className="sla-backdrop"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
          />
          
          <motion.div
            className="sla-sheet"
            initial={{ y: '100%' }}
            animate={controls}
            exit={{ y: '100%' }}
            drag="y"
            dragControls={dragControls}
            dragListener={false}
            dragConstraints={{ top: 0, bottom: 0 }}
            dragElastic={0.2}
            onDragEnd={handleDragEnd}
            style={{ height: '70vh' }}
          >
            {/* Drag Handle */}
            <div 
              className="sla-drag-handle" 
              onPointerDown={(e) => dragControls.start(e)}
            >
              <div className="sla-drag-bar" />
            </div>

            {/* Header */}
            <div className="sla-header">
              <div className="sla-title">
                <HandMetal size={18} className="sla-title-icon" />
                Sign Language Assistant
              </div>
              <button className="sla-close-btn" onClick={onClose} title="Close">
                <X size={18} />
              </button>
            </div>

            {/* Tabs */}
            <div className="sla-tab-bar">
              <button 
                className={`sla-tab ${activeTab === 'live' ? 'active' : ''}`}
                onClick={() => setActiveTab('live')}
              >
                {activeTab === 'live' && <div className="sla-tab-dot" />}
                Live Detection
              </button>
              <button 
                className={`sla-tab ${activeTab === 'learn' ? 'active' : ''}`}
                onClick={() => setActiveTab('learn')}
              >
                Learn
              </button>
              <button 
                className={`sla-tab ${activeTab === 'practice' ? 'active' : ''}`}
                onClick={() => setActiveTab('practice')}
              >
                Practice
              </button>
              <button 
                className={`sla-tab ${activeTab === 'history' ? 'active' : ''}`}
                onClick={() => setActiveTab('history')}
              >
                History
              </button>
              <button 
                className={`sla-tab ${activeTab === 'dataset' ? 'active' : ''}`}
                onClick={() => setActiveTab('dataset')}
                style={{ color: '#0d9488' }}
              >
                Dataset
              </button>
            </div>

            {/* Tab Content Area */}
            <div className="sla-tab-content">
              {activeTab === 'live' && <LiveDetectionTab isActive={isOpen && activeTab === 'live'} onSendToChat={handleSend} />}
              {activeTab === 'learn' && <LearnTab />}
              {activeTab === 'practice' && <PracticeTab activeLetter={detectedLetter} confidence={confidence} isActive={isOpen && activeTab === 'practice'} />}
              {activeTab === 'history' && <HistoryTab onSendToChat={handleSend} />}
              {activeTab === 'dataset' && <DataCollectionTab isActive={isOpen && activeTab === 'dataset'} />}
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
};
