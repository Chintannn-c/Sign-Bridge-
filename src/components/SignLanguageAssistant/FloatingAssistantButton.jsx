import React from 'react';
import { HandMetal } from 'lucide-react';
import './assistant.css';

export const FloatingAssistantButton = ({ onClick, isOpen }) => {
  if (isOpen) return null; // Hide button when sheet is open

  return (
    <button 
      className="sla-fab" 
      onClick={onClick}
      title="Open Sign Language Assistant"
    >
      <div className="sla-fab-icon" style={{ position: 'relative' }}>
        <HandMetal size={18} />
        <div className="sla-fab-pulse" />
      </div>
      Sign Language Assistant
    </button>
  );
};
