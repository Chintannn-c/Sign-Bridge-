import React from 'react';
import { History, Copy, Send, Trash2 } from 'lucide-react';
import { useAssistantHistory } from '../../../hooks/useAssistantHistory';

export const HistoryTab = ({ onSendToChat }) => {
  const { history, deleteEntry, clearHistory } = useAssistantHistory();

  const handleCopy = (text) => {
    navigator.clipboard.writeText(text);
  };

  const handleSend = (text) => {
    if (onSendToChat) onSendToChat(text);
  };

  if (history.length === 0) {
    return (
      <div className="sla-tab-pane">
        <div className="sla-history-empty">
          <History size={32} className="sla-history-empty-icon" />
          <div className="sla-history-empty-text">No history yet</div>
          <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
            Sentences you build will appear here.
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="sla-tab-pane">
      <div className="sla-history-list">
        {history.map(entry => (
          <div key={entry.id} className="sla-history-item">
            <div className="sla-history-text">
              {entry.sentence}
              <div className="sla-history-meta">
                <span>{entry.date} at {entry.time}</span>
                {entry.confidence > 0 && <span>• Conf: {entry.confidence}%</span>}
              </div>
            </div>
            
            <div className="sla-history-actions">
              <button className="sla-history-action" onClick={() => handleCopy(entry.sentence)} title="Copy text">
                <Copy size={12} />
              </button>
              <button className="sla-history-action" onClick={() => handleSend(entry.sentence)} title="Send to chat again">
                <Send size={12} />
              </button>
              <button className="sla-history-action delete" onClick={() => deleteEntry(entry.id)} title="Delete entry">
                <Trash2 size={12} />
              </button>
            </div>
          </div>
        ))}
      </div>

      <button className="sla-history-clear-btn" onClick={clearHistory}>
        <Trash2 size={14} /> Clear all history
      </button>
    </div>
  );
};
