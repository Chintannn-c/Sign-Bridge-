"""
SignBridge SQLite Database Initialization & Management
Database File: backend/database/signbridge.db
"""

from pathlib import Path
import sqlite3

DB_DIR = Path(__file__).resolve().parent
DB_PATH = DB_DIR / "signbridge.db"


def get_db_connection():
    """Returns a SQLite connection with dict-like row factory."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes SQLite database tables if they do not exist."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Table for Conversation & Translation History
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS conversation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            speaker TEXT NOT NULL,          -- 'human' or 'robot'
            raw_text TEXT,                  -- Raw fingerspelled letters (e.g. 'H E L L O')
            refined_sentence TEXT,          -- Refined LLM sentence (e.g. 'Hello, how can I assist you?')
            confidence REAL,               -- Gesture model confidence score (0.0 to 1.0)
            llm_provider TEXT              -- 'Groq', 'Gemini', or 'local_fallback'
        );
    """
    )

    # 2. Table for Dataset Session Metadata
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS dataset_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            letter TEXT NOT NULL,
            session_id TEXT UNIQUE NOT NULL,
            signer_id TEXT,
            frame_count INTEGER,
            file_path TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """
    )

    # Insert sample seed data if empty
    cursor.execute("SELECT COUNT(*) FROM conversation_history")
    if cursor.fetchone()[0] == 0:
        cursor.execute(
            """
            INSERT INTO conversation_history (speaker, raw_text, refined_sentence, confidence, llm_provider)
            VALUES 
            ('human', 'H E L L O', 'Hello! How can I help you today?', 0.96, 'Groq (Llama-3.3-70b)'),
            ('robot', 'WELCOME', 'Welcome to SignBridge ISL Assistant.', 1.0, 'System'),
            ('human', 'W A S H R O O M', 'Where is the washroom located?', 0.92, 'Google Gemini (1.5-Flash)')
        """
        )

    conn.commit()
    conn.close()
    print(f"SQLite Database successfully initialized at {DB_PATH}")


if __name__ == "__main__":
    init_db()
