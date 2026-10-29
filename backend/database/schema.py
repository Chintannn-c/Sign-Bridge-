"""
SignBridge SQLite Database Initialization & Management
Database File: backend/database/signbridge.db
"""

import os
import logging
from pathlib import Path
import sqlite3
from contextlib import closing

logger = logging.getLogger("SignBridge.Database")

_env_db_path = os.getenv("DB_PATH")
if _env_db_path:
    DB_PATH = Path(_env_db_path).resolve()
    DB_DIR = DB_PATH.parent
else:
    DB_DIR = Path(__file__).resolve().parent
    DB_PATH = DB_DIR / "signbridge.db"


def get_db_connection(timeout: float = 5.0):
    """Returns a SQLite connection with dict-like row factory and WAL journal mode."""
    conn = sqlite3.connect(DB_PATH, timeout=timeout)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
    except Exception:
        pass
    return conn


def init_db():
    """Initializes SQLite database tables if they do not exist."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    with closing(get_db_connection()) as conn:
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

        conn.commit()
    logger.info(f"SQLite Database successfully initialized at {DB_PATH}")


def log_conversation(speaker, raw_text=None, refined_sentence=None, confidence=None, llm_provider=None):
    """Inserts a new conversation record into SQLite."""
    try:
        with closing(get_db_connection()) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO conversation_history (speaker, raw_text, refined_sentence, confidence, llm_provider)
                VALUES (?, ?, ?, ?, ?)
                """,
                (speaker, raw_text, refined_sentence, confidence, llm_provider)
            )
            conn.commit()
            return cursor.lastrowid
    except Exception as e:
        logger.error(f"Failed to log conversation to SQLite: {e}")
        return None


def get_recent_history(limit=50):
    """Fetches the most recent conversation records."""
    try:
        with closing(get_db_connection()) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, timestamp, speaker, raw_text, refined_sentence, confidence, llm_provider
                FROM conversation_history
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,)
            )
            return [dict(row) for row in cursor.fetchall()]
    except Exception as e:
        logger.error(f"Failed to fetch conversation history: {e}")
        return []


def log_dataset_session(letter, session_id, signer_id=None, frame_count=0, file_path=None):
    """Inserts or updates a dataset session record into SQLite."""
    try:
        with closing(get_db_connection()) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO dataset_sessions (letter, session_id, signer_id, frame_count, file_path)
                VALUES (?, ?, ?, ?, ?)
                """,
                (letter, session_id, signer_id, frame_count, file_path)
            )
            conn.commit()
            return True
    except Exception as e:
        logger.error(f"Failed to log dataset session to SQLite: {e}")
        return False


if __name__ == "__main__":
    init_db()

