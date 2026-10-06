"""
Sign-Bridge Flask API — Main Application Server

REST API endpoints:
  GET  /api/health            - Server health check
  GET  /api/model/info        - Model metadata and status
  POST /api/translate         - Translate hand landmarks to ISL letter
  POST /api/robot/sign        - Send text to Arduino robotic hands
  GET  /api/robot/status      - Arduino connection status
  POST /api/robot/connect     - Connect to Arduino
  POST /api/robot/disconnect  - Disconnect from Arduino

Run:
    cd backend
    python app.py

The server starts on http://localhost:5000
"""

import os
import sys
import json
import re
import logging
import numpy as np
from typing import TypedDict
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))

# Add parent dir so services can find dataset paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.translator_model import TranslatorModel
from services.arduino_serial import ArduinoSerial
from services.word_recognizer import WordRecognizer
from services.gemini_manager import gemini_manager
from services.groq_manager import groq_manager
from database.schema import init_db, log_conversation, get_recent_history, log_dataset_session

# ─── Configuration ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("SignBridge.API")

try:
    init_db()
except Exception as e:
    logger.warning(f"Database initialization warning: {e}")

app = Flask(__name__)
CORS(app, origins=[  # type: ignore[arg-type] # pyrefly: ignore[bad-argument-type]
    'http://localhost:5173', 'http://localhost:5174', 'http://localhost:5175',
    'http://localhost:3000', 'http://127.0.0.1:5173', 'http://127.0.0.1:5174'
])

# ─── Initialize Services ───────────────────────────────────────────────────
logger.info("Initializing Sign-Bridge API services...")
translator = TranslatorModel()
word_recognizer = WordRecognizer()
arduino = ArduinoSerial()

logger.info(f"Translator mode: {translator.mode}")
logger.info("Services ready.")


def get_smart_fallback_response(query_text: str) -> str:
    """
    Intelligent keyword-based fallback response when LLM APIs are offline/unreachable.
    Responds directly, casually, and like a real human (short, 1 brief sentence).
    """
    q = query_text.lower().strip()
    if len(q) <= 2 and q not in ('hi', 'no', 'ok'):
        return "Could you sign that again?"
    elif any(k in q for k in ['washroom', 'toilet', 'restroom', 'bathroom']):
        return "Down the hall on your left."
    elif any(k in q for k in ['repeat', 'say again', 'once more', 'pardon']):
        return "Sure, what did you want me to repeat?"
    elif any(k in q for k in ['water', 'drink', 'thirsty', 'tea', 'coffee']):
        return "Sure thing! Here's some water for you."
    elif any(k in q for k in ['food', 'hungry', 'eat', 'lunch', 'dinner', 'snack']):
        return "Let's grab a bite! What are you craving?"
    elif any(k in q for k in ['hello', 'hi', 'namaste', 'hey', 'greetings']):
        return "Hey! How's your day going?"
    elif any(k in q for k in ['how are you', 'how do you do', 'how r u']):
        return "Doing great, thanks! How about you?"
    elif any(k in q for k in ['name', 'who are you']):
        return "I'm SignBridge! Nice to meet you."
    elif any(k in q for k in ['thank', 'thanks']):
        return "Anytime! Happy to help."
    elif any(k in q for k in ['help', 'assist', 'support']):
        return "I'm right here! How can I help?"
    elif any(k in q for k in ['bye', 'goodbye', 'see you']):
        return "Take care! See you soon."
    elif any(k in q for k in ['morning']):
        return "Good morning! Hope you have a great day."
    elif any(k in q for k in ['evening', 'night']):
        return "Good evening! How's everything?"
    elif any(k in q for k in ['nice to meet you']):
        return "Nice to meet you too!"
    elif any(k in q for k in ['where', 'location', 'direction']):
        return "Tell me where you want to go and I'll point the way."
    else:
        return "Got it! How can I help with that?"


class LLMResponse(TypedDict):
    text: str
    provider: str
    fallback_used: bool


def generate_llm_response(
    prompt: str,
    system_instruction: str = "You are SignBridge AI assistant.",
    temperature: float = 0.6,
    max_tokens: int = 60
) -> LLMResponse:
    """
    Tiered Automatic Drop-Down LLM Cascade:
    1. Primary Tier: Groq LPU Manager (Dual-Key rotation, auto-discovery & backoff retry).
    2. Secondary Tier: Google Gemini Manager (Dual-Key rotation, auto-discovery & backoff retry).
    3. Final Tier: Smart Local Semantic Engine.
    """
    errors = []

    # 1. Primary Tier Attempt: Centralized Groq LPU Manager
    if groq_manager.is_available():
        try:
            groq_res = groq_manager.generate(
                prompt=prompt,
                system_instruction=system_instruction,
                temperature=temperature,
                max_tokens=max_tokens
            )
            if groq_res and groq_res.get("text"):
                return {
                    "text": groq_res["text"],
                    "provider": groq_res["provider"],
                    "fallback_used": groq_res.get("fallback_used", False)
                }
        except Exception as groq_err:
            errors.append(f"GroqManager error: {groq_err}")

    # 2. Secondary Tier Attempt: Centralized Google Gemini Manager
    if gemini_manager.is_available():
        try:
            gem_res = gemini_manager.generate(
                prompt=prompt,
                system_instruction=system_instruction,
                task_type="text",
                temperature=temperature,
                max_output_tokens=max_tokens
            )
            if gem_res and gem_res.get("text"):
                return {
                    "text": gem_res["text"],
                    "provider": gem_res["provider"],
                    "fallback_used": True
                }
        except Exception as gem_err:
            errors.append(f"GeminiManager error: {gem_err}")

    # 3. Final Tier Attempt: Smart Local Keyword Semantic Engine
    logger.warning(f"All external LLM APIs exhausted or unreachable. Errors: {'; '.join(errors)}")
    fallback_text = get_smart_fallback_response(prompt)
    return {
        "text": fallback_text,
        "provider": "Local Semantic Engine",
        "fallback_used": True
    }


def safe_float(val: object, default: float = 0.0) -> float:
    """Safely convert any raw value to float without errors."""
    try:
        return float(val)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return default


def clean_llm_text(text: str) -> str:
    """Removes thinking blocks, reasoning fences, markdown formatting, and normalizes characters."""
    if not text:
        return ""

    cleaned = text
    # Strip <think> tags, reasoning preambles, and code fences
    cleaned = re.sub(r'<think>[\s\S]*?(?:<\/think>|$)', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'^(?:Here\'?s (?:a )?(?:quick )?thinking process:?|Thinking Process:?|Thought Process:?|Thinking:?)[\s\S]*?(?=\n\n|\n[A-Z0-9]|$)', '', cleaned, flags=re.IGNORECASE)
    if match := re.search(r'(?:Final Answer|Polished Sentence|Translation|Response):\s*([^\n]+)', cleaned, re.IGNORECASE):
        cleaned = match.group(1)
    cleaned = re.sub(r'```(?:[a-zA-Z]*\n)?([\s\S]*?)```', r'\1', cleaned)
    cleaned = re.sub(r'[#*`"\'\u2018\u2019\u201c\u201d]', '', cleaned)
    return re.sub(r'\s+', ' ', cleaned).strip()


def limit_to_human_length(text: str) -> str:
    """
    Ensure the response stays small and conversational (1-2 short sentences max, under 18 words).
    """
    if not text:
        return ""
    # Split by sentence ending punctuation (.!?) keeping punctuation
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    if not sentences:
        return text

    if len(sentences) == 1:
        return sentences[0]

    # If first sentence is very short (e.g. "Hey!", "Sure thing!", "Namaste!"), keep second sentence too
    first_len = len(sentences[0].split())
    if first_len <= 3 and len(sentences) >= 2:
        return f"{sentences[0]} {sentences[1]}"

    # Otherwise return just the first punchy sentence
    return sentences[0]


@app.route('/api/llm/refine', methods=['POST'])
def llm_refine():
    """
    Refine raw ISL letter/word buffer into a natural spoken English sentence.
    Uses automatic fallback (Groq -> Gemini).
    """
    data = request.get_json() or {}
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'error': 'Missing "text" parameter.'}), 400

    # Quick short-circuit: single letter or empty stays minimal
    if len(text) <= 1:
        return jsonify({
            'raw_text': text,
            'refined_sentence': text,
            'llm_provider': 'passthrough',
            'fallback_used': False
        })

    system_instruction = (
        "You are an Indian Sign Language (ISL) translator for a live dialogue system.\n"
        "Convert raw recognized ISL glosses, keywords, or fingerspelled letters into 1 short, natural, everyday English sentence — exactly how a real human speaks in casual conversation.\n\n"
        "Rules:\n"
        "1. Keep it short, casual, and direct. Use natural contractions (e.g., 'What's', 'I'm', 'Where's', 'Can I').\n"
        "   - 'NAME YOU WHAT' -> 'What\'s your name?'\n"
        "   - 'WATER PLEASE' -> 'Can I get some water?'\n"
        "   - 'WHERE WASHROOM' -> 'Where\'s the washroom?'\n"
        "   - 'ME HUNGRY' -> 'I\'m hungry.'\n"
        "   - 'ME DELHI GO' -> 'I\'m going to Delhi.'\n"
        "   - 'TRAIN TIME WHEN' -> 'When does the train arrive?'\n"
        "   - 'THANK YOU' -> 'Thank you!'\n"
        "   - 'HELLO NAMASTE' -> 'Hello! Namaste.'\n"
        "2. If input is fingerspelled letters (e.g. 'H E L L O'), merge them into words.\n"
        "3. NEVER add explanations, meta-commentary, or extra sentences.\n"
        "4. Return ONLY the final spoken sentence."
    )

    try:
        res = generate_llm_response(text, system_instruction=system_instruction, temperature=0.3, max_tokens=150)
        cleaned_text = limit_to_human_length(clean_llm_text(res['text']))
        # Log to SQLite database
        log_conversation(
            speaker='human',
            raw_text=text,
            refined_sentence=cleaned_text,
            confidence=1.0,
            llm_provider=res['provider']
        )
        return jsonify({
            'raw_text': text,
            'refined_sentence': cleaned_text,
            'llm_provider': res['provider'],
            'fallback_used': res['fallback_used']
        })
    except Exception as e:
        logger.error(f"LLM refine failed: {e}", exc_info=True)
        return jsonify({'error': 'LLM refinement service failed.'}), 500


@app.route('/api/llm/simplify', methods=['POST'])
def llm_simplify():
    """
    Simplify complex spoken text into essential keywords for robotic hands.
    Uses automatic fallback (Groq -> Gemini).
    """
    data = request.get_json() or {}
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'error': 'Missing "text" parameter.'}), 400

    system_instruction = (
        "Convert English spoken text into 1 to 4 core Indian Sign Language (ISL) keyword glosses "
        "for robotic hands.\n\n"
        "Rules:\n"
        "1. Extract ONLY key content words (nouns, main verbs, core adjectives, question words).\n"
        "2. Drop filler words, articles (a, an, the), and auxiliary verbs.\n"
        "3. Return ONLY uppercase keywords separated by space (e.g., 'WATER PLEASE', 'WASHROOM LEFT', 'WELCOME').\n"
        "4. Maximum 4 words. No markdown fences, punctuation, or explanations."
    )

    try:
        res = generate_llm_response(text, system_instruction=system_instruction, temperature=0.2, max_tokens=30)
        cleaned_keywords = clean_llm_text(res['text']).upper()
        # Keep only letters and spaces
        cleaned_keywords = re.sub(r'[^A-Z\s]', '', cleaned_keywords)
        cleaned_keywords = re.sub(r'\s+', ' ', cleaned_keywords).strip()
        return jsonify({
            'original_speech': text,
            'robot_keywords': cleaned_keywords,
            'llm_provider': res['provider'],
            'fallback_used': res['fallback_used']
        })
    except Exception as e:
        logger.error(f"LLM simplify failed: {e}", exc_info=True)
        return jsonify({'error': 'LLM simplification service failed.'}), 500


@app.route('/api/llm/answer', methods=['POST'])
def llm_answer():
    """
    Generate a small, warm, human-like AI response to what the user signed/said.
    Uses automatic fallback (Groq -> Gemini -> Smart Local Fallback).
    """
    data = request.get_json() or {}
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'error': 'Missing "text" parameter.'}), 400

    # If single isolated character or gibberish (e.g. 'L', 'X'), respond casually without lecturing
    clean_in = text.strip()
    if len(clean_in) <= 2 and clean_in.lower() not in ('hi', 'no', 'ok'):
        short_prompt_ans = "Could you sign that again?"
        return jsonify({
            'user_text': text,
            'answer': short_prompt_ans,
            'llm_provider': 'direct_guard',
            'fallback_used': False
        })

    system_instruction = (
        "You are a friendly, everyday human conversation partner talking with someone using sign language through SignBridge.\n"
        "Reply directly, naturally, and warmly to what they said.\n\n"
        "CRITICAL RULES:\n"
        "1. KEEP IT SMALL: Respond in exactly 1 brief sentence (5 to 12 words max). Never write long paragraphs.\n"
        "2. SOUND LIKE A REAL HUMAN: Warm, casual, and friendly. Use contractions like 'I\'m', 'what\'s', 'here\'s', 'sure thing'.\n"
        "3. YOU ARE NOT A TEACHER: The user is communicating in real life! If they say 'water', 'food', or 'where is the washroom', answer their real-life need (e.g., 'Here\'s some water!' or 'Down the hall on your left.'). NEVER teach them how to sign or describe handshapes!\n"
        "4. ZERO BOT CLICHES: Never say 'How can I assist you with Indian Sign Language today?', 'As an AI', 'Feel free to sign', or 'Got it!'.\n"
        "5. If asked your name, say: 'I\'m SignBridge! Nice to meet you.'\n"
        "6. Return ONLY the spoken response. No quotes, no preamble, no reasoning."
    )

    try:
        res = generate_llm_response(text, system_instruction=system_instruction, temperature=0.7, max_tokens=180)
        cleaned_answer = limit_to_human_length(clean_llm_text(res['text']))
        # Log to SQLite database
        log_conversation(
            speaker='robot',
            raw_text=text,
            refined_sentence=cleaned_answer,
            confidence=1.0,
            llm_provider=res['provider']
        )
        return jsonify({
            'user_text': text,
            'answer': cleaned_answer,
            'llm_provider': res['provider'],
            'fallback_used': res['fallback_used']
        })
    except Exception as e:
        logger.warning(f"LLM answer API error (using smart local fallback): {e}")
        fallback_ans = get_smart_fallback_response(text)
        log_conversation(
            speaker='robot',
            raw_text=text,
            refined_sentence=fallback_ans,
            confidence=1.0,
            llm_provider='local_smart_fallback'
        )
        return jsonify({
            'user_text': text,
            'answer': fallback_ans,
            'llm_provider': 'local_smart_fallback',
            'fallback_used': True
        })


# ═══════════════════════════════════════════════════════════════════════════
# HEALTH & INFO ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/api/health', methods=['GET'])
def health():
    """Server health check."""
    return jsonify({
        'status': 'ok',
        'service': 'Sign-Bridge Flask API',
        'translator_mode': translator.mode,
        'arduino_connected': arduino.is_connected,
        'groq_available': groq_manager.is_available(),
        'groq_health': groq_manager.get_health_status(),
        'gemini_available': gemini_manager.is_available(),
        'gemini_health': gemini_manager.get_health_status(),
        'word_recognizer_available': word_recognizer.is_available
    })


@app.route('/api/llm/status', methods=['GET'])
def llm_status():
    """Return real-time status of Groq, Gemini multi-key management, and Local LLM tiers."""
    return jsonify({
        'groq': groq_manager.get_health_status(),
        'gemini': gemini_manager.get_health_status(),
        'local_fallback': True
    })


@app.route('/api/model/info', methods=['GET'])
def model_info():
    """Return alphabet model metadata."""
    return jsonify(translator.get_info())


@app.route('/api/words/info', methods=['GET'])
def words_info():
    """Return word recognizer metadata."""
    return jsonify(word_recognizer.get_info())


@app.route('/api/model/reload', methods=['POST', 'GET'])
def model_reload():
    """Reload model weights and metadata from disk."""
    translator._load()
    word_recognizer._load()
    return jsonify({
        'status': 'reloaded',
        'alphabet_model': translator.get_info(),
        'word_model': word_recognizer.get_info()
    })


# ═══════════════════════════════════════════════════════════════════════════
# TRANSLATION ENDPOINTS (Path 1: Signs -> Text)
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/api/translate', methods=['POST'])
def translate():
    """
    Translate hand landmarks to an ISL letter/word.
    """
    data = request.get_json()
    if not data or 'landmarks' not in data:
        return jsonify({
            'error': 'Missing "landmarks" field in request body.',
            'expected': 'Array of 42 landmark points (each with x,y,z) or flat array of 126 floats.'
        }), 400

    landmarks = data['landmarks']

    # Fast guard: if no hands are visible (all zeros or empty), do not predict
    try:
        arr_check = np.asarray(landmarks, dtype=np.float32)
        if arr_check.size == 0 or not np.any(arr_check != 0) or np.all(np.abs(arr_check) < 1e-4):
            return jsonify({
                'letter': '?',
                'confidence': 0.0,
                'mode': translator.mode,
                'rejected': True,
                'rejection_reason': 'no_hands_detected',
                'all_scores': {}
            })
    except Exception:
        return jsonify({
            'letter': '?',
            'confidence': 0.0,
            'mode': translator.mode,
            'rejected': True,
            'rejection_reason': 'invalid_landmarks_payload',
            'all_scores': {}
        })

    try:
        body_anchors = data.get('body_anchors')
        result = translator.predict(landmarks, body_anchors=body_anchors)

        conf_raw = result.get('confidence', 0.0)
        confidence = safe_float(conf_raw, 0.0)

        is_rej = bool(result.get('rejected', False)) or (confidence < 0.50) or (str(result.get('letter', '')) == '?')
        if is_rej:
            result['rejected'] = True
            result['rejection_reason'] = result.get('rejection_reason') or 'low_confidence'
            result['original_letter'] = result.get('letter', '?')
            result['letter'] = '?'
        else:
            result['rejected'] = False

        return jsonify(result)
    except Exception as e:
        logger.error(f"Translation error: {e}", exc_info=True)
        return jsonify({'error': 'Translation processing error.'}), 500


@app.route('/api/translate/batch', methods=['POST'])
def translate_batch():
    """
    Translate a batch of landmark frames (for sentence-level recognition).
    """
    data = request.get_json()
    if not data or 'frames' not in data:
        return jsonify({'error': 'Missing "frames" field.'}), 400

    results = []
    sentence_letters = []

    for frame_landmarks in data['frames']:
        try:
            result = translator.predict(frame_landmarks)
            results.append(result)
            conf_raw = result.get('confidence', 0.0)
            conf_val = safe_float(conf_raw, 0.0)

            if conf_val > 0.5:
                sentence_letters.append(str(result.get('letter', '?')))
        except Exception as e:
            results.append({'letter': '?', 'confidence': 0.0, 'error': str(e)})

    return jsonify({
        'results': results,
        'sentence': ''.join(sentence_letters)
    })


@app.route('/api/translate/word', methods=['POST'])
def translate_word():
    """
    Translate a sequence of landmark frames to an ISL word.
    Uses the CNN-BiLSTM temporal word recognizer.
    """
    if not word_recognizer.is_available:
        return jsonify({
            'error': 'Word recognizer is not available. Train the CNN-BiLSTM model first.',
            'hint': 'Run: cd backend && python train_model_cnn_lstm.py'
        }), 503

    data = request.get_json()
    if not data or 'frames' not in data:
        return jsonify({'error': 'Missing "frames" field.'}), 400

    frames = data['frames']

    # Fast guard: if no hands are visible across frames (all zeros), do not predict
    try:
        arr_check = np.asarray(frames, dtype=np.float32)
        if not np.any(arr_check != 0):
            return jsonify({
                'word': '?',
                'confidence': 0.0,
                'mode': word_recognizer.mode,
                'rejected': True,
                'rejection_reason': 'no_hands_detected',
                'all_scores': {}
            })
    except Exception:
        pass

    try:
        body_anchors = data.get('body_anchors')
        result = word_recognizer.predict(frames, body_anchors=body_anchors)
        if result is None:
            return jsonify({'error': 'Word prediction failed or input invalid.'}), 400

        # Confidence rejection for words
        conf_raw = result.get('confidence', 0.0)
        confidence = safe_float(conf_raw, 0.0)

        if confidence < 0.60 or str(result.get('word', '')) == '?':
            result['rejected'] = True
            result['rejection_reason'] = 'low_confidence'
            result['original_word'] = result.get('word', '?')
            result['word'] = '?'
        else:
            result['rejected'] = False

        return jsonify(result)
    except Exception as e:
        logger.error(f"Word translation error: {e}", exc_info=True)
        return jsonify({'error': 'Word translation processing error.'}), 500


@app.route('/api/history', methods=['GET'])
def get_history():
    """Returns recent conversation history from SQLite database."""
    limit = request.args.get('limit', default=50, type=int)
    history = get_recent_history(limit=limit)
    return jsonify({
        'status': 'ok',
        'history': history,
        'count': len(history)
    })


# -----------------------------------------------------------------------------
# DATASET COLLECTION ENDPOINTS
# -----------------------------------------------------------------------------

@app.route('/api/collect_data', methods=['POST'])
def collect_data():
    """
    Saves recorded landmark frames for a specific ISL letter to build a dataset.
    
    Request JSON body:
    {
        "letter": "A",
        "session_id": "timestamp-uuid",
        "frames": [
            [x1, y1, z1, ...], // 126 coordinates
            ...
        ]
    }
    """
    data = request.get_json()
    if not data or 'frames' not in data or 'letter' not in data:
        return jsonify({'error': 'Missing required fields: frames, letter.'}), 400
        
    letter = str(data['letter']).strip().upper()
    session_id = str(data.get('session_id', 'unknown_session')).strip()
    signer_id = str(data.get('signer_id', 'unknown_signer')).strip()

    # Prevent path traversal and enforce alphanumeric labels
    if not re.match(r'^[A-Z0-9]$', letter) or not re.match(r'^[a-zA-Z0-9_\-]+$', session_id):
        return jsonify({'error': 'Invalid letter or session_id: only alphanumeric characters allowed.'}), 400

    frames = data['frames']
    
    # Save directory
    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset_collected', letter)
    os.makedirs(dataset_dir, exist_ok=True)
    
    file_path = os.path.join(dataset_dir, f"{session_id}.json")
    
    try:
        with open(file_path, 'w') as f:
            json.dump({'letter': letter, 'session_id': session_id, 'signer_id': signer_id, 'frames': frames}, f)
        
        # Log to SQLite dataset_sessions table
        log_dataset_session(
            letter=letter,
            session_id=session_id,
            signer_id=signer_id,
            frame_count=len(frames),
            file_path=file_path
        )
        
        logger.info(f"Saved {len(frames)} frames for letter {letter} to {file_path}")
        return jsonify({'status': 'ok', 'message': f'Saved {len(frames)} frames.'})
    except Exception as e:
        logger.error(f"Failed to save data: {e}", exc_info=True)
        return jsonify({'error': 'Failed to save dataset collection session.'}), 500


# ═══════════════════════════════════════════════════════════════════════════
# ROBOT / ARDUINO ENDPOINTS (Path 2: Text -> Signs)
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/api/robot/status', methods=['GET'])
def robot_status():
    """Return Arduino connection status."""
    return jsonify(arduino.get_status())


@app.route('/api/robot/connect', methods=['POST'])
def robot_connect():
    """
    Connect to Arduino.

    Optional JSON body:
    {
        "port": "COM3",        // Optional, auto-detects if omitted
        "baud_rate": 9600      // Optional, defaults to 9600
    }
    """
    data = request.get_json() or {}
    port = data.get('port')
    baud = data.get('baud_rate', 9600)

    if port:
        arduino.port = port
    arduino.baud_rate = baud

    success = arduino.connect()
    return jsonify({
        'connected': success,
        'status': arduino.get_status()
    })


@app.route('/api/robot/disconnect', methods=['POST'])
def robot_disconnect():
    """Disconnect from Arduino."""
    arduino.disconnect()
    return jsonify({'connected': False, 'message': 'Disconnected.'})


@app.route('/api/robot/sign', methods=['POST'])
def robot_sign():
    """
    Send text to the robotic hands for ISL fingerspelling.

    Request JSON body:
    {
        "text": "HELLO",
        "letter_hold": 1.5,   // Optional: seconds to hold each letter
        "gap": 0.5            // Optional: seconds between letters
    }

    Response:
    {
        "text": "HELLO",
        "signed_letters": ["H", "E", "L", "L", "O"],
        "arduino_connected": true
    }
    """
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing "text" field.'}), 400

    text = data['text']
    letter_hold = data.get('letter_hold', 1.5)
    gap = data.get('gap', 0.5)

    if not arduino.is_connected:
        # Return the servo commands that WOULD be sent (for debugging without hardware)
        from services.arduino_serial import ISL_SERVO_MAP
        commands = []
        for char in text.upper():
            if char in ISL_SERVO_MAP:
                commands.append({
                    'letter': char,
                    'angles': ISL_SERVO_MAP[char]
                })

        return jsonify({
            'text': text,
            'signed_letters': [c['letter'] for c in commands],
            'commands': commands,
            'arduino_connected': False,
            'message': 'Arduino not connected. Showing planned servo commands.'
        })

    # Asynchronous non-blocking background signing
    signed = arduino.sign_text(text, letter_hold=letter_hold, gap=gap, async_mode=True)
    return jsonify({
        'text': text,
        'signed_letters': signed,
        'arduino_connected': True,
        'is_signing': True,
        'message': f'Queued {len(signed)} letters for background signing on robotic hands.'
    })


@app.route('/api/robot/sign-letter', methods=['POST'])
def robot_sign_letter():
    """
    Sign a single letter on the robotic hands.

    Request JSON body:
    {
        "letter": "A",
        "hold_time": 2.0   // Optional: seconds
    }
    """
    data = request.get_json()
    if not data or 'letter' not in data:
        return jsonify({'error': 'Missing "letter" field.'}), 400

    letter = data['letter'].upper()
    hold = data.get('hold_time', 1.5)

    if not arduino.is_connected:
        from services.arduino_serial import ISL_SERVO_MAP, REST_POSE
        angles = ISL_SERVO_MAP.get(letter, REST_POSE)
        return jsonify({
            'letter': letter,
            'angles': angles,
            'arduino_connected': False,
            'message': 'Arduino not connected. Showing planned servo angles.'
        })

    success = arduino.sign_letter(letter, hold_time=hold)
    return jsonify({
        'letter': letter,
        'signed': success,
        'arduino_connected': True
    })


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    host = os.getenv('HOST', '127.0.0.1')
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() in ('true', '1')
    logger.info("=" * 60)
    logger.info("  SIGN-BRIDGE FLASK API SERVER")
    logger.info(f"  http://{host}:{port}")
    logger.info("=" * 60)
    app.run(
        host=host,
        port=port,
        debug=debug
    )
