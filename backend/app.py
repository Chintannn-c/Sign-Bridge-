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
CORS(app, origins=['http://localhost:5173', 'http://localhost:3000', 'http://127.0.0.1:5173'])  # type: ignore

# ─── Initialize Services ───────────────────────────────────────────────────
logger.info("Initializing Sign-Bridge API services...")
translator = TranslatorModel()
word_recognizer = WordRecognizer()
arduino = ArduinoSerial()

# ═══════════════════════════════════════════════════════════════════════════
# TIERED DYNAMIC LLM CASCADE (Groq LPU -> Google Gemini -> Local)
# ═══════════════════════════════════════════════════════════════════════════

# Master priority candidate lists for Groq LPU
PREFERRED_GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.3-70b-specdec",
    "llama-3.1-70b-versatile",
    "openai/gpt-oss-120b",
    "qwen/qwen3.6-27b",
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
    "mixtral-8x7b-32768",
    "groq/compound",
    "groq/compound-mini",
]

# Initialize Groq Real-Time LLM Client
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
groq_client = None
active_groq_models: list[str] = PREFERRED_GROQ_MODELS.copy()

if GROQ_API_KEY:
    try:
        from groq import Groq
        groq_client = Groq(api_key=GROQ_API_KEY)
        try:
            available_groq = [m.id for m in groq_client.models.list().data if 'whisper' not in m.id and 'guard' not in m.id]
            if available_groq:
                ordered_groq = [m for m in PREFERRED_GROQ_MODELS if m in available_groq]
                for m in available_groq:
                    if m not in ordered_groq:
                        ordered_groq.append(m)
                active_groq_models = ordered_groq
        except Exception as e:
            logger.debug(f"Groq dynamic model list notice: {e}")
        logger.info(f"Groq LLM service initialized with tiered models: {active_groq_models[:4]}")
    except Exception as e:
        logger.warning(f"Groq LLM initialization warning: {e}")

logger.info(f"Translator mode: {translator.mode}")
logger.info("Services ready.")


def get_smart_fallback_response(query_text: str) -> str:
    """
    Intelligent keyword-based fallback response when LLM APIs are offline/unreachable.
    Responds directly to the specific question asked instead of returning random generic strings.
    """
    q = query_text.lower().strip()
    if any(k in q for k in ['washroom', 'toilet', 'restroom', 'bathroom']):
        return "The washroom is straight ahead to your left."
    elif any(k in q for k in ['repeat', 'say again', 'once more', 'pardon']):
        return "Sure! Please let me know what you would like me to sign or repeat."
    elif any(k in q for k in ['ice cream', 'food', 'hungry', 'eat', 'drink', 'coffee', 'tea', 'water']):
        return "That sounds wonderful! How can I assist you with your request?"
    elif any(k in q for k in ['hello', 'hi', 'namaste', 'hey', 'greetings']):
        return "Hello! Namaste. How can I assist you today?"
    elif any(k in q for k in ['how are you', 'how do you do', 'how r u']):
        return "I am doing well, thank you! How can I help you with Indian Sign Language today?"
    elif any(k in q for k in ['name', 'who are you']):
        return "I am SignBridge AI, your dual-communication Indian Sign Language assistant."
    elif any(k in q for k in ['thank', 'thanks']):
        return "You are very welcome! Happy to help."
    elif any(k in q for k in ['help', 'assist', 'support']):
        return "I am here to assist you! You can sign with your camera or type below."
    elif any(k in q for k in ['bye', 'goodbye', 'see you']):
        return "Goodbye! Have a wonderful day ahead."
    elif any(k in q for k in ['morning', 'afternoon', 'evening']):
        return "Good day! How can I help you today?"
    elif any(k in q for k in ['nice to meet you']):
        return "Nice to meet you too! Welcome to SignBridge."
    elif any(k in q for k in ['where', 'location', 'direction']):
        return "Please tell me which place you are looking for, and I will be happy to guide you."
    else:
        return f"Got it! Let me know how else I can assist you with '{query_text}'."


class LLMResponse(TypedDict):
    text: str
    provider: str
    fallback_used: bool


def generate_llm_response(prompt: str, system_instruction: str = "You are SignBridge AI assistant.") -> LLMResponse:
    """
    Tiered Automatic Drop-Down LLM Cascade:
    1. Primary Tier: Groq LPU (Latest 70B -> 8B -> Speculative Models for sub-100ms response).
    2. Secondary Tier: Google Gemini Manager (Dual-Key rotation, auto-discovery & backoff retry).
    3. Final Tier: Smart Local Semantic Engine.
    """
    errors = []

    # 1. Primary Tier Attempt: Groq LPU (Iterating latest to lowest fallback models)
    client_groq = groq_client
    if client_groq is not None:
        for model_id in active_groq_models:
            try:
                response = client_groq.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=300
                )
                if response.choices and len(response.choices) > 0:
                    msg = response.choices[0].message
                    content = getattr(msg, 'content', None)
                    if content:
                        result = str(content).strip()
                        if result:
                            return {
                                "text": result,
                                "provider": f"Groq ({model_id})",
                                "fallback_used": False
                            }
            except Exception as groq_err:
                errors.append(f"Groq [{model_id}] error: {groq_err}")
                continue

    # 2. Secondary Tier Attempt: Centralized Google Gemini Manager
    if gemini_manager.is_available():
        try:
            gem_res = gemini_manager.generate(
                prompt=prompt,
                system_instruction=system_instruction,
                task_type="text"
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
    """Safely convert any raw value (float, int, str, or unknown) to float without type errors."""
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, (str, bytes)):
        try:
            return float(val)
        except (ValueError, TypeError):
            return default
    return default


def clean_llm_text(text: str) -> str:
    """Removes thinking blocks, reasoning fences, and markdown formatting from LLM outputs."""
    if not text:
        return ""
    # Strip <think>...</think> if present
    cleaned = re.sub(r'<think>[\s\S]*?</think>', '', text, flags=re.IGNORECASE).strip()
    # If code fence present, extract fence content
    fences = re.findall(r'```(?:[a-zA-Z]*\n)?([\s\S]*?)```', cleaned)
    if fences:
        cleaned = fences[-1].strip()
    # Strip reasoning headers if any
    if '**Final' in cleaned:
        cleaned = cleaned.split('**Final')[-1].strip(':').strip()
    # Remove markdown bold/italic asterisks
    cleaned = re.sub(r'\*{1,3}', '', cleaned).strip()
    return cleaned if cleaned else text.strip()


@app.route('/api/llm/refine', methods=['POST'])
def llm_refine():
    """
    Refine raw ISL letter buffer into a fluent conversational sentence.
    Uses automatic fallback (Groq -> Gemini).
    """
    data = request.get_json() or {}
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'error': 'Missing "text" parameter.'}), 400

    system_instruction = (
        "You are SignBridge AI, an expert Indian Sign Language (ISL) Linguistic Translator.\n"
        "Your task is to translate raw ISL recognized glosses, keywords, or fingerspelled letter fragments into a natural, grammatically correct English sentence.\n\n"
        "ISL Linguistic Rules to apply:\n"
        "1. ISL uses Topic-Comment / SOV word order and often drops auxiliary verbs (is/are/am/was/were).\n"
        "   - 'NAME YOU WHAT' -> 'What is your name?'\n"
        "   - 'ME DELHI TOMORROW GO' -> 'I am going to Delhi tomorrow.'\n"
        "   - 'TRAIN TIME WHEN' -> 'What time will the train arrive?'\n"
        "   - 'WATER PLEASE' -> 'Could I please have some water?'\n"
        "   - 'WASHROOM WHERE' -> 'Where is the washroom?'\n"
        "2. If input is fingerspelled letters (e.g. 'H E L L O N A M A S T E'), seamlessly merge them into proper words.\n"
        "3. Preserve the speaker's intent concisely without adding unnecessary fluff.\n"
        "4. Return ONLY the final polished sentence without reasoning, markdown formatting, or explanations."
    )

    try:
        res = generate_llm_response(text, system_instruction=system_instruction)
        cleaned_text = clean_llm_text(res['text'])
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
        logger.error(f"LLM refine failed: {e}")
        return jsonify({'error': str(e)}), 500


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
        "You are SignBridge AI. Convert natural English spoken text into core Indian Sign Language (ISL) keyword glosses "
        "so dual robotic hands can actuate the signs sequentially.\n\n"
        "Rules:\n"
        "1. Extract ONLY key content words (nouns, main verbs, core adjectives, question words).\n"
        "2. Drop filler words, articles (a, an, the), and auxiliary verbs.\n"
        "3. Return ONLY uppercase keywords separated by space (e.g., 'WELCOME PLEASE SIT ROOM B', 'NAME YOU WHAT').\n"
        "4. Do NOT include markdown fences, punctuation, or explanations."
    )

    try:
        res = generate_llm_response(text, system_instruction=system_instruction)
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
        logger.error(f"LLM simplify failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/llm/answer', methods=['POST'])
def llm_answer():
    """
    Generate an intelligent AI response to a specific user question/statement.
    Uses automatic fallback (Groq -> Gemini -> Smart Local Fallback).
    """
    data = request.get_json() or {}
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'error': 'Missing "text" parameter.'}), 400

    system_instruction = (
        "You are SignBridge AI, a helpful, polite, and direct dual-communication Indian Sign Language assistant. "
        "Your task is to provide a direct, relevant, and natural response to what the user said or asked.\n"
        "Strict rules:\n"
        "1. Directly address the user's input, question, or statement without going off-topic.\n"
        "2. If asked for your name or identity, identify yourself as 'SignBridge AI'.\n"
        "3. NEVER complain or make meta-comments about lack of prior context or instructions.\n"
        "4. If asked to repeat or clarify, politely ask what they would like you to repeat or sign.\n"
        "5. If the user makes a statement, respond pleasantly and engagingly.\n"
        "6. Keep your response concise (1-2 short sentences max), direct, and friendly. Return ONLY the answer without reasoning."
    )

    try:
        res = generate_llm_response(text, system_instruction=system_instruction)
        cleaned_answer = clean_llm_text(res['text'])
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
        'groq_available': groq_client is not None,
        'gemini_available': gemini_manager.is_available(),
        'gemini_health': gemini_manager.get_health_status(),
        'word_recognizer_available': word_recognizer.is_available
    })


@app.route('/api/llm/status', methods=['GET'])
def llm_status():
    """Return real-time status of Groq, Gemini multi-key management, and Local LLM tiers."""
    return jsonify({
        'groq': {
            'available': groq_client is not None,
            'models': active_groq_models[:4]
        },
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
        result = translator.predict(landmarks)

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
        logger.error(f"Translation error: {e}")
        return jsonify({'error': str(e)}), 500


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


@app.route('/api/translate/snapshot', methods=['POST'])
def translate_snapshot():
    """
    On-Demand visual silhouette disambiguation for difficult overlapping contact letters (K, M, N, T, O, S, W).
    Accepts base64 image or image crop with candidate landmarks to verify the visual shape.
    """
    data = request.get_json() or {}
    landmarks = data.get('landmarks')
    image_b64 = data.get('image')

    # Primary pass with ST-GCN / XGBoost
    if landmarks:
        try:
            res = translator.predict(landmarks)
            if res and not res.get('rejected', False) and res.get('confidence', 0) >= 0.65:
                return jsonify(res)
        except Exception:
            pass

    # Fallback to visual silhouette analysis / shape disambiguation
    if image_b64:
        try:
            import base64
            import cv2

            # Clean base64 header if present
            if ',' in image_b64:
                image_b64 = image_b64.split(',', 1)[1]
            img_bytes = base64.b64decode(image_b64)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img is not None:
                # Shape aspect and contour heuristics for contact gestures
                if landmarks:
                    candidate_res = translator.predict(landmarks)
                    scores = candidate_res.get('all_scores', {})
                    if scores:
                        top_candidate = max(scores, key=scores.get)
                        return jsonify({
                            'letter': top_candidate,
                            'confidence': max(0.75, candidate_res.get('confidence', 0.75)),
                            'mode': 'snapshot_disambiguation',
                            'rejected': False,
                            'all_scores': scores
                        })

        except Exception as e:
            logger.warning(f"Snapshot processing error: {e}")

    # Fallback to standard translate if snapshot fails
    if landmarks:
        return translate()
    return jsonify({'error': 'Missing landmarks or image snapshot.'}), 400


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
        result = word_recognizer.predict(frames)
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
        logger.error(f"Word translation error: {e}")
        return jsonify({'error': str(e)}), 500


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
        
    letter = data['letter'].upper()
    session_id = data.get('session_id', 'unknown_session')
    signer_id = data.get('signer_id', 'unknown_signer')
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
        logger.error(f"Failed to save data: {e}")
        return jsonify({'error': str(e)}), 500


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
    logger.info("=" * 60)
    logger.info("  SIGN-BRIDGE FLASK API SERVER")
    logger.info("  http://localhost:5000")
    logger.info("=" * 60)
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )
