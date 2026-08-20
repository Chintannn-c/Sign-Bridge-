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
import logging
import numpy as np
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

# ─── Configuration ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger('sign-bridge-api')

app = Flask(__name__)
CORS(app, origins=['http://localhost:5173', 'http://localhost:3000', 'http://127.0.0.1:5173'])

# ─── Initialize Services ───────────────────────────────────────────────────
logger.info("Initializing Sign-Bridge API services...")
translator = TranslatorModel()
word_recognizer = WordRecognizer()
arduino = ArduinoSerial()

# Initialize Groq Real-Time LLM Client from .env
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
groq_client = None
if GROQ_API_KEY:
    try:
        from groq import Groq
        groq_client = Groq(api_key=GROQ_API_KEY)
        logger.info("Groq Real-Time LLM service initialized successfully from .env!")
    except Exception as e:
        logger.warning(f"Groq LLM initialization warning: {e}")

# Initialize Google Gemini LLM Client from .env
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
gemini_client = None
if GOOGLE_API_KEY:
    try:
        # Try modern google-genai SDK first
        from google import genai
        gemini_client = genai.Client(api_key=GOOGLE_API_KEY)
        logger.info("Google GenAI (Gemini 2.0/2.5) service initialized successfully from .env!")
    except Exception as genai_err:
        try:
            # Fallback to legacy google.generativeai SDK
            import google.generativeai as legacy_genai
            legacy_genai.configure(api_key=GOOGLE_API_KEY)
            gemini_client = legacy_genai.GenerativeModel('gemini-1.5-flash')
            logger.info("Google Gemini (1.5-Flash) service initialized successfully from .env!")
        except Exception as e:
            logger.warning(f"Google Gemini LLM initialization warning: {e}")

logger.info(f"Translator mode: {translator.mode}")
logger.info("Services ready.")


# ═══════════════════════════════════════════════════════════════════════════
# LLM AUTOMATIC FALLBACK ENGINE (Groq -> Google Gemini -> Local)
# ═══════════════════════════════════════════════════════════════════════════

GROQ_MODELS = [
    "groq/compound",
    "groq/compound-mini",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",
]

GEMINI_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-1.5-flash",
]

def generate_llm_response(prompt, system_instruction="You are SignBridge AI assistant."):
    """
    Primary: Groq (groq/compound / openai/gpt-oss-120b) for sub-100ms speed.
    Fallback 1: Google Gemini (2.0/2.5 Flash) via google-genai.
    Fallback 2: Smart Local Semantic Engine.
    """
    errors = []

    # 1. Primary Attempt: Groq LPU (iterates through available active models)
    if groq_client:
        for model_id in GROQ_MODELS:
            try:
                response = groq_client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=300
                )
                result = response.choices[0].message.content.strip()
                if result:
                    return {
                        "text": result,
                        "provider": f"Groq ({model_id})",
                        "fallback_used": False
                    }
            except Exception as groq_err:
                errors.append(f"Groq [{model_id}] error: {groq_err}")
                continue

    # 2. Fallback Attempt: Google Gemini (2.0 / 2.5 Flash)
    if gemini_client:
        for gem_model in GEMINI_MODELS:
            try:
                # If using modern google-genai Client
                if hasattr(gemini_client, 'models') and hasattr(gemini_client.models, 'generate_content'):
                    response = gemini_client.models.generate_content(
                        model=gem_model,
                        contents=f"{system_instruction}\n\nUser Input: {prompt}"
                    )
                    text_out = response.text.strip()
                    if text_out:
                        return {
                            "text": text_out,
                            "provider": f"Google Gemini ({gem_model})",
                            "fallback_used": True
                        }
                # If using legacy GenerativeModel
                elif hasattr(gemini_client, 'generate_content'):
                    res = gemini_client.generate_content(f"{system_instruction}\n\nUser Input: {prompt}")
                    text_out = res.text.strip()
                    if text_out:
                        return {
                            "text": text_out,
                            "provider": "Google Gemini (1.5-Flash)",
                            "fallback_used": True
                        }
            except Exception as gem_err:
                errors.append(f"Gemini [{gem_model}] error: {gem_err}")
                continue

    # 3. Fallback Attempt: Smart Local Keyword Matcher
    logger.warning(f"All external LLMs failed. Errors: {'; '.join(errors)}")
    fallback_text = get_smart_fallback_response(prompt)
    return {
        "text": fallback_text,
        "provider": "Local Semantic Engine",
        "fallback_used": True
    }


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
        "You are SignBridge AI. Convert raw ISL fingerspelled letter fragments or keywords "
        "into a single fluent, natural conversational sentence. Keep response short, helpful, and concise."
    )

    try:
        res = generate_llm_response(text, system_instruction=system_instruction)
        return jsonify({
            'raw_text': text,
            'refined_sentence': res['text'],
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
        "You are SignBridge AI. Extract the essential ISL keywords from the user's speech "
        "so robotic hands can sign them quickly. Return ONLY uppercase keywords separated by space (e.g. 'WELCOME SIT ROOM B')."
    )

    try:
        res = generate_llm_response(text, system_instruction=system_instruction)
        return jsonify({
            'original_speech': text,
            'robot_keywords': res['text'],
            'llm_provider': res['provider'],
            'fallback_used': res['fallback_used']
        })
    except Exception as e:
        logger.error(f"LLM simplify failed: {e}")
        return jsonify({'error': str(e)}), 500


def get_smart_fallback_response(query_text):
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
        "3. NEVER complain or make meta-comments about lack of prior context or instructions (e.g. never say 'You didn't provide instructions' or 'I don't have context').\n"
        "4. If asked to repeat or clarify, politely ask what they would like you to repeat or sign.\n"
        "5. If the user makes a statement (e.g. 'I want ice cream'), respond pleasantly and engagingly (e.g. 'Ice cream sounds great! What flavor would you like?').\n"
        "6. Keep your response concise (1-2 short sentences max), direct, and friendly."
    )

    try:
        res = generate_llm_response(text, system_instruction=system_instruction)
        return jsonify({
            'user_text': text,
            'answer': res['text'],
            'llm_provider': res['provider'],
            'fallback_used': res['fallback_used']
        })
    except Exception as e:
        logger.warning(f"LLM answer API error (using smart local fallback): {e}")
        fallback_ans = get_smart_fallback_response(text)
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
        'gemini_available': gemini_client is not None,
        'word_recognizer_available': word_recognizer.is_available
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

        # Confidence rejection: preserve calibrated rejection decision from translator
        is_rej = result.get('rejected', False) or result.get('confidence', 0) < 0.50 or result.get('letter') == '?'
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
            if result.get('confidence', 0) > 0.5:
                sentence_letters.append(result.get('letter', '?'))
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
        result = word_recognizer.predict(frames)
        if result is None:
            return jsonify({'error': 'Word prediction failed or input invalid.'}), 400

        # Confidence rejection for words
        confidence = result.get('confidence', 0)
        if confidence < 0.60 or result.get('word') == '?':
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
    frames = data['frames']
    
    # Save directory
    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset_collected', letter)
    os.makedirs(dataset_dir, exist_ok=True)
    
    file_path = os.path.join(dataset_dir, f"{session_id}.json")
    
    try:
        with open(file_path, 'w') as f:
            json.dump({'letter': letter, 'session_id': session_id, 'frames': frames}, f)
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

    signed = arduino.sign_text(text, letter_hold=letter_hold, gap=gap)
    return jsonify({
        'text': text,
        'signed_letters': signed,
        'arduino_connected': True,
        'message': f'Successfully signed {len(signed)} letters on robotic hands.'
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
