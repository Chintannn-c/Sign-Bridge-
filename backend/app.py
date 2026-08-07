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
gemini_model = None
if GOOGLE_API_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GOOGLE_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        logger.info("Google Gemini LLM service initialized successfully from .env!")
    except Exception as e:
        logger.warning(f"Google Gemini LLM initialization warning: {e}")

logger.info(f"Translator mode: {translator.mode}")
logger.info("Services ready.")


# ═══════════════════════════════════════════════════════════════════════════
# LLM AUTOMATIC FALLBACK ENGINE (Groq -> Google Gemini)
# ═══════════════════════════════════════════════════════════════════════════

def generate_llm_response(prompt, system_instruction="You are SignBridge AI assistant."):
    """
    Primary: Groq (Llama-3.3-70b-versatile) for sub-100ms speed.
    Fallback: Google Gemini (1.5-Flash) if Groq exhausts quota or hits rate-limits.
    """
    errors = []

    # 1. Primary Attempt: Groq LPU
    if groq_client:
        try:
            logger.info("Attempting Groq API (Primary LLM)...")
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )
            result = response.choices[0].message.content.strip()
            return {
                "text": result,
                "provider": "Groq (Llama-3.3-70b)",
                "fallback_used": False
            }
        except Exception as groq_err:
            msg = f"Groq primary API error/quota exhausted: {groq_err}"
            logger.warning(msg)
            errors.append(msg)

    # 2. Fallback Attempt: Google Gemini 1.5 Flash
    if gemini_model:
        try:
            logger.info("Falling back to Google Gemini API (Secondary LLM)...")
            full_prompt = f"{system_instruction}\n\nUser Input: {prompt}"
            res = gemini_model.generate_content(full_prompt)
            result = res.text.strip()
            return {
                "text": result,
                "provider": "Google Gemini (1.5-Flash)",
                "fallback_used": True
            }
        except Exception as gemini_err:
            msg = f"Google Gemini fallback API error: {gemini_err}"
            logger.error(msg)
            errors.append(msg)

    raise RuntimeError(f"All LLM APIs failed: {'; '.join(errors)}")


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
    elif any(k in q for k in ['hello', 'hi', 'namaste', 'hey', 'greetings']):
        return "Hello! Namaste. How can I assist you today?"
    elif any(k in q for k in ['how are you', 'how do you do', 'how r u']):
        return "I am doing well, thank you! How can I help you with Indian Sign Language today?"
    elif any(k in q for k in ['name', 'who are you']):
        return "I am SignBridge AI, your dual-communication Indian Sign Language assistant."
    elif any(k in q for k in ['thank', 'thanks']):
        return "You are very welcome! Happy to help."
    elif any(k in q for k in ['help', 'assist', 'support']):
        return "I am here to assist you! You can sign or type your message."
    elif any(k in q for k in ['bye', 'goodbye', 'see you']):
        return "Goodbye! Have a wonderful day ahead."
    elif any(k in q for k in ['morning', 'afternoon', 'evening']):
        return "Good day! How can I help you today?"
    elif any(k in q for k in ['nice to meet you']):
        return "Nice to meet you too! Welcome to SignBridge."
    else:
        return f"I received your question: '{query_text}'. I am here to assist you!"


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
        "You are SignBridge AI, a polite and helpful assistant for an Indian Sign Language (ISL) communication system. "
        "Directly answer the user's specific question or respond naturally to their input. "
        "Keep your response concise (1-2 short sentences max), clear, and direct."
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
        'gemini_available': gemini_model is not None,
        'word_recognizer_available': word_recognizer.is_available
    })


@app.route('/api/model/info', methods=['GET'])
def model_info():
    """Return model metadata."""
    return jsonify(translator.get_info())


# ═══════════════════════════════════════════════════════════════════════════
# TRANSLATION ENDPOINTS (Path 1: Signs -> Text)
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/api/translate', methods=['POST'])
def translate():
    """
    Translate hand landmarks to an ISL letter/word.

    Request JSON body:
    {
        "landmarks": [
            // 42 landmarks as [{x, y, z}, ...] or flat [x1,y1,z1, x2,y2,z2, ...]
        ]
    }

    Response:
    {
        "letter": "A",
        "confidence": 0.95,
        "mode": "deep_learning" | "heuristic",
        "all_scores": {"A": 0.95, "B": 0.02, ...}
    }
    """
    data = request.get_json()
    if not data or 'landmarks' not in data:
        return jsonify({
            'error': 'Missing "landmarks" field in request body.',
            'expected': 'Array of 42 landmark points (each with x,y,z) or flat array of 126 floats.'
        }), 400

    landmarks = data['landmarks']

    try:
        result = translator.predict(landmarks)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/translate/batch', methods=['POST'])
def translate_batch():
    """
    Translate a batch of landmark frames (for sentence-level recognition).

    Request JSON body:
    {
        "frames": [
            [landmarks_frame_1],
            [landmarks_frame_2],
            ...
        ]
    }

    Response:
    {
        "results": [
            {"letter": "H", "confidence": 0.92},
            {"letter": "E", "confidence": 0.88},
            ...
        ],
        "sentence": "HE..."
    }
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
    Uses the Bi-LSTM temporal word recognizer.

    Request JSON body:
    {
        "frames": [
            [x1,y1,z1, ...],  // Frame 1: 126 landmark floats
            [x1,y1,z1, ...],  // Frame 2: 126 landmark floats
            ...                // Up to 30 frames
        ]
    }

    Response:
    {
        "word": "HELLO",
        "confidence": 0.93,
        "all_scores": {"HELLO": 0.93, "NAMASTE": 0.04, ...}
    }
    """
    if not word_recognizer.is_available:
        return jsonify({
            'error': 'Word recognizer is not available. Train the Bi-LSTM model first.',
            'hint': 'Run: cd backend && python train_model_lstm.py'
        }), 503

    data = request.get_json()
    if not data or 'frames' not in data:
        return jsonify({'error': 'Missing "frames" field.'}), 400

    frames = data['frames']
    try:
        result = word_recognizer.predict(frames)
        if result is None:
            return jsonify({'error': 'Word prediction failed or input invalid.'}), 400
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
