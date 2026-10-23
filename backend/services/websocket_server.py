"""
Sign-Bridge High-Performance WebSocket Prediction Service.

Provides a persistent bidirectional pipe for real-time ISL landmark predictions:
  - 'letter': Static 126-float landmark frame -> instant letter prediction
  - 'word': Temporal 30-frame sequence -> BiLSTM word prediction

Runs concurrently alongside Flask / Waitress WSGI on a dedicated port (default: 5001).
Zero HTTP request/header overhead — sub-2ms per frame latency.
"""

import asyncio
import json
import logging
import threading
import numpy as np
import websockets

try:
    from .gesture_config import CONFIDENCE_THRESHOLD, CONFIDENCE_THRESHOLD_WORD, MARGIN_THRESHOLD
    from .gesture_pipeline import is_idle_hand, is_neutral_gesture
except ImportError:
    from services.gesture_config import CONFIDENCE_THRESHOLD, CONFIDENCE_THRESHOLD_WORD, MARGIN_THRESHOLD
    from services.gesture_pipeline import is_idle_hand, is_neutral_gesture

logger = logging.getLogger("SignBridge.WS")


def _safe_float(val, default: float = 0.0) -> float:
    try:
        f = float(val)
        return default if np.isnan(f) or np.isinf(f) else f
    except (ValueError, TypeError):
        return default


def handle_letter_prediction(translator, data, model_lock=None):
    landmarks = data.get('landmarks')
    if not landmarks:
        return {
            'type': 'letter',
            'letter': '?',
            'confidence': 0.0,
            'rejected': True,
            'rejection_reason': 'missing_landmarks',
            'all_scores': {},
        }

    try:
        arr = np.asarray(landmarks, dtype=np.float32)
        if arr.size == 0 or not np.any(arr != 0) or np.all(np.abs(arr) < 1e-4) or is_idle_hand(arr):
            return {
                'type': 'letter',
                'letter': '?',
                'confidence': 0.0,
                'rejected': True,
                'rejection_reason': 'idle_or_no_hands',
                'all_scores': {},
            }
        if is_neutral_gesture(arr):
            return {
                'type': 'letter',
                'letter': '?',
                'confidence': 0.0,
                'is_neutral': True,
                'rejected': True,
                'rejection_reason': 'neutral_base_gesture',
                'all_scores': {},
            }
    except Exception:
        return {
            'type': 'letter',
            'letter': '?',
            'confidence': 0.0,
            'rejected': True,
            'rejection_reason': 'invalid_payload',
            'all_scores': {},
        }

    body_anchors = data.get('body_anchors')
    if model_lock:
        with model_lock:
            result = translator.predict(landmarks, body_anchors=body_anchors)
    else:
        result = translator.predict(landmarks, body_anchors=body_anchors)

    conf = _safe_float(result.get('confidence', 0.0))
    is_rej = bool(result.get('rejected', False)) or (conf < CONFIDENCE_THRESHOLD) or (str(result.get('letter', '')) == '?')
    result['rejected'] = is_rej
    if is_rej:
        result['rejection_reason'] = result.get('rejection_reason') or 'low_confidence'
        result['original_letter'] = result.get('letter', '?')
        result['letter'] = '?'
    result['type'] = 'letter'
    return result


def handle_word_prediction(word_recognizer, data, model_lock=None):
    if not word_recognizer or not getattr(word_recognizer, 'is_available', False):
        return {
            'type': 'word',
            'word': '?',
            'confidence': 0.0,
            'rejected': True,
            'rejection_reason': 'word_recognizer_unavailable',
            'all_scores': {},
        }

    frames = data.get('frames')
    if not frames or not isinstance(frames, list) or len(frames) < 10:
        return {
            'type': 'word',
            'word': '?',
            'confidence': 0.0,
            'rejected': True,
            'rejection_reason': 'insufficient_frames',
            'all_scores': {},
        }

    try:
        arr = np.asarray(frames, dtype=np.float32)
        if not np.any(arr != 0):
            return {
                'type': 'word',
                'word': '?',
                'confidence': 0.0,
                'rejected': True,
                'rejection_reason': 'no_hands_detected',
                'all_scores': {},
            }
    except Exception:
        return {
            'type': 'word',
            'word': '?',
            'confidence': 0.0,
            'rejected': True,
            'rejection_reason': 'invalid_payload',
            'all_scores': {},
        }

    body_anchors = data.get('body_anchors')
    if model_lock:
        with model_lock:
            result = word_recognizer.predict(frames, body_anchors=body_anchors)
    else:
        result = word_recognizer.predict(frames, body_anchors=body_anchors)

    if not result:
        return {
            'type': 'word',
            'word': '?',
            'confidence': 0.0,
            'rejected': True,
            'rejection_reason': 'prediction_failed',
            'all_scores': {},
        }

    conf = _safe_float(result.get('confidence', 0.0))
    is_rej = bool(result.get('rejected', False)) or (conf < CONFIDENCE_THRESHOLD_WORD) or (str(result.get('word', '')) == '?')
    result['rejected'] = is_rej
    if is_rej:
        result['rejection_reason'] = result.get('rejection_reason') or 'low_confidence'
        result['original_word'] = result.get('word', '?')
        result['word'] = '?'
    result['type'] = 'word'
    return result


async def ws_handler(websocket, translator, word_recognizer, model_lock=None):
    try:
        async for raw_message in websocket:
            try:
                data = json.loads(raw_message)
                msg_type = data.get('type', 'letter')
                req_id = data.get('req_id')

                if msg_type == 'word':
                    res = handle_word_prediction(word_recognizer, data, model_lock)
                else:
                    res = handle_letter_prediction(translator, data, model_lock)

                if req_id is not None:
                    res['req_id'] = req_id

                await websocket.send(json.dumps(res))
            except Exception as err:
                logger.error(f"WebSocket message error: {err}")
                try:
                    await websocket.send(json.dumps({'error': str(err), 'rejected': True}))
                except Exception:
                    pass
    except websockets.ConnectionClosed:
        pass  # Normal client disconnect (page reload, HMR, or tab close)


def start_websocket_server(translator, word_recognizer, host='127.0.0.1', port=5001, model_lock=None):
    """
    Start the WebSocket server in a background daemon thread.
    """
    def run_loop():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def serve_ws():
            async with websockets.serve(
                lambda ws: ws_handler(ws, translator, word_recognizer, model_lock),
                host,
                port,
                ping_interval=20,
                ping_timeout=20,
                max_size=10 * 1024 * 1024,
            ):
                logger.info(f"WebSocket real-time prediction service running on ws://{host}:{port}")
                await asyncio.Future()

        try:
            loop.run_until_complete(serve_ws())
        except Exception as e:
            logger.error(f"WebSocket server terminated: {e}")

    thread = threading.Thread(target=run_loop, daemon=True, name="SignBridge-WS-Thread")
    thread.start()
    return thread
