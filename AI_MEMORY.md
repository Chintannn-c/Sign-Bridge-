# AI_MEMORY.md

## Architecture Decisions
1. **Two-Handed ISL Requirement vs ASL**: Unlike American Sign Language (ASL) which is primarily single-handed, Indian Sign Language (ISL) is predominantly **two-handed**. Thus, both software vision (MediaPipe 42 landmarks) and hardware robotic hands (2 hands $\times$ 5 servos = 10 servos driven by Arduino Mega/Uno) are engineered for dual-hand processing.
2. **Dual-Provider LLM Fallback (Groq Primary -> Gemini Secondary)**: To ensure sub-100ms real-time text refinement without breaking user chat flows, Groq (`llama-3.3-70b-versatile`) is called first. If Groq encounters rate limits or missing API keys, the system seamlessly falls back to Google Gemini (`gemini-1.5-flash`).
3. **Single Local Database Choice (SQLite)**: If persistent storage is needed for logs, custom gesture calibration, or dataset management, the project uses a single **SQLite** database file (`backend/signbridge.db`). Vector databases are explicitly excluded as standard ML classifiers and LLMs handle real-time inference faster without vector DB overhead.
4. **Vite Proxy for Flask Backend**: Frontend calls relative endpoints like `/api/translate` which Vite proxies to `http://localhost:5000/api`. This avoids CORS headaches during development.
5. **3-Tier ML Model Architecture**: ISL recognition uses a 3-tier pipeline — (a) **XGBoost** for single-frame alphabet classification (highest accuracy, fastest CPU inference), (b) **Bi-LSTM** for multi-frame temporal word recognition (30 frames = ~1 second of movement), (c) **Groq/Gemini LLM** for sentence-level refinement. This separation keeps each tier independently trainable and upgradeable.
6. **Model Loading Priority (XGBoost → Keras → Heuristic)**: `translator_model.py` loads models in strict priority order: XGBoost `.pkl` first, Keras `.h5` second, heuristic fallback third. This ensures the best-available model is always used without manual switching.

---

## Design Patterns
- **Custom React Hooks Pattern**: Core logic and state management are encapsulated in domain hooks (`useISLTranslation`, `useGestureRecognition`, `useHandDetection`, `useAssistantHistory`).
- **Service Layer Pattern (Flask Backend)**: Business logic is decoupled into services (`TranslatorModel` for ML inference, `ArduinoSerial` for serial hardware I/O).
- **Dual-Mode Graceful Fallback**: Components fall back gracefully to **DEMO mode** (simulated typewriter text feeds and pre-recorded pools) if Flask or Arduino hardware is disconnected.
- **Provider Strategy / Automatic Fallback**: `generate_llm_response()` tries primary provider (Groq) first, catches exceptions, logs warnings, and delegates to fallback provider (Gemini).

---

## Important Assumptions
- **MediaPipe Landmark Indexing**: Standard 42 landmark array consists of Left Hand 21 points + Right Hand 21 points $(x, y, z)$.
- **Serial Communication Protocol**: Arduino expects ASCII servo angle frames enclosed in angle brackets `<L1,L2,L3,L4,L5|R1,R2,R3,R4,R5>`.
- **Operating System**: Windows OS host running Python 3 and Node.js with Arduino attached via USB (e.g., `COM3` or auto-detected COM port).

---

## Constraints
- **Real-Time Video Latency**: Vision pipeline must maintain 30 FPS. Processing per frame must stay under **33ms**.
- **Scrap-Material Hardware Limits**: Servos (SG90 micro servos) have mechanical limitations. Rapid 0° to 180° jerks can cause tension line tear on cereal-box cardboard / plastic bottle hand structures; smooth stepping delays are implemented.
- **No Vector DB Overhead**: Vector databases (ChromaDB, FAISS) must NOT be introduced for standard gesture recognition or simple text generation as they add unnecessary latency and dependency bloat.

---

## Coding Conventions
- **Frontend**:
  - React 19 functional components with standard React Hooks.
  - Styling via Vanilla CSS (`index.css` & `assistant.css`) using custom CSS variables (`--bg-primary`, `--accent-color`, glassmorphism styles).
  - No TailwindCSS unless explicitly directed.
- **Backend**:
  - Python 3 with Flask standard REST practices.
  - Logging via standard `logging` module (`sign-bridge-api` logger).
  - Environment variables loaded from `backend/.env` using `python-dotenv`.
- **API Responses**: Standard JSON structure `{ status: "ok" | "error", ... }`.

---

## Things the AI Should Remember
- **Do Not Break Fallback Mechanism**: Always preserve the dual LIVE / DEMO mode capability in `useISLTranslation.js`. If the Flask API or Arduino is offline, the UI must function seamlessly in demo mode.
- **Do Not Modify Hardware Angle Maps Without Testing**: `SERVO_MAPPINGS` dictionary maps ISL letters to physical motor angles. Altering these values affects physical tendon tension.
- **Proxy Configuration**: Backend API routes should always be requested as relative paths `/api/...` in frontend hooks so Vite proxy redirects correctly.
- **Model Priority Order Is Sacred**: `translator_model.py` loads XGBoost → Keras → Heuristic in strict order. Never change this priority without explicit user direction. All three paths must remain functional.
- **WordRecognizer Graceful Degradation**: `word_recognizer.py` must always return `None` when no LSTM model file exists — it must never crash the server or affect the alphabet recognition pipeline.

---

## Common Pitfalls
- **MediaPipe Missing Landmarks**: When hands leave camera frame, landmark array may be `null` or incomplete. Frontend hooks must sanitize inputs before calling `/api/translate`.
- **Serial Port Lock (Windows `PermissionError`)**: PySerial holds an exclusive lock on `COM3`. If another script or Arduino Serial Monitor is open, `ArduinoSerial.connect()` will fail.
- **LLM Rate Limits / Missing Keys**: Always handle `None` return values or exceptions from `generate_llm_response()`.

---

## Frequently Edited Files
- [useISLTranslation.js](file:///c:/React/SignBridge/src/hooks/useISLTranslation.js): Manages live feeds, live buffer text, streaming UI messages, and Flask API connectivity.
- [app.py](file:///c:/React/SignBridge/backend/app.py): Flask application endpoints, LLM fallback engine, and service orchestrator.
- [translator_model.py](file:///c:/React/SignBridge/backend/services/translator_model.py): Gesture recognition ML inference model (XGBoost → Keras → Heuristic) and landmark distance heuristics.
- [word_recognizer.py](file:///c:/React/SignBridge/backend/services/word_recognizer.py): Bi-LSTM temporal word recognition service.
- [arduino_serial.py](file:///c:/React/SignBridge/backend/services/arduino_serial.py): PySerial driver managing USB communication and servo mapping.
- [CameraView.jsx](file:///c:/React/SignBridge/src/components/CameraView.jsx): Live webcam view component rendering landmark canvas overlays.

---

## Project Glossary
- **ISL**: Indian Sign Language (predominantly two-handed gesture manual alphabet and vocabulary).
- **Path 1**: Sign-to-Text/Speech input path (Deaf signer $\rightarrow$ Webcam $\rightarrow$ MediaPipe $\rightarrow$ ML Model $\rightarrow$ Text/TTS).
- **Path 2**: Text/Speech-to-Robot output path (Hearing speaker $\rightarrow$ LLM simplify $\rightarrow$ PySerial $\rightarrow$ Arduino $\rightarrow$ Robot Hands).
- **Landmark**: A 3D coordinate point $(x, y, z)$ output by MediaPipe representing a hand joint.
- **Groq LPU**: High-speed inference hardware powering Llama 3.3 for sub-100ms LLM text refinement.

---

## AI Notes
- **Future Database Integration**: If database features are requested, implement **SQLite** via `sqlite3` or `Flask-SQLAlchemy` stored at `backend/signbridge.db`.
- **Keep Documentation Updated**: Whenever new API routes, ML models, or hardware commands are added, update `PROJECT_CONTEXT.md` and `AI_MEMORY.md`.
