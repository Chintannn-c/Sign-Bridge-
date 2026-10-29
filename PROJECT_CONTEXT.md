# PROJECT_CONTEXT.md — SignBridge System Context & Developer Onboarding Guide

## 1. Project Overview

### 1.1 Purpose & Problem Statement
[SignBridge](file:///c:/React/SignBridge) is an assistive, bidirectional Indian Sign Language (ISL) communication kiosk and robotic actuation system. It bridges the communication divide between Deaf / Hard-of-Hearing (DHH) individuals and hearing individuals.

Unlike American Sign Language (ASL), which is largely single-handed, **Indian Sign Language (ISL)** is predominantly **two-handed**, possesses its own Subject-Object-Verb (SOV) grammatical structure, and places question markers at the end of phrases. SignBridge provides two simultaneous, real-time communication pathways:

1. **Deaf $\rightarrow$ Hearing (Path 1 — Perception & Speech Synthesis)**:
   - Captures continuous video via a standard webcam at 30 FPS.
   - Extracts 42 3D keypoints (126 coordinates) using MediaPipe Dual-Hand / Holistic tracking.
   - Recognizes static fingerspelling alphabets (`A`–`Z`) and digits (`0`–`9`) using a Calibrated Hybrid Ensemble ([ST-GCN](file:///c:/React/SignBridge/backend/models/isl_stgcn_model.pt) + [XGBoost](file:///c:/React/SignBridge/backend/models/isl_xgboost_model.pkl)) operating on 208-D geometric invariant feature vectors.
   - Recognizes dynamic whole-word temporal gestures across 57 classes using a 1D-CNN + BiLSTM neural network ([isl_cnn_lstm_word_model.pt](file:///c:/React/SignBridge/backend/models/isl_cnn_lstm_word_model.pt)) with wrist-relative motion velocity gating.
   - Reconstructs and refines raw recognized glosses into fluent spoken English sentences using an automated dual-provider LLM cascade ([Groq](file:///c:/React/SignBridge/backend/services/groq_manager.py) `llama-3.3-70b-versatile` $\rightarrow$ [Google Gemini](file:///c:/React/SignBridge/backend/services/gemini_manager.py) `gemini-1.5-flash` $\rightarrow$ local rule-based fallback).
   - Speaks the refined sentence out loud using the browser's native Web Speech Synthesis API.
   - Features a **Touchless Hand-Drop Auto-Send**: when hands drop out of the camera view for 1.8 seconds, the accumulated sentence buffer is automatically refined, spoken, and committed to the chat history.

2. **Hearing $\rightarrow$ Deaf (Path 2 — Speech Transcription & Robotic Actuation)**:
   - Captures the hearing person's spoken voice via browser Web Speech Recognition (`webkitSpeechRecognition`).
   - Simplifies complex English sentences into concise ISL robot signing keywords using the LLM cascade.
   - Sends calibrated 10-servo PWM angle arrays over USB PySerial ([arduino_serial.py](file:///c:/React/SignBridge/backend/services/arduino_serial.py)) to an Arduino Mega / Uno microcontroller controlling **dual 5-finger bionic robotic hands** (10x SG90 micro-servos).
   - Displays animated visual reference sheets ([GestureReferenceSheet.jsx](file:///c:/React/SignBridge/src/components/GestureReferenceSheet.jsx)) and SVG tendon diagrams for immediate visual reinforcement.

### 1.2 Current Maturity
- **Status**: **In Active Development / Feature-Complete Working Prototype**.
- **Perception & Vision**: Production-quality live tracking in React 19, sub-2ms WebSocket inference pipe, One-Euro adaptive landmark jitter smoothing, and 63/63 passing backend regression tests.
- **Machine Learning**: 87.43% test accuracy on static alphabet XGBoost; 82.03% on ST-GCN graph; 51.22% validation accuracy on 57-class temporal CNN-BiLSTM (model trained on augmented ISLRTC national dictionary and session captures; expanding vocabulary remains an active development goal).
- **Hardware Integration**: Functional PySerial driver with background queued worker thread, auto-idle 15s servo relaxer, and software fallback when Arduino is disconnected.
- **Packaging**: Containerized via multi-stage Dockerfiles ([Dockerfile.backend](file:///c:/React/SignBridge/Dockerfile.backend), [Dockerfile.frontend](file:///c:/React/SignBridge/Dockerfile.frontend)), Docker Compose ([docker-compose.yml](file:///c:/React/SignBridge/docker-compose.yml)), and Kubernetes manifests ([k8s/](file:///c:/React/SignBridge/k8s/)).

### 1.3 Annotated Repository Layout

```
SignBridge/
├── .agents/                        # AI agent configurations, skills, and workflows (Graphify, Ponytail)
├── backend/                        # Python Flask + Waitress + PyTorch REST and WebSocket server
│   ├── app.py                      # Master Flask application server (19 REST endpoints, WSGI entry point)
│   ├── requirements.txt            # Python dependencies (Flask, PyTorch, XGBoost, MediaPipe, etc.)
│   ├── .env.example                # Template for environment variables and LLM API keys
│   ├── train_unified.py            # Unified training pipeline v2 (extracts landmarks, trains XGBoost, ST-GCN, BiLSTM)
│   ├── benchmark_alphabet_models.py# Standalone CLI evaluator benchmarking XGBoost vs ST-GCN vs Hybrid
│   ├── extract_static_landmarks.py # Batch landmark extractor for static image datasets (RealSign, Mendeley)
│   ├── extract_video_landmarks.py  # Batch landmark extractor for 30-frame temporal word videos
│   ├── stream_and_extract_hf.py    # Zero-disk streaming extractor for Hugging Face ISLRTC dictionary
│   ├── ingest_dataset_2.py         # Batch ingest extractor for 521 high-res real-world photos in dataset_2/
│   ├── generate_centroids.py       # Computes per-class geometric centroids for out-of-dataset rejection
│   ├── generate_idle_dataset.py    # Generates synthetic negative/idle sequence samples for word recognizer
│   ├── test_word_model.py          # Quick CLI inference test runner for BiLSTM word models
│   ├── database/                   # SQLite persistence layer
│   │   ├── schema.py               # Table definitions (conversation_history, dataset_sessions) and queries
│   │   └── signbridge.db           # Persistent SQLite database file (WAL mode)
│   ├── models/                     # Trained ML weights, MediaPipe tasks, and evaluation metadata
│   │   ├── isl_xgboost_model.pkl   # XGBoost 26-class letter classifier (28.5 MB, 87.43% test accuracy)
│   │   ├── isl_stgcn_model.pt      # ST-GCN 26-class kinematic graph model (199 KB, 82.03% test accuracy)
│   │   ├── isl_cnn_lstm_word_model.pt # 1D-CNN + BiLSTM 57-class temporal word model (3.1 MB)
│   │   ├── isl_word_classifier.pkl # Random Forest 57-class sequence model fallback (23.3 MB)
│   │   ├── hand_landmarker.task    # Google MediaPipe hand landmark task binary (7.8 MB)
│   │   ├── letter_centroids.json   # Per-class 98th percentile Euclidean distance centroids
│   │   ├── st_gcn.py               # PyTorch ST-GCN graph neural network definition
│   │   ├── xgb_training_meta.json  # Evaluation metrics and per-class F1 breakdown for XGBoost
│   │   ├── stgcn_training_meta.json# Evaluation metrics for ST-GCN
│   │   ├── hybrid_training_meta.json# Evaluation metrics for Calibrated Hybrid Ensemble
│   │   └── cnn_lstm_training_meta.json # Evaluation metrics for 57-class CNN-BiLSTM
│   ├── services/                   # Modular backend service layer
│   │   ├── translator_model.py     # Hybrid Alphabet Translation Engine (XGBoost ➔ ST-GCN ➔ Keras ➔ Heuristic)
│   │   ├── word_recognizer.py      # Dynamic 30-frame CNN-BiLSTM Temporal Word Recognition Service
│   │   ├── feature_extractor.py    # 208-D scale/rotation/translation geometric invariant feature extractor
│   │   ├── gesture_pipeline.py     # Centroid validation, idle-hand detector, and base neutral gesture filter
│   │   ├── gesture_state_machine.py# Fast-commit gesture state machine (NEUTRAL ➔ ARMED ➔ CAPTURING ➔ COOLDOWN)
│   │   ├── gesture_config.py       # Threshold constants (CONFIDENCE_THRESHOLD, MARGIN_THRESHOLD, timing)
│   │   ├── groq_manager.py         # Resilient Groq LPU Manager (dual-key rotation, auto-discovery, backoff retry)
│   │   ├── gemini_manager.py       # Resilient Google Gemini Manager (dual-key rotation, capability matching)
│   │   ├── arduino_serial.py       # PySerial driver for 10-servo robotic hands with threaded worker queue
│   │   ├── websocket_server.py     # Low-latency WebSocket prediction server (ws://127.0.0.1:5001)
│   │   ├── landmark_utils.py       # Vectorized landmark normalization, canonical hand slotting, augmentation
│   │   ├── data_loader.py          # Leakage-free train/val/test partition manager with SHA-256 deduplication
│   │   └── robot_actuator.py       # Abstract base robot actuator and ROS2 middleware adapter stub
│   ├── tests/                      # Pytest backend test suite (63 passing unit tests)
│   │   ├── test_app_llm.py         # Flask REST endpoint validation, payload clamping, and LLM text cleaner
│   │   ├── test_gemini_manager.py  # Gemini multi-key failover, error classification, and backoff tests
│   │   ├── test_groq_manager.py    # Groq multi-key failover and model discovery tests
│   │   ├── test_gesture_pipeline.py# Centroid distance rejection and idle hand detection tests
│   │   ├── test_gesture_state_machine.py # State transitions, instant commit, and re-arm tests
│   │   ├── test_live_bug_fixes.py  # Regressions for letter H slot ordering and NAMASTE idle gating
│   │   └── test_slot_ordering.py   # Canonical hand slot assignment tests under coordinate flips
│   ├── dataset_collected/          # Partitioned JSON landmark files for alphabets (RealSign, session captures)
│   └── dataset_words/              # 30-frame temporal landmark sequences for ISL words and synthetic idle NONE
├── src/                            # React 19 Frontend Application (Vite 8)
│   ├── main.jsx                    # React entry point mounting App to DOM
│   ├── App.jsx                     # Top-level coordinator rendering ErrorBoundary, Kiosk, and SplashScreen
│   ├── index.css                   # Global Vanilla CSS design system, kiosk layout, variables, responsive styles
│   ├── components/                 # React visual components
│   │   ├── SignBridgeKiosk.jsx     # Master dual-card split-screen kiosk container and keyboard shortcuts
│   │   ├── HumanPanel.jsx          # Left panel: Deaf user video feed, live sentence buffer, font autoscaling
│   │   ├── RobotPanel.jsx          # Right panel: Hearing user chat thread, Web Speech STT, reference drawer
│   │   ├── CameraView.jsx          # Live camera feed, fullscreen toggle, MediaPipe overlay, HUD indicators
│   │   ├── GestureReferenceSheet.jsx # Interactive A–Z gesture cards, SVG finger diagrams, servo angle viewer
│   │   ├── SplashScreen.jsx        # 3.2s cinematic animated onboarding splash screen with wave canvas
│   │   ├── SplashScreen.css        # Glassmorphism, aurora glows, and animated branding styles
│   │   ├── ErrorBoundary.jsx       # Global React error boundary catching render crashes
│   │   └── SignLanguageAssistant/  # Expandable bottom sheet assistant modal
│   │       ├── AssistantBottomSheet.jsx # Bottom sheet container with spring drag-to-dismiss gesture
│   │       ├── HandTrackingOverlay.jsx  # Canvas-based 42-joint hand skeleton overlay
│   │       ├── SentenceBuilder.jsx      # Tag-based interactive sentence builder
│   │       ├── assistant.css            # Bottom sheet styles
│   │       └── tabs/
│   │           ├── LiveDetectionTab.jsx # Real-time probability inspector and top-3 confidence bars
│   │           ├── PracticeTab.jsx      # Gesture practice sandbox with real-time feedback
│   │           ├── LearnTab.jsx         # ISL vocabulary study cards
│   │           ├── HistoryTab.jsx       # Historical chat conversations
│   │           └── DataCollectionTab.jsx# In-browser user gesture recorder saving to /api/collect_data
│   ├── hooks/                      # Custom React hooks encapsulating business logic
│   │   ├── useISLTranslation.js    # Kiosk state, message dialogue stream, typewriter effects, API health
│   │   ├── useGestureRecognition.js# 1500+ line master recognition hook: ring buffer, state machine, WebSocket
│   │   ├── useHandDetection.js     # MediaPipe Holistic/Hands CDN lifecycle, 42-landmark extraction, slot sorting
│   │   ├── useWebcam.js            # MediaDevices camera acquisition, mirror toggle, resolution constraints
│   │   └── useAssistantHistory.js  # Assistant history state persisted to localStorage
│   ├── config/                     # Frontend configuration constants
│   │   ├── gestureConfig.js        # Frontend gesture recognition constants mirroring backend config
│   │   └── splashConfig.js         # Splash screen duration and prefers-reduced-motion timings
│   ├── data/                       # Static dataset definitions and conversational dialogue
│   │   ├── gestureData.js          # ISL letter descriptions, steps, emoji symbols, and YouTube links
│   │   └── islConversations.js     # Pre-scripted conversational scenarios and random dialogue pools
│   └── utils/                      # Helper utilities
│       └── oneEuroFilter.js        # Adaptive speed-based One-Euro low-pass filter for 42 hand joints
├── tests/                          # Frontend unit tests
│   └── splash.test.mjs             # Node test runner suite verifying splash timings and screen transitions
├── k8s/                            # Kubernetes deployment manifests
│   ├── backend-deployment.yaml     # Flask API deployment with readiness/liveness probes and ClusterIP service
│   ├── frontend-deployment.yaml    # React/Nginx deployment and ClusterIP service
│   ├── ingress.yaml                # Nginx Ingress routing /api and /ws to backend, / to frontend
│   ├── configmap.yaml              # Environment configuration map
│   └── secret.yaml                 # API secret keys template
├── Dockerfile.backend              # Multi-stage Python 3.11-slim container running Waitress WSGI
├── Dockerfile.frontend             # Multi-stage Node 20-alpine builder ➔ Nginx alpine runtime
├── docker-compose.yml              # Local multi-container Docker compose definition
├── nginx.conf                      # Nginx reverse proxy configuration for Docker/K8s
├── vite.config.js                  # Vite bundler configuration, vendor code-splitting, /api and /ws proxies
├── package.json                    # Node dependencies, scripts (dev, build, lint, test)
└── AI_MEMORY.md                    # Historical architectural decisions and permanent rules
```

---

## 2. Tech Stack

### 2.1 Languages & Frameworks
- **Frontend**:
  - **JavaScript (ESNext)**: Strict ES Modules (`"type": "module"` in [package.json](file:///c:/React/SignBridge/package.json)).
  - **React 19.2.7**: Functional components, React Hooks (`useCallback`, `useMemo`, `useRef`), and `React.lazy` / `Suspense`.
  - **React-DOM 19.2.7**: DOM renderer.
  - **Framer Motion 12.42.2**: Animation engine for panel transitions, splash screens, and spring bottom sheet physics.
  - **Lucide React 1.27.0**: High-performance SVG iconography.
  - **Vanilla CSS3**: Design system with CSS variables, Glassmorphism backdrop-filters, and CSS Grid/Flexbox layouts ([index.css](file:///c:/React/SignBridge/src/index.css), [assistant.css](file:///c:/React/SignBridge/src/components/SignLanguageAssistant/assistant.css), [SplashScreen.css](file:///c:/React/SignBridge/src/components/SplashScreen.css)).
- **Backend**:
  - **Python 3.11+** (runtime verified on Python 3.12.5).
  - **Flask 3.0.3**: Microframework for REST endpoints.
  - **Flask-CORS 5.0.1**: Cross-Origin Resource Sharing middleware.
  - **Flask-Limiter >= 3.5.0**: In-memory rate limiting (`memory://` storage).
  - **Waitress 3.0.0**: Production multi-threaded WSGI server (`--threads=4`).
  - **Gunicorn 21.2.0**: Alternative production WSGI server for Linux containers.
  - **Websockets 12.0.0**: High-speed asynchronous WebSocket server for frame streaming.
  - **PyTorch (torch >= 2.0.0)**: Deep learning framework powering ST-GCN graph models and CNN-BiLSTM sequence models.
  - **Scikit-Learn >= 1.4.0**: Machine learning metrics, pipelines, and random forest models.
  - **XGBoost >= 2.0.0**: Extreme Gradient Boosting decision trees for static alphabet classification.
  - **TensorFlow >= 2.15.0**: Keras deep learning fallback model loading.
  - **NumPy >= 1.26.0**: Vectorized array mathematics and coordinate transformations.
  - **OpenCV (`opencv-python` / `cv2`)**: Video reading, frame conversion, and image preprocessing.
  - **PySerial >= 3.5**: USB serial communication driver communicating with Arduino microcontrollers.
  - **python-dotenv >= 1.0.0**: Environment configuration loader.

### 2.2 External APIs, Models & Third-Party Services
1. **Groq LPU API**:
   - Primary Generative AI provider using ultra-low latency LPU hardware.
   - Models: `llama-3.3-70b-versatile`, `qwen/qwen3.8-27b`, `allam-2-7b`, `llama-3.1-8b-instant`.
   - Managed via [groq_manager.py](file:///c:/React/SignBridge/backend/services/groq_manager.py) with dual-key auto-rotation (`GROQ_API_KEY_1`, `GROQ_API_KEY_2`).
2. **Google Gemini API**:
   - Secondary Generative AI fallback provider.
   - Dual SDK support: modern `google-genai` and legacy `google-generativeai`.
   - Models: `gemini-2.5-flash`, `gemini-2.0-flash`, `gemini-1.5-flash`.
   - Managed via [gemini_manager.py](file:///c:/React/SignBridge/backend/services/gemini_manager.py) with dual-key auto-rotation (`GEMINI_API_KEY_1`, `GEMINI_API_KEY_2`).
3. **Google MediaPipe**:
   - Browser client: MediaPipe Holistic (`0.5.1675471629`) and MediaPipe Hands (`0.4.1675469240`) loaded dynamically via CDN.
   - Backend extraction: `mediapipe.tasks.python.vision.HandLandmarker` utilizing the precompiled [hand_landmarker.task](file:///c:/React/SignBridge/backend/models/hand_landmarker.task) binary.
4. **Web Speech API**:
   - Native browser `webkitSpeechRecognition` / `SpeechRecognition` for English speech-to-text.
   - Native browser `window.speechSynthesis` for spoken audio feedback.

### 2.3 Databases & Storage
- **SQLite 3**: Local embedded file database at [backend/database/signbridge.db](file:///c:/React/SignBridge/backend/database/signbridge.db) running with `PRAGMA journal_mode=WAL;` for non-blocking concurrent writes.

### 2.4 Build Tools, Package Managers & Runtime Requirements
- **Node.js**: Node 20+ with `npm` (frontend).
- **Vite 8.1.1**: Dev server and Rollup production bundler.
- **Oxlint 1.71.0**: High-speed Rust-based JavaScript linter.
- **Node Test Runner**: `node --test tests/*.test.mjs` for frontend tests.
- **Pytest 9.1.1**: Backend unit test runner with `pytest-anyio`.
- **Operating System**: Cross-platform (Windows 11 host with hardware USB serial COM ports; Linux containers in Docker/Kubernetes).

---

## 3. Architecture & Data Flow

### 3.1 High-Level Component Interaction Diagram

```
                             ┌────────────────────────────────────────────────────────┐
                             │                    USER INTERFACES                     │
                             │  • Deaf User: Webcam Video + Real-Time Skeleton HUD    │
                             │  • Hearing User: Web Speech STT Microphone / Chat      │
                             │  • Assistive Sheet: Practice Sandbox & Learn Cards     │
                             └──────────────────────────┬─────────────────────────────┘
                                                        │
                      ┌─────────────────────────────────┴─────────────────────────────────┐
                      │                                                                   │
         Path 1: ISL ➔ Spoken English                                        Path 2: Spoken English ➔ ISL Robot
                      │                                                                   │
                      ▼                                                                   ▼
       Browser MediaPipe Holistic / Hands                                 Browser Web Speech Recognition
        (Extracts 42 3D Points: 126 Floats)                               (Transcribes Hearing User Voice)
                      │                                                                   │
       One-Euro Adaptive Speed Low-Pass Filter                                            │
                      │                                                                   │
           [WebSocket Pipe ws://:5001]                                            [HTTP POST /api/llm/simplify]
       (or fallback to HTTP POST /api/translate)                                          │
                      │                                                                   ▼
                      ▼                                                    LLM Keyword Simplification Cascade
      Flask / Waitress Backend (app.py)                                    (Groq Llama 3.3 ➔ Gemini 1.5 ➔ Local)
                      │                                                                   │
      ┌───────────────┴───────────────┐                                                   │ Extracted Keywords: "WHERE DOCTOR"
      │                               │                                                   ▼
      ▼ (Static Frame)                ▼ (30-Frame Sequence)                     [HTTP POST /api/robot/sign]
Letter Recognition Engine        Dynamic Word Recognizer                                  │
  • 208-D Invariant Extractor      • Wrist-relative Motion Gate                           ▼
  • Centroid Distance Check        • 1D-CNN + BiLSTM (57 classes)              ArduinoSerial Driver Thread
  • ST-GCN + XGBoost Hybrid        • Fallback: Sequence Random Forest          (Queued non-blocking serial worker)
      │                               │                                                   │
      └───────────────┬───────────────┘                                                   │ ASCII: <L0..L4|R0..R4>\n
                      │ Recognized Letters / Words                                        ▼
                      ▼                                                          Arduino Mega / Uno
       Gesture State Machine (Lock & Steady)                                     10x TowerPro SG90 Micro-Servos
                      │                                                    (Dual 5-Finger Bionic Hands Actuate)
                      ▼
         Live Sentence Buffer (Accumulator)
                      │
     [Touchless 1.8s Drop or Space Trigger]
                      │
                      ▼
        [HTTP POST /api/llm/refine]
   LLM Grammar Cascade (SOV ➔ Natural SVO)
   (Groq Llama 3.3 ➔ Gemini 1.5 ➔ Local Rule)
                      │
                      ▼
        Web Speech Audio Output (TTS)
        + React Chat Thread Message Bubble
```

### 3.2 Entry Points
- **Frontend Web Kiosk**: [src/main.jsx](file:///c:/React/SignBridge/src/main.jsx) $\rightarrow$ [src/App.jsx](file:///c:/React/SignBridge/src/App.jsx) $\rightarrow$ [src/components/SignBridgeKiosk.jsx](file:///c:/React/SignBridge/src/components/SignBridgeKiosk.jsx). Hosted at `http://localhost:5173`.
- **Backend REST API**: [backend/app.py](file:///c:/React/SignBridge/backend/app.py). Started via `waitress-serve --listen=0.0.0.0:5000 --threads=4 app:app` or `python app.py`. Hosted at `http://localhost:5000`.
- **WebSocket Streaming Server**: [backend/services/websocket_server.py](file:///c:/React/SignBridge/backend/services/websocket_server.py). Automatically launched in a daemon thread by `app.py` on `ws://127.0.0.1:5001`.
- **CLI Model Training & Evaluation**:
  - `python backend/train_unified.py` (Full pipeline).
  - `python backend/benchmark_alphabet_models.py` (Accuracy benchmarking).
  - `python backend/generate_centroids.py` (Generates [letter_centroids.json](file:///c:/React/SignBridge/backend/models/letter_centroids.json)).
  - `python backend/generate_idle_dataset.py` (Synthesizes negative idle sequences).

### 3.3 Key Design Patterns
- **Service Layer Pattern**: All backend business logic is isolated into standalone classes in `backend/services/` (`TranslatorModel`, `WordRecognizer`, `ArduinoSerial`, `GroqManager`, `GeminiManager`, `CentroidValidator`, `GestureStateMachine`).
- **Provider Strategy & Fallback Cascade**: High-speed primary provider (Groq) rotates through healthy API keys. If rate-limited or unavailable, it transparently fails over to secondary provider (Gemini), and finally to local deterministic heuristics without crashing or exposing errors to the user.
- **Asynchronous Worker Queue**: [ArduinoSerial](file:///c:/React/SignBridge/backend/services/arduino_serial.py) uses a `queue.Queue()` processed by a background daemon worker thread. REST callers return an immediate HTTP 200 while servo stepping executes smoothly in hardware.
- **RingBuffer & Vectorized Operations**: The frontend uses an allocation-free fixed-capacity [RingBuffer](file:///c:/React/SignBridge/src/hooks/useGestureRecognition.js#L17) for 30-frame temporal windows. The backend uses vectorized batch normalization ([normalize_landmarks_batch](file:///c:/React/SignBridge/backend/services/translator_model.py#L92)) processing $(N, 126)$ arrays without per-frame loops.
- **Dual-Mode Graceful Degradation (LIVE vs. DEMO)**: The entire React UI seamlessly operates in DEMO mode with simulated typewriter feeds and scripted dialogue pools when the backend or Arduino is unavailable.

---

## 4. Module-by-Module Breakdown

### 4.1 Backend Core Services

#### [backend/app.py](file:///c:/React/SignBridge/backend/app.py)
- **Responsibility**: Flask REST application server configuring routes, rate limiters, CORS policies, authentication, and error handling.
- **Key Functions**:
  - `check_api_key() -> bool`: Optional shared-secret authentication comparing `X-API-Key` to `SIGNBRIDGE_API_KEY` using constant-time `hmac.compare_digest`.
  - `generate_llm_response(prompt, system_instruction, ...) -> LLMResponse`: Coordinates the Groq $\rightarrow$ Gemini $\rightarrow$ Local Fallback cascade.
  - `clean_llm_text(text: str) -> str`: Strips `<think>` tags, reasoning fences, markdown formatting, and quotes from LLM outputs.
  - `fallback_refine(raw: str) -> str`: Deterministic heuristic converting raw gloss strings to grammatical sentences when external APIs are offline.
  - `fallback_simplify(text: str, vocab) -> str`: Extracts up to 4 known vocabulary tokens from input text for robotic signing.
- **Dependencies**: [TranslatorModel](file:///c:/React/SignBridge/backend/services/translator_model.py), [WordRecognizer](file:///c:/React/SignBridge/backend/services/word_recognizer.py), [ArduinoSerial](file:///c:/React/SignBridge/backend/services/arduino_serial.py), [GroqManager](file:///c:/React/SignBridge/backend/services/groq_manager.py), [GeminiManager](file:///c:/React/SignBridge/backend/services/gemini_manager.py), [schema.py](file:///c:/React/SignBridge/backend/database/schema.py).

#### [backend/services/translator_model.py](file:///c:/React/SignBridge/backend/services/translator_model.py)
- **Responsibility**: Static alphabet and digit recognition engine mapping 126 landmark coordinates to classes `A`–`Z` and `0`–`9`.
- **Loading Priority**:
  1. Calibrated Hybrid Ensemble (ST-GCN PyTorch + XGBoost `.pkl`).
  2. ST-GCN Kinematic Graph (`isl_stgcn_model.pt`).
  3. XGBoost Classifier (`isl_xgboost_model.pkl`).
  4. TensorFlow/Keras (`isl_gesture_model.h5`).
  5. Geometric Heuristic classifier based on Mendeley ISL rules.
- **Key Functions**:
  - `normalize_landmarks(landmarks)`: Centers coordinates on the mean wrist anchor point and scales by the middle-finger-to-wrist span. Absent hands are zeroed out.
  - `normalize_landmarks_batch(batch)`: Fully vectorized NumPy implementation operating on shape $(N, 126)$.
  - `predict(landmarks, body_anchors=None) -> dict`: Evaluates the model according to the active mode and returns `{ letter, confidence, all_scores, mode, rejected, rejection_reason }`.
  - `_predict_hybrid_ensemble(landmarks) -> dict`: Blends probabilities: $P = 0.65 \times P_{\text{XGB}} + 0.35 \times P_{\text{ST-GCN}}$ (with temperature scaling $T = 1.20$).

#### [backend/services/word_recognizer.py](file:///c:/React/SignBridge/backend/services/word_recognizer.py)
- **Responsibility**: Classifies sliding windows of 30 frames $\times$ 126 coordinates into whole ISL word gestures across 57 classes.
- **Key Classes & Functions**:
  - `compute_wrist_relative_motion(arr, max_jump=0.15) -> float`: Calculates mean per-hand displacement relative to each hand's wrist coordinate across frames, filtering out tracking glitches and hand entries.
  - `WordRecognizer`: Loads `isl_cnn_lstm_word_model.pt` (CNNBiLSTM with 2 layers of bidirectional LSTM, dropout 0.3, and optional attention pooling).
  - `predict(frames_sequence) -> dict`: Validates shape $(30, 126)$, calculates motion velocity, rejects static hands ($\text{velocity} < 0.025$), enforces margin threshold ($\ge 0.06$) and entropy thresholds ($\le 2.20$), and returns `{ word, confidence, margin, motion_velocity, rejected, rejection_reason }`.

#### [backend/services/feature_extractor.py](file:///c:/React/SignBridge/backend/services/feature_extractor.py)
- **Responsibility**: Extracts 208-D scale-, translation-, and rotation-invariant geometric features from 126 raw coordinates.
- **Feature Vector Composition**:
  - 126 normalized Cartesian coordinates (wrist-centered, hand-span scaled).
  - 36 single-hand geometric invariants (18 per hand: tip-to-wrist distances, tip-to-thumb spans, adjacent fingertip distances, finger curl ratios, pinch ratio).
  - 20 knuckle joint bending angles (10 per hand: cosine angles at MCP-PIP-DIP and PIP-DIP-TIP).
  - 6 palm surface normal orientation coordinates (wrist roll and tilt).
  - 6 specialized finger posture invariants (ring extension ratios, O-ring variance, thumb crossing).
  - 14 inter-hand cross-interaction distances (Euclidean distances between fingertips and wrists across both hands for contact signs like `M`, `N`, `P`, `R`).

#### [backend/services/gesture_pipeline.py](file:///c:/React/SignBridge/backend/services/gesture_pipeline.py)
- **Responsibility**: Out-of-dataset rejection, idle/resting hand detection, and base neutral gesture detection.
- **Key Classes & Functions**:
  - `CentroidValidator`: Compares normalized live landmarks against precomputed dataset centroids stored in `letter_centroids.json`. If the Euclidean distance exceeds $1.35\times$ the class threshold, the gesture is rejected with `rejection_reason: 'out_of_dataset_distance'`. Checks both direct orientation and swapped-hand slot configurations.
  - `is_idle_hand(landmarks) -> bool`: Rejects hands resting on desks or below the camera frame ($\text{wrist}_y > 0.88$).
  - `is_neutral_gesture(landmarks) -> bool`: Identifies two open, extended flat palms held facing the camera to re-arm the recognition state machine.

#### [backend/services/gesture_state_machine.py](file:///c:/React/SignBridge/backend/services/gesture_state_machine.py)
- **Responsibility**: Ultra-fast "Lock, then Hold Steady" recognition lifecycle.
- **States**: `NEUTRAL` $\rightarrow$ `ARMED` $\rightarrow$ `CAPTURING` $\rightarrow$ `COOLDOWN` (Locked, Hold Steady) $\rightarrow$ `ARMED`.
- **Logic**:
  - Instant commit: 1 frame $\ge 0.92$ confidence commits immediately.
  - Rapid commit: 3 consecutive identical frames (~100ms) commit.
  - Jitter tolerance: 1 dropped/missed frame tolerated mid-streak.
  - Dynamic re-arm: Minimum lock of 250ms; once the hand stays steady (motion $< 0.015$) for 350ms, it automatically re-arms for the next sign.

#### [backend/services/groq_manager.py](file:///c:/React/SignBridge/backend/services/groq_manager.py) & [gemini_manager.py](file:///c:/React/SignBridge/backend/services/gemini_manager.py)
- **Responsibility**: Resilient, non-blocking API managers for external LLM inference.
- **Key Classes & Mechanics**:
  - `KeyInfo`: Tracks per-key health state (`AVAILABLE`, `COOLDOWN`, `RATE_LIMITED`, `INVALID`), failure counts, and cooldown timestamps.
  - Dynamic Model Discovery: Queries `client.models.list()` on startup to discover available models matching verified priority candidate lists.
  - Exponential Backoff & Cascade: Retries transient errors (500, timeouts) with backoff; rate limits (429) put the active key in cooldown and switch immediately to the secondary key.
  - Security: `mask_key()` guarantees full API keys are never written to logs or error messages.

#### [backend/services/arduino_serial.py](file:///c:/React/SignBridge/backend/services/arduino_serial.py)
- **Responsibility**: Non-blocking PySerial hardware communication driver driving 10x SG90 micro-servos on dual robotic hands.
- **Key Logic**:
  - Default baud rate: `9600`.
  - Format: Formatted ASCII strings `<L0,L1,L2,L3,L4|R0,R1,R2,R3,R4>\n` (values $0^\circ$ extended to $180^\circ$ flexed).
  - Background Worker: Dedicated daemon thread `_process_sign_queue()` consumes letters from a thread-safe queue.
  - Auto-Idle Servo Relaxer: If no signing occurs for $>15.0$ seconds, automatically returns all 10 servos to neutral relaxed rest pose (`[90, 90, 90, 90, 90, 90, 90, 90, 90, 90]`) to prevent motor strain and overheating.

#### [backend/services/websocket_server.py](file:///c:/React/SignBridge/backend/services/websocket_server.py)
- **Responsibility**: Asynchronous bidirectional WebSocket server running on port 5001. Receives raw JSON landmark frames and streams back classification predictions with sub-2ms server-side latency.

#### [backend/database/schema.py](file:///c:/React/SignBridge/backend/database/schema.py)
- **Responsibility**: SQLite database manager initializing tables, logging conversation turns, and tracking dataset collection sessions.

---

### 4.2 Frontend Core Components & Hooks

#### [src/App.jsx](file:///c:/React/SignBridge/src/App.jsx)
- Top-level wrapper mounting [ErrorBoundary](file:///c:/React/SignBridge/src/components/ErrorBoundary.jsx), [SignBridgeKiosk](file:///c:/React/SignBridge/src/components/SignBridgeKiosk.jsx), and [SplashScreen](file:///c:/React/SignBridge/src/components/SplashScreen.jsx). Mounts the kiosk immediately in the background so camera and models initialize while the 3.2s splash plays.

#### [src/components/SignBridgeKiosk.jsx](file:///c:/React/SignBridge/src/components/SignBridgeKiosk.jsx)
- Split-screen container coordinating [HumanPanel](file:///c:/React/SignBridge/src/components/HumanPanel.jsx) (left) and [RobotPanel](file:///c:/React/SignBridge/src/components/RobotPanel.jsx) (right). Registers global kiosk keyboard hotkeys (`Space`, `1`/`H`, `2`/`R`, `C`).

#### [src/components/HumanPanel.jsx](file:///c:/React/SignBridge/src/components/HumanPanel.jsx)
- Left card dedicated to the Deaf user. Renders [CameraView](file:///c:/React/SignBridge/src/components/CameraView.jsx), live accumulating sentence buffer, inactivity auto-send countdown ring, dynamic font autoscaling based on text length, and quick edit/send controls.

#### [src/components/RobotPanel.jsx](file:///c:/React/SignBridge/src/components/RobotPanel.jsx)
- Right card dedicated to the Hearing user. Renders the conversation dialogue thread, Web Speech STT microphone button, text input box, and collapsible split-view [GestureReferenceSheet](file:///c:/React/SignBridge/src/components/GestureReferenceSheet.jsx).

#### [src/components/CameraView.jsx](file:///c:/React/SignBridge/src/components/CameraView.jsx)
- WebRTC camera component. Manages webcam acquisition via `useWebcam`, landmark tracking via `useHandDetection`, gesture recognition via `useGestureRecognition`, canvas overlay via [HandTrackingOverlay](file:///c:/React/SignBridge/src/components/SignLanguageAssistant/HandTrackingOverlay.jsx), and fullscreen HUD mode (`F` key).

#### [src/hooks/useGestureRecognition.js](file:///c:/React/SignBridge/src/hooks/useGestureRecognition.js)
- 1,534-line master recognition hook. Maintains the [RingBuffer](file:///c:/React/SignBridge/src/hooks/useGestureRecognition.js#L17) for 30-frame temporal sequences, establishes persistent low-latency WebSocket connection (`ws://127.0.0.1:5001`), executes majority-vote smoothing across 5 frames, coordinates touchless 1.8s auto-send timers, and triggers `/api/llm/refine` and Web Speech TTS.

#### [src/hooks/useHandDetection.js](file:///c:/React/SignBridge/src/hooks/useHandDetection.js)
- MediaPipe Holistic & Hands CDN manager. Dynamically injects script tags, sets camera options, extracts 42 3D points, runs [trackAndSeparateHands](file:///c:/React/SignBridge/src/hooks/useHandDetection.js#L56) using centroid distance matching to prevent left/right slot swapping, and smooths landmarks via [LandmarkSmoother](file:///c:/React/SignBridge/src/utils/oneEuroFilter.js).

#### [src/utils/oneEuroFilter.js](file:///c:/React/SignBridge/src/utils/oneEuroFilter.js)
- Adaptive speed-based One-Euro low-pass filter. Eliminates high-frequency hand jitter when hands are held still while preserving instantaneous responsiveness during rapid transitions. Points at `(0, 0, 0)` immediately reset filters to prevent ghost landmarks.

---

## 5. Data Layer

### 5.1 Database Models & Schemas ([backend/database/schema.py](file:///c:/React/SignBridge/backend/database/schema.py))
SignBridge uses SQLite 3 stored at [backend/database/signbridge.db](file:///c:/React/SignBridge/backend/database/signbridge.db) with two tables:

```sql
-- 1. Conversation & Translation History
CREATE TABLE IF NOT EXISTS conversation_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    speaker TEXT NOT NULL,          -- 'human' or 'robot'
    raw_text TEXT,                  -- Raw fingerspelled letters (e.g. 'H E L L O')
    refined_sentence TEXT,          -- Refined LLM sentence (e.g. 'Hello, how can I assist you?')
    confidence REAL,               -- Gesture model confidence score (0.0 to 1.0)
    llm_provider TEXT              -- 'Groq', 'Gemini', or 'local_fallback'
);

-- 2. Dataset Session Metadata
CREATE TABLE IF NOT EXISTS dataset_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    letter TEXT NOT NULL,
    session_id TEXT UNIQUE NOT NULL,
    signer_id TEXT,
    frame_count INTEGER,
    file_path TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### 5.2 Datasets & Asset Inventory
1. **RealSign ISL Dataset** (`dataset/RealSign-Indian-Sign-Language-Dataset-main/`):
   - Academic benchmark dataset for Indian Sign Language.
   - Letters: `Letters/A.jpg`–`Z.jpg` (26 canonical single-image references).
   - Training, Validation, and Testing folders containing thousands of multi-signer images.
2. **Custom `dataset_2/`** (`dataset_2/`):
   - 521 ultra-high-resolution (4000$\times$2252) real-world phone captures across all 26 letters (20 photos per letter) taken under varied lighting and angles.
3. **Mendeley ISL Dataset** (`dataset/Mendeley_ISL/`):
   - Reference images and posture rules ([ISL_Mendeley_Alphabets.csv](file:///c:/React/SignBridge/dataset/Mendeley_ISL/extracted/ISL_Mendeley_Alphabets.csv)).
4. **User Session Kiosk Data** (`backend/dataset_collected/`):
   - Partitioned JSON files organized by letter (`A`–`Z`) and digit (`0`–`9`). Each file contains metadata, sample counts, and normalized arrays of 126 floats.
5. **ISL Words Dataset** (`Dataset_Words/` & `backend/dataset_words/`):
   - 30-frame temporal sequences stored as JSON files for 57 word classes plus synthetic negative idle samples (`NONE`).
6. **Hugging Face ISLRTC National Dictionary**:
   - Government of India official ISL video dictionary streamed on-demand via [stream_and_extract_hf.py](file:///c:/React/SignBridge/backend/stream_and_extract_hf.py).

### 5.3 Trained Model Artifacts ([backend/models/](file:///c:/React/SignBridge/backend/models/))
- [isl_xgboost_model.pkl](file:///c:/React/SignBridge/backend/models/isl_xgboost_model.pkl): 28.5 MB, trained on 52,852 augmented vectors, **87.43% test accuracy**, 208-D input.
- [isl_stgcn_model.pt](file:///c:/React/SignBridge/backend/models/isl_stgcn_model.pt): 199 KB, PyTorch spatial-temporal graph model, **82.03% test accuracy**.
- [isl_cnn_lstm_word_model.pt](file:///c:/React/SignBridge/backend/models/isl_cnn_lstm_word_model.pt): 3.1 MB, 1D-CNN + BiLSTM, 57 classes, **51.22% validation accuracy**.
- [isl_word_classifier.pkl](file:///c:/React/SignBridge/backend/models/isl_word_classifier.pkl): 23.3 MB, Random Forest sequence classifier, **29.40% validation accuracy**.
- [hand_landmarker.task](file:///c:/React/SignBridge/backend/models/hand_landmarker.task): 7.8 MB, Google MediaPipe vision task bundle.
- [letter_centroids.json](file:///c:/React/SignBridge/backend/models/letter_centroids.json): Per-class 98th percentile distance thresholds for out-of-dataset rejection.

---

## 6. API / Interface Surface

### 6.1 Backend REST API Endpoints ([backend/app.py](file:///c:/React/SignBridge/backend/app.py))

| Method | Endpoint | Description | Request Body | Response Shape | Auth & Rate Limit |
|---|---|---|---|---|---|
| `GET` | `/api/health` | Service health & model status | None | `{ status: "ok", translator_mode: "hybrid_ensemble", arduino_connected: bool, gemini_available: bool, groq_available: bool, word_recognizer_available: bool, db_available: bool }` | Public |
| `GET` | `/api/llm/status` | Health & cooldown status of LLM keys | None | `{ groq: {...}, gemini: {...} }` | Public |
| `GET` | `/api/model/info` | Letter classifier metadata | None | `{ model_type: "hybrid_ensemble", classes: [...], test_accuracy: 0.8623, ... }` | Public |
| `GET` | `/api/words/info` | Word model metadata | None | `{ model_type: "cnn_bilstm", num_classes: 57, labels: [...], ... }` | Public |
| `POST` | `/api/model/reload` | Thread-safe reload of model weights | None | `{ status: "reloaded", translator_mode: "...", ... }` | Optional `X-API-Key` |
| `POST` | `/api/translate` | Classifies 1 frame (126 floats) | `{ landmarks: [126 floats] }` | `{ letter: "A", confidence: 0.98, all_scores: {...}, rejected: false }` | 1200 / min |
| `POST` | `/api/translate/batch` | Classifies batch of frames | `{ frames: [[126 floats], ...] }` | `{ results: [...], sentence: "HELLO", frame_count: 15 }` | 1200 / min |
| `POST` | `/api/translate/word` | Classifies 30 frames to whole word | `{ frames: [[126 floats] × 30] }` | `{ word: "NAMASTE", confidence: 0.94, margin: 0.32, motion_velocity: 0.048, rejected: false }` | 600 / min |
| `GET` | `/api/gesture/config` | Retrieves active state machine thresholds | None | `{ CONFIDENCE_THRESHOLD: 0.75, STABILITY_FRAMES_K: 3, ... }` | Public |
| `GET` | `/api/history` | Fetches SQLite conversation records | Query param `?limit=50` | `{ history: [ { id, timestamp, speaker, raw_text, refined_sentence, ... } ] }` | Public |
| `POST` | `/api/collect_data` | Saves recorded landmarks to disk & DB | `{ letter: "A", session_id: "...", frames: [...] }` | `{ status: "success", letter: "A", frame_count: 60 }` | 30 / min |
| `POST` | `/api/llm/refine` | Refines raw glosses to fluent sentence | `{ text: "WATER PLEASE" }` | `{ raw_text: "...", refined_sentence: "Can I get some water?", llm_provider: "groq", fallback_used: false }` | 30 / min |
| `POST` | `/api/llm/simplify` | Simplifies speech to robot keywords | `{ text: "Where is doctor?" }` | `{ original_text: "...", robot_keywords: "WHERE DOCTOR", llm_provider: "groq" }` | 30 / min |
| `POST` | `/api/llm/answer` | Answers domain assistant questions | `{ text: "What is your name?" }` | `{ answer: "I'm SignBridge! Nice to meet you.", llm_provider: "groq" }` | 30 / min |
| `GET` | `/api/robot/status` | Arduino connection status | None | `{ connected: bool, port: "COM3", baud_rate: 9600, is_signing: bool, is_at_rest: bool }` | Public |
| `POST` | `/api/robot/connect` | Connects PySerial driver | `{ port: "COM3" }` (optional) | `{ status: "connected", port: "COM3", baud_rate: 9600 }` | Optional `X-API-Key` |
| `POST` | `/api/robot/disconnect` | Closes PySerial connection | None | `{ status: "disconnected", message: "..." }` | Optional `X-API-Key` |
| `POST` | `/api/robot/sign` | Queues text for bionic hand signing | `{ text: "HELLO", hold_time: 1.0 }` | `{ text: "HELLO", signed_letters: ["H","E","L","L","O"], arduino_connected: bool, message: "..." }` | 10 / min |
| `POST` | `/api/robot/sign-letter`| Signs single letter on robotic hand | `{ letter: "A", hold_time: 1.5 }` | `{ letter: "A", signed: true, arduino_connected: bool }` | 10 / min |

### 6.2 WebSocket Streaming Interface ([backend/services/websocket_server.py](file:///c:/React/SignBridge/backend/services/websocket_server.py))
- **URL**: `ws://127.0.0.1:5001` (proxied in Vite as `/ws`).
- **Client Request Formats**:
  - Letter frame: `{"type": "letter", "landmarks": [126 floats], "body_anchors": {...}}`
  - Word sequence: `{"type": "word", "frames": [[126 floats] × 30]}`
- **Server Response**: Instant JSON object with classification result, matching REST shapes with $<2\text{ms}$ latency.

### 6.3 User Interface Surface & Controls
- **Human Panel (Left Card)**: Real-time video canvas, landmark overlay, dynamic font size, touchless auto-send countdown ring, clear buffer button.
- **Robot Panel (Right Card)**: Conversational chat bubbles, microphone toggle for speech recognition, text input field, expandable in-card [GestureReferenceSheet](file:///c:/React/SignBridge/src/components/GestureReferenceSheet.jsx).
- **Assistant Bottom Sheet Modal**: Draggable drawer with 5 tabs: *Live Detection* (confidence bars), *Practice* (interactive gesture verification), *Learn* (vocabulary cards), *History* (SQLite logs), and *Data Collection* (custom recording sandbox).
- **Keyboard Shortcuts**:
  - `Space`: Toggle automated demo mode.
  - `1` or `H`: Trigger human dialogue turn.
  - `2` or `R`: Trigger robot dialogue turn.
  - `F`: Toggle fullscreen camera HUD view.
  - `C`: Clear buffers and reset chat.

---

## 7. Configuration & Environment

### 7.1 Environment Variables

| Variable Name | Required? | Default Value | Description |
|---|---|---|---|
| `GROQ_API_KEY_1` | Recommended | None | Primary Groq API key for LPU inference |
| `GROQ_API_KEY_2` | Optional | None | Secondary Groq API key for auto-rotation |
| `GROQ_API_KEY` | Optional | None | Legacy fallback Groq key |
| `GEMINI_API_KEY_1` | Recommended | None | Primary Google Gemini API key |
| `GEMINI_API_KEY_2` | Optional | None | Secondary Google Gemini key for auto-rotation |
| `GOOGLE_API_KEY` | Optional | None | Legacy fallback Gemini key |
| `SIGNBRIDGE_API_KEY` | Optional | None | Shared secret for `X-API-Key` REST header; if unset, auth is disabled for local dev |
| `PORT` | Optional | `5000` | Flask REST server port |
| `HOST` | Optional | `0.0.0.0` or `127.0.0.1` | Network interface to bind |
| `WS_PORT` | Optional | `5001` | Asynchronous WebSocket server port |
| `WS_HOST` | Optional | `127.0.0.1` | Network interface for WebSocket server |
| `DB_PATH` | Optional | `backend/database/signbridge.db` | Filesystem path for SQLite database |
| `LLM_RATE_LIMIT` | Optional | `"30 per minute"` | Rate limit for `/api/llm/*` endpoints |
| `ROBOT_RATE_LIMIT` | Optional | `"10 per minute"` | Rate limit for `/api/robot/*` endpoints |
| `TRANSLATE_RATE_LIMIT` | Optional | `"1200 per minute"` | Rate limit for `/api/translate` |
| `LLM_DEADLINE_S` | Optional | `"4.0"` | Max seconds to wait for LLM before falling back |
| `CORS_ORIGINS` | Optional | `http://localhost:5173,...` | Comma-separated list of allowed CORS origins |

### 7.2 Step-by-Step Setup & Run Instructions

#### Prerequisites
- Python 3.11 or 3.12 installed.
- Node.js 20+ and npm installed.
- (Optional) Arduino Mega/Uno connected via USB for robotic actuation.

#### Step 1: Clone & Setup Python Virtual Environment
```bash
git clone <repository_url>
cd SignBridge

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install backend dependencies
cd backend
pip install -r requirements.txt
```

#### Step 2: Configure Environment Keys
```bash
# In backend/
copy .env.example .env
# Edit backend/.env and populate your GROQ_API_KEY_1 and GEMINI_API_KEY_1
```

#### Step 3: Setup Frontend Dependencies
```bash
# In project root:
npm install
```

#### Step 4: Run the System Locally
**Terminal 1 (Backend REST & WebSocket Servers):**
```bash
cd backend
python app.py
# Server starts Waitress WSGI on http://127.0.0.1:5000 and WebSocket on ws://127.0.0.1:5001
```

**Terminal 2 (Frontend Dev Server):**
```bash
# In project root:
npm run dev
# Starts Vite at http://localhost:5173 with automated proxies to :5000 (/api) and :5001 (/ws)
```

#### Step 5: Run with Docker Compose (Alternative)
```bash
docker-compose up --build
# Serves frontend on http://localhost:5173 and backend on http://localhost:5000
```

---

## 8. Testing & Quality

### 8.1 Existing Test Suites

1. **Backend Tests ([backend/tests/](file:///c:/React/SignBridge/backend/tests/))**:
   - Framework: **Pytest 9.1.1** with `pytest-anyio`.
   - Test Count: **63 passing unit tests**.
   - Test Execution Time: ~170 seconds (includes simulated API backoff and rate limit cool-downs).
   - What is Covered:
     - [test_app_llm.py](file:///c:/React/SignBridge/backend/tests/test_app_llm.py) (17 tests): Payload clamping, sanitization of `<think>` blocks, sentence limiters, fallback grammar heuristics, rate limiting, and REST endpoint schemas.
     - [test_gemini_manager.py](file:///c:/React/SignBridge/backend/tests/test_gemini_manager.py) (10 tests): Multi-key rotation, transient error retries (500), permanent disabling of invalid keys (401/403), credential masking in logs.
     - [test_groq_manager.py](file:///c:/React/SignBridge/backend/tests/test_groq_manager.py) (10 tests): Dual-key failover cascade, dynamic model discovery, cooldown recovery, rate-limit classification.
     - [test_gesture_pipeline.py](file:///c:/React/SignBridge/backend/tests/test_gesture_pipeline.py) (8 tests): Out-of-dataset centroid distance rejection, resting hand detection, neutral two-open-palms filter.
     - [test_gesture_state_machine.py](file:///c:/React/SignBridge/backend/tests/test_gesture_state_machine.py) (7 tests): Lifecycle transitions, instant commit at $\ge 0.92$, jitter tolerance, dynamic re-arm steady duration.
     - [test_live_bug_fixes.py](file:///c:/React/SignBridge/backend/tests/test_live_bug_fixes.py) (5 tests): Letter 'H' slot invariance regressions, 'NAMASTE' resting-hand hallucination prevention, hand entry glitches.
     - [test_slot_ordering.py](file:///c:/React/SignBridge/backend/tests/test_slot_ordering.py) (6 tests): Canonical slot assignment when MediaPipe inverts left/right labels or emits duplicate labels.

2. **Frontend Tests ([tests/splash.test.mjs](file:///c:/React/SignBridge/tests/splash.test.mjs))**:
   - Framework: **Node.js Native Test Runner** (`node:test`, `node:assert/strict`).
   - Test Count: **3 passing tests**.
   - What is Covered: Splash animation duration limits (3000ms animation + 200ms fade = 3200ms total; 1500ms on prefers-reduced-motion), callback execution, and state unmounting.

### 8.2 How to Run Tests & Linters
- **Run Backend Tests**:
  ```bash
  cd backend
  ..\.venv\Scripts\pytest
  ```
- **Run Frontend Tests**:
  ```bash
  npm test
  ```
- **Run Code Linter**:
  ```bash
  npm run lint
  # Runs oxlint across 29 files (0 errors, 13 non-blocking warnings)
  ```

---

## 9. Health Check

### 9.1 Known Past Bugs & Regressions Resolved
1. **Letter 'H' Dual-Hand Slot Inversion Bug**:
   - *Problem*: In Indian Sign Language, the letter `H` is formed by resting the right palm flat across the open left palm. In a live webcam third-person view, MediaPipe frequently inverted the left/right labels or swapped their index order, causing the 126-D feature vector to swap slots 0..62 and 63..125, resulting in failed recognition.
   - *Fix*: Created [assign_canonical_hand_slots()](file:///c:/React/SignBridge/backend/services/landmark_utils.py#L76) which deterministically assigns detected hands to canonical slots based on horizontal screen coordinates ($x_{\text{left}} \ge x_{\text{right}}$ in third-person view) regardless of MediaPipe's label. Verified by [test_slot_ordering.py](file:///c:/React/SignBridge/backend/tests/test_slot_ordering.py).
2. **Word 'NAMASTE' Idle Hand Hallucination Bug**:
   - *Problem*: Users resting both hands neutrally in front of the camera would trigger false predictions for `NAMASTE` because two still hands closely match the static geometry of prayer hands.
   - *Fix*: Implemented wrist-relative kinematic motion velocity gating in [word_recognizer.py](file:///c:/React/SignBridge/backend/services/word_recognizer.py#L44). Static hands with $\text{velocity} < 0.025$ are rejected as `static_hand_detected`. Verified by [test_live_bug_fixes.py](file:///c:/React/SignBridge/backend/tests/test_live_bug_fixes.py).
3. **Windows IPv6 Localhost Latency Stall**:
   - *Problem*: Node 18+ and modern browsers on Windows resolve `localhost` to IPv6 `::1` first, causing a 2-second timeout before falling back to IPv4 `127.0.0.1` when proxying to Flask.
   - *Fix*: Configured [vite.config.js](file:///c:/React/SignBridge/vite.config.js#L36) to explicitly proxy `/api` to `http://127.0.0.1:5000` and `/ws` to `ws://127.0.0.1:5001`.
4. **Flask Debug Subprocess Port Binding Conflict**:
   - *Problem*: When `FLASK_DEBUG=True`, Werkzeug's auto-reloader creates a child process. Both parent and child processes attempted to bind the WebSocket port 5001 and open exclusive serial locks on `COM3`.
   - *Fix*: Disabled the reloader in `app.py` ([app.py:L1222](file:///c:/React/SignBridge/backend/app.py#L1222)) when running locally: `app.run(..., use_reloader=False)`.

### 9.2 Technical Debt & Architectural Opportunities
1. **Hardcoded API URL in [DataCollectionTab.jsx](file:///c:/React/SignBridge/src/components/SignLanguageAssistant/tabs/DataCollectionTab.jsx#L53)**:
   - Line 53 uses `fetch('http://localhost:5000/api/collect_data', ...)` instead of the relative path `fetch('/api/collect_data', ...)`. In production or Docker environments where port 5000 is not directly exposed to the browser, this will fail. Needs conversion to relative `/api/collect_data`.
2. **PySerial Port Locking on Windows**:
   - Windows grants exclusive locks to COM ports. If an Arduino Serial Monitor or another Python script is open on `COM3`, `arduino_serial.py` fails with `PermissionError`. The backend handles this gracefully by operating in simulation mode, but automatic COM port retry logic could be added.
3. **57-Class BiLSTM Model Accuracy (51.22%)**:
   - As documented in [cnn_lstm_training_meta.json](file:///c:/React/SignBridge/backend/models/cnn_lstm_training_meta.json), expanding the word model from 23 classes to 57 classes dropped validation accuracy to 51.22%. Additional training samples or fine-tuning with temporal transformer backbones would improve dynamic word recognition.

### 9.3 Inconsistencies Between Documentation & Actual Code
1. **Word Model Vocabulary Size & Accuracy**:
   - Earlier documentation claimed 23 words with 87.04% accuracy, while [ARCHITECTURE.md](file:///c:/React/SignBridge/ARCHITECTURE.md#L32) claimed 57 words with 93.3% accuracy.
   - *Verified Code Ground Truth*: [cnn_lstm_training_meta.json](file:///c:/React/SignBridge/backend/models/cnn_lstm_training_meta.json) specifies **57 classes**, with **14,258 training samples** and **51.22% validation accuracy** (`val_accuracy: 0.5122`).
2. **Serial Baud Rate Specification**:
   - [ARCHITECTURE.md:L53](file:///c:/React/SignBridge/ARCHITECTURE.md#L53) stated 9600 baud, while earlier drafts stated 115200 baud.
   - *Verified Code Ground Truth*: [arduino_serial.py:L76](file:///c:/React/SignBridge/backend/services/arduino_serial.py#L76) defines `baud_rate=9600` by default.

---

## 10. Extending the Project

### 10.1 Common Contributor Workflows

#### Adding a New Static Letter or Custom Sign
1. Add reference images or record video frames using the *Data Collection Tab* in the UI or placing images in `dataset/`.
2. Run landmark extraction: `python backend/extract_static_landmarks.py`.
3. Re-train the alphabet model: `python backend/train_unified.py --letters-only`.
4. Regenerate class distance centroids: `python backend/generate_centroids.py`.
5. Add the corresponding 10-servo motor angle mapping array to `ISL_SERVO_MAP` in [arduino_serial.py](file:///c:/React/SignBridge/backend/services/arduino_serial.py#L37).
6. Add the visual card description and steps to `ISL_ALPHABET_DATA` in [gestureData.js](file:///c:/React/SignBridge/src/data/gestureData.js#L9).

#### Adding a New Dynamic Word Gesture
1. Place 30-frame video clips in `Dataset_Words/Words/<WORD_NAME>/`.
2. Extract temporal sequences: `python backend/extract_video_landmarks.py`.
3. Train the word model: `python backend/train_unified.py --words-only`.
4. Update `STATIC_WORD_VOCABULARY` in [app.py](file:///c:/React/SignBridge/backend/app.py#L248) and `_normalize_word_str` in [landmark_utils.py](file:///c:/React/SignBridge/backend/services/landmark_utils.py#L21).

#### Adding a New Backend REST Route
1. Define the route in [backend/app.py](file:///c:/React/SignBridge/backend/app.py) using `@app.route('/api/...', methods=['...'])`.
2. Apply appropriate rate limiters using `@limiter.limit(...)`.
3. If the route mutates models or hardware, verify `check_api_key()` for authorization.
4. Add a corresponding test case in [backend/tests/test_app_llm.py](file:///c:/React/SignBridge/backend/tests/test_app_llm.py).

### 10.2 Developer Rules & Gotchas (Ponytail & Graphify)
1. **Never Break the LIVE / DEMO Fallback**:
   - Components must never assume the Flask backend or Arduino is present. If an API request fails, the UI must fall back gracefully to DEMO typewriter feeds.
2. **Model Priority Order is Sacred**:
   - [TranslatorModel](file:///c:/React/SignBridge/backend/services/translator_model.py) must strictly adhere to the loading hierarchy: Hybrid Ensemble $\rightarrow$ ST-GCN $\rightarrow$ XGBoost $\rightarrow$ Keras $\rightarrow$ Heuristic.
3. **No Vector Database Bloat**:
   - Vector databases (Chroma, Pinecone, FAISS) must NOT be introduced for standard gesture recognition or simple text generation; classical ML classifiers and LLMs handle real-time inference faster with zero database maintenance overhead.
4. **All Frontend API Calls Must Use Relative Paths**:
   - Always call `/api/...` or `/ws` in frontend hooks. Do not hardcode `http://localhost:5000`, which breaks Vite proxying and container deployments.
5. **Always Update the Knowledge Graph**:
   - After making significant structural code changes, run `graphify update .` to keep the AST knowledge graph in `graphify-out/` synchronized.

---

## 11. Quick Start Summary

```bash
# 1. Backend setup & start
cd backend
python -m venv ..\.venv && ..\.venv\Scripts\activate
pip install -r requirements.txt && copy .env.example .env
python app.py  # Starts Waitress on :5000 & WebSocket on :5001

# 2. Frontend setup & start (in a new terminal)
npm install && npm run dev  # Starts Vite on http://localhost:5173
```

---

## 12. Open Questions for the Maintainer

1. **BiLSTM Word Model Retraining**: The 57-class word model currently sits at 51.22% validation accuracy. Is there an active effort to collect more natural video footage for the 34 newer word classes, or should we consider pruning back to the high-accuracy 23-class core vocabulary for production kiosks?
2. **Robotic Hand Motor Calibration**: Are the angle arrays in `ISL_SERVO_MAP` ([arduino_serial.py](file:///c:/React/SignBridge/backend/services/arduino_serial.py#L37)) calibrated for standard 3D-printed tendon hands, or specifically for the scrap-material cardboard hands mentioned in the documentation?
3. **ROS2 Integration Timeline**: Does the team intend to deploy SignBridge on a ROS2 robotic arm platform, and should we transition [ROS2ActuatorStub](file:///c:/React/SignBridge/backend/services/robot_actuator.py#L38) into a fully active `rclpy` node publisher?
4. **Offline Deployment & Local LLM**: For remote installations without stable internet, should we add an optional local lightweight LLM (e.g., Ollama running `llama3.2:3b` or `qwen2.5:3b`) into the fallback chain before the rule-based fallback?