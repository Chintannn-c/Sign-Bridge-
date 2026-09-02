# PROJECT_CONTEXT.md

## 1. Project Overview
- **Project Name**: SignBridge
- **Purpose**: An advanced, bidirectional Indian Sign Language (ISL) translation and robotic communication system designed to bridge the communication gap between Deaf/Hard-of-Hearing individuals and hearing individuals.
- **Core Communication Channels**:
  1. **ISL-to-Speech / Text (Path 1 — Deaf User)**: Real-time webcam AI vision extracting 42 3D MediaPipe landmarks (126 coordinates) and classifying both static manual alphabet fingerspelling (`A`–`Z`, `0`–`9`) and dynamic temporal whole-word gestures (`HELLO`, `NAMASTE`, `THANK_YOU`, etc.) into live text and audio speech.
  2. **Speech / Text-to-ISL Robot (Path 2 — Hearing User)**: Speech-to-Text transcribes the hearing user's voice, a dual-provider LLM simplifies the text into concise ISL keywords, and an Arduino-driven **dual 5-finger bionic robotic hand system** physically executes the fingerspelling while the UI displays animated gesture reference guides.
  3. **Generative AI Refinement & Assistant**: Dual-provider LLM engine (**Groq LPU `llama-3.3-70b-versatile`** with automatic fallback to **Google Gemini `gemini-1.5-flash`**) that converts raw letter buffers into fluent sentences, simplifies speech for robotic signing, and answers user domain questions.
  4. **Touchless Interaction**: Built-in inactivity hand-drop auto-send timer (1.8s countdown) that automatically speaks and commits completed sentences when hands leave the camera frame.
  5. **Dual-Mode Operation**:
     - **LIVE Mode**: Real Flask REST backend, live MediaPipe vision inference, and PySerial Arduino motor actuation.
     - **DEMO Mode**: Simulated typewriter feeds, animated visual reference sheets, and interactive practice cards for demonstrations without hardware.
- **Current Status**: Fully functional, evaluated, and production-ready.

---

## 2. Tech Stack

| Layer | Technologies & Dependencies |
|---|---|
| **Frontend UI** | React 19, Vite 8, Framer Motion, Lucide React icons, Vanilla CSS Design System (`index.css`, `assistant.css`) |
| **Backend API** | Python 3.10+, Flask 3.0, Flask-CORS 5.0, PyTorch 2.x, Scikit-Learn 1.4+, XGBoost 2.x, NumPy, OpenCV (`cv2`) |
| **Vision & Landmarks** | Google MediaPipe Tasks (`HandLandmarker`), MediaPipe Holistic geometry, OpenCV image processing |
| **Generative AI** | Groq Python SDK (`llama-3.3-70b-versatile`), Google Generative AI SDK (`gemini-1.5-flash`) |
| **Hardware & Robotics** | PySerial (`pyserial`), Arduino Uno / Mega (10x SG90 Micro-Servos for Dual 5-Finger Bionic Hands) |
| **DevOps & Deploy** | Docker (`Dockerfile.frontend`, `Dockerfile.backend`), `docker-compose.yml`, Kubernetes (`k8s/`), Nginx |
| **Local Ports** | Frontend: `http://localhost:5173` (Vite with `/api` proxy) \| Backend: `http://localhost:5000` (Flask WSGI) |

---

## 3. System Architecture & Data Flow

```
 ┌─────────────────────────────────────────────────────────────────────────────────┐
 │                               React Frontend (Vite)                             │
 │  • SignBridgeKiosk / HumanPanel (Deaf) / RobotPanel (Hearing)                   │
 │  • MediaPipe CameraView with Real-Time Landmark Overlay & HUD                   │
 │  • Touchless Inactivity Hand-Drop Auto-Send (1.8s Countdown)                    │
 │  • Interactive SignLanguageAssistant (Practice, Learn, History, DataCollection) │
 └────────────────────────────────────────┬────────────────────────────────────────┘
                                          │ Vite Proxy (/api)
                                          ▼
 ┌─────────────────────────────────────────────────────────────────────────────────┐
 │                                Flask REST Backend                               │
 │  • app.py (Endpoints: /api/translate, /api/translate/word, /api/llm/*, /robot/*)│
 ├────────────────────────────────────────┬────────────────────────────────────────┤
 │  AI / ML Translation Core:             │  Robotics & Hardware:                  │
 │  • Tier 1: XGBoost Letters (88.17%)    │  • arduino_serial.py (PySerial)        │
 │  • Tier 1B: ST-GCN Graph (80.38%)      │  • 10-Servo PWM ASCII Controller       │
 │  • Tier 2: CNN-BiLSTM Words (87.04%)   │  • Auto-Reconnecting COM Driver        │
 │  • Tier 3: Groq ➔ Gemini LLM Fallback  │  • Calibrated Flexion/Extension Angles │
 └────────────────────┬───────────────────┴───────────────────┬────────────────────┘
                      │                                       │
                      ▼                                       ▼
         MediaPipe Vision Pipeline                   Physical Arduino Setup
     (42 3D Landmarks / 126 Floats)                (10 Servos / 2 Bionic Hands)
```

### End-to-End Data Flows:
1. **Path 1 (Deaf Signer $\rightarrow$ Speech/Text)**:
   Webcam $\rightarrow$ `CameraView.jsx` $\rightarrow$ MediaPipe HandLandmarker $\rightarrow$ 42 3D coordinates (126 floats) $\rightarrow$ `POST /api/translate` (or `POST /api/translate/word`) $\rightarrow$ XGBoost / CNN-BiLSTM model $\rightarrow$ 300ms temporal consensus smoothing $\rightarrow$ Sentence Buffer $\rightarrow$ Touchless 1.8s hand-drop trigger $\rightarrow$ Web Speech Audio TTS + Chat Transcript.
2. **Path 2 (Hearing User $\rightarrow$ Robot / ISL Reference)**:
   Hearing User Speech / Typing $\rightarrow$ `RobotPanel.jsx` $\rightarrow$ `POST /api/llm/simplify` (Groq/Gemini extracts keywords) $\rightarrow$ `POST /api/robot/sign` $\rightarrow$ `arduino_serial.py` $\rightarrow$ Serial ASCII Packet `<L0..L4|R0..R4>` $\rightarrow$ Arduino actsuate SG90 servos + Frontend displays animated visual gesture reference.
3. **Path 3 (LLM Sentence Refinement)**:
   Raw fingerspelled letters (e.g. `"H E L P W A T E R"`) $\rightarrow$ `POST /api/llm/refine` $\rightarrow$ Groq Llama-3.3-70b (sub-100ms LPU) $\rightarrow$ Fallback to Gemini 1.5-Flash $\rightarrow$ Grammatical, natural English/Hindi sentence (`"Please bring me some water."`).

---

## 4. Complete Directory & File Map

```
SignBridge/
├── backend/
│   ├── app.py                      # Flask REST API server & endpoint definitions
│   ├── requirements.txt            # Python dependencies (Torch, XGBoost, Flask, MediaPipe, etc.)
│   ├── train_unified.py            # Unified Pipeline v2 (extracts & trains XGBoost, ST-GCN, CNN-BiLSTM)
│   ├── extract_static_landmarks.py # Static dataset batch landmark extractor
│   ├── extract_video_landmarks.py  # Video temporal landmark extractor with vectorized NumPy indexing
│   ├── stream_and_extract_hf.py    # Zero-disk Hugging Face streaming ingest pipeline for ISLRTC dictionary
│   ├── ingest_dataset_2.py         # Batch extractor for 521 real-world high-res photos in dataset_2/
│   ├── services/
│   │   ├── translator_model.py     # Hybrid Letter Classifier (XGBoost ➔ ST-GCN ➔ Heuristics)
│   │   ├── word_recognizer.py      # Dynamic CNN-BiLSTM Word Recognizer with motion velocity gating
│   │   ├── data_loader.py          # Leakage-free dataset partition manager (train/val/test routing)
│   │   ├── feature_extractor.py    # 208-D geometric invariant feature extraction engine
│   │   ├── groq_manager.py         # Groq LPU Llama-3.3-70b manager with multi-key rotation
│   │   ├── gemini_manager.py       # Google Gemini 1.5-Flash manager with fallback retry
│   │   └── arduino_serial.py       # Non-blocking PySerial driver for 10-servo robotic bionic hands
│   ├── models/                     # Saved model artifacts & JSON training metadata
│   │   ├── isl_xgboost_model.pkl   # XGBoost Letter Classifier (88.17% test accuracy)
│   │   ├── isl_stgcn_model.pt      # ST-GCN Graph Classifier (80.38% test accuracy)
│   │   ├── isl_cnn_lstm_word_model.pt # CNN-BiLSTM Dynamic Word Model (87.04% val accuracy)
│   │   ├── isl_word_classifier.pkl # Random Forest Word Classifier (72.22% val accuracy)
│   │   ├── hand_landmarker.task    # Google MediaPipe hand landmark model
│   │   ├── xgb_training_meta.json  # XGBoost evaluation metrics & per-class F1 breakdown
│   │   ├── stgcn_training_meta.json# ST-GCN evaluation metrics
│   │   └── cnn_lstm_training_meta.json # CNN-BiLSTM evaluation metrics
│   ├── dataset_collected/          # Partitioned landmark JSONs for letters A–Z and digits 0–9
│   ├── dataset_words/              # 30-frame temporal sequences for 23 ISL words
│   └── tests/                      # Python backend unit tests (20 passing tests)
│       ├── test_groq_manager.py    # Groq manager rate-limiting & fallback unit tests
│       └── test_gemini_manager.py  # Gemini manager multi-key & error handling tests
├── src/
│   ├── App.jsx                     # Core application view switcher
│   ├── main.jsx                    # React 19 application root
│   ├── index.css                   # Global CSS design system, themes, and kiosk layouts
│   ├── components/                 # React UI Components
│   │   ├── SignBridgeKiosk.jsx     # Master kiosk view wrapper & mode switcher
│   │   ├── HumanPanel.jsx          # Deaf signer interface (video feed, sentence buffer, TTS)
│   │   ├── RobotPanel.jsx          # Hearing user interface (STT input, robot feedback, LLM answer)
│   │   ├── CameraView.jsx          # Live video canvas, MediaPipe tracker, HUD bounding boxes
│   │   ├── GestureReferenceSheet.jsx # Interactive A–Z gesture cards & servo angle diagrams
│   │   └── SignLanguageAssistant/  # Expandable AI Assistant Bottom Sheet
│   │       ├── AssistantBottomSheet.jsx # Bottom sheet container
│   │       ├── HandTrackingOverlay.jsx  # Hand skeleton overlay
│   │       ├── SentenceBuilder.jsx      # Interactive sentence builder & keyword tags
│   │       ├── assistant.css            # Assistant modal styling
│   │       └── tabs/
│   │           ├── LiveDetectionTab.jsx # Real-time detection inspector & confidence breakdown
│   │           ├── PracticeTab.jsx      # Gesture practice with real-time feedback
│   │           ├── LearnTab.jsx         # ISL vocabulary study cards
│   │           ├── HistoryTab.jsx       # Chat conversation history
│   │           └── DataCollectionTab.jsx# User gesture recording & custom dataset collection
│   ├── hooks/                      # Custom React Hooks
│   │   ├── useISLTranslation.js    # Master state for translation, message feeds, and API sync
│   │   ├── useGestureRecognition.js# 300ms temporal smoothing, motion filters, word/letter gating
│   │   ├── useHandDetection.js     # MediaPipe HandLandmarker lifecycle & coordinate extraction
│   │   ├── useWebcam.js            # Video stream acquisition & camera device controls
│   │   └── useAssistantHistory.js  # Assistant chat session logging & localStorage persistence
│   ├── data/                       # Static dataset definitions & conversation pools
│   │   ├── gestureData.js          # ISL gesture definitions, SVG finger paths, servo presets
│   │   └── islConversations.js     # Pre-built conversational dialogue pools for demo mode
│   └── utils/                      # Helper utilities
├── k8s/                            # Kubernetes Deployment Manifests
│   ├── backend-deployment.yaml     # Flask backend K8s deployment & service
│   ├── frontend-deployment.yaml    # React frontend K8s deployment & service
│   ├── configmap.yaml              # Environment configuration map
│   ├── secret.yaml                 # API secret keys definition
│   └── ingress.yaml                # Ingress routing rules
├── Dockerfile.backend              # Production multi-stage Python Flask Dockerfile
├── Dockerfile.frontend             # Production Nginx multi-stage React Dockerfile
├── docker-compose.yml              # Local multi-service Docker Compose specification
├── nginx.conf                      # Nginx reverse proxy configuration
├── vite.config.js                  # Vite configuration, build optimizer, and API proxy
├── package.json                    # Node dependencies & npm scripts
├── pyrightconfig.json              # Python static type checker configuration
└── AI_MEMORY.md                    # Permanent AI behavioral memory & constraints
```

---

## 5. Machine Learning Models & Active Performance Metrics

SignBridge operates a 3-tier ensemble recognition pipeline:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 3-Tier Recognition Architecture                                 │
├───────────────────┬───────────────────────────────┬─────────────────┬───────────────────────────┤
│ Tier              │ Active Model Architecture     │ Input Window    │ Verified Benchmark Score  │
├───────────────────┼───────────────────────────────┼─────────────────┼───────────────────────────┤
│ 1. Alphabet       │ XGBoost Ensemble (.pkl)       │ 1 Frame         │ 88.17% Test Accuracy      │
│    (A–Z, 0–9)     │ ➔ ST-GCN Graph (.pt)          │ (208 Geometric) │ 80.38% Test Accuracy      │
│                   │ ➔ Heuristic Geometry          │                 │ 100% Operational Uptime   │
├───────────────────┼───────────────────────────────┼─────────────────┼───────────────────────────┤
│ 2. Dynamic Words  │ 1D-CNN + BiLSTM (.pt)         │ 30 Frames       │ 87.04% Val Accuracy       │
│    (23 Classes)   │ with Motion Velocity Gating   │ (30 × 126)      │ 83.80% 1,000-Seq Benchmark│
├───────────────────┼───────────────────────────────┼─────────────────┼───────────────────────────┤
│ 3. Sentences &    │ Groq Llama-3.3-70b            │ Raw letter /    │ Sub-100ms Latency         │
│    Simplification │ ➔ Google Gemini 1.5-Flash     │ word stream     │ Fluent Grammatical Text   │
└───────────────────┴───────────────────────────────┴─────────────────┴───────────────────────────┘
```

### Detailed Model Metrics

#### 🔤 Alphabet Classifier: XGBoost (`isl_xgboost_model.pkl`)
* **Input**: 208-D invariant geometric features (pairwise Euclidean distance ratios, inter-joint cosine angles, wrist-to-tip vectors).
* **Training Dataset**: 13,213 raw samples augmented to **52,852 vectors** (RealSign, Mendeley, Session captures, and 521 photos from `dataset_2`).
* **Test Dataset**: 4,903 strictly held-out unseen samples.
* **Test Accuracy**: **88.17%** \| **Test Precision**: **89.16%** \| **Test Macro F1**: **87.98%**.
* **High-Accuracy Classes**: `B` (0.994 F1), `Z` (0.995 F1), `K` (0.985 F1), `A` (0.983 F1), `R` (0.980 F1), `C` (0.968 F1), `F` (0.962 F1), `N` (0.953 F1), `V` (0.940 F1), `J` (0.940 F1), `H` (0.939 F1), `L` (0.936 F1), `M` (0.928 F1), `U` (0.906 F1).

#### 🔤 Spatial-Temporal Graph Model: ST-GCN (`isl_stgcn_model.pt`)
* **Input**: Hand skeleton graph adjacency matrix (21 nodes per hand).
* **Test Accuracy**: **80.38%** \| **Test Precision**: **82.01%** \| **Test Macro F1**: **80.36%**.

#### 🗣️ Dynamic Whole-Word Model: 1D-CNN + BiLSTM (`isl_cnn_lstm_word_model.pt`)
* **Input**: 30-frame sliding window (3,780 flattened MediaPipe coordinates).
* **Architecture**: Conv1d(126 $\rightarrow$ 64) $\rightarrow$ BatchNorm $\rightarrow$ Conv1d(64 $\rightarrow$ 128) $\rightarrow$ BiLSTM(128 hidden, 2 layers) $\rightarrow$ Mean Pooling $\rightarrow$ BatchNorm $\rightarrow$ Linear(256 $\rightarrow$ 64) $\rightarrow$ Linear(64 $\rightarrow$ 23).
* **Vocabulary (23 Classes)**: `AGAIN`, `BYE_BYE`, `DEAF`, `DOCTOR`, `FOOD`, `HEARING`, `HELLO`, `HELP`, `INDIA`, `LANGUAGE`, `MAN`, `ME`, `NAMASTE`, `PLEASE`, `SIGN`, `SORRY`, `THANK_YOU`, `WASHROOM`, `WATER`, `WELCOME`, `WHERE`, `WOMAN`, `YOU`.
* **Validation Accuracy**: **87.04%** (83.80% across 1,000 full sequence evaluation).
* **Motion Gate**: Hand movements $<0.020\text{ units/frame}$ are rejected as `static_hand_detected` to eliminate resting-hand hallucinations.

---

## 6. Backend REST API Reference

All backend endpoints are hosted at `http://localhost:5000/api` (proxied by Vite via `/api`):

| Method | Endpoint | Description | Request Payload | Response Schema |
|---|---|---|---|---|
| **GET** | `/api/health` | Service health, model status, and hardware links | None | `{ status: "ok", translator_mode: "hybrid_ensemble", arduino_connected: bool, gemini_available: bool, groq_available: bool, word_recognizer_available: bool }` |
| **GET** | `/api/model/info` | Detailed ML models and labels metadata | None | `{ model_type: "xgboost", classes: [...], loaded: true, val_accuracy: 0.8817 }` |
| **POST** | `/api/translate` | Translates 1 frame of landmarks to ISL letter | `{ landmarks: [ {x,y,z}, ... ] }` | `{ letter: "A", confidence: 0.98, mode: "xgboost", all_scores: {...} }` |
| **POST** | `/api/translate/batch` | Translates batch of frames with sentence output | `{ frames: [ [...] ] }` | `{ results: [...], sentence: "HELLO" }` |
| **POST** | `/api/translate/word` | Classifies 30-frame sequence to ISL word | `{ frames: [ [126 floats] × 30 ] }` | `{ word: "NAMASTE", confidence: 0.98, margin: 0.45, motion_velocity: 0.042, rejected: false }` |
| **POST** | `/api/robot/sign` | Sends text command to physical robotic hands | `{ text: "HELLO" }` | `{ status: "ok", text: "HELLO", arduino_connected: bool }` |
| **GET** | `/api/robot/status` | Queries Arduino COM port connection | None | `{ connected: bool, port: "COM3", baudrate: 115200 }` |
| **POST** | `/api/robot/connect` | Connects PySerial driver to COM port | `{ port: "COM3" }` | `{ status: "connected", port: "COM3" }` |
| **POST** | `/api/robot/disconnect` | Closes Arduino PySerial connection | None | `{ status: "disconnected" }` |
| **POST** | `/api/llm/refine` | Refines raw letter buffer to fluent sentence | `{ text: "H E L P W A T E R" }` | `{ refined_sentence: "Please help me get water.", llm_provider: "groq", fallback_used: false }` |
| **POST** | `/api/llm/simplify` | Simplifies speech into robot keywords | `{ text: "Where is the doctor sitting?" }` | `{ robot_keywords: "WHERE DOCTOR", llm_provider: "groq", fallback_used: false }` |
| **POST** | `/api/llm/answer` | Answers domain questions for assistant sheet | `{ text: "Where is washroom?" }` | `{ answer: "The washroom is down the hall.", user_text: "...", llm_provider: "groq" }` |

---

## 7. Key Mathematical & Business Logic Algorithms

1. **208-D Invariant Geometric Features (`feature_extractor.py`)**:
   - **Wrist-Centric Normalization**: Translates coordinate origin to the wrist (landmark 0).
   - **Scale Invariance**: Normalizes all coordinates by dividing by the Euclidean distance from wrist to middle MCP joint (landmark 9).
   - **Inter-Joint Distance Ratios**: Computes pairwise distances between fingertips and palm base.
   - **Joint Cosine Angles**: Calculates 3D angles at PIP and DIP joints to accurately distinguish finger curls (e.g. `C` vs `O`, `A` vs `S`).
2. **Kinematic Motion Velocity Gating (`word_recognizer.py`)**:
   $$\text{Velocity} = \frac{1}{N-1} \sum_{t=1}^{N-1} \|\mathbf{p}_{t} - \mathbf{p}_{t-1}\|_2$$
   If $\text{Velocity} < 0.020$, the gesture is rejected as `static_hand_detected`.
3. **Temporal Multi-Frame Voting (`useGestureRecognition.js`)**:
   - Commits a letter or word only after it maintains consistent top-1 prediction across a **300ms temporal window** (9–12 frames), eliminating single-frame visual glitches.
4. **Touchless Hand-Drop Auto-Send**:
   - Monitored by `useGestureRecognition.js`: When no hands are detected in the active frame, a **1.8-second countdown** is initiated. When it reaches 0, the sentence buffer is committed, spoken via Web Speech TTS, and broadcast to the dialogue panel.
5. **Dual-Provider LLM Fallback (`groq_manager.py` & `gemini_manager.py`)**:
   - Primary request sent to Groq LPU (`llama-3.3-70b-versatile`, latency $<100\text{ms}$).
   - Catches rate-limits (`429`), timeouts, or network errors and transparently fails over to Google Gemini `gemini-1.5-flash` without interrupting user experience.

---

## 8. Physical Hardware Setup (Dual Bionic Hands)

* **Physical Layout**: Dual 5-finger hands (Left & Right) with 10 independent degrees of freedom.
* **Actuators**: 10x TowerPro SG90 micro-servos (5 per hand) mounted on acrylic/3D printed tendon linkages.
* **Microcontroller**: Arduino Uno / Mega connected over USB serial (`115200` baud).
* **Communication Protocol**: Formatted ASCII strings `<L0,L1,L2,L3,L4|R0,R1,R2,R3,R4>` where values range from $0^\circ$ (full extension) to $180^\circ$ (full flexion).
* **Driver Architecture**: Non-blocking queued background thread in `arduino_serial.py` preventing serial I/O from blocking Flask REST endpoints.

---

## 9. DevOps, Deployment & Verification

- **Dockerization**:
  - `Dockerfile.frontend`: Multi-stage build (Node 20 Alpine builder $\rightarrow$ Nginx Alpine runtime).
  - `Dockerfile.backend`: Python 3.10 slim container with PyTorch, OpenCV, XGBoost, and MediaPipe dependencies pre-installed.
  - `docker-compose.yml`: Orchestrates frontend (`:80`), backend (`:5000`), and volume mappings.
- **Kubernetes**: Complete manifests in `k8s/` (`backend-deployment.yaml`, `frontend-deployment.yaml`, `ingress.yaml`, `configmap.yaml`, `secret.yaml`).
- **Automated Unit Testing**: 20/20 backend unit tests passing in `backend/tests/` verifying Groq manager, Gemini manager, rate-limiters, model fallbacks, and serial recovery.
- **Production Build**: `npm run build` succeeds in under 3.2s with zero TypeScript/linting errors.

---

## 10. Complete Project Change Log

- **2026-08-07**: Initial system design and Flask REST skeleton.
- **2026-08-15**: Implemented React 19 split-screen kiosk UI with MediaPipe live canvas overlay.
- **2026-08-20**: Created non-blocking PySerial Arduino driver and SG90 servo angle maps.
- **2026-08-24**: Implemented dual-provider LLM engine (Groq Llama-3.3-70b $\rightarrow$ Google Gemini 1.5-Flash).
- **2026-08-25**: **Unified ML Pipeline Upgrade v2**:
  - Implemented `train_unified.py` combining RealSign, Mendeley, and session datasets.
  - Trained **XGBoost Letter Classifier** to **88.17% test accuracy** (52,852 augmented samples).
  - Trained **ST-GCN Graph Letter Classifier** to **80.38% test accuracy**.
  - Built zero-disk Hugging Face streaming ingest pipeline (`stream_and_extract_hf.py`) for the ISLRTC national dictionary.
  - Trained **CNN-BiLSTM Dynamic Word Model** to **87.04% validation accuracy** across 23 classes.
  - Implemented motion velocity gating ($0.020$) and margin filtering ($0.08$) to eliminate resting-hand hallucinations.
  - Ingested **`dataset_2`** (521 high-resolution phone captures across all 26 letters).
  - Added Touchless Hand-Drop Auto-Send (1.8s countdown) and 300ms temporal consensus smoothing.
- **2026-09-02**: Comprehensive full-codebase audit and `PROJECT_CONTEXT.md` synchronization.