# PROJECT_CONTEXT.md

## Project Overview
- **Project Name**: SignBridge
- **Purpose**: A bidirectional, real-time communication system bridging the gap between Deaf/Hard-of-Hearing individuals and hearing individuals in India using Indian Sign Language (ISL).
- **Main Features**:
  1. **ISL-to-Text/Speech (Path 1 — Deaf Signer)**: Webcam-based AI vision using MediaPipe landmark extraction (42 3D keypoints / 126 coordinates) and hybrid ML models (XGBoost 88.2%, ST-GCN 80.4%, CNN-BiLSTM 87.0%) to translate static letters and dynamic whole-word gestures into live text and audio speech.
  2. **Text/Speech-to-ISL Robot (Path 2 — Hearing Listener)**: Translates spoken or typed voice into simplified ISL keywords, animated visual gesture guides, and physical fingerspelling executed by a dual 5-finger bionic robotic hand setup (driven by Arduino / PySerial).
  3. **LLM Refinement & Simplification Engine**: Dual-provider fallback engine (Groq LPU `llama-3.3-70b-versatile` with automatic fallback to Google Gemini `gemini-1.5-flash`) that reconstructs raw letter streams into fluent sentences and simplifies complex spoken text into robotic sign keywords.
  4. **Touchless Kiosk Operation**: Hands-free auto-send sensor — when the user drops their hands for 1.8 seconds, the completed sentence is automatically spoken aloud and sent into the chat.
  5. **Dual-Mode Operation**: Supports **LIVE** mode (real-time webcam vision & Arduino serial communication) and **DEMO** mode (simulated typewriter feeds and animated gesture reference sheets for presentations).
- **Current Development Status**: Production-ready core. Frontend UI kiosk/dual display, custom React hooks, Flask REST API, PySerial driver, MediaPipe vision pipeline, XGBoost letter model (88.2%), ST-GCN graph model (80.4%), and CNN-BiLSTM word model (87.0%) are fully operational and tested.

---

## Tech Stack
- **Frontend**: React 19, Vite 8, Framer Motion, Lucide React icons, Vanilla CSS (Design system in `index.css` & `assistant.css`).
- **Backend**: Python 3.10+, Flask 3.0, Flask-CORS 5.0, PyTorch 2.x, Scikit-Learn, XGBoost 2.x, NumPy, OpenCV (`cv2`).
- **Vision & Landmark Extraction**: Google MediaPipe Tasks (`HandLandmarker`), OpenCV (`cv2`).
- **Generative AI & LLMs**: Groq SDK (`llama-3.3-70b-versatile`), Google Generative AI (`gemini-1.5-flash`).
- **Hardware Integration**: PySerial (`pyserial`) for USB serial communication with Arduino Uno/Mega driving 10x SG90 micro-servos (Dual 5-Finger Bionic Hands).
- **DevOps & Containerization**: Docker (`Dockerfile.backend`, `Dockerfile.frontend`), `docker-compose.yml`, Kubernetes manifests (`k8s/`), Nginx.
- **Deployment / Dev Server**: Vite dev server (port 5173 with proxy `/api` -> `http://localhost:5000`), Flask WSGI server (port 5000).

---

## Architecture

### Overall Architecture
```
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                           React Frontend (Vite)                         │
 │  - SignBridgeKiosk / DualDisplayScreen / HumanPanel / RobotPanel        │
 │  - Hooks: useISLTranslation, useHandDetection, useGestureRecognition    │
 │  - Touchless Inactivity Hand-Drop Auto-Send (1.8s Countdown)            │
 └────────────────────────────────────┬────────────────────────────────────┘
                                      │ Vite Proxy (/api)
                                      ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                            Flask REST Backend                           │
 │  - app.py (Endpoints: /api/translate, /api/translate/word, /api/llm/*) │
 ├────────────────────────────────────┬────────────────────────────────────┤
 │   TranslatorModel:                 │   ArduinoSerial (PySerial Driver): │
 │   • XGBoost Letter Model (88.2%)   │   • COM3 / /dev/ttyUSB0            │
 │   • ST-GCN Graph Model (80.4%)     │   • SG90 10-Servo PWM Controller   │
 │   • CNN-BiLSTM Word Model (87.0%)  │   • Non-blocking Command Queue     │
 │   • Heuristic Fallback & Gating    │                                    │
 │   • Groq ➔ Gemini LLM Fallback     │                                    │
 └──────────────────┬─────────────────┴──────────────────┬─────────────────┘
                    │                                    │
                    ▼                                    ▼
       MediaPipe Vision Pipeline                Physical Hardware Setup
    (42 3D Landmarks / 126 Floats)             (10 Servos / 2 Bionic Hands)
```

### Folder Structure
```
SignBridge/
├── backend/
│   ├── app.py                      # Main Flask application & REST routes
│   ├── requirements.txt            # Python dependencies
│   ├── train_unified.py            # Unified Pipeline v2 (Extract + Train XGBoost + ST-GCN + CNN-BiLSTM)
│   ├── extract_static_landmarks.py # Static dataset landmark extractor
│   ├── extract_video_landmarks.py  # Video temporal sequence landmark extractor
│   ├── stream_and_extract_hf.py    # Zero-disk Hugging Face ISLRTC dataset streamer
│   ├── ingest_dataset_2.py         # Batch extractor for dataset_2 (521 high-res photos)
│   ├── services/
│   │   ├── translator_model.py     # Hybrid Letter Classifier (XGBoost ➔ ST-GCN ➔ Heuristics)
│   │   ├── word_recognizer.py      # CNN-BiLSTM Dynamic Word Recognizer (23 classes)
│   │   ├── data_loader.py          # Leakage-free dataset partition manager
│   │   ├── feature_extractor.py    # 208-D invariant geometric feature extractor
│   │   └── arduino_serial.py       # PySerial driver for robotic bionic hands
│   ├── models/                     # Trained model weights (.pkl, .pt) & training metadata
│   │   ├── isl_xgboost_model.pkl   # XGBoost Letter Classifier (88.17% test acc)
│   │   ├── isl_stgcn_model.pt      # ST-GCN Graph Classifier (80.38% test acc)
│   │   ├── isl_cnn_lstm_word_model.pt # CNN-BiLSTM Word Classifier (87.04% val acc)
│   │   └── hand_landmarker.task    # Google MediaPipe hand landmark model
│   ├── dataset_collected/          # Canonical landmark JSONs partitioned A-Z
│   └── dataset_words/              # 30-frame temporal word sequences (23 classes)
├── src/
│   ├── App.jsx                     # Core view router
│   ├── index.css                   # Global styles & design tokens
│   ├── components/                 # React UI components
│   │   ├── CameraView.jsx          # Live MediaPipe camera feed & hand tracking HUD
│   │   ├── DualDisplayScreen.jsx   # Split-screen Deaf / Hearing interface
│   │   ├── GestureReferenceSheet.jsx # Interactive ISL visual gesture guide
│   │   ├── HumanPanel.jsx          # Deaf signer input view & sentence buffer
│   │   ├── RobotPanel.jsx          # Hearing speaker view & robot feedback
│   │   ├── SignBridgeKiosk.jsx     # Main interactive kiosk shell
│   │   └── SignLanguageAssistant/  # Assistant sheet & vocabulary cards
│   ├── hooks/                      # Custom React hooks
│   │   ├── useISLTranslation.js    # Central state for feeds, messages, API sync
│   │   ├── useHandDetection.js     # MediaPipe hand tracking state
│   │   ├── useGestureRecognition.js# 300ms temporal smoothing & word/letter inference
│   │   └── useWebcam.js            # Video capture & camera device controller
│   └── data/                       # Static dataset definitions & conversation pools
├── Dockerfile.backend              # Backend Docker container definition
├── Dockerfile.frontend             # Frontend Docker container definition
├── docker-compose.yml              # Multi-container orchestration
├── package.json                    # Frontend dependencies & scripts
├── vite.config.js                  # Vite configuration & backend proxy
├── PROJECT_CONTEXT.md              # Up-to-date project context (this file)
└── AI_MEMORY.md                    # Permanent AI memory & constraints
```

---

## ML Model Architecture & Active Metrics

### 3-Tier Recognition Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         3-Tier ISL Recognition Pipeline                         │
├─────────────────┬───────────────────────────────┬───────────────┬───────────────┤
│ Tier            │ Model                         │ Input         │ Performance   │
├─────────────────┼───────────────────────────────┼───────────────┼───────────────┤
│ 1. Alphabet     │ XGBoost Classifier (.pkl)     │ 1 Frame       │ 88.17% Test   │
│    (A–Z, 0–9)   │ ➔ ST-GCN Graph Model (.pt)    │ (208 features)│ 80.38% Test   │
│                 │ ➔ Heuristic Fallback          │               │ High Resilient│
├─────────────────┼───────────────────────────────┼───────────────┼───────────────┤
│ 2. Words        │ 1D-CNN + BiLSTM (.pt)         │ 30 Frames     │ 87.04% Val    │
│    (23 Classes) │ with Motion Velocity Gating   │ (30 × 126)    │ 23 Vocabulary │
├─────────────────┼───────────────────────────────┼───────────────┼───────────────┤
│ 3. Sentences    │ Groq Llama-3.3-70b ➔          │ Raw letter/   │ Fluent        │
│                 │ Gemini 1.5-Flash fallback     │ word stream   │ sentence      │
└─────────────────┴───────────────────────────────┴───────────────┴───────────────┘
```

### Active Model Performance

| Model Architecture | Artifact | Test / Val Accuracy | Training Samples | Status |
|---|---|---|---|---|
| **XGBoost (Letters A-Z)** | `isl_xgboost_model.pkl` | **88.17% Test Acc** (89.16% Precision) | 52,852 augmented vectors | ✅ Primary Letter Engine |
| **ST-GCN (Graph Letters)** | `isl_stgcn_model.pt` | **80.38% Test Acc** (82.01% Precision) | 52,852 augmented vectors | ✅ Backup Graph Engine |
| **CNN-BiLSTM (Words)** | `isl_cnn_lstm_word_model.pt` | **87.04% Val Acc** (83.8% Overall 1K Set) | 5,750 balanced sequences | ✅ Primary Word Engine |
| **Random Forest (Words)** | `isl_word_classifier.pkl` | **72.22% Val Acc** | 5,750 balanced sequences | ⚠️ Fallback Backup |

---

## Backend REST Endpoints (`http://localhost:5000/api`)

| Method | Endpoint | Description | Request Body | Response Summary |
|---|---|---|---|---|
| **GET** | `/api/health` | Backend health & service connection status | None | `{ status, translator_mode, arduino_connected, groq_available, gemini_available, word_recognizer_available }` |
| **GET** | `/api/model/info` | ML model metadata & active labels | None | `{ model_type, classes, loaded, val_accuracy }` |
| **POST** | `/api/translate` | Translates single-frame landmarks to ISL letter | `{ landmarks: [ {x,y,z}, ... ] }` | `{ letter, confidence, mode, all_scores }` |
| **POST** | `/api/translate/batch` | Translates array of landmark frames | `{ frames: [ [...] ] }` | `{ results: [...], sentence: "..." }` |
| **POST** | `/api/translate/word` | Translates 30-frame sequence to ISL word (CNN-BiLSTM) | `{ frames: [ [126 floats], ...×30 ] }` | `{ word, confidence, margin, motion_velocity, all_scores }` |
| **POST** | `/api/robot/sign` | Transmits text sequence to robotic hands | `{ text: "HELLO" }` | `{ status, text, arduino_connected }` |
| **GET** | `/api/robot/status` | Queries Arduino serial status | None | `{ connected, port, baudrate }` |
| **POST** | `/api/robot/connect` | Initiates serial connection to Arduino | `{ port: "COM3" }` | `{ status, port }` |
| **POST** | `/api/robot/disconnect`| Closes Arduino serial connection | None | `{ status }` |
| **POST** | `/api/llm/refine` | Refines raw letter buffer to fluent sentence | `{ text: "H E L P W A T E R" }` | `{ refined_sentence, llm_provider, fallback_used }` |
| **POST** | `/api/llm/simplify` | Simplifies speech to robot keywords | `{ text: "Where is the doctor sitting?" }` | `{ robot_keywords, llm_provider, fallback_used }` |
| **POST** | `/api/llm/answer` | Answers user questions with domain assistant | `{ text: "Where is washroom?" }` | `{ answer, user_text, llm_provider, fallback_used }` |

---

## Key Business Logic & Algorithms
1. **Geometric Invariants (208-D)**: Normalizes MediaPipe 126 coordinates relative to wrist, computes pairwise finger-joint Euclidean distances, inter-joint cosine angles, and tip-to-palm ratios to achieve scale/distance invariance.
2. **Motion Velocity Gating**: In `word_recognizer.py`, frame-to-frame displacement rate ($\Delta \text{pos} / \Delta t$) is calculated. Hand gestures with motion $<0.020\text{ units/frame}$ are rejected as `static_hand_detected` to prevent false word hallucinations.
3. **Temporal Multi-Frame Voting**: In `useGestureRecognition.js`, predictions must achieve consensus across a 300ms window before committing to the sentence buffer.
4. **Touchless Hand-Drop Auto-Send**: If hands are not detected in the camera frame for 1.8 seconds, a countdown timer triggers automatic text-to-speech synthesis and sends the message into the dialogue stream.
5. **LLM Dual-Provider Fallback**: Attempts Groq `llama-3.3-70b-versatile` first ($<100\text{ms}$ latency); on rate-limit or network failure, automatically falls back to Google Gemini `gemini-1.5-flash`.
6. **Robotic Hand Motor Control**: Translates text into 10-servo angle ASCII packets (e.g. `<90,0,180,45,90|0,90,90,180,0>`) written over PySerial to SG90 micro-servos on Arduino.
