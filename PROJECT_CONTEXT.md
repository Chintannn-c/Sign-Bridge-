# PROJECT_CONTEXT.md

## Project Overview
- **Project Name**: SignBridge
- **Purpose**: A dual-communication translation system bridging the communication gap between Deaf/Hard-of-Hearing individuals and hearing people in India using Indian Sign Language (ISL).
- **Main Features**:
  1. **ISL-to-Text/Speech (Path 1)**: Webcam-based AI vision using MediaPipe landmark extraction and ML models (Random Forest / Deep Learning / Heuristic) to translate ISL gestures into text and speech output.
  2. **Text/Speech-to-ISL Robot (Path 2)**: Translates spoken/typed text into fingerspelling and gestures physically performed by a scrap-material dual robotic hand setup (driven by Arduino / PySerial).
  3. **LLM Refinement & Simplification Engine**: Dual-provider LLM engine (Groq Llama-3.3-70b with fallback to Google Gemini 1.5-Flash) that refines raw ISL letter buffers into fluent sentences and simplifies complex spoken text into robotic keywords.
  4. **Dual-Mode Operation**: Supports **LIVE** mode (real Flask ML model & PySerial Arduino connection) and **DEMO** mode (simulated typewriter feeds and animated gesture reference sheets for UI demos).
- **Current Development Status**: Active development. Frontend UI kiosk/dual display, custom React hooks, Flask backend REST API, PySerial driver, MediaPipe vision pipeline, and LLM fallback integration are operational.

---

## Tech Stack
- **Frontend**: React 19, Vite 8, Framer Motion, Lucide React icons, Vanilla CSS (Design system in `index.css` & `assistant.css`).
- **Backend**: Python 3, Flask 3.0, Flask-CORS 5.0.
- **Database**: In-memory state & `localStorage` on frontend; static JSON files for servo mappings; SQLite (planned single database for persistence).
- **AI/ML Libraries**: MediaPipe (Holistic / Hand landmarks), Scikit-Learn (Random Forest / KNN), XGBoost (Gradient Boosted Trees), TensorFlow / Keras (Dense NN + Bi-LSTM), NumPy, Groq SDK (`groq`), Google Generative AI (`google-generativeai`).
- **Hardware Integration**: PySerial (`pyserial`) for USB serial communication with Arduino Uno/Mega driving SG90 micro servos.
- **Authentication**: None currently (local kiosk / client-server deployment).
- **Deployment / Dev Server**: Vite dev server (port 5173 with proxy `/api` -> `http://localhost:5000`), Flask WSGI server (port 5000).

---

## Architecture

### Overall Architecture
```
┌─────────────────────────────────────────────────────────────────────────┐
│                           React Frontend (Vite)                         │
│  - SignBridgeKiosk / DualDisplayScreen / HumanPanel / RobotPanel        │
│  - Hooks: useISLTranslation, useHandDetection, useGestureRecognition    │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ Vite Proxy (/api)
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            Flask REST Backend                           │
│  - app.py (Endpoints: /api/translate, /api/robot/*, /api/llm/*)        │
├────────────────────────────────────┬────────────────────────────────────┤
│   TranslatorModel (ML / Heuristic) │   ArduinoSerial (PySerial Driver)  │
│   Groq -> Gemini Fallback Engine   │   Connected via COM / /dev/ttyUSB  │
└──────────────────┬─────────────────┴──────────────────┬─────────────────┘
                   │                                    │
                   ▼                                    ▼
       MediaPipe ML Pipeline                    Physical Arduino
    (ISL Gesture Classifier)                  (10 Servos / 2 Hands)
```

### Folder Structure
```
SignBridge/
├── backend/
│   ├── app.py                      # Main Flask application & routes
│   ├── requirements.txt            # Python dependencies
│   ├── train_model.py              # Keras Dense NN training script
│   ├── train_model_xgb.py          # XGBoost alphabet classifier training script
│   ├── train_model_lstm.py         # Bi-LSTM word sequence classifier training script
│   ├── import_dataset.py           # Dataset processing utility
│   ├── services/
│   │   ├── translator_model.py     # ML & Heuristic ISL recognition service (XGBoost → Keras → Heuristic)
│   │   ├── word_recognizer.py      # Bi-LSTM temporal word recognition service
│   │   └── arduino_serial.py       # PySerial driver for robotic hands
│   ├── models/                     # Saved model artifacts (.pkl, .h5)
│   └── dataset_words/              # Multi-frame word gesture recordings (future)
├── src/
│   ├── App.jsx                     # Core application view switcher
│   ├── index.css                   # Global styles & design tokens
│   ├── components/                 # React UI components
│   │   ├── CameraFeed.jsx          # Live camera feed element
│   │   ├── CameraView.jsx          # MediaPipe camera integration view
│   │   ├── DualDisplayScreen.jsx   # Side-by-side Deaf/Hearing interface
│   │   ├── GestureReferenceSheet.js# ISL alphabet & gesture guide
│   │   ├── HumanPanel.jsx          # Deaf signer input view
│   │   ├── RobotPanel.jsx          # Hearing speaker / robot output view
│   │   ├── SignBridgeKiosk.jsx     # Main interactive kiosk launcher
│   │   └── SignLanguageAssistant/  # Assistant sheet, tabs, & sentence builder
│   ├── hooks/                      # Custom React hooks
│   │   ├── useISLTranslation.js    # Central state for feeds, messages, API sync
│   │   ├── useHandDetection.js     # MediaPipe hand tracking state
│   │   ├── useGestureRecognition.js# Real-time landmark processing
│   │   └── useAssistantHistory.js  # Assistant chat session management
│   └── data/                       # Static datasets & demo pools
│       ├── gestureData.js          # ISL gesture definitions & SVG paths
│       └── islConversations.js     # Simulated conversation pools
├── public/                         # Static web assets
├── package.json                    # Frontend dependencies & scripts
├── vite.config.js                  # Vite configuration & backend proxy
├── plan.txt                        # Implementation roadmap
├── PROJECT_CONTEXT.md              # Long-term project context (this file)
└── AI_MEMORY.md                    # Permanent AI memory & design constraints
```

### Data & API Flow
1. **Path 1 (ISL → Speech/Text)**:
   Webcam → `CameraView.jsx` → MediaPipe Hands/Holistic → 42 landmark points `(x,y,z)` → `POST /api/translate` → `translator_model.py` (XGBoost → Keras → Heuristic priority) → ISL Letter prediction → React state update → Speech Synthesis (TTS).
2. **Path 1b (ISL Word → Text)** *(future)*:
   30-frame landmark buffer → `POST /api/translate/word` → `word_recognizer.py` (Bi-LSTM) → Whole-word prediction (HELLO, NAMASTE, etc.).
3. **Path 2 (Text/Speech → Robot)**:
   Hearing User Speech/Type → `RobotPanel.jsx` → `POST /api/llm/simplify` (Groq/Gemini extracts keywords) → `POST /api/robot/sign` → `arduino_serial.py` → Serial Bytes → Arduino Servos actuate robot fingers.

---

## ML Model Architecture

### 3-Tier Recognition Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         3-Tier ISL Recognition Pipeline                         │
├─────────────────┬───────────────────────────────┬───────────────┬───────────────┤
│ Tier            │ Model                         │ Input         │ Output        │
├─────────────────┼───────────────────────────────┼───────────────┼───────────────┤
│ 1. Alphabet     │ XGBoost Classifier (.pkl)     │ 1 Frame       │ A-Z Letter    │
│                 │ → Keras Dense NN (.h5)        │ (126 features)│               │
│                 │ → Heuristic Fallback          │               │               │
├─────────────────┼───────────────────────────────┼───────────────┼───────────────┤
│ 2. Words        │ Bi-LSTM (2-layer, .h5)        │ 30 Frames     │ Whole word    │
│                 │ (future — needs dataset)      │ (30 × 126)    │ (HELLO, etc.) │
├─────────────────┼───────────────────────────────┼───────────────┼───────────────┤
│ 3. Sentences    │ Groq Llama-3.3-70b →          │ Raw letter/   │ Fluent        │
│                 │ Gemini 1.5-Flash fallback     │ word stream   │ sentence      │
└─────────────────┴───────────────────────────────┴───────────────┴───────────────┘
```

### Tier 1: Alphabet Recognition (Single-Frame)
- **Primary Model**: XGBoost gradient-boosted tree classifier
  - Input: 126 normalized landmark features (42 points × 3 coordinates)
  - Output: Probability distribution over 26 ISL letters (A–Z)
  - Training script: [train_model_xgb.py](file:///c:/React/SignBridge/backend/train_model_xgb.py)
  - Model file: `backend/models/isl_xgboost_model.pkl`
  - **Why XGBoost**: Captures non-linear finger joint relationships better than Dense NN; 0.2ms inference on CPU
- **Secondary Model**: Keras Dense NN (3-layer MLP with BatchNorm & Dropout)
  - Training script: [train_model.py](file:///c:/React/SignBridge/backend/train_model.py)
  - Model file: `backend/models/isl_gesture_model.h5`
  - Current accuracy: **25.5%** (needs more diverse training data)
- **Tertiary Fallback**: Rule-based heuristic using Mendeley ISL landmark descriptions

### Tier 2: Word Recognition (Multi-Frame Temporal)
- **Model**: 2-layer Bidirectional LSTM
  - Input: (30, 126) — 30 consecutive frames × 126 features
  - Output: Probability distribution over word classes
  - Training script: [train_model_lstm.py](file:///c:/React/SignBridge/backend/train_model_lstm.py)
  - Model file: `backend/models/isl_lstm_word_model.h5`
  - Service: [word_recognizer.py](file:///c:/React/SignBridge/backend/services/word_recognizer.py)
  - **Status**: Infrastructure ready; requires multi-frame word dataset collection
  - **Why Bi-LSTM**: Reads landmark positions forward and backward in time, recognizing hand trajectories for dynamic signs

### Tier 3: Sentence Refinement (LLM)
- **Primary**: Groq Llama-3.3-70b-versatile (sub-100ms via LPU hardware)
- **Fallback**: Google Gemini 1.5-Flash
- Converts raw noisy letter buffers → fluent English sentences
- Also handles text simplification for robot hand output

### Model Loading Priority in translator_model.py
```
XGBoost (.pkl) found?  ──→ YES → mode = 'xgboost'  (fastest, best accuracy)
        │ NO
        ▼
Keras (.h5) found?     ──→ YES → mode = 'deep_learning'
        │ NO
        ▼
Heuristic fallback          → mode = 'heuristic'
```

### Current Training Metrics

| Model | Validation Accuracy | Training Samples | Status |
| :--- | :--- | :--- | :--- |
| Keras Dense NN (current `.h5`) | 25.5% | 5,460 (augmented) | ❌ Low accuracy — needs more data |
| XGBoost (when trained) | Expected 85-99% | Same dataset | ⏳ Ready to train |
| Bi-LSTM Words | N/A | No dataset yet | ⏳ Needs word dataset collection |
---

## Important Components

### Frontend Components
- **`useISLTranslation.js`** ([file:///c:/React/SignBridge/src/hooks/useISLTranslation.js]):
  - *Purpose*: Core custom hook managing dual-stream translation feeds, live/demo modes, message logs, and Flask API health monitoring.
- **`DualDisplayScreen.jsx`** ([file:///c:/React/SignBridge/src/components/DualDisplayScreen.jsx]):
  - *Purpose*: Split-screen UI rendering `HumanPanel` (Deaf user) and `RobotPanel` (Hearing user) side-by-side.
- **`CameraView.jsx`** ([file:///c:/React/SignBridge/src/components/CameraView.jsx]):
  - *Purpose*: Integrates browser camera stream with MediaPipe hand tracking overlay and frames capture.
- **`GestureReferenceSheet.jsx`** ([file:///c:/React/SignBridge/src/components/GestureReferenceSheet.jsx]):
  - *Purpose*: Displays interactive ISL A–Z manual alphabet reference and servo angle diagrams.

### Backend Services
- **`backend/app.py`** ([file:///c:/React/SignBridge/backend/app.py]):
  - *Purpose*: Flask application entrypoint defining REST endpoints for health, model info, landmark translation, word translation, serial control, and LLM text processing.
- **`translator_model.py`** ([file:///c:/React/SignBridge/backend/services/translator_model.py]):
  - *Purpose*: Loads ML model artifacts with 3-tier priority (XGBoost → Keras → Heuristic). Normalizes 126 landmark features and runs inference.
- **`word_recognizer.py`** ([file:///c:/React/SignBridge/backend/services/word_recognizer.py]):
  - *Purpose*: Loads Bi-LSTM model for temporal word recognition from 30-frame sequences. Gracefully disabled when no model file exists.
- **`arduino_serial.py`** ([file:///c:/React/SignBridge/backend/services/arduino_serial.py]):
  - *Purpose*: Handles USB serial port auto-discovery, connection management, string-to-angle conversion, and hardware command transmission.

---

## Database
- **Current State**: In-memory state in Python backend and React frontend; static JSON files for servo maps (`servo_presets`); browser `localStorage` for user assistant logs.
- **Planned Database**: Single **SQLite** database (`backend/signbridge.db`).
  - *Table `chat_history`*: Stores conversation transcripts between Deaf and hearing users.
  - *Table `servo_presets`*: Stores custom motor angle sequences for two-handed ISL signs.
  - *Table `gesture_dataset`*: Stores raw MediaPipe landmark arrays for offline retraining.

---

## APIs

### Backend REST Endpoints (`http://localhost:5000/api`)

| Method | Endpoint | Description | Request Body | Response Summary |
| :--- | :--- | :--- | :--- | :--- |
| **GET** | `/api/health` | Backend health & service connection status | None | `{ status, translator_mode, arduino_connected, groq_available, gemini_available }` |
| **GET** | `/api/model/info` | ML model metadata | None | `{ model_type, classes, loaded }` |
| **POST** | `/api/translate` | Translates hand landmarks to ISL letter | `{ landmarks: [ {x,y,z}, ... ] }` | `{ letter, confidence, mode, all_scores }` |
| **POST** | `/api/translate/batch` | Translates array of landmark frames | `{ frames: [ [...] ] }` | `{ results: [...], sentence: "..." }` |
| **POST** | `/api/translate/word` | Translates 30-frame sequence to ISL word (Bi-LSTM) | `{ frames: [ [126 floats], ...×30 ] }` | `{ word, confidence, all_scores }` |
| **POST** | `/api/robot/sign` | Transmits text sequence to robotic hands | `{ text: "HELLO" }` | `{ status, text, arduino_connected }` |
| **GET** | `/api/robot/status` | Queries Arduino serial status | None | `{ connected, port, baudrate }` |
| **POST** | `/api/robot/connect` | Initiates serial connection to Arduino | `{ port: "COM3" }` | `{ status, port }` |
| **POST** | `/api/robot/disconnect`| Closes Arduino serial connection | None | `{ status }` |
| **POST** | `/api/llm/refine` | Refines raw letter buffer to sentence | `{ text: "H E L L O" }` | `{ refined_sentence, llm_provider, fallback_used }` |
| **POST** | `/api/llm/simplify` | Simplifies speech to robot keywords | `{ text: "Please sit in room B" }` | `{ robot_keywords, llm_provider, fallback_used }` |
| **POST** | `/api/llm/answer` | Answers specific user questions using LLMs | `{ text: "Where is washroom?" }` | `{ answer, user_text, llm_provider, fallback_used }` |

---

## Business Logic
- **ISL Hand Landmark Processing**: Normalizes MediaPipe landmark coordinates relative to wrist joint, computes Euclidean distances between key finger joints, and passes normalized features to classifier (XGBoost → Keras → Heuristic priority chain).
- **3-Tier Model Priority**: `translator_model.py` attempts XGBoost (`.pkl`) first for best accuracy, falls back to Keras Dense NN (`.h5`), then to rule-based geometric heuristic.
- **Temporal Word Recognition**: `word_recognizer.py` uses a Bi-LSTM model to classify 30-frame landmark sequences into whole ISL words. Gracefully disabled if no model exists.
- **LLM Dual-Provider Fallback**: Attempts Groq `llama-3.3-70b-versatile` first for sub-100ms response speed; automatically catches errors/quota limits and falls back to Google Gemini `gemini-1.5-flash`.
- **Robotic Hand Motor Control**: Converts input string into uppercase characters, looks up 10-servo angle arrays (5 per hand), formats ASCII serial strings (e.g. `<90,0,180,45,90|0,90,90,180,0>`), and writes to Arduino over PySerial.

---

## Configuration
- **Environment Variables** ([file:///c:/React/SignBridge/backend/.env]):
  - `GROQ_API_KEY`: API key for Groq Llama-3.3 LLM.
  - `GOOGLE_API_KEY`: API key for Google Gemini LLM.
  - `SERIAL_PORT`: Target Arduino COM port (e.g., `COM3` or `/dev/ttyUSB0`).
  - `BAUD_RATE`: Serial baud rate (default: `115200`).
- **Build & Dev Commands**:
  - Frontend: `npm run dev` (Runs Vite server at port 5173).
  - Backend: `cd backend && python app.py` (Runs Flask server at port 5000).

---

## Current Progress
- **Completed**:
  - Full React UI layout (Kiosk, DualDisplay, HumanPanel, RobotPanel, ReferenceSheet).
  - MediaPipe landmark extraction pipeline on frontend.
  - Flask backend REST API infrastructure with CORS and health checks.
  - PySerial driver for dual robotic hand servo control.
  - Dual LLM fallback engine (Groq → Gemini).
  - Simulated DEMO mode with typewriter feeds.
  - **3-tier ML model architecture** (XGBoost → Keras → Heuristic priority chain).
  - **Bi-LSTM word recognition service infrastructure** (ready for dataset collection).
  - **`/api/translate/word` endpoint** for temporal word classification.
- **In Progress**:
  - Expanding two-handed ISL landmark dataset for all 26 manual alphabet letters.
  - Training XGBoost classifier on existing dataset for improved alphabet accuracy.
  - Collecting multi-frame word-level gesture recordings (`backend/dataset_words/`).
- **Upcoming Tasks**:
  - SQLite integration for local transcript logging (`signbridge.db`).
  - Hardware physical calibration suite for scrap-material servo bounds.
  - Train Bi-LSTM word model once word dataset is collected.

---

## Change Log
- **2026-08-07**: Initial creation of `PROJECT_CONTEXT.md` documenting architecture, components, data flows, APIs, and stack details.
- **2026-08-07**: Added `/api/llm/answer` endpoint for dynamic LLM question answering and added a temporary text input toggle button in `HumanPanel.jsx`.
- **2026-08-07**: **Major ML Architecture Upgrade** — Added 3-tier model recognition pipeline:
  - Tier 1: XGBoost alphabet classifier (XGBoost → Keras → Heuristic priority) via `train_model_xgb.py` and updated `translator_model.py`.
  - Tier 2: Bi-LSTM word recognizer infrastructure via `train_model_lstm.py`, `word_recognizer.py`, and `/api/translate/word` endpoint.
  - Tier 3: Groq/Gemini LLM sentence refinement (unchanged).
  - Updated `requirements.txt` with `xgboost>=2.0.0`.

