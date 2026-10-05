# SignBridge Architecture

SignBridge is an assistive two-way communication kiosk bridging Indian Sign Language (ISL) and spoken/text communication with physical robotic hand actuation.

```
                           ┌──────────────────────────────────────────────┐
                           │          KIOSK PRESENTATION LAYER            │
                           │     React 19 + Vite (SignBridgeKiosk)        │
                           └──────┬────────────────────────────────┬──────┘
                                  │                                │
            126 Landmark Floats   │                                │ Spoken / Typed Text
            (Left + Right Hand)   │                                │
                                  ▼                                ▼
┌──────────────────────────────────────────┐    ┌──────────────────────────────────────────┐
│      PERCEPTION ENGINE (Path 1)          │    │       ACTUATION ENGINE (Path 2)          │
│       Python 3 + Flask REST API          │    │       LLM Simplifier + Hardware Driver   │
│                                          │    │                                          │
│ 1. XGBoost (Alphabet: A-Z)               │    │ 1. LLM Keyword Simplification            │
│ 2. CNN-BiLSTM (Temporal: 57 ISL Words)   │    │    (Groq Llama 3.3 -> Gemini 1.5)        │
│ 3. LLM Refinement (Groq -> Gemini)       │    │ 2. PySerial USB Driver (Arduino Mega)    │
│ 4. Web Speech API (Audio TTS Output)     │    │ 3. Dual Robotic Hands (10x SG90 Servos)  │
└──────────────────────────────────────────┘    └──────────────────────────────────────────┘
```

---

## 1. Perception Engine (`backend/services/`)

* **Input**: 42 3D landmarks $(x, y, z)$ from MediaPipe Dual-Hand tracking (126 normalized float features).
* **Classification Pipeline**:
  1. **Alphabet Classifier** (`translator_model.py`): XGBoost `.pkl` $\to$ ST-GCN PyTorch $\to$ Heuristic fallback.
  2. **Temporal Word Classifier** (`word_recognizer.py`): 30-frame temporal window $\times$ 126 features processed by a CNN-BiLSTM neural network (57 ISL classes, 5.53 ms inference latency, 93.3% accuracy).
  3. **Sentence Refinement**: Converts raw gloss strings into natural spoken sentences using Groq (`llama-3.3-70b-versatile`) with automatic fallback to Google Gemini (`gemini-1.5-flash`).

---

## 2. Presentation Layer (`src/components/`)

* **Container**: `SignBridgeKiosk.jsx` renders a two-panel conversational kiosk interface.
* **Human Panel** (`HumanPanel.jsx`): Camera feed (`CameraView.jsx`), real-time landmark canvas overlay, buffer display, and fullscreen HUD mode (`F` key).
* **Robot Panel** (`RobotPanel.jsx`): Dialogue thread, TTS speaker toggle, and manual speech/text input.
* **Kiosk Controls**:
  * `Space`: Toggle automated live demo mode.
  * `1` or `H`: Trigger human turn.
  * `2` or `R`: Trigger assistant response.
  * `F`: Fullscreen camera feed with HUD.
  * `C`: Clear buffers.

---

## 3. Actuation Layer (`backend/services/arduino_serial.py`)

* **Communication**: PySerial at 9600 baud sending 10-element JSON servo angle arrays:
  `[L_thumb, L_index, L_middle, L_ring, L_pinky, R_thumb, R_index, R_middle, R_ring, R_pinky]`
* **Mechanical Bounds**: SG90 micro servos calibrated from $0^\circ$ (open/extended) to $180^\circ$ (closed/flexed).
* **Execution Flow**: Spoken text $\to$ LLM key gloss extraction $\to$ letter/word mapping $\to$ USB serial command $\to$ dual robotic hand actuation.
