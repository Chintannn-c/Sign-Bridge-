<div align="center">

# 🤟 SignBridge

### **Dual-Communication Indian Sign Language (ISL) Assistant**

*Real-Time AI Vision • XGBoost Sign Classifier • Groq & Gemini Dual-LLM Engine • Robotic Hand Hardware Integration*

[![React](https://img.shields.io/badge/Frontend-React_19-blue?logo=react)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Bundler-Vite_8-646CFF?logo=vite)](https://vitejs.dev/)
[![Python](https://img.shields.io/badge/Backend-Python_3.11-3776AB?logo=python)](https://python.org/)
[![Flask](https://img.shields.io/badge/Framework-Flask_3.0-000000?logo=flask)](https://flask.palletsprojects.com/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-EC6B23?logo=xgboost)](https://xgboost.readthedocs.io/)
[![MediaPipe](https://img.shields.io/badge/Vision-MediaPipe_Hands-00979D)](https://mediapipe.dev/)
[![Docker](https://img.shields.io/badge/DevOps-Docker-2496ED?logo=docker)](https://www.docker.com/)
[![Kubernetes](https://img.shields.io/badge/Orchestration-Kubernetes-326CE5?logo=kubernetes)](https://kubernetes.io/)
[![SQLite](https://img.shields.io/badge/Database-SQLite_3-003B57?logo=sqlite)](https://www.sqlite.org/)

</div>

---

## 📌 Project Overview

**SignBridge** is a state-of-the-art dual-communication AI assistant designed to bridge the gap between deaf/mute individuals using **Indian Sign Language (ISL)** and hearing individuals in public kiosks, healthcare facilities, and everyday environments.

The system captures real-time 3D hand skeleton landmarks via webcam, classifies gestures using **XGBoost gradient-boosted decision trees**, refines raw fingerspelled letters into fluent conversational sentences using **Groq (Llama-3.3-70b)** and **Google Gemini 1.5 Flash**, and outputs responses via speech, text, or **Arduino-driven robotic hands**.

---

## ⚡ Key Features

* **🎥 60 FPS Dual-Hand Skeleton Tracking**: MediaPipe Full-Complexity 3D joint tracking smoothed by an adaptive **One-Euro Low-Pass Filter** to eliminate fingertip jitter.
* **🧠 Multi-Tiered Sign Classification Engine**:
  1. **Primary**: High-accuracy XGBoost classifier (`isl_xgboost_model.pkl`) with 98%+ accuracy.
  2. **Secondary**: Keras Dense Deep Learning Network (`isl_gesture_model.h5`).
  3. **Tertiary**: Mendeley geometric heuristic engine for zero-model fallback.
* **💬 Ultra-Fast Dual-LLM Sentence Engine**: Converts raw fingerspelled letter fragments (e.g. `"H E L L O  W A S H R O O M"`) into fluent natural language (`"Where is the washroom located?"`) using **Groq LPUs** (<100ms) with automatic failover to **Google Gemini 1.5 Flash**.
* **🤖 Robotic Hands Hardware Integration**: PySerial motor kinematics driver controlling 5-finger SG90 micro-servos on Arduino Uno/Mega.
* **💾 Persistent SQLite Database**: Stores conversation logs, landmark session metadata, and model performance analytics (`backend/database/signbridge.db`).
* **📦 Production DevOps Ready**: Fully containerized with multi-stage Dockerfiles and production Kubernetes (`k8s/`) cluster manifests.

---

## 📐 System Architecture

```
 🎥 LIVE WEBCAM FEED (60 FPS)
        │
        ▼
 🖐️ MEDIAPIPE HANDS (WebGL)
        │  Extracts 126 3D Coordinates + One-Euro Filter Smoothing
        ▼
 🧠 XGBOOST MODEL (Flask REST API: /api/translate)
        │  Translates Landmarks ──► Raw Letters ("H E L L O")
        ▼
 💬 GROQ LPU / GEMINI 1.5 FLASH (Flask REST API: /api/llm/refine)
        │  Refines Fragments ──► Fluent Sentence ("Hello! How can I help you?")
        ▼
 ┌───────────────────────────────┴───────────────────────────────┐
 │                                                               │
 ▼                                                               ▼
🖥️ REACT KIOSK UI                               🤖 ARDUINO ROBOTIC HANDS
 (Dual Speaker/Signer View)                       (PySerial Servo Driver)
```

---

## 🛠️ Tech Stack

| Domain | Technologies Used |
| :--- | :--- |
| **Frontend UI** | React 19, Vite 8, Framer Motion, Lucide React Icons, Vanilla CSS |
| **Backend API** | Python 3.11, Flask 3.0, Flask-CORS, Gunicorn WSGI |
| **Computer Vision** | MediaPipe Hands (3D Landmarks), OpenCV |
| **Machine Learning** | XGBoost (`XGBClassifier`), TensorFlow / Keras, Scikit-Learn |
| **Generative AI** | Groq SDK (`llama-3.3-70b-versatile`), Google Generative AI (`gemini-1.5-flash`) |
| **Database** | SQLite 3 (`backend/database/signbridge.db`) |
| **Hardware** | PySerial, Arduino Uno/Mega, SG90 Servos |
| **Containerization** | Docker, Docker Compose, Nginx Alpine, Kubernetes (k8s) |

---

## 🚀 Quick Start (Local Development)

### 1. Prerequisites
* **Node.js**: `v20.0+`
* **Python**: `v3.11+`
* **npm**: `v10.0+`

### 2. Clone & Setup Environment

```bash
git clone https://github.com/YOUR_USERNAME/SignBridge.git
cd SignBridge
```

Create and configure your backend environment variables:

```bash
cp backend/.env.example backend/.env
```

Edit `backend/.env` and insert your API keys:
```ini
GROQ_API_KEY=your_groq_api_key_here
GOOGLE_API_KEY=your_google_gemini_api_key_here
PORT=5000
HOST=0.0.0.0
```

### 3. Install Dependencies

#### Frontend:
```bash
npm install
```

#### Backend:
```bash
cd backend
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
python database/schema.py
cd ..
```

### 4. Run Development Servers

#### Terminal 1 — Start Flask Backend:
```bash
cd backend
python app.py
```
*(Runs on `http://localhost:5000`)*

#### Terminal 2 — Start React Frontend:
```bash
npm run dev
```
*(Runs on `http://localhost:5173`)*

---

## 🐳 Running with Docker

Build and run both Frontend and Backend using Docker Compose:

```bash
docker compose up --build
```

* **Frontend**: `http://localhost:5173`
* **Backend API**: `http://localhost:5000`

---

## ☸️ Deploying to Kubernetes

Deploy SignBridge to any Kubernetes cluster (Minikube, EKS, GKE, AKS):

```bash
# Apply ConfigMaps, Secrets, Deployments, and Services
kubectl apply -f k8s/
```

Check cluster pod status:
```bash
kubectl get pods -l app=signbridge
```

---

## 📊 Dataset Collection & Model Training

SignBridge includes automated tools for recording and training custom ISL datasets:

### 1. Capture Interactive Dataset
```bash
python backend/capture_dataset.py --letter A --signer signer1 --session 1
```

### 2. Convert Raw Videos/Images into Landmark JSONs
```bash
python convert_videos_to_json.py
```

### 3. Train XGBoost Model
```bash
cd backend
python train_model_xgb.py
```
*Outputs: `backend/models/isl_xgboost_model.pkl`*

---

## 🔌 REST API Endpoints

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/api/health` | `GET` | System health status, model mode, & API availability |
| `/api/translate` | `POST` | Translates 126 3D landmark floats to ISL letter |
| `/api/llm/refine` | `POST` | Refines raw fingerspelled letters into fluent sentences |
| `/api/llm/answer` | `POST` | Answers user questions with natural AI response |
| `/api/history` | `GET` | Retrieves SQLite conversation logs |
| `/api/robot/sign` | `POST` | Sends keywords to Arduino robotic hands |

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for more information.

---

<div align="center">
  <sub>Built with ❤️ for accessible healthcare and barrier-free communication.</sub>
</div>
