# Graph Report - SignBridge  (2026-10-29)

## Corpus Check
- 467 files · ~4,047,276 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1134 nodes · 1685 edges · 83 communities (69 shown, 10 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 42 edges (avg confidence: 0.92)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `449d8d87`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- react
- train_unified.py
- app.py
- package.json
- test_app_llm.py
- assign_canonical_hand_slots
- ArduinoSerial
- TestGeminiManager
- gemini_manager.py
- .generate
- KeyManager
- TestGroqManager
- KeyInfo
- groq_manager.py
- WordRecognizer
- SpatialGraphConv
- LandmarkSmoother
- .generate
- KeyInfo
- KeyManager
- .get_health_status
- is_neutral_gesture
- .oxlintrc.json
- SignBridge: Dual-Communication Indian Sign Language (ISL) Kiosk & Robotic Actuation System
- What You Must Do When Invoked
- 4. Detailed Implementation Plan
- PROJECT_CONTEXT.md — SignBridge System Context & Developer Onboarding Guide
- Sign-Bridge_Software_Hardware_Build_Guide _9244e8e7.md
- BaseRobotActuator
- AI_MEMORY.md
- graphify reference: extra exports and benchmark
- Ponytail
- Ponytail Help
- ingest_dataset_2.py
- ingest_zip_dataset.py
- schema.py
- 2. Ponytail (Lazy Senior Dev Mode)
- graphify reference: query, path, explain
- CentroidValidator
- ponytail-audit/SKILL.md
- Ponytail Gain
- ponytail-review/SKILL.md
- models/README.md
- Ponytail, lazy senior dev mode
- graphify reference: add a URL and watch a folder
- graphify reference: commit hook and native CLAUDE.md integration
- graphify reference: incremental update and cluster-only
- ponytail-debt/SKILL.md
- graphify reference: GitHub clone and cross-repo merge
- graphify reference: transcribe video and audio
- rules/graphify.md
- extraction-spec.md
- workflows/graphify.md
- GEMINI.md
- SignBridge Architecture
- TranslatorModel
- ._predict_xgb
- SignBridge Codebase Deep QA Audit & Engineering Report
- translator_model.py
- run_phase2a
- useGestureRecognition
- llm_answer
- GestureStateMachine
- ModelSelector
- generate_idle_dataset.py
- train_and_eval_word_variants.py
- 4.1 Backend Core Services
- 4.2 Frontend Core Components & Hooks
- 7.2 Step-by-Step Setup & Run Instructions
- 10.1 Common Contributor Workflows
- 2. Tech Stack
- 1. Project Overview
- 3. Architecture & Data Flow
- 5. Data Layer
- 6. API / Interface Surface
- 9. Health Check
- phase3_websocket_test.py
- phase4_frontend_review.py
- generate_word_report.py

## God Nodes (most connected - your core abstractions)
1. `useGestureRecognition()` - 28 edges
2. `TranslatorModel` - 26 edges
3. `react` - 23 edges
4. `TestGeminiManager` - 18 edges
5. `TestGroqManager` - 18 edges
6. `SignBridge: Dual-Communication Indian Sign Language (ISL) Kiosk & Robotic Actuation System` - 18 edges
7. `KeyInfo` - 16 edges
8. `KeyInfo` - 16 edges
9. `run_phase2a()` - 16 edges
10. `is_neutral_gesture()` - 15 edges

## Surprising Connections (you probably didn't know these)
- `run_phase3()` --calls--> `clean_llm_text()`  [INFERRED]
  qa_analysis/phase3_system_api.py → backend/app.py
- `run_phase3()` --calls--> `limit_sentences()`  [INFERRED]
  qa_analysis/phase3_system_api.py → backend/app.py
- `run_phase3()` --uses--> `ArduinoSerial`  [INFERRED]
  qa_analysis/phase3_system_api.py → backend/services/arduino_serial.py
- `run_phase1()` --calls--> `load_dataset_partitioned()`  [INFERRED]
  qa_analysis/phase1_baseline.py → backend/services/data_loader.py
- `run_phase2a()` --uses--> `CentroidValidator`  [INFERRED]
  qa_analysis/phase2a_letter_model.py → backend/services/gesture_pipeline.py

## Import Cycles
- None detected.

## Communities (83 total, 10 thin omitted)

### Community 0 - "react"
Cohesion: 0.05
Nodes (49): framer-motion, lucide-react, react, App(), CameraView(), ErrorBoundary, HumanPanel, GestureReferenceSheet (+41 more)

### Community 1 - "train_unified.py"
Cohesion: 0.12
Nodes (23): _compute_angle_cos(), extract_features(), extract_holistic_features(), Enriched Feature extraction for Indian Sign Language (ISL) alphabet…, Extracts geometric feature vector enriched with upper-body anchor metrics. If…, Transforms temporal landmark sequences of shape (30, 126) or (N, 30, 126) into…, Compute cosine of the 3D angle at vertex p_b between vectors (p_a - p_b) and…, Extract a 208-D normalized geometric feature vector from 126 raw landmark… (+15 more)

### Community 2 - "app.py"
Cohesion: 0.08
Nodes (45): check_api_key(), clamp_float(), collect_data(), get_gesture_config(), get_history(), health(), llm_status(), model_info() (+37 more)

### Community 3 - "package.json"
Cohesion: 0.06
Nodes (29): dependencies, framer-motion, lucide-react, react, react-dom, description, devDependencies, oxlint (+21 more)

### Community 4 - "test_app_llm.py"
Cohesion: 0.08
Nodes (27): clean_llm_text(), fallback_refine(), fallback_simplify(), get_word_vocabulary(), has(), limit_sentences(), llm_simplify(), Matches whole words in query string using word boundaries. (+19 more)

### Community 5 - "assign_canonical_hand_slots"
Cohesion: 0.06
Nodes (58): extract_mendeley_zip(), main(), process_mendeley_dataset(), process_realsign_dataset(), process_realsign_letters(), process_self_made_dataset(), SignBridge — Unified Batch Static Landmark Extractor (v2) Processes ALL…, Processes ALL ISL self-made dataset images (0-9 and a-z) — NO cap. (+50 more)

### Community 6 - "ArduinoSerial"
Cohesion: 0.13
Nodes (11): ArduinoSerial, Scan COM ports for an Arduino device., Open serial connection to Arduino., Close the serial connection., Send a 10-element servo angle array to the Arduino. Format sent: JSON string…, Look up the ISL servo angles for a single letter and send them. Holds the pose…, Internal synchronous text signing implementation., Fingerspell an entire text string letter-by-letter. If async_mode is True,… (+3 more)

### Community 7 - "TestGeminiManager"
Cohesion: 0.10
Nodes (12): GeminiManager, Centralized, resilient Google Generative AI / Gemini API Manager. Orchestrates:…, Scenario 3: Both keys fail on Model A -> Model B succeeds on Key 1., Scenario 4: Key 1 invalid (401/403) -> Permanently disabled, switches to Key 2., Scenario 5: All keys/models fail -> Raises clean application error, no crash., Scenario 6: Transient 500 error on attempt 1, succeeds on attempt 2., Scenario 7: 400 Invalid Argument fails fast without wasting retries., Verify API keys are never logged in plain text. (+4 more)

### Community 8 - "gemini_manager.py"
Cohesion: 0.21
Nodes (12): ErrorClassifier, GeminiErrorType, GeminiKeyStatus, mask_key(), ModelSelector, Enum, Exception, SignBridge Centralized Google Generative AI / Gemini API Management System.… (+4 more)

### Community 9 - ".generate"
Cohesion: 0.33
Nodes (4): GroqResult, TypedDict, Execute robust generation across Groq (Model x Key) matrix., Low-level Groq completions caller.

### Community 10 - "KeyManager"
Cohesion: 0.15
Nodes (7): KeyManager, Manages collection of API keys, state tracking, and health-aware rotation., Discover and load keys from environment variables., Return all currently usable keys, sorted by priority (KEY_1 -> KEY_2)., Record successful request for a key., Record a failure and apply appropriate cooldown/backoff., Reset all cooldowns and failures.

### Community 11 - "TestGroqManager"
Cohesion: 0.09
Nodes (13): GroqManager, Centralized, resilient Groq LPU API Manager. Orchestrates: Model Selection ->…, Check if at least one Groq API key is configured and not permanently invalid., Comprehensive health status for telemetry and monitoring., Scenario 3: Both keys fail on Model A -> Model B succeeds on Key 1., Scenario 4: Key 1 invalid (401/403) -> Permanently disabled, switches to Key 2., Scenario 5: All keys/models fail -> Raises clean runtime error., Scenario 6: Transient 500 error on attempt 1, succeeds on attempt 2. (+5 more)

### Community 12 - "KeyInfo"
Cohesion: 0.17
Nodes (7): KeyInfo, Initialize modern or legacy SDK client for this key., Check if key is ready for requests., Get ordered list of candidate models (latest -> fallback) suitable for task. If…, Query the API to find which models this specific key has access to., Runtime health and state tracker for an individual API key., Scenario 8: Model auto-discovery dynamically sorts granted models.

### Community 13 - "groq_manager.py"
Cohesion: 0.20
Nodes (11): ErrorClassifier, GroqErrorType, GroqKeyStatus, mask_key(), Enum, Exception, SignBridge Centralized Groq LPU API Management System. Features: - Dual API Key…, Classifies raw Groq SDK and HTTP exceptions into actionable error enums. (+3 more)

### Community 14 - "WordRecognizer"
Cohesion: 0.07
Nodes (26): compute_wrist_relative_motion(), ndarray, Predict a word from a sequence of landmark frames with optional MediaPipe…, Computes motion velocity across sequence using wrist-relative per-hand deltas.…, Ensure frames array is exactly SEQUENCE_LENGTH frames long., Return word recognizer metadata., ISL whole-word gesture recognizer using temporal landmark sequences. Accepts a…, Attempt to load the trained word model. Loading priority: 1. CNN-BiLSTM hybrid… (+18 more)

### Community 15 - "SpatialGraphConv"
Cohesion: 0.27
Nodes (6): build_dual_hand_adjacency(), SignBridge — Spatial-Temporal Graph Convolutional Network (ST-GCN) for ISL…, Builds the 42x42 normalized adjacency matrix with self-loops for dual hands., Spatial Graph Convolution layer. Transforms node features using the adjacency…, SpatialGraphConv, Tensor

### Community 16 - "LandmarkSmoother"
Cohesion: 0.19
Nodes (3): LandmarkSmoother, LowPassFilter, OneEuroFilter

### Community 17 - ".generate"
Cohesion: 0.29
Nodes (5): GeminiResult, TypedDict, Execute robust generation across (Model x Key) matrix. Algorithm: 1. Select…, Low-level SDK caller handling both modern genai and legacy SDKs., ValueError

### Community 18 - "KeyInfo"
Cohesion: 0.18
Nodes (7): KeyInfo, Any, Sanitized dictionary for status reporting (no plain-text keys)., Runtime health and state tracker for an individual Groq API key., Initialize Groq client for this key., Check if key is ready for requests., Scenario 8: Model auto-discovery dynamically sorts granted models.

### Community 19 - "KeyManager"
Cohesion: 0.15
Nodes (7): KeyManager, Manages collection of Groq API keys, state tracking, and health-aware rotation., Discover and load keys from environment variables., Return all currently usable keys, sorted by priority (KEY_1 -> KEY_2)., Record successful request for a key., Record a failure and apply appropriate cooldown/backoff., Reset all cooldowns and failures.

### Community 20 - ".get_health_status"
Cohesion: 0.29
Nodes (4): Any, Sanitized dictionary for status reporting (no plain-text keys)., Check if at least one API key is configured and not permanently invalid., Comprehensive health status for telemetry and monitoring.

### Community 21 - "is_neutral_gesture"
Cohesion: 0.13
Nodes (22): SignBridge Gesture Recognition Configuration and Thresholds. Centralizes all…, is_idle_hand(), is_neutral_gesture(), _is_palm_facing_camera(), _is_single_finger_extended(), _is_thumb_extended(), Any, ndarray (+14 more)

### Community 22 - ".oxlintrc.json"
Cohesion: 0.33
Nodes (5): plugins, rules, react/only-export-components, react/rules-of-hooks, $schema

### Community 23 - "SignBridge: Dual-Communication Indian Sign Language (ISL) Kiosk & Robotic Actuation System"
Cohesion: 0.06
Nodes (33): 1. Clone the Repository, 1. Hardware Specifications, 1. Running Backend Unit Tests, 1. Static Alphabet Recognition (Tier 1), 2. Backend Setup, 2. Dynamic Temporal Word Recognition (Tier 2), 2. Software Architecture (`robot_actuator.py`), 2. Verifying Word Model Inference (+25 more)

### Community 24 - "What You Must Do When Invoked"
Cohesion: 0.08
Nodes (24): For /graphify add and --watch, For /graphify query, For the commit hook and native CLAUDE.md integration, For --update and --cluster-only, /graphify, Honesty Rules, Interpreter guard for subcommands, Part A - Structural extraction for code files (+16 more)

### Community 27 - "4. Detailed Implementation Plan"
Cohesion: 0.09
Nodes (21): 1. Project Objective, 2.1 Path 1 — Human Signs → Text/Speech (Input Path), 2.2 Path 2 — Text/Speech → Robotic Hands (Output Path), 2.3 Data Flow Summary, 2. System Architecture: The Dual Communication Loop, 3. Recommended Project Sequence, 4. Detailed Implementation Plan, 5. Bill of Materials & Cost Estimate (+13 more)

### Community 28 - "PROJECT_CONTEXT.md — SignBridge System Context & Developer Onboarding Guide"
Cohesion: 0.25
Nodes (7): 11. Quick Start Summary, 12. Open Questions for the Maintainer, 4. Module-by-Module Breakdown, 8.1 Existing Test Suites, 8.2 How to Run Tests & Linters, 8. Testing & Quality, PROJECT_CONTEXT.md — SignBridge System Context & Developer Onboarding Guide

### Community 29 - "Sign-Bridge_Software_Hardware_Build_Guide _9244e8e7.md"
Cohesion: 0.11
Nodes (18): 1.1 Path 1 in code terms (ISL → text/speech), 1.2 Path 2 in code terms (text/speech → ISL hands), 1. Software Architecture — Where Everything Runs, 2. Display Interface (What the Browser Looks Like), 3. How the Webcam Captures and Interprets Signs, 4.1 Per-Hand Build Steps (repeat identically for left and right), 4.2 Wiring Both Hands to One Arduino Mega, 4.3 Hardware ↔ Software Integration (+10 more)

### Community 30 - "BaseRobotActuator"
Cohesion: 0.11
Nodes (10): ABC, BaseRobotActuator, SignBridge — Robotics Actuation Interface & ROS2 Stub Provides a standardized…, Abstract base actuator for dual robotic hands / arms., Establish connection to actuator hardware or middleware., Cleanly close actuator connection., Send 10-element servo angle array [0..180]., Actuate robotic hands to form a specific ISL letter. (+2 more)

### Community 31 - "AI_MEMORY.md"
Cohesion: 0.17
Nodes (10): AI Notes, Architecture Decisions, Coding Conventions, Common Pitfalls, Constraints, Design Patterns, Frequently Edited Files, Important Assumptions (+2 more)

### Community 32 - "graphify reference: extra exports and benchmark"
Cohesion: 0.22
Nodes (8): graphify reference: extra exports and benchmark, Step 6b - Wiki (only if --wiki flag), Step 7 - Neo4j export (only if --neo4j or --neo4j-push flag), Step 7a - FalkorDB export (only if --falkordb or --falkordb-push flag), Step 7b - SVG export (only if --svg flag), Step 7c - GraphML export (only if --graphml flag), Step 7d - MCP server (only if --mcp flag), Step 8 - Token reduction benchmark (only if total_words > 5000)

### Community 33 - "Ponytail"
Cohesion: 0.22
Nodes (8): Boundaries, Intensity, Output, Persistence, Ponytail, Rules, The ladder, When NOT to be lazy

### Community 34 - "Ponytail Help"
Cohesion: 0.25
Nodes (7): Configure Default Mode, Deactivate, Levels, More, Ponytail Help, Skills, Update

### Community 35 - "ingest_dataset_2.py"
Cohesion: 0.38
Nodes (6): augment_vector(), get_landmark_vector(), main(), SignBridge — Dataset 2 Batch Ingestion & Landmark Extraction Processes all 521…, Standardizes MediaPipe HandLandmarker result into a 126-float array. Indices…, Generates slight rotation, scale, and jitter variations of a landmark vector.

### Community 36 - "ingest_zip_dataset.py"
Cohesion: 0.38
Nodes (6): augment_landmark_vector(), get_landmark_vector_from_result(), ingest_zip(), SignBridge — Custom Dataset.zip Ingestion & Feature Extraction Reads…, Standardizes MediaPipe HandLandmarker result into a 126-float array. Indices…, Generate subtle kinematic rotations and noise for a static landmark vector.

### Community 37 - "schema.py"
Cohesion: 0.15
Nodes (14): get_db_connection(), get_recent_history(), init_db(), log_conversation(), log_dataset_session(), SignBridge SQLite Database Initialization & Management Database File:…, Inserts or updates a dataset session record into SQLite., Returns a SQLite connection with dict-like row factory and WAL journal mode. (+6 more)

### Community 38 - "2. Ponytail (Lazy Senior Dev Mode)"
Cohesion: 0.33
Nodes (5): 1. Graphify Knowledge Graph, 2. Ponytail (Lazy Senior Dev Mode), Core Rules, SignBridge AI Agent Guidelines, The Ladder

### Community 39 - "graphify reference: query, path, explain"
Cohesion: 0.33
Nodes (5): For /graphify explain, For /graphify path, graphify reference: query, path, explain, Step 0 — Constrained query expansion (REQUIRED before traversal), Step 1 — Traversal

### Community 40 - "CentroidValidator"
Cohesion: 0.09
Nodes (21): CentroidValidator, GestureStateMachine, Path, Enforces the required recognition lifecycle: NEUTRAL -> ARMED -> CAPTURING ->…, Reset state machine to initial NEUTRAL state., Validates live normalized landmarks against stored dataset centroids., Calculates distance from normalized landmarks to the predicted class centroid.…, create_fist_hand() (+13 more)

### Community 41 - "ponytail-audit/SKILL.md"
Cohesion: 0.40
Nodes (4): Boundaries, Hunt, Output, Tags

### Community 42 - "Ponytail Gain"
Cohesion: 0.40
Nodes (4): Boundaries, Honesty boundary, Ponytail Gain, Scoreboard

### Community 43 - "ponytail-review/SKILL.md"
Cohesion: 0.40
Nodes (4): Boundaries, Examples, Format, Scoring

### Community 44 - "models/README.md"
Cohesion: 0.40
Nodes (4): - isl_gesture_model.h5     (Keras model weights), Run `python train_model.py` to generate:, This directory stores trained model weights., - training_meta.json       (Training metadata)

### Community 45 - "Ponytail, lazy senior dev mode"
Cohesion: 0.50
Nodes (3): Not Lazy About, Ponytail, lazy senior dev mode, Rules

### Community 46 - "graphify reference: add a URL and watch a folder"
Cohesion: 0.50
Nodes (3): For /graphify add, For --watch, graphify reference: add a URL and watch a folder

### Community 47 - "graphify reference: commit hook and native CLAUDE.md integration"
Cohesion: 0.50
Nodes (3): For git commit hook, For native CLAUDE.md integration, graphify reference: commit hook and native CLAUDE.md integration

### Community 48 - "graphify reference: incremental update and cluster-only"
Cohesion: 0.50
Nodes (3): For --cluster-only, For --update (incremental re-extraction), graphify reference: incremental update and cluster-only

### Community 49 - "ponytail-debt/SKILL.md"
Cohesion: 0.50
Nodes (3): Boundaries, Output, Scan

### Community 57 - "SignBridge Architecture"
Cohesion: 0.40
Nodes (4): 1. Perception Engine (`backend/services/`), 2. Presentation Layer (`src/components/`), 3. Actuation Layer (`backend/services/arduino_serial.py`), SignBridge Architecture

### Community 58 - "TranslatorModel"
Cohesion: 0.13
Nodes (11): ST-GCN Classifier for Static / Frame-Level Dual-Hand ISL Gestures. Input:…, STGCNHandClassifier, ISL Gesture-to-Text translation model. Supports four ML modes: 1. Hybrid…, Attempt to load trained models; fall back to heuristic. Loading priority: 1.…, Parse the Mendeley CSV into a lookup dictionary., Rule-based heuristic prediction using geometric features extracted from 42 hand…, Parse landmarks into a list of 42 {x, y, z} dicts. Handles both flat arrays and…, Extract geometric features from 42 landmark points. Points 0-20: Left hand,… (+3 more)

### Community 59 - "._predict_xgb"
Cohesion: 0.27
Nodes (7): Coerce landmarks to a (126,) or (N, 126) float array, or raise., Predict the ISL letter from a set of hand landmarks with optional Holistic body…, Run inference through the calibrated ST-GCN + XGBoost Hybrid Ensemble., Run inference through the trained PyTorch ST-GCN model., Run inference through the trained XGBoost model., Run inference through the trained Keras model., validate_landmark_array()

### Community 60 - "SignBridge Codebase Deep QA Audit & Engineering Report"
Cohesion: 0.10
Nodes (19): 1. Executive Summary, 1. Repeated Letter Dropping Bug in Gesture State Machine, 2. Comprehensive Test Results Table, 2. Hardcoded Localhost URL in DataCollectionTab.jsx, 3. Machine Learning Model Metrics, 3. npm High-Severity Vulnerabilities, 4. Documentation vs. Real Code Mismatches, 4. Word Model Low Validation Accuracy & Imbalance (+11 more)

### Community 61 - "translator_model.py"
Cohesion: 0.14
Nodes (15): SignBridge — Benchmark Standalone XGBoost, Standalone ST-GCN, and Calibrated…, run_benchmark(), get_file_sha256(), load_dataset_partitioned(), Any, Path, Clean, Leak-Free Data Loader and Partition Manager for SignBridge ISL Alphabet…, Compute SHA-256 hash of a file. (+7 more)

### Community 62 - "run_phase2a"
Cohesion: 0.20
Nodes (14): compute_centroids(), Script to compute per-class centroids and distance thresholds from…, normalize_landmarks_batch(), Vectorized normalize_landmarks for a (N, 126) float32 array. Mirrors…, compute_ece(), mirror_batch(), predict_batch_fast(), SignBridge QA Analysis - Phase 2A: Vectorized Letter Model (A-Z) Deep… (+6 more)

### Community 63 - "useGestureRecognition"
Cohesion: 0.10
Nodes (22): GESTURE_CONFIG, GESTURE_STATES, GESTURE_STATUS_LABELS, analyzeQuality(), computeLandmarkMotion(), formatWordForSentence(), getFrameHandCentroid(), isHandActive() (+14 more)

### Community 64 - "llm_answer"
Cohesion: 0.18
Nodes (14): generate_llm_response(), get_smart_fallback_response(), get_text(), llm_answer(), llm_refine(), LLMResponse, TypedDict, Safely log conversation to SQLite without propagating errors. (+6 more)

### Community 65 - "GestureStateMachine"
Cohesion: 0.06
Nodes (27): _compute_motion(), GestureStateMachine, Any, SignBridge Ultra-Fast "Lock, then Hold Steady" Gesture State Machine. Single…, Resets machine to initial state., Validates prediction against confidence, margin, and centroid distance limits.…, Single per-frame update driving the state machine. Args: now_ms: current…, Calculates mean Euclidean movement per landmark between consecutive frames. (+19 more)

### Community 66 - "ModelSelector"
Cohesion: 0.33
Nodes (4): ModelSelector, Task-aware Groq model selector with auto-discovery and capability sorting., Get ordered list of candidate models (latest -> fallback). Prioritizes…, Query Groq API to discover active models for this key.

### Community 69 - "train_and_eval_word_variants.py"
Cohesion: 0.22
Nodes (8): Dataset, augment_data(), CNNBiLSTMWordClassifier, evaluate_model(), load_word_data(), SignBridge Word Model Experimentation & Comparative Benchmarking. Evaluates:…, run_word_experiments(), SeqDataset

### Community 70 - "4.1 Backend Core Services"
Cohesion: 0.18
Nodes (11): 4.1 Backend Core Services, [backend/app.py](file:///c:/React/SignBridge/backend/app.py), [backend/database/schema.py](file:///c:/React/SignBridge/backend/database/schema.py), [backend/services/arduino_serial.py](file:///c:/React/SignBridge/backend/services/arduino_serial.py), [backend/services/feature_extractor.py](file:///c:/React/SignBridge/backend/services/feature_extractor.py), [backend/services/gesture_pipeline.py](file:///c:/React/SignBridge/backend/services/gesture_pipeline.py), [backend/services/gesture_state_machine.py](file:///c:/React/SignBridge/backend/services/gesture_state_machine.py), [backend/services/groq_manager.py](file:///c:/React/SignBridge/backend/services/groq_manager.py) & [gemini_manager.py](file:///c:/React/SignBridge/backend/services/gemini_manager.py) (+3 more)

### Community 71 - "4.2 Frontend Core Components & Hooks"
Cohesion: 0.22
Nodes (9): 4.2 Frontend Core Components & Hooks, [src/App.jsx](file:///c:/React/SignBridge/src/App.jsx), [src/components/CameraView.jsx](file:///c:/React/SignBridge/src/components/CameraView.jsx), [src/components/HumanPanel.jsx](file:///c:/React/SignBridge/src/components/HumanPanel.jsx), [src/components/RobotPanel.jsx](file:///c:/React/SignBridge/src/components/RobotPanel.jsx), [src/components/SignBridgeKiosk.jsx](file:///c:/React/SignBridge/src/components/SignBridgeKiosk.jsx), [src/hooks/useGestureRecognition.js](file:///c:/React/SignBridge/src/hooks/useGestureRecognition.js), [src/hooks/useHandDetection.js](file:///c:/React/SignBridge/src/hooks/useHandDetection.js) (+1 more)

### Community 72 - "7.2 Step-by-Step Setup & Run Instructions"
Cohesion: 0.22
Nodes (9): 7.1 Environment Variables, 7.2 Step-by-Step Setup & Run Instructions, 7. Configuration & Environment, Prerequisites, Step 1: Clone & Setup Python Virtual Environment, Step 2: Configure Environment Keys, Step 3: Setup Frontend Dependencies, Step 4: Run the System Locally (+1 more)

### Community 73 - "10.1 Common Contributor Workflows"
Cohesion: 0.33
Nodes (6): 10.1 Common Contributor Workflows, 10.2 Developer Rules & Gotchas (Ponytail & Graphify), 10. Extending the Project, Adding a New Backend REST Route, Adding a New Dynamic Word Gesture, Adding a New Static Letter or Custom Sign

### Community 74 - "2. Tech Stack"
Cohesion: 0.40
Nodes (5): 2.1 Languages & Frameworks, 2.2 External APIs, Models & Third-Party Services, 2.3 Databases & Storage, 2.4 Build Tools, Package Managers & Runtime Requirements, 2. Tech Stack

### Community 75 - "1. Project Overview"
Cohesion: 0.50
Nodes (4): 1.1 Purpose & Problem Statement, 1.2 Current Maturity, 1.3 Annotated Repository Layout, 1. Project Overview

### Community 76 - "3. Architecture & Data Flow"
Cohesion: 0.50
Nodes (4): 3.1 High-Level Component Interaction Diagram, 3.2 Entry Points, 3.3 Key Design Patterns, 3. Architecture & Data Flow

### Community 77 - "5. Data Layer"
Cohesion: 0.50
Nodes (4): 5.1 Database Models & Schemas ([backend/database/schema.py](file:///c:/React/SignBridge/backend/database/schema.py)), 5.2 Datasets & Asset Inventory, 5.3 Trained Model Artifacts ([backend/models/](file:///c:/React/SignBridge/backend/models/)), 5. Data Layer

### Community 78 - "6. API / Interface Surface"
Cohesion: 0.50
Nodes (4): 6.1 Backend REST API Endpoints ([backend/app.py](file:///c:/React/SignBridge/backend/app.py)), 6.2 WebSocket Streaming Interface ([backend/services/websocket_server.py](file:///c:/React/SignBridge/backend/services/websocket_server.py)), 6.3 User Interface Surface & Controls, 6. API / Interface Surface

### Community 79 - "9. Health Check"
Cohesion: 0.50
Nodes (4): 9.1 Known Past Bugs & Regressions Resolved, 9.2 Technical Debt & Architectural Opportunities, 9.3 Inconsistencies Between Documentation & Actual Code, 9. Health Check

## Knowledge Gaps
- **259 isolated node(s):** `$schema`, `plugins`, `react/rules-of-hooks`, `react/only-export-components`, `name` (+254 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 600 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **10 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `WordRecognizer` connect `WordRecognizer` to `app.py`, `translator_model.py`?**
  _High betweenness centrality (0.049) - this node is a cross-community bridge._
- **Why does `TranslatorModel` connect `TranslatorModel` to `app.py`, `CentroidValidator`, `is_neutral_gesture`, `._predict_xgb`, `translator_model.py`, `run_phase2a`?**
  _High betweenness centrality (0.044) - this node is a cross-community bridge._
- **Why does `TestGroqManager` connect `TestGroqManager` to `KeyInfo`, `ModelSelector`, `groq_manager.py`?**
  _High betweenness centrality (0.036) - this node is a cross-community bridge._
- **Are the 5 inferred relationships involving `TranslatorModel` (e.g. with `STGCNHandClassifier` and `CentroidValidator`) actually correct?**
  _`TranslatorModel` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `TestGeminiManager` (e.g. with `ErrorClassifier` and `GeminiErrorType`) actually correct?**
  _`TestGeminiManager` has 6 INFERRED edges - model-reasoned connections that need verification._
- **What connects `$schema`, `plugins`, `react/rules-of-hooks` to the rest of the system?**
  _259 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `react` be split into smaller, more focused modules?**
  _Cohesion score 0.05171907140758154 - nodes in this community are weakly interconnected._