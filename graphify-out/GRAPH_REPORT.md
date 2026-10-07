# Graph Report - SignBridge  (2026-10-07)

## Corpus Check
- 428 files · ~3,776,288 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 772 nodes · 1063 edges · 65 communities (54 shown, 8 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 17 edges (avg confidence: 0.93)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `0a6fd800`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- react
- train_unified.py
- app.py
- package.json
- extract_static_landmarks.py
- stream_and_extract_hf.py
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
- .get_health_status
- .oxlintrc.json
- SignBridge: Dual-Communication Indian Sign Language (ISL) Kiosk & Robotic Actuation System
- What You Must Do When Invoked
- 4. Detailed Implementation Plan
- PROJECT_CONTEXT.md
- Sign-Bridge_Software_Hardware_Build_Guide _9244e8e7.md
- BaseRobotActuator
- AI_MEMORY.md
- graphify reference: extra exports and benchmark
- Ponytail
- Ponytail Help
- ingest_dataset_2.py
- ingest_zip_dataset.py
- extract_video_landmarks.py
- 2. Ponytail (Lazy Senior Dev Mode)
- graphify reference: query, path, explain
- normalize_landmarks
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
- STGCNHandClassifier
- TranslatorModel
- translator_model.py
- load_dataset_partitioned
- .get_info
- ModelSelector
- mask_key

## God Nodes (most connected - your core abstractions)
1. `react` - 21 edges
2. `TranslatorModel` - 20 edges
3. `TestGeminiManager` - 18 edges
4. `TestGroqManager` - 18 edges
5. `SignBridge: Dual-Communication Indian Sign Language (ISL) Kiosk & Robotic Actuation System` - 18 edges
6. `KeyInfo` - 16 edges
7. `KeyInfo` - 16 edges
8. `ArduinoSerial` - 13 edges
9. `lucide-react` - 12 edges
10. `What You Must Do When Invoked` - 12 edges

## Surprising Connections (you probably didn't know these)
- `TranslatorModel` --uses--> `STGCNHandClassifier`  [INFERRED]
  backend/services/translator_model.py → backend/models/st_gcn.py
- `TestGeminiManager` --uses--> `GeminiKeyStatus`  [INFERRED]
  backend/tests/test_gemini_manager.py → backend/services/gemini_manager.py
- `TestGeminiManager` --uses--> `GeminiErrorType`  [INFERRED]
  backend/tests/test_gemini_manager.py → backend/services/gemini_manager.py
- `TestGeminiManager` --uses--> `KeyInfo`  [INFERRED]
  backend/tests/test_gemini_manager.py → backend/services/gemini_manager.py
- `TestGeminiManager` --uses--> `ErrorClassifier`  [INFERRED]
  backend/tests/test_gemini_manager.py → backend/services/gemini_manager.py

## Import Cycles
- None detected.

## Communities (65 total, 8 thin omitted)

### Community 0 - "react"
Cohesion: 0.06
Nodes (45): framer-motion, lucide-react, react, App(), CameraView(), GestureReferenceSheet(), getDynamicFontSize(), HumanPanel (+37 more)

### Community 1 - "train_unified.py"
Cohesion: 0.18
Nodes (16): Transforms temporal landmark sequences of shape (30, 126) or (N, 30, 126) into…, transform_to_kinetic_invariants(), augment_landmarks(), augment_training_partition(), main(), SignBridge — Unified Training Pipeline (v2) One script to rule them all: 1.…, Train XGBoost letter classifier with all combined data., Train PyTorch ST-GCN kinematic hand graph classifier. (+8 more)

### Community 2 - "app.py"
Cohesion: 0.05
Nodes (60): clean_llm_text(), collect_data(), generate_llm_response(), get_history(), get_smart_fallback_response(), health(), limit_to_human_length(), llm_answer() (+52 more)

### Community 3 - "package.json"
Cohesion: 0.07
Nodes (27): dependencies, framer-motion, lucide-react, react, react-dom, description, devDependencies, oxlint (+19 more)

### Community 4 - "extract_static_landmarks.py"
Cohesion: 0.22
Nodes (14): extract_mendeley_zip(), get_landmark_vector_from_result(), main(), process_mendeley_dataset(), process_realsign_dataset(), process_realsign_letters(), process_self_made_dataset(), SignBridge — Unified Batch Static Landmark Extractor (v2) Processes ALL… (+6 more)

### Community 5 - "stream_and_extract_hf.py"
Cohesion: 0.19
Nodes (14): augment_raw_sequence(), clean_word_label(), extract_raw_frames_from_video_bytes(), get_landmark_vector_from_result(), init_mediapipe_detector(), main(), SignBridge — High-Speed Hugging Face ISL Zero-Disk Streaming Landmark Extractor…, Decodes video frames directly from memory bytes buffer and extracts 42 3D… (+6 more)

### Community 6 - "ArduinoSerial"
Cohesion: 0.11
Nodes (12): ArduinoSerial, Sign-Bridge Flask API — Arduino Serial Communication Service Manages PySerial…, Scan COM ports for an Arduino device., Open serial connection to Arduino., Close the serial connection., Send a 10-element servo angle array to the Arduino. Format sent: JSON string…, Look up the ISL servo angles for a single letter and send them. Holds the pose…, Internal synchronous text signing implementation. (+4 more)

### Community 7 - "TestGeminiManager"
Cohesion: 0.11
Nodes (11): GeminiManager, Centralized, resilient Google Generative AI / Gemini API Manager. Orchestrates:…, Scenario 3: Both keys fail on Model A -> Model B succeeds on Key 1., Scenario 4: Key 1 invalid (401/403) -> Permanently disabled, switches to Key 2., Scenario 5: All keys/models fail -> Raises clean application error, no crash., Scenario 6: Transient 500 error on attempt 1, succeeds on attempt 2., Scenario 7: 400 Invalid Argument fails fast without wasting retries., Verify proper classification of SDK exceptions. (+3 more)

### Community 8 - "gemini_manager.py"
Cohesion: 0.20
Nodes (11): ErrorClassifier, GeminiErrorType, GeminiKeyStatus, mask_key(), Enum, Exception, SignBridge Centralized Google Generative AI / Gemini API Management System.…, Classifies raw SDK and HTTP exceptions into actionable error enums. (+3 more)

### Community 9 - ".generate"
Cohesion: 0.22
Nodes (7): GroqResult, TypedDict, Execute robust generation across Groq (Model x Key) matrix., Low-level Groq completions caller., LandmarkValidationError, Raised when incoming landmark data doesn't match the expected shape., ValueError

### Community 10 - "KeyManager"
Cohesion: 0.18
Nodes (6): KeyManager, Manages collection of API keys, state tracking, and health-aware rotation., Discover and load keys from environment variables., Return all currently usable keys, sorted by priority (KEY_1 -> KEY_2)., Record a failure and apply appropriate cooldown/backoff., Reset all cooldowns and failures.

### Community 11 - "TestGroqManager"
Cohesion: 0.11
Nodes (11): GroqManager, Centralized, resilient Groq LPU API Manager. Orchestrates: Model Selection ->…, Scenario 3: Both keys fail on Model A -> Model B succeeds on Key 1., Scenario 4: Key 1 invalid (401/403) -> Permanently disabled, switches to Key 2., Scenario 5: All keys/models fail -> Raises clean runtime error., Scenario 6: Transient 500 error on attempt 1, succeeds on attempt 2., Scenario 7: 400 Invalid Argument fails fast without wasting retries., Verify proper classification of Groq exceptions. (+3 more)

### Community 12 - "KeyInfo"
Cohesion: 0.20
Nodes (6): KeyInfo, Initialize modern or legacy SDK client for this key., Check if key is ready for requests., Record successful request for a key., Runtime health and state tracker for an individual API key., Scenario 8: Model auto-discovery dynamically sorts granted models.

### Community 13 - "groq_manager.py"
Cohesion: 0.24
Nodes (10): ErrorClassifier, GroqErrorType, GroqKeyStatus, ModelSelector, Enum, Exception, SignBridge Centralized Groq LPU API Management System. Features: - Dual API Key…, Classifies raw Groq SDK and HTTP exceptions into actionable error enums. (+2 more)

### Community 14 - "WordRecognizer"
Cohesion: 0.20
Nodes (7): ISL whole-word gesture recognizer using temporal landmark sequences. Accepts a…, Ensure frames array is exactly SEQUENCE_LENGTH frames long., Return word recognizer metadata., Attempt to load the trained word model. Loading priority: 1. CNN-BiLSTM hybrid…, WordRecognizer, SignBridge — Automated Word Model Self-Check & Validation Suite Evaluates…, run_checks()

### Community 15 - "SpatialGraphConv"
Cohesion: 0.27
Nodes (6): build_dual_hand_adjacency(), SignBridge — Spatial-Temporal Graph Convolutional Network (ST-GCN) for ISL…, Builds the 42x42 normalized adjacency matrix with self-loops for dual hands., Spatial Graph Convolution layer. Transforms node features using the adjacency…, SpatialGraphConv, Tensor

### Community 17 - ".generate"
Cohesion: 0.33
Nodes (4): GeminiResult, TypedDict, Execute robust generation across (Model x Key) matrix. Algorithm: 1. Select…, Low-level SDK caller handling both modern genai and legacy SDKs.

### Community 18 - "KeyInfo"
Cohesion: 0.17
Nodes (7): KeyInfo, Get ordered list of candidate models (latest -> fallback). Prioritizes…, Query Groq API to discover active models for this key., Runtime health and state tracker for an individual Groq API key., Initialize Groq client for this key., Check if key is ready for requests., Scenario 8: Model auto-discovery dynamically sorts granted models.

### Community 19 - "KeyManager"
Cohesion: 0.15
Nodes (7): KeyManager, Manages collection of Groq API keys, state tracking, and health-aware rotation., Discover and load keys from environment variables., Return all currently usable keys, sorted by priority (KEY_1 -> KEY_2)., Record successful request for a key., Record a failure and apply appropriate cooldown/backoff., Reset all cooldowns and failures.

### Community 20 - ".get_health_status"
Cohesion: 0.29
Nodes (4): Any, Sanitized dictionary for status reporting (no plain-text keys)., Check if at least one API key is configured and not permanently invalid., Comprehensive health status for telemetry and monitoring.

### Community 21 - ".get_health_status"
Cohesion: 0.29
Nodes (4): Any, Sanitized dictionary for status reporting (no plain-text keys)., Check if at least one Groq API key is configured and not permanently invalid., Comprehensive health status for telemetry and monitoring.

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

### Community 28 - "PROJECT_CONTEXT.md"
Cohesion: 0.10
Nodes (19): 10. DevOps, Deployment & Verification, 11. Complete Project Change Log, 1. Project Overview, 2. Tech Stack, 3. System Architecture & Data Flow, 4. Complete Directory & File Map, 5. Machine Learning Models & Active Performance Metrics, 6. Datasets, Modalities & Feature Engineering (+11 more)

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

### Community 37 - "extract_video_landmarks.py"
Cohesion: 0.19
Nodes (14): augment_raw_sequence(), clean_word_label(), get_landmark_vector_from_result(), main(), _normalize_word_str(), process_video(), SignBridge — Batch Video Landmark Extractor (v2) + Heavy Augmentation Extracts…, Resamples a list of landmark frames to exactly target_length frames. (+6 more)

### Community 38 - "2. Ponytail (Lazy Senior Dev Mode)"
Cohesion: 0.33
Nodes (5): 1. Graphify Knowledge Graph, 2. Ponytail (Lazy Senior Dev Mode), Core Rules, SignBridge AI Agent Guidelines, The Ladder

### Community 39 - "graphify reference: query, path, explain"
Cohesion: 0.33
Nodes (5): For /graphify explain, For /graphify path, graphify reference: query, path, explain, Step 0 — Constrained query expansion (REQUIRED before traversal), Step 1 — Traversal

### Community 40 - "normalize_landmarks"
Cohesion: 0.40
Nodes (3): normalize_landmarks(), Sign-Bridge Flask API — ISL Word Recognition Service Loads a trained model that…, Predict a word from a sequence of landmark frames with optional MediaPipe…

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

### Community 58 - "STGCNHandClassifier"
Cohesion: 0.25
Nodes (4): ST-GCN Classifier for Static / Frame-Level Dual-Hand ISL Gestures. Input:…, STGCNHandClassifier, Attempt to load trained models; fall back to heuristic. Loading priority: 1.…, Parse the Mendeley CSV into a lookup dictionary.

### Community 59 - "TranslatorModel"
Cohesion: 0.17
Nodes (13): ISL Gesture-to-Text translation model. Supports four ML modes: 1. Hybrid…, Predict the ISL letter from a set of hand landmarks with optional Holistic body…, Run inference through the calibrated ST-GCN + XGBoost Hybrid Ensemble., Run inference through the trained PyTorch ST-GCN model., Run inference through the trained XGBoost model., Run inference through the trained Keras model., Rule-based heuristic prediction using geometric features extracted from 42 hand…, Parse landmarks into a list of 42 {x, y, z} dicts. Handles both flat arrays and… (+5 more)

### Community 60 - "translator_model.py"
Cohesion: 0.27
Nodes (8): _compute_angle_cos(), extract_features(), extract_holistic_features(), Enriched Feature extraction for Indian Sign Language (ISL) alphabet…, Extracts geometric feature vector enriched with upper-body anchor metrics. If…, Compute cosine of the 3D angle at vertex p_b between vectors (p_a - p_b) and…, Extract a 208-D normalized geometric feature vector from 126 raw landmark…, Sign-Bridge Flask API — ISL Gesture Translation Model Service Loads trained…

### Community 61 - "load_dataset_partitioned"
Cohesion: 0.25
Nodes (9): SignBridge — Benchmark Standalone XGBoost, Standalone ST-GCN, and Calibrated…, run_benchmark(), get_file_sha256(), load_dataset_partitioned(), Any, Clean, Leak-Free Data Loader and Partition Manager for SignBridge ISL Alphabet…, Compute SHA-256 hash of a file., Load dataset strictly partitioned by origin without frame-level leakage.… (+1 more)

### Community 63 - "ModelSelector"
Cohesion: 0.33
Nodes (4): ModelSelector, Task-aware model selector with auto-discovery and capability matching., Get ordered list of candidate models (latest -> fallback) suitable for task. If…, Query the API to find which models this specific key has access to.

### Community 64 - "mask_key"
Cohesion: 0.50
Nodes (3): mask_key(), Safely mask an API key for logs (e.g., 'gsk_...4X9Z'). Never logs full key., Verify API keys are masked for logs.

## Knowledge Gaps
- **204 isolated node(s):** `$schema`, `plugins`, `react/rules-of-hooks`, `react/only-export-components`, `name` (+199 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 433 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **8 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `TranslatorModel` connect `TranslatorModel` to `app.py`, `STGCNHandClassifier`, `translator_model.py`, `load_dataset_partitioned`, `.get_info`?**
  _High betweenness centrality (0.045) - this node is a cross-community bridge._
- **Why does `load_dataset_partitioned()` connect `load_dataset_partitioned` to `normalize_landmarks`, `train_unified.py`?**
  _High betweenness centrality (0.034) - this node is a cross-community bridge._
- **Why does `ArduinoSerial` connect `ArduinoSerial` to `app.py`?**
  _High betweenness centrality (0.024) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `TestGeminiManager` (e.g. with `ErrorClassifier` and `GeminiErrorType`) actually correct?**
  _`TestGeminiManager` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `TestGroqManager` (e.g. with `ErrorClassifier` and `GroqErrorType`) actually correct?**
  _`TestGroqManager` has 6 INFERRED edges - model-reasoned connections that need verification._
- **What connects `$schema`, `plugins`, `react/rules-of-hooks` to the rest of the system?**
  _204 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `react` be split into smaller, more focused modules?**
  _Cohesion score 0.0625694187338023 - nodes in this community are weakly interconnected._