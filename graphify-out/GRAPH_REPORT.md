# Graph Report - SignBridge  (2026-09-28)

## Corpus Check
- 675 files · ~7,207,049 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 738 nodes · 1014 edges · 57 communities (47 shown, 7 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 17 edges (avg confidence: 0.93)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `8cae6add`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- react
- TranslatorModel
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
- OneEuroFilter
- .generate
- KeyInfo
- KeyManager
- .get_health_status
- ModelSelector
- .oxlintrc.json
- SignBridge: Dual-Communication Indian Sign Language (ISL) Translation & Robotic Actuation System
- What You Must Do When Invoked
- 4. Detailed Implementation Plan
- PROJECT_CONTEXT.md
- Sign-Bridge_Software_Hardware_Build_Guide _9244e8e7.md
- extract_video_landmarks.py
- AI_MEMORY.md
- graphify reference: extra exports and benchmark
- Ponytail
- Ponytail Help
- ingest_dataset_2.py
- ingest_zip_dataset.py
- ISLTranslate/README.md
- 2. Ponytail (Lazy Senior Dev Mode)
- graphify reference: query, path, explain
- ModelSelector
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

## God Nodes (most connected - your core abstractions)
1. `SignBridge: Dual-Communication Indian Sign Language (ISL) Translation & Robotic Actuation System` - 22 edges
2. `react` - 21 edges
3. `TranslatorModel` - 18 edges
4. `TestGeminiManager` - 18 edges
5. `TestGroqManager` - 18 edges
6. `KeyInfo` - 16 edges
7. `KeyInfo` - 16 edges
8. `ArduinoSerial` - 13 edges
9. `lucide-react` - 12 edges
10. `What You Must Do When Invoked` - 12 edges

## Surprising Connections (you probably didn't know these)
- `TestGeminiManager` --uses--> `GeminiKeyStatus`  [INFERRED]
  backend/tests/test_gemini_manager.py → backend/services/gemini_manager.py
- `TestGeminiManager` --uses--> `GeminiErrorType`  [INFERRED]
  backend/tests/test_gemini_manager.py → backend/services/gemini_manager.py
- `TestGeminiManager` --uses--> `KeyInfo`  [INFERRED]
  backend/tests/test_gemini_manager.py → backend/services/gemini_manager.py
- `TestGeminiManager` --uses--> `ErrorClassifier`  [INFERRED]
  backend/tests/test_gemini_manager.py → backend/services/gemini_manager.py
- `TestGeminiManager` --uses--> `ModelSelector`  [INFERRED]
  backend/tests/test_gemini_manager.py → backend/services/gemini_manager.py

## Import Cycles
- None detected.

## Communities (57 total, 7 thin omitted)

### Community 0 - "react"
Cohesion: 0.07
Nodes (42): framer-motion, lucide-react, react, App(), CameraView(), GestureReferenceSheet(), HumanPanel, RobotPanel (+34 more)

### Community 1 - "TranslatorModel"
Cohesion: 0.05
Nodes (49): ST-GCN Classifier for Static / Frame-Level Dual-Hand ISL Gestures. Input:…, STGCNHandClassifier, get_file_sha256(), load_dataset_partitioned(), Any, Clean, Leak-Free Data Loader and Partition Manager for SignBridge ISL Alphabet…, Compute SHA-256 hash of a file., Load dataset strictly partitioned by origin without frame-level leakage.… (+41 more)

### Community 2 - "app.py"
Cohesion: 0.05
Nodes (60): clean_llm_text(), collect_data(), generate_llm_response(), get_history(), get_smart_fallback_response(), health(), llm_answer(), llm_refine() (+52 more)

### Community 3 - "package.json"
Cohesion: 0.07
Nodes (26): dependencies, framer-motion, lucide-react, react, react-dom, devDependencies, oxlint, @types/react (+18 more)

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
Nodes (11): GeminiManager, Centralized, resilient Google Generative AI / Gemini API Manager. Orchestrates:…, Scenario 4: Key 1 invalid (401/403) -> Permanently disabled, switches to Key 2., Scenario 5: All keys/models fail -> Raises clean application error, no crash., Scenario 6: Transient 500 error on attempt 1, succeeds on attempt 2., Scenario 7: 400 Invalid Argument fails fast without wasting retries., Verify proper classification of SDK exceptions., Scenario 1: Key 1 works normally -> Returns response immediately. (+3 more)

### Community 8 - "gemini_manager.py"
Cohesion: 0.20
Nodes (11): ErrorClassifier, GeminiErrorType, GeminiKeyStatus, mask_key(), Enum, Exception, SignBridge Centralized Google Generative AI / Gemini API Management System.…, Classifies raw SDK and HTTP exceptions into actionable error enums. (+3 more)

### Community 9 - ".generate"
Cohesion: 0.33
Nodes (4): GroqResult, TypedDict, Execute robust generation across Groq (Model x Key) matrix., Low-level Groq completions caller.

### Community 10 - "KeyManager"
Cohesion: 0.18
Nodes (6): KeyManager, Manages collection of API keys, state tracking, and health-aware rotation., Discover and load keys from environment variables., Return all currently usable keys, sorted by priority (KEY_1 -> KEY_2)., Record a failure and apply appropriate cooldown/backoff., Reset all cooldowns and failures.

### Community 11 - "TestGroqManager"
Cohesion: 0.09
Nodes (13): GroqManager, Centralized, resilient Groq LPU API Manager. Orchestrates: Model Selection ->…, Check if at least one Groq API key is configured and not permanently invalid., Comprehensive health status for telemetry and monitoring., Scenario 4: Key 1 invalid (401/403) -> Permanently disabled, switches to Key 2., Scenario 5: All keys/models fail -> Raises clean runtime error., Scenario 6: Transient 500 error on attempt 1, succeeds on attempt 2., Scenario 7: 400 Invalid Argument fails fast without wasting retries. (+5 more)

### Community 12 - "KeyInfo"
Cohesion: 0.20
Nodes (6): KeyInfo, Initialize modern or legacy SDK client for this key., Check if key is ready for requests., Record successful request for a key., Runtime health and state tracker for an individual API key., Scenario 8: Model auto-discovery dynamically sorts granted models.

### Community 13 - "groq_manager.py"
Cohesion: 0.20
Nodes (11): ErrorClassifier, GroqErrorType, GroqKeyStatus, mask_key(), Enum, Exception, SignBridge Centralized Groq LPU API Management System. Features: - Dual API Key…, Classifies raw Groq SDK and HTTP exceptions into actionable error enums. (+3 more)

### Community 14 - "WordRecognizer"
Cohesion: 0.18
Nodes (7): Sign-Bridge Flask API — ISL Word Recognition Service Loads a trained model that…, Predict a word from a sequence of landmark frames. Args: frame_sequence: list…, Ensure frames array is exactly SEQUENCE_LENGTH frames long., ISL whole-word gesture recognizer using temporal landmark sequences. Accepts a…, Return word recognizer metadata., Attempt to load the trained word model. Loading priority: 1. CNN-BiLSTM hybrid…, WordRecognizer

### Community 15 - "SpatialGraphConv"
Cohesion: 0.27
Nodes (6): build_dual_hand_adjacency(), SignBridge — Spatial-Temporal Graph Convolutional Network (ST-GCN) for ISL…, Builds the 42x42 normalized adjacency matrix with self-loops for dual hands., Spatial Graph Convolution layer. Transforms node features using the adjacency…, SpatialGraphConv, Tensor

### Community 17 - ".generate"
Cohesion: 0.33
Nodes (4): GeminiResult, TypedDict, Execute robust generation across (Model x Key) matrix. Algorithm: 1. Select…, Low-level SDK caller handling both modern genai and legacy SDKs.

### Community 18 - "KeyInfo"
Cohesion: 0.18
Nodes (7): KeyInfo, Any, Sanitized dictionary for status reporting (no plain-text keys)., Runtime health and state tracker for an individual Groq API key., Initialize Groq client for this key., Check if key is ready for requests., Scenario 8: Model auto-discovery dynamically sorts granted models.

### Community 19 - "KeyManager"
Cohesion: 0.15
Nodes (7): KeyManager, Manages collection of Groq API keys, state tracking, and health-aware rotation., Discover and load keys from environment variables., Return all currently usable keys, sorted by priority (KEY_1 -> KEY_2)., Record successful request for a key., Record a failure and apply appropriate cooldown/backoff., Reset all cooldowns and failures.

### Community 20 - ".get_health_status"
Cohesion: 0.29
Nodes (4): Any, Sanitized dictionary for status reporting (no plain-text keys)., Check if at least one API key is configured and not permanently invalid., Comprehensive health status for telemetry and monitoring.

### Community 21 - "ModelSelector"
Cohesion: 0.33
Nodes (4): ModelSelector, Task-aware Groq model selector with auto-discovery and capability sorting., Get ordered list of candidate models (latest -> fallback). Prioritizes…, Query Groq API to discover active models for this key.

### Community 22 - ".oxlintrc.json"
Cohesion: 0.33
Nodes (5): plugins, rules, react/only-export-components, react/rules-of-hooks, $schema

### Community 23 - "SignBridge: Dual-Communication Indian Sign Language (ISL) Translation & Robotic Actuation System"
Cohesion: 0.06
Nodes (33): 1. Alphabet Recognition (Tier 1), 1. Clone the Repository, 2. Backend Setup, 2. Temporal Word Recognition (Tier 2), 3. Frontend Setup, 4. Running the Application, Acknowledgements, AI/ML Model Details (+25 more)

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

### Community 30 - "extract_video_landmarks.py"
Cohesion: 0.22
Nodes (12): augment_raw_sequence(), clean_word_label(), get_landmark_vector_from_result(), main(), process_video(), SignBridge — Batch Video Landmark Extractor (v2) + Heavy Augmentation Extracts…, Apply heavy augmentation to a raw sequence of frames before resampling.…, Processes a single video into raw landmark frames (variable length). (+4 more)

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

### Community 37 - "ISLTranslate/README.md"
Cohesion: 0.29
Nodes (6): Citation, **Comparison Scores:**, Download Dataset, ISL-Signer Validation, ISLTranslate: Dataset for Translating Indian Sign Language, License

### Community 38 - "2. Ponytail (Lazy Senior Dev Mode)"
Cohesion: 0.33
Nodes (5): 1. Graphify Knowledge Graph, 2. Ponytail (Lazy Senior Dev Mode), Core Rules, SignBridge AI Agent Guidelines, The Ladder

### Community 39 - "graphify reference: query, path, explain"
Cohesion: 0.33
Nodes (5): For /graphify explain, For /graphify path, graphify reference: query, path, explain, Step 0 — Constrained query expansion (REQUIRED before traversal), Step 1 — Traversal

### Community 40 - "ModelSelector"
Cohesion: 0.33
Nodes (4): ModelSelector, Task-aware model selector with auto-discovery and capability matching., Get ordered list of candidate models (latest -> fallback) suitable for task. If…, Query the API to find which models this specific key has access to.

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

## Knowledge Gaps
- **204 isolated node(s):** `$schema`, `plugins`, `react/rules-of-hooks`, `react/only-export-components`, `name` (+199 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 415 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `TranslatorModel` connect `TranslatorModel` to `app.py`?**
  _High betweenness centrality (0.038) - this node is a cross-community bridge._
- **Why does `normalize_landmarks()` connect `TranslatorModel` to `WordRecognizer`?**
  _High betweenness centrality (0.026) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `TestGeminiManager` (e.g. with `ErrorClassifier` and `GeminiErrorType`) actually correct?**
  _`TestGeminiManager` has 6 INFERRED edges - model-reasoned connections that need verification._
- **What connects `$schema`, `plugins`, `react/rules-of-hooks` to the rest of the system?**
  _204 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `react` be split into smaller, more focused modules?**
  _Cohesion score 0.06599597585513078 - nodes in this community are weakly interconnected._
- **Should `TranslatorModel` be split into smaller, more focused modules?**
  _Cohesion score 0.05267778753292362 - nodes in this community are weakly interconnected._
- **Should `app.py` be split into smaller, more focused modules?**
  _Cohesion score 0.05499735589635114 - nodes in this community are weakly interconnected._