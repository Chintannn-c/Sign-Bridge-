<!-- converted from Sign-Bridge_Software_Hardware_Build_Guide .docx -->


SIGN-BRIDGE
Software & Hardware Build Guide

This document is the companion technical build guide to the Sign-Bridge Implementation Plan. It covers the full software architecture, actual starter code (delivered as a separate GitHub-ready repository), hardware assembly and wiring for both hands, the GitHub workflow for a 4-person team, and a researched answer on Indian Sign Language datasets.
A working code repository (firmware, backend, frontend, dataset tooling) accompanies this document — see the accompanying .zip file.

# 1. Software Architecture — Where Everything Runs
Sign-Bridge is a single laptop/PC application with three moving parts, all on the SAME machine (no separate servers needed):

Why this layout: keeping frontend and backend on one machine avoids network/latency complexity for a hardware demo — the browser just points at localhost. If you later want the hearing user on a phone and the hands elsewhere, the same Flask backend can be exposed on the local WiFi network without code changes (just replace localhost with the laptop's LAN IP).
## 1.1 Path 1 in code terms (ISL → text/speech)
- backend/vision/capture_landmarks.py + train_classifier.py build the model offline, once.
- backend/vision/recognize.py runs every frame: webcam → MediaPipe landmarks → classifier → predicted letter/word + confidence.
- backend/app.py's video_worker() thread calls recognize.py continuously and pushes results to the browser over Socket.IO (“sign_update” event) plus streams the annotated webcam feed to /video_feed as MJPEG.
- frontend/app.js listens for “sign_update” and updates the recognized-text panel and confidence bar live.
## 1.2 Path 2 in code terms (text/speech → ISL hands)
- Hearing user types in the browser (frontend/index.html textarea) or clicks the microphone button.
- Typed text: frontend/app.js POSTs to /api/speak → backend/app.py speaks it with tts_stt.speak() AND drives the hands with serial_bridge.py's sign_text().
- Spoken input: frontend/app.js POSTs to /api/listen → backend uses tts_stt.listen_once() (speech-to-text) to get a string, then signs it the same way.
- serial_bridge.py looks up each word in sign_library.json's word library first (whole-word ISL signs); anything not found is fingerspelled letter-by-letter using the same file's letter table.
- Each POSE command is sent over USB serial to the Mega, which smoothly moves all 10 servos and replies “OK” before the next command is sent.

# 2. Display Interface (What the Browser Looks Like)
The frontend is a single page (frontend/index.html) with two side-by-side panels, matching the two communication paths one-to-one so either user immediately understands which side is theirs:
- Left panel (Path 1): live webcam feed with hand-landmark overlay, a large recognized-sign readout, a confidence bar, and a scrolling transcript log of everything recognized so far.
- Right panel (Path 2): a text box for the hearing user to type into, a “Send & Sign” button, a “Speak Instead” microphone button, a “Relax Hands” reset button, and two hand icons (L/R) that light up while the robot is physically signing, so the hearing user gets visual confirmation the message is being delivered.
- A connection-status pill top-right shows whether the backend/Arduino link is live, so the team can spot a disconnected USB cable immediately during a demo.
This is implemented with plain HTML/CSS/JS (frontend/index.html, style.css, app.js in the repo) — no build step or framework needed, which keeps it simple to hand off between 4 teammates and easy to demo on any laptop.

# 3. How the Webcam Captures and Interprets Signs
Sign-Bridge uses Google's MediaPipe Hands library rather than training a model on raw pixels. The pipeline per frame is:
- OpenCV (cv2.VideoCapture) grabs a frame from the laptop's built-in or USB webcam, ~30 times per second.
- MediaPipe Hands processes the frame and, whenever a hand is visible, returns 21 (x, y, z) landmark points per hand (fingertips, knuckles, wrist) — whether it's the left or right hand is also reported.
- Those 21x3 = 63 numbers per hand are flattened into a single feature vector and fed to a trained classifier (RandomForest in the starter code) that outputs the most likely ISL letter/word and a confidence score.
- A small smoothing window (backend/vision/recognize.py) requires the same prediction to repeat across several consecutive frames above a confidence threshold before it's accepted — this avoids the display flickering between letters as a hand moves into position.
- For two-handed ISL letters, both hands' landmark vectors are captured; the starter code classifies on one hand for simplicity, and the dataset guide explains how to extend this to a concatenated 126-number two-hand vector once you're ready (see Section 6).
This landmark-based approach is why a plain webcam is enough — no sensor glove, no depth camera, and no GPU required for real-time use.

# 4. Hardware Assembly — Both Hands
## 4.1 Per-Hand Build Steps (repeat identically for left and right)
- Cut a hand-and-forearm outline from stiff cardboard/flattened plastic bottle, roughly life-size or slightly larger for easier straw-mounting.
- Cut drinking straws into 5-6 short segments per finger (one per knuckle) and glue them along the cardboard finger outline, leaving a small gap at each joint so the finger can bend there.
- Thread non-stretch string (fishing line or dental floss) through all straw segments of one finger; knot/anchor it firmly at the fingertip (a small bead or knot larger than the straw opening works as a stopper).
- Repeat for all 5 fingers, then route all 5 strings down through the palm to the forearm/servo housing.
- Mount a small cardboard or plastic box on the forearm to house the 5 servos for that hand — one servo per finger string.
- Tie each finger's string to its servo horn. When the servo rotates one way it winds the string and curls the finger; add a light elastic band from fingertip back to the base as a return spring so the finger straightens when the servo rotates back.
- Reinforce the palm-to-forearm joint and the servo-mounting box with extra cardboard layers or hot glue — this is the highest-stress point and the most common failure point in scrap builds.
## 4.2 Wiring Both Hands to One Arduino Mega
Both hands share a single Arduino Mega (needed for its 10+ PWM-capable pins). Suggested pin assignment, matching the firmware in firmware/sign_bridge_mega/sign_bridge_mega.ino:

- All 10 servo signal wires go to the pins above.
- All 10 servo ground wires + the Mega's GND pin all connect to a common ground rail (breadboard ground rail works well).
- All 10 servo +5V wires connect to an external 5V 4A power supply's positive rail — NOT the Mega's onboard 5V pin, which cannot supply enough current for 10 servos moving together and will brown out/reset the Mega.
- The external power supply's ground must also be tied to the same common ground rail as the Mega, so signal timing stays synchronized (a shared ground reference is required even though power is separate).
- Connect the Mega to the laptop via USB for both power-to-the-Mega-logic and the serial link the backend uses.
## 4.3 Hardware ↔ Software Integration
- Flash firmware/sign_bridge_mega/sign_bridge_mega.ino to the Mega once, via Arduino IDE (Board: Arduino Mega 2560).
- The Mega then just listens on USB serial at 115200 baud for two plain-text commands: POSE (move all 10 servos to given angles) and HOME (relax to rest).
- backend/serial_bridge.py opens that same serial port from Python and sends POSE/HOME lines built from backend/sign_library.json's per-letter/word angle tables — this file is the single source of truth linking “what sign to make” to “what angles to send.”
- Because scrap joints flex differently per physical build, EVERY angle in sign_library.json starts as a placeholder and must be tuned using backend/calibration_tool.py, which lets you nudge each finger's angle live on the real hand and save the result.

# 5. Using GitHub With a 4-Person Team
Yes — GitHub is the right tool here, both for version control and for splitting work across 4 people without stepping on each other. The provided repository already includes the scaffolding for this:
## 5.1 Suggested Role Split

## 5.2 Branching & Review
- main = always demoable, protected from direct pushes.
- dev = integration branch; features merge here first.
- Feature branches named feature/<area>-<short-description>, e.g. feature/vision-two-hand-classifier.
- Every PR needs at least one teammate's review before merging into dev; periodically merge dev → main once stable.
- A CODEOWNERS file (included) auto-requests the right teammate as reviewer based on which folder a PR touches.
## 5.3 Project Tracking
- Use GitHub Projects (Kanban board) with columns Backlog → To Do → In Progress → In Review → Done.
- Label issues by area (hardware, firmware, backend, frontend, dataset, bug) so work is easy to filter per person.
- Issue templates for bug reports and feature requests are included under .github/ISSUE_TEMPLATE/.
## 5.4 Handling Large Files
- Trained model files and raw dataset captures can get large; the included .gitignore keeps them out of normal commits by default.
- If the team wants dataset files versioned too, use Git LFS (git lfs track "dataset/raw/**") rather than committing large CSV/video files directly.

# 6. ISL Datasets — What Exists, and How to Build Your Own
## 6.1 ISL-Specific Datasets Beyond Kaggle

Caveat: these are image/video datasets, not landmark datasets, so using them with the MediaPipe-landmark pipeline in this repo means re-running MediaPipe over their footage yourself to extract landmarks — or building your own landmark dataset directly, which is faster for an MVP. ISL also has regional signing variation and no single centrally standardized ML-ready dataset the way some languages do; the ISLRTC (Indian Sign Language Research and Training Centre) publishes a reference video glossary that's useful for validating signs even though it isn't packaged for training.
## 6.2 Building Your Own Dataset (Recommended for the MVP)
- Collect all target ISL letters (A-Z) and digits (0-9) you plan to support, noting one-handed vs two-handed from a reference chart, plus 5-10 whole words (hello, thank you, yes, no, help, water, food, family, etc.).
- Aim for ~150-300 landmark samples per class, with variation in signer, lighting, background, and hand distance/rotation — collected across the 4 team members in parallel.
- Also record “no sign” negative samples (empty frame / resting hand) so the classifier can decline to guess.
- Use the included backend/vision/capture_landmarks.py tool: run it once per label, press SPACE to save each sample, and it writes straight to dataset/raw/<label>.csv.
- Have the Dataset/QA lead spot-check samples against the ISL reference chart/ISLRTC glossary before training, then run backend/vision/train_classifier.py and check per-class precision/recall — any class under ~80% recall needs more or better samples.

Full step-by-step instructions are also included in dataset/README.md inside the repository.

Totally fair question — let's slow down and walk through it like a checklist. "Making a dataset" here just means: sit in front of your webcam, hold up an ISL sign, press a key, repeat. That's it. Here's the exact sequence.
Step 0 — One-time setup (do this once, on one laptop)
cd sign-bridge/backend
pip install -r requirements.txt
This installs OpenCV and MediaPipe, which is all the capture script needs.
Step 1 — Pick what you're recording first
Don't try to do all 26 letters + 10 words in one sitting. Start small:
Pick 3 letters to start (e.g. A, B, V — one one-handed, one one-handed, one two-handed)
This proves the whole pipeline works before you invest hours in it
Step 2 — Run the capture script
cd sign-bridge/backend/vision
python capture_landmarks.py --label A --hands 1
A webcam window pops up. Now, physically:
Hold up the ISL sign for "A" in front of your camera
Press the spacebar — you'll see "Saved sample #1" in the terminal
Move your hand slightly (tilt it, move it a bit closer/farther) — don't hold dead-still
Press spacebar again — "Saved sample #2"
Repeat this ~150–200 times for this one letter (yes, that many — it's fast, maybe 5–8 minutes of pressing spacebar while holding the sign in slightly different ways)
Press q to quit when done
That's it — a file called dataset/raw/A.csv now exists with 150–200 rows in it. You never touch the CSV directly.
Step 3 — Repeat for each label
python capture_landmarks.py --label B --hands 1
python capture_landmarks.py --label V --hands 2    # two-handed letter — show both hands
Each command makes its own CSV (B.csv, V.csv).
Step 4 — Split this across your 4 people
This is the boring, time-consuming part, so divide it up:
Person 1 records letters A–G
Person 2 records letters H–N
Person 3 records letters O–T
Person 4 records letters U–Z + your 5–10 target words
Each person runs the same script on their own laptop, then someone collects all the CSVs into one shared dataset/raw/ folder (or pushes them to the GitHub repo).
Step 5 — Sanity check, then train
Once you have CSVs for your first batch of labels:
python train_classifier.py --data ../../dataset/raw --out model.pkl
This prints accuracy per letter. If a letter shows low accuracy, that usually just means "record more samples for that one" — not a bigger problem.
A few honest tips before you start:
Do this in a well-lit room — MediaPipe struggles in dim lighting
Vary your hand position between presses (this is the single biggest thing that makes the classifier actually work later, vs. just memorizing one exact pose)
Don't aim for perfection on your first letter — get A working end-to-end (record → train → check accuracy) before scaling to all 26, so you catch any setup problems early
Want me to walk through what "checking accuracy" actually looks like once you've recorded your first letter, or is the recording process itself clear now?


# 7. What's Included in the Companion Repository
- firmware/sign_bridge_mega/sign_bridge_mega.ino — full Arduino Mega servo control firmware
- backend/app.py, serial_bridge.py, tts_stt.py, calibration_tool.py, sign_library.json — Flask server, Arduino bridge, speech I/O, calibration tool, sign-to-angle mapping
- backend/vision/capture_landmarks.py, train_classifier.py, recognize.py — full dataset collection, training, and real-time recognition pipeline
- frontend/index.html, style.css, app.js — the working display interface described in Section 2
- dataset/README.md — the dataset guide from Section 6, plus dataset/raw/ as the drop folder for collected CSVs
- README.md, CONTRIBUTING.md, CODEOWNERS, .gitignore, .github/ISSUE_TEMPLATE/ — everything needed to push this straight to a new GitHub repository and start team collaboration

All Python files have been syntax-checked and the JSON/JS validated; every servo angle in sign_library.json is explicitly marked as a placeholder pending physical calibration on your specific build, as described in Section 4.3.
| Layer | Technology | Runs where | Talks to |
| --- | --- | --- | --- |
| Firmware | Arduino C++ (.ino) | On the Arduino Mega itself | Backend, over USB serial |
| Backend | Python (Flask + Flask-SocketIO) | On the team laptop wired to the Mega | Webcam, mic, Arduino (serial), Frontend (HTTP/WebSocket) |
| Frontend | HTML/CSS/JS (served by Flask) | Browser on the SAME laptop, http://localhost:5000 | Backend only |
| Hand | Finger | Mega Pin |
| --- | --- | --- |
| Left | Thumb | 2 |
| Left | Index | 3 |
| Left | Middle | 4 |
| Left | Ring | 5 |
| Left | Pinky | 6 |
| Right | Thumb | 7 |
| Right | Index | 8 |
| Right | Middle | 9 |
| Right | Ring | 10 |
| Right | Pinky | 11 |
| Person | Owns | Repo Folder |
| --- | --- | --- |
| Hardware lead | Mechanical build, wiring, firmware | firmware/ |
| Backend/ML lead | Vision pipeline, classifier, serial bridge | backend/, backend/vision/ |
| Frontend lead | Display interface, UX | frontend/ |
| Dataset/QA lead | Data collection, labeling, calibration, testing | dataset/, calibration_tool.py |
| Dataset | Scope | Size | Source | Fit |
| --- | --- | --- | --- | --- |
| INCLUDE (AI4Bharat / IIT Madras) | Word-level ISL, 263 signs, 15 categories | 4,287 videos, ~0.27M frames | github.com/AI4Bharat/INCLUDE; also on Zenodo & HuggingFace | Best fit for the whole-word vocabulary layer |
| ISLTranslate (IIT Kanpur) | Continuous ISL-English sentence pairs | ~31,000 sentence pairs | github.com/Exploration-Lab/ISLTranslate | For later full-sentence translation, not MVP |
| Community alphabet datasets | Fingerspelled A-Z images | ~12,000-13,000 images | Individual GitHub repos (search “Indian Sign Language dataset”) | Bootstrapping an image classifier; verify license & signs first |