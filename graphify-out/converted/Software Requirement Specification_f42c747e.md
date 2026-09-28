<!-- converted from Software Requirement Specification.docx -->

SIGNBRIDGE: A DUAL-COMMUNICATION ROBOTIC TRANSLATOR FOR INDIAN SIGN LANGUAGE (ISL)
A Minor Project Report - Software Requirement Specification
Submitted By:
(23BT04076)       Stuti Mistry
(24BT04D225)      Foram Patel
(24BT04D231)      Chintan Sharma
(24BT04D236)      Parth Thakkar
In partial fulfilment of the requirement for the award of the degree of
Bachelor of Technology In
Semester VI - Computer Science and Engineering (AI-ML)
Department of Computer Science and Engineering
School of Technology
GSFC University,
Vigyan Bhavan, P. O. Fertilizer Nagar,
Vadodara - 391750, Gujarat, India
May 2026

Certificate
This is to certify that the report submitted along with the project entitled 'SIGNBRIDGE: A DUAL-COMMUNICATION ROBOTIC TRANSLATOR FOR INDIAN SIGN LANGUAGE (ISL)' has been carried out by Stuti Mistry (23BT04076), Foram Patel (24BT04D225), Chintan Sharma (24BT04D231), and Parth Thakkar (24BT04D236) under supervision for the partial fulfillment of the degree of Bachelor of Technology in Computer Science and Engineering (AI-ML) at GSFC University, Vadodara.
SIGNATURE (Guide / HOD)				SIGNATURE (Internal Examiner)
Date: May, 2026					Date: May, 2026

List Of Tables

List Of Figures

TABLE OF CONTENTS
CHAPTER 1: INTRODUCTION                                           1
1.1 Project Description                                         1
Problem Statement                                           1
Objectives                                                  2
1.2 Features                                                    2
1.3 Tools and Technology Used                                   3
1.4 Requirements for Hardware and Software                      4
CHAPTER 2: LITERATURE REVIEW                                      5
2.1 Origin of the Problem & History of the Topic                5
2.2 Comparative Study of Existing Technical Solutions and Their Drawbacks 6
CHAPTER 3: SYSTEM DESIGN                                          8
3.1 Use Case                                                    8
3.2 Activity Diagram                                            10
CHAPTER 4: IMPLEMENTATION OF PROJECT                              12
4.1 System Architecture & Data Flow                             12
4.2 Module Implementation                                       13
CHAPTER 5: SOFTWARE TESTING                                       16
5.1 Rationale for Testing                                       16
5.2 Levels of Testing                                           16
5.3 Testing Methods                                             17
5.3.1 Functionality Testing                                 17
5.3.2 UX & Feedback                                         18
CHAPTER 6: LIMITATIONS AND FUTURE ENHANCEMENT                     19
6.1 Existing Limitations                                        19
6.2 Future Improvements                                         20
CHAPTER 7: CONCLUSION                                             21
CHAPTER 8: REFERENCES                                             22

CHAPTER 1: INTRODUCTION
1.1 Project Description
Problem Statement
The first obstacle faced when establishing accessibility for Deaf and Hard-of-Hearing individuals lies within the 'Communication Gap.' Society relies predominantly on spoken and written languages, while Deaf signers communicate through Indian Sign Language (ISL).
Unlike American Sign Language (ASL), which uses a mostly one-handed manual alphabet, Indian Sign Language is fundamentally a two-handed sign language—most letters (A–Z), digits (0–9), and practical vocabulary require synchronized movements from both hands. Existing sign language translation tools are often limited by three major flaws: (1) they are one-way only (translating signs to text or text to signs, but rarely both); (2) they rely on expensive, intrusive sensor gloves that restrict movement; and (3) they utilize single-hand robotic models that cannot physically perform two-handed ISL manual alphabets.
Objectives
The goal of SignBridge is to develop an affordable, two-way (dual) robotic translation system that bridges the gap between Deaf signers and hearing individuals using webcam-based AI vision and low-cost robotic hands. The system aims to:
- Two-Way (Dual) Communication: Provide a bidirectional translation loop (ISL signs -> Spoken Speech/Text for hearing users, and Spoken Speech/Text -> Physical ISL robotic hand gestures for Deaf signers).
- Non-Invasive Vision Pipeline: Eliminate wearable sensor gloves by using standard laptop webcams with Google MediaPipe 3D hand-landmark tracking.
- Dual Scrap-Material Robotic Hands: Build low-cost dual 5-finger robotic hands driven by 10 micro servos to physically perform ISL manual alphabets and gestures.
- Public Kiosk Accessibility: Provide a touch-optimized, full-screen interactive interface suitable for deployment at hospital reception desks, public service counters, and educational institutions.
1.2 Features
The system is built around a guided 'Dual-Communication' progression:
Steps to be taken include:
- Commencing the Kiosk Session — Initializing the live webcam vision pipeline, server health check, and Arduino hardware link.
- ISL Gesture Input Path (Deaf -> Hearing) — Real-time 3D hand joint tracking, TensorFlow deep-learning classification, live text display, and spoken audio output via Text-to-Speech (TTS).
- Sentence Builder Assistance — Providing interactive touchscreen editing tools including Undo, Delete, Clear, Add Space, Copy, and Speak Aloud.
- Speech & Text Input Path (Hearing -> Deaf) — Speech-to-text microphone audio transcription or direct text input entry.
- Robotic Hand Actuation — Looking up pre-calibrated servo angle sets and transmitting serial commands to actuate the dual robotic hands in sequence.
- Dual Display Stream — Rendering live side-by-side panel status for both the signer ('YOU') and the robotic translator ('ROBOT').
1.3 Tools and Technology Used
For the project to be both powerful and user-friendly, the following development stack was used:
- React + Vite — Frontend library and build tool for managing state and rendering the responsive touchscreen kiosk UI.
- Python & Flask — Micro web framework hosting the backend REST API endpoints for translation model inference and serial communication.
- Google MediaPipe — Computer vision framework for real-time extraction of 42 3D hand landmark points (126 coordinate features per frame).
- TensorFlow / Keras — Machine learning platform used to train the Multi-Layer Perceptron (MLP) ISL gesture classifier.
- Arduino Mega 2560 — Microcontroller board utilized for simultaneous 10-channel PWM servo motor control.
- SG90 Micro Servos — 9g micro servo motors providing tendon-pulling tension for finger movements.
- PySerial — Python library for USB serial data communication between the Flask backend and Arduino board.
1.4 Requirements for Hardware and Software
- Hardware — Processor with standard CPU/GPU execution support (Intel i3/i5 or equivalent), 4GB+ RAM, Integrated/USB 720p Webcam, Microphone, Speakers, 1x Arduino Mega 2560, 10x SG90 Micro Servos, 5V 4A External Power Supply, dual scrap-material hand frames.
- Software Framework — Developed using React 18, Vite 5, Python 3.10+, Flask 3.0, MediaPipe, TensorFlow 2.x, PySerial, Arduino IDE.
- Operating System — Windows 10/11, macOS, or Linux.

CHAPTER 2: LITERATURE REVIEW
2.1 Origin of the Problem & History of the Topic
Sign Language Recognition (SLR) research began in the 1990s with physical sensor gloves (such as DataGlove) equipped with flex sensors and accelerometers. While precise, sensor gloves proved unsuited for widespread adoption due to high costs, delicate wiring, and tactile discomfort for native signers.
With the rise of computer vision in the 2010s, researchers shifted toward camera-based recognition using Convolutional Neural Networks (CNNs). However, Indian Sign Language (ISL) presented unique structural challenges compared to Western sign languages like ASL or BSL. Most ISL letters require two hands overlapping or touching key points (e.g. index finger touching opposite palm). This structural requirement makes single-hand gesture tools insufficient for real-world ISL communication.
2.2 Comparative Study of Existing Technical Solutions and Their Drawbacks
Below is a comparative analysis of existing communication solutions vs. the SignBridge dual-communication platform:
Literature Review Summary Table

CHAPTER 3. SYSTEM DESIGN
3.1 Use Case
The SignBridge system involves three primary user roles / actors: Deaf Signer, Hearing Speaker, and System Administrator.
- Deaf Signer — Signs in front of webcam; views real-time text output; uses live sentence builder controls (Undo, Delete, Clear, Speak).
- Hearing Speaker — Speaks into microphone or types text message; watches dual robotic hands physically perform ISL signs.
- System Administrator — Connects/disconnects Arduino Mega serial port; runs dataset importer scripts and model training pipelines.
3.2 Activity Diagram
The operational workflow consists of two parallel pipelines within the Dual-Communication Loop:
1. Input Path (Signer -> Hearing User): Webcam capture -> MediaPipe 42 keypoint extraction -> Landmark Normalization (wrist center & scale) -> Keras MLP Classifier inference -> Recognized text display -> Text-to-Speech audio engine.
2. Output Path (Hearing User -> Signer): Microphone speech-to-text / typed text input -> Character parsing -> Servo angle lookup table (ISL_SERVO_MAP) -> USB PySerial transmission -> Arduino Mega PWM driver -> 10x SG90 servo motor physical hand movement.

CHAPTER 4. IMPLEMENTATION OF PROJECT
4.1 System Architecture & Data Flow
SignBridge is structured into three integrated architectural tiers:
- 1. Presentation Layer (Frontend Kiosk UI): React + Vite single-page application rendering the touchscreen split kiosk UI (HumanPanel and RobotPanel).
- 2. Intelligence & API Layer (Backend Engine): Python Flask REST API (app.py) providing endpoints for model prediction (/api/translate), dataset collection (/api/collect_data), and robot control (/api/robot/sign).
- 3. Physical Hardware Layer (Robotic Servo Controller): Arduino Mega 2560 running a custom C++ sketch driving 10 PWM SG90 servos across dual scrap-material hand frames.
4.2 Module Implementation
Key implementation highlights include:
- Vision Module (CameraView.jsx): MediaPipe integration extracts 21 keypoints per hand in real-time.
- Classifier Module (translator_model.py): Loads isl_gesture_model.h5 and normalizes 126-element landmark arrays before classification.
- Arduino Serial Controller (arduino_serial.py): Manages serial connection at 9600 baud rate and maps letters A-Z to 10-servo angle arrays.

CHAPTER 5. SOFTWARE TESTING
5.1 Rationale for Testing
Testing ensures real-time gesture recognition accuracy (>95%), low API latency (<100ms), and reliable physical servo actuation without mechanical binding.
5.2 Levels of Testing
- Unit Testing — Tested MediaPipe landmark parsing, normalization functions, and PySerial message formatting.
- Integration Testing — Verified HTTP communication between React frontend and Flask API, and USB communication between Flask and Arduino.
- System Testing — Executed end-to-end live testing of both communication paths under varying room lighting.
5.3 Testing Methods
5.3.1 Functionality Testing
Verified classification accuracy across all 26 ISL letters and confirmed correct servo angles for each sign.
5.3.2 UX & Feedback
Evaluated touchscreen button responsiveness (Undo, Delete, Clear, Speak) and clear visual panel hierarchy.

CHAPTER 6. LIMITATIONS AND FUTURE ENHANCEMENT
6.1 Existing Limitations
- 1. Mechanical Wear of Scrap Materials: Cardboard and plastic joints require periodic recalibration under string tension.
- 2. Ambient Lighting Dependence: Extremely dark environments impact webcam landmark detection accuracy.
6.2 Future Improvements
- 1. 3D-Printed Robotic Chassis: Replacing scrap frames with 3D-printed biomimetic hands and N20 gear motors.
- 2. Sequence Classification for Whole Words: Integrating LSTM/Transformer models to recognize dynamic whole-word phrases.

CHAPTER 7. CONCLUSION
SignBridge successfully demonstrates a two-way (dual) robotic translation system for Indian Sign Language. By combining MediaPipe vision tracking, TensorFlow deep learning, and low-cost Arduino servo control, SignBridge provides an accessible solution bridging the communication gap.
CHAPTER 8: REFERENCES
- 1. IISc Bangalore & AI4Bharat INCLUDE Dataset Paper (2020).
- 2. Mendeley ISL Benchmark Dataset (2-Handed ISL Alphabets), Dataset 98mzk82wbb.
- 3. Google MediaPipe Framework Documentation (mediapipe.dev).
- 4. TensorFlow & Keras API Documentation (tensorflow.org).
- 5. Arduino Mega 2560 Hardware & Servo Library Reference (arduino.cc).
| Table No. | Title |
| --- | --- |
| Table 1.1 | Hardware and Software Specifications |
| Table 2.1 | Comparative Analysis of Existing Technical Solutions and Their Drawbacks |
| Table 2.2 | Literature Review Summary of Key Research Papers |
| Table 3.1 | Abbreviations and Acronyms Reference |
| Table 5.1 | Software Testing Matrix and Verification Objectives |
| Table 6.1 | Future Improvements and Extension Roadmap |
| Figure No. | Title |
| --- | --- |
| Figure 1.1 | SignBridge Dual-Communication Loop Architecture |
| Figure 3.1 | System Use Case Diagram |
| Figure 3.2 | System Activity Diagram (Input & Output Pipelines) |
| Figure 4.1 | MediaPipe 21 Hand Landmark Topology |
| Figure 4.2 | SignBridge Touchscreen Kiosk User Interface |
| Solution | Interactivity | Bidirectionality | 2-Hand Support | Drawbacks |
| --- | --- | --- | --- | --- |
| Printed Charts / Books | Low | None | Static Images | Passive; no real-time translation or feedback. |
| Flex Sensor Gloves | Medium | 1-Way (Sign -> Text) | Rarely (High Cost) | Intrusive wiring, fragile components, high maintenance cost. |
| ASL 1-Hand Robotic Models | Medium | 1-Way (Text -> Sign) | No (Single Hand) | Incompatible with two-handed ISL alphabets. |
| SignBridge Kiosk System | High | 2-Way Dual Loop | Yes (Dual 10-Servo) | Low-cost scrap materials require angle calibration. |
| Sr no. | Paper / Dataset Title | Summary & Findings | Tools, Technologies & Methodologies |
| --- | --- | --- | --- |
| 1 | INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition (IISc Bangalore & AI4Bharat) | Developed a comprehensive benchmark ISL dataset containing thousands of video clips across 263 ISL signs. Demonstrated that two-handed spatial keypoints significantly outperform raw RGB frames for ISL gesture classification. | Tools: MediaPipe Holistic, Python, PyTorch
Methodology: 3D Hand Landmark Extraction, Spatio-Temporal Keypoint Normalization. |
| 2 | Mendeley ISL Benchmark Dataset: 2-Handed ISL Manual Alphabets & Posture Features | Established a standardized 2-handed ISL alphabet dataset (A-Z) documenting left and right hand postures, finger extension ratios, and touch points for geometric heuristic and deep learning validation. | Tools: Mendeley Data, MediaPipe Hands, Python
Methodology: 42-point 3D keypoint mapping, posture feature extraction. |
| 3 | Google MediaPipe: On-Device Real-Time Hand Tracking Framework | Presents a real-time 21 3D hand landmark estimation pipeline running efficiently on low-power CPU/GPU devices without specialized depth sensors. | Tools: MediaPipe Hands, TensorFlow Lite
Methodology: BlazePalm detector + Hand Landmark regression network. |