"""
Sign-Bridge Flask API — Arduino Serial Communication Service

Manages PySerial connection to Arduino Mega for driving
10x SG90 micro servos (5 per hand) to physically form ISL signs.

Each ISL letter maps to a 10-element servo-angle array:
  [L_thumb, L_index, L_middle, L_ring, L_pinky,
   R_thumb, R_index, R_middle, R_ring, R_pinky]

Angle 0 = finger fully extended (open)
Angle 180 = finger fully curled (closed fist)
"""

import serial
import serial.tools.list_ports
import time
import json
import logging

logger = logging.getLogger(__name__)

# ─── ISL Alphabet → Servo Angle Lookup Table ───────────────────────────────
# Each entry: [L_thumb, L_index, L_middle, L_ring, L_pinky,
#              R_thumb, R_index, R_middle, R_ring, R_pinky]
# These are initial reference angles and MUST be fine-tuned
# during Phase 7 calibration with the physical scrap-material hands.

ISL_SERVO_MAP = {
    'A': [  0,  180, 180, 180, 180,   180,   0, 180, 180, 180],  # LH thumb up, RH index touches
    'B': [ 60,   60, 180, 180, 180,    60,  60, 180, 180, 180],  # Both open loops
    'C': [ 90,   90, 120, 150, 180,     0,   0,   0,   0,   0],  # LH C-arc, RH base palm
    'D': [180,    0, 180, 180, 180,    90,  90, 180, 180, 180],  # LH index up, RH semicircle
    'E': [180,    0, 180, 180, 180,   180,   0, 180, 180, 180],  # LH index finger, RH touch index
    'F': [180,   90,  90, 180, 180,   180,  90,  90, 180, 180],  # Both cross fingers
    'G': [180,  180, 180, 180, 180,   180, 180, 180, 180, 180],  # Both fists out
    'H': [  0,    0,   0,   0,   0,     0,   0,   0,   0,   0],  # LH flat palm, RH sweep palm
    'I': [180,  180,   0, 180, 180,   180,   0, 180, 180, 180],  # LH middle finger, RH touch index
    'J': [  0,    0,   0,   0,   0,   180,   0, 180, 180, 180],  # LH flat palm, RH trace J hook
    'K': [180,    0, 180, 180, 180,   180,  90, 180, 180, 180],  # LH index up, RH hook index
    'L': [  0,    0, 180, 180, 180,     0,   0,   0,   0,   0],  # LH L-shape, RH base palm
    'M': [  0,    0,   0,   0,   0,   180,   0,   0,   0, 180],  # LH flat palm, RH 3-fingers
    'N': [  0,    0,   0,   0,   0,   180,   0,   0, 180, 180],  # LH flat palm, RH 2-fingers
    'O': [180,  180, 180,   0, 180,   180,   0, 180, 180, 180],  # LH ring finger, RH touch index
    'P': [180,    0, 180, 180, 180,    90,  90, 180, 180, 180],  # LH index up, RH circle loop
    'Q': [ 90,   90, 180, 180, 180,    90,  90, 180, 180, 180],  # Both loop halves
    'R': [  0,    0,   0,   0,   0,   180,  90, 180, 180, 180],  # LH flat palm, RH curled index
    'S': [180,  180, 180, 180,   0,   180, 180, 180, 180,   0],  # Both pinky interlock
    'T': [  0,    0,   0,   0,   0,   180,   0, 180, 180, 180],  # LH vertical palm, RH edge touch
    'U': [180,  180, 180, 180,   0,   180,   0, 180, 180, 180],  # LH pinky finger, RH touch index
    'V': [  0,    0,   0,   0,   0,   180,   0,   0, 180, 180],  # LH flat palm, RH V-shape taps
    'W': [180,    0,   0,   0, 180,   180,   0,   0,   0, 180],  # Both W-peaks
    'X': [180,   90, 180, 180, 180,   180,  90, 180, 180, 180],  # Both cross index
    'Y': [  0,    0,   0,   0,   0,     0,  180, 180, 180,  0],  # LH flat palm, RH Y-shape taps
    'Z': [  0,    0,   0,   0,   0,   180,   0, 180, 180, 180],  # LH flat palm, RH draw Z air
}

# A neutral rest position (all fingers half-open)
REST_POSE = [90, 90, 90, 90, 90, 90, 90, 90, 90, 90]


class ArduinoSerial:
    """
    Manages connection to Arduino Mega via USB Serial.
    Sends servo angle commands as JSON arrays.
    """

    def __init__(self, port=None, baud_rate=9600, timeout=2):
        self.port = port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.connection = None
        self.is_connected = False

    def auto_detect_port(self):
        """Scan COM ports for an Arduino device."""
        ports = serial.tools.list_ports.comports()
        for p in ports:
            desc = (p.description or '').lower()
            mfr = (p.manufacturer or '').lower()
            if 'arduino' in desc or 'arduino' in mfr or 'ch340' in desc or 'cp210' in desc:
                logger.info(f"Auto-detected Arduino on {p.device}: {p.description}")
                return p.device
        logger.warning("No Arduino detected on any COM port.")
        return None

    def connect(self):
        """Open serial connection to Arduino."""
        if self.is_connected and self.connection:
            return True

        target_port = self.port or self.auto_detect_port()
        if not target_port:
            logger.error("Cannot connect: no Arduino port found or specified.")
            self.is_connected = False
            return False

        try:
            self.connection = serial.Serial(
                port=target_port,
                baudrate=self.baud_rate,
                timeout=self.timeout
            )
            # Wait for Arduino to reset after serial open
            time.sleep(2)
            self.is_connected = True
            self.port = target_port
            logger.info(f"Connected to Arduino on {target_port} at {self.baud_rate} baud.")
            return True
        except serial.SerialException as e:
            logger.error(f"Serial connection failed: {e}")
            self.is_connected = False
            return False

    def disconnect(self):
        """Close the serial connection."""
        if self.connection and self.connection.is_open:
            self.connection.close()
        self.is_connected = False
        logger.info("Arduino disconnected.")

    def send_angles(self, angles):
        """
        Send a 10-element servo angle array to the Arduino.
        Format sent: JSON string e.g. '[0,180,180,180,180,180,0,180,180,180]\n'
        """
        if not self.is_connected or not self.connection:
            logger.warning("Cannot send: Arduino not connected.")
            return False

        try:
            payload = json.dumps(angles) + '\n'
            self.connection.write(payload.encode('utf-8'))
            self.connection.flush()
            logger.debug(f"Sent angles: {angles}")
            return True
        except serial.SerialException as e:
            logger.error(f"Failed to send angles: {e}")
            self.is_connected = False
            return False

    def sign_letter(self, letter, hold_time=1.5):
        """
        Look up the ISL servo angles for a single letter and send them.
        Holds the pose for `hold_time` seconds before returning.
        """
        letter = letter.upper()
        angles = ISL_SERVO_MAP.get(letter, REST_POSE)
        success = self.send_angles(angles)
        if success:
            time.sleep(hold_time)
        return success

    def sign_text(self, text, letter_hold=1.5, gap=0.5):
        """
        Fingerspell an entire text string letter-by-letter.
        Skips spaces with a short pause.
        Returns list of letters successfully signed.
        """
        signed = []
        for char in text.upper():
            if char == ' ':
                # Rest pose between words
                self.send_angles(REST_POSE)
                time.sleep(gap * 2)
                continue
            if char in ISL_SERVO_MAP:
                self.sign_letter(char, hold_time=letter_hold)
                signed.append(char)
                # Brief gap between consecutive letters
                self.send_angles(REST_POSE)
                time.sleep(gap)
        return signed

    def get_status(self):
        """Return current connection status."""
        return {
            'connected': self.is_connected,
            'port': self.port,
            'baud_rate': self.baud_rate
        }
