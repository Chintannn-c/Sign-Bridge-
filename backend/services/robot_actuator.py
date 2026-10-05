"""
SignBridge — Robotics Actuation Interface & ROS2 Stub
Provides a standardized base class for robotic arm/hand actuation,
supporting PySerial (Arduino), WebSockets, and ROS2 topic publishing.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class BaseRobotActuator(ABC):
    """Abstract base actuator for dual robotic hands / arms."""

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to actuator hardware or middleware."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Cleanly close actuator connection."""
        pass

    @abstractmethod
    def send_angles(self, angles: List[int]) -> bool:
        """Send 10-element servo angle array [0..180]."""
        pass

    @abstractmethod
    def sign_letter(self, letter: str, hold_seconds: float = 1.0) -> bool:
        """Actuate robotic hands to form a specific ISL letter."""
        pass


class ROS2ActuatorStub(BaseRobotActuator):
    """
    ROS2 Middleware Adapter Stub.
    Publishes joint angles to `/signbridge/joint_trajectory` or `/signbridge/servo_angles`.
    Activate by initializing with an active ROS2 node or rclpy context.
    """

    def __init__(self, topic_name: str = "/signbridge/servo_angles"):
        self.topic_name = topic_name
        self.is_connected = False
        self._publisher = None

    def connect(self) -> bool:
        try:
            # ponytail: runtime import avoids hard rclpy dependency when running standard USB serial
            import rclpy
            from std_msgs.msg import Int32MultiArray
            logger.info(f"ROS2 stub connected to topic: {self.topic_name}")
            self.is_connected = True
            return True
        except ImportError:
            logger.info("ROS2 (rclpy) not installed in local environment; running in stub simulation mode.")
            self.is_connected = True
            return True

    def disconnect(self) -> None:
        self.is_connected = False

    def send_angles(self, angles: List[int]) -> bool:
        if len(angles) != 10:
            return False
        logger.debug(f"[ROS2 Stub] Published angles: {angles} to {self.topic_name}")
        return True

    def sign_letter(self, letter: str, hold_seconds: float = 1.0) -> bool:
        logger.info(f"[ROS2 Stub] Signing letter '{letter}' (hold: {hold_seconds}s)")
        return True
