"""
SignBridge Centralized Groq LPU API Management System.

Features:
  - Dual API Key Management (GROQ_API_KEY_1, GROQ_API_KEY_2, GROQ_API_KEY fallback)
  - Non-blocking Key Health Tracking & Cooldown Management
  - Dynamic Model Auto-Discovery via client.models.list()
  - Granular Error Classification (RATE_LIMIT, QUOTA_EXHAUSTED, AUTH_ERROR, MODEL_NOT_FOUND, SERVER_ERROR, etc.)
  - Controlled Exponential Backoff Retries & (Model x Key) Fallback Cascade
  - Zero Credential Leakage (Masked key identifiers in logs and public responses)
"""

import os
import time
import logging
from enum import Enum
from typing import Optional, Any, Dict, List, Tuple, TypedDict

logger = logging.getLogger("SignBridge.GroqManager")


class GroqKeyStatus(Enum):
    AVAILABLE = "AVAILABLE"
    COOLDOWN = "COOLDOWN"
    RATE_LIMITED = "RATE_LIMITED"
    QUOTA_EXHAUSTED = "QUOTA_EXHAUSTED"
    INVALID = "INVALID"


class GroqErrorType(Enum):
    RATE_LIMIT = "RATE_LIMIT"
    QUOTA_EXHAUSTED = "QUOTA_EXHAUSTED"
    AUTHENTICATION_ERROR = "AUTHENTICATION_ERROR"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    PERMISSION_ERROR = "PERMISSION_ERROR"
    SERVER_ERROR = "SERVER_ERROR"
    TIMEOUT = "TIMEOUT"
    NETWORK_ERROR = "NETWORK_ERROR"
    INVALID_REQUEST = "INVALID_REQUEST"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


# Master priority candidate lists (sorted from latest/highest capability down to fast fallbacks)
PREFERRED_GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.3-70b-specdec",
    "llama-3.1-70b-versatile",
    "openai/gpt-oss-120b",
    "qwen/qwen3.6-27b",
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
    "mixtral-8x7b-32768",
    "groq/compound",
    "groq/compound-mini",
]


def mask_key(key: Optional[str]) -> str:
    """Safely mask an API key for logs (e.g., 'gsk_...4X9Z'). Never logs full key."""
    if not key:
        return "NONE"
    if len(key) <= 8:
        return "****"
    return f"{key[:4]}...{key[-4:]}"


class KeyInfo:
    """Runtime health and state tracker for an individual Groq API key."""

    def __init__(self, key_id: str, api_key: str):
        self.key_id: str = key_id
        self.api_key: str = api_key.strip()
        self.masked: str = mask_key(self.api_key)
        self.status: GroqKeyStatus = GroqKeyStatus.AVAILABLE
        self.success_count: int = 0
        self.failure_count: int = 0
        self.cooldown_until: float = 0.0
        self.consecutive_failures: int = 0
        self.last_used: Optional[float] = None
        self.last_error_type: Optional[GroqErrorType] = None
        self.last_error_message: Optional[str] = None
        self.client: Any = None
        self.discovered_models: List[str] = []
        self.discovery_time: float = 0.0

        self._init_sdk_client()

    def _init_sdk_client(self):
        """Initialize Groq client for this key."""
        try:
            from groq import Groq
            self.client = Groq(api_key=self.api_key)
        except Exception as e:
            logger.warning(f"[Groq] Failed to initialize Groq SDK for {self.key_id}: {e}")
            self.status = GroqKeyStatus.INVALID

    def is_available(self, now: Optional[float] = None) -> bool:
        """Check if key is ready for requests."""
        if self.status == GroqKeyStatus.INVALID or self.status == GroqKeyStatus.QUOTA_EXHAUSTED:
            return False

        current_time = now or time.time()
        if self.status in (GroqKeyStatus.COOLDOWN, GroqKeyStatus.RATE_LIMITED):
            if current_time >= self.cooldown_until:
                self.status = GroqKeyStatus.AVAILABLE
                self.consecutive_failures = 0
                return True
            return False

        return self.status == GroqKeyStatus.AVAILABLE

    def to_health_dict(self) -> Dict[str, Any]:
        """Sanitized dictionary for status reporting (no plain-text keys)."""
        now = time.time()
        remaining_cooldown = max(0.0, round(self.cooldown_until - now, 1)) if self.cooldown_until > now else 0.0
        return {
            "key_id": self.key_id,
            "masked_key": self.masked,
            "status": self.status.value,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "consecutive_failures": self.consecutive_failures,
            "cooldown_remaining_sec": remaining_cooldown,
            "last_error": self.last_error_type.value if self.last_error_type else None,
            "discovered_models_count": len(self.discovered_models),
        }


class KeyManager:
    """Manages collection of Groq API keys, state tracking, and health-aware rotation."""

    def __init__(self):
        self.keys: List[KeyInfo] = []
        self._load_keys()

    def _load_keys(self):
        """Discover and load keys from environment variables."""
        self.keys = []
        k1 = os.environ.get("GROQ_API_KEY_1") or os.environ.get("GROQ_API_KEY")
        k2 = os.environ.get("GROQ_API_KEY_2")

        if k1 and k1.strip():
            self.keys.append(KeyInfo("KEY_1", k1))

        if k2 and k2.strip() and k2.strip() != (k1.strip() if k1 else ""):
            self.keys.append(KeyInfo("KEY_2", k2))

        # Legacy single key fallback if different
        legacy_k = os.environ.get("GROQ_API_KEY")
        if legacy_k and legacy_k.strip() and all(k.api_key != legacy_k.strip() for k in self.keys):
            self.keys.append(KeyInfo("KEY_LEGACY", legacy_k))

        valid_count = sum(1 for k in self.keys if k.status != GroqKeyStatus.INVALID)
        logger.info(f"[Groq] KeyManager initialized with {len(self.keys)} key(s) ({valid_count} active).")

    def get_available_keys(self) -> List[KeyInfo]:
        """Return all currently usable keys, sorted by priority (KEY_1 -> KEY_2)."""
        now = time.time()
        return [k for k in self.keys if k.is_available(now)]

    def record_success(self, key_info: KeyInfo):
        """Record successful request for a key."""
        key_info.success_count += 1
        key_info.consecutive_failures = 0
        key_info.status = GroqKeyStatus.AVAILABLE
        key_info.last_used = time.time()
        logger.debug(f"[Groq] Request succeeded on {key_info.key_id} (Total Successes: {key_info.success_count})")

    def record_failure(self, key_info: KeyInfo, error_type: GroqErrorType, error_msg: str):
        """Record a failure and apply appropriate cooldown/backoff."""
        key_info.failure_count += 1
        key_info.consecutive_failures += 1
        key_info.last_error_type = error_type
        key_info.last_error_message = error_msg
        now = time.time()

        if error_type == GroqErrorType.RATE_LIMIT:
            # Exponential backoff cooldown: 10s -> 20s -> 40s -> max 120s
            cooldown_sec = min(120.0, 10.0 * (2 ** (key_info.consecutive_failures - 1)))
            key_info.status = GroqKeyStatus.RATE_LIMITED
            key_info.cooldown_until = now + cooldown_sec
            logger.warning(f"[Groq] {key_info.key_id} rate-limited (HTTP 429). Cooldown for {cooldown_sec:.0f}s. Switching key.")

        elif error_type == GroqErrorType.QUOTA_EXHAUSTED:
            key_info.status = GroqKeyStatus.QUOTA_EXHAUSTED
            key_info.cooldown_until = now + 900.0
            logger.error(f"[Groq] {key_info.key_id} quota exhausted. Disabled for 15 mins. Switching key.")

        elif error_type == GroqErrorType.AUTHENTICATION_ERROR:
            key_info.status = GroqKeyStatus.INVALID
            logger.error(f"[Groq] {key_info.key_id} authentication failed (Invalid API Key). Disabled permanently.")

        elif error_type in (GroqErrorType.SERVER_ERROR, GroqErrorType.TIMEOUT):
            cooldown_sec = min(30.0, 5.0 * key_info.consecutive_failures)
            key_info.status = GroqKeyStatus.COOLDOWN
            key_info.cooldown_until = now + cooldown_sec
            logger.warning(f"[Groq] {key_info.key_id} transient error ({error_type.value}). Cooldown for {cooldown_sec:.0f}s.")

    def reset_health(self):
        """Reset all cooldowns and failures."""
        for k in self.keys:
            if k.status != GroqKeyStatus.INVALID:
                k.status = GroqKeyStatus.AVAILABLE
                k.consecutive_failures = 0
                k.cooldown_until = 0.0


class ErrorClassifier:
    """Classifies raw Groq SDK and HTTP exceptions into actionable error enums."""

    @staticmethod
    def classify(exc: Exception) -> Tuple[GroqErrorType, str]:
        err_str = str(exc)
        err_lower = err_str.lower()

        if "429" in err_str or "rate_limit" in err_lower or "rate limit" in err_lower or "too many requests" in err_lower:
            return GroqErrorType.RATE_LIMIT, err_str

        if "quota" in err_lower or "insufficient_quota" in err_lower or "billing" in err_lower or "exceeded your current quota" in err_lower:
            return GroqErrorType.QUOTA_EXHAUSTED, err_str

        if "401" in err_str or "403" in err_str or "invalid_api_key" in err_lower or "unauthenticated" in err_lower or "authentication" in err_lower or "invalid api key" in err_lower:
            return GroqErrorType.AUTHENTICATION_ERROR, err_str

        if "404" in err_str or "model_not_found" in err_lower or "model not found" in err_lower or "does not exist" in err_lower or "decommissioned" in err_lower:
            return GroqErrorType.MODEL_NOT_FOUND, err_str

        if "500" in err_str or "503" in err_str or "unavailable" in err_lower or "internalservererror" in err_lower or "server error" in err_lower:
            return GroqErrorType.SERVER_ERROR, err_str

        if "timeout" in err_lower or "timed out" in err_lower or "deadline exceeded" in err_lower:
            return GroqErrorType.TIMEOUT, err_str

        if "connection" in err_lower or "network" in err_lower or "failed to establish a new connection" in err_lower:
            return GroqErrorType.NETWORK_ERROR, err_str

        if "400" in err_str or "invalid_request_error" in err_lower or "bad request" in err_lower:
            return GroqErrorType.INVALID_REQUEST, err_str

        return GroqErrorType.UNKNOWN_ERROR, err_str


class ModelSelector:
    """Task-aware Groq model selector with auto-discovery and capability sorting."""

    @staticmethod
    def get_candidate_models(key_info: KeyInfo) -> List[str]:
        """
        Get ordered list of candidate models (latest -> fallback).
        Prioritizes discovered models on this key.
        """
        now = time.time()
        if not key_info.discovered_models or (now - key_info.discovery_time > 3600):
            ModelSelector.discover_models_for_key(key_info)

        if key_info.discovered_models:
            ordered = [m for m in PREFERRED_GROQ_MODELS if m in key_info.discovered_models]
            for m in key_info.discovered_models:
                if m not in ordered:
                    ordered.append(m)
            return ordered or PREFERRED_GROQ_MODELS

        return PREFERRED_GROQ_MODELS

    @staticmethod
    def discover_models_for_key(key_info: KeyInfo):
        """Query Groq API to discover active models for this key."""
        if not key_info.client:
            return

        discovered = []
        try:
            raw_list = key_info.client.models.list().data
            for m in raw_list:
                mid = getattr(m, "id", "")
                if mid and "whisper" not in mid and "guard" not in mid:
                    discovered.append(mid)
        except Exception as e:
            logger.debug(f"[Groq] Model discovery notice on {key_info.key_id}: {e}")

        key_info.discovered_models = discovered
        key_info.discovery_time = time.time()
        if discovered:
            logger.info(f"[Groq] Discovered {len(discovered)} supported models on {key_info.key_id}.")


class GroqResult(TypedDict):
    text: str
    provider: str
    model: str
    key_id: str
    fallback_used: bool
    retries: int


class GroqManager:
    """
    Centralized, resilient Groq LPU API Manager.
    Orchestrates: Model Selection -> API Key Selection -> Backoff Retry -> Model Fallback -> Response.
    """

    def __init__(self):
        self.key_manager: KeyManager = KeyManager()
        self.max_retries: int = int(os.environ.get("GROQ_MAX_RETRIES", "3"))
        self.max_model_fallbacks: int = int(os.environ.get("GROQ_MAX_MODEL_FALLBACKS", "4"))
        self.timeout_sec: float = float(os.environ.get("GROQ_TIMEOUT", "20.0"))

    def is_available(self) -> bool:
        """Check if at least one Groq API key is configured and not permanently invalid."""
        return len(self.key_manager.get_available_keys()) > 0

    def generate(
        self,
        prompt: str,
        system_instruction: str = "You are SignBridge AI assistant.",
        temperature: float = 0.3,
        max_tokens: int = 300
    ) -> GroqResult:
        """
        Execute robust generation across Groq (Model x Key) matrix.
        """
        available_keys = self.key_manager.get_available_keys()
        if not available_keys:
            all_keys = self.key_manager.keys
            if all_keys:
                min_wait = min((k.cooldown_until - time.time() for k in all_keys if k.cooldown_until > time.time()), default=5.0)
                logger.warning(f"[Groq] All keys are currently in cooldown/quota. Shortest wait: {min_wait:.1f}s.")
            raise RuntimeError("Groq service temporarily unavailable: No active API keys available.")

        primary_key = available_keys[0]
        candidate_models = ModelSelector.get_candidate_models(primary_key)[: self.max_model_fallbacks]

        total_retries = 0
        overall_errors: List[str] = []

        for model_idx, model_name in enumerate(candidate_models):
            active_keys = self.key_manager.get_available_keys()
            if not active_keys:
                break

            for key_info in active_keys:
                logger.info(f"[Groq] Attempting request | Model: {model_name} | Key: {key_info.key_id}")

                for attempt in range(1, self.max_retries + 1):
                    try:
                        text_output = self._call_sdk(
                            key_info=key_info,
                            model_name=model_name,
                            prompt=prompt,
                            system_instruction=system_instruction,
                            temperature=temperature,
                            max_tokens=max_tokens
                        )

                        if text_output and text_output.strip():
                            self.key_manager.record_success(key_info)
                            is_fallback = (model_idx > 0) or (key_info.key_id != "KEY_1")
                            return {
                                "text": text_output.strip(),
                                "provider": f"Groq ({model_name})",
                                "model": model_name,
                                "key_id": key_info.key_id,
                                "fallback_used": is_fallback,
                                "retries": total_retries
                            }

                    except Exception as e:
                        error_type, err_msg = ErrorClassifier.classify(e)
                        total_retries += 1
                        err_summary = f"[{model_name} / {key_info.key_id}] {error_type.value}: {str(e)[:120]}"
                        overall_errors.append(err_summary)

                        if error_type == GroqErrorType.INVALID_REQUEST:
                            logger.error(f"[Groq] Invalid request format: {err_msg}")
                            raise ValueError(f"Invalid Groq request: {err_msg}")

                        if error_type == GroqErrorType.MODEL_NOT_FOUND:
                            logger.warning(f"[Groq] Model '{model_name}' not available on {key_info.key_id}. Falling back to next model.")
                            break

                        self.key_manager.record_failure(key_info, error_type, err_msg)

                        if error_type in (GroqErrorType.RATE_LIMIT, GroqErrorType.QUOTA_EXHAUSTED, GroqErrorType.AUTHENTICATION_ERROR):
                            break

                        if error_type in (GroqErrorType.SERVER_ERROR, GroqErrorType.TIMEOUT, GroqErrorType.NETWORK_ERROR):
                            if attempt < self.max_retries:
                                backoff = min(6.0, 1.0 * (2 ** (attempt - 1)))
                                logger.info(f"[Groq] Transient error on {key_info.key_id}. Retrying in {backoff:.1f}s (Attempt {attempt}/{self.max_retries})...")
                                time.sleep(backoff)
                            else:
                                logger.warning(f"[Groq] Exhausted {self.max_retries} attempts on {key_info.key_id}. Switching key/model.")
                                break

        error_details = "; ".join(overall_errors[-4:]) if overall_errors else "All keys/models exhausted"
        logger.error(f"[Groq] All Groq models and keys exhausted. Technical summary: {error_details}")
        raise RuntimeError(f"Groq service temporarily unavailable. (Attempts: {total_retries})")

    def _call_sdk(
        self,
        key_info: KeyInfo,
        model_name: str,
        prompt: str,
        system_instruction: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """Low-level Groq completions caller."""
        client = key_info.client
        if not client:
            raise RuntimeError(f"Groq client not initialized for {key_info.key_id}")

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=self.timeout_sec
        )

        if response.choices and len(response.choices) > 0:
            msg = response.choices[0].message
            content = getattr(msg, "content", None)
            if content:
                return str(content)

        raise RuntimeError(f"Empty response choice from Groq ({model_name})")

    def get_health_status(self) -> Dict[str, Any]:
        """Comprehensive health status for telemetry and monitoring."""
        return {
            "service": "Groq LPU Manager",
            "is_available": self.is_available(),
            "total_keys": len(self.key_manager.keys),
            "keys": [k.to_health_dict() for k in self.key_manager.keys],
            "max_retries": self.max_retries,
            "max_model_fallbacks": self.max_model_fallbacks,
            "priority_models": PREFERRED_GROQ_MODELS[:5],
        }


# Global singleton instance
groq_manager = GroqManager()
