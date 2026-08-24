"""
SignBridge Centralized Google Generative AI / Gemini API Management System.

Features:
  - Dual API Key Management (GEMINI_API_KEY_1, GEMINI_API_KEY_2, GOOGLE_API_KEY fallback)
  - Non-blocking Key Health Tracking & Cooldown Management
  - Dynamic Model Auto-Discovery with Task-Specific Capability Matching
  - Granular Error Classification (RATE_LIMIT, QUOTA_EXHAUSTED, AUTH_ERROR, MODEL_NOT_FOUND, SERVER_ERROR, etc.)
  - Controlled Exponential Backoff Retries & (Model x Key) Fallback Cascade
  - Zero Credential Leakage (Masked key identifiers in logs and public responses)
  - Dual SDK Compatibility (Modern google-genai and legacy google.generativeai)
"""

import os
import time
import logging
from enum import Enum
from typing import Optional, Any, Dict, List, Tuple, TypedDict

logger = logging.getLogger("SignBridge.GeminiManager")


class GeminiKeyStatus(Enum):
    AVAILABLE = "AVAILABLE"
    COOLDOWN = "COOLDOWN"
    RATE_LIMITED = "RATE_LIMITED"
    QUOTA_EXHAUSTED = "QUOTA_EXHAUSTED"
    INVALID = "INVALID"


class GeminiErrorType(Enum):
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


# Master priority lists for task types (sorted from latest/highest capability down to fast fallbacks)
TASK_PRIORITY_MODELS: Dict[str, List[str]] = {
    "text": [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.0-pro-exp-02-05",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
        "gemini-1.5-pro",
        "gemini-pro",
    ],
    "vision": [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    ],
    "structured": [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
    ],
}


def mask_key(key: Optional[str]) -> str:
    """Safely mask an API key for logs (e.g., 'AIza...4X9Z'). Never logs full key."""
    if not key:
        return "NONE"
    if len(key) <= 8:
        return "****"
    return f"{key[:4]}...{key[-4:]}"


class KeyInfo:
    """Runtime health and state tracker for an individual API key."""

    def __init__(self, key_id: str, api_key: str):
        self.key_id: str = key_id
        self.api_key: str = api_key.strip()
        self.masked: str = mask_key(self.api_key)
        self.status: GeminiKeyStatus = GeminiKeyStatus.AVAILABLE
        self.success_count: int = 0
        self.failure_count: int = 0
        self.cooldown_until: float = 0.0
        self.consecutive_failures: int = 0
        self.last_used: Optional[float] = None
        self.last_error_type: Optional[GeminiErrorType] = None
        self.last_error_message: Optional[str] = None
        self.client: Any = None
        self.sdk_mode: Optional[str] = None  # 'genai' | 'legacy'
        self.discovered_models: List[str] = []
        self.discovery_time: float = 0.0

        self._init_sdk_client()

    def _init_sdk_client(self):
        """Initialize modern or legacy SDK client for this key."""
        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
            self.sdk_mode = "genai"
        except Exception:
            try:
                import google.generativeai as legacy_genai
                # For legacy, configure is global so we store the module
                self.client = legacy_genai
                self.sdk_mode = "legacy"
            except Exception as e:
                logger.warning(f"[Gemini] Failed to load SDK for {self.key_id}: {e}")
                self.status = GeminiKeyStatus.INVALID

    def is_available(self, now: Optional[float] = None) -> bool:
        """Check if key is ready for requests."""
        if self.status == GeminiKeyStatus.INVALID or self.status == GeminiKeyStatus.QUOTA_EXHAUSTED:
            return False
        
        current_time = now or time.time()
        if self.status in (GeminiKeyStatus.COOLDOWN, GeminiKeyStatus.RATE_LIMITED):
            if current_time >= self.cooldown_until:
                self.status = GeminiKeyStatus.AVAILABLE
                self.consecutive_failures = 0
                return True
            return False
        
        return self.status == GeminiKeyStatus.AVAILABLE

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
            "sdk_mode": self.sdk_mode or "none",
            "last_error": self.last_error_type.value if self.last_error_type else None,
            "discovered_models_count": len(self.discovered_models),
        }


class KeyManager:
    """Manages collection of API keys, state tracking, and health-aware rotation."""

    def __init__(self):
        self.keys: List[KeyInfo] = []
        self._load_keys()

    def _load_keys(self):
        """Discover and load keys from environment variables."""
        self.keys = []
        k1 = os.environ.get("GEMINI_API_KEY_1") or os.environ.get("GOOGLE_API_KEY")
        k2 = os.environ.get("GEMINI_API_KEY_2")

        if k1 and k1.strip():
            self.keys.append(KeyInfo("KEY_1", k1))
        
        if k2 and k2.strip() and k2.strip() != (k1.strip() if k1 else ""):
            self.keys.append(KeyInfo("KEY_2", k2))
        
        # If legacy GOOGLE_API_KEY exists as a distinct 3rd key
        legacy_k = os.environ.get("GOOGLE_API_KEY")
        if legacy_k and legacy_k.strip() and all(k.api_key != legacy_k.strip() for k in self.keys):
            self.keys.append(KeyInfo("KEY_LEGACY", legacy_k))

        valid_count = sum(1 for k in self.keys if k.status != GeminiKeyStatus.INVALID)
        logger.info(f"[Gemini] KeyManager initialized with {len(self.keys)} key(s) ({valid_count} active).")

    def get_available_keys(self) -> List[KeyInfo]:
        """Return all currently usable keys, sorted by priority (KEY_1 -> KEY_2)."""
        now = time.time()
        return [k for k in self.keys if k.is_available(now)]

    def record_success(self, key_info: KeyInfo):
        """Record successful request for a key."""
        key_info.success_count += 1
        key_info.consecutive_failures = 0
        key_info.status = GeminiKeyStatus.AVAILABLE
        key_info.last_used = time.time()
        logger.debug(f"[Gemini] Request succeeded on {key_info.key_id} (Total Successes: {key_info.success_count})")

    def record_failure(self, key_info: KeyInfo, error_type: GeminiErrorType, error_msg: str):
        """Record a failure and apply appropriate cooldown/backoff."""
        key_info.failure_count += 1
        key_info.consecutive_failures += 1
        key_info.last_error_type = error_type
        key_info.last_error_message = error_msg
        now = time.time()

        if error_type == GeminiErrorType.RATE_LIMIT:
            # Exponential backoff cooldown: 15s -> 30s -> 60s -> max 120s
            cooldown_sec = min(120.0, 15.0 * (2 ** (key_info.consecutive_failures - 1)))
            key_info.status = GeminiKeyStatus.RATE_LIMITED
            key_info.cooldown_until = now + cooldown_sec
            logger.warning(f"[Gemini] {key_info.key_id} rate-limited. Cooldown for {cooldown_sec:.0f}s. Switching key.")

        elif error_type == GeminiErrorType.QUOTA_EXHAUSTED:
            # Longer cooldown for quota exhaustion (e.g. 15 minutes)
            key_info.status = GeminiKeyStatus.QUOTA_EXHAUSTED
            key_info.cooldown_until = now + 900.0
            logger.error(f"[Gemini] {key_info.key_id} quota exhausted. Disabled for 15 mins. Switching key.")

        elif error_type == GeminiErrorType.AUTHENTICATION_ERROR:
            key_info.status = GeminiKeyStatus.INVALID
            logger.error(f"[Gemini] {key_info.key_id} authentication failed (Invalid API Key). Disabled permanently.")

        elif error_type in (GeminiErrorType.SERVER_ERROR, GeminiErrorType.TIMEOUT):
            cooldown_sec = min(30.0, 5.0 * key_info.consecutive_failures)
            key_info.status = GeminiKeyStatus.COOLDOWN
            key_info.cooldown_until = now + cooldown_sec
            logger.warning(f"[Gemini] {key_info.key_id} transient error ({error_type.value}). Cooldown for {cooldown_sec:.0f}s.")

    def reset_health(self):
        """Reset all cooldowns and failures."""
        for k in self.keys:
            if k.status != GeminiKeyStatus.INVALID:
                k.status = GeminiKeyStatus.AVAILABLE
                k.consecutive_failures = 0
                k.cooldown_until = 0.0


class ErrorClassifier:
    """Classifies raw SDK and HTTP exceptions into actionable error enums."""

    @staticmethod
    def classify(exc: Exception) -> Tuple[GeminiErrorType, str]:
        err_str = str(exc)
        err_lower = err_str.lower()

        # Check HTTP status codes or message patterns
        if "429" in err_str or "resourceexhausted" in err_lower or "rate limit" in err_lower or "too many requests" in err_lower:
            return GeminiErrorType.RATE_LIMIT, err_str

        if "quota" in err_lower or "billing" in err_lower or "exceeded your current quota" in err_lower:
            return GeminiErrorType.QUOTA_EXHAUSTED, err_str

        if "401" in err_str or "403" in err_str or "api_key_invalid" in err_lower or "unauthenticated" in err_lower or "permission_denied" in err_lower or "api key not valid" in err_lower:
            return GeminiErrorType.AUTHENTICATION_ERROR, err_str

        if "404" in err_str or "not found" in err_lower or "is not found for api version" in err_lower or "unsupported model" in err_lower or "model not supported" in err_lower:
            return GeminiErrorType.MODEL_NOT_FOUND, err_str

        if "500" in err_str or "503" in err_str or "unavailable" in err_lower or "internalservererror" in err_lower or "server error" in err_lower:
            return GeminiErrorType.SERVER_ERROR, err_str

        if "timeout" in err_lower or "timed out" in err_lower or "deadline exceeded" in err_lower:
            return GeminiErrorType.TIMEOUT, err_str

        if "connection" in err_lower or "network" in err_lower or "failed to establish a new connection" in err_lower:
            return GeminiErrorType.NETWORK_ERROR, err_str

        if "400" in err_str or "invalidargument" in err_lower or "bad request" in err_lower:
            return GeminiErrorType.INVALID_REQUEST, err_str

        return GeminiErrorType.UNKNOWN_ERROR, err_str


class ModelSelector:
    """Task-aware model selector with auto-discovery and capability matching."""

    @staticmethod
    def get_candidate_models(key_info: KeyInfo, task_type: str = "text") -> List[str]:
        """
        Get ordered list of candidate models (latest -> fallback) suitable for task.
        If models have been auto-discovered for this key, prioritize active ones.
        """
        preferred = TASK_PRIORITY_MODELS.get(task_type, TASK_PRIORITY_MODELS["text"])

        now = time.time()
        # Refresh discovery every 1 hour (3600s)
        if not key_info.discovered_models or (now - key_info.discovery_time > 3600):
            ModelSelector.discover_models_for_key(key_info)

        if key_info.discovered_models:
            # Sort discovered models based on master preferred priority
            ordered = [m for m in preferred if m in key_info.discovered_models]
            for m in key_info.discovered_models:
                if m not in ordered and any(p in m for p in ["gemini-2", "gemini-1.5", "gemini"]):
                    ordered.append(m)
            return ordered or preferred

        return preferred

    @staticmethod
    def discover_models_for_key(key_info: KeyInfo):
        """Query the API to find which models this specific key has access to."""
        if not key_info.client:
            return

        discovered = []
        try:
            if key_info.sdk_mode == "genai" and hasattr(key_info.client, "models"):
                for m in key_info.client.models.list():
                    mid = getattr(m, "name", "") or getattr(m, "id", "")
                    mid = mid.replace("models/", "").strip()
                    if mid and "gemini" in mid and "embedding" not in mid and "aqa" not in mid:
                        discovered.append(mid)
            elif key_info.sdk_mode == "legacy" and hasattr(key_info.client, "list_models"):
                for m in key_info.client.list_models():
                    mid = getattr(m, "name", "").replace("models/", "").strip()
                    methods = getattr(m, "supported_generation_methods", [])
                    if "generateContent" in methods and "gemini" in mid:
                        discovered.append(mid)
        except Exception as e:
            logger.debug(f"[Gemini] Model discovery notice on {key_info.key_id}: {e}")

        key_info.discovered_models = discovered
        key_info.discovery_time = time.time()
        if discovered:
            logger.info(f"[Gemini] Discovered {len(discovered)} supported models on {key_info.key_id}.")


class GeminiResult(TypedDict):
    text: str
    provider: str
    model: str
    key_id: str
    fallback_used: bool
    retries: int


class GeminiManager:
    """
    Centralized, resilient Google Generative AI / Gemini API Manager.
    Orchestrates: Model Selection -> API Key Selection -> Backoff Retry -> Model Fallback -> Response.
    """

    def __init__(self):
        self.key_manager: KeyManager = KeyManager()
        self.max_retries: int = int(os.environ.get("GEMINI_MAX_RETRIES", "3"))
        self.max_model_fallbacks: int = int(os.environ.get("GEMINI_MAX_MODEL_FALLBACKS", "4"))
        self.timeout_sec: float = float(os.environ.get("GEMINI_TIMEOUT", "30.0"))

    def is_available(self) -> bool:
        """Check if at least one API key is configured and not permanently invalid."""
        return len(self.key_manager.get_available_keys()) > 0

    def generate(
        self,
        prompt: str,
        system_instruction: str = "You are SignBridge AI assistant.",
        task_type: str = "text",
        temperature: float = 0.3,
        max_output_tokens: int = 300
    ) -> GeminiResult:
        """
        Execute robust generation across (Model x Key) matrix.
        
        Algorithm:
          1. Select candidate models for task (latest -> fallback).
          2. For each model (up to MAX_MODEL_FALLBACKS):
               3. For each available key (KEY_1 -> KEY_2):
                    4. Execute request with controlled retries on transient errors.
                    5. On success: record metrics, return sanitized result.
                    6. On rate-limit/quota: mark key cooldown, switch to next key.
                    7. On model-not-found: break to next model.
          8. If all combinations exhausted, raise a clean runtime error.
        """
        available_keys = self.key_manager.get_available_keys()
        if not available_keys:
            # Check if all keys exist but are in cooldown
            all_keys = self.key_manager.keys
            if all_keys:
                min_wait = min((k.cooldown_until - time.time() for k in all_keys if k.cooldown_until > time.time()), default=5.0)
                logger.warning(f"[Gemini] All keys are currently in cooldown/quota. Shortest wait: {min_wait:.1f}s.")
            raise RuntimeError("Gemini service temporarily unavailable: No active API keys available.")

        # Get candidate models based on primary active key
        primary_key = available_keys[0]
        candidate_models = ModelSelector.get_candidate_models(primary_key, task_type=task_type)
        candidate_models = candidate_models[: self.max_model_fallbacks]

        total_retries = 0
        overall_errors: List[str] = []

        for model_idx, model_name in enumerate(candidate_models):
            # Refresh available keys in case cooldowns expired
            active_keys = self.key_manager.get_available_keys()
            if not active_keys:
                break

            for key_info in active_keys:
                logger.info(f"[Gemini] Attempting request | Model: {model_name} | Key: {key_info.key_id}")
                
                # Attempt request with exponential backoff on transient errors
                for attempt in range(1, self.max_retries + 1):
                    try:
                        text_output = self._call_sdk(
                            key_info=key_info,
                            model_name=model_name,
                            prompt=prompt,
                            system_instruction=system_instruction,
                            temperature=temperature,
                            max_output_tokens=max_output_tokens
                        )

                        if text_output and text_output.strip():
                            self.key_manager.record_success(key_info)
                            is_fallback = (model_idx > 0) or (key_info.key_id != "KEY_1")
                            return {
                                "text": text_output.strip(),
                                "provider": f"Google Gemini ({model_name})",
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

                        # Fast-fail invalid requests (bad parameters) without wasting retries
                        if error_type == GeminiErrorType.INVALID_REQUEST:
                            logger.error(f"[Gemini] Invalid request format: {err_msg}")
                            raise ValueError(f"Invalid Gemini request: {err_msg}")

                        # Model not found on this account -> skip to next model immediately
                        if error_type == GeminiErrorType.MODEL_NOT_FOUND:
                            logger.warning(f"[Gemini] Model '{model_name}' not available on {key_info.key_id}. Falling back to next model.")
                            break

                        # Record key failure & evaluate if we should switch keys
                        self.key_manager.record_failure(key_info, error_type, err_msg)

                        if error_type in (GeminiErrorType.RATE_LIMIT, GeminiErrorType.QUOTA_EXHAUSTED, GeminiErrorType.AUTHENTICATION_ERROR):
                            # Switch to next key immediately
                            break

                        # Transient server error or timeout -> retry on same key with exponential backoff
                        if error_type in (GeminiErrorType.SERVER_ERROR, GeminiErrorType.TIMEOUT, GeminiErrorType.NETWORK_ERROR):
                            if attempt < self.max_retries:
                                backoff = min(8.0, 1.0 * (2 ** (attempt - 1)))
                                logger.info(f"[Gemini] Transient error on {key_info.key_id}. Retrying in {backoff:.1f}s (Attempt {attempt}/{self.max_retries})...")
                                time.sleep(backoff)
                            else:
                                logger.warning(f"[Gemini] Exhausted {self.max_retries} attempts on {key_info.key_id}. Switching key/model.")
                                break

        # If we reached here, all (Model x Key) combinations failed
        error_details = "; ".join(overall_errors[-4:]) if overall_errors else "All keys/models exhausted"
        logger.error(f"[Gemini] All Gemini models and keys exhausted. Technical summary: {error_details}")
        raise RuntimeError(f"Gemini service temporarily unavailable. (Attempts: {total_retries})")

    def _call_sdk(
        self,
        key_info: KeyInfo,
        model_name: str,
        prompt: str,
        system_instruction: str,
        temperature: float,
        max_output_tokens: int
    ) -> str:
        """Low-level SDK caller handling both modern genai and legacy SDKs."""
        client = key_info.client
        if not client:
            raise RuntimeError(f"SDK client not initialized for {key_info.key_id}")

        full_prompt = f"{system_instruction}\n\nUser Input: {prompt}" if system_instruction else prompt

        if key_info.sdk_mode == "genai" and hasattr(client, "models"):
            # Modern google-genai SDK
            response = client.models.generate_content(
                model=model_name,
                contents=full_prompt,
            )
            raw_text = getattr(response, "text", None)
            if raw_text is not None:
                return str(raw_text)

        elif key_info.sdk_mode == "legacy" and hasattr(client, "GenerativeModel"):
            # Legacy google.generativeai SDK
            # Reconfigure legacy module globally with current key
            client.configure(api_key=key_info.api_key)
            model_obj = client.GenerativeModel(model_name)
            response = model_obj.generate_content(full_prompt)
            raw_text = getattr(response, "text", None)
            if raw_text is not None:
                return str(raw_text)

        raise RuntimeError(f"Unrecognized SDK mode or empty response from {model_name}")

    def get_health_status(self) -> Dict[str, Any]:
        """Comprehensive health status for telemetry and monitoring."""
        return {
            "service": "Google Gemini Manager",
            "is_available": self.is_available(),
            "total_keys": len(self.key_manager.keys),
            "keys": [k.to_health_dict() for k in self.key_manager.keys],
            "max_retries": self.max_retries,
            "max_model_fallbacks": self.max_model_fallbacks,
            "priority_models_text": TASK_PRIORITY_MODELS["text"][:5],
        }


# Global singleton instance
gemini_manager = GeminiManager()
