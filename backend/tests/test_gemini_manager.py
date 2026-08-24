"""
Unit and integration tests for SignBridge GeminiManager.
Tests all 8 required failure, fallback, and rotation scenarios:
  1. Test 1: Key 1 works -> Success
  2. Test 2: Key 1 rate-limited -> Key 2 works
  3. Test 3: Both keys fail on Model A -> Model B succeeds
  4. Test 4: Key 1 invalid -> Key 2 succeeds
  5. Test 5: All keys fail -> Graceful runtime error
  6. Test 6: Temporary server error -> Retry with backoff -> Success
  7. Test 7: Invalid request -> Fast fail
  8. Test 8: Latest model unavailable -> Supported fallback model selected
"""

import os
import unittest
from unittest.mock import patch, MagicMock

# Set mock env before importing
os.environ["GEMINI_API_KEY_1"] = "AIzaSyFakeKey1_MockedTesting1234"
os.environ["GEMINI_API_KEY_2"] = "AIzaSyFakeKey2_MockedTesting5678"
os.environ["GEMINI_MAX_RETRIES"] = "2"
os.environ["GEMINI_MAX_MODEL_FALLBACKS"] = "3"

from services.gemini_manager import (
    GeminiManager,
    KeyManager,
    KeyInfo,
    GeminiKeyStatus,
    GeminiErrorType,
    ErrorClassifier,
    ModelSelector,
    TASK_PRIORITY_MODELS,
    mask_key
)


class TestGeminiManager(unittest.TestCase):

    def setUp(self):
        self.manager = GeminiManager()
        # Reset key health
        self.manager.key_manager.reset_health()

    def test_mask_key(self):
        """Verify API keys are never logged in plain text."""
        self.assertEqual(mask_key("AIzaSy1234567890abcdef"), "AIza...cdef")
        self.assertEqual(mask_key(""), "NONE")
        self.assertEqual(mask_key(None), "NONE")
        self.assertEqual(mask_key("123"), "****")

    def test_error_classifier(self):
        """Verify proper classification of SDK exceptions."""
        e_rate = Exception("429 ResourceExhausted: rate limit exceeded")
        self.assertEqual(ErrorClassifier.classify(e_rate)[0], GeminiErrorType.RATE_LIMIT)

        e_quota = Exception("Quota exceeded for quota metric 'GenerateContent'")
        self.assertEqual(ErrorClassifier.classify(e_quota)[0], GeminiErrorType.QUOTA_EXHAUSTED)

        e_auth = Exception("403 API_KEY_INVALID: API key not valid")
        self.assertEqual(ErrorClassifier.classify(e_auth)[0], GeminiErrorType.AUTHENTICATION_ERROR)

        e_not_found = Exception("404 models/gemini-future-99 is not found")
        self.assertEqual(ErrorClassifier.classify(e_not_found)[0], GeminiErrorType.MODEL_NOT_FOUND)

        e_500 = Exception("500 InternalServerError: backend error")
        self.assertEqual(ErrorClassifier.classify(e_500)[0], GeminiErrorType.SERVER_ERROR)

        e_bad_req = Exception("400 InvalidArgument: prompt too long")
        self.assertEqual(ErrorClassifier.classify(e_bad_req)[0], GeminiErrorType.INVALID_REQUEST)

    def test_scenario_1_key1_works(self):
        """Scenario 1: Key 1 works normally -> Returns response immediately."""
        with patch.object(self.manager, "_call_sdk", return_value="Hello, SignBridge!") as mock_call:
            res = self.manager.generate("Translate HELLO ME")
            self.assertEqual(res["text"], "Hello, SignBridge!")
            self.assertEqual(res["key_id"], "KEY_1")
            self.assertFalse(res["fallback_used"])
            self.assertEqual(mock_call.call_count, 1)

    def test_scenario_2_key1_rate_limited_switch_to_key2(self):
        """Scenario 2: Key 1 hits 429 rate limit -> Automatically switches to Key 2."""
        def side_effect(key_info, model_name, **kwargs):
            if key_info.key_id == "KEY_1":
                raise Exception("429 ResourceExhausted: rate limit exceeded")
            return "Response from Key 2"

        with patch.object(self.manager, "_call_sdk", side_effect=side_effect):
            res = self.manager.generate("Test prompt")
            self.assertEqual(res["text"], "Response from Key 2")
            self.assertEqual(res["key_id"], "KEY_2")
            self.assertTrue(res["fallback_used"])

            # Verify Key 1 is marked as RATE_LIMITED with cooldown
            k1 = next(k for k in self.manager.key_manager.keys if k.key_id == "KEY_1")
            self.assertEqual(k1.status, GeminiKeyStatus.RATE_LIMITED)
            self.assertGreater(k1.cooldown_until, 0)

    def test_scenario_3_both_keys_fail_model_a_fallback_to_model_b(self):
        """Scenario 3: Both keys fail on Model A -> Model B succeeds on Key 1."""
        def side_effect(key_info, model_name, **kwargs):
            if model_name == TASK_PRIORITY_MODELS["text"][0]:
                raise Exception("404 models/gemini-2.5-flash is not found for api version")
            return f"Success on {model_name} via {key_info.key_id}"

        with patch.object(self.manager, "_call_sdk", side_effect=side_effect):
            res = self.manager.generate("Test model fallback")
            self.assertIn("Success on", res["text"])
            self.assertEqual(res["key_id"], "KEY_1")
            self.assertTrue(res["fallback_used"])
            self.assertNotEqual(res["model"], TASK_PRIORITY_MODELS["text"][0])

    def test_scenario_4_key1_invalid_credentials(self):
        """Scenario 4: Key 1 invalid (401/403) -> Permanently disabled, switches to Key 2."""
        def side_effect(key_info, model_name, **kwargs):
            if key_info.key_id == "KEY_1":
                raise Exception("403 API_KEY_INVALID: API key not valid")
            return "Key 2 Valid Output"

        with patch.object(self.manager, "_call_sdk", side_effect=side_effect):
            res = self.manager.generate("Check invalid auth")
            self.assertEqual(res["text"], "Key 2 Valid Output")
            self.assertEqual(res["key_id"], "KEY_2")

            # Verify Key 1 is permanently INVALID
            k1 = next(k for k in self.manager.key_manager.keys if k.key_id == "KEY_1")
            self.assertEqual(k1.status, GeminiKeyStatus.INVALID)
            self.assertFalse(k1.is_available())

    def test_scenario_5_all_keys_and_models_fail_gracefully(self):
        """Scenario 5: All keys/models fail -> Raises clean application error, no crash."""
        with patch.object(self.manager, "_call_sdk", side_effect=Exception("503 Service Unavailable")):
            with self.assertRaises(RuntimeError) as ctx:
                self.manager.generate("All fail test")
            self.assertIn("temporarily unavailable", str(ctx.exception))

    def test_scenario_6_transient_server_error_retry_backoff(self):
        """Scenario 6: Transient 500 error on attempt 1, succeeds on attempt 2."""
        attempts = {"count": 0}

        def side_effect(key_info, model_name, **kwargs):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise Exception("500 InternalServerError")
            return "Recovered after retry"

        with patch.object(self.manager, "_call_sdk", side_effect=side_effect), \
             patch("time.sleep") as mock_sleep:
            res = self.manager.generate("Retry test")
            self.assertEqual(res["text"], "Recovered after retry")
            self.assertEqual(mock_sleep.call_count, 1)

    def test_scenario_7_invalid_request_fast_fails(self):
        """Scenario 7: 400 Invalid Argument fails fast without wasting retries."""
        with patch.object(self.manager, "_call_sdk", side_effect=Exception("400 InvalidArgument: Malformed")):
            with self.assertRaises(ValueError) as ctx:
                self.manager.generate("Bad request test")
            self.assertIn("Invalid Gemini request", str(ctx.exception))

    def test_scenario_8_latest_model_auto_discovery(self):
        """Scenario 8: Model auto-discovery dynamically sorts granted models."""
        k_info = KeyInfo("KEY_1", "mock_key")
        k_info.discovered_models = ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-custom-enterprise"]
        candidates = ModelSelector.get_candidate_models(k_info, task_type="text")
        
        # gemini-2.0-flash should precede gemini-1.5-flash based on master priority
        self.assertIn("gemini-2.0-flash", candidates)
        self.assertIn("gemini-1.5-flash", candidates)
        self.assertLess(candidates.index("gemini-2.0-flash"), candidates.index("gemini-1.5-flash"))


if __name__ == "__main__":
    unittest.main()
