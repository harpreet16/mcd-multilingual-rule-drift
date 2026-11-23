"""Tests for the Bedrock client. No real AWS calls are made — boto3 is mocked throughout."""
import sys
import types
import unittest
import warnings
from unittest.mock import MagicMock, patch, call

from contracts.models import ServerUnreachable


# ── helpers to build fake boto3 responses ────────────────────────────────────

def _ok_response(text, stop_reason="end_turn"):
    return {"output": {"message": {"content": [{"text": text}]}}, "stopReason": stop_reason}


def _error(code, message="error"):
    """Build an exception that looks like a botocore ClientError."""
    err = Exception(message)
    err.response = {"Error": {"Code": code, "Message": message}}
    return err


# ── inject a fake boto3 so the import works without it installed ──────────────

class FakeBoto3:
    """Minimal boto3 stand-in that returns a controllable client."""
    def __init__(self, mock_client):
        self._client = mock_client

    def client(self, service, region_name=None):
        return self._client


def _patch_boto3(mock_client):
    fake = FakeBoto3(mock_client)
    return patch.dict(sys.modules, {"boto3": fake})


# ── tests ─────────────────────────────────────────────────────────────────────

class MessageConversionTests(unittest.TestCase):
    """The client translates OpenAI-style messages to Bedrock Converse format."""

    def setUp(self):
        self.mock_client = MagicMock()

    def _run(self, messages):
        self.mock_client.converse.return_value = _ok_response("ok")
        with _patch_boto3(self.mock_client):
            from importlib import reload
            import models.bedrock.client as m
            reload(m)
            chat = m.make_chat("model-id", region="us-east-1")
            return chat(messages)

    def test_string_content_is_wrapped_in_bedrock_format(self):
        self._run([{"role": "user", "content": "hello"}])
        sent = self.mock_client.converse.call_args[1]["messages"]
        self.assertEqual(sent[0]["content"], [{"text": "hello"}])

    def test_already_bedrock_format_passes_through(self):
        self._run([{"role": "user", "content": [{"text": "already bedrock"}]}])
        sent = self.mock_client.converse.call_args[1]["messages"]
        self.assertEqual(sent[0]["content"], [{"text": "already bedrock"}])


class NormalReplyTests(unittest.TestCase):
    def setUp(self):
        self.mock_client = MagicMock()
        self.mock_client.converse.return_value = _ok_response("hello")

    def _chat(self, **kwargs):
        with _patch_boto3(self.mock_client):
            from importlib import reload
            import models.bedrock.client as m
            reload(m)
            chat = m.make_chat("model-id", region="us-east-1", **kwargs)
            return chat([{"role": "user", "content": "hi"}])

    def test_complete_reply_returns_text_and_false(self):
        text, cut_off = self._chat()
        self.assertEqual(text, "hello")
        self.assertFalse(cut_off)

    def test_converse_is_called_once(self):
        self._chat()
        self.assertEqual(self.mock_client.converse.call_count, 1)

    def test_temperature_is_passed(self):
        self._chat(temperature=0)
        cfg = self.mock_client.converse.call_args[1]["inferenceConfig"]
        self.assertEqual(cfg["temperature"], 0)

    def test_max_tokens_is_passed(self):
        self._chat(max_tokens=2048)
        cfg = self.mock_client.converse.call_args[1]["inferenceConfig"]
        self.assertEqual(cfg["maxTokens"], 2048)

    def test_no_system_key_when_system_message_is_none(self):
        self._chat()
        self.assertNotIn("system", self.mock_client.converse.call_args[1])

    def test_system_message_is_passed_when_set(self):
        self._chat(system_message="/no_think")
        self.assertEqual(self.mock_client.converse.call_args[1]["system"],
                         [{"text": "/no_think"}])


class CutOffTests(unittest.TestCase):
    """When the reply hits the token limit it is asked again with double the budget."""

    def setUp(self):
        self.mock_client = MagicMock()

    def _chat(self, side_effects, **kwargs):
        self.mock_client.converse.side_effect = side_effects
        with _patch_boto3(self.mock_client):
            from importlib import reload
            import models.bedrock.client as m
            reload(m)
            chat = m.make_chat("model-id", region="us-east-1",
                               max_tokens=4096, max_tokens_cap=8192, **kwargs)
            return chat([{"role": "user", "content": "hi"}])

    def test_cut_off_reply_is_asked_again_with_double_the_tokens(self):
        self.mock_client.converse.side_effect = [
            _ok_response("partial", "max_tokens"),
            _ok_response("finished", "end_turn"),
        ]
        text, cut_off = self._chat([
            _ok_response("partial", "max_tokens"),
            _ok_response("finished", "end_turn"),
        ])
        self.assertEqual(text, "finished")
        self.assertFalse(cut_off)
        budgets = [c[1]["inferenceConfig"]["maxTokens"]
                   for c in self.mock_client.converse.call_args_list]
        self.assertEqual(budgets, [4096, 8192])

    def test_still_cut_off_at_the_cap_is_returned_as_cut_off(self):
        text, cut_off = self._chat([
            _ok_response("p1", "max_tokens"),
            _ok_response("p2", "max_tokens"),
        ])
        self.assertTrue(cut_off)
        budgets = [c[1]["inferenceConfig"]["maxTokens"]
                   for c in self.mock_client.converse.call_args_list]
        self.assertEqual(budgets, [4096, 8192])   # never exceeds the cap


class ThrottlingTests(unittest.TestCase):
    """ThrottlingException is retried with jittered exponential backoff."""

    def _chat(self, side_effects, env_overrides=None):
        mock_client = MagicMock()
        mock_client.converse.side_effect = side_effects
        env = {"AWS_DEFAULT_REGION": "us-east-1", **(env_overrides or {})}
        with _patch_boto3(mock_client):
            from importlib import reload
            import models.bedrock.client as m
            reload(m)
            with patch("time.sleep"), patch("random.uniform", return_value=0.5):
                with patch.dict("os.environ", env):
                    chat = m.make_chat("model-id", region="us-east-1")
                    return chat([{"role": "user", "content": "hi"}]), mock_client

    def test_throttling_is_retried_and_eventually_succeeds(self):
        result, mock_client = self._chat([
            _error("ThrottlingException"),
            _error("ThrottlingException"),
            _ok_response("done"),
        ])
        text, cut_off = result
        self.assertEqual(text, "done")
        self.assertFalse(cut_off)
        self.assertEqual(mock_client.converse.call_count, 3)

    def test_exhausted_retries_raises(self):
        with self.assertRaises(Exception):
            self._chat([_error("ThrottlingException")] * 10)

    def test_non_retryable_error_raises_immediately(self):
        mock_client = MagicMock()
        mock_client.converse.side_effect = _error("ValidationException")
        with _patch_boto3(mock_client):
            from importlib import reload
            import models.bedrock.client as m
            reload(m)
            chat = m.make_chat("model-id", region="us-east-1")
            with self.assertRaises(Exception):
                chat([{"role": "user", "content": "hi"}])
        self.assertEqual(mock_client.converse.call_count, 1)


class ThrottlingRetryBehaviourTests(unittest.TestCase):
    """Detailed tests for jitter, logging and env-var tuning."""

    def _run(self, side_effects, env_overrides=None):
        mock_client = MagicMock()
        mock_client.converse.side_effect = side_effects
        env = {"AWS_DEFAULT_REGION": "us-east-1", **(env_overrides or {})}
        sleeps = []
        with _patch_boto3(mock_client):
            from importlib import reload
            import models.bedrock.client as m
            reload(m)
            with patch("time.sleep", side_effect=lambda s: sleeps.append(s)):
                with patch("random.uniform", return_value=1.0):
                    with patch.dict("os.environ", env):
                        chat = m.make_chat("model-id", region="us-east-1")
                        try:
                            result = chat([{"role": "user", "content": "hi"}])
                        except Exception as exc:
                            result = exc
        return result, mock_client, sleeps

    def test_jitter_is_added_to_each_wait(self):
        # random.uniform patched to 1.0, so wait = min(60, 2**attempt + 1.0)
        # attempt 0 → 2.0, attempt 1 → 3.0
        _, _, sleeps = self._run([
            _error("ThrottlingException"),
            _error("ThrottlingException"),
            _ok_response("done"),
        ])
        self.assertAlmostEqual(sleeps[0], 2.0, places=5)
        self.assertAlmostEqual(sleeps[1], 3.0, places=5)

    def test_wait_is_capped_at_bedrock_max_wait(self):
        # 8 throttles with max_wait=5 — wait must never exceed 5
        side_effects = [_error("ThrottlingException")] * 7 + [_ok_response("done")]
        _, _, sleeps = self._run(side_effects, {"BEDROCK_MAX_WAIT": "5"})
        self.assertTrue(all(s <= 5.0 + 1e-9 for s in sleeps),
                        f"a wait exceeded max_wait=5: {sleeps}")

    def test_retry_is_printed_to_stdout(self):
        import io, sys
        mock_client = MagicMock()
        mock_client.converse.side_effect = [
            _error("ThrottlingException"),
            _ok_response("done"),
        ]
        with _patch_boto3(mock_client):
            from importlib import reload
            import models.bedrock.client as m
            reload(m)
            with patch("time.sleep"), patch("random.uniform", return_value=0.5):
                with patch.dict("os.environ", {"AWS_DEFAULT_REGION": "us-east-1"}):
                    chat = m.make_chat("model-id", region="us-east-1")
                    captured = io.StringIO()
                    sys.stdout = captured
                    try:
                        chat([{"role": "user", "content": "hi"}])
                    finally:
                        sys.stdout = sys.__stdout__
        output = captured.getvalue()
        self.assertIn("throttled", output.lower())
        self.assertIn("model-id", output)
        self.assertIn("1/", output)   # "attempt 1/N"

    def test_default_max_retries_is_8(self):
        # 7 throttles then success = 8 total calls, which fits within the default of 8
        side_effects = [_error("ThrottlingException")] * 7 + [_ok_response("done")]
        result, mock_client, _ = self._run(side_effects)
        self.assertEqual(result[0], "done")
        self.assertEqual(mock_client.converse.call_count, 8)

    def test_bedrock_max_retries_env_var_is_respected(self):
        # With BEDROCK_MAX_RETRIES=3, 3 throttles should exhaust retries
        side_effects = [_error("ThrottlingException")] * 3
        result, mock_client, _ = self._run(side_effects, {"BEDROCK_MAX_RETRIES": "3"})
        self.assertIsInstance(result, Exception)
        self.assertEqual(mock_client.converse.call_count, 3)


class ServerUnreachableTests(unittest.TestCase):
    def test_connection_error_raises_server_unreachable(self):
        mock_client = MagicMock()
        err = Exception("connection refused")
        err.response = {"Error": {"Code": "EndpointConnectionError"}}
        mock_client.converse.side_effect = err
        with _patch_boto3(mock_client):
            from importlib import reload
            import models.bedrock.client as m
            reload(m)
            chat = m.make_chat("model-id", region="us-east-1")
            with self.assertRaises(ServerUnreachable):
                chat([{"role": "user", "content": "hi"}])


class ThinkStripTests(unittest.TestCase):
    """<think> blocks are stripped from replies and a warning is issued."""

    def test_think_block_is_removed(self):
        mock_client = MagicMock()
        mock_client.converse.return_value = _ok_response(
            "<think>internal reasoning</think>The actual answer"
        )
        with _patch_boto3(mock_client):
            from importlib import reload
            import models.bedrock.client as m
            reload(m)
            chat = m.make_chat("model-id", region="us-east-1")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                text, _ = chat([{"role": "user", "content": "hi"}])
        self.assertEqual(text, "The actual answer")
        self.assertTrue(any("think" in str(w.message).lower() for w in caught),
                        "expected a warning about stripped think blocks")

    def test_reply_without_think_block_has_no_warning(self):
        mock_client = MagicMock()
        mock_client.converse.return_value = _ok_response("clean answer")
        with _patch_boto3(mock_client):
            from importlib import reload
            import models.bedrock.client as m
            reload(m)
            chat = m.make_chat("model-id", region="us-east-1")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                chat([{"role": "user", "content": "hi"}])
        self.assertFalse(any("think" in str(w.message).lower() for w in caught))


class EnvironmentVariableTests(unittest.TestCase):
    """The client reads region and retry count from the environment."""

    def test_region_defaults_to_env_var(self):
        mock_client = MagicMock()
        mock_client.converse.return_value = _ok_response("ok")
        fake = FakeBoto3(mock_client)
        captured_regions = []

        class CapturingFakeBoto3:
            def client(self, service, region_name=None):
                captured_regions.append(region_name)
                return mock_client

        with patch.dict(sys.modules, {"boto3": CapturingFakeBoto3()}):
            with patch.dict("os.environ", {"AWS_DEFAULT_REGION": "eu-west-1"}):
                from importlib import reload
                import models.bedrock.client as m
                reload(m)
                chat = m.make_chat("model-id")
                chat([{"role": "user", "content": "hi"}])
        self.assertIn("eu-west-1", captured_regions)

    def test_explicit_region_overrides_env_var(self):
        mock_client = MagicMock()
        mock_client.converse.return_value = _ok_response("ok")
        captured_regions = []

        class CapturingFakeBoto3:
            def client(self, service, region_name=None):
                captured_regions.append(region_name)
                return mock_client

        with patch.dict(sys.modules, {"boto3": CapturingFakeBoto3()}):
            with patch.dict("os.environ", {"AWS_DEFAULT_REGION": "eu-west-1"}):
                from importlib import reload
                import models.bedrock.client as m
                reload(m)
                chat = m.make_chat("model-id", region="ap-southeast-1")
                chat([{"role": "user", "content": "hi"}])
        self.assertIn("ap-southeast-1", captured_regions)


if __name__ == "__main__":
    unittest.main()
