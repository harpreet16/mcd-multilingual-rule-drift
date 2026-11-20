"""Chat client for Amazon Bedrock (Qwen3-32B and Llama 3.3-70B).

Behaves like models/sol/client.py:
  - make_chat(...) returns chat(messages) -> (text, cut_off)
  - cut_off is True when the reply hit the token limit
  - ask again with double the tokens when a reply is cut off, up to max_tokens_cap
  - retry with a growing wait when Bedrock says it is busy
  - raise ServerUnreachable when it cannot connect at all
  - pass system_message to every call when one is set (switches Qwen thinking off)
  - strip any <think>...</think> text that leaks into the reply and warn about it

The judge (grader/judge.py) reuses this client with Claude Haiku's model ID, so keep it general.

Environment variables
---------------------
AWS_ACCESS_KEY_ID       IAM access key (required)
AWS_SECRET_ACCESS_KEY   IAM secret key (required)
AWS_DEFAULT_REGION      Region to use (default: us-west-2)
BEDROCK_MAX_RETRIES     How many times to retry a throttled call (default: 5)
BEDROCK_TIMEOUT         Seconds to wait for one API call (default: 60)

boto3 also reads AWS_SESSION_TOKEN when you use temporary credentials.
Never set any of these in code — only in the environment or a local .env file.
"""
import os
import re
import time
import warnings

from contracts.models import ServerUnreachable

# Throttling and service errors that are worth retrying.
_RETRYABLE = frozenset({
    "ThrottlingException",
    "ServiceUnavailableException",
    "ModelNotReadyException",
    "RequestThrottledException",
})


def _make_bedrock_client(region):
    """Return a boto3 bedrock-runtime client using credentials from the environment."""
    try:
        import boto3
    except ImportError as exc:
        raise ImportError("boto3 is required for the Bedrock backend. Install it with: pip install boto3") from exc
    return boto3.client("bedrock-runtime", region_name=region)


def _to_bedrock_messages(messages):
    """Convert OpenAI-style messages to the Bedrock Converse format.

    OpenAI format:  [{"role": "user", "content": "text"}, ...]
    Bedrock format: [{"role": "user", "content": [{"text": "text"}]}, ...]
    """
    result = []
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            result.append({"role": m["role"], "content": [{"text": content}]})
        elif isinstance(content, list):
            # Already in Bedrock format — pass through unchanged.
            result.append(m)
        else:
            result.append({"role": m["role"], "content": [{"text": str(content)}]})
    return result


def _strip_think(text):
    """Remove <think>...</think> blocks from a reply. Return (cleaned_text, count_stripped).

    Qwen3 models can emit hidden reasoning wrapped in these tags even when thinking
    is switched off. The count is returned so callers can warn about it.
    """
    cleaned, n = re.subn(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip(), n


def _post_once(bedrock_client, model_id, messages, budget, temperature, system_message):
    """Make one Converse call. Return (text, cut_off).

    Does not retry — the caller handles that.
    Raises ServerUnreachable on connection failures.
    Re-raises any other ClientError after extracting the error code.
    """
    try:
        import botocore.exceptions
    except ImportError:
        import boto3.exceptions as botocore_exceptions

    kwargs = {
        "modelId": model_id,
        "messages": _to_bedrock_messages(messages),
        "inferenceConfig": {"maxTokens": budget, "temperature": temperature},
    }
    if system_message:
        kwargs["system"] = [{"text": system_message}]

    try:
        response = bedrock_client.converse(**kwargs)
    except Exception as exc:
        # Classify the exception using the error code when available.
        code = getattr(getattr(exc, "response", None), "get", lambda k, d=None: d)("Error", {}).get("Code", "")
        if not code:
            # Try botocore's exception structure.
            code = type(exc).__name__
        if code in ("EndpointConnectionError", "ConnectTimeoutError",
                    "ReadTimeoutError", "ConnectionClosedError"):
            raise ServerUnreachable(model_id) from exc
        # For retryable errors the caller re-raises and retries.
        raise

    output_text = response["output"]["message"]["content"][0]["text"].strip()
    cut_off = response.get("stopReason") == "max_tokens"
    return output_text, cut_off


def _post_with_retry(bedrock_client, model_id, messages, budget, temperature, system_message,
                     max_retries, max_tokens_cap):
    """Call Bedrock with exponential backoff on throttling and cutoff doubling.

    Returns (text, cut_off). The text has <think> blocks stripped.
    Raises ServerUnreachable when the endpoint cannot be reached.
    Raises RuntimeError when retries are exhausted.
    """
    while True:
        for attempt in range(max_retries):
            try:
                text, cut_off = _post_once(bedrock_client, model_id, messages, budget,
                                           temperature, system_message)
                break  # success
            except ServerUnreachable:
                raise
            except Exception as exc:
                code = type(exc).__name__
                # Try to get the real AWS error code.
                try:
                    code = exc.response["Error"]["Code"]  # type: ignore[attr-defined]
                except (AttributeError, KeyError, TypeError):
                    pass
                if code not in _RETRYABLE or attempt == max_retries - 1:
                    raise
                wait = 2 ** attempt + 1   # 2, 3, 5, 9, ... seconds
                time.sleep(wait)

        # Strip thinking blocks and warn if any were found.
        text, n_stripped = _strip_think(text)
        if n_stripped:
            warnings.warn(
                f"Bedrock ({model_id}): stripped {n_stripped} <think> block(s) from a reply. "
                "Check that the system_message is working.",
                stacklevel=4,
            )

        # An empty reply after stripping is treated as cut off and asked again.
        if not cut_off and text:
            return text, False

        if budget >= max_tokens_cap:
            return text, True   # still cut off or empty at the cap — return as cut off

        budget = min(budget * 2, max_tokens_cap)


def make_chat(model_id, region=None, max_tokens=4096, max_tokens_cap=8192,
              temperature=0, system_message=None):
    """Return a chat(messages) function that calls Bedrock and returns (text, cut_off).

    Parameters
    ----------
    model_id        : Bedrock model ID, e.g. "us.meta.llama3-3-70b-instruct-v1:0"
    region          : AWS region. Defaults to AWS_DEFAULT_REGION env var, then "us-west-2"
    max_tokens      : Token budget for the first attempt
    max_tokens_cap  : Maximum token budget (asked again with double if cut off, up to this)
    temperature     : Sampling temperature. Use 0 for reproducible results
    system_message  : Optional system message, e.g. "/no_think" for Qwen3 models

    Environment variables read
    --------------------------
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION  (standard boto3)
    BEDROCK_MAX_RETRIES   override the retry count (default 5)
    BEDROCK_TIMEOUT       not used directly — boto3 config handles socket timeouts
    """
    effective_region = region or os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
    max_retries = int(os.environ.get("BEDROCK_MAX_RETRIES", "5"))
    bedrock_client = _make_bedrock_client(effective_region)

    def chat(messages):
        return _post_with_retry(
            bedrock_client, model_id, messages, max_tokens, temperature,
            system_message, max_retries, max_tokens_cap,
        )

    return chat
