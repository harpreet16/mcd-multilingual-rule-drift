"""Chat client for Amazon Bedrock (Qwen3-32B and Llama 3.3-70B).

Behaves like models/sol/client.py:
  - make_chat(...) returns chat(messages) -> (text, cut_off)
  - cut_off is True when the reply hit the token limit
  - ask again with double the tokens when a reply is cut off, up to max_tokens_cap
  - retry with jittered exponential backoff when Bedrock throttles
  - raise ServerUnreachable when it cannot connect at all
  - pass system_message to every call when one is set (switches Qwen thinking off)
  - strip any <think>...</think> text that leaks into the reply and warn about it

The judge (grader/judge.py) reuses this client with Claude Haiku's model ID, so keep it general.

Environment variables
---------------------
AWS_ACCESS_KEY_ID       IAM access key (required)
AWS_SECRET_ACCESS_KEY   IAM secret key (required)
AWS_DEFAULT_REGION      Region to use (default: us-west-2)
BEDROCK_MAX_RETRIES     How many times to retry a throttled call (default: 8)
BEDROCK_MAX_WAIT        Maximum seconds to wait between retries (default: 60)
BEDROCK_TIMEOUT         Reserved for future socket-level timeout tuning

boto3 also reads AWS_SESSION_TOKEN when you use temporary credentials.
Never set any of these in code — only in the environment or a local .env file.

Throttling behaviour
--------------------
When Bedrock returns ThrottlingException (or related errors), the client waits
and retries. The wait grows exponentially with jitter so that parallel workers
do not all retry at the same moment and immediately throttle again:

    wait = min(BEDROCK_MAX_WAIT, 2 ** attempt + random.uniform(0, 2))

Each retry is logged to stderr so you can see it happening during a run.
After BEDROCK_MAX_RETRIES attempts the exception is re-raised and the runner
marks that conversation as failed (without stopping the rest of the run).
"""
import os
import random
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
        raise ImportError(
            "boto3 is required for the Bedrock backend. Install it with: pip install boto3"
        ) from exc
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


def _backoff_wait(attempt, max_wait):
    """Return seconds to wait before retry attempt, with jitter.

    wait = min(max_wait, 2 ** attempt + uniform(0, 2))

    Jitter prevents parallel workers from all retrying at the same moment
    and immediately throttling each other again (thundering herd).

    Examples (without jitter component):
      attempt 0 → ~1s   attempt 3 → ~10s
      attempt 1 → ~3s   attempt 4 → ~18s
      attempt 2 → ~5s   attempt 7 → ~130s (capped at max_wait)
    """
    return min(max_wait, 2 ** attempt + random.uniform(0, 2))


def _post_once(bedrock_client, model_id, messages, budget, temperature, system_message):
    """Make one Converse call. Return (text, cut_off).

    Does not retry — the caller handles that.
    Raises ServerUnreachable on connection failures.
    Re-raises any other exception (including throttling) for the caller to handle.
    """
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
        code = getattr(getattr(exc, "response", None), "get", lambda k, d=None: d)("Error", {}).get("Code", "")
        if not code:
            code = type(exc).__name__
        if code in ("EndpointConnectionError", "ConnectTimeoutError",
                    "ReadTimeoutError", "ConnectionClosedError"):
            raise ServerUnreachable(model_id) from exc
        raise

    output_text = response["output"]["message"]["content"][0]["text"].strip()
    cut_off = response.get("stopReason") == "max_tokens"
    return output_text, cut_off


def _error_code(exc):
    """Extract the AWS error code from an exception, falling back to the class name."""
    try:
        return exc.response["Error"]["Code"]  # type: ignore[attr-defined]
    except (AttributeError, KeyError, TypeError):
        return type(exc).__name__


def _post_with_retry(bedrock_client, model_id, messages, budget, temperature, system_message,
                     max_retries, max_tokens_cap, max_wait):
    """Call Bedrock with jittered exponential backoff on throttling and cutoff doubling.

    Returns (text, cut_off). The text has <think> blocks stripped.
    Raises ServerUnreachable when the endpoint cannot be reached.
    Raises the original exception when retries are exhausted.

    Each throttle retry is printed to stderr so it is visible in the runner output.
    """
    while True:
        for attempt in range(max_retries):
            try:
                text, cut_off = _post_once(bedrock_client, model_id, messages, budget,
                                           temperature, system_message)
                break  # success — exit the retry loop
            except ServerUnreachable:
                raise
            except Exception as exc:
                code = _error_code(exc)
                if code not in _RETRYABLE or attempt == max_retries - 1:
                    raise
                wait = _backoff_wait(attempt, max_wait)
                print(
                    f"Bedrock throttled ({model_id}, attempt {attempt + 1}/{max_retries}), "
                    f"waiting {wait:.1f}s ...",
                    flush=True,
                )
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
            return text, True   # still cut off or empty at the cap

        budget = min(budget * 2, max_tokens_cap)


def make_chat(model_id, region=None, max_tokens=4096, max_tokens_cap=8192,
              temperature=0, system_message=None):
    """Return a chat(messages) function that calls Bedrock and returns (text, cut_off).

    Parameters
    ----------
    model_id        : Bedrock model ID, e.g. "us.meta.llama3-3-70b-instruct-v1:0"
    region          : AWS region. Defaults to AWS_DEFAULT_REGION env var, then "us-west-2"
    max_tokens      : Token budget for the first attempt
    max_tokens_cap  : Maximum token budget (doubled on cutoff, up to this limit)
    temperature     : Sampling temperature. Use 0 for reproducible results
    system_message  : Optional system message, e.g. "/no_think" for Qwen3 models

    Environment variables read
    --------------------------
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION  (standard boto3)
    BEDROCK_MAX_RETRIES   override the retry count (default 8)
    BEDROCK_MAX_WAIT      cap the wait between retries in seconds (default 60)
    """
    effective_region = region or os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
    max_retries = int(os.environ.get("BEDROCK_MAX_RETRIES", "8"))
    max_wait    = float(os.environ.get("BEDROCK_MAX_WAIT", "60"))
    bedrock_client = _make_bedrock_client(effective_region)

    def chat(messages):
        return _post_with_retry(
            bedrock_client, model_id, messages, max_tokens, temperature,
            system_message, max_retries, max_tokens_cap, max_wait,
        )

    return chat
