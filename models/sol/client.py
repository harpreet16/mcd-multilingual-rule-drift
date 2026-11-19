"""Chat client for a vLLM server on Sol. It speaks the OpenAI style chat API."""
import json
import time
import urllib.error
import urllib.request

from contracts.models import ServerUnreachable


def reply_with_bigger_budget(post, messages, max_tokens, cap):
    """Ask once. If the reply is cut off or empty, ask again with double the tokens, up to cap.

    The temperature is 0, so a bigger budget only lets the same answer finish.
    post returns (text, cut_off), and so does this function.
    """
    budget = max_tokens
    while True:
        text, cut_off = post(messages, budget)
        if (not cut_off and text.strip()) or budget >= cap:
            return text, cut_off
        budget = min(budget * 2, cap)


def make_chat(base_url, model_id, max_tokens=4096, max_tokens_cap=8192, temperature=0,
              extra_body=None, retries=5, timeout=600):
    """Return a chat(messages) function that talks to the server at base_url and returns (text, cut_off)."""

    def post(messages, budget):
        body = json.dumps({"model": model_id, "messages": messages, "max_tokens": budget,
                           "temperature": temperature, **(extra_body or {})}).encode()
        headers = {"Content-Type": "application/json"}
        for attempt in range(retries):
            try:
                request = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", body, headers)
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    choice = json.load(response)["choices"][0]
                return choice["message"]["content"] or "", choice.get("finish_reason") == "length"
            except urllib.error.HTTPError as error:
                # A busy or overloaded server is worth a retry. A bad request is not.
                if error.code not in (429, 500, 502, 503) or attempt == retries - 1:
                    detail = error.read().decode("utf-8", "replace")[:300]
                    raise RuntimeError(f"HTTP {error.code} from the model server: {detail or error.reason}") from error
                time.sleep(2 ** attempt)
            except (TimeoutError, urllib.error.URLError, ConnectionError, OSError) as error:
                if isinstance(getattr(error, "reason", error), ConnectionRefusedError):
                    raise ServerUnreachable(base_url) from error
                if attempt >= 2:
                    raise
                time.sleep(2 ** attempt)

    def chat(messages):
        return reply_with_bigger_budget(post, messages, max_tokens, max_tokens_cap)

    return chat
