"""Chat client for Amazon Bedrock (Qwen3 32B and Llama 3.3 70B). Not built yet.

Use the Converse API. Behave like models/sol/client.py:
  - return (text, cut_off), where cut_off is True when the reply hit the token limit
  - ask again with double the tokens when a reply is cut off, up to max_tokens_cap
  - retry with a growing wait when Bedrock says it is busy
  - raise ServerUnreachable when it cannot connect at all
  - add system_message to every call when there is one (it switches Qwen's thinking off)
  - if a reply still contains <think> text, remove it and count how many were removed

The judge (grader/judge.py) reuses this client, so keep it general.
Check early that Sol compute nodes can reach Bedrock, because the judge is run from Sol.
Bedrock charges per token and every turn sends the whole history again, so estimate the cost first.
"""


def make_chat(model_id, region, max_tokens=4096, max_tokens_cap=8192, temperature=0, system_message=None):
    """Return a chat(messages) function that calls Bedrock."""
    raise NotImplementedError("Not built yet")
