"""The five models and where they run. Change a model here and nowhere else.

Why these numbers (from the earlier runs):
  context_window 65536   32768 was too small: 9 long Hindi conversations failed because the history grew past it
  workers 16 on Sol      the earlier runs used 8 and only used 17 to 20% of the GPU's history memory.
                         vLLM queues extra requests, so more cannot crash it
  workers 4 on Bedrock   Bedrock limits how fast it can be called. Raise it if it copes
  Qwen3 32B on Bedrock   its weights (about 64 GB) leave only about 8 GB for history on one 80 GB GPU,
                         roughly 30,000 tokens in total, which is not enough for long Hindi conversations

Qwen "thinking": some Qwen models write hidden reasoning (<think> ... </think>) before the answer, which costs
tokens and can leak into the reply. Qwen3-4B-Instruct-2507 never thinks. Qwen3-32B does by default, so the
system message below switches it off. Check the first finished replies for any <think> text.
"""

MODELS = {
    "qwen3_4b": {
        "model_id": "Qwen/Qwen3-4B-Instruct-2507",
        "backend": "sol", "context_window": 65536, "workers": 16,
    },
    "llama32_3b": {
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "backend": "sol", "context_window": 65536, "workers": 16,
    },
    "llama31_8b": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "backend": "sol", "context_window": 65536, "workers": 16,
    },
    "qwen3_32b": {
        "model_id": "qwen.qwen3-32b-v1:0",                 # Bedrock id, taken from the earlier project
        "backend": "bedrock", "workers": 4,
        "system_message": "You are a helpful assistant. /no_think",
    },
    "llama33_70b": {
        "model_id": "us.meta.llama3-3-70b-instruct-v1:0",  # Bedrock id, taken from the earlier project
        "backend": "bedrock", "workers": 4,
    },
}

# The same for every model, so results are comparable.
# A reply that is cut off is asked again with double the tokens, up to max_tokens_cap.
SETTINGS = {"temperature": 0, "max_tokens": 4096, "max_tokens_cap": 8192}

# Claude Haiku 3.5 on Bedrock — used by grader/judge.py as the LLM judge.
# Shashwat: import this and pass it to models.bedrock.client.make_chat().
# Confirmed available in us-west-2. Uses an inference profile ID (us. prefix).
JUDGE_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
