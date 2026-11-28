# Do not include this folder in any analysis

This is a **partial** run: 181 responses over 48 of the 100 conversations. It was never finished.

It is in the repo for one reason only: comparing it against `runs/qwen3_4b/` is a free
reproducibility check, because both are the same model at temperature 0 on different hardware
(Colab A100 40 GB / vLLM 0.34.0 here, Sol 80 GB / vLLM 0.33.0 there). The replies should be
near identical, and where they are not, that is worth knowing.

It is **not** a sixth model and **not** a replacement for `runs/qwen3_4b/`, which is complete.

**If you are writing analysis code, list the five model folders explicitly:**

    qwen3_4b  qwen3_32b  llama31_8b  llama32_3b  llama33_70b

A glob such as `runs/*/responses.jsonl` will pick this folder up and quietly mix a half-finished
run into your numbers. There are no grades here, so it would also skew any coverage figure.
