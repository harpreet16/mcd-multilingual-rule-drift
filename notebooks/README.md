# notebooks

`colab_run.ipynb` — run one Sol model on a Colab A100 instead of Sol.

It is a port of `models/sol/submit.sbatch`, step for step: preflight, start vLLM, play every
conversation twice, report. It calls the same `models.sol.preflight`, so a Colab run records the
same `jobs.jsonl` provenance line as a Sol run — dataset SHA-256, git commit, GPU and settings.

Use it when Sol is busy or you do not have an allocation. Needs an A100: the preflight refuses
anything with less than 40 GB or compute capability under 8.0, because the 65,536-token window plus
16 conversations at once does not fit on a T4 or an L4.

Set `MODEL` to `qwen3_4b` or `llama32_3b` and run the cells in order. Output goes to Drive, so a
Colab disconnect costs one turn, not the run: open the notebook again, run every cell, and the
runner skips what is already finished.

`Llama-3.2-3B` is gated — accept Meta's licence on Hugging Face with the account your token belongs
to first. `Qwen3-4B` is not, so start there.
