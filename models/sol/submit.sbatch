#!/bin/bash
#SBATCH --job-name=mcd-run
#SBATCH --time=0-05:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -A class_cse59888883fall2026   # change this to your own Sol account
#SBATCH -q class
#SBATCH -p public
#SBATCH --output=/scratch/%u/mcd_runs/%x_%j.out
#
# Run one model on Sol: start vLLM, run every variant, stop vLLM.
# Usage (from the repo folder, with HF_TOKEN set):
#   sbatch models/sol/submit.sbatch llama31_8b                              # default variants file
#   sbatch models/sol/submit.sbatch llama31_8b pilots/variants.jsonl        # another variants file
#   VARIANTS=pilots/variants.jsonl OUT=/scratch/$USER/try sbatch models/sol/submit.sbatch llama31_8b
# Order of choice for the variants file: second argument, then $VARIANTS, then the default below.
# Relative paths are read from the repo folder (where you ran sbatch).
# Safe to run again: finished conversations are kept and only the rest is redone.
set -uo pipefail

KEY="${1:?usage: sbatch models/sol/submit.sbatch <model key> [variants file]}"
VARIANTS="${2:-${VARIANTS:-dataset/data/variants.jsonl}}"
OUT="${OUT:-/scratch/$USER/mcd_runs/$KEY}"
LOGDIR="/scratch/$USER/mcd_runs"

if [ -z "${HF_TOKEN:-}" ]; then echo "HF_TOKEN is not set"; exit 1; fi
cd "$SLURM_SUBMIT_DIR" || exit 1
# Check the file before starting vLLM, which takes minutes.
if [ ! -f "$VARIANTS" ]; then echo "variants file not found: $VARIANTS (from $(pwd))"; exit 1; fi
echo "variants: $VARIANTS  out: $OUT"
mkdir -p "$LOGDIR" "$OUT"

module load mamba/latest
source activate mcd-vllm        # an environment that has vllm installed
export HF_HOME="/scratch/$USER/hf_cache"

# The model id and window come from models/settings.py, so there is one place to change them.
read -r MODEL_ID CONTEXT < <(python3 -c "from models.settings import MODELS; m = MODELS['$KEY']; print(m['model_id'], m['context_window'])")
echo "node: $(hostname)  gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "model: $MODEL_ID  window: $CONTEXT"

vllm serve "$MODEL_ID" --dtype half --max-model-len "$CONTEXT" --gpu-memory-utilization 0.90 \
      --port 8000 > "$LOGDIR/vllm_$KEY.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null' EXIT

# Wait up to 15 minutes. The first start downloads the model.
READY=0
for _ in $(seq 1 180); do
  if curl -sf http://localhost:8000/v1/models > /dev/null; then READY=1; echo "server ready"; break; fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then echo "vLLM stopped:"; tail -n 30 "$LOGDIR/vllm_$KEY.log"; exit 1; fi
  sleep 5
done
[ "$READY" = 1 ] || { echo "server not ready after 15 minutes"; exit 1; }

# The second pass redoes anything that failed in the first.
for pass in 1 2; do
  python3 -m runner --model "$KEY" --variants "$VARIANTS" --out "$OUT"
done
python3 -m runner --model "$KEY" --variants "$VARIANTS" --out "$OUT" --status
echo ALL_DONE
