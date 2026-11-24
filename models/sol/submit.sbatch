#!/bin/bash
#SBATCH --job-name=mcd-run
#SBATCH --time=0-05:00:00
#SBATCH --gres=gpu:1                     # any GPU; preflight stops at once if it cannot run the model
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -A class_cse59888883fall2026     # change this to your own Sol account
#SBATCH -q class
#SBATCH -p public
#SBATCH --output=/scratch/%u/mcd_runs/slurm_%j.out
#
# Run one Sol model on every variant: check the inputs, start vLLM, play the conversations, stop vLLM.
#
# Usage, from the repo folder, with HF_TOKEN set (one time first: mkdir -p /scratch/$USER/mcd_runs):
#   sbatch models/sol/submit.sbatch llama31_8b                            # dataset/data/variants.jsonl
#   sbatch models/sol/submit.sbatch llama31_8b my_subset.jsonl            # another variants file
#   VARIANTS=my_subset.jsonl OUT=/scratch/$USER/try sbatch models/sol/submit.sbatch llama31_8b
# Any variants file must hold remixer output: preflight checks every turn against dataset/data/conversations.jsonl.
# Variants file: second argument, else $VARIANTS, else dataset/data/variants.jsonl (relative to the repo folder).
#
# Everything a run makes is in one folder, OUT (default /scratch/$USER/mcd_runs/<model>):
#   responses.jsonl        the result: one finished conversation per line. The grader reads this
#   partial.json           conversations stopped half way, so a resubmit continues them ({} when all done)
#   jobs.jsonl             one line per job: dataset sha256, git commit, GPU, vLLM version, settings
#   logs/job_<id>.log      everything this job printed
#   logs/vllm_<id>.log     the vLLM server log
#
# Safe to submit again (for example after the time limit): finished conversations are kept, the rest is redone.
set -uo pipefail

KEY="${1:?usage: sbatch models/sol/submit.sbatch <model key> [variants file]}"
VARIANTS="${2:-${VARIANTS:-dataset/data/variants.jsonl}}"
OUT="${OUT:-/scratch/$USER/mcd_runs/$KEY}"
JOB="${SLURM_JOB_ID:-local}"

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p "$OUT/logs"
exec > >(tee -a "$OUT/logs/job_$JOB.log") 2>&1    # copy all output into the run folder
echo "job $JOB  model $KEY  variants $VARIANTS  out $OUT"

if [ -z "${HF_TOKEN:-}" ]; then echo "HF_TOKEN is not set"; exit 1; fi
module load mamba/latest
source activate mcd-vllm                         # an environment that has vllm installed
export HF_HOME="/scratch/$USER/hf_cache"

# 1. Check the GPU and the inputs before starting vLLM. Its last line is the model id, window and dtype from models/settings.py.
PREFLIGHT=$(python3 -m models.sol.preflight --model "$KEY" --variants "$VARIANTS" --out "$OUT") || exit 1
echo "$PREFLIGHT" | sed '$d'
read -r MODEL_ID CONTEXT DTYPE <<< "$(echo "$PREFLIGHT" | tail -n 1)"
echo "model: $MODEL_ID  window: $CONTEXT  dtype: $DTYPE  node: $(hostname)"

# 2. Start vLLM. The port comes from the job id, so two jobs on one node never talk to each other's server.
PORT=$((20000 + JOB % 20000))
vllm serve "$MODEL_ID" --dtype "$DTYPE" --max-model-len "$CONTEXT" --gpu-memory-utilization 0.90 \
      --seed 0 --port "$PORT" > "$OUT/logs/vllm_$JOB.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null' EXIT

# Wait up to 15 minutes (the first start downloads the model), and check it serves the right model.
READY=0
for _ in $(seq 1 180); do
  if curl -sf "http://localhost:$PORT/v1/models" | grep -q "\"$MODEL_ID\""; then READY=1; echo "server ready on port $PORT"; break; fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then echo "vLLM stopped:"; tail -n 30 "$OUT/logs/vllm_$JOB.log"; exit 1; fi
  sleep 5
done
[ "$READY" = 1 ] || { echo "server not ready after 15 minutes"; exit 1; }

# 3. Play every conversation. The second pass redoes anything that failed in the first.
RUN=(python3 -m runner --model "$KEY" --variants "$VARIANTS" --out "$OUT" --base-url "http://localhost:$PORT/v1")
for pass in 1 2; do
  echo "pass $pass"
  "${RUN[@]}"
done

# 4. Report. ALL_DONE only when every variant is finished.
STATUS=$("${RUN[@]}" --status)
echo "status: $STATUS"
if echo "$STATUS" | grep -q '"to_do": 0,'; then
  echo ALL_DONE
else
  echo "INCOMPLETE: submit the same command again to finish the rest"
  exit 1
fi
