#!/usr/bin/env bash
#SBATCH --job-name=pr-build-gpu-envs
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --account=uzk
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
# Build the two GPU conda envs (no container): pr_serve (vLLM 0.19.0, as submitted) and pr_train
# (LoRA SFT stack). Installs only; the CUDA check runs in the GPU smoke job. Refuses to overwrite.
set -euo pipefail
PROJECT_DIR=/msc/home/ssingh54/partial-refs; cd "$PROJECT_DIR"
export CONDA_PKGS_DIRS=$PROJECT_DIR/.cache/conda_pkgs PIP_CACHE_DIR=$PROJECT_DIR/.cache/pip
INFO=$PROJECT_DIR/results/env/gpu_envs_${SLURM_JOB_ID}; mkdir -p "$INFO"
set +u; source "$HOME/miniconda3/etc/profile.d/conda.sh"; set -u
echo "node=$(hostname) job=$SLURM_JOB_ID start=$(date -Is)"
for spec in "pr_serve:env/requirements-gpu-serve.txt" "pr_train:env/requirements-gpu-train.txt"; do
  name=${spec%%:*}; req=${spec#*:}
  if conda env list | awk '{print $1}' | grep -qx "$name"; then echo "$name exists; skipping" >&2; continue; fi
  conda create -y -q -n "$name" -c conda-forge --override-channels python=3.12 pip
  set +u; conda activate "$name"; set -u
  pip install -q -r "$req"
  pip check > "$INFO/${name}_pip_check.txt" 2>&1 || echo "pip check reported problems for $name (see $INFO)"
  pip freeze > "env/${name}.lock.txt"
  python -c "import importlib.metadata as m; [print(p, m.version(p)) for p in ['torch','transformers','vllm','peft','accelerate'] if __import__('importlib').util.find_spec(p.replace('-','_'))]" | tee "$INFO/${name}_versions.txt"
  pip list 2>/dev/null | grep -i gptqmodel && { echo "gptqmodel present in $name: remove it" >&2; exit 1; } || true
  set +u; conda deactivate; set -u
done
echo "done=$(date -Is)"
