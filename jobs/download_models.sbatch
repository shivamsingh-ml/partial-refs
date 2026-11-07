#!/usr/bin/env bash
#SBATCH --job-name=pr-dl-models
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --account=uzk
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
# Download model weights into $PROJECT_DIR/models (gitignored). Public repos only; no token used.
#   released V2V2R model (LoRA merged into AWQ)  lucent517/v2v2r_cbct_vqa  -> models/Qwen3.5-9B-AWQ-dental-cbct-sft
#   bf16 base for retraining                    Qwen/Qwen3.5-9B           -> models/Qwen3.5-9B
# Records the resolved commit of each repo for provenance.
set -euo pipefail
PROJECT_DIR=/msc/home/ssingh54/partial-refs; cd "$PROJECT_DIR"
export HF_HOME=$PROJECT_DIR/.cache/hf HF_HUB_DISABLE_TELEMETRY=1
set +u; source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate pr_serve; set -u
echo "node=$(hostname) job=$SLURM_JOB_ID start=$(date -Is)"; df -h "$PROJECT_DIR" | tail -1
python - <<'PY'
import json, os
from huggingface_hub import snapshot_download, HfApi
api = HfApi()
out = {}
for repo, local in [("lucent517/v2v2r_cbct_vqa", "models/Qwen3.5-9B-AWQ-dental-cbct-sft"),
                    ("Qwen/Qwen3.5-9B", "models/Qwen3.5-9B")]:
    info = api.model_info(repo)
    path = snapshot_download(repo_id=repo, revision=info.sha, local_dir=local)
    size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(path) for f in fs)
    out[repo] = {"sha": info.sha, "local_dir": local, "bytes": size}
    print(repo, info.sha, f"{size/1e9:.1f} GB", flush=True)
os.makedirs("results/env", exist_ok=True)
json.dump(out, open(f"results/env/models_{os.environ['SLURM_JOB_ID']}.json", "w"), indent=2)
PY
df -h "$PROJECT_DIR" | tail -1; echo "done=$(date -Is)"
