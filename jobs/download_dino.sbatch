#!/usr/bin/env bash
#SBATCH --job-name=pr-dl-dino
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --account=uzk
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
# Download DINO image models into $PROJECT_DIR/models (gitignored), for per-tooth embeddings (PAPER_PLAN
# step 2). DINOv2: Apache-2.0, not gated. DINOv3: Meta's DINOv3 licence, gated on Hugging Face; Shivam accepted
# it with his account (2026-11-12); the download uses his token from ~/.secrets/hf.env (never printed).
# Records the resolved commit of each repo; a repo that fails (e.g. licence not accepted) is reported, not fatal.
#   sbatch jobs/download_dino.sbatch [v2|v3|all]   (default v2)
set -euo pipefail
PROJECT_DIR=/msc/home/ssingh54/partial-refs; cd "$PROJECT_DIR"
export HF_HOME=$PROJECT_DIR/.cache/hf HF_HUB_DISABLE_TELEMETRY=1
set +u; source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate pr_train
[ -f "$HOME/.secrets/hf.env" ] && { source "$HOME/.secrets/hf.env"; export HF_TOKEN; echo "using HF token from ~/.secrets/hf.env"; }
set -u
WHICH=${1:-v2}
echo "node=$(hostname) job=$SLURM_JOB_ID start=$(date -Is) which=$WHICH"
WHICH=$WHICH python - <<'PY'
import json, os
from huggingface_hub import HfApi, snapshot_download
out = {}
which = os.environ["WHICH"]
for repo, local in [r for r in [("facebook/dinov2-small", "models/dinov2-small"), ("facebook/dinov2-base", "models/dinov2-base"),
                    ("facebook/dinov3-vits16-pretrain-lvd1689m", "models/dinov3-vits16"),
                    ("facebook/dinov3-vitb16-pretrain-lvd1689m", "models/dinov3-vitb16")]
                    if which == "all" or which in r[0]]:
    try:
        sha = HfApi().model_info(repo).sha
        snapshot_download(repo, local_dir=local, revision=sha, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.md"])
        out[repo] = {"local": local, "commit": sha}
    except Exception as e:  # noqa: BLE001
        out[repo] = {"error": f"{type(e).__name__}: {e}"[:300]}
print(json.dumps(out, indent=2))
prev = json.load(open("models/dino_versions.json")) if os.path.exists("models/dino_versions.json") else {}
open("models/dino_versions.json", "w").write(json.dumps(prev | out, indent=2))
PY
echo "done=$(date -Is)"
