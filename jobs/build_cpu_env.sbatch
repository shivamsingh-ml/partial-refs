#!/usr/bin/env bash
#SBATCH --job-name=pr-build-cpu-env
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --account=uzk
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# Build the cbct_partialref conda env (HANDOFF §5.2), download the METEOR wordnet data, freeze the
# resolved versions, and run the test suite once. Refuses to overwrite an existing env.
set -euo pipefail
PROJECT_DIR=/msc/home/ssingh54/partial-refs
cd "$PROJECT_DIR"
export CONDA_PKGS_DIRS=$PROJECT_DIR/.cache/conda_pkgs PIP_CACHE_DIR=$PROJECT_DIR/.cache/pip
export NLTK_DATA=$PROJECT_DIR/.cache/nltk_data
INFO=$PROJECT_DIR/results/env/cpu_env_${SLURM_JOB_ID}; mkdir -p "$INFO" "$NLTK_DATA"
set +u; source "$HOME/miniconda3/etc/profile.d/conda.sh"; set -u
echo "node=$(hostname) job=$SLURM_JOB_ID start=$(date -Is) conda=$(conda --version)"
if conda env list | awk '{print $1}' | grep -qx cbct_partialref; then
  echo "cbct_partialref already exists; not overwriting" >&2; exit 1
fi
conda env create -f env/environment.yml
set +u; conda activate cbct_partialref; set -u
python -m nltk.downloader -d "$NLTK_DATA" wordnet omw-1.4
pip freeze --exclude-editable > env/requirements-cpu.lock.txt
conda env export > "$INFO/conda_env_export.yml"
python - <<'PY'
import importlib
for m in ["numpy", "scipy", "pandas", "pyarrow", "sklearn", "statsmodels", "numpyro", "jax",
          "rapidfuzz", "datasketch", "SimpleITK", "nibabel", "nltk", "partialref"]:
    mod = importlib.import_module(m)
    print(f"{m:12s} {getattr(mod, '__version__', 'ok')}")
from importlib.metadata import version, distribution
print("radfact-lite", version("radfact-lite"), distribution("radfact-lite").read_text("direct_url.json"))
from nltk.corpus import wordnet
print("wordnet synsets:", len(list(wordnet.all_synsets("n"))) > 0)
PY
python -m pytest -q -p no:cacheprovider
echo "done=$(date -Is)"
