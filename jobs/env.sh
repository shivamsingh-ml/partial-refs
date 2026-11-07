#!/usr/bin/env bash
# Sourced by every job (HANDOFF §5.3). Unresolved <<FILL>> values abort loudly; never guess them.
set +u
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"   # confirmed 2026-11-07
[ -n "$CONDA_SH" ] || { echo "env.sh: CONDA_SH not set (<<FILL>>)" >&2; return 1 2>/dev/null || exit 1; }
source "$CONDA_SH"
conda activate cbct_partialref
set -u
export PROJECT_DIR="${PROJECT_DIR:-/msc/home/ssingh54/partial-refs}"   # confirmed 2026-11-07
export DATA_DIR="${DATA_DIR:-/msc/home/ssingh54/Datasets/ToothFairy4}"   # agreed 2026-11-07; verify layout on upload
export SIF_PATH="${SIF_PATH:-}"   # <<FILL>>; required only by GPU jobs, which check it themselves
export HF_HOME=$PROJECT_DIR/.cache/hf
export XDG_CACHE_HOME=$PROJECT_DIR/.cache
export PIP_CACHE_DIR=$PROJECT_DIR/.cache/pip
export APPTAINER_CACHEDIR=$PROJECT_DIR/.cache/apptainer
export VLLM_CACHE_ROOT=$PROJECT_DIR/.cache/vllm
export NLTK_DATA=$PROJECT_DIR/.cache/nltk_data    # METEOR wordnet (HANDOFF §5.2)
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
# secrets: only if a step needs them and Shivam approved it
# set +u; source ~/.secrets/hf.env; set -u
