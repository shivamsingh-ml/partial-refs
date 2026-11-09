#!/usr/bin/env bash
#SBATCH --job-name=pr-review
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --account=uzk
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
# Report review app (partialref.review.app) on a COMPUTE node (never on the login node).
# Report text is only shown in your browser through an SSH tunnel; nothing is saved unless you enable
# logging in the app (IDs and codes only, results/review/). The access token goes to an owner-only file,
# not to this log (the home directory is group-readable).
#   sbatch jobs/review_app.sbatch results/extract/<parse run> [en|it]
#   then read logs/pr-review-<jobid>.out for the tunnel command; stop with scancel <jobid> when done.
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: sbatch jobs/review_app.sbatch <parse run dir> [en|it]" >&2; exit 2; }
source /msc/home/ssingh54/partial-refs/jobs/env.sh
cd "$PROJECT_DIR"
[ -d "$1/per_report" ] || { echo "no per_report/ in $1" >&2; exit 1; }
export REVIEW_PARSE_RUN=$1 REVIEW_LANG=${2:-en}
export REVIEW_MANIFEST=results/manifest/20261107-1830-c2aa718-manifest REVIEW_LOG_ROOT=results/review
TOKEN_FILE=$HOME/.secrets/review_token
mkdir -p "$(dirname "$TOKEN_FILE")"; chmod 700 "$(dirname "$TOKEN_FILE")"
( umask 077; python -c "import secrets; print(secrets.token_urlsafe(18))" > "$TOKEN_FILE" )
export REVIEW_TOKEN=$(cat "$TOKEN_FILE")
PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
NODE=$(hostname)
cat <<EOF
node=$NODE job=${SLURM_JOB_ID:-none} start=$(date -Is) git=$(git rev-parse --short HEAD) parse=$REVIEW_PARSE_RUN lang=$REVIEW_LANG
---------------------------------------------------------------------------------------------
On YOUR laptop (new terminal), with the same ssh login you always use for the cluster:
    ssh -N -L 8501:${NODE}:${PORT} <your usual cluster login, e.g. ssingh54@<login host>>
Then open  http://localhost:8501  and paste the token from:   cat $TOKEN_FILE
Stop the app when done:  scancel ${SLURM_JOB_ID:-<jobid>}
---------------------------------------------------------------------------------------------
EOF
exec streamlit run src/partialref/review/app.py --server.address 0.0.0.0 --server.port "$PORT" \
  --server.headless true --browser.gatherUsageStats false --server.enableXsrfProtection true \
  --server.fileWatcherType none
