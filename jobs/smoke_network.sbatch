#!/usr/bin/env bash
#SBATCH --job-name=pr-smoke-net
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --account=uzk
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:02:00
#SBATCH --output=logs/%x-%j.out
# HANDOFF §4.6: can compute nodes reach the internet? Headers only, 10 s timeout per host.
set -uo pipefail
echo "node=$(hostname) job=${SLURM_JOB_ID:-none} date=$(date -Is)"
for u in https://github.com https://huggingface.co https://registry-1.docker.io/v2/; do
  code=$(curl -sI --max-time 10 -o /dev/null -w "%{http_code}" "$u")
  echo "$code $u (exit=$?)"
done
