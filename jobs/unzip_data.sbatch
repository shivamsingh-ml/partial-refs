#!/usr/bin/env bash
#SBATCH --job-name=pr-unzip-tf4
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --account=uzk
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
# Unpack ToothFairy4.zip into ~/Datasets/ToothFairy4/ (never overwrites), check every file's CRC
# while extracting, and record the zip checksum. Prints counts only; no report text is read.
set -euo pipefail
ZIP=/msc/home/ssingh54/Datasets/ToothFairy4.zip
DEST=/msc/home/ssingh54/Datasets/ToothFairy4
INFO=/msc/home/ssingh54/partial-refs/results/env/data_${SLURM_JOB_ID}; mkdir -p "$INFO"
echo "node=$(hostname) job=$SLURM_JOB_ID start=$(date -Is)"; df -h "$(dirname "$ZIP")" | tail -1
[ -e "$DEST" ] && { echo "DEST exists, refusing to extract over it: $DEST" >&2; exit 1; }
# zip sha256 recorded by job 614550: 40dbbf2dbc00547269594062b91059363c941151c8bece668ee9db8c18cfa073
mkdir -p "$DEST"
# `unzip` is not installed on compute nodes (job 614550) -> Python zipfile: it checks each member's
# CRC while reading and rejects absolute / ".." paths on extract.
set +u; source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate cbct_partialref; set -u
python - "$ZIP" "$DEST" <<'PY'
import sys, zipfile
zf = zipfile.ZipFile(sys.argv[1])
names = zf.namelist()
bad = [n for n in names if n.startswith("/") or ".." in n.split("/")]
if bad: sys.exit(f"unsafe paths in zip: {len(bad)}")
zf.extractall(sys.argv[2])
print(f"extracted {len(names)} entries")
PY
echo "files: $(find "$DEST" -type f | wc -l) (expected 2622)"
echo "cases: $(find "$DEST" -mindepth 1 -maxdepth 1 -type d | wc -l) (expected 622)"
echo "volumes: $(find "$DEST" -path '*/cbct/volume.nii.gz' | wc -l)  en: $(find "$DEST" -path '*/reports_en/*.txt' | wc -l)  it: $(find "$DEST" -path '*/reports_it/*.txt' | wc -l)"
chmod -R go-rwx "$DEST"                # DUA: only the owner can read the data
du -sh --apparent-size "$DEST"; df -h "$DEST" | tail -1
echo "done=$(date -Is)"
