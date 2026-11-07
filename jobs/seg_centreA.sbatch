#!/usr/bin/env bash
#SBATCH --job-name=pr-seg-A
#SBATCH --partition=gpu
#SBATCH --qos=3g.20gb
#SBATCH --account=uzk
#SBATCH --gres=gpu:3g.20gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
# Segmentation-only masks for centre A (Sfax), which has no hand-drawn labels. Used ONLY for
# visibility features (HANDOFF §2.6), never as model input for evaluation.
# Calls the submitted pipeline's own segmentation entry point (external/odin2026/segmentation.py:
# run_segmentation -> auto-orientation to the model's RPI frame, mask returned in TF4 input space),
# i.e. the old U-Mamba2 `fold_all` model (trained on ToothFairy3 = P/F/S only; A was never seen).
# Labels are mapped back to ToothFairy3 ids via the submission's own
# vqa/schema/tf3_single_pulp_label_map.json (train_to_original); merged pulp -> 199.
# MIG slice is fine here (no vLLM). Resumable: existing masks are skipped.
#   sbatch jobs/seg_centreA.sbatch [LIMIT]       (LIMIT=3 for the diagnostic run)
set -euo pipefail
PROJECT_DIR=/msc/home/ssingh54/partial-refs; cd "$PROJECT_DIR"
LIMIT=${1:-0}
APP=$PROJECT_DIR/external/odin2026
SEG_MODEL=/msc/home/ssingh54/U-Mamba2/nnunet/nnUNet_results/Dataset703_ToothFairy3SinglePulp/nnUNetTrainer_TF3_Task1_1500ep_accum2__nnUNetResEncUNetLPlans_torchres__3d_fullres_torchres_mambabot2_ps160x288x288_bs1
OUT=$PROJECT_DIR/results/seg/centreA_umamba2_foldall      # stable dir so reruns resume
ROOT=$OUT/model_root; mkdir -p "$ROOT"; ln -sfn "$SEG_MODEL" "$ROOT/nnunet_seg"
export MODEL_ROOT=$ROOT NNUNET_SUBDIR=nnunet_seg PYTHONHASHSEED=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "node=$(hostname) job=$SLURM_JOB_ID start=$(date -Is) git=$(git rev-parse --short HEAD) limit=$LIMIT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
set +u; source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate umamba2; set -u
cd "$APP"
python - "$OUT" "$LIMIT" <<'PY'
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, SimpleITK as sitk
from segmentation import run_segmentation          # submitted pipeline entry point (cwd = external/odin2026)

out, limit = Path(sys.argv[1]), int(sys.argv[2])
(out / "masks").mkdir(parents=True, exist_ok=True)
lm = json.load(open("vqa/schema/tf3_single_pulp_label_map.json"))["train_to_original"]
lut = np.zeros(256, np.uint8)
for k, v in lm.items():
    lut[int(k)] = 199 if v == "COMBINED_PULP" else int(v)
cases = pd.read_parquet("/msc/home/ssingh54/partial-refs/results/manifest/20261107-1830-c2aa718-manifest/cases.parquet")
cases = cases[cases.centre == "A"].sort_values("case_id")
if limit:
    cases = cases.head(limit)
log = []
for r in cases.itertuples():
    dst = out / "masks" / f"{r.case_id}.nii.gz"
    if dst.exists():
        continue
    t0 = time.time()
    img = sitk.ReadImage(r.volume_path)
    mask = run_segmentation(img)                       # input space, sequential train labels
    arr = lut[sitk.GetArrayFromImage(mask).astype(np.int64)]
    m2 = sitk.GetImageFromArray(arr); m2.CopyInformation(img)
    sitk.WriteImage(m2, str(dst), useCompression=True)
    n_teeth = len({int(v) for v in np.unique(arr) if 11 <= v <= 48})
    rec = {"case_id": r.case_id, "seconds": round(time.time() - t0, 1), "n_labels": int(len(np.unique(arr))),
           "n_teeth_labels": n_teeth, "jaws": [int(j) for j in (1, 2) if (arr == j).any()]}
    log.append(rec); print("CASE", json.dumps(rec), flush=True)
pd.DataFrame(log).to_csv(out / f"log_{int(time.time())}.csv", index=False)
print("DONE", len(log), "new masks;", len(list((out / "masks").glob("*.nii.gz"))), "total")
PY
echo "done=$(date -Is)"
