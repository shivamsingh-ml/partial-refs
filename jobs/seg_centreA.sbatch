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
# run_segmentation already returns the predictor's FDI scheme (task1_inference.BasePredictor.postprocessing:
# teeth 11-48, incisive/lingual canals 51-53, merged pulp 50). We map that to ToothFairy3 ids by composing
# the predictor's own sequential->FDI rule with vqa/schema/tf3_single_pulp_label_map.json (train_to_original):
# teeth unchanged, 51/52/53 -> 103/104/105, pulp 50 -> 199. (v1 of this job applied train_to_original to the
# already-FDI mask: tooth IDs scrambled, 47/48 and canals lost -> DECISIONS 2026-11-09; v1 masks are invalid.)
# MIG slice is fine here (no vLLM). Resumable: existing masks are skipped.
#   sbatch jobs/seg_centreA.sbatch [LIMIT]       (LIMIT=3 for the diagnostic run)
set -euo pipefail
PROJECT_DIR=/msc/home/ssingh54/partial-refs; cd "$PROJECT_DIR"
LIMIT=${1:-0}
APP=$PROJECT_DIR/external/odin2026
SEG_MODEL=/msc/home/ssingh54/U-Mamba2/nnunet/nnUNet_results/Dataset703_ToothFairy3SinglePulp/nnUNetTrainer_TF3_Task1_1500ep_accum2__nnUNetResEncUNetLPlans_torchres__3d_fullres_torchres_mambabot2_ps160x288x288_bs1
OUT=$PROJECT_DIR/results/seg/centreA_umamba2_foldall_v2   # stable dir so reruns resume (v1 = invalid labels)
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
# predictor's sequential -> FDI rule, same formula as task1_inference.BasePredictor (orig_mapping)
seq_to_fdi = {i: i for i in range(47)}
seq_to_fdi.update({43: 51, 44: 52, 45: 53, 46: 50})
seq_to_fdi.update({i: i + 2 for i in range(19, 27)})
seq_to_fdi.update({i: i + 4 for i in range(27, 35)})
seq_to_fdi.update({i: i + 6 for i in range(35, 43)})
assert len(set(seq_to_fdi.values())) == 47, "predictor mapping not one-to-one"
lut = np.full(256, 255, np.uint8)                      # 255 = unexpected label -> fail below
for k, v in lm.items():
    lut[seq_to_fdi[int(k)]] = 199 if v == "COMBINED_PULP" else int(v)
assert all(lut[t] == t for t in [11, 18, 21, 28, 31, 38, 41, 48]) and tuple(lut[[51, 52, 53, 50]]) == (103, 104, 105, 199)
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
    mask = run_segmentation(img)                       # input space, predictor FDI scheme
    raw = sitk.GetArrayFromImage(mask).astype(np.int64)
    assert raw.max() < 256 and not (lut[np.unique(raw)] == 255).any(), f"{r.case_id}: unexpected labels {np.unique(raw)}"
    arr = lut[raw]
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
