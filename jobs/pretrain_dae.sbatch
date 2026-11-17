#!/usr/bin/env bash
#SBATCH --job-name=pr-pretrain-dae
#SBATCH --partition=gpu
#SBATCH --qos=a100-sxm4-80gb
#SBATCH --account=uzk
#SBATCH --gres=gpu:a100-sxm4-80gb:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
# Repeat U-Mamba2's self-supervised (DAE) pretraining on our 424 training scans only (2026-11-17). Their code in
# external/umamba2_pretrain_dae (imported verbatim, then: our scan list, our intensity values), their readme command
# (documentation/competitions/Toothfairy3/readme.md) with our plans; environment ~/envs/pr_dae (env/pr_dae.lock.txt).
#   MODE=diag sbatch --export=ALL jobs/pretrain_dae.sbatch
#       2 rounds into data/pretrain/diag_<job>: checks that their loader orients every scan as nnU-Net does (one scan
#       per centre), runs 2 rounds, converts the round-1 checkpoint and loads it into our segmentation network.
#   MODE=full sbatch --export=ALL --time=3-00:00:00 jobs/pretrain_dae.sbatch
#       ${EPOCHS:-200} rounds into data/pretrain/dae_v1; when resubmitted, passes --resume (their only resume switch) if a
#       ckpt_epoch_*.pth exists there. Final weights = the last-round checkpoint (DECISIONS 2026-11-17).
# 80 GB A100: batch 2 at 160x288x288 in float32 (their recipe, no AMP) needs an estimated 50-55 GB (reviews 2026-11-17).
set -euo pipefail
PROJECT_DIR=/msc/home/ssingh54/partial-refs; cd "$PROJECT_DIR"
MODE=${MODE:?MODE=diag|full}
PY=$HOME/envs/pr_dae/bin/python
TF3=/msc/home/ssingh54/sae_tooth/data/raw/toothfairy3
PLANS=$PROJECT_DIR/data/nnunet/preprocessed/Dataset713_TF3SinglePulpDev/nnUNetResEncUNetLPlans_torchres.json
CONF=3d_fullres_torchres_mambabot2_ps160x288x288_bs1
REF=/msc/home/ssingh54/U-Mamba2/data/tf3_pretrained/ssl_umamba2_3d_depth7_160x288x288.pth
export PR_DAE_DATALIST=$PROJECT_DIR/data/pretrain/datalist_v1.json
export WANDB_MODE=disabled PYTHONHASHSEED=0 PYTHONPATH=$PROJECT_DIR/src
SHA=$(git rev-parse --short HEAD)$([ -n "$(git status --porcelain -- src jobs external configs)" ] && echo "-dirty" || true)
echo "node=$(hostname) job=$SLURM_JOB_ID mode=$MODE sha=$SHA start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
[ -f "$PLANS" ] || { echo "plans missing: $PLANS"; exit 1; }
if [ ! -f "$PR_DAE_DATALIST" ]; then
  "$PY" -m partialref.baseline.pretrain_datalist --splits configs/splits/v1 --fixes configs/seg/label_fixes_v1.json \
    --tf3-dir "$TF3" --out "$PR_DAE_DATALIST"
fi
if [ "$MODE" = diag ]; then OUT=$PROJECT_DIR/data/pretrain/diag_${SLURM_JOB_ID}; EPOCHS=2; SAVE=1
else OUT=$PROJECT_DIR/data/pretrain/dae_v1; EPOCHS=${EPOCHS:-200}; SAVE=5; fi
mkdir -p "$OUT"
echo "$SHA" > "$OUT/git_sha_${SLURM_JOB_ID}.txt"
case "$SHA" in *-dirty) [ "$MODE" = full ] && { echo "refusing a full run from a dirty tree"; exit 1; } ;; esac
sha256sum "$PR_DAE_DATALIST" | tee "$OUT/datalist_sha256_${SLURM_JOB_ID}.txt"
RESUME=""; ls "$OUT"/ckpt_epoch_*.pth >/dev/null 2>&1 && RESUME="--resume"
echo "resume flag: ${RESUME:-none}"

if [ "$MODE" = diag ]; then
  # their loader (MONAI LoadImaged -> Transposed(0,3,2,1)) must give the array nnU-Net reads with SimpleITK
  "$PY" - "$PR_DAE_DATALIST" <<'PY' | tee "$OUT/orientation_check.txt"
import json, sys, numpy as np, SimpleITK as sitk
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Transposed
d = json.load(open(sys.argv[1]))
rows = d["training"] + d["validation"]
picks = {}
for r in rows:
    picks.setdefault(r["case_id"][0], r)
t = Compose([LoadImaged(keys=["image"]), EnsureChannelFirstd(keys=["image"]), Transposed(keys=["image"], indices=(0, 3, 2, 1))])
ok = True
for c, r in sorted(picks.items()):
    a = np.asarray(t({"image": r["image"]})["image"])[0]
    b = sitk.GetArrayFromImage(sitk.ReadImage(r["image"]))
    same = a.shape == b.shape and np.array_equal(a.astype(np.float64), b.astype(np.float64))
    ok &= same
    print(json.dumps({"case": r["case_id"], "monai_shape": list(a.shape), "sitk_shape": list(b.shape), "equal": bool(same)}))
sys.exit(0 if ok else 1)
PY
fi

cd external/umamba2_pretrain_dae
"$PY" main_runner.py --batch_size=2 --sw_batch_size=1 --epoch=$EPOCHS --mask_patch_size=16 --loss_type=all_img \
  --base_lr=5e-5 --min_lr=5e-6 --warmpup_epoch=4 --warmup_lr=5e-7 --cache_dataset --cache_rate=0.2 \
  --model_type=nnunet --save_freq=$SAVE --log_dir="$OUT/logdir" --output="$OUT" --out_channels=1 --choice "all" \
  --mm_con 0.03 --temperature 0.5 --img_size 288 --roi_x 160 --roi_y 288 --roi_z 288 --mask_ratio 0.3 \
  --nnunet_plan "$PLANS" --nnunet_conf "$CONF" $RESUME
cd "$PROJECT_DIR"

if [ "$MODE" = diag ]; then
  CK=$(ls "$OUT"/ckpt_epoch_*.pth | sort | tail -1)
  "$PY" -m partialref.baseline.dae_to_nnunet --ckpt "$CK" --reference "$REF" \
    --out "$OUT/dae_nnunet_weights_diag.pth" --plans "$PLANS" --config "$CONF" \
    --dataset-json data/nnunet/preprocessed/Dataset713_TF3SinglePulpDev/dataset.json | tee "$OUT/convert_check.txt"
  grep -h -o -E "mem [0-9]+MB|EPOCH [0-9]+ training takes [0-9:]+" "$OUT"/log_rank*.txt 2>/dev/null | sort | uniq -c | tail -10 || true
fi
echo "done=$(date -Is)"
