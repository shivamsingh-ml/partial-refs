#!/usr/bin/env bash
#SBATCH --job-name=pr-seg-prep
#SBATCH --partition=cpu
#SBATCH --account=uzk
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
# nnU-Net fingerprint, plans, folds and preprocessing for Dataset713 (424 training scans), V2V2R's recipe
# (2026-11-17). No GPU, no training.
#   STAGE=plans      sbatch --export=ALL jobs/seg_preprocess.sbatch
#       nnU-Net fingerprint on our scans (+ its dataset integrity check), plans = V2V2R's plans with only the dataset
#       name and intensity values changed (partialref.baseline.seg_plans), splits_final.json from split v1, then a
#       3-scan preprocessing test in memory (nothing written to the preprocessed data folder).
#   STAGE=preprocess sbatch --export=ALL jobs/seg_preprocess.sbatch
#       nnUNetv2_preprocess as in V2V2R's readme (configuration ps128x256x256; the ps160x288x288 training
#       configuration inherits the same data identifier), then a file count check.
set -euo pipefail
PROJECT_DIR=/msc/home/ssingh54/partial-refs; cd "$PROJECT_DIR"
STAGE=${STAGE:?STAGE=plans|preprocess}
DS=Dataset713_TF3SinglePulpDev; DS_ID=713; PLANS=nnUNetResEncUNetLPlans_torchres
SRC_PLANS=/msc/home/ssingh54/U-Mamba2/nnunet/nnUNet_preprocessed/Dataset703_ToothFairy3SinglePulp/$PLANS.json
SHA=$(git rev-parse --short HEAD)$([ -n "$(git status --porcelain -- src jobs configs)" ] && echo "-dirty" || true)
RUN=$PROJECT_DIR/results/seg/$(date +%Y%m%d-%H%M)-${SHA}-seg-prep-${STAGE}-${SLURM_JOB_ID}
mkdir -p "$RUN"
export nnUNet_raw=$PROJECT_DIR/data/nnunet/raw nnUNet_preprocessed=$PROJECT_DIR/data/nnunet/preprocessed
export nnUNet_results=$PROJECT_DIR/data/nnunet/results OMP_NUM_THREADS=1 PYTHONHASHSEED=0
PR_PY=$HOME/miniconda3/envs/cbct_partialref/bin/python
set +u; source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate umamba2; set -u
echo "node=$(hostname) job=$SLURM_JOB_ID stage=$STAGE run=$RUN start=$(date -Is)"
python -c "import nnunetv2; print('nnunetv2 from', nnunetv2.__file__)"
PRE=$nnUNet_preprocessed/$DS

if [ "$STAGE" = plans ]; then
  [ -e "$PRE/$PLANS.json" ] && { echo "plans already exist: $PRE/$PLANS.json"; exit 1; }
  nnUNetv2_extract_fingerprint -d $DS_ID -np 8 --verify_dataset_integrity
  "$PR_PY" -m partialref.baseline.seg_plans --source-plans "$SRC_PLANS" --fingerprint "$PRE/dataset_fingerprint.json" \
    --dataset $DS --splits configs/splits/v1 --fixes configs/seg/label_fixes_v1.json --raw "$nnUNet_raw/$DS" \
    --out-dir "$PRE" --report "$RUN/plans_report.json"
  cp "$PRE/dataset_fingerprint.json" "$PRE/splits_final.json" "$PRE/dataset.json" "$RUN/"
  undo() { rm -f "$PRE/$PLANS.json" "$PRE/splits_final.json"; echo "3-scan test failed: plans and folds removed, rerun after fixing"; }
  if ! python - "$PRE/$PLANS.json" "$nnUNet_raw/$DS" <<'PY' | tee "$RUN/preprocess_test.txt"; then
import sys, json, numpy as np, SimpleITK as sitk
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
plans, raw = sys.argv[1], sys.argv[2]
pm = PlansManager(plans)
dj = json.load(open(f"{raw}/dataset.json"))
ip = pm.foreground_intensity_properties_per_channel["0"]
lo, hi = (ip["percentile_00_5"] - ip["mean"]) / ip["std"], (ip["percentile_99_5"] - ip["mean"]) / ip["std"]
for cfg in ("3d_fullres_torchres_mambabot2_ps128x256x256_bs1", "3d_fullres_torchres_mambabot2_ps160x288x288_bs1"):
    cm = pm.get_configuration(cfg)
    print(cfg, "data_identifier", cm.data_identifier, "normalization", cm.normalization_schemes,
          "spacing", cm.spacing, "patch", cm.patch_size, "network", cm.network_arch_class_name)
cm = pm.get_configuration("3d_fullres_torchres_mambabot2_ps128x256x256_bs1")
for case in ("ToothFairy3P_001", "ToothFairy3F_006", "ToothFairy3S_0048"):
    img, lab = f"{raw}/imagesTr/{case}_0000.nii.gz", f"{raw}/labelsTr/{case}.nii.gz"
    data, seg, props = DefaultPreprocessor(verbose=False).run_case([img], lab, pm, cm, dj)
    raw_shape = tuple(int(x) for x in props["shape_before_cropping"])   # as read, before crop_to_nonzero
    lab_shape = sitk.GetArrayFromImage(sitk.ReadImage(lab)).shape
    d = data[0]
    print(json.dumps({"case": case, "raw_shape": list(raw_shape), "preprocessed_shape": list(d.shape),
                      "seg_shape": list(seg.shape[1:]), "spacing": [float(x) for x in props["spacing"]],
                      "data_min": round(float(d.min()), 3), "data_max": round(float(d.max()), 3),
                      "expected_clip_range": [round(lo, 3), round(hi, 3)],
                      "foreground_mean": round(float(d[seg[0] > 0].mean()), 3),
                      "seg_labels": sorted(int(v) for v in np.unique(seg)),
                      "label_file_shape_equal": list(lab_shape) == list(raw_shape),
                      "cropped": list(d.shape) != list(raw_shape),
                      "same_shape": list(lab_shape) == list(raw_shape) and list(d.shape) == list(seg.shape[1:])}))
PY
    undo; exit 1
  fi
  if grep -q '"same_shape": false' "$RUN/preprocess_test.txt"; then undo; exit 1; fi
elif [ "$STAGE" = preprocess ]; then
  [ -e "$PRE/$PLANS.json" ] || { echo "run STAGE=plans first"; exit 1; }
  nnUNetv2_preprocess -d $DS_ID -c 3d_fullres_torchres_mambabot2_ps128x256x256_bs1 -plans_name $PLANS -np 8
  N_RAW=$(ls "$nnUNet_raw/$DS/labelsTr" | wc -l)
  N_PKL=$(ls "$PRE/nnUNetPlans_3d_fullres_torchres_ctnorm"/*.pkl | wc -l)
  echo "raw cases $N_RAW, preprocessed cases $N_PKL" | tee "$RUN/counts.txt"
  du -sh "$PRE/nnUNetPlans_3d_fullres_torchres_ctnorm" | tee -a "$RUN/counts.txt"
  [ "$N_RAW" = "$N_PKL" ] || { echo "case count mismatch"; exit 1; }
fi
echo "done=$(date -Is)"
