#!/usr/bin/env bash
#SBATCH --job-name=pr-seg-memcheck
#SBATCH --partition=gpu
#SBATCH --qos=a16
#SBATCH --account=uzk
#SBATCH --gres=gpu:a16:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out
# Does V2V2R's segmentation training (U-Mamba2, exact released trainer + configuration) fit in 16 GB?
# (2026-11-12, Shivam: retraining on a 16 GB desktop A4000 instead of the cluster.) Runs the real trainer's
# train_step (forward, deep-supervision loss, backward, accumulation 2, optimizer step) on RANDOM data of the
# training crop size 160x288x288, batch 1. Memory does not depend on image content. No dataset, no checkpoint
# written; nnUNet_results/preprocessed point to a scratch folder so the released model folder is untouched.
# Reports peak GPU memory and seconds per step. A16 speed is NOT the A4000's (A16 is slower); memory is.
#   sbatch jobs/seg_memory_check.sbatch
set -euo pipefail
PROJECT_DIR=/msc/home/ssingh54/partial-refs; cd "$PROJECT_DIR"
SRC=/msc/home/ssingh54/U-Mamba2/nnunet/nnUNet_results/Dataset703_ToothFairy3SinglePulp/nnUNetTrainer_TF3_Task1_1500ep_accum2__nnUNetResEncUNetLPlans_torchres__3d_fullres_torchres_mambabot2_ps160x288x288_bs1
SHA=$(git rev-parse --short HEAD)$([ -n "$(git status --porcelain -- jobs)" ] && echo "-dirty" || true)
RUN=$PROJECT_DIR/results/seg/$(date +%Y%m%d-%H%M)-${SHA}-memcheck-${SLURM_JOB_ID}
mkdir -p "$RUN/nnunet_results" "$RUN/nnunet_preprocessed"
export nnUNet_results=$RUN/nnunet_results nnUNet_preprocessed=$RUN/nnunet_preprocessed nnUNet_raw=$RUN
unset nnUNet_compile           # nnU-Net default = torch.compile on, as in the original training logs
source ~/miniconda3/etc/profile.d/conda.sh; conda activate umamba2
echo "node=$(hostname) job=$SLURM_JOB_ID run=$RUN"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python - "$SRC" "$RUN" <<'PY' 2>&1 | tee "$RUN/memcheck.txt"
import json, sys, time
import numpy as np, torch
import nnunetv2.training.nnUNetTrainer.nnUNetTrainer as base
from nnunetv2.training.nnUNetTrainer.variants.competitions.toothfairy3 import nnUNetTrainer_TF3_Task1_1500ep_accum2 as T

src, run = sys.argv[1], sys.argv[2]
base.infer_dataset_class = lambda *_: None          # no preprocessed data needed for a memory check
plans = json.load(open(f"{src}/plans.json"))
dsj = json.load(open(f"{src}/dataset.json"))
cfg = "3d_fullres_torchres_mambabot2_ps160x288x288_bs1"
tr = T(plans, cfg, "all", dsj, torch.device("cuda"))
tr.initialize()
ps = tr.configuration_manager.patch_size
bs = tr.configuration_manager.batch_size
nlab = tr.label_manager.num_segmentation_heads
print(f"trainer={T.__name__} config={cfg} patch={ps} batch={bs} classes={nlab} accumulate={tr.accumulate_steps}")
print(f"network={type(tr.network).__name__} params={sum(p.numel() for p in tr.network.parameters())/1e6:.1f}M")
scales = tr._get_deep_supervision_scales()
g = torch.Generator().manual_seed(0)
def batch():
    x = torch.randn((bs, 1, *ps), generator=g)
    t = [torch.randint(0, nlab, (bs, 1, *[int(round(p * s)) for p, s in zip(ps, sc)]), generator=g).float()
         for sc in scales]
    return {"data": x, "target": t}
torch.cuda.reset_peak_memory_stats()
times = []
for i in range(30):
    b = batch()
    torch.cuda.synchronize(); t0 = time.time()
    tr.train_step(b, step_optimizer=(i % tr.accumulate_steps == tr.accumulate_steps - 1))
    torch.cuda.synchronize(); times.append(time.time() - t0)
    if i in (0, 1, 9, 29):
        print(f"step {i+1}: {times[-1]:.2f} s, peak allocated {torch.cuda.max_memory_allocated()/2**30:.2f} GiB, "
              f"peak reserved {torch.cuda.max_memory_reserved()/2**30:.2f} GiB")
tot = torch.cuda.get_device_properties(0).total_memory / 2**30
res = {"gpu": torch.cuda.get_device_name(0), "gpu_total_gib": round(tot, 2),
       "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
       "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2),
       "median_s_per_step_after_warmup": round(float(np.median(times[5:])), 3),
       "steps_per_epoch": tr.num_iterations_per_epoch, "epochs": tr.num_epochs, "patch": list(ps), "batch": bs}
json.dump(res, open(f"{run}/memcheck.json", "w"), indent=2)
print(json.dumps(res))
PY
echo "done=$(date -Is)"
