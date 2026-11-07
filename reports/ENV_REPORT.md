# ENV_REPORT — MSC first-session checklist (HANDOFF §4)

Date: 2026-11-07. Read-only inspection plus two test jobs (614514, 614515).
No installs, no data read: only file/directory names were listed. Raw job logs:
`logs/pr-smoke-net-614514.out`, `logs/pr-smoke-gpu-614515.out` (gitignored).

## 1. Identity

| Check | Result |
| --- | --- |
| hostname | `s0-l03` (login node) |
| user | `ssingh54` |
| date | Sat Nov 7 17:03 CET 2026 |

## 2. Filesystem

| Check | Result |
| --- | --- |
| Home | `/msc/home/ssingh54` → resolves to `/muwscs01/legacy/msc/home/ssingh54` |
| Filesystem | GPFS `muwscs01`, 250 TB, **248 TB used, 2.8 TB free (99 %)**, shared by all users |
| Quota | `quota` not installed; `mmlsquota`: "No quota enabled file system found" → no per-user quota visible; the limit is the shared pool |
| Backups | None (HANDOFF §0.9) |
| `~/projects/` | does not exist |
| This repo | `~/partial-refs` (where HANDOFF.md was placed) — **location needs confirmation** |

Risk: with 2.8 TB free on a shared FS, large outputs (segmentation probabilities, vLLM caches, SIF
copies) must be kept small and short-lived (HANDOFF §12.2, §12.7).

## 3. SLURM

Account: **`uzk`** (only association). Partitions:

| Partition | Nodes | Time limit | Notes |
| --- | --- | --- | --- |
| `cpu` (default) | 6 (`s0-n[22-27]`), 256 CPUs / ~2 TB RAM each | none on partition; QoS `cpu` max 3 days | DefMemPerCPU 8 GB |
| `gpu` | 7 (`s0-n[00-03,11-12,15]`) | partition default 8 h, QoS max below | DefMemPerCPU 8 GB |
| `storage` | 2 (`s0-s[00-01]`) | — | 8 CPUs, small; for transfers presumably |

GPU job pattern (from Shivam's existing jobs, e.g. `~/U-Mamba2/jobs/job_tf4_all.sh`): the QoS
must match the GRES, e.g. `--partition=gpu --qos=h100 --gres=gpu:h100:1`.

| GRES / QoS | GPUs in cluster | QoS max wall | Max jobs/user (as listed) |
| --- | --- | --- | --- |
| `gpu:a100:N` / `a100` (A100-PCIE-40GB) | 21 | 3-00:00:00 | 4 |
| `a100_long` | (same A100s) | 5-00:00:00 | 2 |
| `gpu:a100-sxm4-80gb:N` / `a100-sxm4-80gb` | 8 | 3-00:00:00 | 4 |
| `gpu:h100:N` / `h100` | 4 | 3-00:00:00 | 2 |
| `gpu:b200:N` / `b200` | 8 | 3-00:00:00 | 2 |
| `gpu:a30:N` / `a30` | 1 | 3-00:00:00 | — |
| `gpu:a16:N` / `a16` | 8 | 3-00:00:00 | 4 |
| `gpu:3g.20gb:N` / `3g.20gb` (A100 MIG) | 2 | 3-00:00:00 | 2 |
| `gpu:1g.5gb:N` / `1g.5gb` (A100 MIG) | 7 | 08:00:00 | 4 |
| `cpu` | — | 3-00:00:00 | 8 |

(The last column is `MaxJobsPU` read from `sacctmgr show qos`; column alignment was ambiguous —
treat as indicative.)

## 4. Modules / containers / conda

| Check | Result |
| --- | --- |
| `module` | **not available** (no Lmod/Environment Modules) |
| apptainer / singularity | `/usr/bin/apptainer`, `/usr/bin/singularity` |
| conda | `~/miniconda3/condabin/conda`; init script used by Shivam's jobs: `~/miniconda3/etc/profile.d/conda.sh` |
| mamba / micromamba | not found |
| conda envs | base, cbctvlm-core, cbctvlm-m3d, cbctvlm-med3dvlm, cbctvlm-medgemma, cbctvlm-radfm, jawfm, provenance, tmd-trace, umamba2 |
| `cbct_base` (V2V2R) / `cbct_partialref` | **neither exists** |

## 5. Existing assets

| Asset | Result |
| --- | --- |
| ToothFairy4 under `~/Datasets/` | **not found** — `~/Datasets/` does not exist, although `~/U-Mamba2/jobs/job_tf4_all.sh` still sets `TF4=$HOME/Datasets` |
| ToothFairy4 elsewhere in `~` (depth 5) | no directory named `*toothfairy4*` found |
| V2V2R checkout | **not found** in `~` (depth 5) |
| V2V2R SIF (`*.sif`) | **not found** in `~` (depth 5) |
| Model `Qwen3.5-9B-AWQ-dental-cbct-sft` | **not found**; only the base model in HF cache: `~/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B` |
| U-Mamba2 | `~/U-Mamba2/` present; env script `~/U-Mamba2/jobs/env.sh` (conda env `umamba2`, `NNUNET_BASE=~/U-Mamba2/nnunet`) |
| Segmentation weights (candidates, not verified) | `~/U-Mamba2/checkpoints/checkpoint_final.pth`; model dir referenced by `job_tf4_all.sh`: `$nnUNet_results/Dataset703_ToothFairy3SinglePulp/nnUNetTrainer_TF3_Task1_1500ep_accum2__nnUNetResEncUNetLPlans_torchres__3d_fullres_torchres_mambabot2_ps160x288x288_bs1` |
| Docker image `ssingh54/toothfairy4-v2v2r` | not checked (needs tag + Docker Hub login) |

Cases/reports were not counted because the dataset was not found.

### 5.1 Container `ssingh54/toothfairy4-v2v2r:odin2026-task1` (registry metadata only, not pulled)

Read from Docker Hub manifest, config and BuildKit SLSA provenance (saved off-repo in the session
scratchpad):

| Field | Value |
| --- | --- |
| Index digest (amd64 image) | `sha256:b4abf81185e7e315d5b1f43a12a235d34b279b6b8c1150099b76c998799eb571` |
| Size | 35 layers, 14.29 GB compressed |
| Built | 2026-08-17T11:42Z |
| Base | `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel` |
| **Source (provenance)** | `https://github.com/shivamsingh-ml/tf-submission.git`, context `algorithm/`, **revision `d9c27a85f1e1f18ef47093778eb75d74c3e8a279`** |
| Workdir / entrypoint | `/opt/app`, `python inference.py` (Grand Challenge algorithm layout) |
| Code in image | `inference.py task1_inference.py config.py orientation.py segmentation.py facts.py extract_facts.py report_side.py`, `umamba2/`, `vqa/code/`, `vqa/schema/` |
| VLM runtime | separate venv `/opt/vlmenv` (`requirements-vlm.txt`, CUDA 12.x torch + vLLM) |
| Compile caches | `/opt/vllm_cache`, `/opt/inductor_cache`, `/opt/triton_cache` (env vars point there) |
| Runtime env | `HF_HUB_OFFLINE=1`, `VLM_MAX_MODEL_LEN=32768`, `VLM_GPU_UTIL=0.90`, `VLM_DTYPE=float16`, `VLM_PREFIX_CACHING=1` |

Implication: the submission was built from `tf-submission`, not from the public `v2v2R` repo
(whose HEAD is dated 2026-10-07). The submodule pin question becomes "which `v2v2R` commit matches
`tf-submission@d9c27a8:algorithm/vqa/{code,schema}`".

## 6. Network

| Host | Login node `s0-l03` | Compute node `s0-n25` (job 614514) |
| --- | --- | --- |
| github.com | 200 | 200 |
| huggingface.co | 200 | 200 |
| registry-1.docker.io/v2/ | 401 (reachable; auth required) | 401 (reachable; auth required) |

Compute nodes have outbound internet, so `HF_HUB_OFFLINE` is not forced; still pre-download for
reproducibility.

## 7. GPU smoke test (job 614515, 14 s, node `s0-n12`)

| Check | Result |
| --- | --- |
| GPU | NVIDIA A100-PCIE-40GB, MIG slice `1g.5gb` (4.8 GB) |
| Driver | 580.178.04 (CUDA 13.0 driver API) |
| torch | **conda env `umamba2`** (fallback; no SIF available): Python 3.11.15, torch 2.5.1+cu124, `cuda_available True` |
| Inside container | **not tested** — no SIF found (HANDOFF §5.4 `apptainer inspect` / `pip freeze` still to do) |

## 8. `<<FILL>>` values

### Answered by Shivam (2026-11-07)

| Item | Value |
| --- | --- |
| Project root | `~/partial-refs` (`/msc/home/ssingh54/partial-refs`); `results/` and `.cache/` stay inside it |
| Container | Docker `ssingh54/toothfairy4-v2v2r:odin2026-task1` → to be pulled as SIF (Plan B, HANDOFF §5.4); SIF path set after the pull |
| Released model | pull HF `lucent517/v2v2r_cbct_vqa` into `$HF_HOME` |
| GitHub | personal account `shivamsingh-ml`, private repo `partial-refs`; SSH key on the cluster already authenticates to GitHub |
| DUA / external APIs | not allowed (default; nothing is sent externally) |
| Reader-study size | decided after M1 |
| GPU type | start with the smallest GPU that fits, scale up only if needed |

### Observed, awaiting confirmation

| Item | Observed value |
| --- | --- |
| GPU partition / account | `gpu` / `uzk`, with `--qos` = GRES type |
| Max walltime | 3 days (QoS), 5 days with `a100_long`; `1g.5gb` 8 h |
| CPU partition | `cpu`, QoS `cpu`, max 3 days |
| Conda init path | `~/miniconda3/etc/profile.d/conda.sh` (used in Shivam's U-Mamba2 jobs) |
| Quota command | none — GPFS quota not enabled; FS 99 % full |

### Still open

1. **ToothFairy4 path + version** — Shivam uploads to `~/Datasets/ToothFairy4/` (agreed); version still to record.
2. ~~Which code defines "V2V2R"~~ — resolved: own splits, public `v2v2R` used as a toolkit (DECISIONS.md).
3. **Segmentation weights** — `~/U-Mamba2/checkpoints/checkpoint_final.pth` vs the `Dataset703_…` nnU-Net results folder.
4. ~~40-case held-out list~~ — not needed (own splits).
5. ~~SMB backup path~~ — resolved: lab PC pulls weekly with rsync (README.md).
