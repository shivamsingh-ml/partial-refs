# HANDOFF — "When the Reference Is Partial" (MedIA project)

Agent handoff for setting up and running the project on the MedUni Vienna HPC cluster (MSC).
Read this whole file before doing anything. Values marked `<<FILL>>` are unknown and must be
confirmed with Shivam before use. Never guess them.

Proposal doc (human-readable version): https://claude.ai/artifact/NWrA3qS1Wh3S6KwtB5jGRj

**Revision 2026-11-07 (Shivam):** we use **our own splits** (locked test set + 5-fold CV + LOCO,
§6.4) and **retrain V2V2R's LoRA per fold**; the released V2V2R weights and its 40-case held-out
list are no longer used for any reported evaluation. V2V2R is used as a toolkit, not reproduced.
**Toolkit = only the code actually submitted to the challenge**, recovered verbatim from the
Docker image `odin2026-task1` into the folder `external/odin2026/` (see its `PROVENANCE.md`).
No submodules; the public V2V2R repo is not part of this project. **No container:** GPU work runs in conda envs pinned to the image's versions
(e.g. `vllm==0.19.0`); the SIF route (§5.4) is a fallback.
Affected: §0.4, §1, §3, §5.4, §6.4, M0, M1, M3, M4, §10, §13. See DECISIONS.md (2026-11-07).

---

## 0. How to work on this project (operating rules)

These rules override any default habit.

1. **Gates are hard stops.** Each milestone ends in a gate (G0–G5). At a gate, write
   `reports/REPORT_Mx.md` with the numbers, then stop and wait for Shivam's decision.
2. **Shivam reviews code before it runs.** Before submitting any GPU job, or any CPU job longer
   than ~30 min, show the diff/script and the exact `sbatch` command, then wait for approval.
3. **Cheap diagnostic first.** Every pipeline runs on 3–5 cases (`LIMIT`/`CASE_IDS`) before a
   full run. Full runs only after the small run's outputs are inspected.
4. **No unnecessary wrappers.** Call the entry points of the submitted code
   (`external/odin2026/`: `vqa/code/competition_runner.py`,
   `vqa/code/run_vqa_inference.py`, `vqa/code/train_vision_lora.py`, … — exact roles mapped in M0
   and recorded in DECISIONS.md) directly. Only write new code for genuinely new functionality.
   If that code needs a change, edit it in place as an ordinary commit that says why (the import
   commit is verbatim, so `git diff` against it shows every change); don't wrap it.
5. **Diagnose, don't work around.** When something breaks, find the root cause and fix it inline.
   Use a backup plan (§12) only when the root cause is outside our control.
6. **Verify empirically.** Don't assume a file layout, label map, ID convention or orientation:
   check it on real files and record what you found in `DECISIONS.md`.
7. **Never fabricate results.** Every number in a report must come from a file in `results/`
   produced by code in this repo at a recorded git SHA. If something wasn't run, say so.
8. **Data use agreement (DUA).** ToothFairy4 data (volumes, masks, report text) is
   access-controlled (CC BY-NC-SA + data-use terms). Never commit it, never push it, never paste
   report text into GitHub issues/PRs, and **never send it to an external API** (OpenAI, etc.)
   unless Shivam confirms in writing that the DUA allows it. Local models only by default.
9. **No backups on the cluster.** The 200 TB NVMe shared filesystem is NOT backed up. Code lives
   in git and is pushed at least daily. Small derived results (JSON/CSV without report text) are
   synced off-cluster weekly (§5.6).
10. **Log decisions.** Every non-trivial choice (threshold, mapping, exclusion, fallback taken)
    goes into `DECISIONS.md` with date, reason and evidence path.
11. **Login node is shared.** Only editing, git, `ls`/`head`, and jobs < ~2 min on the login
    node. Everything else through SLURM. There are no interactive compute nodes.

---

## 1. Project in one paragraph

We grade report-generation models by comparing them with clinician-written reference reports.
But references are **partial**: clinicians omit findings they see (routine fillings, crowns,
subtle caries). Current metrics (RadFact, structured F1) count a correct-but-unreported finding as
a false positive. We (a) estimate per-finding **reporting propensity** π_f (probability a reader
writes finding f when it is present) from cases that have ≥2 independent reports, (b) build
**propensity-corrected metrics** plus a human-vs-human **noise floor** and two mandatory
**controls** (no-image corpus prior; acquisition-header-only model), (c) **re-evaluate** existing
systems (V2V2R and ablations, retrained on our own splits), (d) build a **segmentation-aware joint decoder** whose recall gains
are visible only under corrected metrics, and (e) validate with a **dentist re-reading study** and
a **second, non-dental domain**. Target: Medical Image Analysis (MedIA), submission by end of
June 2027.

---

## 2. The science the code must implement

### 2.1 Propensity model (per finding f, per centre c)

- A finding instance (case, finding type, tooth position if tooth-level) is truly present with
  prevalence p.
- If present, each reader mentions it independently with probability π.
- If absent, readers mention it with probability φ (false-positive rate). **Base model: φ = 0.**
- Readers of the same case are exchangeable (same π).

Likelihood for an instance with n reports, k of which mention it:

```
L(p, π, φ) = p · C(n,k) π^k (1−π)^(n−k)  +  (1−p) · C(n,k) φ^k (1−φ)^(n−k)
```

With φ = 0 the second term is `(1−p)·[k = 0]`. Fit by maximum likelihood over **all** cases
(single-report cases contribute too; identifiability comes from n ≥ 2).

Closed form for n = 2, φ = 0 (use as a sanity check against the MLE):

```
P(both) = p π²        P(exactly one) = 2 p π (1−π)
R = P(exactly one) / P(both)   →   π = 2 / (R + 2)   and   p = P(both) / π²
```

Hierarchy: π and p get centre-specific values with partial pooling (logit-normal prior across
centres; rare findings pooled into finding families). Report 95% intervals by case-level
bootstrap (≥ 1000 resamples) and, for the hierarchical fit, posterior intervals.

Known bias: readers at the same centre share house style, so omissions are positively correlated.
That inflates agreement and makes π look **too high** → corrections are conservative. The
simulation study (M0.5) must quantify this.

### 2.2 Corrected precision

For a predicted positive the reference does not mention, the probability it is still correct,
given n reports none of which mention it:

```
P(present | 0 of n mention) = p (1−π)^n / ( p (1−π)^n + (1−p) (1−φ)^n )
```

Corrected precision = (matched predictions + Σ over unmatched predictions of that probability)
/ (number of predictions). Report raw and corrected side by side, always.

### 2.3 Recall

Reference-based recall is kept. Stated assumption: omission is independent of whether the model
detects the finding. Sensitivity analysis: recompute recall assuming omitted findings are harder
(parameter ρ ∈ {0, 0.25, 0.5}: probability the model misses an omitted finding is inflated by ρ).

### 2.4 Noise floor

On multi-report cases, score report A against report B (both directions) with every metric. The
human-vs-human score and the spread across {A, B, union} are reported as the ceiling and error bar
for that metric.

### 2.5 Controls (mandatory baselines for every comparison)

- **Corpus prior:** no image. Emits the findings whose training-fold prevalence exceeds a
  threshold tuned on training folds.
- **Acquisition-only:** features from the NIfTI header only (volume shape, voxel spacing, physical
  extent in mm, log-transformed); one regularized logistic regression per finding.

Rationale: George et al. (arXiv 2609.13238) showed a no-image prior scores near image models on
ToothFairy4. Controls show how much of any score needs the image.

### 2.6 Visibility (added 2026-11-07, agreed with Shivam)

Scans differ in field of view by centre (P/S ≈ 51 mm slabs, F ≈ 82 mm, A ≈ 90 mm; DECISIONS.md).
A reader cannot omit what is not in the scan, so every quantity is **conditional on visibility**:

- π_f = P(reader mentions f | f present **and visible**); p_f = prevalence among visible instances.
  Instances that are not visible are excluded from the propensity fit, agreement tables and noise
  floor — they are neither omissions nor negatives.
- **Visibility comes from the image only** (scan geometry, masks / segmentation), never from the
  reports (that would be circular).
- Partly visible structures: fixed rule "≥ X % of the structure inside the volume", X set before
  looking at outcomes; sensitivity analysis over X.
- **Model claims about non-visible anatomy get no omission credit:** the §2.2 correction applies
  only to visible instances; claims outside the field of view are scored as errors and reported
  separately (FOV-conflict rate).
- Report statements such as "not assessable / outside the field of view" are not findings; the
  per-report parser must label them separately.
- Field-of-view class is a split stratum and a covariate in every centre comparison.

**Agreed rules (2026-11-07; thresholds frozen after the image-facts job, before any report analysis):**
three tiers — (1) *mask-checkable* (missing teeth, implants, crowns, bridges): propensity from reader
agreement + validated against TF3 masks; (2) *visibility from image, no truth* (caries, fillings, root
canal treatment, periapical lesions, canal–tooth relation, sinus findings, condyles): propensity from
reader agreement, visibility from masks/geometry; (3) *visibility undecidable*: reported separately as
a sensitivity analysis, not in headline numbers. Per group: jaw-level findings visible if the jaw label
exceeds a minimum size; per-tooth findings if the tooth label is present; periapical lesion only if the
tooth's root tip is not cut by the scan boundary; missing tooth only if the slot lies between present
teeth or the quadrant's jaw label does not touch the lateral/AP boundary (else "not assessable");
canal visible if its label is not mostly cut; sinus visible if its label is ≥ X % of a typical complete
sinus (full-height F scans), X = 50 % with 25/75 % sensitivity; **condyles** visible if the mandible
label's posterior-superior ends lie inside the volume (not touching the top boundary). Reader statements
such as "not included in the field of view" never define visibility; they are used to validate it.

---

## 3. Repository layout

New repo: `partial-refs` (private, personal GitHub account `shivamsingh-ml`,
`git@github.com:shivamsingh-ml/partial-refs.git`). The submitted challenge code lives in
`external/odin2026/` as a plain folder, imported verbatim in its own commit.

```
partial-refs/
  CLAUDE.md                 # short: points to HANDOFF.md, repeats §0 rules 1–11
  HANDOFF.md                # this file
  DECISIONS.md              # dated decision log
  README.md
  pyproject.toml            # package `partialref`, src layout
  env/
    environment.yml         # cbct_partialref conda env (CPU path)
    requirements-gpu.txt    # only what the container lacks, if anything
  configs/
    paths.example.yaml      # template; real paths.yaml is gitignored
    audit.yaml  propensity.yaml  metrics.yaml  controls.yaml  decoder.yaml
    splits/                 # our split files (case IDs only, committed; §6.4)
  external/
    odin2026/               # exact /opt/app of the submitted image (PROVENANCE.md, SHA256SUMS)
  src/partialref/
    data/        manifest.py  cases.py  reports.py  centres.py  header_features.py
    audit/       inventory.py  duplicates.py  representativeness.py  mask_vs_report.py
    extract/     per_report.py          # per-report structured findings (NO union merge)
    propensity/  closed_form.py  mle.py  hierarchical.py  simulate.py
    metrics/     structured_f1.py  corrected.py  noise_floor.py  radfact_bridge.py
    controls/    corpus_prior.py  acquisition_only.py
    decoder/     scoring.py  calibrate.py  seg_confidence.py  factor_graph.py  select.py
    reader_study/ export_packets.py  import_reads.py
    provenance.py                        # run_id, git SHA, config + manifest hash
  jobs/
    env.sh                  # sourced by every job (see §5.3)
    cpu.sbatch              # generic CPU job: runs `python -m partialref.<module> ...`
    cpu_array.sbatch        # array over case chunks
    vllm_score.sbatch       # GPU: start vLLM, wait for health, run client, stop
    seg_probs.sbatch        # GPU: segmentation with probabilities → per-structure stats
    judge.sbatch            # GPU: RadFact-lite judge
  tests/                    # pytest; synthetic data only, never real reports
  results/                  # gitignored; one dir per run_id (§9)
  reports/                  # REPORT_Mx.md — numbers only, no report text
  figures/                  # standalone scripts reading results/, one per figure
  paper/                    # modular LaTeX: one .tex per section
```

`.gitignore` must exclude: `results/`, `data/`, `*.nii.gz`, `*.npz`, `*.sif`, `configs/paths.yaml`,
`**/reports_raw/`, any file containing report text, `.env`, `~/.secrets`.

Add a pre-commit hook that refuses commits containing `.nii`, `.npz`, `.sif`, or files over 5 MB.

---

## 4. First session checklist (do this before writing project code)

Write findings to `reports/ENV_REPORT.md`.

1. `hostname; whoami; date` — confirm login node and user `ssingh54`.
2. Filesystem: `df -h ~ /msc/home/ssingh54`; check quotas (`<<FILL: quota command>>`); find a
   project root with space, e.g. `/msc/home/ssingh54/projects/partial-refs` `<<CONFIRM>>`.
3. SLURM: `sinfo -s`, `sacctmgr show assoc user=ssingh54 format=account,partition,qos`,
   `scontrol show partition`. Record GPU partition name, GRES string (e.g. `gpu:a100:1`), max
   walltime, CPU partition, account. Do not guess.
4. Modules/containers: `module avail 2>&1 | head -100`; `which apptainer singularity conda mamba micromamba`.
5. Existing assets — locate, don't re-download:
   - ToothFairy4 data under `/msc/home/ssingh54/Datasets/` (`ls`, count cases/reports).
   - V2V2R checkout and its container: `find ~ -maxdepth 3 -name "*.sif" 2>/dev/null`;
     V2V2R README refers to `SIF_PATH`.
   - Model: `models/Qwen3.5-9B-AWQ-dental-cbct-sft` or HF `lucent517/v2v2r_cbct_vqa`.
   - Segmentation: U-Mamba2 setup at `~/U-Mamba2/` (env script `~/U-Mamba2/jobs/env.sh`),
     trained weights location `<<FILL>>`.
   - Docker image `ssingh54/toothfairy4-v2v2r` (Docker Hub; repository page needs login).
6. Network: can the login node reach github.com, huggingface.co, registry-1.docker.io?
   Can compute nodes? (Test with a 1-minute CPU job running `curl -sI`.)
7. GPU smoke job (2 min): `nvidia-smi`, CUDA driver version, `python -c "import torch; print(torch.cuda.is_available())"` inside the container.

Stop and send ENV_REPORT.md to Shivam with the list of `<<FILL>>` values still missing.

---

## 5. Environment setup on MSC

### 5.1 Two environments, on purpose

| Env | Where it runs | What it is for |
| --- | --- | --- |
| `cbct_partialref` (conda) | login node + CPU jobs | audit, propensity, metrics, controls, figures, tests. No torch, no vLLM. |
| Container (Apptainer SIF) | GPU jobs | vLLM serving/scoring, VLM SFT, segmentation probabilities, RadFact judge. |

This mirrors V2V2R's split (`cbct_base` CPU env + container for the vLLM server).

### 5.2 Conda env (CPU)

Revised 2026-11-07: `env/environment.yml` + `env/requirements-cpu.txt` (no V2V2R dependency),
built by `jobs/build_cpu_env.sbatch`. `radfact-lite` stays pinned to git SHA `053f680` (the pin
V2V2R used — keep it; PyPI 0.1.0 gives non-comparable scores). Contents:
`numpy scipy pandas pyarrow scikit-learn statsmodels numpyro jax[cpu] rapidfuzz datasketch
nibabel SimpleITK pyyaml pydantic matplotlib pytest pre-commit`.

Then: `python -m nltk.downloader wordnet omw-1.4` (METEOR silently changes scale without it;
every score must record `meteor_backend: nltk-wordnet`).

### 5.3 `jobs/env.sh`

Pattern Shivam already uses in jobs: `set +u` before sourcing env scripts (they reference unset
variables), then `set -u` again.

```bash
#!/usr/bin/env bash
set +u
source "<<FILL: conda init path>>"
conda activate cbct_partialref
set -u
export PROJECT_DIR=/msc/home/ssingh54/projects/partial-refs      # <<CONFIRM>>
export DATA_DIR=/msc/home/ssingh54/Datasets/ToothFairy4           # <<CONFIRM exact dir>>
export SIF_PATH=<<FILL>>
export HF_HOME=$PROJECT_DIR/.cache/hf
export XDG_CACHE_HOME=$PROJECT_DIR/.cache
export PIP_CACHE_DIR=$PROJECT_DIR/.cache/pip
export APPTAINER_CACHEDIR=$PROJECT_DIR/.cache/apptainer
export VLLM_CACHE_ROOT=$PROJECT_DIR/.cache/vllm
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
# secrets: only if a step needs them and Shivam approved it
# set +u; source ~/.secrets/hf.env; set -u
```

### 5.4 Container

**Revised 2026-11-07: no container by default.** GPU envs are conda envs pinned to the versions
in the image's `requirements-vlm.txt` / `requirements.txt` (`vllm==0.19.0`, …) plus a training env
for the image's LoRA SFT scripts; the pull below is the fallback (`jobs/pull_sif.sbatch`).

Plan A (original): reuse the existing V2V2R SIF (`SIF_PATH`). Record `apptainer inspect` output and
`pip freeze` from inside it in `ENV_REPORT.md`.
Plan B: `apptainer pull docker://ssingh54/toothfairy4-v2v2r:<<tag>>` on the login node.
Plan C–E: see §12.1.

Important V2V2R lesson: the vLLM / torch compile caches must exist at the container's exact
paths, or every job recompiles. Bind a persistent cache dir and keep the same bind path in every
job.

### 5.5 SLURM templates (shape; fill partition/account/GRES from §4)

```bash
#!/usr/bin/env bash
#SBATCH --job-name=pr-<step>
#SBATCH --partition=<<FILL gpu partition>>
#SBATCH --account=<<FILL>>
#SBATCH --gres=<<FILL e.g. gpu:a100:1>>
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=<<FILL ≤ max walltime>>
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source "$PROJECT_DIR/jobs/env.sh"
```

GPU jobs that use vLLM: start the server in the background, poll `GET /v1/models` until ready
(V2V2R measured 15–25 min load on A100), run all cases in one job with `RESUME=1`, then stop the
server. Never one job per case.

### 5.6 Backups (cluster storage has none)

- Code: `git push` at least daily.
- `results/**/metrics.json`, `results/**/*.csv` **without report text**, `reports/`, `figures/`
  outputs: weekly `rsync` **pulled from the lab PC** (decided 2026-11-07; command in README.md).
  The SMB share was dropped: it may expose the same un-backed-up filesystem. Never copy raw data or report text off-cluster except to approved locations.

---

## 6. Data

### 6.1 Expected layout (V2V2R convention — verify on disk)

```
dataset/{training,validate}/
  images/   {case}_0000.nii.gz
  masks/    {case}.nii.gz
  reports/  {case}_1.txt, {case}_2.txt, ...   (a case may have several readers)
  facts/    {case}.json                         (segmentation-derived facts)
```

Known facts (from published sources; re-verify):
- ~622 public cases from 4 centres: P 412, A 95, F 63, S 52 (George et al.).
- Reports exist in Italian and English; ~1,000 reports total. About 367 cases have >1 report.
- The official hidden test set (50 cases, unseen centre) has no public references — it cannot be
  used for corrected metrics.

### 6.2 Things to verify before any analysis (write results to `DECISIONS.md`)

1. **Centre from case ID.** IDs look like `A008`. Confirm the letter prefix = sub-dataset
   (P/A/F/S) against the official dataset description.
2. **Language.** Confirm `{case}_1.txt` and `{case}_2.txt` are two *English* reports by
   different readers — NOT the Italian original and its English translation of the same report.
   If both languages are present, use English only and pair by reader, not by file.
3. **Mask label map.** V2V2R remapped ToothFairy3's 77 classes to 47 (pulp merged, unused IDs
   dropped). Load the label map from the V2V2R code/config; check tooth labels follow FDI.
4. **Orientation.** Standardize with `sitk.DICOMOrient(img, "RPI")` (silent bugs otherwise).
   Check with anatomical centroids (mandible below maxilla, left/right teeth on the expected side).
5. **facts.json.** Confirm which fields come from masks (absent teeth, implants, IAN proximity…)
   and how they were computed.

### 6.3 Manifest

`python -m partialref.data.manifest` writes `results/manifest/<run_id>/manifest.parquet`:
case_id, centre, split, n_reports, report paths, language, volume shape, spacing, physical
extent (mm), sha256 of each file. Every later run records the manifest hash.

### 6.4 Splits (leakage matters) — our own, revised 2026-11-07

- **Design after looking at the data.** The split is proposed in M1 (step 3b) from the inventory,
  the centre characterisation and the independence check, then approved by Shivam. Inputs: case ID,
  centre, number of (independent, English) reports, header features. Never report text.
- **Locked test set (~20 %)**, case-level, stratified by centre and by single- vs multi-report
  (and other strata if M1 shows strong centre differences). Not used for any tuning, threshold,
  pooling choice, calibration or model selection. Evaluated once, at the end, per metric.
- **5-fold case-level CV on the remaining ~80 % (development set)**, same stratification. All
  development (tuning, calibration, propensity cross-fitting, decoder choices) happens here.
- **Leave-one-centre-out (LOCO)** as a separate generalisation analysis.
- Split files (case IDs only, seed recorded) live in `configs/splits/` and are committed; every
  run records their hash.
- **All systems use the same split.** V2V2R's LoRA SFT is retrained per fold with its own
  `vqa/code/train_vision_lora.py` (submitted code); the released weights (trained on 528 of the public cases) must not be
  scored on our folds or test set. They may be used only for environment smoke tests.
- Propensity estimates used to score fold k are fit on reports from the other folds
  (cross-fitting). Estimates used for the locked test set are fit on the development set only.
- **Segmentation must not see the test set (added 2026-11-07).** The existing U-Mamba2 model
  (`fold_all`) was trained on all of ToothFairy3 = every P/F/S scan, so it overlaps any test set
  drawn from P/F/S. Order: (1) split approved → (2) retrain U-Mamba2 on TF3 labels of the
  development P/F/S cases + the 5 TF3-only scans (P120, P121, P262, P263, P378), never on test
  cases → (3) per-fold VLM LoRA. Inside the CV folds the segmentation model has seen the held-out
  fold (development only; accepted at first); per-fold segmentation retraining decided at G3. The
  old `fold_all` model is used only for non-evaluated purposes (e.g. rough checks on A).
  Hand-drawn TF3 labels used as ground truth about the image (visibility, mask-vs-report, report
  repair) are not leakage.
- **Centre A has only 2 multi-report cases.** A contributes almost nothing to estimating π; π for
  A's reports must come from pooled / reader-level estimates (the same `report_x` values that read
  P cases also read A cases — if `report_x` is a reader ID, reader-level π transfers to A). A's main
  roles: LOCO test of generalisation (scanner, field of view, site; segmentation unseen) and raw
  metrics; corrected metrics on A carry the transfer assumption and are reported as such.

---

## 7. Milestones

Order (revised 2026-11-08): M0 → **GT (go/no-go test)** → M0.5 → M1 → G1 → M2 → G2 → M3 → G3 → M4 → G4 → M5/M6.
**No V2V2R retraining, segmentation retraining or decoder work before the GT decision.**

### GT — Go/no-go test (before any further heavy compute)

Purpose: answer cheaply whether (i) reader disagreement is mostly *omission* and (ii) the effect is
large enough to change conclusions. Budget: ≤ 3 GPU-hours (parser) + CPU. Uses provisional
visibility (tier 1–2 rules of §2.6 with simple thresholds, recorded in DECISIONS.md); the final
thresholds are frozen later in M1. Thresholds below are **guidance values**: Shivam takes the
go/no-go decision after seeing the T1–T4 numbers, and the reasoning is written to DECISIONS.md either
way (internal decision; the paper's analyses stay pre-specified). Parser input: **English reports**
(team cannot read Italian; manual checks must be possible); translation noise is measured
automatically by also parsing the Italian originals and comparing extracted findings per report.
Few-shot examples for the parser are synthetic (written by Claude, no real report text).

| Test | What | Pass if |
| --- | --- | --- |
| T1 Parser | Local LLM parses all 1000 EN reports (Lumetti ontology; IT also parsed for a translation check). **Judged by Shivam's blinded review against the report text** (review app, ~60 held-out reports with extra reports for rarer types; revised 2026-11-09); agreement with Lumetti's per-patient labels is an internal early warning only and is **not reported** | precision and recall ≥ 0.90 in the review for the common finding types; no systematic error pattern |
| T2 Propensity | π per common finding type, pooled over centres, visible instances, double-read cases (excl. P116/P117/P260/P261), bootstrap CIs | ≥ 3 finding types with upper 95 % CI of π < 0.90 and CI width ≤ 0.30 |
| T3 Mask agreement | Tier-1 findings (missing teeth, implants, crowns/bridges): π from two readers vs π measured against TF3 masks; false-mention rate φ vs masks | CIs overlap for ≥ 2 of 3 types and φ ≤ 0.05 |
| T4 Effect size | Plug estimated (p, π) into a simulated scoring of realistic systems (a conservative and a complete one) | corrected − raw precision ≥ 5 points for common types, and/or the ranking of the two systems flips |

Decision rules: all pass → continue with M0.5/M1/M2/M3 as planned. **T1 fails** → fix the parser first
(nothing else is interpretable). **T3 fails** → disagreement is not mainly omission: switch to a
false-positive-aware model (φ from masks / 3-report cases) and partial-identification bounds (§12.5 D).
**T2 or T4 fails** (π ≈ 1 or tiny effect) → pivot to a measurement/benchmark-analysis paper on
reference incompleteness, with V2V2R improvements as the second pillar. Report: `reports/REPORT_GT.md`.

### M0 — Setup and smoke test

Tasks:
- [ ] §4 first-session checklist → `reports/ENV_REPORT.md`.
- [ ] Repo skeleton (§3), submitted code in `external/odin2026/`, pre-commit hook, conda env, `jobs/env.sh`.
- [ ] Manifest (§6.3) and the §6.2 verifications.
- [ ] Smoke run of V2V2R inference on 3 cases using its own entry points:
      `STAGE=images` → `STAGE=infer` → `STAGE=post` with `CASE_IDS=...`, `RESUME=1`.
      Environment test only (no crashes, valid JSON, report produced); no metrics are reported
      from it, so the released weights may be used here.
- [ ] Smoke run of the RadFact-lite judge on those 3 reports (local Qwen3-14B judge).

**G0 (stop):** environment works end to end on 3 cases; all `<<FILL>>` values resolved or listed.

### M0.5 — Simulation study (no real data needed; start immediately)

Purpose: prove the estimator works before touching real reports; becomes a paper figure.

`partialref.propensity.simulate` generates synthetic cases with known ground truth:
- K finding types with prevalences p_f ∈ [0.01, 0.6]; propensities π_f ∈ [0.2, 0.95].
- Centres with different π (house style); 1–3 reports per case with a realistic distribution
  (≈ 60% of cases with ≥2 reports).
- Reader correlation ρ (shared omission within a centre) ∈ {0, 0.1, 0.3}.
- False-positive rate φ ∈ {0, 0.01, 0.03}.

Checks (pytest + a results table):
- Closed form and MLE recover π, p within CI at ρ = 0, φ = 0; CI coverage ≈ 95%.
- With ρ > 0, π is overestimated (direction confirmed, size quantified).
- With φ > 0 and only 2 reports, show the bias; with 3-report cases or a mask "reader", show
  recovery.
- Corrected precision on simulated model outputs is closer to the true precision than raw
  precision, across settings.

**Output:** `reports/REPORT_M0.5.md` + `figures/fig_sim_recovery.py`.

### M1 — Data audit

All CPU. Outputs in `results/audit/<run_id>/`; summary in `reports/REPORT_M1.md`.

1. **Inventory and centre characterisation.** Reports per case by centre and official split;
   number of cases with 1/2/3 reports; language. Per centre: report length, number of findings,
   missing teeth (masks), volume shape / spacing / field of view (headers). Show how centres
   differ — this decides the split strata (step 3b).
2. **Independence check (most important).** For every pair of reports of the same case:
   - character 5-gram TF-IDF cosine, normalized Levenshtein ratio (`rapidfuzz`), MinHash Jaccard;
   - flag pairs above thresholds (start: cosine > 0.9 or Levenshtein ratio > 0.85 — tune by
     inspecting the distribution; record final thresholds in DECISIONS.md);
   - Shivam manually inspects the top 20 flagged pairs and 20 random pairs (export IDs + scores
     only; he reads the text on the cluster).
   - Also check: are the two reports from different centres' conventions? (They shouldn't be.)
3. **Representativeness.** Compare multi-report vs single-report cases on: centre mix, number of
   missing teeth (from masks), number of findings, report length, field of view. Report
   standardized mean differences (SMD); flag |SMD| > 0.2.
3b. **Split design (§6.4) — mini-gate.** After steps 1–3 (near-duplicate pairs change which cases
   count as multi-report), propose the locked test set, 5 folds and LOCO with per-split tables
   (centre mix, multi-report share, key header features). **Stop for Shivam's approval.** Steps
   4–7 that tune anything (thresholds, pooling, parser checks) use the development set only.
4. **Per-report structured findings (no union).** Target vocabulary (decided 2026-11-07): the
   finding ontology of Lumetti et al. (MICCAI, repo `AImageLab-zip/CBCT-Report`; 15 finding types with
   tooth/side/quadrant slots and qualifiers), re-implemented in our own code and cited — their files
   are not copied (no licence stated). Parser = local LLM only (never GPT/external API). Parser check:
   the union of our per-report findings per case is compared with their released, clinician-validated
   per-patient labels (internal use only, never redistributed), plus a small manual per-report check
   by Shivam (~10 reports, on the cluster). V2V2R parses each report with Qwen3-14B in two
   stages and then **union-merges** reports of the same case. We need the **stage-1 per-report**
   output. Locate that code in `external/odin2026/vqa/code/` (`parse_reports_to_gt.py`), add a flag to dump per-report
   findings, or call stage 1 directly. Output one row per (case, report, finding, tooth).
   Validate on 30 reports against manual reading by Shivam (precision/recall of the parser per
   finding type).
5. **Agreement tables.** Per finding type (and tooth-level instance) and centre: counts of
   both-mention / one-mentions / neither on 2-report cases; full k-of-n counts for 3-report cases.
6. **Mask vs report.** For mask-visible findings (missing teeth; implants; crowns/fillings/
   endodontic material if the label map has them — verify), compare each report with the mask.
   This estimates π directly and tests φ ≈ 0 for these findings.
7. **Propensity estimates.** Closed form + MLE + hierarchical, per finding × centre, with CIs;
   flag findings with < 30 positive mentions (proposed threshold) for pooling.

**G1 (stop) — proposed criteria for Shivam to confirm:**
- Split files approved and committed (step 3b).
- ≤ 5% of report pairs flagged as near-duplicates after inspection (otherwise exclude them and
  re-check support).
- Multi-report subset representative (|SMD| ≤ 0.2) or reweighting planned.
- Parser per-finding F1 ≥ 0.9 on the 30-report check for the findings we'll use.
- π estimates for core findings have CI width ≤ 0.3.
- Mask-vs-report agreement consistent with the propensity model (π from masks within CI of π from
  reader pairs, for mask-visible findings).

### M2 — Corrected metrics, noise floor, controls

1. `metrics/structured_f1.py`: LLM-free tuple-level P/R/F1 (case, finding, tooth), the
   **primary** metric for corrections, because propensity is defined per finding type. Definitions
   follow Lumetti et al. (pres-F1, inst-F1, rep-F1, char-acc), re-implemented and cited; our
   contribution is their propensity-corrected versions. V2V2R outputs are mapped to the ontology
   transparently (mapping coverage reported).
2. `metrics/corrected.py`: §2.2 corrected precision, §2.3 recall + sensitivity analysis.
3. `metrics/noise_floor.py`: §2.4, per metric and per centre.
4. `metrics/radfact_bridge.py`: RadFact-lite (pinned SHA) as the secondary metric. To correct it,
   map each RadFact proposition to a finding type (rule-based + the per-report parser); unmapped
   propositions are scored raw and counted separately.
5. Controls (§2.5) trained per fold.
6. Validation on mask-visible findings: masks act as near-truth; check corrected precision is
   closer to mask-based precision than raw precision.

**G2 (stop):** corrected metric beats raw metric at tracking truth in the simulation AND on
mask-visible findings; noise floor computed; controls run on all folds.

### M3 — Re-evaluation of existing systems

On our splits (§6.4): 5-fold CV on the development set for all development; the locked test set
once at the end; LOCO as a separate analysis.
- V2V2R full pipeline with LoRA **retrained per fold** (`vqa/code/train_vision_lora.py`, ~1,190 steps per
  fold on A100; approved 2026-11-07); ablations from the paper (segmentation-only, base
  Qwen3.5-9B, +LoRA); `POSTPROCESS_CONFIG=configs/postprocess/no_source_rules.yaml`; controls.
  Retraining first on one fold as a cheap diagnostic (§0.3) before all folds.
- Both fixed-reference and mean-over-references protocols; paired bootstrap CIs (≥ 10,000
  resamples, matched cases).
- Our numbers are not directly comparable with the challenge leaderboard or the V2V2R paper
  (different training data per fold); say so in the paper.
- Ranking table: raw vs corrected, per metric. Which pairwise comparisons flip?

Optional: if other teams' outputs are obtainable (e.g. EviDent-CBCT), include them only with
written permission.

**G3 (stop):** Shivam decides whether the re-ranking result is strong enough to carry the paper's
headline, or whether the decoder must carry more weight.

### M4 — Segmentation-aware joint decoder

Build on the V2V2R VLM with the per-fold LoRA adapters from M3; develop on the CV folds, report
the locked test set once at the end.

1. **Scoring instead of sampling** (`decoder/scoring.py`). The challenge entry decoded at
   temperature 1.0, top-p 0.8. For every closed schema field, compute the log-probability of each
   allowed answer and normalize across options (sum token log-probs for multi-token options).
   - The schema asks for a `visual_evidence` text before each answer. Generate that evidence
     greedily first, then score the answer options conditioned on it.
   - Implementation order: Plan A/B/C in §12.4. Verify the API against the **installed** vLLM
     version (V2V2R used vLLM 0.19.0); parameter names for logprobs and guided choice have
     changed across versions.
2. **Calibration** (`decoder/calibrate.py`). Temperature scaling per field (or per field family)
   on training folds; report ECE and reliability diagrams per field.
3. **Segmentation confidence** (`decoder/seg_confidence.py`, GPU job `seg_probs.sbatch`).
   Run segmentation with probability output (nnU-Net v2 `--save_probabilities`), compute
   per-structure statistics **immediately**, then delete the probability files (they are huge and
   storage has no backup). Per tooth: expected voxel count Σp, mean max-probability inside the
   mask, mean entropy, fraction of voxels with p < 0.5. Store as one small parquet per case.
4. **Factor graph** (`decoder/factor_graph.py`). Per case:
   - Variables: per tooth — present, endodontic, post-and-core, crown, filling, caries, root
     remnant, implant, impacted; global — maxilla/sinus/condyle inclusion, atrophy, lesions.
     Take the exact field list from `external/odin2026/vqa/schema/`.
   - Unary factors: calibrated VLM probabilities, one per view (panoramic, tooth composite, 3D
     render), replacing V2V2R's source-selection rules.
   - Pairwise factors: co-occurrence (e.g. post-and-core ↔ endodontic) learned from per-report
     findings by pseudo-likelihood logistic regressions with L1, using counts **corrected for
     omission with π** (raw co-occurrence counts are biased by omission).
   - Soft anatomical factors: mask facts (tooth absent, maxilla out of view) weighted by
     segmentation confidence, so a wrong mask cannot hard-veto a real finding.
   - Inference: exact enumeration per tooth (≤ 2^9 states); cross-tooth factors (bridges) via
     a few rounds of loopy belief propagation, or dropped in v1 (record in DECISIONS.md).
5. **Selection** (`decoder/select.py`). Choose the emitted set maximizing approximate expected F1:
   sort by marginal q_i, pick the top-k maximizing `2 Σ_{i∈S} q_i / (|S| + Σ_all q_i)`.
   Two targets: latent truth (q = presence marginal) and observed reference (q × π). Report both.
6. **Report assembly**: reuse V2V2R's template assembly unchanged, so text metrics stay
   comparable.

Experiments (from the proposal): E6 sampled vs greedy vs scored+calibrated; E7 current rules vs
deterministic projection (EviDent-style baseline) vs joint inference; E8 hard vs soft anatomical
factors under injected segmentation errors (delete a tooth from the mask, merge two teeth, shift
FDI labels by one) at controlled rates.

**G4 (stop):** decoder results under raw and corrected metrics, per finding (focus: post-and-core,
caries, root remnant).

### M5 — Second domain (preparation only until Shivam picks a dataset)

Do NOT register for or download credentialed datasets (PhysioNet, etc.) without Shivam.
Prepare `reports/SECOND_DOMAIN_OPTIONS.md`: candidate datasets with ≥2 independent reports or
adjudicated/multi-reader labels per study, licence, access process, size, whether reports are
public. A known pointer: PU-DPO (arXiv 2608.05341) used radiologist re-reads of CheXpert studies;
check whether those labels are released. Verify every property from the dataset's own page.

The code must already be domain-agnostic: findings, readers and centres are generic columns; no
dental assumptions outside `data/` adapters.

### M6 — Dentist re-reading study tooling

- `reader_study/export_packets.py`: sample cases (stratified by centre and by disagreement level),
  produce a blinded checklist form per case (CSV/XLSX: finding × tooth grid), no original report
  shown, random order, reader ID field.
- `reader_study/import_reads.py`: ingest completed forms, validate, compute agreement (Cohen's κ)
  if two readers.
- Size: 50–100 cases, final number set after M1 (`<<FILL>>`).

---

## 8. Engineering standards

- Python ≥ 3.10, src layout, type hints, `ruff` + `pytest`. Small modules, no notebooks in the
  pipeline (notebooks allowed in `scratch/`, gitignored).
- Config via YAML + pydantic validation. No hard-coded paths in code; everything from
  `configs/paths.yaml` / env vars.
- Determinism: seeds in config; `PYTHONHASHSEED=0`; record library versions.
- Every CLI is `python -m partialref.<area>.<module> --config ...`. Jobs call these directly.
- Resumable: per-case outputs, skip existing unless `--overwrite`.
- Tests use synthetic data only (`tests/fixtures/` generated by `simulate.py`). Never commit real
  report text, even in fixtures.
- Figures: one standalone script per figure in `figures/`, reading only `results/`.

## 9. Provenance (every run)

`results/<area>/<run_id>/` with `run_id = YYYYMMDD-HHMM-<short git sha>-<tag>` containing:
`config.yaml` (resolved), `provenance.json` (git SHA, dirty flag, manifest hash,
SLURM job ID, node, GPU, container digest, conda `env export` hash, `meteor_backend`,
`radfact_lite_sha`, judge model + version), `metrics.json`, logs. Reports cite run IDs.

---

## 10. Things the agent must never do

- Commit or push data, masks, probabilities, model weights, or any report text.
- Send ToothFairy4 content to external APIs without written DUA confirmation.
- Edit `external/odin2026/` without a commit message saying why (and a DECISIONS.md entry for
  anything that changes outputs).
- Change the `radfact-lite` pin, METEOR backend, or BLEU aggregation (corpus-level, NLTK method-1
  smoothing in the official grader) — scores stop being comparable.
- Run GPU or long CPU work on the login node; submit hundreds of tiny jobs (use arrays).
- `rm -rf` anything outside `$PROJECT_DIR/results/tmp` or `.cache`.
- Evaluate the released V2V2R model on cases it was trained on (in practice: never score the
  released weights on our folds or test set; per-fold retrained adapters only).
- Use the locked test set for any tuning, selection or threshold choice.
- Report a number not produced by a recorded run.

---

## 11. Timeline (proposed; see proposal doc)

| Phase | Window | Gate |
| --- | --- | --- |
| M0 setup + M0.5 simulation | Nov 2026 | G0 |
| M1 data audit | Nov 2026 | G1 (audit gate) |
| M2 corrected metrics + controls | Dec 2026 – Jan 2027 | G2 |
| M3 re-evaluation; M6 reader-study tooling | Jan – Mar 2027 | G3 |
| M4 joint decoder | Feb – Apr 2027 | G4 |
| M5 second domain | Mar – May 2027 | G5 |
| Writing → MedIA | May – Jun 2027 | submission |

---

## 12. Backup plans

Rule: try Plan A; on failure, diagnose the root cause first (§0.5). Move down the list only when
the cause is outside our control. Record the switch in DECISIONS.md.

### 12.1 Environment and container
- **A:** existing V2V2R SIF at `SIF_PATH`.
- **B:** `apptainer pull docker://ssingh54/toothfairy4-v2v2r:<tag>` on the login node.
- **C:** Docker Hub blocked from the cluster → build/pull the SIF on the lab PC or laptop,
  transfer via SMB.
- **D:** build from V2V2R's `env/container_requirements.txt` (local reference copy, not in the repo: `.cache/v2v2R-reference/`; a spec, not a tested image —
  verify a first build) with `apptainer build --fakeroot` if allowed, else on the lab PC.
- **E:** no container runtime at all → conda env with CUDA-matched torch + vLLM wheels; pin to
  the versions recorded from the SIF (`pip freeze`).
- Conda slow/broken → `mamba`/`micromamba`; last resort `python -m venv` on a module-provided
  Python.

### 12.2 SLURM and compute
- Long queue → smaller GPU requests, shorter walltime for backfill, array jobs; batch all cases
  into one vLLM job; checkpoint with `RESUME=1`.
- Job killed at walltime → per-case outputs make reruns resume; split by `CASE_IDS` chunks.
- No A100 available → the AWQ 9B model fits smaller GPUs: reduce `max_model_len`, lower
  `gpu_memory_utilization`, batch size 1; judge (Qwen3-14B) may need quantized weights.
- GPU out of memory in vLLM → reduce context (tooth composites are large images), lower
  concurrency, enable prefix caching off/on as diagnosed.
- Compute nodes have no internet → pre-download all models/caches on the login node into
  `$HF_HOME`; set `HF_HUB_OFFLINE=1` in jobs.
- Quota/disk full → move caches to project dir; delete segmentation probability files after
  stats; compress parquet; never store per-voxel outputs long-term.

### 12.3 Data and parsing
- Per-report parser (Qwen3-14B) unavailable or too slow → **B:** rule-based extractor for core
  findings (FDI numbers via regex; keywords for absent, implant, crown, endodontic, impacted,
  post-and-core, caries, root remnant) validated on the same 30 reports. **C:** restrict the
  paper's core analysis to findings the rule-based extractor handles at F1 ≥ 0.9.
- Reports turn out to be IT/EN pairs of the same reading → use English only; recount
  multi-report cases; if too few remain, go to §12.5 B/C.
- Label map lacks restorations/material classes → mask-vs-report only for missing teeth and
  implants; note it.
- Orientation or spacing anomalies (Set A known to differ in acquisition) → fix with `DICOMOrient`
  + record; exclude a case only with Shivam's approval.

### 12.4 VLM scoring (decoder step 1)
- **A:** offline vLLM (`LLM.generate`) scoring candidate continuations with prompt log-probs —
  exact likelihood of each option after the generated evidence.
- **B:** vLLM server with constrained choice (guided/structured choice) + top log-probs per field.
- **C:** Hugging Face `transformers` teacher-forced scoring with the merged LoRA model (slower,
  but exact and version-proof). Use for a 5-case cross-check of A/B in any case.
- Option tokenization differs between "true"/"false" spellings → normalize to the exact strings
  in the schema; verify with the tokenizer on 10 fields before full runs.

### 12.5 Propensity estimation (scientific risk)
- **A:** hierarchical model per finding × centre.
- **B:** pooled across centres (single π per finding) with centre as a sensitivity analysis.
- **C:** finding families instead of single findings when support is low.
- **D:** independence fails broadly or too few clean multi-report cases → **partial
  identification**: report every metric as a band over a plausible π range (e.g. from mask-based
  estimates and literature), plus sensitivity curves. No point estimate of π needed; still a
  methodological contribution.
- φ > 0 suspected → fit φ on mask-visible findings (mask as reference), use 3-report cases for a
  latent-class fit, report results with φ = 0 and fitted φ.
- Strong same-centre reader correlation → report π as an upper bound; corrections as
  conservative; quantify with the simulation (M0.5).

### 12.6 Metrics and judge
- RadFact judge (local Qwen3-14B) unavailable → structured F1 is the primary metric anyway;
  postpone RadFact to a later job; never switch the judge to an external API without DUA approval.
- Proposition-to-finding mapping too lossy → report corrected structured F1 as the main result,
  RadFact raw + noise floor only.

### 12.7 Segmentation confidence
- `--save_probabilities` too large/slow → **B:** keep only per-structure statistics computed in
  the same job, from logits in memory if the code path allows. **C:** test-time augmentation
  disagreement (mirroring) as the confidence signal. **D:** mask-geometry proxy (tooth volume vs
  typical volume for that FDI position; connected-component count).
- Segmentation weights not found → use the masks provided in the dataset for facts; confidence
  via §12.7 D only; record the limitation.

### 12.8 Decoder results
- Joint inference no better than rules on raw metrics → expected; check corrected metrics and
  per-finding results. If still no gain → the paper leads with evaluation (M1–M3); the decoder
  becomes a smaller section/ablation.
- Pairwise factors overfit → stronger L1, fewer pairs (only clinically motivated ones listed by
  Shivam), or unary-only + soft anatomy.

### 12.9 Validation
- Reader study delayed → proceed with simulation + mask-visible validation; reader study added
  before submission or at revision.
- No suitable second domain → **B:** semi-synthetic omission study: take a dataset with dense
  structured labels, inject omissions with known π (incl. centre effects and reader correlation),
  show recovery and metric correction. **C:** discuss transfer as limitation; consider a journal
  with lower generality expectations (Shivam decides).

---

## 13. Open items for Shivam (fill before M0 ends)

| Item | Value |
| --- | --- |
| Project root on MSC | `~/partial-refs` |
| GPU partition / GRES / account / max walltime | `gpu` / `gpu:<type>:N` with matching `--qos` / `uzk` / 3 days (`a100_long` 5 days); smallest GPU that fits first (ENV_REPORT §3) |
| CPU partition | `cpu`, QoS `cpu`, max 3 days |
| Conda init path | `~/miniconda3/etc/profile.d/conda.sh` |
| Existing V2V2R SIF path / Docker tag | `ssingh54/toothfairy4-v2v2r:odin2026-task1`; code extracted, no SIF needed |
| ToothFairy4 exact path + version | `~/Datasets/ToothFairy4/` (upload pending); version `<<FILL>>` |
| Segmentation weights path (U-Mamba2, TF3) | `<<FILL>>` |
| 40-case held-out list location | not needed (own splits, revision 2026-11-07) |
| GitHub org/repo for `partial-refs` | personal `shivamsingh-ml`, private |
| SMB backup path | not used — lab PC pulls weekly (README.md) |
| DUA: external APIs allowed? (default: no) | no |
| Reader-study size and readers | after M1 |

---

## 14. Key references (for context; read the abstracts before related code)

- V2V2R paper: https://papers.miccai.org/miccai-2026-sat/paper/ODIN_challenges_024.pdf
- V2V2R code: https://github.com/CC-AIDent-Vienna/v2v2R
- EviDent-CBCT (dental-logic projection, PU-weighted supervision): https://arxiv.org/pdf/2610.02375
- George et al., ToothFairy4 metric analysis and no-image prior: https://arxiv.org/pdf/2609.13238
- PU-DPO (omission noise in chest X-ray reports): https://arxiv.org/abs/2608.05341
- REREF (metric sensitivity to reporting style): https://arxiv.org/pdf/2609.19093v1
