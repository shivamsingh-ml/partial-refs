# DECISIONS

Dated log of non-trivial choices (HANDOFF §0 rule 10). Format: date — decision — reason — evidence.

## 2026-11-07

- **Repo location is `~/partial-refs`** (`/muwscs01/legacy/msc/home/ssingh54/partial-refs`), not
  `~/projects/partial-refs` as suggested in HANDOFF §4.2. `~/projects/` does not exist; HANDOFF.md
  was placed in `~/partial-refs` by Shivam. **Pending Shivam's confirmation** (listed in ENV_REPORT.md).
  Evidence: `reports/ENV_REPORT.md` §2.

- **Pre-commit hook as a plain git hook** in `.githooks/pre-commit`, enabled with
  `git config core.hooksPath .githooks`, rather than the `pre-commit` framework. Reason: no install
  needed, the rule (block `.nii/.nii.gz/.npz/.sif` and files > 5 MB) is a 20-line script, and it
  checks the staged blob size. Can be wrapped by the `pre-commit` framework later if wanted.

- **Simulation: reader-correlation model** (`partialref.propensity.simulate`). For a present
  instance read by n readers, with probability ρ all readers share one Bernoulli(π) draw, otherwise
  each reader draws independently. This keeps the marginal mention probability at exactly π and
  gives pairwise correlation exactly ρ between two readers' mentions (given presence), so ρ has
  the meaning "shared omission" from HANDOFF §2.1 / §7 M0.5 without changing π. False positives
  on absent instances are drawn independently with rate φ.

- **Simulation: centre house style** enters as a centre offset on logit(π_f), drawn
  N(0, `centre_logit_sd`²), clipped so π stays in [0.01, 0.99]. Prevalence p_f is shared across
  centres by default (`centre_prev_logit_sd = 0`), configurable.

- **MLE parameterisation with φ free:** φ = π · sigmoid(z). Reason: the two-component mixture is
  symmetric under (p, π, φ) ↔ (1−p, φ, π) (label switching); constraining φ < π picks the
  "present ⇒ more mentions" labelling. With φ = 0 (base model, HANDOFF §2.1) the fit uses
  logit(p), logit(π) only.

- **Case-level bootstrap in its own module** `partialref.propensity.bootstrap` (not in HANDOFF §3
  layout). Reason: shared by `closed_form` and `mle` (and later `metrics`); resampling is by case
  (cluster bootstrap), because instances of the same case are not independent.

- **V2V2R submodule pinned at `c5a6025e962adad89b75ea5c4a3998efe1827566`** (HEAD of the default
  branch on 2026-11-07; commit date 2026-10-07, "Update model references in README"). Provisional:
  re-pin to the commit used for the released model once Shivam gives that SHA (record the move here).

- **Test env for this session: existing conda env `jawfm`** (numpy 2.4.6, scipy 1.17.1, pandas 3.1.1,
  pydantic 2.14.3, pytest 9.2.0), run with `PYTHONPATH=src` via `srun -p cpu`, because
  `cbct_partialref` does not exist yet and installs were out of scope. Switch to `cbct_partialref`
  once `env/environment.yml` is built from V2V2R's `env/environment.yml` (HANDOFF §5.2).
  Result: 36 passed (srun on s0-n22, 2026-11-07).

- **Answers from Shivam (2026-11-07):** project root stays `~/partial-refs`, with `results/` and
  `.cache/` inside it; container = Docker `ssingh54/toothfairy4-v2v2r:odin2026-task1` (pull to SIF);
  released model = HF `lucent517/v2v2r_cbct_vqa`; GitHub = personal account (private); external
  APIs not allowed (default); reader-study size after M1; GPU policy = smallest GPU that fits
  first, scale up only on evidence (OOM / too slow). ToothFairy4 will be uploaded by Shivam via scp.

- **Submitted code ≠ any v2v2R commit (evidence, 2026-11-07).** BuildKit provenance of
  `ssingh54/toothfairy4-v2v2r:odin2026-task1` names `github.com/shivamsingh-ml/tf-submission`
  (private) @ `d9c27a85f1e1f18ef47093778eb75d74c3e8a279`, dir `algorithm/`. Blob-hash comparison of
  its `vqa/{code,schema}` (54 files) against all 31 v2v2R commits: no commit matches; only
  `schema/schema.json` is identical everywhere (and `build_vqa_pairs.py` in 2026-08-20 commits).
  Whitespace-insensitive diff vs v2v2R HEAD `c5a6025`: preprocess renderers near-identical
  (≤ 16 lines changed each); substantive changes in `run_vqa_inference.py` (+216/−195 of 1525),
  `postprocess_pred.py` (+346/−61 of 3410), `synthesize_report.py` (+390/−80 of 1128),
  `parse_reports_to_gt.py` (+306/−74), `build_vqa_pairs.py` (+191/−0); `competition_runner.py` has
  no counterpart at HEAD. Label maps (`tf3_single_pulp_label_map.json`,
  `tf3_official_original_to_train_label_map.json`, `structural_label_map*.json`) exist only in
  tf-submission. Script: session scratchpad `match_commits.py` (reads git trees only; few-shot
  example files compared by hash, never opened). **Which code defines "V2V2R" for M0/M3 is
  Shivam's decision** — submodule stays at `c5a6025` until then.

- **Own splits; V2V2R as a toolkit (Shivam, 2026-11-07).** Replaces the HANDOFF plan of scoring
  the released V2V2R weights on its 40 held-out cases. Reasons: (1) the propensity / corrected-metric
  core uses only reference reports and does not depend on V2V2R; (2) 40 cases (few of them
  multi-report) give too little power for the corrected metrics and paired comparisons; (3) one
  protocol for every system. Protocol: locked ~20 % test set + 5-fold CV on the development set +
  LOCO, stratified by centre and single/multi-report; strata finalised only after the M1 inventory,
  centre characterisation and independence check (new M1 step 3b, mini-gate). V2V2R LoRA retrained
  per fold (cost accepted). Released weights only for environment smoke tests. Consequences: no
  need for the 40-case list or the released model for evaluation; the `tf-submission` vs `v2v2R`
  code question is moot — submodule stays at public `v2v2R` `c5a6025` purely to record the tool
  version. Not comparable with leaderboard / V2V2R paper numbers (state in paper). HANDOFF.md §1,
  §3, §6.4, M0, M1, M3, M4, §10, §13 edited accordingly.

- **The challenge image is the only exact record of the submitted code (evidence, 2026-11-07).**
  Downloaded only the 8 code layers (20, 22, 25, 27, 28, 32, 33, 34; 1.4 MB, sha256 digests
  verified) of `ssingh54/toothfairy4-v2v2r@sha256:b4abf811…` via the registry API — no Docker,
  no full pull. Compared 307 files under `/opt/app` with `tf-submission@d9c27a8:algorithm/` by git
  blob hash: 275 identical, 11 differ (incl. `run_vqa_inference.py`, `postprocess_pred.py`,
  `synthesize_report.py`, `competition_runner.py`, `report_side.py`), 21 only in the image (incl. a
  full LoRA SFT pipeline: `train_vision_lora.py`, `sft_collator.py`, `build_sft_targets.py`,
  `select_sft_pool.py`, `merge_vision_lora.py`, `lora_arms.py`, `dequantize_awq.py`). Commit is
  2026-08-13, image built 2026-08-17 from an uncommitted working tree. None of the 23
  differing/image-only `vqa/code` files matches any v2v2R commit verbatim. Pins in the image:
  `vllm==0.19.0` (VLM venv), `transformers==4.43.4` (main env). Scripts: session scratchpad
  `extract_code_layers.sh`; few-shot example files compared by hash only, never opened.

- **Toolkit = submitted code; no container (Shivam, 2026-11-07).** The exact image code was
  committed (locally, push pending approval) to `shivamsingh-ml/tf-submission` as branch
  `odin2026-task1-image`, commit `e31c95d2831cb2c58169d6f276d1b5b6f177cfb7` = `d9c27a8` + 11 modified
  + 19 added files under `algorithm/`; verified: all 305 image files (excluding two upstream
  nnU-Net test `.nii.gz`) reproduced byte-exactly in the commit tree. Built with git plumbing in a
  blobless clone (no checkout, no large blobs fetched). Local copy of the extracted code + layer
  tarballs + registry metadata: `.cache/odin2026-task1-code/`. To be pinned as submodule
  `external/tf-submission`; `external/v2v2R` stays as reference only. GPU work in conda envs
  pinned to the image's versions; SIF pull kept only as fallback. HANDOFF §0.4, §3, §5.4, §10 and
  CLAUDE.md updated.

- **Supersedes the two entries above on submodules/branch (Shivam, 2026-11-07): only the submitted
  code, as a plain folder.** `external/odin2026/` = verbatim `/opt/app` of the image (305 files,
  `SHA256SUMS`, `PROVENANCE.md`), imported in its own commit; later edits are ordinary commits.
  The `tf-submission` snapshot branch (`e31c95d`) was never pushed and has been deleted; Shivam's
  `tf-submission` repo is untouched. The `external/v2v2R` submodule (never committed) was removed;
  a local reference copy is kept outside git at `.cache/v2v2R-reference/` (`c5a6025`). The CPU env no
  longer reads V2V2R files: `env/requirements-cpu.txt` carries the `radfact-lite@053f680` pin.
  `vqa/code/fewshot_examples/` (11 files) is gitignored until Shivam confirms DUA status (may hold
  report-derived text; never opened); it stays in `.cache/odin2026-task1-code/`. Remote: `origin` =
  `git@github.com:shivamsingh-ml/partial-refs.git` (SSH; the cluster key authenticates).

- **ToothFairy4 as uploaded (zip table of contents only, 2026-11-07).** `~/Datasets/ToothFairy4.zip`,
  47.7 GB, 2622 files (48.1 GB unpacked). Layout differs from HANDOFF §6.1: one folder per case,
  `{case}/cbct/volume.nii.gz`, `{case}/reports_en/cbct_<X>_<Y>.txt`, `{case}/reports_it/` with the
  same file names. **No masks and no facts, no official train/validate split** in this release.
  622 cases: P 412, A 95, F 63, S 52 (S IDs have 4 digits, e.g. `S0000`). 1000 EN + 1000 IT reports;
  EN and IT file names identical in every case → each EN file is the translation of the IT file with
  the same name (one reading, two languages), so readers are counted by file, EN only.
  EN reports per case: 1→255, 2→360, 3→3, 4→4 (367 multi-report). By centre: A {1:93, 2:2},
  F {2:62, 3:1}, P {1:154, 2:254, 4:4}, S {1:8, 2:42, 3:2}. `<Y>` is unique per report (371–3958).
  **Hypothesis, to verify against the dataset documentation:** `<X>` (16 values, 14–35) is a reader
  ID — in 363/367 multi-report cases all reports have different X, never the same X. Consequences
  if confirmed: reader-level propensity and a direct test of reader exchangeability. Centre A has
  almost no multi-report cases → its π cannot be estimated alone (pooling, HANDOFF §12.5 B/C).
  Missing masks: mask-vs-report checks (M1.6) need our own segmentation (U-Mamba2) first.

- **Prior knowledge from Shivam's `~/cbct-vlm-transfer` project (2026-11-07; to be re-derived by
  our manifest, not trusted blindly).** Sources: `docs/DATASET.md` §0, `FINDINGS.md` #1, #14, #25,
  `reports/intensity_percase.csv` (622 rows), `manifests/qc_flags.csv` (read as numbers/headings
  only; no report text).
  1. **Centres:** P/F/S = ToothFairy3 public Sets A/B/C (417/63/52 released → 412/63/52 with
     paired reports); **A = external Sfax (Tunisia) cohort**, different site and scanner.
     Answers HANDOFF §6.2.1 pending a check against the official dataset description.
  2. **Field of view (FOV) differs by centre and is permanent:** z-extent median P 50.7 mm
     (45.3–53.4, 100 % < 60 mm), S 51.6 mm, F 82.2 mm, A 90.0 mm; in-plane A ~83 mm, F ~123 mm,
     P/S ~101–111 mm. P/S are thin slabs around the teeth ("cut" scans).
  3. **Acquisition is confounded with centre:** A = LPS, spacing 0.12–0.45 mm (10 distinct),
     often not air-calibrated / saturated; P/F/S = LAS, 0.30 mm isotropic.
  4. **ToothFairy3 ships 77-class labels** for the P/F/S scans → masks may exist on disk for
     527 cases (U-Mamba2 data, `sae_tooth` preds named `ToothFairy3P_<n>`); mapping TF4 ID ↔ TF3 ID
     to be verified.
  **Proposed consequence (needs Shivam's OK):** a finding outside the scanned volume cannot be
  "omitted". Propensity, agreement tables and corrected precision must be computed only on
  instances that are **visible in the FOV** (per finding: in-FOV indicator from geometry and, where
  available, masks); FOV class becomes a split stratum and a covariate; the acquisition-only
  control will largely encode centre.

- **Visibility conditioning adopted (Shivam, 2026-11-07).** Propensity, agreement, noise floor and
  corrected precision are defined on visible instances only; visibility from image geometry/masks,
  never from reports; partial-visibility threshold fixed in advance + sensitivity analysis; model
  claims outside the FOV get no omission credit (scored as errors, reported separately). New
  HANDOFF §2.6.
- **Data unpacked (job 614552, 2026-11-07):** `~/Datasets/ToothFairy4/`, 2622 files, 622 cases,
  622 volumes, 1000 EN + 1000 IT reports, 45 GB, permissions owner-only. Zip sha256
  `40dbbf2d…a073` (job 614550). `unzip` is not installed on compute nodes → Python `zipfile`.
- **Manifest 5-case diagnostic (job 614554):** A003–A009; headers float32, qform/sform codes 1,
  LPS; spacing/extent/orientation match `cbct-vlm-transfer/reports/intensity_percase.csv` for all
  5. A comparator bug (int vs float after outer merge made shapes look mismatched) was fixed with
  a regression test before the full run.

- **Full manifest (job 614556, run `20261107-1830-c2aa718-manifest`, manifest hash
  `72b68659…8805`):** 622 cases (A 95, F 63, P 412, S 52), 1000 EN reports, 367 multi-report,
  EN/IT names match everywhere, no missing volumes, 622 distinct volume sha256.
- **Duplicate data removed (Shivam's instruction, 2026-11-07).** `~/Datasets/ToothFairy4.zip`
  deleted after extraction checks. `~/cbct-vlm-transfer/data/raw` was byte-identical to our copy
  (job 614558: 2622/2622 files `cmp`-identical; 3 random volumes also matched the manifest sha256),
  so it was replaced by a symlink to `~/Datasets/ToothFairy4` (keeps that project's 7 scripts that
  reference `data/raw` working). ~93 GB freed in total.
- **`cbct-vlm-transfer` preprocessed copies deleted (Shivam, 2026-11-07):** `data/prep_air` and the
  seven `prep_crop*/prep_pad*/prep_xcrop*` folders (~74 GB, real folders, no links). Regenerable from
  `data/raw` with that project's `scripts/ingest.py` / `preprocess.py`; note left in
  `cbct-vlm-transfer/data/DELETED_PREP_README.txt`.
- **Inventory (job 614559) → `reports/REPORT_M1_inventory.md`.** Geometry identical to the earlier
  project's table for 622/622 cases. New flags: P116/P117/P260/P261 carry 4 EN reports with two per
  `report_x`; 15 EN files in 5 identical-content groups across cases, all `report_x = 14`
  (template reuse); centre A has 2 multi-report cases.
- **P116/P117/P260/P261 — likely misfiled reports (evidence from file names only, 2026-11-07).**
  P120, P121, P262, P263 (and P378) are absent from TF4 (no folder/scan/reports) but present in
  ToothFairy3 with image + label. Per-reader running report numbers follow case order: reader 21
  P115 977, P116 978, P117 981, P118 982, P119 991, P122 1183; reader 26 P119 1454 … P122 1450.
  P116's extra reports (21/992, 26/1453) sit exactly where P120's are expected, P117's extras
  (21/997, 26/1451) where P121's are expected → probably P120/P121 reports filed under P116/P117.
  P260/P261 extras (24/1594, 24/1322; 31/1014|1036, 31/922|1015) do not fit cleanly. **Interim:**
  exclude P116/P117/P260/P261 from multi-report analyses. **Possible repair (needs Shivam's OK
  per case):** after per-report findings exist, match each extra report to TF3 masks of the
  candidate scans (score only) and reassign; would recover P120/P121/P262/P263.
- **Segmentation leakage handled by retraining without the test set (Shivam, 2026-11-07).** The
  existing U-Mamba2 (`fold_all`, trained on all ToothFairy3 = all P/F/S scans) cannot provide
  inputs for evaluating anything on P/F/S. After the split: retrain U-Mamba2 on development P/F/S +
  5 TF3-only scans; locked test set clean for segmentation and VLM. Within-CV overlap accepted for
  development; per-fold segmentation decided at G3. HANDOFF §6.4 updated (also: role of centre A,
  which has 2 multi-report cases).
- **Independence check (job 614574, run `20261107-1911-f6a9649-duplicates`, EN, P116/P117/P260/P261
  excluded):** 984 reports, 369 within-case pairs, **0 flagged** (cosine > 0.9 or Levenshtein
  > 0.85). Within-case cosine median 0.18 (q95 0.46, max 0.77); a report's nearest *other-case*
  report has median cosine 0.47 → reader style dominates, pairs are independent writing. 30 reports
  have a cross-case twin with cosine > 0.9 (templates). Official dataset page: most cases "have
  exactly two reports redacted by different maxillofacial surgeons with >5 years of experience";
  EN reports are GPT-5.2 translations, only random subsets clinician-checked (test-set translations
  reviewed); official counts (625 cases, 1001 reports, max 3 per case) differ from our download
  (622, 1000, four cases with 4) → different release version; to ask Shivam which. File naming
  is not documented, so `report_x` = reader stays a hypothesis.
- **TF3 ↔ TF4 scan identity and orientation (job 614578, run `20261107-1920-2bf3051-tf3-link`):**
  all 527 P/F/S TF4 volumes are **voxel-identical** to `ToothFairy3<C>_<id>_0000` after flipping the
  TF3 array on axes 1 and 2 (y, z); headers: TF4 LAS, TF3 LPS (headers only differ in y → the z flip
  is unrecorded). TF3 labels lie on the TF3 image grid (527/527). Anatomy with TF3 labels in each
  header's RAS world: upper jaw above lower jaw in TF4 frame for 224/228 testable cases (TF3 frame:
  4/228) → **TF4 headers are right for superior/inferior; TF3's are upside down.** Exceptions, all
  centre S: S0002, S0004, S0015, S0034 (TF4 frame says upper below lower) → visual check needed.
  Left canal (label 3) at larger world x than right canal (4) in 527/527 cases in both frames →
  either both headers mirror left/right or TF3 names sides from the viewer; unresolved — test later
  by report-vs-mask laterality agreement (FDI quadrants). **To use TF3 masks on TF4 scans: flip
  label arrays on axes 1 and 2, then use the TF4 affine.** 299 cases lack jaw labels.
- **Left/right still unresolved; prior work inconclusive (2026-11-07).** `~/U-Mamba2/quadrant_mirror.csv`
  (edentulous-quadrant test): undecided 529/532 (P: 1 MIRRORED, 1 OTHER; F: 1 AGREE).
  `check_flips.py` (implant FDI sites, report vs segmentation facts, as-is vs mirrored) and flip-test
  logs 550053/550065 (A set): transforms with a z flip match the reports, but flip-x and no-flip-x
  both match → laterality not identified. Plan: our own report-vs-TF3-mask laterality test (implants,
  missing teeth, all P/F/S) after per-report findings exist.
- **The 4 S "upside-down" flags (tooth/canal labels, TF4 frame):** S0004, S0034 — upper teeth above
  lower teeth → scan fine, jaw-label artefact. S0002 — canals below lower teeth, "upper jaw" label is
  a 9,768-voxel fragment → scan fine, label artefact. **S0015 — canals above lower teeth and an
  81,556-voxel upper-jaw label below the lower jaw → probably stored upside down in TF4**; Shivam to
  confirm visually on the cluster (images are not viewed by Claude: DUA).
- **Release version:** zip entries dated 2026-04 (15 volumes), 2026-05 (reports, dirs), 2026-07
  (607 volumes, reports), latest 2026-07-29 → our copy is a late-July-2026 build; the official page
  (625 cases / 1001 reports / max 3 per case) likely describes another version.
- **GPU envs built (job 614588, 2026-11-07), no container:** `pr_serve` — vllm 0.19.0, torch 2.10.0,
  transformers 4.57.6 (as in the submitted image); `pr_train` — torch 2.11.0+cu130, transformers 5.9.0,
  peft 0.20.0, accelerate 1.13.0, flash-linear-attention 0.5.2, no gptqmodel. `pip check` clean for
  both; lock files `env/pr_serve.lock.txt`, `env/pr_train.lock.txt`. CUDA not yet exercised (GPU
  smoke run pending).
- **Models (job 614596):** `Qwen/Qwen3.5-9B` (bf16, for retraining) → `models/Qwen3.5-9B`, commit
  `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, 19.3 GB. `lucent517/v2v2r_cbct_vqa` (released model =
  LoRA merged into AWQ) is a **gated** repo → needs Shivam's HF token in `~/.secrets/hf.env`.
  Source of the plain AWQ base `models/Qwen3.5-9B-AWQ` (merge target for retrained adapters) is
  unknown → ask Shivam; fallback: rebuild it from the released checkpoint by restoring the bf16
  visual/attention tensors from `Qwen/Qwen3.5-9B` (bit-identical by AWQ's skip list, per
  `merge_vision_lora.py`).
- **Facts from submitted code (`external/odin2026/config.py`):** (1) the facts the released VLM was
  trained on mislabel 3 of 4 quadrants (sequential seg labels read as FDI); inference reproduces this
  on purpose (`REMAP_MODEL_TO_FDI=0`) **[wrong, corrected 2026-11-11 below: the mask is already FDI]**.
  Per-fold retraining could use corrected facts — Shivam's
  decision. (2) Orientation lead: "per the ToothFairy4 dataset correction, the reports align with the
  RAW image content, so we do NOT reorient by affine" → relevant to the left/right question.
- **`external/odin2026/inference.py` edited (first change to the imported code):** `INPUT_PATH` /
  `OUTPUT_PATH` read `INPUT_DIR` / `OUTPUT_DIR` env vars (defaults `/input`, `/output` unchanged) so the
  pipeline runs outside the container. No behavioural change.
- **What the masks can confirm (TF3 labels):** per-tooth presence (64 tooth + 32 pulp labels),
  implant (10), crown (9), bridge (8), jaws (1, 2), canals (3, 4, 103–105), sinuses (5, 6), pharynx
  (7). No label for fillings, root canal fillings or condyles. Schema v7.1 has explicit visibility
  fields (`maxilla_scope.maxilla_included`, sinus/condyle `scope`).
- **Models downloaded (job 614599, HF token from `~/.secrets/hf.env`):** `lucent517/v2v2r_cbct_vqa`
  @ `065348c6a8ea3da85f6504841246f5ed4a442447` (12.4 GB, released model, smoke test only);
  `QuantTrio/Qwen3.5-9B-AWQ` @ `938f8e3ef86c9d1e9bec3705e149694c172592f1` (12.4 GB, AWQ base = merge
  target, source confirmed by Shivam); `Qwen/Qwen3.5-9B` @ `c2022362…` (bf16, training base).
- **Smoke run submitted (job 614600, approved):** `jobs/smoke_v2v2r.sbatch` on P001, F001, A003,
  MIG `3g.20gb`; environment test only, no metrics.
- **Smoke run 614600 FAILED for the VLM (correcting an earlier premature "passed").** Segmentation
  (58–77 s/case on MIG 3g.20gb) and fact extraction worked for P001, F001, A003; vLLM 0.19 crashed at
  startup in every case: `ValueError: invalid literal for int(): 'MIG-…'` (it parses
  CUDA_VISIBLE_DEVICES as integers; SLURM gives MIG slices a UUID). The written reports were the
  template fallback (`REPORT_FALLBACK_ON_ERROR=1`), not VLM output. Fix: full A100 40GB (`--qos=a100
  --gres=gpu:a100:1`, next size up per the GPU policy, justified by this evidence); smoke job now sets
  `REPORT_FALLBACK_ON_ERROR=0` and checks `competition_runner exited 0` per case.
- **Visibility rules adopted (Shivam, 2026-11-07):** three tiers + per-group rules incl. jawbone-based
  condyle rule; written into HANDOFF §2.6. Thresholds frozen after the image-facts job.
- **Lumetti et al., "Ontology-Grounded Structured Prediction for Dental CBCT Reporting" (MICCAI paper
  0618; repo `AImageLab-zip/CBCT-Report`)** — closest prior work, on the same ToothFairy3 reports
  (893 reports / 529 TF3 cases). Facts: (1) reports written by **2 radiologists + 6 maxillofacial
  surgeons (8 readers)** → our `report_x` (~15 distinct values within P/F/S) is probably **not** a reader
  ID; reader-level analysis needs real reader IDs (ask organisers). (2) Released: ontology
  (`ontology.ttl`, 15 finding types with tooth/side/quadrant slots and qualifiers, incl. ToothAbsence,
  MandibularCondyleFinding; no visibility class), SHACL shapes, **per-patient consolidated** labels
  (`data/ttl_reports`, clinician-validated); per-report extractions (`ttl_reports_single`) **not** released.
  (3) Labels built with GPT-5.2 from both reports consolidated (union). (4) Human agreement on 358
  two-report patients: pres-F1 0.81, inst-F1 0.70, rep-F1 0.67 (a published noise floor / evidence of
  partial references). (5) GT vs predicted masks: char-acc 85.4 → 51.0 (segmentation quality matters).
  Repo files read: README, API directory listings, `ontology.ttl` only (no report or TTL instance text).
- **Use of Lumetti et al. (Shivam, 2026-11-07):** (1) their finding ontology and metric definitions
  are the evaluation "ruler" — re-implemented in our code and cited, files not copied (repo states no
  licence); a published, clinician-designed vocabulary not built around V2V2R is the fairer ruler.
  (2) Their released per-patient labels may be used **internally** to check our report parser (never
  redistributed, never presented as our result). (3) Their baseline model is **not** used; V2V2R
  (+ ablations, controls, our decoder) stays the centrepiece. (4) **Reader-level analysis dropped**
  (no reader IDs; organisers cannot be contacted); centre level only. HANDOFF M1 step 4 and M2 step 1
  updated.
- **M0 smoke run PASSED (job 614601, full A100 40GB, released weights, environment test only):** the
  submitted V2V2R pipeline ran end to end without container on P001, F001, A003 — segmentation
  (`umamba2` env), facts, vLLM 0.19 (`pr_serve`) + VQA + report, `competition_runner exited 0`
  (`VLM_OK`) for all three, fallback disabled; 5.0–6.6 min/case; valid JSON reports (91–103 words).
  No metrics computed. Note: segmentation alone runs fine on a MIG slice; vLLM needs a full GPU.
- **Go/no-go test GT inserted before further heavy compute (Shivam, 2026-11-08).** Prompted by an
  honest risk review: (1) two-reader disagreement may be partly false calls / interpretation, not
  omission (only masks can check, tier 1 only); (2) parser errors mimic omission; (3) per-finding
  counts allow solid π only for ~6–8 common finding types pooled over centres; (4) "so what" risk if
  no conclusion changes. GT = parser check (T1), π estimates (T2), mask agreement (T3), effect size
  (T4), with proposed pass/fail thresholds and decision rules in HANDOFF §7 (GT). No V2V2R/segmentation
  retraining or decoder work until the GT decision. Thresholds are proposals until confirmed.
- **GT adjustments (Shivam, 2026-11-08):** parser input = **English** reports (no one in the team reads
  Italian, so manual checks must be in English); translation noise measured automatically by parsing
  the Italian originals too and comparing per-report findings. Few-shot examples synthetic (written by
  Claude). GT thresholds are guidance; decision after seeing the numbers, reasoning recorded.
- **Parser T1 first run (job 614962, Qwen3-14B, parser v1, 2026-11-08):** 1000 EN + 1000 IT + EN repeat;
  16/15 reports truncated at 2048 output tokens. Slot-level F1 vs Lumetti (EN, 519 cases): 0.817 overall
  (type level 0.917); ≥0.90 only for canal 0.977, restorations 0.940, endodontic 0.933 (caries 0.889).
  Run-to-run F1 0.985 (negligible vs ~0.70 inter-reader). EN vs IT 0.858 (EN matches Lumetti better than
  IT: 0.817 vs 0.796). Diagnostics (`partialref.extract.parser_diagnostics`, `convention_checks`, counts
  only) → mostly convention mismatches: Lumetti codes tooth-specific periodontal defects as
  ToothDevelopmentPathology; pontics as ToothAbsence + ProstheticDevice with contiguous spans; edentulous
  arch = all 16 teeth absent; their quadrants/canal sides agree with FDI digits (96/96, 349/357) while our
  parser never attached teeth to side findings (mirrored misses: canal 31/37, quadrant 49/77); plus parser
  over-calls of condyles (15 in P/F/S), sinus findings and planned implants. → parser v2 prompt (commit
  7e79d28), 4096-token limit.
- **Parser development split (2026-11-08):** cases hashed into 'tune' and 'heldout' halves
  (`score_parser.parser_half`); further prompt changes may only look at 'tune'; T1 is reported on
  'heldout'. v1 baseline slot F1: tune 0.805, heldout 0.830. (v2 rules were derived from aggregate
  counts over all cases — modest leakage, stated.)
- **Lumetti ontology version (2026-11-08):** paper = 13 finding types (5,703 instances); released
  ontology (repo commit e809ec3) = 15 types (adds ToothAbsence, MandibularCondyleFinding); released labels
  = 14 types occurring, 7,434 findings = 5,703 + 1,731 ToothAbsence (exact) → labels regenerated with
  the extended ontology (our inference from the counts); no condyle findings (condyles outside the thin
  TF3 scans). We use and cite the released version.
- **Lumetti repo has no licence → handling (2026-11-08):** default all-rights-reserved. We use ideas,
  metric definitions and the finding vocabulary with citation; our code, prompt and examples are
  independent (their extract.py, extraction_guide.ttl, example_*.ttl, shapes.ttl and src/ were never
  opened). Their unused files were removed from the cluster copy (`~/Datasets/CBCT-Report`, sparse
  checkout now `data/ttl_reports` only). Their labels: internal parser check only, never redistributed,
  only aggregate agreement reported. Conservative fallback if needed: manual 30-report parser check.
  Attribution added to `src/partialref/extract/fewshot.py`.
- **Independent review of T2 code (subagent, code + synthetic data only, 2026-11-08) → fixes:**
  (A1, severe) pandas 3 reads missing parquet strings as NaN, so `error is not None` marked every report
  as failed when any report failed → T2 would silently use 0 cases; fixed with `notna()` + regression test.
  (A2) non-identifiable π (no instance with n≥2, k≥1) still returned finite π / narrow CIs that could pass
  GT flags → now NaN (also per bootstrap resample); GT flags require ≥10 two-reader "both" instances,
  ≥30 double-read mentioned instances and ≥95 % identified resamples. (A3) condyle sides used world-x
  halves while canal/sinus sides use TF3 label names (recorded as opposite) → one condyle value for both
  sides until laterality is resolved. Also: default exclusions P116/P117/P260/P261 + S0015; headline fit on
  double-read instances (all-cases fit = sensitivity row); per_report extractions table built from all
  per-report files. Reviewer's synthetic end-to-end check re-run on fixed code: 1 failed report now drops
  1 case (59/60 used). 70 tests pass.
- **Provisional visibility rules for GT (recorded as HANDOFF §7 GT requires):** tooth present for Caries,
  EndodonticTreatment, ConservativeRestoration, ToothDevelopmentPathology, PeriapicalLesion (+ root tip
  not cut); missing-tooth rule (§2.6: between present teeth of the quadrant, or quadrant teeth not
  touching a lateral/posterior face) for ToothAbsence, DentalImplant, ProstheticDevice,
  PostExtractionSocket, RootResidue; canal/sinus: label present (deviation from §2.6: sinus %-of-complete
  rule needs a reference volume, canal "not mostly cut" not yet quantified); condyles: both mandible halves
  touch neither top nor posterior face; quadrant findings: jaw in the scan (label or teeth). Open risks
  noted by the reviewer: tooth-level laterality of TF3 masks (test via `--mirror-teeth`, compare
  mentions on invisible slots); pooling over slots/centres biases n=2 π upwards (conservative for T2).
- **TF3 tooth masks are left/right mirrored relative to the reports' FDI numbers (T2 laterality check,
  2026-11-08).** Same parse (v2, Qwen3-14B, run `results/extract/20261108-2311-8113799-parse-qwen3-14b-n0-614989`),
  same visibility features; only `--mirror-teeth` differs. Reported findings on teeth the mask calls
  absent, normal → mirrored: EndodonticTreatment 279 → 86, ConservativeRestoration 167 → 28,
  ToothDevelopmentPathology 240 → 83, PeriapicalLesion 70 → 34, Caries 36 → 9 (arch-level types
  unchanged, as expected). Mirrored is clearly right → for P/F/S (TF3-derived masks) tooth IDs are
  mirrored (1↔2, 3↔4) before any comparison with reports. Centre A (our own predicted masks) is not
  mirrored until checked the same way. Canal/sinus sides are not mirrored yet (canal is visible on both
  sides in nearly all cases, so this check cannot tell). π changes ≤ 0.015 for every type (pooling hides
  it), but per-slot analyses (T3) need the mirror. Evidence:
  `results/propensity/20261108-2349-66fd4d1-propensity-gt-v2{,-mirror}/summary.json`.
- **Laterality diagnostics (job 615021, 14B parse v2, 2026-11-09), evidence `results/propensity/20261109-0011-375148b-laterality/`.**
  (1) Per case, informative cases (≥1 reported tooth finding on a mask-absent tooth, either way): mirrored
  mask fits better in 214, unmirrored in 13, tie 37 (P 161/9/29, F 30/3/6, S 23/1/2) → the flip is a
  dataset-wide convention, not individual scans. Unmirrored better by ≥2: F028, F055, P230, S0019 →
  candidates for individually flipped scans or misfiled reports; added to the visual check list with
  S0015, not excluded yet. (2) Residual after mirroring: 214 instances; 162 reported by one of several
  readers (reader or parser error), 25 by ≥2 readers (mask/visibility problem: Endo 15, TDP 7, Resto 3),
  27 in single-read cases. (3) Reader-pair side swaps (36 vs 46) 22 vs up/down control (36 vs 26) 34, out of
  3,128 unmatched mentions in 1,346 reader pairs; neighbour shifts 57 → no evidence of side confusion by
  clinicians above chance; numbering confusions explain < 2 % of between-reader disagreement, so
  disagreement is overwhelmingly omission (what π measures). Re-run on the frozen parser.
- **Centre A predicted masks v1 are invalid at tooth level (bug in our jobs/seg_centreA.sbatch, found
  2026-11-09).** `run_segmentation` already returns the predictor's FDI scheme
  (`task1_inference.BasePredictor.postprocessing`: teeth 11-48, canals 51-53, pulp 50); our job applied
  `train_to_original` (sequential → TF3) a second time. Effect: tooth IDs shifted (FDI 21-26 → 23-28, 27/28 →
  31/32, 31-34 → 35-38, 35-38 → 41-44, 41/42 → 47/48, 43-45 → 103-105, 46 → 199), FDI 47/48 and the
  incisive/lingual canals and pulp lost (→ 0). Evidence: in
  `results/audit/20261108-2033-b59b375-dirty-visibility-features-A/tooth_features.parquet` FDI 21, 22, 33, 34,
  45, 46 never occur in 95 cases — exactly the IDs this double mapping cannot produce. Not used in any GT
  result (T2 and laterality ran on P/F/S only). Mandible, maxilla, IAN canals, sinuses (ids 1-6) were
  unaffected, so the earlier A sinus/condyle visibility shares stand; tooth-based A features (root tips,
  quadrant edges) do not. Fix: compose the predictor's sequential→FDI rule with `train_to_original`
  (teeth unchanged, 51/52/53 → 103/104/105, 50 → 199), fail on unexpected labels, new output dir
  `results/seg/centreA_umamba2_foldall_v2`; re-run segmentation + A visibility features.
- **V2V2R already corrects left/right at inference** (`external/odin2026/segmentation.py`, verified in Aug 2026
  on F030/P061/A046 single-implant reports): TF4 volumes are stored L-first (LAS/LPS), the model was trained on
  R-first TF3 arrays → flip array axis 2 for L-first input and flip the mask back. Same root cause as the
  TF3-label vs report mirroring found in T2 (214 vs 13 cases), now confirmed at scale. Consequence: centre A
  masks from `run_segmentation` are already in report convention → `mirror_centres` stays "PFS" (A not
  mirrored), to be confirmed by the laterality check on A once v2 masks exist.
- **Independent review of all code incl. parser (subagent, code + synthetic data only, 2026-11-09) → fixes,
  then fix verification by the same reviewer.** Parser: schema-valid output that breaks the per-type anchor
  rule no longer loses the report (salvage per finding: side/quadrant derived from tooth digits, tooth ids
  split per side; unanchorable findings dropped and counted); a report with any dropped finding counts as
  failed everywhere (T1 universe, T2/laterality/audit), since a dropped finding looks like an omission;
  atomic per-report writes. Scoring: the scored universe is fixed in advance (fully parsed cases ∩ Lumetti
  files) instead of "cases with ≥1 instance in both" (which hid misses in empty/failed cases); run-to-run and
  EN-vs-IT checks are now per report (case, report, type, slot) — the earlier "run-to-run F1 0.985" was a
  per-case union and understated per-report parser noise; Lumetti type level uses findings.parquet (56
  findings without anchor kept). Visibility: a quadrant without mask teeth is assessable only if the whole
  jaw (both halves) stays clear of lateral/posterior faces (stricter than §2.6's per-half wording, chosen
  because the jaw-half ↔ report-side mapping is not settled; check the share of edentulous arches lost once
  features are re-run). side_geometry: quadrant ↔ report-side mapping per centre (mirrored P/F/S vs report-
  convention A). DUA: vLLM request-log check in the EXIT trap; audit sheet without free text, .tsv, excluded
  from rsync. Not a problem: report file names (2000/2000 parse). All T1 numbers so far were produced with
  the old scorer and will be re-scored.
- **Overnight results 2026-11-09 (jobs 615028-615031, git 794924d; run ids carry "dirty" only because
  reports/STATUS.md was uncommitted).**
  (1) **Parser re-scored with the fixed scorer** (`results/extract/*-rescore-*`). Slot F1 vs Lumetti, EN:
  14B tune 0.815 / held-out 0.837; **32B tune 0.845 / held-out 0.848** (IT 0.830 / 0.836); type level
  held-out 14B 0.938, 32B 0.948. Per report: **run-to-run (32B, temperature 0) 0.988**; **EN vs IT 0.885**
  (lowest: ToothDevelopmentPathology 0.77, PathologicalLesion 0.70, PeriapicalLesion 0.80). Failed reports
  32B: EN 4, IT 6 (14B EN 17); all normal length (generation loops), so they are a parser problem to fix.
  (2) **Side geometry** (`results/audit/*-side-geometry`): with tooth labels mirrored, canal labels are on the
  opposite side in 499/500 determinable P/F/S cases (sinus 47/51) → TF3 side labels (canal, sinus) follow
  the same mirrored convention as the teeth; TF4 header left/right agrees with the reports (494/495). →
  canal/sinus Left/Right swapped for P/F/S (`swap_side_centres = PFS`).
  (3) **Empty-quadrant rule (review #4)**: jaw fully clear of lateral/posterior faces: mandible 414/527
  (P 73.5 %, F 96.8 %, S 96.2 %), maxilla 224/228 → the stricter rule costs about a quarter of P mandibles
  for toothless quadrants only; accepted.
  (4) **Report-scan fingerprint** (`results/audit/*-report-scan-match`, 32B EN): median own contradiction
  rate 0 in every centre; own scan best fit for 81-84 % of reports. No evidence of misfiled reports: the 16
  reports of P116/P117/P260/P261 all fit their own scans (P260/P261: 10-14 agreeing items, ≤ 2
  contradictions); P120/P121/P262/P263 exist only in TF3, so "same patient scanned twice" is untested.
  30 misfiled flags are not misfiling: the largest come from **8 reports in which the parser marks a whole
  arch (16 teeth) absent while the mask has ≥ 6 teeth there** (report prefixes 30, 29, 32; other reader of
  the same case fits) → suspected parser over-firing of the "edentulous arch = 16 absent" convention;
  to be checked by Shivam on the report text. 66 of 76 whole-arch absences agree with the mask (0-2 teeth).
  (5) **Centre A segmentation v2, 3-case check** (job 615031): A003/A004/A005 have teeth in all four
  quadrants (25/21/30 teeth; A005 has 41-48 incl. 47/48), canals 103-105 and pulp 199, no unexpected
  labels → label fix works; ready for all 95.
- **Parser v3 (2026-11-09), reviewed (subagent; must-fix applied).** Cause found with Shivam's reading of two
  reports (no text recorded): the Italian abbreviation "e.d." (elemento dentario = tooth) survives in 33 EN
  translations (224 IT reports use it, 111 "elemento/i dentari"); the parser read "e.d." as "edentulous" and
  turned "teeth present from X to Y" into 16 ToothAbsence. In reports with "e.d.", 104/174 (60 %) of
  ToothAbsence calls fall on mask-present teeth vs 264/3065 (9 %) elsewhere. v3: prompt defines
  e.d./el./elemento dentario = tooth, presence statements give no finding, ranges across the midline run
  through it, missing teeth without numbers → ToothAbsence with `region` (UpperArch/LowerArch/QuadrantI-IV,
  slot "region:<R>"), absent-vs-present consistency instruction + per-report conflict count, new synthetic
  example 5; retry ladder (penalty 1.1 → 1.3 → 1.05 with temperature 0.3, seed 0) for failed parses AND for
  salvaged parses that dropped findings; region slots excluded at slot level vs Lumetti, kept at type level.
  Post-run checks: full-edentulous arches still 16 teeth (not region), v2→v3 ToothAbsence change, conflicts,
  share of reports ending on the sampled retry.
- **Estimand for region-level mentions (Shivam, 2026-11-09): main = B, tooth-level completeness of the
  reference** — a vague "partial edentulism" reader counts as not mentioning the unnamed teeth (plain n),
  because corrected metrics compare findings tooth by tooth. **Secondary = A**, reporting propensity of
  readers who name teeth (`--region-mode uninformative`: that reader drops out for the unnamed teeth; a tooth
  the same report names with a presence-implying type stays informative). Both reported, with the share of
  reader-tooth pairs affected.
- **Parser evaluation without Lumetti numbers (Shivam, 2026-11-09).** Lumetti et al.'s labels never enter π,
  visibility or the metrics; they were only a parser check. Their repo has no licence and the authors cannot
  be contacted → **no agreement numbers against their labels are reported** in the paper; internal use as an
  early warning continues. Their ontology and the 9 conventions aligned with their released labels (bridge
  spans incl. pontics; pontic = also ToothAbsence; tooth-specific bone loss = ToothDevelopmentPathology;
  toothless arch = 16 teeth; side/quadrant from tooth digits; tooth ids on canal/atrophy findings; condyle
  only if abnormal; sinus only pathology; planned implants not findings) are cited and listed in the paper,
  stating they were derived from aggregate label counts. **T1 is judged by Shivam's blinded review against
  the report text** (HANDOFF §7 T1 updated), plus mask checks and consistency (run-to-run, EN vs IT, two models).
- **Human review plan (Shivam, 2026-11-09), staged:** (1) after the parser freeze, ~60 held-out reports with
  extra reports for rarer types (T1, ~3-5 h); (2) if GO, all ~740 double-read P/F/S reports reviewed and
  corrected in the app → π from human-verified per-report findings (main), parser-only as sensitivity;
  (3) optional single-read/centre A reports; throughout, ~5 % re-reviewed blind after a week (intra-rater
  consistency). Review only the frozen parser. Correction mode in the app is built when stage 2 starts.
- **Local review on Shivam's own laptop (Shivam, 2026-11-09).** Shivam approves his own machine as a location
  for ToothFairy4 report text (public dataset, not shared further) → `partialref.review.export_bundle` packs
  report texts (EN + IT), the frozen parse's per-report findings and a relative-path report table into
  `results/review/bundles/*.tar.gz` (mode 600, git-ignored, not matched by the weekly rsync); copied with scp,
  no cloud sync. The app runs locally on 127.0.0.1 (`python -m partialref.review.local <bundle>`); the review
  log (IDs and codes only) comes back by scp and is validated by `partialref.review.import_log` (unknown IDs,
  unexpected fields, free text, invalid codes rejected) before use. The cluster version (jobs/review_app.sbatch,
  token in an owner-only file because the home directory is group-readable) stays as an alternative.
- **T1 criterion replaced (Shivam, 2026-11-09).** The old guidance "slot F1 ≥ 0.90 for ≥ 6 common types" was a
  conventional bar proposed on 8 Nov, not derived from data, and measured agreement with Lumetti rather than
  correctness. What matters is whether parser error changes the conclusion about π: a missed mention looks like
  an omission (observed π ≈ true π × parser recall; false additions push the other way), so the tolerable error
  depends on how far π is from 1 (caries π 0.26 survives a recall of 0.85; root canal treatment π 0.83 would not).
  New T1, per finding type: parser recall and precision from Shivam's blinded review (with CIs) → π corrected for
  parser error with uncertainty propagated → the type passes if the corrected π's upper 95 % CI is < 0.90; failing
  types are reported as "parser-limited" and kept out of headline claims; plus no systematic error pattern.
  HANDOFF §7 T1 row and decision rule updated. The same "does it change the conclusion" logic is to be applied
  when T3/T4 guidance values are reviewed.
- **Literature / novelty check and plan additions (Shivam, 2026-11-09), `reports/LITERATURE_NOVELTY.md`.** Three
  parallel web searches (report-generation evaluation; statistical methods; dental report generation). Verdict:
  nobody found estimating per-finding reporting propensity from double reads to correct report-generation
  *evaluation*, or using masks to decide whether a reference's silence is evidence of absence; but the problem is
  recognised concurrently — EviDent-CBCT (arXiv 2610.02375, ToothFairy4, hand-set non-mention weights for training)
  and PU-DPO (arXiv 2608.05341, CXR, union-of-readers truth, training only) — and the statistics (Dawid–Skene,
  Hui–Walter, capture–recapture) and propensity-scored / PU metrics are established → framed as a principled
  transfer with new elements. Added L1–L13 to HANDOFF §7 (refinements inside existing milestones; L2 and L11
  optional), incl. a second (PU-style) corrected-precision estimator, heterogeneity and correlated readers in the
  simulation, Rogan–Gladen parser correction, random (not discrepant-only) adjudication, and an analysis-plan
  freeze (`reports/ANALYSIS_PLAN.md`) before final runs. M4–M6 kept as planned.
- **T1 failure rule made concrete (Shivam, 2026-11-09):** T1 fails if a systematic error pattern appears or if
  fewer than 3 finding types survive parser correction with upper 95 % CI of π < 0.90 (same count as T2's guidance).
- **V2V2R challenge paper read (2026-11-09; Singh, Lu, Sagl, ODIN 2026 challenges 024 + supplement).** Already
  published: masks as spatial index, FOV statements in prompts and constraint rules, silence rules (fields with
  precision < 0.30 off: crowns, caries, post-and-core, root remnants), union-of-reports ground truth, Table S3
  inter-reader agreement. Its discussion already names reporting habits / missed findings as label noise → our
  project quantifies and corrects that. Table S3 implies case-level two-reader π (endodontic ≈ 0.94 … caries ≈ 0.37;
  case-level counting inferred from the numbers, script not in the submitted code → confirm with Chen). Added
  HANDOFF L14: re-test the silence rules under corrected metrics; compare our tooth-level π with Table S3.
  `reports/LITERATURE_NOVELTY.md` §1b added.
- **Parser v3.1 (2026-11-10).** Reason: in the v3 run (`results/extract/20261109-1742-42d9d35-parse-qwen3-32b-v3-n0-615200`)
  toothless arches were coded as a region instead of 16 teeth (17 region mentions on arches the mask shows as
  toothless; 117 Lumetti absences lost, 98 of them in region cases) and 96 reports had a tooth coded missing AND
  present. Change: toothless arch/quadrant = all its teeth, "edentulous except X" = all but X, region only for
  partial edentulism or a sector/saddle/span/area without numbers, whole-arch ridge/jaw = 16 teeth; residual root
  and impacted tooth are never ToothAbsence; synthetic example 6. Conflicts are RECORDED per kind
  (`absence_conflict_details`), not resolved: the mask check of conflict teeth showed most are really absent
  (RootResidue 35 absent / 20 present, periodontal TDP 21 / 6, deciduous 4 / 0, eruption 4 / 4), so automatic
  removal would be wrong; `apply_absence_convention` exists but is applied nowhere (later sensitivity row). The
  counts above are ad-hoc checks on the v3 run from this session, not saved as a results file; the post-run
  checks on v3.1 will write them. JSON schema unchanged from v3. Reviewed twice by an independent agent (GO) and
  re-verified before submission (102 tests, schema identical to HEAD, token headroom, parquet round trip).
- **External model may read tune-half report text (Shivam's written confirmation, 2026-11-10).** Terms checked the
  same day: ODIN 2026 datasets page and the Ditto ToothFairy4 page state CC BY-NC-SA 4.0, sign-up to download and
  a citation requirement; no separate data-use agreement or clause on third-party processing was found, and the
  providers themselves state the English reports are gpt-5.2 translations. Not checked by Claude: anything shown
  at Ditto account creation, and the Claude account's data settings (Shivam's responsibility). Not MedUni Wien
  data, so no institutional data policy applies (Shivam). Scope: Claude reads report TEXT of the tune half only,
  for parser error analysis (disagreement-ranked reports first, then a random sample of ~50 agreeing reports,
  extend to the whole tune half only if that sample shows many hidden errors). Held-out half stays unseen;
  no images/masks; parser, labels and metrics stay local-model only; report text never in commits, logs, results
  or prompt examples (examples stay synthetic). The paper states that an external model assisted error analysis
  during parser development. HANDOFF §0.8 and CLAUDE.md rule 8 updated.
- **Parser v3.2 (2026-11-10).** Reason: v3.1 (job 615640, cancelled after the EN pass,
  `results/extract/20261110-1548-42c1f97-parse-qwen3-32b-v31-n0-615640`; evidence, codes only, in its `diagnosis/`
  folder) did not improve ToothAbsence against Lumetti's labels (slot F1, tune half: v3 0.784, v3.1 0.782) and
  coded more arches as a region instead of teeth (lower arch as region: 19 reports in v3, 54 in v3.1; upper 27
  and 29; all 1,000 EN reports, `transitions_lower.csv` / `transitions_upper.csv`).
  Causes, found by reading tune-half report text next to the codes, the masks and Lumetti's labels: (1) scan
  coverage wording ("maxilla partially included") gave a region (140 tune reports with that wording and no
  edentul* word: 0 region codes in v3, 11 in v3.1); (2) "complete edentulism" with the jaw only in the section
  heading gave a region instead of 16 teeth; (3) "only tooth X present" gave a region instead of all other
  teeth; (4) edentulous tooth-group sectors on both sides gave one arch region; (5) "edentulous saddle distal to
  X" gave an arch region; (6) numbered missing teeth got an extra region. Change: the model no longer writes out
  a toothless arch. It states `region` + `region_scope` (AllMissing | SomeMissing) + `remaining_tooth_ids`, and
  the code fills `tooth_ids` for AllMissing (teeth of the region minus the remaining ones). SomeMissing, or a
  region without scope (all v3 / v3.1 files), stays a region slot, so stored runs validate unchanged.
  Conventions: a whole tooth-group sector with its side stated (incisor x1-x2, canine x3, premolar x4-x5, molar
  x6-x8) and "distal to X" are identified teeth, third molars included (as in Lumetti's labels and the masks on
  the tune cases read); a partly edentulous sector, an anterior / posterior / lateral sector and a tooth group
  without a side are SomeMissing; inside an AllMissing region every tooth with a tooth-present finding (caries,
  filling, root canal filling, periapical lesion, residual root, impacted tooth) is a remaining tooth, implant
  positions and pontics are not; scan-coverage wording is never ToothAbsence. Synthetic examples 7 and 8 added,
  4-6 state the scope. Also: `dropped_reasons` per report (finding type + validator message, never input values),
  `absence_regions_by_scope` in the run summary, `CASE_IDS` in `jobs/parse_reports.sbatch` (run dir and score
  tags get `-subset`; an unknown or empty id list stops the run), review app shows an AllMissing finding as the
  statement instead of 14-16 teeth. Independent review: GO for the tune-half run; its should-fix points
  (partly edentulous sector, teeth with findings in a toothless arch, optional convention surviving
  re-validation, subset labelling, review-app anchor) are fixed. Left open for the post-run check: third-molar
  false positives of the sector / distal rules, AllMissing arches against the masks, a tooth in both `tooth_ids`
  and `remaining_tooth_ids`, example key order, "maxilla" in not_assessable for a partly included maxilla
  (not used for π). Next: tune half, English only (312 cases, 508 reports), compared with v3 and v3.1 on the
  same cases; the full run only if that is clearly better.
- **Ontology: stay on Lumetti's finding types as they are (Shivam, 2026-11-10).** No own or extended ontology and
  no derived views (ToothDevelopmentPathology split, lesion merge, prosthetic split, explicit negatives) for
  now. Reason: comparability with the published labels and the challenge metric, and defensibility; a derived
  view would only help if sub-type π differ or cross-type mismatches are common, which is not measured. A cheap
  CPU check of both is planned after the parser is frozen; nothing changes before that.
- **Parser v3.2 tune-half result and freeze rule (Shivam, 2026-11-10).** v3.2 against Lumetti's labels on the
  tune half, English (312 cases, 508 reports): case level (type present in the case) micro F1 0.940, tooth
  level 0.854. Case level is ≥ 0.876 for 13 of 14 types; PathologicalLesion is the exception (0.699 case,
  0.575 tooth; boundary to PeriapicalLesion). ToothAbsence at tooth level is 0.767 (fp 380, from expanding
  toothless arches), while against the masks v3.2 is the best version (lower arch, 263 cases: F1 0.820 v3 /
  0.795 v3.1 / 0.875 v3.2). Evidence:
  `results/extract/20261110-1714-4142c99-parser-check-qwen3-32b-v32-tune-en-n0-subset-all/parser_vs_lumetti.csv`,
  `results/extract/20261110-1700-4142c99-parse-qwen3-32b-v32-tune-n0-subset-615701/diagnosis/README.txt`.
  **Rule:** no v3.3. The parser is frozen at v3.2 if, in the full run, the held-out per-type agreement with
  Lumetti is in the same range as on the tune half (differences of the size expected between two parsers).
  Weak types (PathologicalLesion) are marked "use with caution" or left out of the main π table, not fixed.
  Lumetti agreement stays an internal check (not in the paper) and does not replace the blinded review that
  the parser-error correction of π (T1) needs. Order: 5-case diagnostic EN+IT+repeat (job 615781), then the
  full run.
- **Translation errors in the English reports (2026-11-10).** Rule-based screen of all 1,000 IT/EN pairs plus
  Claude's reading of 99 tune-half pairs (DUA exception, tune half only). Confirmed meaning-changing
  mistranslations: 3 reports, all reader 30 (P045 30_1278, P049 30_1266, P022 30_1443): the Italian
  abbreviation for "dental elements" in a "teeth present" statement became missing teeth / edentulism. Random
  sample of 45 full tune pairs: 0 meaning-changing errors (95 % upper bound about 6–7 %). Of the rule flags on
  the tune half only the polarity flag found real errors (3 of 5); the others were false alarms of the rules.
  Held-out candidates from the rules are listed but not read. Evidence (IDs, flags, verdicts; no text):
  `results/audit/20261110-1737-4142c99-translation-screen/`. A local-model auditor for full coverage
  (`src/partialref/extract/translation_audit.py`, tests, job script) is written, reviewed and committed, but **parked and
  not run**: off the critical path; the reviewer's fixes (planted-error rules, recall counting, a faithful
  example, provenance per job, fail-fast) come before any run.
- **Paper focus: methods paper, π inside the decoder (Shivam, 2026-11-10).** The headline is the
  segmentation-aware joint decoder (M4) that uses the estimated reporting propensity π; the evaluation work
  (π with visibility, corrected metrics) is its foundation, not a separate paper. π enters the decoder in the
  co-occurrence factors (counts corrected for omission) and in the selection step (targets q and q × π,
  replacing V2V2R's hand-set silence rules), and the evaluation (corrected precision). Candidate additions,
  not yet decided: π-aware calibration (a non-mention is not a clean negative) and π-weighted non-mention
  loss in the LoRA training (EviDent-CBCT uses hand-set weights). Consequences: the GT gate stays, with the
  question "is π good enough to build the decoder on?" (T1 blinded review and T2 as planned, T3/T4 lighter);
  no retraining before that decision, as before; the split proposal can be prepared in parallel (no
  compute). Second domain (M5) and reader study (M6) become optional. **Venue is open** (MedIA not fixed;
  Artificial Intelligence in Medicine and others to be checked at the gate). The boundary with Chen's
  verifier-grounded generation paper must be agreed before decoder work starts.
- **V2V2R code audit in two parts, each before its retraining (Shivam, 2026-11-10).** The submitted code in
  `external/odin2026/` is checked for inconsistencies and bugs by several independent reviewer agents
  (read-only; no report text, no images), with two questions: what fits our plan as it is, and what must
  change. **Part 1, before the segmentation retraining:** segmentation training and inference
  (`segmentation.py`, `orientation.py`, `report_side.py`, `umamba2/`): orientation, header and affine
  handling, left/right and up/down conventions, label maps, mirroring in augmentation and test time, how the
  training cases are selected. Reason: many files have side swaps, header and orientation problems (TF3 vs
  TF4 headers, mirrored TF3 tooth labels, the double-mapped labels on centre A; entries of 2026-11-07 to
  2026-11-09), and a problem found after the retraining would cost the GPU job twice. Part 1 can run before
  the GT decision (no compute). **Part 2, before the per-fold LoRA retraining (M3):** VLM, facts, schema,
  post-processing and report assembly, including the open `facts.py` question (were the VLM training facts
  built from sequential labels?). Fixes follow §0.4: edit in place, one commit per fix that says why.
  Findings go to `reports/V2V2R_CODE_AUDIT.md`.
- **Parser v3.2 frozen (Shivam, 2026-11-11).** Full run: job 615806, run
  `results/extract/20261110-2029-f4df9e5-parse-qwen3-32b-v32-n0-615806` (Qwen3-32B, 80 GB A100, clean tree
  at `f4df9e5`): 1000/1000 reports parsed in EN, IT and the EN repeat, 0 errors, 0 server errors, 0 logged
  request lines; EN: 23 salvaged outputs, 12 findings dropped by the validator (all without tooth number or
  quadrant), 90 reports with a tooth both missing and carrying a finding (partner: RootResidue 68,
  EndodonticTreatment 34, ConservativeRestoration 18; 43 tune / 47 held-out reports). **Freeze rule
  (2026-11-10) met:** agreement with Lumetti's labels (internal check, not for the paper), EN, tune →
  held-out: type F1 0.941 → 0.944, slot F1 0.853 → 0.860 (`*-parser-check-qwen3-32b-v32-en-n0-tune`,
  `-heldout`). Largest held-out drops at slot level: Caries 0.962 → 0.898, EndodonticTreatment 0.960 →
  0.901 (type level unchanged; read as sampling variation). **Weak type, marked and not fixed:**
  PathologicalLesion (held-out 0.737 type / 0.621 slot). Not judgeable against Lumetti:
  MandibularCondyleFinding (no reference), MaxillarySinusFinding (13 references). Run-to-run (EN, per
  report, slot) F1 0.989 (v3: 0.984); EN vs IT 0.900 (v3: 0.885). No v3.3: prompt, schema, validator and
  model stay as in commit `4142c99`; any later change needs a new version and a new full run. Next: blinded
  review of ~60 held-out reports (T1); the missing-and-present conflicts are judged there.
- **Audit findings that the study depends on are fixed, each at its stage (Shivam, 2026-11-10).** Source:
  `reports/V2V2R_CODE_AUDIT.md`. Before the segmentation retraining: S1, S2, S3, S4, S7. Before predicted
  masks enter our pipeline: S5, S8, S10. Before the LoRA retraining: V1, V3, V4, V5, S6, V12, V13. Before
  baseline inference and scoring: V7, V8, V9, V10, V14. Before the decoder: S9, log-probabilities from the
  VLM call, captions without mask facts for the decoder arm. Not needed: dead and side scripts, the
  few-shot scripts, V2 if our own labels replace the reader merge. Rules: one commit per fix in
  `external/odin2026/` that says why, diff shown first; where a fix changes what the submitted system
  outputs, it gets a switch whose default is the submitted behaviour, so that the submitted and the
  corrected baseline can both be reported. Open decisions (Shivam): start weights, mirror bug, left/right
  convention, thresholds, baseline definition, silence labels.
- **GT gate stays on all P/F/S double-read cases (Shivam, 2026-11-10).** T2–T4 describe the reference
  reports; no system is trained, tuned or scored there, so the gate is not split into development and test.
  Two rules keep the locked test set clean: (1) any π that feeds or scores a system is refitted on the
  right subset (other folds for fold k; development set only for the locked test set and for centre A);
  the gate's all-cases π is never reused for that. (2) The gate decides go or no-go only; which finding
  types enter the final evaluation and how centres are pooled is set by a fixed rule applied to the
  development-set π and frozen in the analysis plan before the first test-set scoring. Pre-registration of
  the gate / analysis plan was discussed and set aside for now.
- **Correction of the 2026-11-07 entry "Facts from submitted code" (audit S10, 2026-11-11).** That entry
  says inference reproduces the sequential-labels-read-as-FDI quirk on purpose (`REMAP_MODEL_TO_FDI=0`).
  The code shows otherwise: the predictor already writes FDI numbers (`orig_mapping`,
  `umamba2/task1_inference.py:387-399`), so the mask is FDI with the flag at 0, and setting the flag to 1
  would map twice. The comments in `facts.py:4-14` and `config.py:39-47` are wrong and are corrected with
  the S10 fix. Still open (ask Chen): which label scheme the facts of the released adapter's training data
  used.
- **Correction of HANDOFF M3 (audit V9, 2026-11-11).** The switch
  `POSTPROCESS_CONFIG=configs/postprocess/no_source_rules.yaml` named there does not exist in the submitted
  code (`RULES` in `vqa/code/source_rules.py:71-82` is a constant). Per-rule and per-silence switches are
  part of the V9 fix; HANDOFF now says so.
- **Split draft prepared, not approved (2026-11-10).** From case IDs, centre, number of EN reports and
  manual-mask tooth counts only: locked test 107 of 527 P/F/S cases (20 %, 74 double-read), five folds of
  84, centre A (95) external, segmentation training list 425 (development + 5 TF3-only scans); strata
  centre × single/double-read, ordered by "teeth in both jaws"; seed 20261110. Open for Shivam: 20 % or
  25 %, the third stratum, whether the cases with an open visual check may be test cases (P242 is one in
  the draft), S0015 in segmentation training. Nothing is committed to `configs/splits/` before approval.
- **T1 review sample: fixed list, two strata (2026-11-11).** The review app only shuffled whole cases; the
  planned sample did not exist in code. New `partialref.review.sample` draws one list with a recorded seed
  (IDs and codes only): **base** = 60 random held-out reports of double-read P/F/S cases (default
  exclusions P116/P117/P260/P261/S0015), proportional by centre; every parsed finding gets a verdict and
  missed findings are entered → parser precision and recall per finding type. **extra** = for each type the
  parser found in fewer than 15 base reports, further random held-out reports in which the parser found
  that type; they are selected on the parser's own output, so they count only towards that type's
  precision, never recall and never another type. The app (`REVIEW_LIST`) follows the list one report at a
  time, shows verdict boxes only for the types an extra report was drawn for, has 6 rows for missed
  findings (was 3: a cap would flatter recall) and warns when a finding was left without verdict; the last
  saved record of a report counts. The bundle built with `--review-list` holds only the listed reports.
  The log format is unchanged. List drawn from the frozen run with seed 20261111: 348 reports / 174 cases
  in the population; base P 42, F 11, S 7; 53 extra reports (Caries, PeriapicalLesion, RootResidue,
  PostExtractionSocket 9 each, MaxillarySinusFinding 11, PathologicalLesion and DentalImplant 5 each,
  MandibularCondyleFinding 4 = all there are); 113 reports in total
  (`results/review/lists/review-list-20261110-2029-f4df9e5-parse-qwen3-32b-v32-n0-615806-s20261111.json`).
  Limits to state with T1: verdicts are per parsed finding, not per tooth; about 15 reports per rare type
  give wide intervals; MandibularCondyleFinding (5 reports) will not be estimable.
- **T2 result and T3/T4 code after independent review (2026-11-11).** T2 on the frozen parse (jobs
  616001/616002, `results/propensity/20261111-1056-6901b70-propensity-gt-v32-B` main, `-A` secondary, clean
  tree): 10 of 13 estimable types pass both guidance checks (upper 95 % limit < 0.90, width ≤ 0.30); e.g.
  Caries 0.27 [0.16, 0.40], ConservativeRestoration 0.47 [0.40, 0.54], ToothAbsence 0.70 [0.64, 0.75] (A:
  0.73), MandibularCanalCourse 0.89 [0.88, 0.91]; RootResidue and PostExtractionSocket too wide,
  MaxillarySinusFinding low support, MandibularCondyleFinding not estimable. Not yet corrected for parser
  error (T1). New code: `propensity/mask_agreement.py` (T3), `propensity/effect_size.py` (T4),
  `metrics/corrected.py`. The reviewer agent's findings changed the design: (1) for ToothAbsence a reader
  "mentions" a gap if it codes ToothAbsence, DentalImplant or PostExtractionSocket at that position (implant
  sites have no tooth label; the parser codes an implant as DentalImplant only); the ToothAbsence-only coding
  is an extra row. T2 is not rerun for this: positions no reader mentions only inform p, not π. (2) A
  two-reader π above the mask π is NOT evidence of false mentions: with φ = 0 the two-reader fit estimates
  E[π²]/E[π] ≥ E[π], so it means π varies between findings (two-reader π is an upper bound, HANDOFF L3);
  false mentions are judged by φ against the masks (upper 95 % limit). (3) ProstheticDevice is compared only
  for Crown / FixedBridge / SuspendedPontic; implants and fixed prostheses per case (no tooth number on those
  mask labels), plus rows with a single mask component. **HANDOFF §2.2 corrected precision is not adopted
  as is (Shivam: the handoff's methods may be changed):** for a system with skill it under-corrects, most
  for rare findings, and it does not remove the advantage a system gains by staying silent on under-reported
  types (`tests/test_gate_t3_t4.py`); T4 therefore reports raw − true precision next to corrected − raw, and
  a better estimator (HANDOFF L1) is needed before systems are scored.
- **T3 first results and the upper-jaw question (2026-11-11).** Run
  `results/propensity/20261111-1107-68c8ce5-mask-agreement-gt` (double-read): DentalImplant per case π_mask
  0.95 vs two-reader 0.97, φ_mask 0.006; ProstheticDevice (Crown/FixedBridge/SuspendedPontic) per case 0.77
  vs 0.78, but φ_mask 0.106 (upper limit 0.14 > 0.05); ToothAbsence per position 0.41 vs 0.73 (φ 0.017).
  Breakdown (codes only, scratch script, not kept): of the mask-absent positions in double-read cases,
  neither reader mentions 14 % in the lower jaw, but 96 % in the upper jaw of centre P (F 10 %, S 44 %);
  front versus back teeth does not explain it. Hypothesis: centre P's short scans (~5 cm) leave upper
  positions outside the scan, so a missing upper label means "not in the scan"; alternatives: readers skip
  the upper jaw, or annotators leave partly visible teeth unlabelled. Image-only rules tried on the upper
  positions (maxilla label present; no upper root tip cut; maxilla clear of the top face) do not separate
  them cleanly. T3 now reports ToothAbsence per jaw (reviewed). Open: a top-face rule in `slot_assessable`
  if the hypothesis holds. ProstheticDevice: 10 double-read cases where both readers name a fixed
  prosthesis and the mask has no crown/bridge label go to Shivam for a visual check (mask gap or
  parser/reader error); the case list is kept locally, not in the repo.
- **Visibility rule for empty upper-jaw positions (Shivam, 2026-11-11).** Shivam's visual check of short
  centre-P scans: where the mask has only a few upper teeth and no upper jaw bone, only the tips of those
  teeth reach into the picture and the rest of the upper jaw is not scanned (in another such scan both
  readers also state that the maxilla is not assessable). The old rule ("upper arch in the scan" = maxilla
  label OR any upper mask tooth) therefore counted unscanned upper positions as visible gaps. New main rule
  (`--upper-gap-rule bone`): for ToothAbsence, DentalImplant, ProstheticDevice, PostExtractionSocket and
  RootResidue, an upper position WITHOUT a mask tooth is visible only if the maxilla label has at least
  2000 mm³ (`--upper-bone-min-mm3`; chosen from image data only: just below the smallest maxilla of the tall
  centre-F scans, 2495 mm³, which show the whole upper jaw; P median 819 mm³, i.e. a thin strip). Positions
  with a mask tooth, the lower jaw, tooth-requiring and quadrant/side types are unchanged. The previous rule
  stays available (`teeth_or_bone`) and its T2/T3 runs of this morning are the sensitivity analysis.
  Expected effect from counts before the change: π ToothAbsence 0.70 → ~0.72, ProstheticDevice 0.60 →
  ~0.61. Reviewed by an independent agent (thin bone strips and centre-A false positives led to the volume
  threshold; `report_scan_match` keeps the old rule pinned for reproducibility).
