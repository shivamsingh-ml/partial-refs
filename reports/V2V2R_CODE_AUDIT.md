# V2V2R code audit

**Status 2026-11-10:** complete (Part 1 segmentation, Part 2 vision-language model, facts, report assembly
and evaluation). Nothing in the code has been changed; this is a list of findings and proposed changes for
Shivam to decide on (DECISIONS.md 2026-11-10, "V2V2R code audit").

**How it was done.** Seven independent read-only agents reviewed `external/odin2026/` and the training
set-up under `/msc/home/ssingh54/U-Mamba2`: three on segmentation (training pipeline, orientation and side
handling, open brief) and four on the rest (training labels and LoRA training, view generation, inference
and report assembly, open brief including the evaluation code). Claude then read the code behind the main
findings. No GPU job was run, no image or mask voxels were viewed, no report text was read, and
`vqa/code/fewshot_examples/` was not opened.

**The five points that matter most for the plan**

1. **No split is enforced anywhere.** Segmentation training uses every case (S1); the LoRA leak guards pass
   on nothing in our layout (V3); the released adapter was trained on 528 cases and the post-processing
   rules were fitted on the 40 validation cases (V4). Everything must be rebuilt per fold from a committed
   case list.
2. **Silence is a supervised negative in the training labels (V1).** This is the hand-set rule that a
   π-weighted loss would replace, and it means the model's probabilities already contain reporting habits.
3. **The decoder's inputs do not exist yet.** The segmentation predictor gives no probabilities (S9), the
   VLM call records no log-probabilities or failures (V8), and image captions state mask facts, so the
   two sources are not independent (V13).
4. **"Not imaged" and "not answered" become "absent" or "normal"** in the facts, the post-processing and
   the report text (S6, V8, V10, V11).
5. **A data-use review of the imported code is pending (V15).** Shivam does this himself.

**Evidence levels used below**

| Tag | Meaning |
|---|---|
| read | Claude read the cited lines and they do what is described |
| synthetic | an agent reproduced the behaviour on synthetic arrays, headers or inputs; Claude did not re-run it |
| results | taken from a file under `results/` |
| agent | reported by an agent from reading the code; Claude did not re-read it |
| open | not verified; a check is listed in the "checks not run" tables |

Paths starting with `EXT/` are under `external/odin2026/`, `NN/` under `external/odin2026/umamba2/nnunetv2/`,
and `UM/` under `/msc/home/ssingh54/U-Mamba2/` (outside the repo, no backup).

---

## Part 1 — Segmentation (U-Mamba2)

### 1. Summary

The submitted segmentation model can be retrained with the same trainer, network and plans. Four things
must be settled first:

1. **A case list.** Nothing in the pipeline restricts which cases are used, and `configs/splits/` does not
   exist. Without it the retrained model, its intensity statistics and possibly its starting weights see
   the test cases.
2. **The training recipe must be in the repo.** The conversion script, the config-injection script, the
   plans and the job scripts exist only under `UM/`.
3. **The left/right convention must be written down and kept.** The model names sides by array position,
   and one line at inference compensates. Changing the training labels or the reader without changing that
   line mirrors every mask with no error.
4. **The predictor cannot give probabilities.** The decoder (HANDOFF M4 step 3) needs segmentation
   confidence, and the submitted predictor returns only an argmax of a non-normalised accumulator.

Two bugs in the submitted code also affect our numbers today: the post-processing size thresholds differ
by side and by voxel spacing (S5), and the fact extraction reports teeth outside the scan as absent (S6).

### 2. Findings

**S1. No case restriction in training** — read
- `UM/scripts/convert_tf3_singlepulp.py` converts every label file; fingerprint and plans use every raw
  case; fold `all` trains on everything in the preprocessed folder (`NN/training/nnUNetTrainer/nnUNetTrainer.py:553-613`).
- ToothFairy3 has 532 cases (F 63, P 417, S 52).
- The intensity normalisation (clip −997 to 3608, mean 810.6, sd 1020.9) was computed over all 532 cases.
- There is no held-out validation: fold `all` validates on training cases and `NN/run/run_training.py:213`
  skips the final validation.
- **Change:** commit the case list first; build a new dataset ID from development cases only; re-run
  fingerprint, planning, injection and preprocessing; compare the new plans with the old ones.

**S2. Training recipe is outside the repo** — read
- Only under `UM/`: the conversion script, `scripts/inject_umamba2_config.py`, plans, fingerprint,
  `dataset.json`, job scripts, pretrained weights and logs. The raw and preprocessed nnU-Net data were deleted.
- The converter in the repo (`NN/dataset_conversion/Dataset119_ToothFairy3_All.py:337-342`) uses the
  77-class mapping, not the 47-class one the model was trained on.
- The `umamba2` conda environment imports `UM/nnunetv2`, not the repo copy. The two trees are identical
  today, but an edit in the repo copy would not run, and the repo SHA does not pin the executed code.
- **Change:** bring the conversion script (with a case filter, keeping its special case for
  `ToothFairy3S_0048`), the injection script and the job scripts into the repo; point the environment at the
  repo copy and log `nnunetv2.__file__` in every job.

**S3. Possible leak through the starting weights** — open
- `UM/jobs/train_tf3.sh:36-38` starts from `UM/data/tf3_pretrained/ssl_umamba2_3d_depth7_160x288x288.pth`,
  which per its readme was trained on ToothFairy3 images.
- Not verified: whether these weights were loaded in the June run, and which images they were trained on.
- **Decision for Shivam:** retrain from these weights or from scratch.

**S4. Mirror augmentation turns padding into pulp** — read; lookup reproduced in numpy, not in torch
- The loader pads the label patch with −1 (`NN/training/dataloading/data_loader.py:192`).
- `LRMirrorTransform` (`NN/training/nnUNetTrainer/variants/competitions/toothfairy3.py:42-60`) swaps
  left/right labels with a lookup table of 47 entries; index −1 reads the last entry, class 46 (pulp).
- The transform that removes −1 runs after the mirror, so padded voxels are trained as pulp whenever the
  x-flip is drawn.
- The share of samples affected is not measured. The old logs show pulp pseudo-Dice 0.28–0.58 against
  about 0.97 for teeth; a link is suspected, not shown. `extract_facts.py` does not use pulp.
- **Decision for Shivam:** fix it in the retraining (one line: remove −1 before the mirror) or keep it for
  fidelity to the submitted system.

**S5. Post-processing thresholds differ by side and by voxel spacing** — read
- `EXT/task1_inference.py:336-382` lists a minimum component size per class in voxels, applied on the
  original grid. Left and right differ: tooth 32 is 1,092 voxels and tooth 42 is 3,597.
- Tooth 42's threshold is 97 mm³ at 0.3 mm, 12 mm³ at 0.15 mm and 328 mm³ at 0.45 mm. Only 12 of 95
  centre A cases are at 0.3 mm (results).
- A deleted tooth becomes "absent" in the facts and in our visibility features. The effect on real masks
  is not measured.
- **Change:** convert the thresholds to mm³ and log every deletion per tooth.

**S6. Fact extraction treats "not imaged" as "absent"** — read (absence logic); synthetic (the rest)
- Upper absences are suppressed only when a maxilla label exists and touches the upper border
  (`EXT/extract_facts.py:155-162`). On a mandible-only scan all 16 upper teeth are reported absent, plus
  two "complete edentulism" phrases.
- An edentulous maxilla fully inside the scan is called "partial" (fewer than 4 upper teeth).
- Implants in an edentulous jaw are assigned to the wrong tooth slot (`EXT/extract_facts.py:101-114`).
- The superior–inferior axis is taken from voxel-index centroids (`EXT/extract_facts.py:117-126`); a
  synthetic slab picked the wrong axis, and the fallback with one jaw missing is right only for S-up arrays.
- **Change:** add a "not assessable" state before the decoder uses these facts. This belongs to Part 2 and
  is the same omission problem the paper is about.

**S7. Left/right is learned by array position** — read; results
- Training mirrors all three axes and swaps left/right labels only on array axis 2, so the model learns
  sides from position along x.
- `EXT/segmentation.py:93` flips x when the first header letter is "L" and flips the mask back. The comment
  and `EXT/config.py:17-19` (`MODEL_ORIENTATION = "RPI"`) give a wrong reason: ToothFairy3 headers are
  L-first and no reorientation was done in training. The flip works because ToothFairy3 labels are
  mirrored against the reports.
- All 622 volumes are L-first, so the no-flip branch has never run. Only the first letter is read; permuted
  or oblique headers are not handled (synthetic).
- **Consequence for retraining:** train on ToothFairy3 arrays and labels as they are and keep mirror axes
  0 and 1; or change the convention and remove the flip in the same commit. Record the convention next to
  the checkpoint. Masks from `run_segmentation` are left/right swapped against ToothFairy3 labels, so a
  Dice against those labels must account for it.

**S8. Our loader mirrors by centre letter, not by mask source** — read
- `src/partialref/propensity/agreement.py:82-83, 108, 120` and `src/partialref/audit/side_geometry.py:104`
  mirror every P/F/S case (`--mirror-centres`, default `PFS`).
- This is right for ToothFairy3-derived masks. Predicted masks for P/F/S from `run_segmentation` are
  already in report convention and would be mirrored a second time.
- `label_source` is already written by `visibility_features.py:98` but not used.
- **Change (our code):** key the mirroring on `label_source`. Needed before any predicted mask for P/F/S
  enters the propensity pipeline.

**S9. The predictor cannot give probabilities** — read
- `EXT/task1_inference.py:217` divides every patch by its own maximum; the weight normalisation is
  commented out (lines 204, 220, 222); only the argmax is returned.
- The stock `--save_probabilities` path has no x-flip, no label-swap mirroring and no post-processing, so
  it would describe a different mask, and it writes 47 channels at full resolution.
- The loss also smooths labels on purpose (`NN/training/loss/compound_losses.py:218-234`): 10 % of the
  target goes to neighbouring tooth numbers, pulp and the opposite sinus. Tooth probabilities top out near
  0.9 and the neighbour mass is not uncertainty.
- **Change:** remove line 217, restore the normalisation, and compute softmax and per-tooth statistics in
  memory in the same run that produces the mask. Check calibration on 3–5 cases afterwards.

**S10. Label scheme stated wrongly in three places** — read; synthetic
- `run_segmentation` returns FDI labels (11–48, 50 pulp, 51–53 canals); the map is applied in
  `EXT/task1_inference.py:387-399`.
- `EXT/config.py:39-40` lists sequential ids for the orientation check, which receives the FDI mask: FDI
  27/28 count as lower teeth and 43–48 as neither. The check only warns, and our centre A job never calls it.
- `EXT/facts.py:4-14`, `EXT/config.py:42-47` and DECISIONS.md line 245 say inference reproduces a quirk
  (sequential labels read as FDI) on purpose. It does not: the mask is already FDI. Switching
  `REMAP_MODEL_TO_FDI` on would map labels twice.
- Whether the VLM was trained on facts with that quirk is a Part 2 question.
- **Change:** correct the comments and the DECISIONS entry; remove or guard the flag; give the orientation
  check FDI ids.

**S11. Smaller items** — read
- "Axis 2 = left/right" is hard-coded in the trainer and in test-time mirroring. It holds only while the
  plans keep the axis order, which they do for isotropic data.
- The label layout is written in at least eight places (conversion mapping, trainer swap pairs, class
  weights, loss, `segmentation.py`, `config.py`, `task1_inference.py`, `facts.py`). They agree today.
- `NN/training/loss/compound_losses.py` hard-codes 47 classes and `device='cuda'`.
- The returned mask header goes through a NIfTI round trip; the origin drifts by at most 1.9e-05 mm
  (synthetic). `jobs/seg_centreA.sbatch` avoids this by copying the input header.
- `jobs/seg_centreA.sbatch:14` claims "auto-orientation to RPI", which is not what happens.
- Our tooth volume includes pulp for hand labels but not for predictions (`visibility_features.py:142`).

### 3. What can be reused unchanged

- The trainer class (`nnUNetTrainer_TF3_Task1_1500ep_accum2`), network (UMambaBot2), plans family, patch
  size 160×288×288, spacing 0.3 mm and the 47-class mapping.
- The repo copy of nnunetv2 is identical to the code that trained the submitted model, and the checkpoint
  is the final `fold_all` one (same sha256).
- The left/right swap pairs are correct. Test-time mirroring over y and x with the channel swap is valid.
- The x-flip is applied once to the image and undone once on the mask; the mask is voxel-aligned with the input.
- Non-isotropic input is resampled correctly and the crop is reverted correctly.
- `jobs/seg_centreA.sbatch` (v2) composes the label mapping correctly and asserts on unexpected labels.
- Resuming training with `--c` worked in June.

**Cost of retraining:** about 10 min per epoch on the 40 GB A100, so 1,500 epochs take 10.4 days — four
chained 72 h jobs or the `a100_long` QoS. The job script needs `--account=uzk`.

### 4. Decisions for Shivam

1. Starting weights: ToothFairy3 SSL weights or from scratch (S3).
2. Mirror bug: fix or keep for fidelity (S4).
3. Left/right convention for retraining: keep ToothFairy3 arrays and labels as they are (recommended, no
   code change at inference) or correct the labels and remove the flip (S7).
4. Whether the post-processing thresholds change for the baseline or only for our pipeline (S5).

### 5. Checks not run (each needs approval)

| Question | What would settle it |
|---|---|
| Were the SSL weights loaded in June? | `sacct` for the June jobs and the training log header |
| Share of samples hit by the mirror bug | CPU dry run of the dataloader on synthetic or a few cases |
| Does torch wrap index −1 like numpy here? | one-line torch check in the `umamba2` environment |
| Is centre A's header true on left/right? | full centre A segmentation, then `partialref.propensity.laterality --centres A` |
| Are centre A intensities inside the clip range? | SLURM job with per-case percentiles per centre |
| Effect of the thresholds on real masks | 3 cases with and without post-processing at 0.15 mm and 0.45 mm |
| How often the superior–inferior axis is wrong | `extract_facts.py` on the existing centre A masks, tabulate `si_axis` |
| Calibration of real probabilities | 3–5 case diagnostic after the predictor change |
| Suspect label cases | visual check by Shivam: P192 (teeth on the opposite side), P242 and S0019 (sinus sides), S0015, F028, F055, P230 |

---

## Part 2 — VLM, facts, post-processing, report assembly, evaluation

`V/` is `external/odin2026/vqa/code/`.

### 1. Summary

The view rendering, the question builder, the schema, the prompt and collator code, the LoRA and merge
scripts and the template writer can be reused. Four things must change before per-fold retraining and
before the decoder is built on top:

1. **Training labels.** Our parser output must enter as the per-case label file, with "not mentioned" as
   null. The submitted label builder turns silence into `False`.
2. **Splits.** Guards, the negative cap, pool selection, teacher evidence and the post-processing rules are
   all computed over the whole pool or over validation cases. Each must run per fold.
3. **Inference.** Decoding needs a fixed temperature and seed, every call needs a status record, and the
   call must return log-probabilities for the decoder.
4. **Evaluation.** The local metric script differs from the official one and drops failed cases.

### 2. Findings

#### Training labels and LoRA training

**V1. Silence is trained as a negative** — read; synthetic
- `V/parse_reports_to_gt.py:1475-1523`: for every present tooth, a finding that no report sentence names
  becomes `False` (caries, root fracture, endodontic treatment, filling, crown, post, periapical lesion,
  furcation, bone lesion). Only absent positions get null. Eruption defaults to fully erupted and bone loss
  to the arch default.
- `V/parse_reports_to_gt.py:1170-1175` closes the world for tooth presence: if a report lists absences,
  every unnamed tooth is present, and the reverse (agent).
- Silent sinus, atrophy and periodontal fields are set to normal (`:963-1036`, agent).
- `V/build_sft_targets.py:379-386` supervises every non-null field. Arch rows have no "unstated" signal.
- Synthetic check by the agent: a report stating one absence gave 15 present teeth with 10 supervised
  fields each on teeth that were never mentioned.
- **Consequence:** the model's probability for a finding already contains the reporting propensity. A
  decoder that corrects with π on top would count it twice, unless training masks silent fields.
- **Change:** write `{case_id}_gt.json` from our parser output and pass it with `build_sft_targets.py
  --gt-dir`. Null means not mentioned or not assessable; `False` only where our absence convention allows
  it. Add an "unstated" mask for arch facts. This is also where a π-weighted loss would go.

**V2. Two readers are merged asymmetrically** — agent
- Finding lists are unioned (`V/parse_reports_to_gt.py:1699-1704`); presence ties go to "present" (`:1690`).
- For scalar fields a silent reader counts as a vote for the neutral value and ties go to the first reader
  (`:1655-1659`, `:1725-1728`), so the result depends on reader order.
- Only the first bridge and the first bone lesion are kept (`:1720-1723`).
- Multi-report cases run without `--consensus` get no label file and vanish downstream (`:1944-1946`).
- **Change:** not needed if our own label file replaces this step (V1); otherwise fix the tie and
  truncation rules.

**V3. Leak guards pass on nothing** — read
- `V/train_vision_lora.py:332-369`, `V/build_sft_targets.py:591-610` and `V/select_sft_pool.py:132-135`
  define "forbidden" as the content of `dataset/validate/images` and `outputs/sft_pool/heldout.txt` under
  V2V2R's root. If the directory is missing they print PASS with 0 cases checked.
- The released adapter saw 528 training and 30 evaluation cases, listed in
  `models/Qwen3.5-9B-AWQ-dental-cbct-sft/merge_meta.json` (agent). It was trained on 21 of the 24
  "held-out scoring" cases hand-coded in `V/survey_findings.py` (agent, computed from `train_meta.json`).
- **Change:** make `--case-list` mandatory, assert against the fold's test list and fail when that list is
  empty. Train every fold from the base model, never from the released weights.

**V4. Rules, silences and model selection were fitted on evaluation cases** — read (`source_rules.py`); agent (rest)
- `V/source_rules.py:43-45` states that all ten source rules rest on the 40 validation cases and nine are
  unconfirmed on training data.
- The sentence switches in `V/synthesize_report.py` (lines 221, 269, 323, 451, 609-658) were set from
  precision on validation runs. `V/rate_matched_null.py` compares LoRA arms on a validation run.
- `V/postprocess_pred.py:3027-3067` and `V/audit_facts.py` use statistics over the references of all 622 cases.
- The negative cap in `V/build_sft_targets.py:415-478` runs over the whole pool; pool selection
  (`V/select_sft_pool.py`) ranks cases by report–mask agreement; teacher evidence (`V/draft_evidence.py`) is
  written with the label in hand.
- **Consequence:** in 5-fold or locked-test evaluation the submitted rules give optimistic scores. The
  hidden-test leaderboard number is not affected.
- **Change:** refit rules and switches inside each training fold, or replace them by the decoder. Run the
  cap, the selection and the evidence drafting per fold.

**V5. Training run robustness** — read (seed, guards); agent (rest)
- `get_peft_model` (`V/train_vision_lora.py:440`) runs before any seed is set, so the LoRA initialisation
  is not reproducible. There is no resume, although checkpoints are saved per epoch.
- A failing row (over 8,192 tokens or missing image) is replaced by its neighbour (`:85-94`). The counters
  live in dataloader workers, and the released `train_meta` shows empty drop statistics, so the drop count
  of the released run is unknown.
- Evaluation is off by default and random when on. Paths are hard-coded under V2V2R's root.
- A code comment reports a 60 GiB peak on one row; the 40 GB A100 is doubtful (not measured).
- No requirements file pins the training environment.
- Loss masking, batch size 1, schedule, gradient accumulation and bf16 are correct by reading.

**V6. Training and inference differ in places** — agent
- Prompts, captions, image order and `enable_thinking=False` are shared by construction.
- Arch rows and tooth rows without teacher evidence train with an empty, masked `visual_evidence`; at
  inference the model writes the evidence first.
- Tooth rows exist only for teeth present in the mask. A tooth absent in the report but present in the
  mask supervises "absent" on an image that shows a tooth.
- The script that generated the training images (`jobs/aksssr_pipeline.sh`) is referenced in six files but
  is not in the import (read), so image settings at training and inference cannot be compared.
- Whether the facts used in training came from sequential or FDI masks cannot be settled from code (S10).

#### Inference

**V7. Decoding is sampled and unseeded** — read
- `V/run_vqa_inference.py:134-165` sends no temperature by default; no top_p or seed is sent; vLLM is
  started without a seed (`V/competition_runner.py:150-159`). A truncated answer is resent at temperature 0.7.
- The released model's `generation_config.json` sets sampling with temperature 1.0 (agent). Whether vLLM
  applies it is open. The base Qwen3.5-9B has no such file, so a retrained model would decode differently.
- **Change:** send temperature, top_p and seed explicitly, and record them.

**V8. Failed calls become findings** — read (tooth result); synthetic (scenario)
- A call exception becomes `{}` plus one stderr line (`V/run_vqa_inference.py:1174-1237`, agent).
  `finish_reason` is never checked and the script exits 0.
- Every tooth result, including an empty one, is stored as `detected = "yes"` (`:1356-1358`).
- Agent scenario with every call failing: the report still says all 32 teeth present, both canals regular,
  no lesions.
- A crash of the whole runner gives a mask-only template with the no-lesion sentence
  (`external/odin2026/report_side.py:123-128`), and the work directory is deleted.
- A tooth under 5 voxels gets no image, then `no_image`, which counts as absent (agent).
- **Change:** record a status per call in the prediction file; stop writing `detected = "yes"` for failed
  calls; keep the work directory.

#### Post-processing and report text

**V9. The M3 switch in HANDOFF does not exist** — read
- `POSTPROCESS_CONFIG=configs/postprocess/no_source_rules.yaml` appears only in HANDOFF.md line 604.
- The ten rules are a module constant (`V/source_rules.py:71-82`). The only switch is `--facts-dir`, which
  turns all rules off at once and also removes the filling sentence and the mask-absent gate (agent), so it
  is not a clean ablation. A missing facts file switches all rules off silently (agent).
- The silence rules are hard-coded returns in `V/synthesize_report.py` (lines 344, 473, 703, 894, 1169; agent).
- **Change:** a per-rule switch with facts still loaded; named switches for the silences, default as submitted.

**V10. Rule and assembly bugs** — read (maxilla drop); synthetic (rest)
- One sampled answer deletes the upper jaw: `V/postprocess_pred.py:3216-3229` drops all maxillary findings
  when the model answers "not included", even if the mask has 16 upper teeth.
- `apply_absent` (`V/source_rules.py:139-157`) rewrites the tooth list but not the pattern, so the report
  can state a whole quadrant absent.
- `apply_impaction` (`:195-245`) keeps wisdom teeth only, so an impacted canine is deleted; a fully erupted
  wisdom tooth with non-normal orientation is declared impacted.
- If the mask has no implant or crown, VLM claims pass unfiltered; if it has one, the VLM positions are replaced.
- Out-of-scan upper teeth are removed from the absence list only when the maxilla is "excluded", not when
  it is "partial".

**V11. Statements written without a prediction, and predictions never written** — agent; synthetic
- Written without a prediction: canal "regular course" (not a schema field), canal location (defaults to
  lingual), scope (defaults to "partially included"), the no-lesion sentence when the bone-quality answer is
  missing, mucosal thickening when the sinus answer is null.
- Predicted but never written: caries, root fracture, root remnant, crown, post-and-core, filling quality,
  periapical lesion, periodontal status per tooth, bone resorption, intrasinusal teeth, sinus content,
  impaction of non-wisdom teeth.
- **Why it matters here:** the submitted system is silent on these finding types by construction. These
  silences are what the selection step with π is meant to replace.

#### Views and captions

**V12. Side and orientation in the rendered views** — synthetic (panoramic, sinus); agent (3D, code only)
- When no jaw has teeth on both sides (one-sided, edentulous, implant-only), nothing is flipped but the
  L/R markers and captions are still written (`V/create_panoramic.py:1396-1437`, `V/create_3d_renders.py:483-494`).
- The fallbacks disagree: sinus view uses the header convention, tooth composites the mirrored one.
- 3D side views take the side from the header; every other view takes it from the labels.
- Upside-down volumes are not handled in any view.
- The sinus view crops by array halves; a sinus in the lower half of a tall scan gets no marker while the
  caption says the cross marks it.
- Laterality is circular: the patient side comes from the mask's own tooth labels, so a mirrored mask gives
  a consistently mirrored report.

**V13. Field of view, intensity and size assumptions** — read (window, 3D default); agent (rest)
- Field of view is decided by three different rules. `V/create_3d_renders.py:940-948` falls back to
  "maxilla partially included"; the facts never contain "complete", so every complete maxilla gets this
  caption on the view where the scope question is asked.
- Tooth composites use a fixed window (centre 1200, width 4400) that assumes calibrated intensities
  (`V/create_tooth_detail.py:405-421`); the runner does not pass the percentile option. Centre A is often
  not calibrated.
- Composites, sinus tiles and 3D meshes assume isotropic voxels.
- Tooth-presence thresholds differ between steps: facts 50 mm³, composites 5 voxels, 3D 10 voxels.
- Captions of tooth composites state crown, implant, bridge, canal proximity and "absent" from the mask
  facts (`V/create_tooth_detail.py:1013-1032`). **For the decoder this means the VLM answer is not
  independent of the mask.**

#### Evaluation code

**V14. The local metric is not the challenge metric** — read
- `V/official_ranking.py:394-400` judges RadFact against the first reference only, with a local judge model.
- Cases whose judging failed are dropped from mean precision and recall (`:622-627`) while BLEU and METEOR
  keep them.
- The clinical score is the harmonic mean of mean precision and mean recall. Applied to the published
  hidden-test precision 0.534 and recall 0.332 this gives 0.409, not the published 0.392, so the official
  aggregation is different.
- `V/evaluate_predictions.py:111-119` scores empty-against-empty as 1.0 and macro-averages. No confidence
  intervals exist anywhere.
- **Change:** fail on a failed batch, report per-case scores and intervals, state the reference policy.
  Our own metrics (`partialref`) replace this for the paper; the script matters only for reproducing the
  submitted baseline.

#### Data use

**V15. Data-use review of the imported code is pending** — agent; not checked by Claude
- Some comments in the imported code under `V/` need a data-use review by Shivam before the code is
  distributed further. The details are kept outside the repository.
- Two optional code paths can call an external API: `V/build_fewshot_exemplars.py` and
  `V/official_ranking.py --judge-backend openai`. Neither is on the submitted path, and neither may be run
  on ToothFairy4 data.
- **For Shivam:** do the review and decide what is removed.

### 3. What can be reused unchanged

- The four view generators as code (deterministic, no GPU), with the limits in V12 and V13.
- `build_vqa_pairs.py`: 9 arch calls plus one per imaged tooth; closed answer sets are machine-readable in
  `qa_pairs.jsonl`.
- `schema.json` (v7.1), `sft_prompt.py`, `sft_collator.py`, `lora_arms.py`, `merge_vision_lora.py`, `normalize_pred.py`.
- `postprocess_prediction(pred, facts)` and `synthesize_report(summary)` as functions.
- The few-shot exemplars are not on the submitted path or in the training path.
- No left/right inconsistency was found in the facts or the source rules themselves.

### 4. Fit to the decoder (HANDOFF M4)

**Hooks**
- Log-probabilities: in `call_vllm_messages` / `infer_call` of `V/run_vqa_inference.py`. Call and view
  identity must be carried through, because the merge of global calls flattens them. `visual_evidence`
  comes first in the answer, so options are scored after it.
- Output: the decoder can write a `{case}_pred.json` and call `postprocess_prediction` then
  `synthesize_report`, or write the summary JSON directly.

**Obstacles**
- The panoramic call gives one exclusive 6-way label per tooth with merged categories, while the composite
  gives separate booleans; combining them needs a many-to-one likelihood.
- Images and tooth calls exist only for mask-present teeth, and captions state mask facts (V13).
- The model's probabilities contain reporting propensity as long as silence is trained as negative (V1).
- Label smoothing in segmentation (S9) and sampled decoding (V7) both distort confidences.

**Schema fields without a clear counterpart in our ontology** (agent, not checked field by field):
`visual_evidence`, `with_root_fracture`, arch `periodontal_extent` and tooth `bone_loss`, implant
`osseointegration_status`, sinus `scope` / `mucosa_state` / `sinus_content`, condyle and maxilla scope,
bridge abutments and pontics.

### 5. Decisions for Shivam

1. V15: outcome of the data-use review, and whether the external API code paths are removed.
2. Baseline definition: is the M3 baseline the submitted system with its rules as they are (fitted on
   validation cases, stated as such), or with rules refitted per fold (V4)?
3. Training labels for the retrained baseline: the submitted silence-as-negative labels (faithful
   baseline) or masked labels from our parser (V1). Both may be needed: one as baseline, one for the decoder.
4. Which bugs are fixed in the baseline (V8, V10) and which are kept and reported.
5. HANDOFF line 604 needs correcting once the switch design is agreed (V9).

### 6. Checks not run (each needs approval)

| Question | What would settle it |
|---|---|
| Does vLLM apply the model's `generation_config.json`? | the "Default sampling parameters" line in an existing server log |
| Label scheme and mask source of the training facts | `present_label_ids` in one facts file under Chen's project folder |
| Flags used for the released adapter (arch rows, evidence) | the job script or the rows file next to the released training output |
| Overlap of validation-40, held-out-24 and the 528 training cases with our split | compare the lists once `configs/splits/` exists |
| Rows dropped per fold | `sft_collator.py --rows <fold.jsonl>` on CPU via `srun` |
| Peak GPU memory of LoRA training | one-row backward probe on the target GPU |
| Official aggregation and reference policy | compare `official_ranking.py` with the organisers' evaluation script |
| Code difference between the submitted image and the original repo commits | `git diff` in the original repository (Chen) |
| Per-case view assumptions on real masks (side vote, maxilla volume, sinus height, small teeth, spacing, intensities) | the agent's check script, after review and copying into the repo, via `srun` |
| Missing referenced files | ask Chen for `jobs/aksssr_pipeline.sh`, `jobs/evaluation.sh`, `postprocess_pipeline.md`, `vision_sft_plan.md`, the teacher-evidence folder |
| Parts not read in detail | `synthesize_report.py` lines 700-1329, `normalize_pred.py` repairs, `build_vqa_pairs.py:650-770` |
