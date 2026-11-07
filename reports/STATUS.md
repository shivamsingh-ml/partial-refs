# STATUS — where we stopped (end of session, Sat 7 Nov 2026)

Read HANDOFF.md (rules, plan) and DECISIONS.md (evidence) first. This file is only the bookmark.

## Done
- **M0 complete.** Cluster envs (`cbct_partialref` CPU; `pr_serve` vLLM 0.19; `pr_train` LoRA SFT;
  `umamba2` segmentation). Submitted V2V2R code recovered byte-exactly → `external/odin2026/`.
  Models in `models/` (released V2V2R, `QuantTrio/Qwen3.5-9B-AWQ`, `Qwen/Qwen3.5-9B`).
  **Smoke run passed** on a full A100 (job 614601; MIG slices cannot host vLLM 0.19).
- **M1 so far:** manifest (622 cases, 1000 EN reports, 367 multi-report); inventory report
  (`reports/REPORT_M1_inventory.md`); independence check (0/369 near-copy pairs); TF3↔TF4 identity
  (527/527 voxel-identical after flipping TF3 axes 1,2; TF3 headers upside down; left/right open).
- **Decisions this session:** own split (locked ~20 % test + 5-fold CV + LOCO); V2V2R retrained per
  fold; segmentation retrained without test cases; visibility conditioning (three tiers, HANDOFF §2.6);
  Lumetti et al. ontology + metrics as the ruler (re-implemented, cited), their labels internal only,
  their baseline not used; reader-level analysis dropped; second domain shortlist (VinDr-CXR + REFLACX).
- Supervisor overview page updated: https://claude.ai/artifact/X4ACD1gdeUbLe21RsbS6DF

## Running / queued when we stopped
- **Visibility features, P/F/S** (job 614607) → `results/audit/*-visibility-features/`.
- **Centre A segmentation**: diagnostic (job 614609, 3 cases; first two fine) then full 95 cases
  (job 614610, queued with `afterok`) → `results/seg/centreA_umamba2_foldall/masks/` (resumable).

## Next (Monday), in order
1. Check both jobs (`sacct`, logs in `logs/`). If the A segmentation finished, add an A adapter to
   `partialref.audit.visibility_features` (predicted masks are already on the TF4 grid: no flip;
   pulp = 199) and run it for A.
2. Show Shivam per-centre distributions (jaw labels present, root tips cut, sinus volume vs F,
   condyle region cut, quadrant edges) → **freeze visibility thresholds** in DECISIONS.md before any
   report analysis.
3. **Split design** (M1 step 3b): propose locked test + 5 folds + LOCO, stratified by centre,
   multi-report, field-of-view class; exclude P116/P117/P260/P261 from multi-report analyses → Shivam approves.
4. Then: local report parser onto the Lumetti ontology (local LLM only), checked against their
   released per-patient labels (internal) + ~10 manual reports; M0.5 simulation study run.

## Waiting on Shivam
- Visual check of **S0015** (probably stored upside down).
- PhysioNet credentialing if VinDr-CXR / REFLACX is chosen as second domain.
- Optional: delete `~/partial-refs/hf.env` (ignored by git; jobs use `~/.secrets/hf.env`) and
  consider regenerating the HF token (it left the cluster in a rejected push).
