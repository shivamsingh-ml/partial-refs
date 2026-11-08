# STATUS — where we stopped

## Update Mon 9 Nov 2026, early morning (latest — read this first)
- T2 done on the 14B parse (provisional): π canal 0.88, endo 0.83, implant 0.68, restoration 0.51, caries 0.26
  (double-read, mirrored masks). **TF3 tooth masks are L/R mirrored vs report FDI numbers** (214 vs 13 cases);
  clinician side swaps are at chance level. V2V2R's own segmentation already corrects L/R (segmentation.py).
- **Bug (ours): centre A v1 masks double-mapped labels** → tooth-level A features invalid; fixed in
  `jobs/seg_centreA.sbatch` (new dir `_v2`); needs the GPU re-run (3-case check first, then 95).
- Independent review of all code incl. parser: 15 issues, all fixed + verified (DECISIONS 2026-11-09).
  All T1 numbers so far used the old scorer → re-score.
- Qwen3-32B (full bf16, A100 80GB) parse job 615019: EN done (3 failed reports vs 17 for 14B), IT/repeat
  running. Old-scorer held-out slot F1 0.847 (14B 0.833) — not final.
- Queued overnight (see DECISIONS / logs): Lumetti reader re-run (cases.parquet) → re-score 14B + 32B;
  P/F/S visibility features v2 + side geometry + report-scan fingerprint + laterality on the 32B parse;
  centre A segmentation 3-case diagnostic.
- **Next:** (1) check those logs; (2) if the 3-case A masks contain all 4 quadrants' FDI IDs, run all 95, then
  A visibility + laterality on A; (3) decide canal/sinus side swap from side geometry; (4) parser freeze
  (choose on tune, report held-out) + audit sheet for Shivam (open question: add "Lumetti-only" and
  "both agree" samples); (5) T3, T4, REPORT_GT.md. Waiting on Shivam: visual check of S0015, F028, F055,
  P230, S0019; V2V2R facts.py comment (were VLM training facts sequential-label?).

## Update Sun 8 Nov 2026
- Plan changed: **go/no-go test GT first** (HANDOFF §7 GT, thresholds = guidance). No V2V2R/segmentation
  retraining or decoder work before the GT decision.
- Built: finding ontology (after Lumetti et al., re-implemented), synthetic few-shot prompt, local-LLM
  per-report parser (`partialref.extract.per_report`), Lumetti label reader (529 cases, run
  `20261108-2017-e80c4fe-lumetti-labels`), parser scoring, mock-server end-to-end test (59 tests pass).
  vLLM 0.19 request format, flags and JSON grammar verified on CPU. Qwen3-14B downloaded (`models/Qwen3-14B`).
- Centre A segmentation done (95/95, job 614610); centre A visibility features done (job 614955; run is
  marked dirty → rerun cleanly before freezing thresholds).
- **Queued:** job 614957 = parser 5-case check (EN; first 5 IDs are A cases → pipeline test only, no
  accuracy score); job 614962 = full parse of all reports EN + IT + EN repeat, chained `afterok`, each
  scored against Lumetti labels. Estimated start Mon ~08:20 (A100 40GB queue).
- **Monday:** (1) check 614957/614962 logs (only grep the vLLM server log for errors; never read report
  text); (2) T1 numbers: parser vs Lumetti per finding type, EN-vs-IT, run-to-run; (3) agree provisional
  visibility rules with Shivam; (4) build T2 (π estimates), T3 (mask comparison), T4 (effect-size
  simulation) → `reports/REPORT_GT.md`. If 614957 failed, fix and resubmit both (614962 never starts).

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

## Next (Monday), in order — revised 2026-11-08: GO/NO-GO TEST FIRST (HANDOFF §7 GT)
0. Confirm GT thresholds and parser decisions with Shivam; then T1 parser → T2 π → T3 masks → T4 effect size.
   Visibility (steps 1–2) is needed for T2/T3 in provisional form; the split (step 3) waits until after GT.
1. Check both jobs (`sacct`, logs in `logs/`). If the A segmentation finished, add an A adapter to
   `partialref.audit.visibility_features` (predicted masks are already on the TF4 grid: no flip;
   pulp = 199) and run it for A.
   **P/F/S features finished** (job 614607, run `20261107-2050-da2f91d-visibility-features`). First look,
   to investigate before any threshold: (a) maxilla label present F 97 %, P 30 %, S 85 % although P has
   median 8 upper teeth → "jaw included = label OR teeth present"; (b) mandible touches the top face
   in 100 % of cases incl. F → check for stray voxels (use a robust measure, e.g. voxels on the top
   slice) before the condyle rule; (c) left-sinus volume median P 0, S 0, F 741 mm³ → F is NOT a
   "complete sinus" reference; need another reference (A predictions / literature); (d) root tip cut
   by the scan edge: F 17.5 %, P 35 %, S 44 % of present teeth.
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
