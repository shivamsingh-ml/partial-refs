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
