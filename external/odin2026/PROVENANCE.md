# external/odin2026 — code submitted to the ODIN 2026 / ToothFairy4 Task 1 challenge

This folder is the **exact `/opt/app` of the submitted Docker image**, copied verbatim. The commit
that added it changes nothing; every later edit is an ordinary commit, so
`git diff <import-commit> -- external/odin2026/` shows all our changes.

| | |
| --- | --- |
| Image | `ssingh54/toothfairy4-v2v2r:odin2026-task1` |
| amd64 manifest | `sha256:b4abf81185e7e315d5b1f43a12a235d34b279b6b8c1150099b76c998799eb571` |
| Built | 2026-08-17 (from an uncommitted working tree of `shivamsingh-ml/tf-submission` on top of `d9c27a8`) |
| Extracted | 2026-11-07, registry API, layers 20, 22, 25, 27, 28, 32, 33, 34 only (layer sha256 digests verified); no Docker, no full pull |
| Files | 305 (`SHA256SUMS`) |

Not in git, but listed in `SHA256SUMS` and kept on the cluster under
`.cache/odin2026-task1-code/` (with the original layer tarballs and registry metadata):

- `vqa/code/fewshot_examples/` (11 files) — may contain text derived from ToothFairy4 reports;
  excluded from git until Shivam confirms it is safe under the DUA. The pipeline needs them at
  run time: copy them back from `.cache/odin2026-task1-code/opt_app/vqa/code/fewshot_examples/`.
- `umamba2/nnunetv2/tests/example_data/*.nii.gz` (2 files) — upstream nnU-Net test volumes, not
  needed; not listed in `SHA256SUMS`.

Verify: `cd external/odin2026 && sha256sum -c SHA256SUMS` (fewshot files need to be copied back first).
