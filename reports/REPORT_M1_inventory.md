# REPORT M1 — inventory and centre characterisation (step 1)

Numbers only, no report text. Sources (all produced by code in this repo):

- Manifest: `results/manifest/20261107-1830-c2aa718-manifest/` (job 614556, git `c2aa718`, clean;
  manifest hash `72b68659d0a65905ec745d87e067c5806b4b8556e28bfeb6763c8e279f2b8805`).
- Inventory: `results/audit/20261107-1832-c2aa718-dirty-inventory/` (job 614559; code = `c2aa718`,
  "dirty" only because DECISIONS.md had uncommitted notes).
- Data: `~/Datasets/ToothFairy4/` (622 case folders; zip sha256 `40dbbf2d…a073`).

## 1. Cases and reports

| Centre | Cases | EN reports | 1 report | 2 | 3 | 4 | Multi-report share |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A (Sfax, external) | 95 | 97 | 93 | 2 | 0 | 0 | 0.021 |
| F (TF3 Set B) | 63 | 127 | 0 | 62 | 1 | 0 | 1.000 |
| P (TF3 Set A) | 412 | 678 | 154 | 254 | 0 | 4 | 0.626 |
| S (TF3 Set C) | 52 | 98 | 8 | 42 | 2 | 0 | 0.846 |
| **All** | **622** | **1000** | **255** | **360** | **3** | **4** | **0.590** |

- Every EN report has an IT file with the same name (622/622 cases) → one reading in two languages;
  readers are counted per file, EN only.
- No file-name or UTF-8 problems (1000/1000 EN, 1000/1000 IT).
- Centre names A/F/P/S → sites: from `cbct-vlm-transfer` FINDINGS #25; still to confirm against the
  official dataset description.

## 2. Candidate reader ID (`report_x`, from `cbct_<X>_<Y>.txt`)

- 16 distinct values. In the 367 multi-report cases: all reports have different X in **363**, the
  same X in **0**, mixed in **4** (P116, P117, P260, P261 — see §5).
- Reports per X and centre:

| X | A | F | P | S |  | X | A | F | P | S |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 14 | 0 | 60 | 42 | 0 |  | 28 | 12 | 0 | 1 | 52 |
| 21 | 7 | 0 | 52 | 0 |  | 29 | 6 | 11 | 0 | 44 |
| 22 | 0 | 0 | 52 | 0 |  | 30 | 6 | 2 | 50 | 0 |
| 23 | 7 | 1 | 15 | 0 |  | 31 | 7 | 0 | 52 | 0 |
| 24 | 7 | 0 | 74 | 0 |  | 32 | 8 | 0 | 51 | 0 |
| 25 | 6 | 0 | 70 | 0 |  | 33 | 7 | 0 | 52 | 0 |
| 26 | 8 | 2 | 67 | 2 |  | 34 | 4 | 0 | 54 | 0 |
| 27 | 7 | 51 | 0 | 0 |  | 35 | 5 | 0 | 46 | 0 |

- Observations (hypotheses until the dataset documentation confirms X = reader): the same X values
  report on several centres (e.g. 14 on F and P; 28/29 mostly S; most X report a few A cases), so
  "centre" and "reader" are crossed, not nested — centre effects and reader effects can in
  principle be separated.

## 3. Report length (EN characters per report)

| Centre | Median [min–max] |
| --- | --- |
| A | 867 [301–2780] |
| F | 616 [280–1500] |
| P | 540 [142–2160] |
| S | 582 [177–1570] |

A's reports are the longest — consistent with its larger field of view (more anatomy to describe).

## 4. Scan geometry (NIfTI headers)

| Centre | Orientation | Spacing (mm) | Field of view x × y (mm) | Height z (mm) | z < 60 mm |
| --- | --- | --- | --- | --- | --- |
| A | LPS (95) | 0.175 [0.12–0.45] | 83 [78–160] × 83 [70–160] | 90 [60–120] | 0 / 95 |
| F | LAS (63) | 0.30 | 123 [123–154] × 123 [123–154] | 82 [41–89] | 1 / 63 |
| P | LAS (412) | 0.30 | 111 [94–139] × 101 [79–127] | 51 [45–53] | 412 / 412 |
| S | LAS (52) | 0.30 | 111 [111–139] × 107 [87–139] | 52 [41–62] | 49 / 52 |

- **Cross-check with the earlier project** (`cbct-vlm-transfer/reports/intensity_percase.csv`):
  622/622 cases present in both; **0 mismatches** in shape (x, y, z), spacing (x, y, z), extent
  (x, y, z) and orientation.
- **Data copy check:** our copy and `cbct-vlm-transfer/data/raw` were byte-identical (2622/2622
  files, job 614558); the duplicate was replaced by a symlink.
- Acquisition is confounded with centre (A: different orientation, spacing, field of view) — the
  acquisition-only control will largely encode centre.

## 5. Flags for the audit (M1 steps 2–3)

1. **P116, P117, P260, P261:** 4 distinct EN reports each, two per X (e.g. 21, 21, 26, 26); volumes
   differ. ToothFairy3 has 4 P scans with no TF4 reports (P120, P121, P262, P263) next to them →
   possible report mix-up or two readings per reader. Per-reader running report numbers support
   a mix-up for P116/P117 (their extra reports sit exactly where P120/P121's are expected); for
   P260/P261 the numbering is ambiguous. Interim: excluded from multi-report analyses; possible
   repair via mask matching later (DECISIONS.md).
2. **Identical report files across cases:** 15 EN files in 5 groups (320–380 characters), all from
   X = 14, never twice in the same case (P004/P018/P023/P027/P057; P005/P006/P025/P070;
   P041/P086; F008/F010; F018/F033) → template reuse by one reader; matters for the reader-level
   analysis and the independence check.
3. **Centre A has 2 multi-report cases** → its reporting propensity cannot be estimated on its own
   (pool with other centres / use reader effects, HANDOFF §12.5).
4. **No masks in the TF4 release;** ToothFairy3 labels exist for all 527 P/F/S cases
   (`~/sae_tooth/data/raw/toothfairy3/labelsTr`, IDs match by number) — identity of scans still to
   verify from headers / voxels before use.

## 6. Not done yet

Independence check of report pairs (text similarity), representativeness (SMD), visibility per
finding, split design (step 3b), per-report findings. Nothing in this report depends on report
content beyond character counts and file hashes.
