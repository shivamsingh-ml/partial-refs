# ToothFairy3 / ToothFairy4 case split v1 — standard split for all ToothFairy3 studies

Decided 2026-11-11 (DECISIONS.md), adopted as the standard ToothFairy3 split for future studies on 2026-11-17.
Built by `partialref.data.splits` (seed 20261110) from case IDs, centre, number of English reports and
manual-mask tooth counts only; no image content, no report text, no results.

## Files

- `splits.csv` — one row per case: `case_id`, `centre` (P/F/S/A), `role` (`dev` / `test` / `external_A`),
  `fold` (0–4 for `dev`, −1 otherwise).
- `splits.json` — seed, inputs and their SHA-256, `never_test` cases with reasons, the five ToothFairy3-only scans
  (`tf3_only_for_segmentation`), counts.
- `summary.txt` — counts per fold / test.

## What is in it

| Set | Scans | P | F | S | Use |
|---|---|---|---|---|---|
| Development (5 folds of 84) | 420 | 329 | 50 | 41 | training, cross-validation, all tuning |
| ToothFairy3-only scans | 5 | 5 | – | – | development (segmentation training only; no ToothFairy4 reports) |
| **Locked test** | **107** | 83 | 13 | 11 | **final evaluation once; never for training, tuning, checkpoint or threshold choice** |
| Centre A (external) | 95 | – | – | – | no manual masks; external evaluation of the full system, no tuning on it |

Together, development + test cover all 532 ToothFairy3 scans (ToothFairy3 file `ToothFairy3<C>_<id>` = case `<C><id>`;
the ToothFairy4 volumes of P/F/S are voxel-identical to them after flipping array axes 1 and 2).

## How it was made

Within every centre × single/double-read cell, cases were sorted by jaw coverage (both arches / one / none),
then tooth count, then a seeded random number; the test set (20 %) was drawn by systematic sampling along that
order and the rest dealt into five folds in turn. Cases that must not be in the test set stay in development:
misfiled reports (P116, P117, P260, P261), S0015 (unusable jaw labels) and cases with an open visual check
(F028, F055, P230, S0019, P192, P242). Possible same-patient pairs are therefore never split across test and
development. Cases are never moved after the commit.

## Balance (`partialref.data.split_balance`, run `results/splits/20261117-1045-960f355-split-balance`)

Test − development, standardised mean difference (SMD; |SMD| < 0.1 negligible, 0.1–0.2 small). Permutation p
shuffles test labels within centre × read cell with the never-test cases fixed; uncorrected for 14 properties.

| Property | Dev | Test | SMD | p |
|---|---|---|---|---|
| Toothless* | 3.8 % | 3.7 % | −0.00 | 1.0 |
| Upper teeth present* | 75.0 % | 74.8 % | −0.01 | 1.0 |
| Both arches* | 73.8 % | 72.9 % | −0.02 | 0.90 |
| Teeth per scan* | 19.9 | 20.0 | 0.01 | 0.93 |
| Upper jaw labelled | 43.1 % | 43.9 % | 0.02 | 0.90 |
| Any root tip cut by field of view | 74.5 % | 73.8 % | −0.02 | 0.90 |
| Share of teeth with root tip cut | 29.8 % | 28.9 % | −0.04 | 0.73 |
| Bridge label | 19.0 % | 18.7 % | −0.01 | 1.0 |
| Axial slices (compare within centre) | 181 | 182 | 0.03 | 0.32 |
| **Implant label** | 12.6 % | **19.6 %** | **0.19** | 0.10 |
| **Crown label** | 35.0 % | **41.1 %** | **0.13** | 0.26 |
| **Any crown cut by field of view** | 10.7 % | **15.0 %** | **0.13** | 0.23 |

\* design variables: balanced by construction (the split was sorted by them).

## Known limitations (state them in any paper that uses this split)

1. **Implants and crowns are somewhat more frequent in the test set** (implants 21 test scans vs ~13–14 expected;
   per centre P 15.7 vs 10.9 %, F 30.8 vs 18.0 %, S 36.4 vs 19.5 %). Not a balancing variable; small effect; it
   makes the test set slightly harder for implants and crowns, equally for every method compared.
2. **Only 13 full-arch centre-F scans in the test set.** Centre F is the only centre with both full arches in the
   field of view (P and S often crop the upper jaw). Questions specific to full arches use "test ∩ F" (13 scans)
   and need wide intervals.
3. Public models trained on all of ToothFairy3 (e.g. released challenge weights, U-Mamba2's self-supervised
   weights) have seen the test scans; the test set is clean only for models trained here on development scans.

## Rules for every study using it

- Train and tune on development scans only (folds for cross-validation); the test set is evaluated once at the
  end of a study, with the protocol fixed beforehand.
- Use the case lists from `splits.csv` unchanged; a study that needs a different subset defines it as a subset
  of these roles (e.g. development ∩ F), never by moving cases.
- Segmentation training on these scans uses the label corrections in `configs/seg/label_fixes_v1.json`
  (S0015 excluded, 11 scans corrected; DECISIONS 2026-11-16).
