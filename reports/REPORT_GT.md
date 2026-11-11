# REPORT_GT — go/no-go gate on the reference reports (T1–T4)

Status: **decided 2026-11-11: GO in two stages (§9).** Gate definition: HANDOFF §7 GT.
Guidance thresholds are guidance; the decision and its reasoning go to DECISIONS.md.

## 1. Question and short answer

**Question.** Are the ToothFairy4 reference reports incomplete enough, and can that incompleteness be
measured well enough, to build the π-aware decoder on it?

**Short answer.**

| Check | Question | Result |
| --- | --- | --- |
| T1 | Does the report parser read the reports correctly? | **Passed provisionally.** Small check of 20 held-out reports on 7 finding types: 92 of 99 findings correct, one repeating error (bridge spans). Full T1 before the final runs. |
| T2 | Do readers leave findings out (π clearly below 1)? | **Passed.** 10 of 13 estimable types have upper 95 % limit < 0.90 and width ≤ 0.30. |
| T3 | Is reader disagreement omission, not false mentions? | **Passed.** Mask-based and two-reader π agree for missing teeth and implants; false-mention rate ≤ 2 %. Crowns/bridges agree on π; their 11 % "false mentions" are missing mask labels (visual check). |
| T4 | Is the effect on evaluation large? | **Passed, large.** A system with true precision 0.80 scores 0.22–0.71 against one reference report, depending on the type; a system that stays silent on under-reported types looks 21 points more precise than an equally good complete one. |

Recommendation: **GO**, with the conditions in §8.

## 2. Inputs and provenance

| Item | Location | Code |
| --- | --- | --- |
| Parser v3.2 full run (EN, IT, EN repeat), frozen | `results/extract/20261110-2029-f4df9e5-parse-qwen3-32b-v32-n0-615806` | `f4df9e5`, clean |
| T2 main (estimand B) | `results/propensity/20261111-1342-76165e3-propensity-gt-v32-B-upperbone` | `76165e3`, clean |
| T2 secondary (estimand A) | `results/propensity/20261111-1342-76165e3-propensity-gt-v32-A-upperbone` | `76165e3`, clean |
| T2 with the previous visibility rule (sensitivity) | `results/propensity/20261111-1056-6901b70-propensity-gt-v32-B` (and `-A`) | `6901b70`, clean |
| T3 | `results/propensity/20261111-1326-132ed35-mask-agreement-gt-upperbone` | `132ed35`, clean |
| T4 | `results/propensity/20261111-1343-76165e3-effect-size-gt-upperbone` | `76165e3`, clean |
| T1 small check (review log) | `results/review/20261110-2029-f4df9e5-parse-qwen3-32b-v32-n0-615806/review_log.jsonl` | list `…-s20261111-first20-mechanical.json` |

Population: P/F/S double-read cases; default exclusions P116/P117/P260/P261 (misfiled reports) and S0015
(possibly upside down); cases with a failed or salvaged-with-drop report removed (8 cases, 9 reports).
514 cases used in T2. Case-level bootstrap, 1000 resamples. All T2–T4 code was reviewed by an independent
agent before it ran on real data (findings and fixes: DECISIONS 2026-11-11).

## 3. T1 — parser

**Full run health** (job 615806): 1000/1000 EN and IT reports parsed, 0 errors, 0 logged request lines;
EN run-to-run agreement F1 0.989 (per report, slot level); EN vs IT 0.900. Internal check against
Lumetti et al.'s labels (not for the paper): held-out in the range of the tune half, so the parser was
frozen (DECISIONS 2026-11-11).

**Small human check** (Shivam, 2026-11-11): the first 20 base reports of the fixed held-out review list
(seed 20261111), judged only for the 7 "mechanical" types (tooth number plus plain wording).
Counts from the review log (summarised with an ad-hoc script; the full T1 will compute them with repo code):

| Type | correct | wrong tooth/side | not in report | unclear | missed teeth/sides |
| --- | --- | --- | --- | --- | --- |
| MandibularCanalCourse | 35 | 0 | 0 | 0 | 4 |
| ToothAbsence | 16 | 1 | 1 | 0 | 1 |
| EndodonticTreatment | 16 | 0 | 0 | 0 | 0 |
| ConservativeRestoration | 13 | 0 | 0 | 1 | 0 |
| ProstheticDevice | 8 | 3 | 0 | 1 | 11 |
| DentalImplant | 3 | 0 | 0 | 0 | 0 |
| Caries | 1 | 0 | 0 | 0 | 0 |
| **All** | **92 / 99** | 4 | 1 | 2 | 16 |

**Error patterns** seen in the check, then counted on the 508 tune-half EN reports with word searches
(exploratory counts, scratch scripts not kept; text read by the scripts only, counts printed):

| Pattern | Count (tune half) | Assessment |
| --- | --- | --- |
| Bridge "from X to Y" (≥ 3 teeth) stored with only its end teeth | 7 of 19 bridges | real, repeating; ~2 % of crown/bridge mentions; π ProstheticDevice slightly low |
| Canal sentence without side or tooth: parser codes one side | 52 of 508 reports (Right 28, Left 24) | no side bias; side invented; review convention "no side = both sides" |
| Tooth group ("molar sector") or "teeth present" coded as a vague region | 33 region findings, 3 with no absence wording at all | rare |
| Tooth list copied from another finding into a bridge | not provable by counting (crowns often sit on root-canal-treated teeth) | open, checked again in full T1 |

Not covered yet: the 8 types that need dental knowledge (PathologicalLesion, PeriapicalLesion,
ToothDevelopmentPathology, JawAtrophy, RootResidue, PostExtractionSocket, sinus, condyle). PathologicalLesion
is already marked weak from the internal check.

## 4. T2 — reporting propensity π

Pooled P/F/S, double-read, main estimand B (a vague region statement counts as not naming the teeth).

| Finding type | π | 95 % CI | width | passes |
| --- | --- | --- | --- | --- |
| Caries | 0.27 | 0.16–0.40 | 0.24 | yes |
| PeriapicalLesion | 0.27 | 0.16–0.37 | 0.21 | yes |
| PathologicalLesion | 0.29 | 0.19–0.40 | 0.21 | yes |
| PostExtractionSocket | 0.45 | 0.24–0.64 | 0.40 | no (width) |
| ConservativeRestoration | 0.47 | 0.40–0.54 | 0.14 | yes |
| RootResidue | 0.49 | 0.30–0.64 | 0.34 | no (width) |
| ProstheticDevice | 0.61 | 0.55–0.66 | 0.11 | yes |
| ToothDevelopmentPathology | 0.62 | 0.56–0.67 | 0.11 | yes |
| JawAtrophy | 0.67 | 0.61–0.73 | 0.12 | yes |
| DentalImplant | 0.71 | 0.56–0.84 | 0.28 | yes |
| ToothAbsence | 0.72 | 0.66–0.77 | 0.11 | yes |
| EndodonticTreatment | 0.85 | 0.80–0.89 | 0.09 | yes |
| MandibularCanalCourse | 0.89 | 0.88–0.91 | 0.04 | no (upper ≥ 0.90) |
| MaxillarySinusFinding | 0.36 | 0.21–0.54 | 0.33 | no (low support) |
| MandibularCondyleFinding | — | — | — | not estimable |

- Estimand A (a vague reader drops out for the unnamed teeth) differs only for ToothAbsence: 0.75.
- Previous visibility rule (before the upper-jaw fix, §7): ToothAbsence 0.70, ProstheticDevice 0.60, all
  other types identical.
- Per centre (double-read; small numbers, descriptive only) the house styles differ strongly, e.g.
  ConservativeRestoration F 0.06, P 0.55, S 0.64; Caries F 0.02, P 0.33, S 0.60. Pooling across centres
  and the centre-level model (HANDOFF §2.1) matter for the decoder.
- Not yet corrected for parser error (full T1). Assumes independent readers given presence; positively
  correlated omissions (house style) make these π too high, i.e. the real incompleteness larger.

## 5. T3 — reader-based π against the manual masks

For mask-checkable findings, π measured directly against the TF3 masks (π_mask), the two-reader π on the
same instances fitted blind to the masks (π_readers), and the false-mention rate against the masks
(φ_mask). Double-read cases.

| Finding (unit) | π_mask (95 % CI) | π_readers (95 % CI) | φ_mask (upper limit) |
| --- | --- | --- | --- |
| ToothAbsence (tooth position, all permanent) | 0.71 (0.66–0.75) | 0.75 (0.70–0.79) | 0.017 (0.022) |
| ToothAbsence, lower jaw | 0.74 (0.69–0.79) | 0.79 (0.75–0.83) | 0.020 (0.026) |
| ToothAbsence, upper jaw (221 gaps) | 0.51 (0.37–0.64) | 0.41 (0.12–0.63) | 0.011 (0.020) |
| DentalImplant (case) | 0.95 (0.87–1.00) | 0.97 (0.92–1.00) | 0.006 (0.016) |
| ProstheticDevice: crown/bridge (case) | 0.77 (0.70–0.82) | 0.78 (0.72–0.83) | 0.106 (0.140) |

- **Readers rarely mention what is not there** (φ ≤ 2 %) for missing teeth and implants.
- **The two-reader π is unbiased or slightly high.** For missing teeth it is 4 points above the
  mask-based π, as expected when π varies between findings (obvious gaps are reported more often); the
  two-reader π is then an upper bound and corrections based on it are conservative.
- **Crowns/bridges:** π agrees. φ_mask 0.106 is above the 0.05 guidance, but in all 10 double-read cases
  where both readers name a crown/bridge and the mask has no crown/bridge label, Shivam saw the crowns in
  the scan (2026-11-11; one bridge was drawn as a tooth). These 20 of the 45 "false" mentions are mask
  gaps; even if all 25 single-reader ones were reader errors, φ ≤ ~0.06. The case list is kept locally.
- Implants and crowns/bridges are compared per case (their mask labels carry no tooth number), which
  checks P(at least one mentioned | at least one present), not the per-tooth π of T2.

## 6. T4 — effect on evaluation

Expected scores of a system with true precision 0.80 against **one** reference report, using T2's p and π
(φ = 0; true precision and sensitivity are assumed scenario values):

| Finding type | π | raw precision | raw − true | §2.2 corrected | corrected − raw |
| --- | --- | --- | --- | --- | --- |
| Caries | 0.27 | 0.22 | −0.58 | 0.24 | +0.02 |
| PeriapicalLesion | 0.27 | 0.22 | −0.59 | 0.25 | +0.03 |
| PathologicalLesion | 0.29 | 0.23 | −0.57 | 0.36 | +0.12 |
| ConservativeRestoration | 0.47 | 0.38 | −0.42 | 0.43 | +0.05 |
| ProstheticDevice | 0.61 | 0.49 | −0.31 | 0.51 | +0.02 |
| ToothAbsence | 0.72 | 0.57 | −0.23 | 0.60 | +0.03 |
| EndodonticTreatment | 0.85 | 0.68 | −0.12 | 0.68 | +0.00 |

(All 13 supported types: `effect_by_type.csv`.) Every type loses at least 8 points against an incomplete
reference (raw − true from −0.09 for the canal to −0.59 for periapical lesions).

Two systems with the same true precision (0.80) and sensitivity (0.70) on every type they report:
"complete" reports all 13 types, "conservative" only the 2 types with π ≥ 0.80. Raw precision 0.49 vs
0.70: the conservative system looks **21 points** more precise because of the references alone. Its true
recall is 0.11 instead of 0.70, so F1 still ranks the complete system first.

**The HANDOFF §2.2 corrected precision is not fit for use as is.** For a system with skill it
under-corrects, most for rare findings (Caries +0.02 against a bias of −0.58), it over-corrects where p ≈ 1
(MandibularCanalCourse: corrected 0.997 for a true 0.80), and it widens rather than removes the advantage of
the silent system (0.70 → 0.89 vs 0.49 → 0.55). A better estimator (HANDOFF L1, PU-style) is needed before
any system is scored.

## 7. Changes made while running the gate (2026-11-11)

All reviewed by an independent agent; details and evidence in DECISIONS.md.

1. **Implant sites count as noticed gaps (T3).** The parser codes "implant in 36" as DentalImplant only; an
   implant site has no tooth label in the mask.
2. **Reading of T3:** π_readers above π_mask means π varies between findings, not false mentions; false
   mentions are judged by φ.
3. **Upper-jaw visibility rule (T2, T3).** In short centre-P scans only the tips of a few upper teeth can
   reach into the picture (Shivam's visual check). An empty upper position now counts as visible only if
   the mask has at least 2000 mm³ of upper jaw bone (just below the smallest maxilla of the tall centre-F
   scans). Effect: π ToothAbsence 0.70 → 0.72, ProstheticDevice 0.60 → 0.61; T3 missing teeth 0.41 → 0.71
   (mask-based), now agreeing with the two-reader π.
4. **T1 in two steps** (a small check now, the full T1 before the final runs) with a scoped review mode.

Also confirmed again by Shivam's visual checks: TF3 tooth labels are mirrored left/right relative to the
reports (10 of 10 crown cases on the mirror tooth), and TF3 masks do not label all crowns and bridges.

## 8. Recommendation and conditions

**GO** for the π-aware decoder, under these conditions:

1. **Full T1 before the final runs:** all 113 listed reports; a dentist for the 8 knowledge types if one
   is available, otherwise those types are reported as "parser not verified" and kept out of headline
   claims. Records are evidence per (report, type) within their scope.
2. **Parser v3.3 before the LoRA retraining,** not now: fix bridge spans, canal side without a side cue
   ("no side = both"), group wording as teeth; one new full run. The parser output becomes training labels
   there, so that is the cheap moment.
3. **A corrected-precision estimator that works for skilled systems** (HANDOFF L1) before any system is
   scored; the §2.2 formula is kept only as a reference.
4. **π for the canal counted per report,** not per side (side invention, §3).
5. **Centre house style is large:** the decoder and the corrected metrics use centre-level π with partial
   pooling, not one pooled π.
6. Per HANDOFF §6.4: any π that feeds or scores a system is refitted on the right subset (other folds;
   development set only for the locked test set). The gate's all-cases π is never reused for that.

Open, not blocking: the split (four points for Shivam), the decoder boundary with Chen, the DUA review of
comment lines in the imported V2V2R code, and the visual check of S0015.

## 9. Decision (Shivam, 2026-11-11)

**GO in two stages.** The measurement result (T2–T4) is solid and already supports a paper on reference
incompleteness and its effect on evaluation. The π-aware decoder is the riskier part: proving that it helps
needs a corrected-precision estimator that works for skilled systems (§6), its largest gains are expected
for types without mask truth (caries, lesions), and the design must answer the circularity question (π used
both to build and to score the system).

- **Stage 1 (CPU only):** (a) a corrected-precision estimator validated in simulation for skilled systems
  (HANDOFF L1); (b) a simulation of the decoder idea with the real p and π and realistically imperfect
  models: does π-aware selection beat fixed silence rules on true scores, and by how much? (c) the split
  approved and committed; (d) the boundary with Chen's verifier paper agreed.
- **Second decision point before any GPU-heavy work** (segmentation retraining, per-fold LoRA, decoder): go
  on only if (a) works and (b) shows a clear gain. Otherwise the measurement paper is written from the gate
  results plus (a), and the decoder is deferred.
- The conditions of §8 apply to stage 2.

