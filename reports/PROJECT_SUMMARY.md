# Project summary — "When the Reference Is Partial"

State: Monday 9 November 2026, evening. Written in plain words for Shivam (and anyone he shares it with).
Every number below comes from a file produced by code in this repository (paths in `DECISIONS.md`); no
report text, images or masks are reproduced here. Items marked **provisional** will change once the
final parser (v3) and visibility rules are frozen.

---

## 1. The problem in one paragraph

When a model writes a radiology report, we grade it by comparing it with a report a clinician wrote
(the "reference"). Current metrics treat the reference as the complete truth: if the model mentions a
finding the clinician did not write down, it is counted as a **mistake** (false positive). But
clinicians do not write down everything they see — routine fillings, crowns, small caries, a missing
wisdom tooth are often left out. So the reference is **partial**, and a model that is *more complete*
than the clinician gets *punished*. Our project measures how partial the references are, corrects the
metrics for it, and checks whether conclusions about which model is better change.

## 2. What the project does, step by step

1. **Measure reporting propensity (π)** for each finding type: the probability that a clinician writes
   a finding down when it is really there and visible in the scan.
2. **Correct the metrics** with π (a correct-but-unreported finding is no longer a full mistake), and
   compute a **noise floor** (how well one clinician scores against another clinician).
3. Add two **controls** that show how much of any score really needs the image: a "corpus prior"
   (no image, just predicts common findings) and an "acquisition-only" model (only scan header numbers).
4. **Re-evaluate** existing systems — mainly **V2V2R**, Shivam's ToothFairy4 challenge model, which
   stays the centrepiece — on our own data splits, raw vs corrected.
5. Later: a **segmentation-aware decoder** for V2V2R whose gains show up mainly under corrected metrics,
   a **dentist re-reading study**, and a **second, non-dental dataset**.

Target journal: **Medical Image Analysis (MedIA)**, submission by end of June 2027.

Since 8 November a **go/no-go test (GT)** comes first: four cheap tests (T1–T4, §9) decide whether the
idea holds before any heavy compute (no V2V2R retraining, no segmentation retraining, no decoder work
until Shivam's GT decision).

## 3. Reporting propensity π — what it is, how it is computed, and whether it makes sense

### 3.1 The idea

Many cases were read by **two clinicians independently**. For one possible finding at one place
(e.g. "filling on tooth 36") there are three outcomes: both wrote it, one wrote it, neither wrote it.

- If findings were always written down when present, the two clinicians would almost always agree
  ("both" or "neither"). Many "only one wrote it" outcomes mean findings are often left out.
- The ratio of "only one" to "both" tells us π, exactly as in capture–recapture counting of animals
  (two independent "catches" of the same population).

### 3.2 The model (HANDOFF §2.1)

For each finding type separately, every possible **instance** = (case, finding type, slot); the slot is
a tooth (FDI number), a side (Left/Right) or a quadrant, depending on the finding type.

- With probability **p** (prevalence) the finding is really present at that slot.
- If present, each clinician writes it down independently with probability **π**.
- If absent, a clinician writes it with probability **φ** (false-positive rate). Base model: **φ = 0**.
- Clinicians of the same case are interchangeable (same π).

For an instance read by n clinicians, k of whom mention it, the likelihood is
`p · C(n,k) π^k (1−π)^(n−k) + (1−p) · [k = 0]` (with φ = 0). We fit p and π by maximum likelihood over
all instances of that finding type. **Two-reader shortcut (check):** with b = number of "both" and
o = number of "only one", **π = 2b / (o + 2b)**.

Worked example (14B parser, provisional): mandibular canal course, both = 540, only one = 139 →
π = 1080 / 1219 = **0.886**; the full maximum-likelihood fit gives 0.882. They agree, as they should.

Uncertainty: **case-level bootstrap** (1000 resamples of whole cases, because instances of one case are
not independent) → 95 % intervals. Safeguards (from the first independent code review): π is reported
as "not estimable" when no instance was read by ≥ 2 clinicians with ≥ 1 mention; a finding type only
counts as "supported" with ≥ 10 "both" instances, ≥ 30 mentioned double-read instances and ≥ 95 % of
bootstrap resamples identifiable.

### 3.3 Visibility — the key addition (HANDOFF §2.6, agreed 7 Nov)

A clinician cannot omit what is **not in the scan**. Scans differ hugely by centre (P and S are thin
~51 mm slabs around the teeth). So π is defined only on instances that are **visible**, and visibility
comes **from the image only** (masks, scan geometry), never from the reports. Three tiers:
(1) mask-checkable (missing teeth, implants, crowns, bridges) — π can be validated against masks;
(2) visible from the image but no truth (caries, fillings, root canal treatment, periapical lesions,
canal, sinus, condyles); (3) undecidable — reported separately only.

Provisional rules currently in code: tooth-level findings (caries, root canal treatment, filling,
development pathology, periapical lesion) need the tooth in the mask (periapical also needs the root tip
not cut by the scan edge); missing-tooth-type findings need the position to be "assessable" (between
present teeth of that quadrant, or the quadrant's teeth not touching the scan's side/back face; a
quadrant with no teeth in the mask needs the whole jaw clear of those faces); canal/sinus: label
present on that side; condyles: both mandible halves clear of the top and back faces; quadrant
findings: that jaw in the scan.

### 3.4 Two versions of π for vague statements (decided 9 Nov)

Some reports say "partial edentulism of the upper arch" without naming teeth. **Main analysis (B):**
π = how complete the reference is **tooth by tooth** — the vague report does not contain "tooth 16
missing", so it counts as not mentioning 16 (this is what corrected metrics need, because they compare
tooth by tooth). **Secondary (A):** π = how often clinicians who *name* teeth write a missing tooth down —
the vague reader is left out for those teeth. Both will be reported.

### 3.5 How π is used: corrected precision (HANDOFF §2.2)

For a model's finding that no reference mentions, the probability that it is nevertheless correct is
`P(present | 0 of n mention) = p(1−π)^n / (p(1−π)^n + (1−p)(1−φ)^n)`. Corrected precision = (matched
findings + sum of these probabilities for unmatched findings) / all findings. Raw and corrected are
always reported side by side. Recall stays reference-based, with a sensitivity analysis for the
possibility that omitted findings are also harder for models.

### 3.6 Does it make sense? Honest assessment

What supports it:
- **References are partial — independently confirmed.** Lumetti et al. (MICCAI 2026, same ToothFairy
  reports) report human-vs-human agreement on 358 two-report patients of only pres-F1 0.81, inst-F1 0.70,
  rep-F1 0.67. Our own π values (§8) are far below 1 for most finding types.
- **Disagreement is mostly omission, not mislabelling.** Our laterality check (§7.2): left/right swaps
  between two clinicians (36 vs 46) were 22, the same as the chance control (36 vs 26: 34), and all
  tooth-number confusions together explain < 2 % of the 3,128 one-reader-only mentions.
- **The estimator works on simulated data** (tests: true π 0.60 and 0.85 recovered within the CI from
  3,000 simulated cases; closed form and MLE agree).

What could bias it, and in which direction:
- **Shared house style** (clinicians at one centre omit the same things) → π looks **too high** →
  corrections too small (conservative). The planned simulation study (M0.5) will size this.
- **Parser errors** look like omissions → π looks **too low**. That is why the parser is tested hard (T1)
  and why parser noise is measured (run-to-run per report 0.988, §6).
- **φ = 0 may be wrong** (clinicians also write things that are not there). Only masks can check this,
  for tier-1 findings (T3, still to run). If φ is clearly > 0, we switch to a false-positive-aware model.
- **Pooling** over all teeth and centres mixes easy and hard slots; with two readers this biases π
  upwards (conservative), noted by the reviewer.
- **Interchangeable readers:** we do not know who wrote which report (file prefix `report_x` is probably
  not a reader ID: Lumetti says 8 readers, we see 16 prefix values), so reader-level π is not possible.
- **Corrected precision uses the average prevalence p of a slot.** A model's unmatched predictions are
  not random slots — a good model predicts where findings are likely — so the formula **under-corrects**
  (again conservative). Worth stating in the paper; masks can check it for tier-1 findings (M2 step 6).

Bottom line: the method is sound for what it claims (completeness of references for visible findings),
its main biases point in the conservative direction, and the remaining big unknown — whether
disagreement contains false mentions (φ) — is exactly what T3 tests next.

---

## 4. The data

### 4.1 ToothFairy4 (our copy)

- Uploaded by Shivam on 7 Nov: zip 47.7 GB, 2,622 files, unpacked 45 GB in `~/Datasets/ToothFairy4/`
  (owner-only permissions). Release looks like a late-July-2026 build (latest file date 2026-07-29).
- One folder per case: `cbct/volume.nii.gz`, `reports_en/`, `reports_it/`. **No masks, no "facts", no
  official train/test split** in this release.
- **622 cases**: **P 412, A 95, F 63, S 52**. Centres: P/F/S = ToothFairy3 public Sets A/B/C;
  **A = external cohort from Sfax (Tunisia)**, different site and scanner.
- **1,000 English + 1,000 Italian reports.** The **Italian files are the originals** written by the
  clinicians; the English files are **GPT-5.2 translations** (by the dataset providers; only random
  subsets were clinician-checked). Each English file pairs with the Italian file of the same name
  (one reading, two languages).
- Official dataset page says 625 cases / 1,001 reports / max 3 per case; ours has 622 / 1,000 / four
  cases with 4 → a different release version. Official page: most cases "have exactly two reports
  redacted by different maxillofacial surgeons with >5 years of experience".

### 4.2 Reports per case

| Centre | Cases | EN reports | 1 report | 2 | 3 | 4 | Share with ≥ 2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | 95 | 97 | 93 | 2 | 0 | 0 | 2 % |
| F | 63 | 127 | 0 | 62 | 1 | 0 | 100 % |
| P | 412 | 678 | 154 | 254 | 0 | 4 | 63 % |
| S | 52 | 98 | 8 | 42 | 2 | 0 | 85 % |
| **All** | **622** | **1,000** | **255** | **360** | **3** | **4** | **59 % (367 cases)** |

- Centre A has only **2** multi-report cases → it contributes almost nothing to π; its role is a
  "new site" generalisation test later.
- File names `cbct_<X>_<Y>.txt`: X has 16 values (14–35); in 363 of 367 multi-report cases all reports
  have different X. X crosses centres (e.g. 14 on F and P). Lumetti's paper says the reports were
  written by 2 radiologists + 6 maxillofacial surgeons (8 readers) → X is probably **not** a reader ID.
- Report length (EN characters, median [min–max]): A 867 [301–2780], F 616 [280–1500],
  P 540 [142–2160], S 582 [177–1570].

### 4.3 Scans (headers)

| Centre | Orientation | Voxel size (mm) | Field of view x × y (mm) | Height z (mm) | Height < 60 mm |
| --- | --- | --- | --- | --- | --- |
| A | LPS | 0.175 [0.12–0.45] | 83 × 83 | 90 [60–120] | 0 / 95 |
| F | LAS | 0.30 | 123 × 123 | 82 [41–89] | 1 / 63 |
| P | LAS | 0.30 | 111 × 101 | 51 [45–53] | 412 / 412 |
| S | LAS | 0.30 | 111 × 107 | 52 [41–62] | 49 / 52 |

P and S are thin slabs around the teeth; acquisition is fully confounded with centre.

### 4.4 Checks done on the data

- **Data copies:** our copy was byte-identical to `~/cbct-vlm-transfer/data/raw` (2,622/2,622 files) →
  that copy replaced by a link; the zip and that project's preprocessed copies were deleted (~93 GB +
  ~74 GB freed). Geometry matched the earlier project's table for 622/622 cases.
- **Independence of the two reports of a case:** 984 reports, 369 within-case pairs, **0** near-duplicate
  pairs (thresholds cosine > 0.9 or Levenshtein > 0.85). Within-case similarity median 0.18 (max 0.77);
  a report's most similar report in *another* case has median 0.47 → reader writing style dominates,
  the two reports of a case were written independently. 15 EN files in 5 identical groups across cases,
  all X = 14 (template reuse by one reader, never twice in one case).
- **Four special cases P116, P117, P260, P261** carry 4 reports each (two per X). P120, P121, P262, P263
  (and P378) exist in ToothFairy3 but not in ToothFairy4. Earlier hypothesis: their reports were filed
  under P116 etc. **New evidence (9 Nov):** all 16 of these reports fit their own scans well (P260/P261:
  10–14 agreeing items, ≤ 2 contradictions) → not obviously misfiled; maybe the same patients scanned
  twice. Still excluded from π for now.
- **S0015** may be stored upside down; F028, F055, P230, S0019 fit their reports slightly better without
  the left/right correction (§7.1). All five await Shivam's visual check.

### 4.5 ToothFairy3 masks (for P/F/S)

- ToothFairy3 provides hand-drawn 77-class label masks for all **527 P/F/S** scans.
- The TF4 volumes are **voxel-identical** to the TF3 volumes after flipping two array axes (527/527).
  TF3 headers are upside down (TF4 headers correct for up/down: upper jaw above lower jaw in 224/228
  testable cases).
- What masks can confirm: each tooth present/absent (64 tooth + 32 pulp labels), implants, crowns,
  bridges, jaws, canals, sinuses, pharynx. **Not** fillings, root canal fillings or condyles.
- **Left/right (solved 8–9 Nov):** the TF3 tooth labels are **mirrored** relative to the reports' FDI
  numbers (§7.1), and so are TF3's canal and sinus labels (499/500 and 47/51 cases); the TF4 header's
  left/right agrees with the reports (494/495). We mirror TF3 labels for P/F/S before any comparison.
  V2V2R's own segmentation already corrects this when it runs (found in August 2026, now confirmed at scale).

### 4.6 What the scans show (visibility features, P/F/S, run of 9 Nov)

| | F | P | S |
| --- | --- | --- | --- |
| Mandible (lower jaw) label present | 100 % | 100 % | 100 % |
| Maxilla (upper jaw) label present | 97 % | 30 % | 85 % |
| Mandible touches the top of the scan (condyles cut) | 100 % | 100 % | 100 % |
| Median left sinus volume (mm³) | 741 | 0 | 0 |
| Median teeth present, upper / lower | 14 / 14 | 8 / 13 | 14 / 14.5 |
| Teeth with root tip cut by the scan edge | 17.5 % | 35 % | 44 % |
| Whole mandible clear of side/back faces | 97 % | 73.5 % | 96 % |

Consequences: condyles are never fully inside P/F/S scans (condyle findings cannot be estimated there);
sinuses are mostly outside P and S scans; P often lacks the upper jaw.

Centre A has no hand-drawn masks; we segmented it with the old U-Mamba2 model. Jaw/canal/sinus results
stand (sinuses visible in ~95 % of A cases, condyles inside ~43 %), but the tooth-level A masks were
wrong (our bug, §10) and are being regenerated later.

---

## 5. Finding types (the ontology)

We use the finding vocabulary of **Lumetti et al.** (MICCAI 2026, same reports), re-implemented in our own
code and cited (their repository has no licence, so none of their files were copied). Paper: 13 types;
released version: **15 types**; we use the released version.

| Finding type | Anchored to | Plain meaning |
| --- | --- | --- |
| ToothAbsence | tooth (or region, v3) | missing tooth (incl. bridge pontic positions; fully toothless arch = 16 teeth) |
| ToothDevelopmentPathology | tooth | impacted/unerupted/ectopic/extra teeth, incomplete roots; tooth-specific bone loss around one tooth |
| Caries | tooth | decay |
| PeriapicalLesion | tooth | lesion at the root tip |
| EndodonticTreatment | tooth | root canal filling |
| ConservativeRestoration | tooth | filling |
| RootResidue | tooth | remaining root |
| PostExtractionSocket | tooth | recent extraction socket |
| DentalImplant | tooth | implant present |
| ProstheticDevice | tooth | crown, bridge, post-and-core, abutment, removable prosthesis, orthodontic appliance |
| MandibularCanalCourse | side | course of the lower jaw nerve canal and its contact with teeth |
| MandibularCondyleFinding | side | abnormal jaw joint head |
| MaxillarySinusFinding | side | sinus pathology |
| JawAtrophy | quadrant | general bone loss of a region |
| PathologicalLesion | quadrant | cysts, tumours, other bone lesions |

Plus our own addition `not_assessable` (statements that a structure is outside the scan — not findings),
and 14 qualifier groups (severity, orientation, eruption, canal trajectory, prosthesis type, …).

**Lumetti's released labels:** 529 files (one per patient, all reports of a patient merged into one list,
extracted with GPT-5.2 and clinician-validated), 7,434 findings, 7,741 instances, 14 of 15 types occur
(no condyle findings), 56 findings without a tooth/side/quadrant. Counts per type: missing tooth 1,731,
canal 1,069, filling 956, development pathology 940, prosthesis 616, jaw atrophy 606, root canal
treatment 530, caries 217, periapical 204, implant 173, bone lesion 156, extraction socket 115,
root residue 82, sinus 39. We use these labels **only internally** to check our parser; never
redistributed, never presented as our result.

---

## 6. The report parser (T1)

Turns each report into structured findings with a **local** language model (never an external API),
served with vLLM on the cluster; output forced into our JSON format and validated. Prompt in English with
**made-up example reports** (no real report text). Each report is parsed separately (no merging).
Both English and Italian are parsed; English is the main input (no one in the team reads Italian).

Development split: cases are hashed into a **tune** half (we may look at its errors to improve the
prompt) and a **held-out** half (reported, never inspected) — to avoid fooling ourselves.

| Version / model | Failed reports (EN) | Slot F1 tune | Slot F1 held-out | Type F1 held-out | Notes |
| --- | --- | --- | --- | --- | --- |
| v1, Qwen3-14B (8 Nov) | 16 cut off | 0.805 | 0.830 | — | old scorer |
| v2, Qwen3-14B | 17 | 0.815 | 0.837 | 0.938 | re-scored with fixed scorer |
| v2, **Qwen3-32B** (full bf16) | 4 | **0.845** | **0.848** | **0.948** | re-scored with fixed scorer |
| v2, Qwen3-32B, Italian input | 6 | 0.830 | 0.836 | — | |
| v3, Qwen3-32B | — | — | — | — | **running tonight** (job 615200) |

("Slot F1" = agreement with Lumetti on finding type + tooth/side/quadrant; "type F1" = finding type
only. These measure **agreement** with Lumetti, not accuracy — Lumetti's labels have their own errors.)

Per finding type, Qwen3-32B v2, held-out slot F1 (number of Lumetti instances): canal 0.968 (495),
filling 0.947 (456), implant 0.944 (74), sinus 0.923 (19), root canal treatment 0.918 (231), root residue
0.912 (35), jaw atrophy 0.909 (231), extraction socket 0.851 (55), caries 0.819 (101; tune 0.971),
prosthesis 0.813 (479), **missing tooth 0.782 (772)**, periapical 0.769 (76), development pathology
0.752 (418), bone lesion 0.683 (74).

Consistency (Qwen3-32B v2, per report): **run-to-run 0.988** (the parser is very stable);
**English vs Italian 0.885** (the translation or the language changes ~1 in 9 findings; lowest for
development pathology 0.77, bone lesions 0.70, periapical 0.80). Since nobody can read Italian, this
stays a reported limitation.

Parser history and fixes:
- v1 → v2 (8 Nov): conventions measured from Lumetti's labels (bridges + pontics, tooth-specific bone
  loss = development pathology, toothless arch = 16 teeth, sides from tooth numbers, no condyle/sinus/
  planned-implant over-calls); output limit 4096 tokens; one retry with repetition penalty for loops.
- Independent code review (9 Nov): 15 issues found and fixed (e.g. reports lost when an answer had the
  right format but a missing side; scoring that only looked at cases with findings on both sides;
  run-to-run checks that compared per-case unions instead of per report). All earlier numbers re-scored.
- **Failed reports are not dropped any more** (Shivam's decision): failures were normal-length reports
  where the model looped; v3 retries with stronger settings (penalty 1.1 → 1.3 → small seeded randomness).
- **The "e.d." bug (found 9 Nov with Shivam's reading of two reports):** Italian reports abbreviate
  "tooth" as "e.d." (elemento dentario); 33 English translations kept it (224 Italian reports use it).
  The parser read it as "edentulous" and turned "teeth present from X to Y" into 16 missing teeth. In
  such reports 60 % of missing-tooth calls fell on teeth the mask shows present (vs 9 % elsewhere) —
  about 30 % of all such errors. v3 fixes the prompt, adds region-level missing teeth, a within-report
  check (a tooth cannot be both missing and filled), and the retry sequence.

Other parser findings: the whole-arch "missing" calls agree with the masks in 66 of 76 cases; the
report-to-scan fingerprint (missing-tooth pattern of each report vs every scan) found **no misfiled
reports** (own scan is the best fit for 81–84 % of reports; median contradiction rate 0).

---

## 7. Laterality and consistency checks

### 7.1 Left/right of masks vs reports
With TF3 tooth labels as they are, reports put findings on "missing" teeth far more often than after
mirroring: root canal treatment 279 → 86, filling 167 → 28, development pathology 240 → 83,
periapical 70 → 34, caries 36 → 9. Per case: mirrored fits better in 214, unmirrored in 13, tie in 37 →
the mirroring is a dataset-wide convention, not individual scans.

### 7.2 What remains after mirroring
214 findings on mask-absent teeth: 162 written by only one of several clinicians (reader or parser
error), 25 by ≥ 2 clinicians (probably the mask: root canal 15, development pathology 7, filling 3),
27 in single-read cases. Clinician side swaps are at chance level (22 vs 34), neighbour shifts 57.

---

## 8. First π estimates (T2, provisional)

Qwen3-14B parser v2, P/F/S, double-read cases only, mirrored masks, 508 cases used (14 dropped for failed
parses; P116, P117, P260, P261, S0015 excluded). **Provisional**: computed before the "e.d." fix (which
mainly affects missing teeth), before the canal/sinus side fix and before estimand B.

| Finding type | π | 95 % CI | Prevalence p (per visible slot) |
| --- | --- | --- | --- |
| Mandibular canal course | 0.88 | 0.86–0.90 | ≈ 1.00 |
| Root canal treatment | 0.83 | 0.78–0.87 | 0.06 |
| Implant | 0.68 | 0.57–0.79 | 0.02 |
| Jaw atrophy | 0.64 | 0.58–0.70 | 0.35 |
| Missing tooth | 0.62 | 0.56–0.68 | 0.21 |
| Prosthesis | 0.59 | 0.52–0.66 | 0.10 |
| Development pathology | 0.58 | 0.52–0.64 | 0.09 |
| Filling | 0.51 | 0.43–0.59 | 0.14 |
| Root residue | 0.47 | 0.29–0.62 | 0.01 |
| Bone lesion | 0.36 | 0.25–0.47 | 0.20 |
| Periapical lesion | 0.32 | 0.20–0.43 | 0.05 |
| Caries | 0.26 | 0.16–0.38 | 0.05 |
| Extraction socket | 0.25 | 0.11–0.42 | 0.02 |
| Sinus finding | 0.32 | (not supported: too few) | — |
| Condyle finding | not estimable (condyles never fully in P/F/S scans) | | |

Reading: clinicians write down a root canal treatment ~4 times in 5, a filling about half the time, and
caries only about 1 time in 4 when it is there and visible. The canal "prevalence ≈ 1" is because the
ontology codes the canal course as a finding even when it is normal. Normal and mirrored masks gave the
same π within 0.015 (pooling hides left/right); per-tooth analyses (T3) need the mirror.

---

## 9. The go/no-go test (GT) — status

| Test | Question | Guidance to pass | Status |
| --- | --- | --- | --- |
| T1 Parser | Is the parser good enough that errors don't masquerade as omissions? | slot F1 ≥ 0.90 for ≥ 6 common types (≥ 100 instances); no systematic error in a manual check | **not yet passing.** 32B v2 held-out: ≥ 0.90 for 7 types, but only **4 of them are common** (canal 0.97, filling 0.95, jaw atrophy 0.91, root canal treatment 0.92); the other common types are below (caries 0.82, prosthesis 0.81, missing tooth 0.78, development pathology 0.75). v3 targets missing teeth; manual check via the blinded audit sheet still to do |
| T2 Propensity | Are references really partial? | ≥ 3 types with upper CI of π < 0.90 and CI width ≤ 0.30 | provisionally **yes** (most types; to be redone with v3) |
| T3 Mask agreement | Is disagreement omission, not false mention? (missing teeth, implants, crowns/bridges vs masks) | CIs overlap for ≥ 2 of 3 types and φ ≤ 0.05 | **to do** |
| T4 Effect size | Does correcting change conclusions? | corrected − raw precision ≥ 5 points and/or two systems swap rank | **to do** |

Decision rules (HANDOFF §7): T1 fails → fix the parser first; T3 fails → switch to a false-positive-aware
model; T2 or T4 fails → pivot to a measurement/benchmark-analysis paper with V2V2R as the second pillar.

---

## 10. Everything else that happened (chronological, 7–9 Nov)

**7 Nov (setup):** environment checks (`reports/ENV_REPORT.md`): SLURM account `uzk`, partitions `cpu` /
`gpu` / `storage`, GPU QoS must match the GPU type, 21 A100 40GB plus A100 80GB, H100, B200 and MIG
slices, max wall 3 days, no module system, conda, no visible quota (shared filesystem ~99 % full, 4.6 TB free).
Repository created (`github.com/shivamsingh-ml/partial-refs`, private), pre-commit hook blocking image
files and files > 5 MB, decision log, M0.5 estimator code with tests. **V2V2R code:** extracted the exact
submitted code from the challenge Docker image (307 files compared; no public commit matched it) into
`external/odin2026/` — no container needed. Own data splits decided (locked ~20 % test set, 5-fold CV,
leave-one-centre-out; V2V2R retrained per fold later). GPU environments built (`pr_serve`: vLLM 0.19,
`pr_train`: torch 2.11 / transformers 5.9). Models downloaded (Qwen3.5-9B, AWQ base, V2V2R released model,
later Qwen3-14B and Qwen3-32B). **V2V2R smoke test passed** end to end on 3 cases (A100 40GB, 5–6.6 min per
case; vLLM does not run on MIG slices). Lumetti et al. found and their use agreed (ontology + internal
label check; their model not used; reader-level analysis dropped). Second-domain shortlist written
(`reports/SECOND_DOMAIN_OPTIONS.md`: VinDr-CXR primary, REFLACX secondary). An HF token file was once
committed by accident locally; GitHub blocked the push, it was removed from history and never published.

**8 Nov:** go/no-go test inserted; parser built and run (v1 → v2); T2 code built, independently reviewed
(3 serious bugs fixed, incl. one that would have silently used 0 cases); T2 run; left/right mirroring found.

**9 Nov (overnight + day):** Qwen3-32B parse (1 h 12 min on A100 80GB); second full independent review
(15 issues, all fixed and verified); visibility v2, side geometry, fingerprint, laterality; the "e.d."
bug; parser v3 built, reviewed (must-fix applied), committed (`4bfd6f0`) and queued. **Our own bug:** the
centre A segmentation job converted tooth labels twice (scrambling tooth numbers, losing teeth 47/48);
fixed and verified on 3 cases (all four quadrants present); the full 95-case run is postponed until after
the GT decision because GT does not need centre A. Also noticed: V2V2R's `facts.py` comment says its VLM
was trained on facts with wrong quadrant numbers, while the submitted pipeline now feeds correct FDI
numbers — Shivam to check whether training and inference disagree.

**Compute so far for this project:** ~3.8 GPU-hours + ~37 CPU-core-hours (Shivam's total since 1 Nov:
50.5 GPU-hours, mostly other projects; fair share 0.70 vs 0.72 maximum; usage half-life 14 days).
Disk: `sae_tooth/acts` (77 GB) deleted on Shivam's request.

**Data protection (DUA):** report text, volumes and masks are never committed, pushed or sent to
external services; only counts and IDs are written. On 9 Nov a short report excerpt was pasted into the
chat by mistake; flagged immediately — Shivam to decide whether it needs reporting under the DUA.

---

## 11. Open items and next steps

1. **Tonight:** parser v3 (Qwen3-32B, English + Italian + repeat, job 615200, expected ~23:15).
2. **Tomorrow:** v3 scores (tune/held-out/per report/EN vs IT) → freeze the parser (choose on tune);
   post-run checks (toothless arches still 16 teeth, change in missing-tooth calls, conflicts, sampled retries).
3. T2 again with v3, both estimands (B main, A secondary); then **T3** (masks vs reports for missing teeth,
   implants, crowns/bridges; φ) and **T4** (effect size) → `reports/REPORT_GT.md` → Shivam's go/no-go.
4. Blinded audit sheet for Shivam (~60–80 items, IDs only; parser-only, Lumetti-only, both-agree samples)
   to measure parser and Lumetti accuracy against the report text.
5. Waiting on Shivam: visual check of S0015, F028, F055, P230, S0019; V2V2R training-facts question;
   DUA decision about the pasted excerpt.
6. After GO: simulation study (M0.5 report), data split (M1 step 3b, Shivam approves), centre A
   segmentation (95 cases), corrected metrics and noise floor (M2), V2V2R re-evaluation per fold (M3),
   decoder (M4), dentist re-reading study and second domain (M5/M6).
