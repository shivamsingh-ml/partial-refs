# Paper plan (draft for Shivam, 2026-11-12) — replaces the draft of 2026-11-11

## In one sentence

Report generators trained on incomplete clinical reports learn to stay silent; we build a CBCT report
generator that examines each tooth with a dedicated model, grounds checkable findings in the segmentation,
and does not learn silence, and we evaluate it against image-derived truth instead of single reports.

**Type:** methods paper (target MICCAI 2027, or MedIA with fuller evaluation). **Fallback:** the same system
as an application paper (e.g. AIIM, a dental or radiology imaging journal) if the new part gains little.

## What the paper contains

| Part | Content | Status |
|---|---|---|
| Motivation | Single reports are incomplete (π per finding, gate T2); checked against the outlines (T3); scoring against one report rewards silence (T4). One paragraph on why: different authors cover different findings (author profiles). | done |
| **New method** | **Tooth-by-tooth perception:** the segmentation cuts out each tooth; a dedicated model examines it for fillings, crowns, root canal fillings, caries and root-tip infection; its results feed the report. Plus **findings that go together** (e.g. root canal treatment → look for root-tip infection). | to build |
| System around it | Checkable findings (missing teeth, implants, crowns) read from the segmentation; training labels where "not mentioned" is unknown, labels only from authors who report that finding type, both reports of double-read cases used; reporting thresholds chosen sensibly instead of switching finding types off. | to build |
| Evaluation | Against the outlines where they give truth (missing teeth, implants, crowns); a dentist's check of a sample for caries and root-tip infection; the usual single-report score shown alongside. Locked test set and centre A once at the end. | partly ready (split v1, outlines) |
| Baseline | The current V2V2R-style system, retrained on our development folds. | to do (GPU) |

## How V2V2R is used: baseline and foundation

| V2V2R part | In the new system |
|---|---|
| Segmentation (U-Mamba2): teeth, jaws, canals, implants | kept, retrained without the test cases; the per-tooth model needs it to cut out each tooth |
| Facts from the segmentation (teeth present, implants, …) | kept, with the audit fixes ("segmentation first") |
| Per-tooth image views (`create_tooth_detail.py`) | reused as input for the per-tooth model (candidate 3) |
| Language model with LoRA (Qwen3.5-9B), per-tooth questions | kept, retrained with clean labels; gets the per-tooth model's results as extra input |
| Silence rules (switch off weak finding types) | replaced by sensibly chosen thresholds |
| Report assembly | kept |

New system = V2V2R + per-tooth model, clean labels, no silence rules. Baseline = V2V2R as it is,
retrained on the development folds. Cited as the published V2V2R paper; no challenge result needed.

## Candidates for the dedicated per-tooth model

| # | Model | Good for | Cost |
|---|---|---|---|
| 1 | Intensity features inside each tooth (bright material in crown / root / on top; dark spots in the crown; dark zone at the root tip) + small classifier | fillings, crowns, root canal fillings; first try at root-tip infection | CPU, hours |
| 2 | Features from our segmentation network inside each tooth outline + small head | all types | GPU, short |
| 3 | 3–5 2D slices per tooth + pretrained 2D image model + small head (builds on V2V2R's per-tooth views) | caries, lesions | GPU, moderate |
| 4 | Small 3D network on the tooth box (+ outline channel) | all types | GPU, moderate–large |
| 5 | Large pretrained 3D medical model as feature extractor (availability for dental CBCT to be checked) | possibly strongest | GPU, uncertain |

Order: 1 first; then 2 or 3 for caries and root-tip infection; 4 and 5 only if needed.

## Steps and decision points

| Step | Work | Compute | Decision |
|---|---|---|---|
| 1. Cheap first test | Candidate 1 on the development folds: fillings, crowns, root canal fillings, root-tip infection; compared with what reports say for the same teeth (labels from authors who report the type) and with outline truth where it exists. | CPU, confirmed per job | **D1:** does per-tooth analysis detect these findings clearly? Yes → continue. No → rethink the new part before any GPU. |
| 2. Hard findings | Candidate 2 or 3 for caries and root-tip infection, one fold; plus "findings that go together". | GPU, small; confirmed per job | **D2:** clear gain on the hard findings? Yes → methods paper. No → application paper with what works. |
| 3. Baseline | Segmentation retrained without the test cases (audit fixes); report generator retrained once on the development cases. | GPU: about 10 days segmentation + one training run | — |
| 4. Full system | Per-tooth model + segmentation-first + clean labels + thresholds, on all folds. | GPU | **D3:** beats the baseline against outline truth and in the dentist's check? |
| 5. Final | Locked test set and centre A once; full T1 review of the parser; dentist's sample check. | inference only | paper |

## Risks

- Caries and root-tip infection are hard to see in CBCT; the per-tooth model may not reach useful accuracy.
- Labels come from incomplete reports; mitigated by author-aware label selection and double reads, but not removed.
- A dentist is needed for the evaluation of caries and lesions.
- One dataset; a second one would strengthen a MedIA submission.
- Publication timing: the challenge document sets a 6-month embargo after MICCAI 2026 (from about April
  2027) for participants' own results, except at the ODIN workshop; Shivam judges whether it applies.

## Rule for compute

No job on the cluster without Shivam's confirmation for that job (command, CPU/GPU, cores, time shown first).
