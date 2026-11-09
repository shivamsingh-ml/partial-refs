# Literature and novelty check — "When the Reference Is Partial"

Date: 9 November 2026. Three parallel web searches (report-generation evaluation; statistical methods and
analogues; dental report generation). Every reference was checked by the searching agent against its
arXiv / DOI / venue page unless marked **UNVERIFIED**. Re-check all 2026 preprints for newer versions before
submission. No project data was used.

## 1. Verdict in one paragraph

Nobody we found **estimates per-finding reporting propensity from independent double reads and uses it to
correct report-generation metrics**, and nobody uses **segmentation masks to decide whether a reference's
silence is evidence of absence**. But the *problem* is now recognised: concurrent 2026 work names "omission
noise" (PU-DPO, chest X-ray) and "non-exhaustive report supervision" (EviDent-CBCT, on ToothFairy4 itself),
and both handle it **at training time only**, with **fixed or naive assumptions**, and without correcting
**evaluation**. The statistical machinery is classic (Dawid–Skene, Hui–Walter, capture–recapture), and
propensity-scored metrics for missing labels exist in recommender systems, search evaluation and
positive-unlabeled (PU) learning. Our contribution is therefore best framed as a **principled transfer** with
genuinely new elements (§4), positioned explicitly against EviDent-CBCT and PU-DPO.

## 2. Closest prior work (ranked)

| # | Work | What they did | Overlap | What it means for us |
|---|---|---|---|---|
| 1 | **EviDent-CBCT** — Hao et al., arXiv 2610.02375 (Oct 2026), "…under Non-Exhaustive Report Supervision" | ToothFairy4 system; says a non-mention may mean absence *or* non-reporting; training down-weights non-mentioned tooth negatives with **hand-set** per-class weights (missing 0.95 … caries 0.20, root fracture 0.10); concedes its metrics measure agreement with reports, not accuracy; no evaluation correction | **Strong** (same dataset, same insight) | Must cite and differentiate: they *guess* propensities for training; we *estimate* them from double reads and *correct evaluation*. Their hand-set weights are a hypothesis our π can test directly (figure: their weights vs our π per type). |
| 2 | **PU-DPO** — Kobayashi et al., arXiv 2608.05341 (Aug 2026; NeurIPS 2026 poster per search, not opened) | CXR "omission noise"; 2 radiologists re-read CheXpert images for 3 findings; omission vs original report: effusion 11.3 %, atelectasis 38.2 %, cardiomegaly 43.8 %; truth = union of both readers; PU treatment in DPO **training** | **Strong on problem, partial on method** | Cite as concurrent; adopt their vocabulary ("omission noise"); show their union-of-readers truth is biased relative to the latent model (compare both on our data). Omission depends on clinical context (e.g. cardiomegaly in ICU) → allow π to vary by context/centre. |
| 3 | **"Reporting Practice Matters"** — Jeong et al., arXiv 2609.19093 (Sep 2026) | Rewrites references preserving content (ReRef); rankings of 9 CXR models change (RadCliQ); releases 120 radiologist-validated reference pairs | Strong on "reference choice changes rankings"; none on propensity | Main evaluation-side comparator; second-domain test: is our corrected metric more stable under their rewrites? |
| 4 | **George et al.**, "Clinical Reasoning Under a Partially Observed Objective in CBCT Report Generation", arXiv 2609.13238 | ToothFairy4: image encoder at chance (AUC 0.486); acquisition centre predicts sentences better (0.718) than the image model (0.663); header predicts anatomical coverage (mandible AUC 0.945, condyle 0.872); 8 fixed + 5 header-gated sentences reach METEOR 0.354 | Partial | Our no-image / header-only controls are well motivated; **our mask-based visibility must beat (or include) a header-only proxy**. |
| 5 | **Lumetti et al.**, MICCAI 2026 — per ODIN news page, **the ToothFairy4 dataset paper** | 893 reports / 529 TF3 volumes, 2 radiologists + 6 maxillofacial surgeons, ontology (paper 13 types; released 15), human agreement pres-F1 0.81 / inst-F1 0.70 on 358 two-report patients (numbers read from the PDF Shivam provided on 7 Nov) | Partial | Dataset citation and ontology; uses agreement as a ceiling, not to correct metrics. |
| 6 | **Tanno et al.**, Nature Medicine 31:599–608 (2024/25) | Clinically significant errors also in original human CXR reports (14.0 % only in the human report); AI reports often rated equal or better | Partial | Evidence that references are fallible → motivates the noise floor. |
| 7 | **Yu et al.**, Patterns 4:100802 (2023) — ReXVal | Radiologists count errors (incl. omission) in 200 report pairs; RadGraph-F1, RadCliQ | Partial | Standard meta-evaluation set-up to cite. |
| 8 | **Choudhury et al.**, Med Phys 37:1788 (2010) | Latent-class analysis of LIDC CT readers: reader-panel truth biased CAD sensitivity by −23 to −27 % vs −2 to −6 % with latent classes; reader sensitivity ≈ 55 % | Method precedent | The closest radiology use of our latent model; supports latent vs union-of-readers. |
| 9 | **ODIN 2026 Task 1 leaderboard** (odin2026.grand-challenge.org) and design doc (Zenodo 19727377) | Aggregate = 0.8·RadFact + 0.1·BLEU-4 + 0.1·METEOR (inferred arithmetically); blinded clinician **Arena**; **V2V2R (Singh, Lu, Sagl) 1st** (RadFact 0.392, aggregate 0.361, Arena 86.8 %); "When Is a Claim Worth Making?" 2nd (higher Arena, lower aggregate; paper UNVERIFIED); EviDent 3rd | Context | The **Arena vs aggregate disagreement** is a natural validity check: do corrected metrics agree better with clinicians' preferences? |
| 10 | **Mahmoud & LaBella**, ODIN 2026 satellite (papers.miccai.org/miccai-2026-sat/ODIN_challenges_012) | Nearest-neighbour report retrieval: BLEU-4 0.289 / METEOR 0.505 (10 validation cases), copies unsupported findings | Partial | Cheap retrieval baseline to include. |
| 11 | **Wuokko-Landén et al.**, PLOS ONE 2024 | Dental findings overlooked in 225 sinus CT reports (re-reading found 22 apical periodontitis, 16 severe bone loss) | Partial | Dental evidence of omission. |
| 12 | Others (context/baselines): VisualCheXbert (CHIL 2021; report labels ≠ image labels), ReXamine-Global (arXiv 2408.16208; site/style effects), CBCTRepD (arXiv 2603.10933; omission as an error class in a reader study), PanDent (arXiv 2607.27378, MedIA; template references, FDI-level metrics), RadFact/MAIRA-2, GREEN, RaTEScore, FineRadScore, CheXprompt, CRG Score (ignores extra findings instead of modelling them), ReXrank; dental LLMs (MMOral/OralGPT, DentalGPT, MMDental) as background. UNVERIFIED: "Fluency Is Not Faithfulness" (ODIN 2026). | | |

## 3. Statistical precedents (what we build on, what reviewers will expect)

- **Dawid & Skene (1979)**, Appl Stat 28:20–28 — latent rater error rates by EM without gold standard; our model is
  the binary, exchangeable-reader, φ = 0 special case → cite as the parent model.
- **Hui & Walter (1980)**, Biometrics 36:167–171 — two tests × two populations with different prevalence identify
  error rates; assumptions discussed by Toft et al. (2005). → **φ may be identifiable with 2 readers across centres
  with different prevalence**, assuming π and φ equal across centres (just-identified, no spare df; needs π > φ).
- **Chao (1987)**, Biometrics 43:783–791; Chao et al. (2001) Stat Med tutorial — heterogeneous detectability biases
  two-sample estimates; lower-bound estimators. → our π is **overestimated** under heterogeneity (π̂ ≈ E[π²]/E[π]),
  so unreported findings are undercounted (conservative).
- **Conditional dependence:** Vacek (1985; via citing papers), Brenner (1995; abstract), Dendukuri & Joseph (2001),
  **Albert & Dodd (2004)** — misspecified reader dependence biases latent-class estimates and is undetectable with
  few raters; partial gold-standard verification (our masks) is the remedy.
- **Imperfect reference standards (reviews):** Rutjes et al. (2007, HTA), Reitsma et al. (2009, J Clin Epidemiol),
  Umemneku Chikere et al. (2019, PLOS ONE), Joseph et al. (1995, Am J Epidemiol, Bayesian LCA); **Hadgu (1996,
  Lancet): discrepant analysis inflates accuracy** → never adjudicate only discordant cases.
- **Propensity-scored metrics for missing labels:** Jain, Prabhu & Varma (KDD 2016, PSP@k) and the critique by
  **Schultheis et al. (KDD 2022)**: heuristic propensity models are unreliable → our argument for estimating π from
  reader overlap. **PU evaluation:** Elkan & Noto (KDD 2008), Claesen et al. (arXiv 2015), Jain, White & Radivojac
  (AAAI 2017), Bekker & Davis (2020 survey; SAR setting 2019 ≈ our per-type/visibility conditioning).
- **Information retrieval with incomplete judgments:** Zobel (SIGIR 1998), Buckley & Voorhees (bpref, SIGIR 2004),
  Yilmaz & Aslam (infAP, CIKM 2006), Sakai (SIGIR 2007) — sampling-based estimation of metrics under incomplete
  judgments.
- **Capture–recapture of independent reviewers:** Eick et al. (ICSE 1992), Briand et al. (IEEE TSE 2000) — with few
  reviewers, underestimation is common.
- **Misclassification correction:** Rogan & Gladen (1978) — corrected rate = (apparent + Sp − 1)/(Se + Sp − 1);
  can leave [0, 1]; uncertainty in Se/Sp must be propagated.
- Not found: any capture–recapture estimate of radiologists' omission rates from double reading.

## 4. What is genuinely new in our project

1. **Per-finding reporting propensity estimated from independent double reads** (latent class / capture–recapture),
   not hand-set (EviDent) and not union-of-readers (PU-DPO).
2. **Visibility conditioning from the image** (masks, field of view) at tooth/side level — a reference's silence
   about something outside the scan is not an omission.
3. **Validation of the propensity model against segmentation masks** (partial gold standard).
4. **Propensity-corrected evaluation metrics** for report generation, with a **human-vs-human noise floor** computed
   with the same metrics (apparently absent from the literature) and **no-image / header-only controls**.
5. **Re-ranking real systems** (V2V2R and others) under raw vs corrected metrics, with the clinician Arena as an
   external check.
6. **Parser-error correction** of the propensity estimates (measured by blinded human review).
7. **Dental CBCT** as the first domain; chest X-ray (REFLACX, VinDr-CXR) as the second.

What is **not** new and must be credited: that references omit findings; that reference choice changes rankings;
latent-class reader models; propensity-scored / PU metrics; capture–recapture.

## 5. Lessons → proposed changes to our plan

| # | Change | Why (source) |
|---|---|---|
| L1 | **Second corrected-precision estimator (PU-style):** expected true positives among unmatched predictions ≈ model recall on mentioned slots × P(present \| unmentioned) × number of unmentioned slots; report both estimators | current estimator is exact only for a model without skill (Jain/White/Radivojac 2017; Claesen 2015) |
| L2 | **Estimate φ across centres (Hui–Walter)** as a complement to T3, with stated assumptions; use the few 3-report cases for goodness of fit | identifiability (Hui & Walter 1980) |
| L3 | **Heterogeneity and reader dependence in the simulation study** (beta-distributed detectability; correlated readers); frame π as an upper bound / unreported counts as a lower bound | Chao 1987/2001; Albert & Dodd 2004; Briand 2000 |
| L4 | **Parser correction = Rogan–Gladen at mention level per type**, review sample resampled inside the case bootstrap; check parser-error correlation within a case; validate in simulation | Rogan & Gladen 1978 |
| L5 | **Random, stratified adjudication** of unmatched predictions (not only disagreements), ideally a dentist with the image | Hadgu 1996; infAP sampling |
| L6 | **Compare latent π with union-of-readers truth** (PU-DPO's approach) on our data | Choudhury 2010; PU-DPO |
| L7 | **Figure: EviDent's hand-set weights vs our estimated π** per finding type | EviDent-CBCT |
| L8 | **Baselines:** no-image/centre prior (George), report retrieval (Mahmoud & LaBella), Lumetti's structured predictor (public code), V2V2R; EviDent/other outputs only with permission | dental thread |
| L9 | **Visibility must beat a header-only proxy** (or include it) | George et al. |
| L10 | **Allow π to vary by context** (centre as covariate; hierarchical model already planned) | PU-DPO |
| L11 | **Validity check against the clinician Arena ranking** (if obtainable) | ODIN leaderboard |
| L12 | Second domain: REFLACX (multi-read free text) + VinDr-CXR; stability under ReRef rewrites | report-generation thread |
| L13 | Use the Lumetti ontology/metric names, per-centre analysis (A = Sfax, unseen centre), FDI-level evaluation, acknowledge GPT-5.2 translation noise | dental thread |

## 6. Checks before citing

- Re-open 2026 preprints (EviDent-CBCT, PU-DPO, Jeong et al., George et al., CheXthought) for later versions.
- UNVERIFIED items: "When Is a Claim Worth Making?" paper, "Fluency Is Not Faithfulness", CBCT inter-observer κ
  primaries, Vacek 1985 and Brenner 1995 (only via citing papers / abstract), the PU-DPO NeurIPS page.
- The equality "Lincoln–Petersen count = Chao's M_h lower bound at t = 2" is the agent's own algebra — verify
  before using it as a framing.
