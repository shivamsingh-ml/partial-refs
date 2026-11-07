# Second domain — candidate datasets (HANDOFF M5)

Status: shortlist from web research, 2026-11-07. **No registration or download has been made.**
Properties marked ✔ were read on a primary source (paper or dataset page) listed under each entry;
properties marked ? still need checking on the dataset's own page before Shivam decides.

## What the method needs

1. ≥ 2 **independent** readers per study (k-of-n mention counts identify reporting propensity π;
   with ≥ 3 readers the false-positive rate φ is identifiable too).
2. Per-reader findings: free-text reports (parsed into findings) or structured labels.
3. Images (so models can be evaluated) and, ideally, a near-truth subset (consensus / adjudication)
   to validate corrected metrics — the role ToothFairy3 masks play in the dental domain.
4. A licence that allows research use; access we can obtain in weeks, not months.

## Shortlist

| | VinDr-CXR | REFLACX | CheXthought | LIDC-IDRI |
| --- | --- | --- | --- | --- |
| Modality | Chest X-ray | Chest X-ray (MIMIC-CXR images) | Chest X-ray | Chest CT |
| Readers per study | ✔ 3 radiologists, independent (15,000 training images) | ✔ 5 radiologists on 109 images (phases 1–2); 1 reader on ~2,500 (phase 3) | ? "multi-read", 501 radiologists, 50,312 images | ✔ 4 radiologists (blinded + unblinded phases) |
| Per-reader output | ✔ structured labels + boxes | ✔ free-text report transcription + labels with certainty + ellipses | ? reasoning traces + gaze; report/label content unclear | ✔ nodule marks + characteristics (XML) |
| Near-truth subset | ✔ test set (3,000) labelled by consensus of 5 | – | ? | – |
| φ identifiable (≥ 3 readers) | ✔ yes | ✔ yes (5 readers, but only 109 images) | ? | ✔ yes |
| Access | ? PhysioNet, credentialed DUA (GitHub says registration + DUA; PhysioNet page not yet seen) | ✔ PhysioNet credentialed DUA; MIMIC-CXR access (CITI training) also needed | ? not stated in abstract | ? TCIA, reportedly open; licence to check |
| Main concern | labels, not free text | small multi-reader subset | content and independence unknown | ? released marks may be from the **unblinded** phase (not independent) |

### Recommendation (for Shivam to decide)

- **Primary: VinDr-CXR.** Large (15,000 × 3 independent readers), φ identifiable, and the 5-reader
  consensus test set lets us check that corrected metrics track near-truth better than raw ones — the
  same validation logic as masks in the dental study. Findings are structured labels, which our
  pipeline handles directly (the method is defined on findings, not on text).
- **Secondary, same format as the dental study: REFLACX** phases 1–2 (109 CXRs × 5 free-text reports):
  a small report-level replication of propensity estimation from real free-text reports.
- **Check before deciding: CheXthought** (could be large-scale multi-read with text; needs the full
  paper's data section) and **LIDC-IDRI** (open access, but independence of released reads is doubtful).
- **Fallback (HANDOFF §12.9 B):** semi-synthetic omission study on a densely labelled dataset.

### Next steps

1. Read the data sections of VinDr-CXR (PhysioNet page) and CheXthought (full paper) and fill the
   ? cells. 2. Shivam chooses and applies for access (PhysioNet credentialing takes time — start
   early). 3. Write a `partialref.data` adapter; run propensity, corrected metrics, noise floor and a
   re-ranking of public models on that domain.

## Sources

- VinDr-CXR paper (Scientific Data): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9300612/ ;
  arXiv https://arxiv.org/pdf/2012.15029 ; code/access notes https://github.com/vinbigdata-medical/vindr-cxr
- REFLACX paper (Scientific Data): https://pmc.ncbi.nlm.nih.gov/articles/PMC9206650/ ;
  PhysioNet https://physionet.org/content/reflacx-xray-localization/
- CheXthought abstract: https://arxiv.org/abs/2604.26288
- LIDC-IDRI protocol (via LUNA16 paper): https://arxiv.org/pdf/1612.08012 ; https://pmc.ncbi.nlm.nih.gov/articles/PMC3041807
- PU-DPO (omission noise in CXR report generation; related work, not a dataset): https://arxiv.org/pdf/2608.05341
