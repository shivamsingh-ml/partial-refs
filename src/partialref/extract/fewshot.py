"""Prompt and few-shot examples for the per-report parser.

ALL example reports below are SYNTHETIC: written for this project in the general style of dental
CBCT reports, not taken or paraphrased from any ToothFairy report (DUA: real report text is never
committed). Their expected outputs follow partialref.extract.ontology.

Attribution: the finding types, anchors, qualifier values and several definitions used in the prompt
follow the finding ontology of Lumetti et al. (MICCAI 2026, "Ontology-Grounded Structured Prediction for
Dental CBCT Reporting"); the coding conventions (bridges, tooth-level periodontal defects, edentulous
arches) were measured from their released labels. Their repository states no licence: none of their
files (code, prompts, examples) were opened or copied; this prompt and its examples are our own.
"""

from __future__ import annotations

import json

from partialref.extract.ontology import QUALIFIERS

SYSTEM_PROMPT = f"""You convert ONE dental CBCT radiology report into structured findings.
Return only JSON that matches the given schema: {{"findings": [...], "not_assessable": [...]}}.

Finding types (choose exactly one per finding) and their anchor:
- ToothAbsence (tooth_ids): tooth missing (extracted, agenesis), including pontic positions of a bridge.
  A FULLY EDENTULOUS arch = ToothAbsence for all 16 teeth of that arch (11-18 and 21-28, or 31-38 and
  41-48). NOT for impacted/unerupted teeth (ToothDevelopmentPathology), NOT where a root residue
  remains (RootResidue), and NEVER for teeth or arches that are not visible / not included in the scan.
- ToothDevelopmentPathology (tooth_ids): (a) impacted, semi-included, unerupted, ectopic,
  supernumerary, retained primary teeth, incomplete root formation (eruption_status, bone_inclusion,
  tooth_orientation, development_feature); (b) TOOTH-SPECIFIC periodontal defects: vertical bone
  defect, bone loss / resorption around a named tooth, furcation involvement, pericoronal resorption
  (periodontal_feature, severity).
- Caries (tooth_ids). PeriapicalLesion (tooth_ids): lesion at the root apex. RootResidue (tooth_ids).
- EndodonticTreatment (tooth_ids): root canal filling; treatment_quality if stated.
- ConservativeRestoration (tooth_ids): filling / composite / amalgam restoration.
- ProstheticDevice (tooth_ids): crown, bridge, post-and-core, healing abutment, removable prosthesis,
  orthodontic appliance; ONE prosthesis_type per finding (crown + post-and-core = two findings).
  A bridge lists EVERY tooth from the first to the last abutment, pontics included; each pontic
  position is ALSO a ToothAbsence.
- DentalImplant (tooth_ids): an implant that IS PRESENT. Planned implants or sites "suitable for an
  implant" are NOT findings.
- PostExtractionSocket (tooth_ids): recent extraction socket.
- MandibularCanalCourse (side): course of the inferior alveolar canal; canal_trajectory,
  canal_tooth_contact; ALWAYS add tooth_ids of the teeth it relates to when they are named. Canal
  contact qualifiers belong ONLY here, never on other types.
- MandibularCondyleFinding (side): ONLY explicitly abnormal condyles (flattening, erosion, osteophyte,
  asymmetry, remodelling). Normal, regular, not visible or not mentioned condyles = NO finding.
- MaxillarySinusFinding (side): ONLY explicit sinus pathology (mucosal thickening, cyst,
  opacification, polyp). A normal / well-aerated sinus = NO finding.
- JawAtrophy (quadrant): generalised alveolar bone atrophy / resorption of a region (not tied to one
  tooth); severity, periodontal_feature; add tooth_ids if teeth are named.
- PathologicalLesion (quadrant): bone lesions not at a root apex: cysts, tumours, osteosclerosis,
  condensing osteitis, other radiolucent/radiopaque lesions. Periodontal or alveolar bone loss is NOT a
  PathologicalLesion (use JawAtrophy or ToothDevelopmentPathology).

Rules:
- Teeth use FDI two-digit numbers: permanent 11-48, primary 51-85. Expand ranges: "from 14 to 16"
  -> 14, 15, 16.
- One finding per type and anchor; several teeth with the same finding may share one finding.
- Negative or normal statements ("no caries", "regular condyles") produce NO finding.
- Statements that a structure is not visible / not assessable / outside the field of view go to
  "not_assessable" (maxilla, mandible, sinus_left, sinus_right, condyle_left, condyle_right,
  canal_left, canal_right, upper_teeth, lower_teeth, other), never to findings.
- Quadrants: QuadrantI upper right (1x), QuadrantII upper left (2x), QuadrantIII lower left (3x),
  QuadrantIV lower right (4x). Sides: Left / Right of the patient.
- When tooth numbers are named, DERIVE side / quadrant FROM THE NUMBERS (canal with 3x teeth = Left,
  4x = Right; quadrant from the first digit), not from words like "left"/"right".
- Use only these qualifier names and values: {json.dumps(QUALIFIERS, separators=(",", ":"))}
- Do not invent findings that the report does not state."""

FEWSHOT = [
    (
        "CBCT of both arches. Teeth 18, 28, 36 and 46 are missing. Tooth 38 is impacted, "
        "mesioangulated and fully included in bone; its roots are in close relationship with the "
        "left mandibular canal, which runs lingually. Root canal treatment on 26, reaching the "
        "apex, with a periapical radiolucency. Moderate vertical bone defect at 47. Composite "
        "restorations on 15 and 16. No caries. "
        "Condyles regular. The right maxillary sinus shows mild mucosal thickening.",
        {"findings": [
            {"finding_type": "ToothAbsence", "tooth_ids": ["18", "28", "36", "46"]},
            {"finding_type": "ToothDevelopmentPathology", "tooth_ids": ["38"],
             "qualifiers": {"eruption_status": ["Impacted"], "tooth_orientation": ["Mesioangulated"],
                            "bone_inclusion": ["FullInclusion"]}},
            {"finding_type": "MandibularCanalCourse", "side": "Left", "tooth_ids": ["38"],
             "qualifiers": {"canal_trajectory": ["PrevalentlyLingual"],
                            "canal_tooth_contact": ["CloseRelationship"]}},
            {"finding_type": "EndodonticTreatment", "tooth_ids": ["26"],
             "qualifiers": {"treatment_quality": ["ReachesApex"]}},
            {"finding_type": "PeriapicalLesion", "tooth_ids": ["26"],
             "qualifiers": {"radiodensity": ["Radiolucent"]}},
            {"finding_type": "ToothDevelopmentPathology", "tooth_ids": ["47"],
             "qualifiers": {"periodontal_feature": ["VerticalDefect"], "severity": ["Moderate"]}},
            {"finding_type": "ConservativeRestoration", "tooth_ids": ["15", "16"]},
            {"finding_type": "MaxillarySinusFinding", "side": "Right",
             "qualifiers": {"severity": ["Mild"]}}],
         "not_assessable": []},
    ),
    (
        "Examination limited to the mandible; the maxilla is not included in the field of view. "
        "Fixed bridge from 44 to 46 with 45 as pontic. Implant in position 36. Residual root of 47. "
        "Moderate horizontal bone resorption in the lower left quadrant. Both mandibular canals "
        "follow a regular course.",
        {"findings": [
            {"finding_type": "ProstheticDevice", "tooth_ids": ["44", "45", "46"],
             "qualifiers": {"prosthesis_type": ["FixedBridge"]}},
            {"finding_type": "ToothAbsence", "tooth_ids": ["45"]},
            {"finding_type": "DentalImplant", "tooth_ids": ["36"]},
            {"finding_type": "RootResidue", "tooth_ids": ["47"]},
            {"finding_type": "JawAtrophy", "quadrant": "QuadrantIII",
             "qualifiers": {"severity": ["Moderate"], "periodontal_feature": ["HorizontalBoneLoss"]}},
            {"finding_type": "MandibularCanalCourse", "side": "Left",
             "qualifiers": {"canal_trajectory": ["Regular"]}},
            {"finding_type": "MandibularCanalCourse", "side": "Right",
             "qualifiers": {"canal_trajectory": ["Regular"]}}],
         "not_assessable": ["maxilla", "sinus_left", "sinus_right"]},
    ),
    (
        "Crown with post-and-core on 21. Deep carious lesion on 37 approaching the pulp. Recent "
        "extraction socket at 47. Retained deciduous tooth 63 with agenesis of 13. The condyles "
        "cannot be evaluated because they lie outside the scanned volume.",
        {"findings": [
            {"finding_type": "ProstheticDevice", "tooth_ids": ["21"],
             "qualifiers": {"prosthesis_type": ["Crown"]}},
            {"finding_type": "ProstheticDevice", "tooth_ids": ["21"],
             "qualifiers": {"prosthesis_type": ["PostAndCore"]}},
            {"finding_type": "Caries", "tooth_ids": ["37"], "qualifiers": {"severity": ["Severe"]}},
            {"finding_type": "PostExtractionSocket", "tooth_ids": ["47"]},
            {"finding_type": "ToothAbsence", "tooth_ids": ["13"]},
            {"finding_type": "ToothDevelopmentPathology", "tooth_ids": ["63"],
             "qualifiers": {"development_feature": ["Deciduous"]}}],
         "not_assessable": ["condyle_left", "condyle_right"]},
    ),
    (
        "Edentulous lower arch with severe generalised atrophy of the posterior mandible. Upper arch: "
        "teeth 17 and 27 are missing; healing abutment on 24; the area of 26 is suitable for implant "
        "placement. Maxillary sinuses well aerated. Condyles symmetric and regular. The right "
        "mandibular canal runs superficially below the ridge in the region of 45 and 46.",
        {"findings": [
            {"finding_type": "ToothAbsence", "tooth_ids": ["31", "32", "33", "34", "35", "36", "37", "38",
                                                           "41", "42", "43", "44", "45", "46", "47", "48"]},
            {"finding_type": "JawAtrophy", "quadrant": "QuadrantIII", "qualifiers": {"severity": ["Severe"]}},
            {"finding_type": "JawAtrophy", "quadrant": "QuadrantIV", "qualifiers": {"severity": ["Severe"]}},
            {"finding_type": "ToothAbsence", "tooth_ids": ["17", "27"]},
            {"finding_type": "ProstheticDevice", "tooth_ids": ["24"],
             "qualifiers": {"prosthesis_type": ["HealingAbutment"]}},
            {"finding_type": "MandibularCanalCourse", "side": "Right", "tooth_ids": ["45", "46"],
             "qualifiers": {"canal_trajectory": ["Superficialized"]}}],
         "not_assessable": []},
    ),
]


def messages(report_text: str) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for text, out in FEWSHOT:
        msgs.append({"role": "user", "content": f"REPORT:\n{text}"})
        msgs.append({"role": "assistant", "content": json.dumps(out)})
    msgs.append({"role": "user", "content": f"REPORT:\n{report_text}"})
    return msgs
