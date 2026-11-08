"""Finding vocabulary for per-report extraction (HANDOFF M1 step 4, GT/T1).

Re-implemented in our own code following the finding ontology of Lumetti et al. (MICCAI 2026,
"Ontology-Grounded Structured Prediction for Dental CBCT Reporting", repo AImageLab-zip/CBCT-Report;
cited, files not copied: the repository states no licence). Additions of ours:
- ``not_assessable``: statements that a structure is not visible / not evaluable in the scan
  (needed for visibility, HANDOFF §2.6); they are not findings.
Negative statements ("no caries") produce no finding, as in the original ontology.

Each finding has exactly one type and is anchored to a *slot*: FDI tooth ids, a side, or a
quadrant, depending on the type. For scoring, a multi-tooth finding expands to one instance per
tooth (case, type, slot).
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class FindingType(str, Enum):
    MandibularCanalCourse = "MandibularCanalCourse"
    JawAtrophy = "JawAtrophy"
    PathologicalLesion = "PathologicalLesion"
    ToothAbsence = "ToothAbsence"
    ToothDevelopmentPathology = "ToothDevelopmentPathology"
    Caries = "Caries"
    PeriapicalLesion = "PeriapicalLesion"
    EndodonticTreatment = "EndodonticTreatment"
    ConservativeRestoration = "ConservativeRestoration"
    RootResidue = "RootResidue"
    PostExtractionSocket = "PostExtractionSocket"
    DentalImplant = "DentalImplant"
    ProstheticDevice = "ProstheticDevice"
    MandibularCondyleFinding = "MandibularCondyleFinding"
    MaxillarySinusFinding = "MaxillarySinusFinding"


SIDE_TYPES = {FindingType.MandibularCanalCourse, FindingType.MandibularCondyleFinding,
              FindingType.MaxillarySinusFinding}
QUADRANT_TYPES = {FindingType.JawAtrophy, FindingType.PathologicalLesion}
TOOTH_TYPES = set(FindingType) - SIDE_TYPES - QUADRANT_TYPES


def slot_kind(t: FindingType) -> str:
    return "side" if t in SIDE_TYPES else "quadrant" if t in QUADRANT_TYPES else "tooth"


QUALIFIERS: dict[str, list[str]] = {
    "severity": ["Mild", "Moderate", "Severe"],
    "tooth_orientation": ["Vertical", "Mesioangulated", "Distoangulated", "Horizontal", "Inverted",
                          "Linguoversed", "Vestibuloversed"],
    "eruption_status": ["Erupted", "PartiallyErupted", "Impacted"],
    "bone_inclusion": ["FullInclusion", "SemiInclusion"],
    "canal_trajectory": ["Regular", "PrevalentlyLingual", "PrevalentlyVestibular", "ApicalLingual",
                         "ApicalVestibular", "CentroLingual", "Central", "Superficialized"],
    "canal_tooth_contact": ["IntimateContact", "CloseRelationship", "BetweenRoots"],
    "treatment_quality": ["Adequate", "Inadequate", "ReachesApex", "DoesNotReachApex", "BeyondApex"],
    "lesion_kind": ["Granuloma", "RadicularCyst", "DentigerousCyst", "FollicularCyst",
                    "LateralPeriodontalCyst", "RetentionCyst", "ResiduaCyst", "InflammatoryCyst",
                    "Pericoronitis", "Cementoma"],
    "radiodensity": ["Radiolucent", "Radiopaque", "MixedDensity"],
    "margins": ["RegularMargins", "IrregularMargins"],
    "morphology": ["BifidCanal", "AnteriorLoop", "CorticalThinning", "CorticalExpansion",
                   "CorticalErosion", "Destructive", "CysticAppearing", "Osteosclerosis",
                   "CondensingOsteitis"],
    "prosthesis_type": ["Crown", "FixedBridge", "SuspendedPontic", "RemovableProsthesis",
                        "PostAndCore", "HealingAbutment", "OrthodonticAppliance"],
    "development_feature": ["Supernumerary", "Deciduous", "IncompleteRootFormation", "OpenApex"],
    "periodontal_feature": ["VerticalDefect", "HorizontalBoneLoss", "CircumferentialResorption",
                            "PericoronalResorption", "PeriradicularResorption",
                            "PeriimplantResorption", "FurcationInvolvement"],
}
MULTI_VALUED = {"tooth_orientation", "canal_trajectory", "morphology", "development_feature",
                "periodontal_feature"}

# Lumetti property names -> ours (used when reading their labels)
LUMETTI_PROPERTY = {
    "hasSeverity": "severity", "hasToothOrientation": "tooth_orientation",
    "hasEruptionStatus": "eruption_status", "hasBoneInclusion": "bone_inclusion",
    "hasCanalTrajectory": "canal_trajectory", "hasCanalToothContact": "canal_tooth_contact",
    "hasTreatmentQuality": "treatment_quality", "hasLesionKind": "lesion_kind",
    "hasRadiodensity": "radiodensity", "hasMargins": "margins", "hasMorphology": "morphology",
    "hasProsthesisType": "prosthesis_type", "hasDevelopmentFeature": "development_feature",
    "hasPeriodontalFeature": "periodontal_feature",
}

FDI_RE = re.compile(r"^([1-4][1-8]|[5-8][1-5])$")   # permanent 11-48, primary 51-85 (as in Lumetti labels)
Side = Literal["Left", "Right"]
Quadrant = Literal["QuadrantI", "QuadrantII", "QuadrantIII", "QuadrantIV"]
NotAssessableStructure = Literal[
    "maxilla", "mandible", "sinus_left", "sinus_right", "condyle_left", "condyle_right",
    "canal_left", "canal_right", "upper_teeth", "lower_teeth", "other"]


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    finding_type: FindingType
    tooth_ids: list[str] = Field(default_factory=list)
    side: Side | None = None
    quadrant: Quadrant | None = None
    qualifiers: dict[str, list[str]] = Field(default_factory=dict)

    @field_validator("tooth_ids")
    @classmethod
    def _fdi(cls, v: list[str]) -> list[str]:
        bad = [t for t in v if not FDI_RE.match(str(t))]
        if bad:
            raise ValueError(f"invalid FDI ids: {bad}")
        return sorted(set(map(str, v)))

    @field_validator("qualifiers")
    @classmethod
    def _quals(cls, v: dict[str, list[str]]) -> dict[str, list[str]]:
        out = {}
        for k, vals in v.items():
            if k not in QUALIFIERS:
                raise ValueError(f"unknown qualifier: {k}")
            vals = [vals] if isinstance(vals, str) else list(vals)
            bad = [x for x in vals if x not in QUALIFIERS[k]]
            if bad:
                raise ValueError(f"invalid values for {k}: {bad}")
            if len(vals) > 1 and k not in MULTI_VALUED:
                raise ValueError(f"{k} is single-valued")
            out[k] = vals
        return out

    @model_validator(mode="after")
    def _slot(self) -> Finding:
        kind = slot_kind(self.finding_type)
        if kind == "tooth" and not self.tooth_ids:
            raise ValueError(f"{self.finding_type.value} needs tooth_ids")
        if kind == "side" and self.side is None:
            raise ValueError(f"{self.finding_type.value} needs side")
        if kind == "quadrant" and self.quadrant is None:
            raise ValueError(f"{self.finding_type.value} needs quadrant")
        return self

    def slots(self) -> list[str]:
        """Scoring slots: one per tooth (tooth types), else the side / quadrant."""
        kind = slot_kind(self.finding_type)
        if kind == "tooth":
            return list(self.tooth_ids)
        return [self.side] if kind == "side" else [self.quadrant]


class ReportExtraction(BaseModel):
    """Parser output for ONE report."""
    model_config = ConfigDict(extra="forbid")
    findings: list[Finding] = Field(default_factory=list)
    not_assessable: list[NotAssessableStructure] = Field(default_factory=list)


def instances(case_id: str, findings: list[Finding]) -> list[tuple[str, str, str]]:
    """(case, type, slot) instances, de-duplicated."""
    return sorted({(case_id, f.finding_type.value, s) for f in findings for s in f.slots()})


def json_schema() -> dict:
    """JSON schema for guided decoding (vLLM structured output)."""
    return ReportExtraction.model_json_schema()
