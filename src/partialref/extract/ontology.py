"""Finding vocabulary for per-report extraction (HANDOFF M1 step 4, GT/T1).

Re-implemented in our own code following the finding ontology of Lumetti et al. (MICCAI 2026,
"Ontology-Grounded Structured Prediction for Dental CBCT Reporting", repo AImageLab-zip/CBCT-Report;
cited, files not copied: the repository states no licence). Additions of ours:
- ``not_assessable``: statements that a structure is not visible / not evaluable in the scan
  (needed for visibility, HANDOFF §2.6); they are not findings.
- ``region`` on ToothAbsence (parser v3, 2026-11-09): missing teeth stated WITHOUT tooth numbers
  ("partial edentulism of the upper arch"). Its slot is "region:<Region>"; it counts at type level. In the
  propensity model (main analysis) the unnamed teeth of the region count as not mentioned (tooth-level
  completeness of the reference); secondary analysis: that reader is uninformative for them.
Negative statements ("no caries") produce no finding, as in the original ontology.

Each finding has exactly one type and is anchored to a *slot*: FDI tooth ids, a side, or a
quadrant, depending on the type. For scoring, a multi-tooth finding expands to one instance per
tooth (case, type, slot).
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Annotated, Literal

from pydantic import (BaseModel, ConfigDict, Field, StringConstraints, create_model, field_validator,
                      model_validator)


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


def _qual_field(name: str):
    """List of allowed values; single-valued qualifiers allow at most one entry. Encoded in the JSON
    schema (enum + maxItems) so that guided decoding cannot produce invalid qualifiers."""
    lit = Literal[tuple(QUALIFIERS[name])]  # type: ignore[valid-type]
    return (list[lit], Field(default_factory=list, max_length=None if name in MULTI_VALUED else 1))


Qualifiers = create_model(
    "Qualifiers", __config__=ConfigDict(extra="forbid"),
    **{name: _qual_field(name) for name in QUALIFIERS},
)

FDIStr = Annotated[str, StringConstraints(pattern=FDI_RE.pattern)]
Region = Literal["UpperArch", "LowerArch", "QuadrantI", "QuadrantII", "QuadrantIII", "QuadrantIV"]
REGION_PREFIX = "region:"
_PERM = {q: [f"{q}{i}" for i in range(1, 9)] for q in "1234"}
REGION_TEETH: dict[str, list[str]] = {
    "UpperArch": _PERM["1"] + _PERM["2"], "LowerArch": _PERM["3"] + _PERM["4"],
    "QuadrantI": _PERM["1"], "QuadrantII": _PERM["2"], "QuadrantIII": _PERM["3"], "QuadrantIV": _PERM["4"]}
# types that imply the tooth is present: a ToothAbsence on the same tooth in the same report is a conflict
TOOTH_PRESENT_TYPES = {"Caries", "EndodonticTreatment", "ConservativeRestoration", "ToothDevelopmentPathology",
                       "PeriapicalLesion", "RootResidue"}


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    finding_type: FindingType
    tooth_ids: list[FDIStr] = Field(default_factory=list)
    side: Side | None = None
    quadrant: Quadrant | None = None
    region: Region | None = None
    qualifiers: Qualifiers = Field(default_factory=Qualifiers)  # type: ignore[valid-type]

    @field_validator("tooth_ids")
    @classmethod
    def _fdi(cls, v: list[str]) -> list[str]:
        return sorted(set(map(str, v)))

    def qualifier_dict(self) -> dict[str, list[str]]:
        return {k: list(v) for k, v in self.qualifiers.model_dump().items() if v}

    @model_validator(mode="after")
    def _slot(self) -> Finding:
        kind = slot_kind(self.finding_type)
        if self.finding_type != FindingType.ToothAbsence and self.region is not None:
            self.region = None                 # region is only meaningful for ToothAbsence
        if kind == "tooth" and not self.tooth_ids and not (
                self.finding_type == FindingType.ToothAbsence and self.region is not None):
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
            return list(self.tooth_ids) if self.tooth_ids else [REGION_PREFIX + self.region]
        return [self.side] if kind == "side" else [self.quadrant]


class ReportExtraction(BaseModel):
    """Parser output for ONE report."""
    model_config = ConfigDict(extra="forbid")
    findings: list[Finding] = Field(default_factory=list)
    not_assessable: list[NotAssessableStructure] = Field(default_factory=list)


def instances(case_id: str, findings: list[Finding]) -> list[tuple[str, str, str]]:
    """(case, type, slot) instances, de-duplicated."""
    return sorted({(case_id, f.finding_type.value, s) for f in findings for s in f.slots()})


def absence_conflicts(findings: list[Finding]) -> list[str]:
    """Teeth that one report calls absent AND gives a finding implying the tooth is present."""
    absent = {t for f in findings if f.finding_type == FindingType.ToothAbsence for t in f.tooth_ids}
    present = {t for f in findings if f.finding_type.value in TOOTH_PRESENT_TYPES for t in f.tooth_ids}
    return sorted(absent & present)


def json_schema() -> dict:
    """JSON schema for guided decoding (vLLM structured output)."""
    return ReportExtraction.model_json_schema()
