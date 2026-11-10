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
- ``region_scope`` / ``remaining_tooth_ids`` (parser v3.2, 2026-11-10): a WHOLE arch or quadrant without teeth
  is stated by the model as region + "AllMissing" (+ the teeth named as the only ones remaining); the tooth
  list is then filled in HERE, by code (the model made errors when it had to list 14-16 teeth or a complement).
  "SomeMissing" (or no scope) keeps the region slot as before.
Negative statements ("no caries") produce no finding, as in the original ontology.

Each finding has exactly one type and is anchored to a *slot*: FDI tooth ids, a side, or a
quadrant, depending on the type. For scoring, a multi-tooth finding expands to one instance per
tooth (case, type, slot).
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    create_model,
    field_validator,
    model_validator,
)


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
RegionScope = Literal["AllMissing", "SomeMissing"]
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
    region_scope: RegionScope | None = None                     # ToothAbsence with a region only
    remaining_tooth_ids: list[FDIStr] = Field(default_factory=list)   # with AllMissing: teeth still present
    qualifiers: Qualifiers = Field(default_factory=Qualifiers)  # type: ignore[valid-type]

    @field_validator("tooth_ids", "remaining_tooth_ids")
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
        if self.region is None:
            self.region_scope, self.remaining_tooth_ids = None, []
        elif self.region_scope == "AllMissing":
            # whole region without teeth: every tooth of it except those named as remaining. Filled in by
            # code; region / scope / remaining stay as the model gave them (idempotent on re-validation).
            missing = set(REGION_TEETH[self.region]) - set(self.remaining_tooth_ids)
            if not missing:
                raise ValueError("ToothAbsence AllMissing with every tooth of the region remaining")
            self.tooth_ids = sorted(set(self.tooth_ids) | missing)
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


# Optional downstream convention (NOT applied at parsing; v3.1 review + mask check 2026-11-10): a tooth that
# the same report codes as RootResidue, or as ToothDevelopmentPathology of the eruption / development kind
# (impacted, included, orientation, incomplete root / open apex), is not ToothAbsence. Periodontal-only,
# deciduous and supernumerary TDP are NOT used (the mask mostly confirms those teeth as absent).
_TDP_PRESENT_QUALS = {"eruption_status", "bone_inclusion", "tooth_orientation"}
_TDP_PRESENT_DEV = {"IncompleteRootFormation", "OpenApex"}


def _implies_present(f: Finding) -> str | None:
    """Kind of present-implying finding, or None."""
    if f.finding_type == FindingType.RootResidue:
        return "RootResidue"
    if f.finding_type == FindingType.ToothDevelopmentPathology:
        q = f.qualifier_dict()
        if set(q) & _TDP_PRESENT_QUALS or set(q.get("development_feature", [])) & _TDP_PRESENT_DEV:
            return "TDP_eruption_development"
        if q.get("periodontal_feature"):
            return "TDP_periodontal"
        if q.get("development_feature"):
            return "TDP_deciduous_supernumerary"
        return "TDP_unspecified"
    return None


def absence_conflict_details(findings: list[Finding]) -> list[dict]:
    """Every tooth a report codes as ToothAbsence AND as a presence-implying finding, with that finding's kind."""
    absent = {t for f in findings if f.finding_type == FindingType.ToothAbsence for t in f.tooth_ids}
    out = []
    for f in findings:
        kind = f.finding_type.value if f.finding_type.value in TOOTH_PRESENT_TYPES else None
        if kind == "ToothDevelopmentPathology" or kind == "RootResidue":
            kind = _implies_present(f)
        if kind:
            out += [{"tooth": t, "with": kind} for t in sorted(absent & set(f.tooth_ids))]
    return out


def apply_absence_convention(findings: list[Finding]) -> tuple[list[Finding], int]:
    """Optional: remove ToothAbsence tooth ids that the same report codes as RootResidue or as eruption /
    development TDP. A ToothAbsence that had tooth ids and loses all of them is dropped (never falls back to its
    region). Returns (findings, number of tooth ids removed)."""
    keep_present = {t for f in findings if _implies_present(f) in ("RootResidue", "TDP_eruption_development")
                    for t in f.tooth_ids}
    out, removed = [], 0
    for f in findings:
        if f.finding_type == FindingType.ToothAbsence and keep_present & set(f.tooth_ids):
            teeth = [t for t in f.tooth_ids if t not in keep_present]
            removed += len(f.tooth_ids) - len(teeth)
            if not teeth:
                continue
            upd = {"tooth_ids": teeth}
            if f.region_scope == "AllMissing":      # else re-validation would add the removed teeth back
                upd["remaining_tooth_ids"] = sorted(set(f.remaining_tooth_ids)
                                                    | (keep_present & set(REGION_TEETH[f.region])))
            f = f.model_copy(update=upd)
        out.append(f)
    return out, removed


def absence_conflicts(findings: list[Finding]) -> list[str]:
    """Teeth that one report calls absent AND gives a finding implying the tooth is present."""
    absent = {t for f in findings if f.finding_type == FindingType.ToothAbsence for t in f.tooth_ids}
    present = {t for f in findings if f.finding_type.value in TOOTH_PRESENT_TYPES for t in f.tooth_ids}
    return sorted(absent & present)


def json_schema() -> dict:
    """JSON schema for guided decoding (vLLM structured output)."""
    return ReportExtraction.model_json_schema()
