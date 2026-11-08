"""Image-only visibility features per case (HANDOFF §2.6, "measure first").

For P/F/S cases the hand-drawn ToothFairy3 labels are mapped onto the TF4 voxel grid (flip of array
axes 1 and 2, then the TF4 affine; verified voxel-identical for 527/527 cases, DECISIONS.md).
Nothing here reads reports, and no visibility threshold is applied: this module only measures.

Per case (one row):
- scan geometry; for each structure group: voxel count, volume (mm^3), number of connected components,
  and which scan faces it touches. Faces are named in world terms (sup/inf, left/right, ant/post)
  using the TF4 affine (RAS+ world), so "touches_sup" means "cut by the top of the scan".
- mandible split at its own centroid into left/right halves: does each half touch the top face
  (condyle cut) or a lateral / posterior face?
- per quadrant (1-4): jaw-label or tooth extent touching a lateral or posterior face.
Per tooth (long table): present, volume, root-direction face touched (upper teeth: sup, lower: inf)
-> root tip cut by the scan boundary.

Usage:
    python -m partialref.audit.visibility_features --manifest results/manifest/<run_id> --tf3-dir DIR [--limit N] [--workers 8]
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

from partialref.audit.tf3_link import tf3_paths
from partialref.provenance import make_run_id, write_provenance

TF3_TO_TF4_FLIP = (0, 1, 1)
STRUCTURES = {
    "mandible": [1], "maxilla": [2], "canal_left": [3], "canal_right": [4],
    "sinus_left": [5], "sinus_right": [6], "pharynx": [7], "bridge": [8], "crown": [9],
    "implant": [10], "incisive_canals": [103, 104], "lingual_canal": [105],
}
UPPER_TEETH = [q * 10 + i for q in (1, 2) for i in range(1, 9)]
LOWER_TEETH = [q * 10 + i for q in (3, 4) for i in range(1, 9)]
ALL_TEETH = UPPER_TEETH + LOWER_TEETH


def face_axes(affine: np.ndarray) -> dict[str, tuple[int, int]]:
    """Map world faces to (array axis, end) with end 0 = index 0, 1 = last index.

    For each world axis (x: right, y: anterior, z: superior in RAS+), find the array axis that
    dominates it and which array end points in the + world direction.
    """
    out = {}
    for w, (pos, neg) in enumerate([("right", "left"), ("ant", "post"), ("sup", "inf")]):
        col = affine[w, :3]
        ax = int(np.argmax(np.abs(col)))
        plus_end = 1 if col[ax] > 0 else 0
        out[pos] = (ax, plus_end)
        out[neg] = (ax, 1 - plus_end)
    return out


def touches(mask: np.ndarray, faces: dict[str, tuple[int, int]]) -> dict[str, bool]:
    res = {}
    for name, (ax, end) in faces.items():
        sl = [slice(None)] * 3
        sl[ax] = -1 if end else 0
        res[name] = bool(mask[tuple(sl)].any())
    return res


def _flip(x: np.ndarray, fl=TF3_TO_TF4_FLIP) -> np.ndarray:
    for ax, f in enumerate(fl):
        if f:
            x = np.flip(x, ax)
    return x


def case_features(case_id: str, tf4_path: str, tf3_dir: str,
                  pred_dir: str | None = None) -> tuple[dict, list[dict]]:
    """Hand-drawn TF3 labels (flipped onto the TF4 grid) or, with ``pred_dir``, predicted masks that
    are already on the TF4 grid in TF3 label ids (centre A; merged pulp = 199, ignored here)."""
    if pred_dir:
        lab_p, source, flip = Path(pred_dir) / f"{case_id}.nii.gz", "umamba2_foldall_pred", None
    else:
        lab_p, source, flip = tf3_paths(case_id, Path(tf3_dir))[1], "tf3_manual", TF3_TO_TF4_FLIP
    img = nib.load(tf4_path)
    row: dict = {"case_id": case_id, "label_source": source, "label_found": lab_p.is_file()}
    if not lab_p.is_file():
        return row, []
    L = np.asarray(nib.load(str(lab_p)).dataobj).astype(np.int16)
    if flip:
        L = _flip(L, flip)
    if L.shape != img.shape[:3]:
        row["shape_mismatch"] = True
        return row, []
    aff = img.affine
    vox_mm3 = float(np.prod(img.header.get_zooms()[:3]))
    faces = face_axes(aff)
    row["shape"] = "x".join(map(str, L.shape))

    for name, ids in STRUCTURES.items():
        m = np.isin(L, ids)
        n = int(m.sum())
        row[f"{name}_vox"] = n
        row[f"{name}_mm3"] = round(n * vox_mm3, 1)
        if n:
            row[f"{name}_ncomp"] = int(ndimage.label(m)[1])
            for k, v in touches(m, faces).items():
                row[f"{name}_touches_{k}"] = v

    # mandible halves at the mandible's own world-x centroid: condyle region = top of each half
    mand = L == 1
    if mand.any():
        idx = np.argwhere(mand)
        wx = (aff @ np.c_[idx, np.ones(len(idx))].T)[0]
        cx = float(np.median(wx))
        for side, sel in [("left", wx < cx), ("right", wx >= cx)]:
            half = np.zeros_like(mand)
            pts = idx[sel]
            half[tuple(pts.T)] = True
            t = touches(half, faces)
            row[f"mandible_{side}_touches_sup"] = t["sup"]
            row[f"mandible_{side}_touches_lateral"] = t[side]
            row[f"mandible_{side}_touches_post"] = t["post"]

    teeth_rows = []
    for t in ALL_TEETH:
        m = np.isin(L, [t, 100 + t])
        n = int(m.sum())
        r = {"case_id": case_id, "fdi": t, "present": n > 0, "vox": n, "mm3": round(n * vox_mm3, 1)}
        if n:
            tt = touches(m, faces)
            root_face = "sup" if t in UPPER_TEETH else "inf"
            r.update({"root_tip_cut": tt[root_face], "crown_cut": tt["inf" if root_face == "sup" else "sup"],
                      "touches_lateral": tt["left"] or tt["right"], "touches_post": tt["post"]})
        teeth_rows.append(r)

    # per quadrant: does anything of that quadrant (its teeth, or its jaw half) touch a lateral/post face?
    pres = {r["fdi"]: r for r in teeth_rows}
    for q, side, jaw in [(1, "right", 2), (2, "left", 2), (3, "left", 1), (4, "right", 1)]:
        qt = [pres[q * 10 + i] for i in range(1, 9) if pres[q * 10 + i]["present"]]
        row[f"q{q}_n_teeth"] = len(qt)
        row[f"q{q}_teeth_touch_lateral_or_post"] = any(r["touches_lateral"] or r["touches_post"] for r in qt)
        jaw_name = "maxilla" if jaw == 2 else "mandible"
        row[f"q{q}_jaw_label_present"] = row.get(f"{jaw_name}_vox", 0) > 0
    row["n_upper_teeth"] = sum(pres[t]["present"] for t in UPPER_TEETH)
    row["n_lower_teeth"] = sum(pres[t]["present"] for t in LOWER_TEETH)
    return row, teeth_rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--tf3-dir", type=Path, default=Path("/msc/home/ssingh54/sae_tooth/data/raw/toothfairy3"))
    ap.add_argument("--pred-mask-dir", type=Path, help="predicted masks on the TF4 grid (centre A)")
    ap.add_argument("--centres", default="PFS")
    ap.add_argument("--results-dir", default=Path("results"), type=Path)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--tag", default="visibility-features")
    a = ap.parse_args(argv)

    cases = pd.read_parquet(a.manifest / "cases.parquet")
    cases = cases[cases["centre"].isin(list(a.centres))].sort_values("case_id")
    if a.limit:
        cases = cases.head(a.limit)
    pred = str(a.pred_mask_dir) if a.pred_mask_dir else None
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        res = list(ex.map(case_features, cases["case_id"], cases["volume_path"],
                          [str(a.tf3_dir)] * len(cases), [pred] * len(cases)))
    df = pd.DataFrame([r for r, _ in res]).merge(cases[["case_id", "centre"]], on="case_id")
    teeth = pd.DataFrame([t for _, ts in res for t in ts])

    run_id = make_run_id(a.tag)
    out = a.results_dir / "audit" / run_id
    out.mkdir(parents=True, exist_ok=False)
    df.to_parquet(out / "case_features.parquet", index=False)
    teeth.to_parquet(out / "tooth_features.parquet", index=False)

    def share(col: str) -> dict:
        return df.groupby("centre")[col].mean().round(3).to_dict() if col in df else {}

    summary = {
        "run_id": run_id, "n_cases": int(len(df)),
        "jaw_label_present": {j: (df[f"{j}_vox"] > 0).groupby(df["centre"]).mean().round(3).to_dict()
                              for j in ["mandible", "maxilla"]},
        "mandible_touches_sup": share("mandible_touches_sup"),
        "sinus_left_mm3_median": df.groupby("centre")["sinus_left_mm3"].median().round(0).to_dict(),
        "n_upper_teeth_median": df.groupby("centre")["n_upper_teeth"].median().to_dict(),
        "n_lower_teeth_median": df.groupby("centre")["n_lower_teeth"].median().to_dict(),
        "teeth_present_root_tip_cut_share": (
            teeth[teeth["present"]].merge(df[["case_id", "centre"]], on="case_id")
            .groupby("centre")["root_tip_cut"].mean().round(3).to_dict() if len(teeth) else {}),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    write_provenance(out, extra={"run_id": run_id, "manifest_run": str(a.manifest), "tf3_dir": str(a.tf3_dir),
                                 "pred_mask_dir": pred, "centres": a.centres,
                                 "label_flip_tf3_to_tf4": None if pred else TF3_TO_TF4_FLIP})
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
