"""Test-set evaluation of the retrained V2V2R segmentation network (2026-11-18).

The network is run exactly as V2V2R's pipeline runs it: `run_segmentation` of the submitted code
(external/odin2026/segmentation.py: its own predictor with test-time left/right mirroring and label swapping, its
L-first flip, its label remap), on the ToothFairy4 volume the pipeline receives, with MODEL_ROOT pointing at the
model folder to evaluate. Nothing of that code is changed or re-implemented here.

Reference = the scan's ToothFairy3 manual mask converted to V2V2R's output grid and label scheme by
`partialref.baseline.v2v2r_masks` (mirror=True, the variant that matches V2V2R's output: Dice 0.98 vs its own masks,
DECISIONS 2026-11-13; image identity TF3 (flipped y,z) == TF4 re-checked per scan). Test scans carry no label fixes
(the fix table covers training scans only).

Score: Dice per label per scan (labels present in the reference or the prediction), plus per tooth: missed (in the
reference, Dice 0) and extra (predicted, absent from the reference). Summary per label group and per centre.

Cheap check first: the same two steps on 3 development scans with V2V2R's released weights must give tooth Dice
near the 0.98 found before (that network saw these scans); this tests orientation and scoring end to end.

    python -m partialref.baseline.seg_eval predict --cases test|P001,F001 --splits configs/splits/v1/splits.csv \\
        --tf4-dir <TF4> --model-root <dir with nnunet_seg/> --pred-dir <dir>
    python -m partialref.baseline.seg_eval score --cases test|P001,F001 --splits configs/splits/v1/splits.csv \\
        --tf3-dir <TF3> --tf4-dir <TF4> --pred-dir <dir> --out results/seg/<run_id>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from partialref.audit.tf3_link import tf3_paths
from partialref.baseline.v2v2r_masks import CANALS, PULP, convert_labels, to_tf4_grid
from partialref.perception.tooth_features import TEETH
from partialref.provenance import write_provenance

ODIN = Path(__file__).resolve().parents[3] / "external" / "odin2026"
GROUPS = {"teeth": list(TEETH), "pulp": [PULP], "jawbones": [1, 2], "alveolar_canals": [3, 4], "sinuses": [5, 6],
          "pharynx": [7], "bridge_crown_implant": [8, 9, 10], "incisive_lingual_canals": sorted(CANALS.values())}


def resolve_cases(spec: str, splits_csv: Path) -> list[str]:
    s = pd.read_csv(splits_csv)
    if spec == "test":
        return sorted(s.loc[s["role"] == "test", "case_id"])
    cases = [c.strip() for c in spec.split(",") if c.strip()]
    unknown = set(cases) - set(s["case_id"])
    if unknown:
        raise ValueError(f"cases not in the split: {sorted(unknown)}")
    return cases


def tf4_volume(case_id: str, tf4_dir: Path) -> Path:
    return tf4_dir / case_id / "cbct" / "volume.nii.gz"


def predict(cases: list[str], tf4_dir: Path, model_root: Path, pred_dir: Path) -> list[dict]:
    """V2V2R's run_segmentation on each TF4 volume; masks written in the volume's own space. Skips existing masks."""
    if not (model_root / "nnunet_seg").is_dir():
        raise FileNotFoundError(f"{model_root}/nnunet_seg missing")
    os.environ["MODEL_ROOT"] = str(model_root)                 # read by the submitted config.py at import
    os.environ.setdefault("NNUNET_SUBDIR", "nnunet_seg")
    sys.path.insert(0, str(ODIN))
    import SimpleITK as sitk
    from segmentation import run_segmentation                  # the submitted code, unchanged

    import config                                              # the submitted config, as segmentation imported it
    print(json.dumps({"NNUNET_MODEL_FOLDER": str(config.NNUNET_MODEL_FOLDER), "NNUNET_CHECKPOINT":
                      config.NNUNET_CHECKPOINT, "USE_MIRRORING": bool(config.USE_MIRRORING)}), flush=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for cid in cases:
        out = pred_dir / f"{cid}.nii.gz"
        if out.is_file():
            rows.append({"case_id": cid, "seconds": None, "skipped": True, "axcodes": None, "error": None})
            continue
        t0 = time.time()
        row = {"case_id": cid, "skipped": False, "error": None}
        try:
            img = sitk.ReadImage(str(tf4_volume(cid, tf4_dir)))
            row["axcodes"] = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(img.GetDirection())
            # the reference's left/right convention (mirror=True) was established on L-first volumes, which
            # run_segmentation flips; an R-first volume would need its own check -> not scored silently
            if not row["axcodes"].upper().startswith("L"):
                raise ValueError(f"volume stored {row['axcodes']} (not L-first)")
            mask = run_segmentation(img)
            tmp = pred_dir / f".{cid}.tmp.nii.gz"
            sitk.WriteImage(mask, str(tmp), True)
            tmp.rename(out)                                    # a killed job never leaves a half-written mask
        except Exception as e:                                               # noqa: BLE001
            row["error"] = f"{type(e).__name__}: {e}"[:300]    # no mask -> score reports the scan as failed
        row["seconds"] = round(time.time() - t0, 1)
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


def reference(case_id: str, tf3_dir: Path, tf4_dir: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    """TF3 manual mask in V2V2R's output scheme and the TF4 grid (nibabel array order), plus the TF4 image."""
    img_p, lab_p = tf3_paths(case_id, tf3_dir)
    tf4 = nib.load(str(tf4_volume(case_id, tf4_dir)))
    I3, I4 = np.asarray(nib.load(str(img_p)).dataobj), np.asarray(tf4.dataobj)
    if I3.shape != I4.shape or not np.array_equal(to_tf4_grid(I3), I4):
        raise ValueError("TF3 image (flipped y,z) is not voxel-identical to the TF4 volume")
    L = np.asarray(nib.load(str(lab_p)).dataobj).astype(np.int16)
    return to_tf4_grid(convert_labels(L, mirror=True)), tf4


def score_case(case_id: str, tf3_dir: str, tf4_dir: str, pred_dir: str) -> list[dict]:
    try:
        ref, tf4 = reference(case_id, Path(tf3_dir), Path(tf4_dir))
        pimg = nib.load(str(Path(pred_dir) / f"{case_id}.nii.gz"))
        if pimg.shape[:3] != ref.shape or not np.allclose(pimg.affine, tf4.affine, atol=1e-3):
            raise ValueError(f"prediction grid {pimg.shape} differs from the TF4 volume {ref.shape}")
        pred = np.asarray(pimg.dataobj).astype(np.int16)
        rv, pv = np.bincount(ref.ravel(), minlength=256), np.bincount(pred.ravel(), minlength=256)
        rows = []
        for k in sorted((set(np.flatnonzero(rv)) | set(np.flatnonzero(pv))) - {0}):
            inter = int(((ref == k) & (pred == k)).sum())
            rows.append({"case_id": case_id, "label": int(k), "ref_vox": int(rv[k]), "pred_vox": int(pv[k]),
                         "dice": 2 * inter / int(rv[k] + pv[k]), "error": None})
        return rows
    except Exception as e:                                                   # noqa: BLE001
        return [{"case_id": case_id, "label": -1, "ref_vox": 0, "pred_vox": 0, "dice": np.nan,
                 "error": f"{type(e).__name__}: {e}"[:300]}]


def summarise(df: pd.DataFrame, centre: dict[str, str]) -> dict:
    ok = df[df["error"].isna()]
    inref = ok[ok["ref_vox"] > 0]
    out: dict = {"n_cases": int(df["case_id"].nunique()), "n_failed": int(df.loc[df["error"].notna(), "case_id"].nunique()),
                 "failed": sorted(df.loc[df["error"].notna(), "case_id"].unique().tolist())}
    for g, labels in GROUPS.items():
        r = inref[inref["label"].isin(labels)]
        per_case = r.groupby("case_id")["dice"].mean()                       # scan Dice = mean over its labels
        out[g] = {"n_cases": int(per_case.size), "mean_dice": round(float(per_case.mean()), 4) if per_case.size else None,
                  "median_dice": round(float(per_case.median()), 4) if per_case.size else None,
                  "by_centre": {c: round(float(v), 4) for c, v in
                                per_case.groupby(per_case.index.map(centre)).mean().items()}}
    t = ok[ok["label"].isin(TEETH)]
    out["teeth_counts"] = {"in_reference": int((t["ref_vox"] > 0).sum()),
                           "missed": int(((t["ref_vox"] > 0) & (t["dice"] == 0)).sum()),
                           "extra": int(((t["ref_vox"] == 0) & (t["pred_vox"] > 0)).sum()),
                           "below_0.5": int(((t["ref_vox"] > 0) & (t["dice"] < 0.5)).sum())}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["predict", "score"])
    ap.add_argument("--cases", required=True, help="'test' or comma-separated case ids")
    ap.add_argument("--splits", required=True, type=Path)
    ap.add_argument("--tf3-dir", type=Path)
    ap.add_argument("--tf4-dir", required=True, type=Path)
    ap.add_argument("--model-root", type=Path)
    ap.add_argument("--pred-dir", required=True, type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--model", default="", help="score: model folder and checkpoint sha256, kept in provenance")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    cases = resolve_cases(a.cases, a.splits)
    if a.mode == "predict":
        rows = predict(cases, a.tf4_dir, a.model_root, a.pred_dir)
        pd.DataFrame(rows).to_csv(a.pred_dir / "predict_times.csv", mode="a", index=False,
                                  header=not (a.pred_dir / "predict_times.csv").exists())
        return 0
    with ProcessPoolExecutor(a.workers) as ex:
        res = ex.map(score_case, cases, [str(a.tf3_dir)] * len(cases), [str(a.tf4_dir)] * len(cases),
                     [str(a.pred_dir)] * len(cases))
        df = pd.DataFrame([r for rows in res for r in rows])
    a.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out / "dice_per_label.csv", index=False)
    s = pd.read_csv(a.splits)
    summary = summarise(df, dict(zip(s["case_id"], s["centre"])))
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2))
    write_provenance(a.out, {"cases": a.cases, "n_cases": len(cases), "pred_dir": str(a.pred_dir), "model": a.model,
                             "tf3_dir": str(a.tf3_dir), "tf4_dir": str(a.tf4_dir)},
                     packages=["numpy", "nibabel", "pandas"])
    print(json.dumps(summary, indent=1))
    return 1 if summary["n_failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
