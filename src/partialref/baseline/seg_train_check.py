"""Checks on the training data for retraining V2V2R's segmentation network on our development split (2026-11-12).

Before any GPU training: every candidate scan is checked on the real ToothFairy3 files, so the training set has
no test scans, no header problems, no left/right or up/down swaps and no unexpected labels. Also tests whether
per-tooth pulp can be recovered from the network's single merged pulp class (our per-tooth module uses each
tooth's own pulp; V2V2R's network merges all pulps into one class). Reviewed 2026-11-12 (agent, synthetic run).

Candidates: development cases of `configs/splits/v1/splits.csv` + the ToothFairy3-only scans of
`splits.json` (`tf3_only_for_segmentation`). Never test, never centre A. Exclusions are decided after this check
(DECISIONS: list frozen after the visual checks); this script only reports.

Per case (`per_case.parquet`, numbers only):
- files present; label on the image grid (shape + affine); qform vs sform of both files; SimpleITK's spacing /
  origin / direction of image vs label (nnU-Net and V2V2R's conversion read with SimpleITK); axis codes; image
  dtype, min, max.
- raw label values integral and within 0..32767 before casting; values outside ToothFairy3's label list; voxels
  V2V2R's conversion (`U-Mamba2/scripts/convert_tf3_singlepulp.py`, imported unchanged, incl. its S_0048 quirk
  60/160 -> 38/138) sets to background; converted labels within 0..46.
- label vs image: mean image value inside the tooth labels as stored and with the label array flipped on each
  axis; teeth are bright, so the label as stored must give the highest value (catches a label array flipped
  against its image under an identical header).
- centroids (voxel index, nibabel array axes 0/1/2) of every tooth, canal 3/4 and jaw 1/2 label.
- pulp test: pulps (111-148) merged into one class, each pulp voxel given the label of the nearest tooth voxel
  (Euclidean, voxel units); share of pulp voxels that get their own tooth back.

Summary (`summary.json`), from the centroids of all cases (differences in mm):
- left/right axis = axis with the largest median |centroid(Q1) - centroid(Q2)|; sign = convention. Per case and
  per check: other sign, a different dominant axis, or |difference| < 5 mm -> flagged; cases without the
  needed labels are listed as unchecked; cases with NO left/right check at all are listed.
- the left/right axis must be nibabel axis 0 = SimpleITK/nnU-Net axis 2: U-Mamba2's training augmentation
  (LRMirrorTransform) swaps left/right labels only when it mirrors that axis.
- canals 3/4 per case on the same side as the lower quadrants' convention (3 with Q3, 4 with Q4).
- single teeth on the wrong side of the arch midline (midpoint of the two quadrants of the same jaw) or of the
  mid-plane between the arches, if more than 3 mm off.
- up/down the same way from upper (Q1+Q2) vs lower (Q3+Q4) teeth, and jaw labels 2 vs 1.

    python -m partialref.baseline.seg_train_check --tf3-dir <TF3> --splits configs/splits/v1 \\
        --umamba2-dir <U-Mamba2 repo> [--out results/seg/<run_id>] [--cases P001,F006] [--workers 4]
"""

from __future__ import annotations

import argparse
import importlib.util
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

Q = {1: list(range(11, 19)), 2: list(range(21, 29)), 3: list(range(31, 39)), 4: list(range(41, 49))}
TEETH = [t for q in Q.values() for t in q]
UPPER, LOWER = Q[1] + Q[2], Q[3] + Q[4]
PULPS = [100 + t for t in TEETH]
OTHER = list(range(1, 11)) + [103, 104, 105]
TF3_LABELS = set(OTHER) | set(TEETH) | set(PULPS) | {0}
S0048_QUIRK = {60: 38, 160: 138}           # as V2V2R's conversion script
MIDLINE_MM = 3.0                           # a tooth this close to the midline / mid-plane is not given a side
MIN_SEP_MM = 5.0                           # a group difference below this is too small to call a side
LR_AXIS_EXPECTED = 0                       # nibabel axis 0 = SimpleITK / nnU-Net axis 2 (LRMirrorTransform)


def load_mapping(umamba2_dir: Path) -> dict[int, int]:
    """V2V2R's own label mapping, imported from its conversion script unchanged."""
    p = umamba2_dir / "scripts" / "convert_tf3_singlepulp.py"
    spec = importlib.util.spec_from_file_location("convert_tf3_singlepulp", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                                             # type: ignore[union-attr]
    return mod.mapping_DS119_singlepulp()


def candidates(splits_dir: Path) -> list[str]:
    s = pd.read_csv(splits_dir / "splits.csv")
    tf3_only = json.loads((splits_dir / "splits.json").read_text())["tf3_only_for_segmentation"]
    dev = s.loc[s["role"] == "dev", "case_id"].tolist()
    bad = set(tf3_only) & set(s.loc[s["role"] != "dev", "case_id"])
    if bad:
        raise ValueError(f"TF3-only scans listed as test/external: {sorted(bad)}")
    return sorted(set(dev) | set(tf3_only))


def pulp_reassign(L: np.ndarray) -> dict:
    """Merge all pulps, give each pulp voxel the nearest tooth label; how many get their own tooth back."""
    pulp = (L >= 111) & (L <= 148)
    n = int(pulp.sum())
    if n == 0:
        return {"pulp_vox": 0}
    tooth = np.isin(L, TEETH)
    if not tooth.any():
        return {"pulp_vox": n, "pulp_correct_share": 0.0, "pulp_vox_without_own_tooth": n}
    sl = ndimage.find_objects((pulp | tooth).astype(np.uint8))[0]
    Ls, ps, ts = L[sl], pulp[sl], tooth[sl]
    _, idx = ndimage.distance_transform_edt(~ts, return_indices=True)
    nearest = Ls[tuple(i[ps] for i in idx)]
    own = Ls[ps] - 100
    present = set(np.unique(Ls[ts]).tolist())
    has_tooth = np.isin(own, list(present))
    per = {int(t): round(float((nearest[own == t] == t).mean()), 4) for t in np.unique(own)}
    return {"pulp_vox": n, "pulp_correct_share": float((nearest == own).mean()),
            "pulp_correct_share_tooth_present": float((nearest[has_tooth] == own[has_tooth]).mean())
            if has_tooth.any() else np.nan,
            "pulp_vox_without_own_tooth": int((~has_tooth).sum()),
            "pulps_without_own_tooth": json.dumps(sorted(int(t) for t in set(own.tolist()) - present)),
            "pulp_correct_per_tooth": json.dumps(per),
            "n_pulps_below_0.9": sum(v < 0.9 for v in per.values())}


def centroids(L: np.ndarray, labels: list[int]) -> dict[str, list[float]]:
    """Voxel-index centroid per label, from each label's bounding box (no full-volume float arrays)."""
    out = {}
    objs = ndimage.find_objects(L, max_label=int(max(labels)))
    for k in labels:
        sl = objs[k - 1] if k - 1 < len(objs) else None
        if sl is None:
            continue
        idx = np.nonzero(L[sl] == k)
        out[str(k)] = [round(float(i.mean() + s.start), 2) for i, s in zip(idx, sl)]
    return out


def _sitk_geom(path: Path) -> dict | None:
    try:
        import SimpleITK as sitk
    except ImportError:
        return None
    r = sitk.ImageFileReader()
    r.SetFileName(str(path))
    r.ReadImageInformation()
    return {"size": list(r.GetSize()), "spacing": list(r.GetSpacing()), "origin": list(r.GetOrigin()),
            "direction": list(r.GetDirection())}


def _qs(img: nib.Nifti1Image) -> dict:
    q, qc = img.get_qform(coded=True)
    s, sc = img.get_sform(coded=True)
    return {"qcode": int(qc), "scode": int(sc),
            "q_eq_s": bool(q is not None and s is not None and np.allclose(q, s, atol=1e-4)) if qc and sc else None}


def check_case(case_id: str, tf3_dir: str, mapping: dict[int, int]) -> dict:
    row: dict = {"case_id": case_id, "image_present": False, "label_present": False}
    try:
        img_p, lab_p = tf3_paths(case_id, Path(tf3_dir))
        row.update(image_file=img_p.name, image_present=img_p.is_file(), label_present=lab_p.is_file())
        if not (img_p.is_file() and lab_p.is_file()):
            return row
        im, lb = nib.load(str(img_p)), nib.load(str(lab_p))
        qi, ql = _qs(im), _qs(lb)
        row.update(shape=list(im.shape[:3]), label_shape=list(lb.shape[:3]),
                   same_grid=im.shape[:3] == lb.shape[:3] and bool(np.allclose(im.affine, lb.affine, atol=1e-4)),
                   zooms=[round(float(z), 4) for z in im.header.get_zooms()[:3]],
                   axcodes="".join(nib.aff2axcodes(im.affine)), label_axcodes="".join(nib.aff2axcodes(lb.affine)),
                   image_qcode=qi["qcode"], image_scode=qi["scode"], image_q_eq_s=qi["q_eq_s"],
                   label_qcode=ql["qcode"], label_scode=ql["scode"], label_q_eq_s=ql["q_eq_s"],
                   image_dtype=str(im.get_data_dtype()), label_dtype=str(lb.get_data_dtype()))
        gi, gl = _sitk_geom(img_p), _sitk_geom(lab_p)
        if gi is not None:
            row["sitk_same_geometry"] = (gi["size"] == gl["size"] and all(
                np.allclose(gi[k], gl[k], atol=1e-4) for k in ("spacing", "origin", "direction")))
            row["sitk_direction"] = json.dumps([round(x, 4) for x in gi["direction"]])
            row["sitk_spacing"] = json.dumps([round(x, 4) for x in gi["spacing"]])
        Lraw = np.asarray(lb.dataobj)
        row["label_integral"] = bool(np.issubdtype(Lraw.dtype, np.integer) or np.all(Lraw == np.round(Lraw)))
        row["label_min"], row["label_max"] = float(Lraw.min()), float(Lraw.max())
        if not row["label_integral"] or row["label_min"] < 0 or row["label_max"] > 32767:
            row["error"] = "label values not integral or outside 0..32767"
            return row
        L = Lraw.astype(np.int16)
        del Lraw
        if "ToothFairy3S_0048" in img_p.name:                                # V2V2R's known quirk
            for a, b in S0048_QUIRK.items():
                L[L == a] = b
        vals, cnt = np.unique(L, return_counts=True)
        counts = {int(v): int(c) for v, c in zip(vals, cnt)}
        row["label_counts"] = json.dumps(counts)
        row["unknown_labels"] = json.dumps(sorted(v for v in counts if v not in TF3_LABELS))
        row["dropped_by_conversion"] = json.dumps({v: c for v, c in counts.items() if v != 0 and v not in mapping})
        lut = np.zeros(max(max(counts), max(mapping)) + 1, np.uint8)
        for k, v in mapping.items():
            lut[k] = v
        conv = set(np.unique(lut[L]).tolist())
        row["converted_labels_ok"] = all(0 <= v <= 46 for v in conv)
        row["n_converted_classes"] = len(conv)
        # label vs image: teeth are bright -> the label as stored must sit on the brightest place
        I = np.asarray(im.dataobj, dtype=np.float32)
        row["image_min"], row["image_max"] = float(I.min()), float(I.max())
        tooth = np.isin(L, TEETH)
        if tooth.any() and I.shape == tooth.shape:
            asis = float(I[tooth].mean())
            flips = {ax: float(I[np.flip(tooth, ax)].mean()) for ax in (0, 1, 2)}
            row["tooth_mean_asis"] = asis
            row["tooth_mean_flips"] = json.dumps({k: round(v, 1) for k, v in flips.items()})
            row["label_fits_image"] = bool(asis > max(flips.values()))
        del I, tooth
        row["centroids"] = json.dumps(centroids(L, TEETH + [1, 2, 3, 4]))
        row.update(pulp_reassign(L))
        row["error"] = None
    except Exception as e:                                                   # noqa: BLE001
        row["error"] = f"{type(e).__name__}: {e}"[:300]
    return row


def _mean(c: dict, labels) -> np.ndarray | None:
    pts = [c[str(t)] for t in labels if str(t) in c]
    return np.mean(pts, axis=0) if pts else None


def _bool(s: pd.Series) -> pd.Series:
    return s.astype("boolean").fillna(False).astype(bool)


def summarise(df: pd.DataFrame) -> dict:
    present = _bool(df["image_present"]) & _bool(df["label_present"])
    ok = df[df["error"].isna() & present]
    C = {r.case_id: json.loads(r.centroids) for r in ok.itertuples()}
    zoom = {r.case_id: np.array(r.zooms, float) for r in ok.itertuples()}

    def diffs(a, b) -> tuple[dict, list]:
        out, unchecked = {}, []
        for cid, c in C.items():
            x, y = _mean(c, a), _mean(c, b)
            if x is None or y is None:
                unchecked.append(cid)
            else:
                out[cid] = (x - y) * zoom[cid]                                # mm per array axis
        return out, sorted(unchecked)

    def check(a, b, name: str) -> dict:
        d, unchecked = diffs(a, b)
        res = {"name": name, "n": len(d), "unchecked": unchecked}
        if not d:
            return res
        D = np.array(list(d.values()))
        ax = int(np.argmax(np.median(np.abs(D), axis=0)))
        sign = int(np.sign(np.median(D[:, ax])))
        flagged = []
        for cid, v in d.items():
            why = []
            if sign == 0 or np.sign(v[ax]) != sign:
                why.append("other sign")
            if int(np.argmax(np.abs(v))) != ax:
                why.append(f"dominant axis {int(np.argmax(np.abs(v)))}")
            if abs(v[ax]) < MIN_SEP_MM:
                why.append(f"only {abs(v[ax]):.1f} mm apart")
            if why:
                flagged.append({"case_id": cid, "why": why, "mm": [round(float(x), 1) for x in v]})
        res.update(axis=ax, sign=sign, median_mm=round(float(np.median(D[:, ax])), 2), flagged=flagged)
        res["signs"] = {cid: int(np.sign(v[ax])) for cid, v in d.items()}
        return res

    lr_up, lr_lo = check(Q[1], Q[2], "upper left/right: Q1 - Q2"), check(Q[4], Q[3], "lower left/right: Q4 - Q3")
    canal = check([4], [3], "canals: 4 - 3 (4 sits with Q4)")
    ud, jaw = check(UPPER, LOWER, "up/down: upper - lower teeth"), check([2], [1], "jaws: upper (2) - lower (1)")
    lr_cases = set(lr_up.get("signs", {})) | set(lr_lo.get("signs", {}))
    no_lr_check = sorted(set(C) - lr_cases)
    # all left/right checks must use the same axis and sign, and it must be nnU-Net's mirror axis
    lr_axes = {x["name"]: (x.get("axis"), x.get("sign")) for x in (lr_up, lr_lo, canal)}
    lr_consistent = len({v for v in lr_axes.values() if v[0] is not None}) == 1
    canal_vs_teeth = []
    for cid, s in canal.get("signs", {}).items():
        ref = lr_lo.get("signs", {}).get(cid, lr_up.get("signs", {}).get(cid))
        if ref is not None and s != ref:
            canal_vs_teeth.append(cid)
    tooth_flags = []
    for cid, c in C.items():
        z = zoom[cid]
        if "axis" in lr_up:
            a, s = lr_up["axis"], lr_up["sign"]
            for left, right in ((Q[1], Q[2]), (Q[4], Q[3])):
                m1, m2 = _mean(c, left), _mean(c, right)
                if m1 is None or m2 is None:
                    continue
                mid = (m1[a] + m2[a]) / 2
                for t in left + right:
                    if str(t) in c:
                        off = (c[str(t)][a] - mid) * z[a] * s                 # > 0: on the Q1/Q4 side
                        side = 1 if t in left else -1
                        if abs(off) > MIDLINE_MM and np.sign(off) != side:
                            tooth_flags.append({"case_id": cid, "tooth": t, "check": "left/right",
                                                "mm_from_midline": round(float(off), 1)})
        if "axis" in ud:
            a, s = ud["axis"], ud["sign"]
            up, lo = _mean(c, UPPER), _mean(c, LOWER)
            if up is not None and lo is not None:
                mid = (up[a] + lo[a]) / 2
                for t in TEETH:
                    if str(t) in c:
                        off = (c[str(t)][a] - mid) * z[a] * s                 # > 0: on the upper side
                        side = 1 if t in UPPER else -1
                        if abs(off) > MIDLINE_MM and np.sign(off) != side:
                            tooth_flags.append({"case_id": cid, "tooth": t, "check": "up/down",
                                                "mm_from_mid_plane": round(float(off), 1)})
    for x in (lr_up, lr_lo, canal, ud, jaw):
        x.pop("signs", None)
    pc = ok[ok["pulp_vox"].fillna(0) > 0] if "pulp_vox" in ok else ok.iloc[:0]
    pulp_vox = int(pc["pulp_vox"].sum()) if len(pc) else 0

    def col(name):
        return ok[name] if name in ok else pd.Series(dtype=object)

    return {
        "n_candidates": len(df), "n_checked": len(ok),
        "errors": df.loc[df["error"].notna(), ["case_id", "error"]].to_dict("records"),
        "missing_files": df.loc[~present, "case_id"].tolist(),
        "label_not_on_image_grid": ok.loc[~_bool(ok["same_grid"]), "case_id"].tolist(),
        "sitk_geometry_differs": ok.loc[~_bool(col("sitk_same_geometry")), "case_id"].tolist()
        if "sitk_same_geometry" in ok else "SimpleITK not available",
        "qform_ne_sform": ok.loc[(col("image_q_eq_s") == False) | (col("label_q_eq_s") == False),
                                 "case_id"].tolist(),
        "axcodes": col("axcodes").value_counts().to_dict(),
        "label_axcodes_differ": ok.loc[ok["axcodes"] != ok["label_axcodes"], "case_id"].tolist(),
        "sitk_directions": col("sitk_direction").value_counts().to_dict(),
        "spacings": col("zooms").map(lambda z: "x".join(map(str, z))).value_counts().head(10).astype(int).to_dict(),
        "image_dtypes": col("image_dtype").value_counts().to_dict(),
        "label_dtypes": col("label_dtype").value_counts().to_dict(),
        "label_not_fitting_image": ok.loc[~_bool(col("label_fits_image")), "case_id"].tolist(),
        "unknown_labels": {r.case_id: json.loads(r.unknown_labels) for r in ok.itertuples()
                           if r.unknown_labels != "[]"},
        "dropped_by_conversion": {r.case_id: json.loads(r.dropped_by_conversion) for r in ok.itertuples()
                                  if r.dropped_by_conversion != "{}"},
        "converted_labels_not_ok": ok.loc[~_bool(ok["converted_labels_ok"]), "case_id"].tolist(),
        "left_right": [lr_up, lr_lo, canal], "left_right_same_axis_and_sign": lr_consistent,
        "left_right_axis_is_nnunet_mirror_axis": lr_up.get("axis") == LR_AXIS_EXPECTED,
        "cases_without_left_right_check": no_lr_check,
        "canal_side_differs_from_teeth": sorted(canal_vs_teeth),
        "up_down": [ud, jaw], "single_teeth_wrong_side": tooth_flags,
        "pulp": {"cases": len(pc), "pulp_voxels": pulp_vox,
                 "correct_share_all": float((pc["pulp_correct_share"] * pc["pulp_vox"]).sum() / pulp_vox)
                 if pulp_vox else None,
                 "case_median": float(pc["pulp_correct_share"].median()) if len(pc) else None,
                 "case_min": float(pc["pulp_correct_share"].min()) if len(pc) else None,
                 "cases_below_0.95": pc.loc[pc["pulp_correct_share"] < 0.95, "case_id"].tolist(),
                 "teeth_below_0.9": int(pc["n_pulps_below_0.9"].fillna(0).sum()),
                 "pulp_voxels_without_own_tooth": int(pc["pulp_vox_without_own_tooth"].fillna(0).sum())},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tf3-dir", required=True, type=Path)
    ap.add_argument("--splits", required=True, type=Path)
    ap.add_argument("--umamba2-dir", required=True, type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--cases", help="comma-separated subset (diagnostic run)")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    mapping = load_mapping(a.umamba2_dir)
    cases = candidates(a.splits)
    if a.cases:
        sub = [c.strip() for c in a.cases.split(",") if c.strip()]
        if set(sub) - set(cases):
            raise SystemExit(f"not candidates (test/external/unknown): {sorted(set(sub) - set(cases))}")
        cases = sub
    out = a.out or Path("results/seg") / make_run_id("seg-train-check")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {c: ex.submit(check_case, c, str(a.tf3_dir), mapping) for c in cases}
        for c, f in futs.items():
            try:
                rows.append(f.result())
            except Exception as e:                                           # noqa: BLE001 (e.g. worker killed)
                rows.append({"case_id": c, "image_present": None, "label_present": None,
                             "error": f"worker: {type(e).__name__}: {e}"[:300]})
    df = pd.DataFrame(rows)
    df.to_parquet(out / "per_case.parquet", index=False)
    s = summarise(df)
    (out / "summary.json").write_text(json.dumps(s, indent=2, default=str))
    (out / "candidates.txt").write_text("\n".join(cases) + "\n")
    write_provenance(out, {"tf3_dir": str(a.tf3_dir), "splits": str(a.splits), "umamba2_dir": str(a.umamba2_dir),
                           "n_cases": len(cases)}, packages=["numpy", "scipy", "pandas", "pyarrow", "nibabel",
                                                             "SimpleITK"])
    brief = {k: v for k, v in s.items() if k not in ("left_right", "up_down", "single_teeth_wrong_side", "pulp")}
    brief["left_right_flagged"] = {x["name"]: x.get("flagged") for x in s["left_right"]}
    brief["up_down_flagged"] = {x["name"]: x.get("flagged") for x in s["up_down"]}
    brief["unchecked"] = {x["name"]: len(x["unchecked"]) for x in s["left_right"] + s["up_down"]}
    brief["single_teeth_wrong_side"] = len(s["single_teeth_wrong_side"])
    brief["pulp"] = s["pulp"]
    print(json.dumps(brief, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
