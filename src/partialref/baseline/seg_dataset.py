"""Build the nnU-Net training set for retraining V2V2R's segmentation network on our development split (2026-11-16).

Steps (one CPU job):
1. Staging folder in ToothFairy3's own layout (`imagesTr/`, `labelsTr/`, `dataset.json`) holding only the training
   scans: development cases of `configs/splits/v1` + the ToothFairy3-only scans, minus the fix table's exclusions
   (S0015). Images are symlinks to the TF3 originals. Labels are symlinks too, except for scans in the fix table:
   those get a corrected copy (scope "train", `partialref.baseline.label_fixes`), written with the original NIfTI
   header and data type.
2. V2V2R's conversion script (`U-Mamba2/scripts/convert_tf3_singlepulp.py`) is run UNCHANGED on the staging folder
   (it reads `imagesTr`/`labelsTr`/`dataset.json`, applies its own label mapping and its S_0048 fix, symlinks the
   images and writes nnU-Net's `dataset.json`). File names stay the TF3 names, so its S_0048 rule still applies.
3. Verification, every scan, independent of the script: case list = expected list (no test / centre A / excluded
   scan), image link resolves to the TF3 original, converted label == V2V2R mapping applied to the (fixed) TF3 label
   voxel by voxel, SimpleITK geometry of the converted label == the TF3 image's, fixed copies differ from the
   original only where the fix table says (voxel count per scan), 47 classes in dataset.json. Stops on any failure.

Outputs: `<raw_dir>/<dataset>/` (nnU-Net raw; image links point straight at the TF3 originals) and `<out>/` with
`cases.txt`, `verify.parquet`, `summary.json`, provenance. Staging and the unverified build are kept outside the raw
folder (`<raw_dir>/../staging/`, nnU-Net finds datasets by "DatasetNNN" prefix); the build moves into the raw folder
only when every check passed, so a failed build can never be picked up by training. An existing dataset of the
same name is never rebuilt over.

    python -m partialref.baseline.seg_dataset --tf3-dir <TF3> --splits configs/splits/v1 \\
        --fixes configs/seg/label_fixes_v1.json --umamba2-dir <U-Mamba2> \\
        --raw-dir data/nnunet/raw --dataset Dataset713_TF3SinglePulpDev --out results/seg/<run_id> [--np 4]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from partialref.audit.tf3_link import tf3_paths
from partialref.baseline.label_fixes import apply_fixes, load_fixes
from partialref.baseline.seg_train_check import S0048_QUIRK, candidates, load_mapping
from partialref.provenance import make_run_id, write_provenance

N_CLASSES = 47


def training_cases(splits: Path, fixes: dict) -> list[str]:
    cases = [c for c in candidates(splits) if c not in fixes.get("exclude", {})]
    missing = set(fixes["fixes"]) - set(cases)
    if missing:
        raise ValueError(f"fix table names scans outside the training list: {sorted(missing)}")
    return cases


def stage(cases: list[str], tf3_dir: Path, fixes: dict, staging: Path) -> dict[str, int]:
    """TF3-layout folder with only `cases`; fixed label copies for scans in the fix table. Returns voxels changed."""
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "imagesTr").mkdir(parents=True)
    (staging / "labelsTr").mkdir()
    shutil.copy2(tf3_dir / "dataset.json", staging / "dataset.json")
    changed = {}
    for cid in cases:
        img_p, lab_p = tf3_paths(cid, tf3_dir)
        if not (img_p.is_file() and lab_p.is_file()):
            raise FileNotFoundError(f"{cid}: TF3 image or label missing")
        os.symlink(img_p.resolve(), staging / "imagesTr" / img_p.name)
        if cid in fixes["fixes"]:
            lb = nib.load(str(lab_p))
            L0 = np.asarray(lb.dataobj)
            L, _ = apply_fixes(L0.astype(np.int16), cid, fixes, "train")
            dtype = lb.get_data_dtype()
            if L.min() < np.iinfo(dtype).min or L.max() > np.iinfo(dtype).max:
                raise ValueError(f"{cid}: fixed labels do not fit {dtype}")
            nib.save(nib.Nifti1Image(L.astype(dtype), lb.affine, lb.header), str(staging / "labelsTr" / lab_p.name))
            changed[cid] = int((L != L0).sum())
        else:
            os.symlink(lab_p.resolve(), staging / "labelsTr" / lab_p.name)
    return changed


def convert(umamba2_dir: Path, staging: Path, out_ds: Path, n_proc: int) -> None:
    """V2V2R's conversion script, unchanged."""
    if out_ds.exists():
        raise FileExistsError(f"{out_ds} exists; refusing to mix with an earlier build")
    script = umamba2_dir / "scripts" / "convert_tf3_singlepulp.py"
    subprocess.run([sys.executable, str(script), "--tf3_raw", str(staging), "--out", str(out_ds),
                    "--np", str(n_proc)], check=True)


def verify_case(cid: str, tf3_dir: str, out_ds: str, fixes: dict, mapping: dict[int, int],
                expect_changed: int) -> dict:
    import SimpleITK as sitk
    row: dict = {"case_id": cid}
    img_p, lab_p = tf3_paths(cid, Path(tf3_dir))
    o_img, o_lab = Path(out_ds) / "imagesTr" / img_p.name, Path(out_ds) / "labelsTr" / lab_p.name
    row["image_link_ok"] = o_img.is_symlink() and Path(os.readlink(o_img)) == img_p.resolve()   # direct link
    # expected = V2V2R mapping applied to the TF3 label after our train-scope fixes (and its S_0048 rule)
    L = sitk.GetArrayFromImage(sitk.ReadImage(str(lab_p))).astype(np.int16)        # SimpleITK order, as the script
    Lf, _ = apply_fixes(L, cid, fixes, "train") if cid in fixes["fixes"] else (L, [])
    row["fix_voxels"] = int((Lf != L).sum())
    row["fix_voxels_ok"] = row["fix_voxels"] == expect_changed
    # as convert_tf3_singlepulp.remap_one: labels in the mapping -> new id, all others -> 0, then S_0048's 60/160
    lut = np.zeros(max(int(Lf.max()), max(mapping)) + 1, np.uint8)
    for k, v in mapping.items():
        lut[k] = v
    exp = lut[Lf]
    if "ToothFairy3S_0048" in lab_p.name:
        for a, b in S0048_QUIRK.items():
            exp[Lf == a] = mapping.get(b, 0)
    out = sitk.ReadImage(str(o_lab))
    got = sitk.GetArrayFromImage(out)
    row["label_equal"] = got.shape == exp.shape and bool(np.array_equal(got, exp))
    ref = sitk.ImageFileReader()
    ref.SetFileName(str(img_p))
    ref.ReadImageInformation()
    row["geometry_ok"] = (out.GetSize() == ref.GetSize()
                          and np.allclose(out.GetSpacing(), ref.GetSpacing(), atol=1e-5)
                          and np.allclose(out.GetOrigin(), ref.GetOrigin(), atol=1e-4)
                          and np.allclose(out.GetDirection(), ref.GetDirection(), atol=1e-5))
    row["classes"] = sorted(int(v) for v in np.unique(got))
    row["ok"] = row["image_link_ok"] and row["fix_voxels_ok"] and row["label_equal"] and row["geometry_ok"]
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tf3-dir", required=True, type=Path)
    ap.add_argument("--splits", required=True, type=Path)
    ap.add_argument("--fixes", required=True, type=Path)
    ap.add_argument("--umamba2-dir", required=True, type=Path)
    ap.add_argument("--raw-dir", required=True, type=Path)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--np", type=int, default=4)
    a = ap.parse_args(argv)
    fixes = load_fixes(a.fixes)
    mapping = load_mapping(a.umamba2_dir)
    cases = training_cases(a.splits, fixes)
    out = a.out or Path("results/seg") / make_run_id("seg-dataset")
    out.mkdir(parents=True, exist_ok=True)
    # staging and the unverified build live OUTSIDE nnU-Net's raw folder (nnU-Net finds datasets by "DatasetNNN"
    # prefix and fails on two matches); the build is moved into the raw folder only after every check passed
    final = a.raw_dir / a.dataset
    if final.exists():
        raise FileExistsError(f"{final} exists; refusing to rebuild over a dataset that may be in use")
    work = a.raw_dir.parent / "staging"
    staging, build = work / a.dataset, work / f"{a.dataset}_build"
    for p in (staging, build):
        if p.exists():
            shutil.rmtree(p)                                       # only ever inside <raw-dir>/../staging
    changed = stage(cases, a.tf3_dir, fixes, staging)
    convert(a.umamba2_dir, staging, build, a.np)
    # the script links images to the staging links; point them straight at the TF3 originals instead (same files),
    # so the dataset does not depend on the staging folder
    for f in (build / "imagesTr").iterdir():
        target = f.resolve(strict=True)
        f.unlink()
        os.symlink(target, f)
    built = sorted(p.name.removesuffix(".nii.gz") for p in (build / "labelsTr").glob("*.nii.gz"))
    built_img = sorted(p.name.removesuffix("_0000.nii.gz") for p in (build / "imagesTr").iterdir())
    expected = sorted(tf3_paths(c, a.tf3_dir)[1].name.removesuffix(".nii.gz") for c in cases)
    dj = json.loads((build / "dataset.json").read_text())
    rows = []
    with ProcessPoolExecutor(a.np) as ex:
        futs = {c: ex.submit(verify_case, c, str(a.tf3_dir), str(build), fixes, mapping, changed.get(c, 0))
                for c in cases}
        for c, fu in futs.items():
            try:
                rows.append(fu.result())
            except Exception as e:                                           # noqa: BLE001
                rows.append({"case_id": c, "ok": False, "classes": [], "error": f"{type(e).__name__}: {e}"[:300]})
    df = pd.DataFrame(rows)
    df.to_parquet(out / "verify.parquet", index=False)
    (out / "cases.txt").write_text("\n".join(cases) + "\n")
    summary = {"n_cases": len(cases), "n_built": len(built), "case_list_equal": built == expected == built_img,
               "n_classes_dataset_json": len(set(dj["labels"].values())), "numTraining": dj["numTraining"],
               "excluded": sorted(fixes.get("exclude", {})), "fixed_scans": changed,
               "n_ok": int(df["ok"].sum()), "not_ok": df.loc[~df["ok"], "case_id"].tolist(),
               "max_class": int(max((max(r) for r in df["classes"] if len(r)), default=-1))}
    ok = (summary["case_list_equal"] and summary["n_ok"] == len(cases) and summary["numTraining"] == len(cases)
          and summary["n_classes_dataset_json"] == N_CLASSES and 0 <= summary["max_class"] < N_CLASSES)
    if ok:
        a.raw_dir.mkdir(parents=True, exist_ok=True)
        build.rename(final)
    summary.update(passed=bool(ok), raw_dataset=str(final) if ok else None,
                   unverified_build=None if ok else str(build))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    write_provenance(out, {"tf3_dir": str(a.tf3_dir), "splits": str(a.splits), "fixes": str(a.fixes),
                           "umamba2_dir": str(a.umamba2_dir), "raw_dataset": summary["raw_dataset"]},
                     packages=["numpy", "nibabel", "SimpleITK", "pandas"])
    print(json.dumps(summary, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
