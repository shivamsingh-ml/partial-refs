"""Corrections to ToothFairy3 manual labels from a fix table (`configs/seg/label_fixes_v1.json`, 2026-11-16).

Applied to a copy of the label array; the raw files are never changed. Each fix has a scope: "train" (V2V2R
segmentation training data, where all pulps are merged into one class, so pulp-only fixes are skipped) or
"module" (our per-tooth module, which reads each tooth's own pulp). Ops run in the listed order:

- relabel   {label, to}                    every voxel of `label` becomes `to`
- swap      {a, b}                          `a` and `b` exchange
- component_to {label, touching, to}        each 26-connected piece of `label` that touches `touching`
                                            (shares a face, edge or corner) becomes `to`

`expect_voxels` (int; for swap [n_a, n_b]) must equal the voxels the op changes, otherwise ValueError; this
stops a fix from applying to the wrong file or a changed release. `null` is only allowed while drafting
(`allow_unchecked=True`, used by `--dry-run`). A fix must never create a label that is already present
elsewhere (relabel/component_to onto an existing tooth label raises unless `merge: true`).

    python -m partialref.baseline.label_fixes --fixes configs/seg/label_fixes_v1.json --tf3-dir <TF3> \\
        --dry-run [--cases P061,S0038]       # prints voxel counts per op, one scan at a time
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

SCOPES = ("train", "module")
TEETH = set(range(11, 19)) | set(range(21, 29)) | set(range(31, 39)) | set(range(41, 49))
_CONN = np.ones((3, 3, 3), bool)


def load_fixes(path: Path) -> dict:
    d = json.loads(Path(path).read_text())
    for cid, ops in d["fixes"].items():
        if cid in d.get("exclude", {}):
            raise ValueError(f"{cid} is both excluded and fixed")
        for op in ops:
            need = {"relabel": ("label", "to"), "swap": ("a", "b"), "component_to": ("label", "touching", "to")}
            if op.get("op") not in need:
                raise ValueError(f"{cid}: unknown op {op.get('op')}")
            missing = [k for k in (*need[op["op"]], "scope", "why") if k not in op]
            if missing:
                raise ValueError(f"{cid}: {op['op']} lacks {missing}")
            if not set(op["scope"]) <= set(SCOPES) or not op["scope"]:
                raise ValueError(f"{cid}: bad scope {op['scope']}")
    return d


def _touching_pieces(L: np.ndarray, label: int, touching: int) -> np.ndarray:
    m = L == label
    cc, n = ndimage.label(m, structure=_CONN)
    if n == 0:
        return np.zeros_like(m)
    near = ndimage.binary_dilation(L == touching, structure=_CONN)
    hit = np.unique(cc[near & m])
    return np.isin(cc, hit[hit > 0])


def apply_op(L: np.ndarray, op: dict, allow_unchecked: bool = False) -> tuple[np.ndarray, int | list[int]]:
    """Return (new array, voxels changed). Raises on a count mismatch or an unintended duplicate."""
    L = L.copy()
    kind = op["op"]
    if kind == "swap":
        ma, mb = L == op["a"], L == op["b"]
        n = [int(ma.sum()), int(mb.sum())]
        L[ma], L[mb] = op["b"], op["a"]
    else:
        m = (L == op["label"]) if kind == "relabel" else _touching_pieces(L, op["label"], op["touching"])
        n = int(m.sum())
        to = op["to"]
        present = bool((L[~m] == to).any())
        if to != 0 and not op.get("merge") and to in TEETH | {100 + t for t in TEETH} and present:
            raise ValueError(f"{kind} {op.get('label')}->{to}: label {to} already present (set merge: true?)")
        if op.get("merge") and not present:
            raise ValueError(f"{kind} {op.get('label')}->{to}: merge set but label {to} is absent")
        L[m] = to
    exp = op.get("expect_voxels")
    if exp is None:
        if not allow_unchecked:
            raise ValueError(f"{kind}: expect_voxels missing")
    elif (list(exp) if isinstance(exp, list) else exp) != n:
        raise ValueError(f"{kind} {op}: changed {n} voxels, expected {exp}")
    if n == 0 or (isinstance(n, list) and 0 in n):
        raise ValueError(f"{kind} {op}: nothing to change (label absent)")
    return L, n


def apply_fixes(L: np.ndarray, case_id: str, table: dict, scope: str,
                allow_unchecked: bool = False) -> tuple[np.ndarray, list[dict]]:
    """Apply all ops of one case for `scope`; returns the new array and a log (op, voxels changed)."""
    if scope not in SCOPES:
        raise ValueError(scope)
    if case_id in table.get("exclude", {}):
        raise ValueError(f"{case_id} is excluded: {table['exclude'][case_id]}")
    log = []
    for op in table["fixes"].get(case_id, []):
        if scope not in op["scope"]:
            continue
        L, n = apply_op(L, op, allow_unchecked)
        log.append({"case_id": case_id, **{k: v for k, v in op.items() if k not in ("why", "scope")},
                    "changed": n})
    return L, log


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixes", required=True, type=Path)
    ap.add_argument("--tf3-dir", required=True, type=Path)
    ap.add_argument("--cases", help="comma-separated subset")
    ap.add_argument("--dry-run", action="store_true", help="print voxel counts per op (expect_voxels may be null)")
    a = ap.parse_args(argv)
    import nibabel as nib

    from partialref.audit.tf3_link import tf3_paths
    table = load_fixes(a.fixes)
    cases = [c.strip() for c in a.cases.split(",")] if a.cases else sorted(table["fixes"])
    for cid in cases:
        _, lab_p = tf3_paths(cid, a.tf3_dir)
        L = np.asarray(nib.load(str(lab_p)).dataobj).astype(np.int16)
        for scope in SCOPES:
            out, log = apply_fixes(L, cid, table, scope, allow_unchecked=a.dry_run)
            for e in log:
                print(json.dumps({"scope": scope, **e}))
            # left/right and up/down of the touched teeth are re-checked by seg_train_check --fixes
            del out
    return 0


if __name__ == "__main__":
    sys.exit(main())
