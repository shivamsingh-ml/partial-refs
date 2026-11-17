"""Synthetic tests for the pretraining preparation (partialref.baseline.pretrain_datalist, dae_to_nnunet)."""

from __future__ import annotations

import json

import pytest

from partialref.baseline import pretrain_datalist as pdl


def _ref():
    import torch
    return {"network_weights": {"stem.conv.weight": torch.zeros(4, 1, 3, 3, 3), "stem.conv.bias": torch.zeros(4),
                                "decoder.seg_layers.0.weight": torch.zeros(1, 4, 1, 1, 1)}}


def test_convert_strips_prefix_and_checks():
    torch = pytest.importorskip("torch")                      # not in the CPU env; runs in the pretraining env
    from partialref.baseline import dae_to_nnunet as dn
    ck = {"model": {"encoder." + k: torch.randn(v.shape) for k, v in _ref()["network_weights"].items()}, "epoch": 3}
    w = dn.convert(ck, _ref())
    assert set(w) == {"network_weights"} and set(w["network_weights"]) == set(_ref()["network_weights"])
    bad = {"model": dict(ck["model"], **{"decoder_extra.w": torch.zeros(1)})}
    with pytest.raises(ValueError, match="without"):
        dn.convert(bad, _ref())
    short = {"model": {k: v for k, v in ck["model"].items() if "bias" not in k}}
    with pytest.raises(ValueError, match="missing"):
        dn.convert(short, _ref())
    wrong = {"model": dict(ck["model"], **{"encoder.stem.conv.bias": torch.zeros(5)})}
    with pytest.raises(ValueError, match="shapes"):
        dn.convert(wrong, _ref())
    nan = {"model": dict(ck["model"], **{"encoder.stem.conv.bias": torch.full((4,), float("nan"))})}
    with pytest.raises(ValueError, match="non-finite"):
        dn.convert(nan, _ref())


def test_datalist(tmp_path):
    tf3 = tmp_path / "tf3"
    (tf3 / "imagesTr").mkdir(parents=True)
    ids = [f"P{i:03d}" for i in range(30)] + ["F001", "S0015", "P100"]
    for c in ids:
        (tf3 / "imagesTr" / f"ToothFairy3{c[0]}_{c[1:]}_0000.nii.gz").write_bytes(b"x")
    rows = ["case_id,centre,role,fold"] + [f"{c},{c[0]},dev,0" for c in ids if c != "P100"] + ["P100,P,test,-1"]
    sp = tmp_path / "s"
    sp.mkdir()
    (sp / "splits.csv").write_text("\n".join(rows) + "\n")
    (sp / "splits.json").write_text(json.dumps({"tf3_only_for_segmentation": []}))
    d = pdl.build(sp, {"exclude": {"S0015": "x"}}, tf3)
    names = [r["case_id"] for r in d["training"] + d["validation"]]
    assert "P100" not in names and "S0015" not in names and len(names) == len(set(names)) == 31
    assert [r["case_id"] for r in d["validation"]] == ["P005", "P015", "P025"]     # every 10th per centre, offset 5
    assert all(r["image"].startswith("/") for r in d["training"])
