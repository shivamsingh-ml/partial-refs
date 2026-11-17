"""Convert a U-Mamba2 DAE pretraining checkpoint into the file nnU-Net's `-pretrained_weights` loads (2026-11-17).

Their pretraining (`external/umamba2_pretrain_dae`, `utils.save_checkpoint`) saves `{"model": state_dict, ...}` of a
wrapper whose only submodule is the nnU-Net network, called `encoder` (`models/simmim_advanced.nnUNet`), so every key
is `encoder.<network key>`. nnU-Net (`nnunetv2/run/load_pretrained_weights.py`) expects `{"network_weights":
<network state_dict>}`. The released file `ssl_umamba2_3d_depth7_160x288x288.pth` has that layout (1,170 keys); the
conversion step that produced it is not in their repository, so this module does it: strip `encoder.`, keep every
key, store under `network_weights`.

Checks (any failure stops): every checkpoint key starts with `encoder.`; key names and tensor shapes equal the
reference file (their released weights) one to one; all tensors finite; optional: the result loads into the
segmentation network built from our plans with nnU-Net's own `load_pretrained_weights` (it asserts every
non-segmentation-layer key and shape), every loaded tensor outside the segmentation layers equals the file, and most
InstanceNorm weights have moved away from their initial value 1 (trained, not fresh).

    python -m partialref.baseline.dae_to_nnunet --ckpt <output>/ckpt_epoch_0199.pth \\
        --reference /msc/home/ssingh54/U-Mamba2/data/tf3_pretrained/ssl_umamba2_3d_depth7_160x288x288.pth \\
        --out data/pretrain/<run>/dae_nnunet_weights.pth [--plans <plans.json> --config <name> --dataset-json <json>]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PREFIX = "encoder."


def convert(ckpt: dict, reference: dict) -> dict:
    sd = ckpt["model"]
    bad = [k for k in sd if not k.startswith(PREFIX)]
    if bad:
        raise ValueError(f"{len(bad)} checkpoint keys without '{PREFIX}': {bad[:5]}")
    out = {k[len(PREFIX):]: v.detach().clone().cpu() for k, v in sd.items()}
    ref = reference["network_weights"]
    missing, extra = sorted(set(ref) - set(out)), sorted(set(out) - set(ref))
    if missing or extra:
        raise ValueError(f"key names differ from the reference: missing {missing[:5]}, extra {extra[:5]}")
    shapes = [k for k in ref if tuple(ref[k].shape) != tuple(out[k].shape)]
    if shapes:
        raise ValueError(f"shapes differ from the reference: {[(k, tuple(out[k].shape), tuple(ref[k].shape)) for k in shapes[:5]]}")
    nonfinite = [k for k, v in out.items() if v.is_floating_point() and not torch.isfinite(v).all()]
    if nonfinite:
        raise ValueError(f"non-finite values in {nonfinite[:5]}")
    return {"network_weights": out}


def check_loads(weights_path: Path, plans: Path, config: str, dataset_json: Path) -> None:
    """Load into the segmentation network of our plans with nnU-Net's own function (strict on non-seg layers)."""
    from nnunetv2.run.load_pretrained_weights import load_pretrained_weights
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    pm = PlansManager(str(plans))
    cm = pm.get_configuration(config)
    dj = json.loads(dataset_json.read_text())
    lm = pm.get_label_manager(dj)
    net = get_network_from_plans(cm.network_arch_class_name, cm.network_arch_init_kwargs,
                                 cm.network_arch_init_kwargs_req_import,
                                 determine_num_input_channels(pm, cm, dj), lm.num_segmentation_heads,
                                 allow_init=True, deep_supervision=True)
    load_pretrained_weights(net, str(weights_path), verbose=False)
    after = net.state_dict()
    file_w = torch.load(weights_path, map_location="cpu", weights_only=False)["network_weights"]
    keys = [k for k in after if ".seg_layers." not in k]
    not_equal = [k for k in keys if not torch.equal(after[k].cpu(), file_w[k].cpu())]
    if not_equal:
        raise AssertionError(f"{len(not_equal)} tensors differ from the file after loading: {not_equal[:5]}")
    # trained, not fresh: InstanceNorm affine weights start at 1 and biases at 0
    norm_w = [k for k in keys if ".norm.weight" in k]
    moved = sum(not torch.allclose(after[k], torch.ones_like(after[k])) for k in norm_w)
    print(json.dumps({"loaded_into_network": True, "tensors_checked_equal_to_file": len(keys),
                      "norm_weights": len(norm_w), "norm_weights_moved_from_1": moved}))
    if norm_w and moved < 0.5 * len(norm_w):
        raise AssertionError(f"only {moved} of {len(norm_w)} norm weights differ from 1: weights look untrained")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--reference", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--plans", type=Path)
    ap.add_argument("--config")
    ap.add_argument("--dataset-json", type=Path)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise FileExistsError(a.out)
    ckpt = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    ref = torch.load(a.reference, map_location="cpu", weights_only=False)
    w = convert(ckpt, ref)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(w, a.out)
    print(json.dumps({"ckpt": str(a.ckpt), "epoch": ckpt.get("epoch"), "n_keys": len(w["network_weights"]),
                      "out": str(a.out)}))
    if a.plans:
        check_loads(a.out, a.plans, a.config, a.dataset_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
