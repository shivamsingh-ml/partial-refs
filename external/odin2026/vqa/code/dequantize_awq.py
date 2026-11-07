#!/usr/bin/env python3
"""
dequantize_awq.py -- unpack Qwen3.5-9B-AWQ's 4-bit MLPs into a bf16 checkpoint.

*** A PATH NOT TAKEN. This script has never been run to completion and no
*** dequantized checkpoint exists. It is kept for the measurement in it and
*** for the reasoning at the bottom, not because anything calls it.

It was vision_sft_plan.md §4.3 fallback 1, promoted to the main path on
2026-08-14 when the transfer gate fired (§4.3b), and dropped on 2026-08-15.
The idea was to make TRAINING see the weight values DEPLOYMENT will use.

FIRST, THE THING THAT IS EASY TO GET BACKWARDS. Dequantizing does not make a
better model, or a more precise one. `w = (q - z) * s` has at most 16 distinct
values per group whether it is stored in 4 bits or 16, so this writes the SAME
numbers into a fatter container: 12 GB in, ~19 GB out. Nothing is recovered.
The only thing it buys is that an ordinary nn.Linear full of already-rounded
values has an ordinary backward -- no gptqmodel, no custom kernel, no fp16.

WHY, IN ONE MEASUREMENT
───────────────────────
Arm 1 was merged into both checkpoints and served on the same 22 held-out cases:

    AWQ baseline, draw 1        213 claims
    AWQ baseline, draw 2        230 claims     <- 8% sampling noise floor
    AWQ  + arm 1                171
    bf16 + arm 1                543            <- 3.2x the AWQ number

Same adapter, same merged delta, same prompts. The only difference between the
two served checkpoints is the 93 frozen MLP modules -- 4-bit in one, bf16 in the
other. AWQ's scales are ACTIVATION-AWARE: they were calibrated on the
un-adapted activation distribution, so an adapter that shifts vision features
pushes activations off that calibration and the quantisation error grows. The
0.572-vs-0.552 base comparison never saw this because it had no adapter attached.

Training against the ORIGINAL bf16 therefore fits an adapter to weights that are
not the ones it will be served with. Dequantising AWQ gives a checkpoint whose
MLP values ARE the deployed ones -- the 4-bit rounding is already baked in --
while every layer stays an ordinary nn.Linear with an ordinary backward. No
gptqmodel, which cannot be used here at all (it patches Triton's autotuner at
import and kills fla's GatedDeltaNet kernels).

THE PACKING ORDER IS THE WHOLE RISK
───────────────────────────────────
AWQ packs eight 4-bit values per int32 along the OUTPUT dimension, and not in
natural order: the nibbles are interleaved [0,4,1,5,2,6,3,7]. Unpack them in
order 0..7 and you get a weight matrix whose columns are permuted within every
group of eight -- a checkpoint that loads, runs, and is silently wrong.

So the result is CHECKED, not asserted. models/Qwen3.5-9B is the same model
before quantisation, so a correct unpack lands within a few percent of it and a
wrong one is uncorrelated. --reference makes that a gate: per module, relative
error and cosine similarity against the original, and a refusal if either says
the unpack is wrong. Without a reference the script still runs but says loudly
that nothing verified it.

    python code/dequantize_awq.py \\
        --src models/Qwen3.5-9B-AWQ \\
        --out models/Qwen3.5-9B-AWQdq \\
        --reference models/Qwen3.5-9B

CPU only, ~19 GB out, a few minutes. Shards are streamed one at a time.

The unpack IS verified, on model.language_model.layers.1.mlp.down_proj against
models/Qwen3.5-9B, and those two numbers are the reusable part of this file:

    interleaved [0,4,1,5,2,6,3,7]   rel 0.1040   cos 1.000
    naive       [0,1,2,3,4,5,6,7]   rel 1.2638   cos 0.240

Note what that says about the tolerances: 4-bit group-128 error is a TENTH of
the weight norm, not the small number one would assume, so relative error is a
gross-breakage check and cosine is the real discriminator. A column-permuted
matrix keeps its magnitudes and sails through a loose rel bound.

WHY IT WAS DROPPED
──────────────────
Not because it was wrong. Three things together:

  * It answers only the FROZEN-MLP half of the transfer gap. The 93 quantized
    modules would hold deployed values; every other module -- and every module
    an arm actually trains -- is bf16 either way.
  * The design moved instead to arm 5 (`vision+language` in lora_arms.py),
    which puts adapters on all 214 modules AWQ leaves in bf16, including the
    24 Gated DeltaNet layers no arm had touched. More of the network is now
    adapted around the frozen MLPs rather than downstream of them.
  * The 19 GB shadow is a training-time artifact that has to be built, stored
    and kept in step with the AWQ checkpoint it shadows, for a benefit no
    measurement had yet separated from arm 5's own effect.

If the gap survives arm 5, this is the next thing to try, and the packing
order above is the part that would otherwise have to be re-derived.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

# AWQ's nibble interleave. Indexing an unpacked [.., 8] block with this puts the
# values back in natural output order. See the note above -- this list is the
# difference between a correct checkpoint and a plausible one.
AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]


def unpack_4bit(packed, torch):
    """[rows, cols] int32 -> [rows, cols*8] uint8, AWQ interleave undone."""
    shifts = torch.arange(0, 32, 4, dtype=torch.int32)
    out = torch.bitwise_right_shift(packed[:, :, None], shifts[None, None, :])
    out = torch.bitwise_and(out, 0xF).to(torch.uint8)          # [rows, cols, 8]
    out = out[:, :, AWQ_REVERSE_ORDER]
    return out.reshape(packed.shape[0], -1)


def dequant_module(qweight, qzeros, scales, torch):
    """AWQ triple -> nn.Linear-shaped [out_features, in_features] bf16."""
    iw = unpack_4bit(qweight, torch).to(torch.int32)            # [in, out]
    iz = unpack_4bit(qzeros, torch).to(torch.int32)             # [in/G, out]
    sc = scales.to(torch.float32)                               # [in/G, out]

    in_features = iw.shape[0]
    groups = sc.shape[0]
    if in_features % groups:
        raise ValueError(f"in_features {in_features} not divisible by {groups} groups")
    per = in_features // groups

    # Expand the per-group zero/scale down the input dimension.
    iz = iz.repeat_interleave(per, dim=0)
    sc = sc.repeat_interleave(per, dim=0)
    w = (iw - iz).to(torch.float32) * sc                        # [in, out]
    return w.t().contiguous().to(torch.bfloat16)                # [out, in]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reference", type=Path,
                    help="the pre-quantisation checkpoint. Turns the packing "
                         "order from an assumption into a checked fact.")
    ap.add_argument("--max-rel", type=float, default=0.30,
                    help="loose sanity bound on relative error against the "
                         "reference. Measured on down_proj: 0.104 for a correct "
                         "unpack, 1.264 for the naive 0..7 order -- so 4-bit "
                         "group-128 error is a tenth of the weight norm and NOT "
                         "the small number one might assume. Cosine is the real "
                         "gate; this only catches gross breakage.")
    ap.add_argument("--min-cos", type=float, default=0.98,
                    help="refuse below this cosine similarity. This is the "
                         "check that actually catches a permutation: a "
                         "column-shuffled matrix keeps its magnitudes and so "
                         "can pass a loose relative-error bound.")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    if args.out.exists() and not args.force:
        sys.exit(f"[FAIL] {args.out} exists (use --force)")

    shards = sorted(args.src.glob("*.safetensors"))
    if not shards:
        sys.exit(f"[FAIL] no safetensors in {args.src}")

    ref_index = {}
    if args.reference:
        for f in sorted(args.reference.glob("*.safetensors")):
            with safe_open(str(f), "pt") as h:
                for k in h.keys():
                    ref_index[k] = f
        print(f"[INFO] reference: {len(ref_index)} tensor(s) in {args.reference.name}")
    else:
        print("[WARN] no --reference: the packing order will NOT be verified. "
              "A wrong unpack produces a checkpoint that loads and is silently "
              "wrong -- see this file's header.")

    staging = args.out.with_suffix(".incomplete")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    n_deq = 0
    worst = []
    for shard in shards:
        tensors = {}
        with safe_open(str(shard), "pt") as h:
            keys = list(h.keys())
            qmods = sorted({k[:-len(".qweight")] for k in keys
                            if k.endswith(".qweight")})
            passthrough = [k for k in keys
                           if not any(k.startswith(m + ".") and
                                      k.rsplit(".", 1)[1] in
                                      ("qweight", "qzeros", "scales")
                                      for m in qmods)]
            for k in passthrough:
                tensors[k] = h.get_tensor(k)
            for m in qmods:
                w = dequant_module(h.get_tensor(m + ".qweight"),
                                   h.get_tensor(m + ".qzeros"),
                                   h.get_tensor(m + ".scales"), torch)
                tensors[m + ".weight"] = w
                n_deq += 1

                if ref_index.get(m + ".weight") is not None:
                    with safe_open(str(ref_index[m + ".weight"]), "pt") as rh:
                        r = rh.get_tensor(m + ".weight").to(torch.float32)
                    a = w.to(torch.float32)
                    rel = ((a - r).norm() / r.norm().clamp(min=1e-9)).item()
                    cos = torch.nn.functional.cosine_similarity(
                        a.flatten(), r.flatten(), dim=0).item()
                    worst.append((rel, cos, m))
                    if rel > args.max_rel or cos < args.min_cos:
                        shutil.rmtree(staging)
                        sys.exit(
                            f"[FAIL] {m}: relative error {rel:.3f} "
                            f"(max {args.max_rel}), cosine {cos:.4f} "
                            f"(min {args.min_cos}).\n"
                            f"       That is the signature of a wrong nibble "
                            f"order, not of 4-bit rounding. Nothing written.")
        save_file(tensors, str(staging / shard.name), metadata={"format": "pt"})
        print(f"[INFO] {shard.name}: {len(tensors)} tensor(s), "
              f"{len([1 for k in tensors if k.endswith('.weight')])} weight(s)")
        del tensors

    # ── config: the checkpoint is no longer quantized, and must not claim to be
    cfg = json.loads((args.src / "config.json").read_text(encoding="utf-8"))
    cfg.pop("quantization_config", None)
    if isinstance(cfg.get("text_config"), dict):
        cfg["text_config"].pop("quantization_config", None)
    (staging / "config.json").write_text(json.dumps(cfg, indent=2) + "\n",
                                         encoding="utf-8")

    for f in args.src.iterdir():
        if f.is_file() and f.name not in ("config.json",) and f.suffix != ".safetensors":
            shutil.copy2(f, staging / f.name)

    # The index names shards, and every shard was rewritten under its own name,
    # but the weight_map keys changed: qweight/qzeros/scales became .weight.
    idx = args.src / "model.safetensors.index.json"
    if idx.exists():
        blob = json.loads(idx.read_text(encoding="utf-8"))
        wm = {}
        for k, shard in (blob.get("weight_map") or {}).items():
            if k.endswith((".qweight", ".qzeros", ".scales")):
                wm[k.rsplit(".", 1)[0] + ".weight"] = shard
            else:
                wm[k] = shard
        blob["weight_map"] = wm
        (staging / idx.name).write_text(json.dumps(blob, indent=2) + "\n",
                                        encoding="utf-8")

    if worst:
        worst.sort(reverse=True)
        print(f"\n[INFO] verified {len(worst)} module(s) against the reference")
        print(f"       worst relative error {worst[0][0]:.4f} ({worst[0][2]})")
        print(f"       worst cosine        {min(w[1] for w in worst):.5f}")
        print(f"[PASS] the packing order is correct: a wrong nibble order gives "
              f"cosine near 0, not {min(w[1] for w in worst):.3f}")
    staging.rename(args.out)
    print(f"\n[PASS] {n_deq} module(s) dequantized -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
