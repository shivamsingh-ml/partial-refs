"""DINO embeddings of tooth slices (PAPER_PLAN step 2; 2026-11-12). Runs in the `pr_train` environment
(torch + transformers; no pandas there), so this file imports nothing from partialref.

Input: slices.npz from partialref.perception.tooth_slices. Each grey slice is repeated to 3 channels, resized
to 224 x 224, normalised with ImageNet mean/std (as the DINOv2 and DINOv3 processors do; DINOv2 uses 14 px
patches -> 16 x 16 tokens, DINOv3 16 px patches -> 14 x 14 tokens plus 4 register tokens) and passed through a local
DINO model (Hugging Face format, e.g. models/dinov2-base). Per slice: the CLS token and the mean of the patch
tokens. Output embeddings.npz: emb (n, 3 slices x 2 x d) float32, case_id, fdi_report, truncated; plus
info.json with model path, git SHA, device and timing.

    python src/partialref/perception/dino_embed.py --slices results/perception/<slices run> \\
        --model models/dinov2-base --out results/perception/<new run name> [--batch 64] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def preprocess(batch_u8: np.ndarray, torch):
    """(b, H, W) uint8 -> (b, 3, 224, 224) float tensor, ImageNet-normalised."""
    x = torch.from_numpy(batch_u8.astype(np.float32) / 255.0)[:, None]
    x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    x = x.repeat(1, 3, 1, 1)
    m, s = torch.from_numpy(MEAN)[None, :, None, None], torch.from_numpy(STD)[None, :, None, None]
    return (x - m) / s


def embed(slices: np.ndarray, model, torch, device: str, batch: int) -> np.ndarray:
    """slices (n, k, H, W) uint8 -> (n, k * 2 * d): per slice [CLS, mean patch token]. The token layout is
    [CLS, register tokens, patch tokens]: DINOv2 has 0 registers, DINOv3 has 4 (config.num_register_tokens),
    so patch tokens start at 1 + num_register_tokens (transformers docs, 2026-11-12)."""
    first_patch = 1 + int(getattr(model.config, "num_register_tokens", 0) or 0)
    n, k = slices.shape[:2]
    flat = slices.reshape(n * k, *slices.shape[2:])
    outs = []
    with torch.no_grad():
        for i in range(0, len(flat), batch):
            x = preprocess(flat[i:i + batch], torch).to(device)
            h = model(pixel_values=x).last_hidden_state
            outs.append(torch.cat([h[:, 0], h[:, first_patch:].mean(dim=1)], dim=1).float().cpu().numpy())
    if not outs:
        return np.zeros((n, 0), np.float32)
    return np.concatenate(outs).reshape(n, -1)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slices", required=True, type=Path)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    import torch
    from transformers import AutoModel
    d = np.load(a.slices / "slices.npz")
    S = d["slices"][: a.limit] if a.limit else d["slices"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(str(a.model), local_files_only=True).to(device).eval()
    t0 = time.time()
    emb = embed(S, model, torch, device, a.batch)
    a.out.mkdir(parents=True, exist_ok=False)
    n = len(S)
    np.savez_compressed(a.out / "embeddings.npz", emb=emb.astype(np.float32), case_id=d["case_id"][:n],
                        fdi_report=d["fdi_report"][:n], fdi_mask=d["fdi_mask"][:n], truncated=d["truncated"][:n])
    root = str(Path(__file__).resolve().parents[3])
    sha = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], check=False, capture_output=True, text=True).stdout.strip()
    # code folders only, untracked files included (an uncommitted new script must count as dirty)
    dirty = bool(subprocess.run(["git", "-C", root, "status", "--porcelain", "--", "src", "jobs"], check=False,
                                capture_output=True, text=True).stdout.strip())
    versions = Path(root) / "models" / "dino_versions.json"
    info = {"slices_run": str(a.slices), "model": str(a.model), "n_teeth": n, "emb_dim": int(emb.shape[1]) if n else 0,
            "device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "seconds": round(time.time() - t0, 1), "git_sha": sha, "git_dirty": dirty, "torch": torch.__version__,
            "argv": sys.argv,
            "model_versions": json.loads(versions.read_text()) if versions.exists() else None}
    (a.out / "info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
