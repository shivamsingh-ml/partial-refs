"""
Seam between segmentation and your fact extraction (extract_facts.py v3).

Label scheme (corrected, audit S10). run_segmentation already returns FDI numbers (teeth 11-48, pulp 50,
canals 51-53: task1_inference.orig_mapping), which is what extract_facts.py v3 expects, so the mask goes in
unchanged. The original comment here said the model's sequential labels (0-46) reached extract_facts and that
this quirk was kept on purpose; that is not what the code does. _MODEL_TO_FDI below would map FDI labels a
second time, so REMAP_MODEL_TO_FDI=1 is refused. (Which scheme the facts of the VLM's training data used is a
separate question.)

extract() re-derives the S-I axis from anatomy, so the RPI mask goes in as-is.

Return value = frozen handoff schema. Agree its shape with her VLM.
"""

import numpy as np
import SimpleITK as sitk

from config import REMAP_MODEL_TO_FDI
from extract_facts import extract

# Inverse of convert_tf3_singlepulp.py's FDI->sequential map, into the label
# scheme extract_facts declares (teeth FDI 11-48, canals 51-53, pulp 50).
_MODEL_TO_FDI = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10,
    11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18,   # Upper Right -> FDI 11-18
    19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 26, 25: 27, 26: 28,   # Upper Left  -> FDI 21-28
    27: 31, 28: 32, 29: 33, 30: 34, 31: 35, 32: 36, 33: 37, 34: 38,   # Lower Left  -> FDI 31-38
    35: 41, 36: 42, 37: 43, 38: 44, 39: 45, 40: 46, 41: 47, 42: 48,   # Lower Right -> FDI 41-48
    43: 51, 44: 52, 45: 53, 46: 50,                                    # canals + pulp
}


def _remap_to_fdi(arr: np.ndarray) -> np.ndarray:
    lut = np.zeros(max(_MODEL_TO_FDI) + 1, dtype=np.int16)
    for src, dst in _MODEL_TO_FDI.items():
        lut[src] = dst
    return lut[np.clip(arr, 0, lut.shape[0] - 1)]


def run_fact_extraction(cbct_rpi: sitk.Image, mask_rpi: sitk.Image) -> dict:
    arr = sitk.GetArrayFromImage(mask_rpi).astype(np.int16)   # (z, y, x)
    if REMAP_MODEL_TO_FDI:   # audit S10: the mask is already FDI; remapping would map it twice
        raise RuntimeError("REMAP_MODEL_TO_FDI=1 is not supported: run_segmentation already returns FDI labels")
    facts, phrases = extract(arr, mask_rpi.GetSpacing())       # spacing (x, y, z)
    return {"structured": facts, "phrases": phrases}