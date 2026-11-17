# Source of this folder

Verbatim copy of `documentation/competitions/Pretrain_DAE/` from U-Mamba2
(https://github.com/zhiqin1998/U-Mamba2, commit 2046d29785087b656ca69fa02dd40e43e69cfb42, 2026-05-18),
the self-supervised (disruptive autoencoder) pretraining used for V2V2R's U-Mamba2 segmentation network.
Licence: CC BY-NC 4.0 (`LICENSE_U-Mamba2`, copied from the U-Mamba2 repository root).

Imported unchanged on 2026-11-17 so that our changes are visible as separate commits (HANDOFF §0.4). Every later
change is its own commit that says why. Reason for rerunning the pretraining: the released weights were pretrained on
all 532 ToothFairy3 images (`dataset_tf3.json`), which include our locked test set (DECISIONS 2026-11-16).
