# partial-refs — When the Reference Is Partial

Propensity-corrected evaluation of report-generation models when clinician reference reports omit
findings. Target: Medical Image Analysis (MedIA).

- Project plan and operating rules: [HANDOFF.md](HANDOFF.md)
- Decision log: [DECISIONS.md](DECISIONS.md)
- Environment report: [reports/ENV_REPORT.md](reports/ENV_REPORT.md)

## Setup (CPU env, after `env/environment.yml` exists)

```bash
git clone git@github.com:shivamsingh-ml/partial-refs.git && cd partial-refs
git config core.hooksPath .githooks        # blocks .nii/.npz/.sif and files > 5 MB
pip install -e ".[dev]"
pytest -q                                   # synthetic data only
```

ToothFairy4 data is access-controlled: never commit it, push it, or send it to an external API.

## Weekly backup (cluster storage has no backup — HANDOFF §5.6)

Code is backed up by `git push`. Small derived results are pulled **from the lab PC** once a week
(run this on the lab PC, not on the cluster; replace `<msc-login-host>`):

```bash
MSC=ssingh54@<msc-login-host>:/msc/home/ssingh54/partial-refs
DEST=~/partial-refs-backup
rsync -av --prune-empty-dirs --exclude='*parser-audit-sheet*/' \
  --include='*/' --include='metrics.json' --include='provenance.json' --include='config.yaml' \
  --include='*.csv' --exclude='*' "$MSC/results/" "$DEST/results/"
rsync -av "$MSC/reports/" "$DEST/reports/"
rsync -av "$MSC/figures/" "$DEST/figures/"
```

Only numbers, configs and figures: by design no file under `results/` that matches these patterns
contains report text, images or masks (HANDOFF §0.8). Never widen the include list to raw data.
