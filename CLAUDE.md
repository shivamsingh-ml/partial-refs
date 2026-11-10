# CLAUDE.md — partial-refs

**Read [HANDOFF.md](HANDOFF.md) in full before doing anything.** It is the source of truth.
Its §0 operating rules override default habits. Decisions go in [DECISIONS.md](DECISIONS.md).

## §0 rules (repeated from HANDOFF.md)

1. **Gates are hard stops.** Each milestone ends in a gate (G0–G5): write `reports/REPORT_Mx.md`
   with the numbers, then stop and wait for Shivam's decision.
2. **Shivam reviews code before it runs.** Before any GPU job, or CPU job longer than ~30 min,
   show the diff/script and the exact `sbatch` command, then wait for approval.
3. **Cheap diagnostic first.** Every pipeline runs on 3–5 cases (`LIMIT`/`CASE_IDS`) before a full run.
4. **No unnecessary wrappers.** Call the submitted code in `external/odin2026/` directly. Edit it
   in place with a commit that says why (the import commit is verbatim); don't wrap it.
5. **Diagnose, don't work around.** Fix root causes; backup plans (HANDOFF §12) only when the cause
   is outside our control.
6. **Verify empirically.** Check file layouts, label maps, IDs, orientation on real files; record in DECISIONS.md.
7. **Never fabricate results.** Every reported number comes from a file in `results/` produced by
   code in this repo at a recorded git SHA.
8. **DUA.** ToothFairy4 data (volumes, masks, report text) is never committed, pushed, pasted into
   issues/PRs/chat, or sent to an external API. Local models only.
   **Exception confirmed by Shivam 2026-11-10 (DECISIONS.md):** Claude may read report TEXT of the tune
   half only, for parser error analysis. Never the held-out half, never images/masks, never for parsing
   or metrics; report text never goes into commits, logs, results or prompt examples.
9. **No backups on the cluster.** Push code daily; sync small derived results weekly (HANDOFF §5.6).
10. **Log decisions** in DECISIONS.md with date, reason, evidence path.
11. **Login node is shared.** Only editing, git, `ls`/`head`, jobs < ~2 min on the login node.
    Everything else through SLURM (`sbatch`/`srun`).

## Additional standing rules from Shivam

- Ask before running any command on the cluster; propose the exact commands first.
- Show diffs before committing. No `Co-Authored-By: Claude` (or any Claude attribution) in commits/PRs.
- Never guess a `<<FILL>>` value; list missing ones instead.
- Tests use synthetic data only (`partialref.propensity.simulate`), never real reports.

## Cluster facts (verified 2026-11-07, see reports/ENV_REPORT.md)

- SLURM account `uzk`; partitions `cpu` (default), `gpu`, `storage`.
- GPU jobs need a QoS matching the GRES, e.g. `--qos=a100 --gres=gpu:a100:1`. Max wall 3 days
  (`a100_long`: 5 days).
- No `module` system. Apptainer at `/usr/bin/apptainer`. Conda at `~/miniconda3`.
