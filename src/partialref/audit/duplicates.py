"""Independence check of report pairs (HANDOFF M1 step 2).

For every pair of reports of the same case (EN by default): character 5-gram TF-IDF cosine,
normalised Levenshtein similarity (rapidfuzz) and MinHash Jaccard (datasketch, char 5-shingles).
Also, for every report, its most similar report from a *different* case (template reuse).

Report text is read inside this process only; **outputs contain IDs and scores, never text**.
Starting thresholds (HANDOFF): cosine > 0.9 or Levenshtein > 0.85 → flagged; final thresholds are
chosen after Shivam inspects the exported top-20 flagged and 20 random pairs on the cluster.

Usage:
    python -m partialref.audit.duplicates --manifest results/manifest/<run_id> [--lang en] [--exclude P116,P117]
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from datasketch import MinHash
from rapidfuzz.distance import Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer

from partialref.provenance import make_run_id, write_provenance

COS_T, LEV_T = 0.9, 0.85


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def minhash(text: str, k: int = 5, num_perm: int = 128) -> MinHash:
    m = MinHash(num_perm=num_perm, seed=1)
    for i in range(max(1, len(text) - k + 1)):
        m.update(text[i:i + k].encode("utf-8"))
    return m


def score_pairs(reports: pd.DataFrame, texts: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """reports: one row per report (case_id, report_id, report_x), aligned with ``texts``."""
    norm = [normalise(t) for t in texts]
    tfidf = TfidfVectorizer(analyzer="char", ngram_range=(5, 5)).fit_transform(norm)
    sims = (tfidf @ tfidf.T).toarray()
    mh = [minhash(t) for t in norm]
    rep = reports.reset_index(drop=True)

    rows = []
    for _, g in rep.groupby("case_id"):
        for i, j in itertools.combinations(g.index, 2):
            rows.append({
                "case_id": rep.at[i, "case_id"],
                "report_a": rep.at[i, "report_id"], "report_b": rep.at[j, "report_id"],
                "x_a": rep.at[i, "report_x"], "x_b": rep.at[j, "report_x"],
                "cosine": float(sims[i, j]),
                "levenshtein": float(Levenshtein.normalized_similarity(norm[i], norm[j])),
                "minhash_jaccard": float(mh[i].jaccard(mh[j])),
            })
    pairs = pd.DataFrame(rows)
    if not pairs.empty:
        pairs["flagged"] = (pairs["cosine"] > COS_T) | (pairs["levenshtein"] > LEV_T)

    same_case = rep["case_id"].to_numpy()[:, None] == rep["case_id"].to_numpy()[None, :]
    other = np.where(same_case, -1.0, sims)
    nn = other.argmax(axis=1)
    nearest = pd.DataFrame({
        "case_id": rep["case_id"], "report_id": rep["report_id"], "report_x": rep["report_x"],
        "nn_case_id": rep["case_id"].to_numpy()[nn], "nn_report_id": rep["report_id"].to_numpy()[nn],
        "nn_report_x": rep["report_x"].to_numpy()[nn], "nn_cosine": other[np.arange(len(rep)), nn],
    })
    return pairs, nearest


def quantiles(s: pd.Series) -> dict:
    return {f"q{int(q * 100):02d}": round(float(s.quantile(q)), 4) for q in (0, .05, .25, .5, .75, .95, .99, 1)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--lang", default="en", choices=["en", "it"])
    ap.add_argument("--exclude", type=lambda s: [x for x in s.split(",") if x], default=[])
    ap.add_argument("--results-dir", default=Path("results"), type=Path)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="duplicates")
    a = ap.parse_args(argv)

    reps = pd.read_parquet(a.manifest / "reports.parquet")
    reps = reps[(reps["lang"] == a.lang) & ~reps["case_id"].isin(a.exclude)].sort_values(
        ["case_id", "report_id"]).reset_index(drop=True)
    texts = [Path(p).read_text(encoding="utf-8", errors="replace") for p in reps["path"]]
    pairs, nearest = score_pairs(reps[["case_id", "report_id", "report_x"]], texts)
    del texts

    run_id = make_run_id(a.tag)
    out = a.results_dir / "audit" / run_id
    out.mkdir(parents=True, exist_ok=False)
    pairs.to_csv(out / "pairs_within_case.csv", index=False)
    nearest.to_csv(out / "nearest_other_case.csv", index=False)

    rng = np.random.default_rng(a.seed)
    flagged = pairs[pairs["flagged"]].sort_values("cosine", ascending=False)
    review = pd.concat([
        flagged.head(20).assign(review_group="top_flagged"),
        pairs.iloc[rng.choice(len(pairs), size=min(20, len(pairs)), replace=False)].assign(review_group="random"),
    ])[["review_group", "case_id", "report_a", "report_b", "cosine", "levenshtein", "minhash_jaccard"]]
    review.to_csv(out / "review_ids.csv", index=False)   # IDs + scores only; Shivam reads texts on the cluster

    summary = {
        "run_id": run_id, "lang": a.lang, "excluded_cases": a.exclude,
        "n_reports": int(len(reps)), "n_pairs": int(len(pairs)),
        "n_flagged": int(pairs["flagged"].sum()), "share_flagged": round(float(pairs["flagged"].mean()), 4),
        "thresholds": {"cosine": COS_T, "levenshtein": LEV_T},
        "within_case": {m: quantiles(pairs[m]) for m in ["cosine", "levenshtein", "minhash_jaccard"]},
        "nearest_other_case_cosine": quantiles(nearest["nn_cosine"]),
        "n_reports_with_cross_case_cosine_gt_0.9": int((nearest["nn_cosine"] > COS_T).sum()),
        "within_vs_cross_median": {"within_case_cosine": round(float(pairs["cosine"].median()), 4),
                                   "nearest_other_case_cosine": round(float(nearest["nn_cosine"].median()), 4)},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    write_provenance(out, extra={"run_id": run_id, "manifest_run": str(a.manifest)},
                     packages=["numpy", "pandas", "scikit-learn", "rapidfuzz", "datasketch"])
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
