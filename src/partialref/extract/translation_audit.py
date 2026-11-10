"""Translation audit: does the English report say the same as the Italian original? (LOCAL LLM, codes only)

The English ToothFairy4 reports are translations of the Italian originals. For every report that exists in
both languages a local model (vLLM server, started by jobs/translation_audit.sbatch) goes through the
Italian text segment by segment and codes where the English differs: category, whether the clinical
meaning changes, and the teeth concerned. Report text is read inside this process only; outputs hold ids,
segment numbers and codes, never text.

Recall check (--plant N): for N pairs, errors are planted by code into the English text (a changed tooth
number, right <-> left, present <-> absent, a dropped segment) and the audit is repeated. A planted error
counts as detected when the audit reports more meaning-changing problems than for the unchanged pair.

Usage (inside the job, server already up):
    python -m partialref.extract.translation_audit --manifest results/manifest/<run> \
        --server http://127.0.0.1:PORT --model qwen3-32b --half tune --plant 60 \
        [--known results/audit/<run>/tune_read_verdicts.csv] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from partialref.extract.ontology import FDIStr
from partialref.extract.per_report import _post, detect_mode
from partialref.extract.score_parser import parser_half
from partialref.provenance import make_run_id, write_provenance


class Category(str, Enum):
    PresenceAbsence = "PresenceAbsence"
    ToothNumber = "ToothNumber"
    Side = "Side"
    Jaw = "Jaw"
    Negation = "Negation"
    FindingChanged = "FindingChanged"
    FindingOmitted = "FindingOmitted"
    Untranslated = "Untranslated"
    Other = "Other"


ADDED = "FindingAdded"            # category used in our tables for english_only entries


class Problem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    category: Category
    meaning_changed: bool
    tooth_ids: list[FDIStr] = Field(default_factory=list)


class SegmentCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")
    it: int = Field(ge=1)
    problems: list[Problem] = Field(default_factory=list)


class EnglishOnly(BaseModel):
    model_config = ConfigDict(extra="forbid")
    en: int = Field(ge=1)
    meaning_changed: bool
    tooth_ids: list[FDIStr] = Field(default_factory=list)


class PairAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    segments: list[SegmentCheck]
    english_only: list[EnglishOnly] = Field(default_factory=list)


SYSTEM_PROMPT = """You check the English translation of ONE Italian dental CBCT radiology report. The Italian is the
original. Find every place where the English says something clinically different from the Italian.
Both texts are given as numbered segments. Return only JSON that matches the given schema:
{"segments": [{"it": <Italian segment number>, "problems": [...]}, ...], "english_only": [...]}.

Go through the Italian segments in order and return exactly ONE entry per Italian segment. "problems" is []
when the English carries the same clinical content. Wording, word order, style, and how the content is
split into segments do not matter: the matching English text may sit in a segment with another number.
Then list in "english_only" every English segment that states a clinical finding with no source anywhere
in the Italian ({"en": <English segment number>, "meaning_changed": ..., "tooth_ids": [...]}).

Problem categories:
- PresenceAbsence: teeth stated as present / in the arch / erupted in one language and as missing /
  absent / edentulous in the other (either direction).
- ToothNumber: a tooth number, a tooth range, a quadrant or a region is different.
- Side: right versus left, or one side versus both sides.
- Jaw: upper versus lower jaw (maxilla versus mandible).
- Negation: a finding stated in one language and denied in the other (teeth present versus missing is
  PresenceAbsence, not Negation).
- FindingChanged: the same place, but another finding or another attribute: kind of lesion or treatment,
  grade / severity, adequate versus inadequate, impacted versus erupted, in contact versus not in contact,
  direction of the canal course, what the scan includes.
- FindingOmitted: clinical content of the Italian segment is missing from the English.
- Untranslated: an Italian word or abbreviation is left in Italian in the English.
- Other: any other difference.

"meaning_changed": true when a dentist who reads only the English would record a different finding, tooth,
side, jaw, presence / absence or grade than from the Italian. false when the meaning is still clear
(an untranslated abbreviation, a slightly inexact synonym). Report a problem only when you can point to it
in the two texts; do not guess.
"tooth_ids": FDI numbers of the teeth concerned, two digits as strings (for ToothNumber: the Italian
and the English number); [] if no tooth is concerned.

Italian conventions:
- "e.d.", "E.D.", "ed", "el.", "elemento", "elementi" = dental element(s) = tooth / teeth. "Presenti e.d.
  da 31 a 35" = teeth 31 to 35 are PRESENT. "e.d." never means edentulous.
- Tooth numbers may be written 3.6 or 3,6 (= 36). "ottavo" = third molar (tooth 8 of a quadrant),
  "settimo" = second molar, "sesto" = first molar; "IV quadrante" = fourth quadrant.
- "in arcata" = present in the arch; "incluso" / "in inclusione" = impacted; "semi-incluso" = partially
  impacted; "estruso" = extruded; "avulso" / "avulsione" = missing / extracted; "edentulia" = edentulism;
  "sella edentula" = edentulous saddle; "residuo radicolare" = root remnant; "otturazione" = filling;
  "congruo" / "incongruo" = adequate / inadequate; "corona in sospensione" = pontic; "NAI" = inferior
  alveolar nerve (IAN); "dx" = right; "sx" / "sn" = left."""


# ALL example texts are SYNTHETIC: written for this project, not taken or paraphrased from any ToothFairy
# report (DUA: real report text is never committed). The first example carries the one error pattern found
# on the tune half (wording pattern only): a "teeth present" statement with the abbreviation, translated as
# missing teeth.
FEWSHOT: list[tuple[str, str, dict]] = [
    (
        ("Mandibola: presenti e.d. da 33 a 43.\nAssenza di e.d. 36 e 46.\nImpianto endosseo in sede 3.5, ben "
         "osteointegrato.\nCanale mandibolare destro in stretto rapporto con l'apice di 48 incluso.\nNon lesioni "
         "osteolitiche.\nMascellare superiore non incluso nella scansione."),
        ("Mandible: edentulism from 33 to 43.\nAbsence of teeth 36 and 46.\nEndosseous implant at site 35, well "
         "osseointegrated.\nLeft mandibular canal in close relationship with the apex of impacted 48.\nNo "
         "osteolytic lesions.\nA periapical lesion is noted at tooth 47.\nMaxilla not included in the scan."),
        {"segments": [
            {"it": 1, "problems": [{"category": "PresenceAbsence", "meaning_changed": True,
                                    "tooth_ids": ["31", "32", "33", "41", "42", "43"]}]},
            {"it": 2, "problems": []},
            {"it": 3, "problems": []},
            {"it": 4, "problems": [{"category": "Side", "meaning_changed": True, "tooth_ids": ["48"]}]},
            {"it": 5, "problems": []},
            {"it": 6, "problems": []}],
         "english_only": [{"en": 6, "meaning_changed": True, "tooth_ids": ["47"]}]},
    ),
    (
        ("Arcata inferiore: e.d. 38 in inclusione ossea, mesioverso. Carie a carico di e.d. 2,6 e 2,7. "
         "Trattamento endodontico incongruo di 46 con lesione periapicale. Edentulia completa del mascellare "
         "superiore. Atrofia ossea severa."),
        ("Lower arch: e.d. 38 with bony impaction, mesioverted. Caries involving teeth 26 and 27. Adequate "
         "endodontic treatment of tooth 46 with a periapical lesion. Complete edentulism of the maxilla with "
         "moderate bone atrophy."),
        {"segments": [
            {"it": 1, "problems": [{"category": "Untranslated", "meaning_changed": False, "tooth_ids": ["38"]}]},
            {"it": 2, "problems": []},
            {"it": 3, "problems": [{"category": "FindingChanged", "meaning_changed": True, "tooth_ids": ["46"]}]},
            {"it": 4, "problems": []},
            {"it": 5, "problems": [{"category": "FindingChanged", "meaning_changed": True, "tooth_ids": []}]}],
         "english_only": []},
    ),
    (
        ("Elemento 4.7 con otturazione coronale.\nResiduo radicolare in sede 15.\nOttavi inferiori in arcata.\n"
         "Seno mascellare sinistro con ispessimento mucoso.\nCondili mandibolari non inclusi nel volume."),
        ("Tooth 46 with a coronal filling.\nThe lower third molars are missing.\nLeft maxillary sinus with mucosal "
         "thickening.\nThe mandibular condyles are not included in the volume."),
        {"segments": [
            {"it": 1, "problems": [{"category": "ToothNumber", "meaning_changed": True, "tooth_ids": ["46", "47"]}]},
            {"it": 2, "problems": [{"category": "FindingOmitted", "meaning_changed": True, "tooth_ids": ["15"]}]},
            {"it": 3, "problems": [{"category": "PresenceAbsence", "meaning_changed": True,
                                    "tooth_ids": ["38", "48"]}]},
            {"it": 4, "problems": []},
            {"it": 5, "problems": []}],
         "english_only": []},
    ),
]

SEG_RE = re.compile(r"\n+|(?<=[.;])\s+(?=[A-ZÀ-Ý])")   # line break, or sentence end before a capital letter


def split_segments(text: str) -> list[str]:
    return [s.strip() for s in SEG_RE.split(text.strip()) if s.strip()]


def pair_prompt(it_segs: list[str], en_segs: list[str]) -> str:
    def num(segs: list[str]) -> str:
        return "\n".join(f"[{i}] {s}" for i, s in enumerate(segs, 1))
    return f"ITALIAN (original):\n{num(it_segs)}\n\nENGLISH (translation):\n{num(en_segs)}"


def messages(it_text: str, en_text: str) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for it, en, out in FEWSHOT:
        msgs.append({"role": "user", "content": pair_prompt(split_segments(it), split_segments(en))})
        msgs.append({"role": "assistant", "content": json.dumps(out)})
    msgs.append({"role": "user", "content": pair_prompt(split_segments(it_text), split_segments(en_text))})
    return msgs


def request_body(model: str, it_text: str, en_text: str, mode: str, repetition_penalty: float | None = None) -> dict:
    body = {"model": model, "messages": messages(it_text, en_text), "temperature": 0.0, "max_tokens": 3072,
            "chat_template_kwargs": {"enable_thinking": False}}
    if repetition_penalty:
        body["repetition_penalty"] = repetition_penalty
    schema = PairAudit.model_json_schema()
    if mode == "response_format":
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "pair_audit", "schema": schema}}
    else:
        body["structured_outputs"] = {"json": schema}
    return body


def clean_audit(audit: PairAudit, n_it: int, n_en: int) -> tuple[PairAudit, dict]:
    """Keep entries whose segment number exists, merge repeated numbers, count Italian segments without an
    entry (n_unchecked) and entries pointing outside the text (n_out_of_range)."""
    by_it: dict[int, list[Problem]] = {}
    out_of_range = 0
    for s in audit.segments:
        if s.it > n_it:
            out_of_range += 1
            continue
        by_it.setdefault(s.it, []).extend(s.problems)
    eng = [e for e in audit.english_only if e.en <= n_en]
    out_of_range += len(audit.english_only) - len(eng)
    cleaned = PairAudit(segments=[SegmentCheck(it=i, problems=p) for i, p in sorted(by_it.items())], english_only=eng)
    return cleaned, {"n_it": n_it, "n_en": n_en, "n_unchecked": n_it - len(by_it), "n_out_of_range": out_of_range}


def problem_rows(audit: PairAudit) -> list[dict]:
    """One row per problem: language side, segment number, category, meaning_changed, teeth. Codes only."""
    rows = [{"where": "it", "segment": s.it, "category": p.category.value, "meaning_changed": p.meaning_changed,
             "tooth_ids": ";".join(sorted(p.tooth_ids))} for s in audit.segments for p in s.problems]
    rows += [{"where": "en", "segment": e.en, "category": ADDED, "meaning_changed": e.meaning_changed,
              "tooth_ids": ";".join(sorted(e.tooth_ids))} for e in audit.english_only]
    return rows


def audit_one(server: str, model: str, it_text: str, en_text: str, mode: str,
              penalties: tuple[float | None, ...] = (None, 1.1)) -> tuple[PairAudit | None, dict]:
    """Audit one pair. A failed or incomplete answer (Italian segments without an entry) is retried once with
    a repetition penalty; the attempt with the fewest unchecked segments is kept."""
    n_it, n_en = len(split_segments(it_text)), len(split_segments(en_text))
    best: tuple[PairAudit | None, dict] | None = None
    for k, pen in enumerate(penalties, 1):
        t0 = time.time()
        meta: dict = {"repetition_penalty": pen, "attempt": k, "n_it": n_it, "n_en": n_en}
        audit = None
        try:
            resp = _post(f"{server}/v1/chat/completions", request_body(model, it_text, en_text, mode, pen))
            meta.update(prompt_tokens=resp.get("usage", {}).get("prompt_tokens"),
                        completion_tokens=resp.get("usage", {}).get("completion_tokens"),
                        finish_reason=resp["choices"][0].get("finish_reason"))
            audit, info = clean_audit(PairAudit.model_validate_json(resp["choices"][0]["message"]["content"]), n_it, n_en)
            meta.update(info)
            meta["error"] = None
        except ValidationError as e:                   # also covers non-JSON / truncated output
            meta["error"] = f"validation: {e.error_count()} errors"
        except urllib.error.HTTPError as e:
            meta["error"] = f"http {e.code}"
        except Exception as e:  # noqa: BLE001  (record and continue)
            meta["error"] = type(e).__name__
        meta["seconds"] = round(time.time() - t0, 2)

        def badness(a, m):
            return (a is None, m.get("n_unchecked") or 0)
        if best is None or badness(audit, meta) < badness(*best):
            best = (audit, meta)
        if badness(*best) == (False, 0):
            break
    return best  # type: ignore[return-value]


# ---- planted errors (recall check). Each function returns the changed English text and a code-only detail,
# ---- or None when the text offers no place for this kind of error.
TOOTH_TOKEN = re.compile(r"(?<![\d.,])([1-4][1-8])(?!\d|[.,]\d|\s?(?:mm|cm|%|°))")
SIDE_TOKEN = re.compile(r"\b(right|left)\b", re.IGNORECASE)
POLARITY = [(re.compile(r"\babsence of\b", re.IGNORECASE), "presence of"),
            (re.compile(r"\bpresence of\b", re.IGNORECASE), "absence of"),
            (re.compile(r"\b(is|are) (?:absent|missing)\b", re.IGNORECASE), r"\1 present"),
            (re.compile(r"\b(is|are) present\b", re.IGNORECASE), r"\1 absent")]
EXPECTED = {"tooth": {"ToothNumber"}, "side": {"Side"}, "polarity": {"PresenceAbsence", "Negation"},
            "drop": {"FindingOmitted"}}


def _like(new: str, old: str) -> str:
    return new.upper() if old.isupper() else new.capitalize() if old[:1].isupper() else new


def _sub(text: str, m: re.Match, new: str) -> str:
    return text[:m.start()] + new + text[m.end():]


def plant(en_text: str, kind: str, rng: random.Random) -> tuple[str, str] | None:
    if kind == "tooth":
        hits = list(TOOTH_TOKEN.finditer(en_text))
        present = {m.group(1) for m in hits}
        rng.shuffle(hits)
        for m in hits:
            q, u = m.group(1)[0], int(m.group(1)[1])
            for v in rng.sample([u - 1, u + 1], 2):
                if 1 <= v <= 8 and f"{q}{v}" not in present:
                    return _sub(en_text, m, f"{q}{v}"), f"{m.group(1)}->{q}{v}"
        return None
    if kind == "side":
        hits = list(SIDE_TOKEN.finditer(en_text))
        if not hits:
            return None
        m = rng.choice(hits)
        return _sub(en_text, m, _like("left" if m.group(1).lower() == "right" else "right", m.group(1))), ""
    if kind == "polarity":
        hits = [(m, rep) for rx, rep in POLARITY for m in rx.finditer(en_text)]
        if not hits:
            return None
        m, rep = rng.choice(sorted(hits, key=lambda h: h[0].start()))
        return _sub(en_text, m, _like(m.expand(rep).lower(), m.group(0))), ""
    if kind == "drop":
        segs = split_segments(en_text)
        cand = [i for i, s in enumerate(segs) if TOOTH_TOKEN.search(s)]
        if len(segs) < 2 or not cand:
            return None
        i = rng.choice(cand)
        return "\n".join(segs[:i] + segs[i + 1:]), f"en_segment {i + 1}"
    raise ValueError(kind)


def _h(*parts: str) -> int:
    return int(hashlib.sha256(":".join(parts).encode()).hexdigest()[:16], 16)


def n_meaning(audit: dict | None) -> int | None:
    if audit is None:
        return None
    return sum(r["meaning_changed"] for r in problem_rows(PairAudit.model_validate(audit)))


def known_class(verdict: str) -> str:
    v = str(verdict).lower()
    return "mistranslation" if v.startswith("mistranslation") else "minor" if v.startswith("minor") \
        else "abbreviation_kept" if v.startswith("abbreviation") else "faithful"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--server", required=True)
    ap.add_argument("--model", required=True, help="served model name")
    ap.add_argument("--model-path", default="", help="recorded in provenance")
    ap.add_argument("--half", default="all", choices=["tune", "heldout", "all"], help="parser_half of the case")
    ap.add_argument("--limit-pairs", type=int, default=0, help="first N pairs only (diagnostic)")
    ap.add_argument("--plant", type=int, default=0, help="recall check on N pairs (x 4 kinds of planted error)")
    ap.add_argument("--known", type=Path, help="CSV case_id, report_id, verdict, extent from a human/Claude read")
    ap.add_argument("--out-dir", type=Path, help="existing run dir to resume / extend")
    ap.add_argument("--results-dir", default=Path("results"), type=Path)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--tag", default="translation-audit")
    a = ap.parse_args(argv)

    reps = pd.read_parquet(a.manifest / "reports.parquet")
    wide = reps[reps["lang"].isin(["it", "en"])].pivot(index=["case_id", "report_id"], columns="lang", values="path")
    pairs = wide.dropna(subset=["it", "en"]).reset_index().sort_values(["case_id", "report_id"])
    pairs["half"] = pairs["case_id"].map(parser_half)
    if a.half != "all":
        pairs = pairs[pairs["half"] == a.half]
    if a.limit_pairs:
        pairs = pairs.head(a.limit_pairs)
    if pairs.empty:
        raise SystemExit("no report pairs selected")
    out = a.out_dir or (a.results_dir / "audit" / make_run_id(f"{a.tag}-{a.half}"))
    (out / "per_pair").mkdir(parents=True, exist_ok=True)
    (out / "planted").mkdir(exist_ok=True)
    mode = detect_mode(a.server, a.model)
    order = sorted(pairs.itertuples(), key=lambda r: _h("plant-sample", r.case_id, r.report_id))[: a.plant]
    jobs = [(r, None) for r in pairs.itertuples()] + [(r, k) for r in order for k in EXPECTED]
    print(f"structured-output mode: {mode}; pairs: {len(pairs)}; planted audits: up to {len(order) * len(EXPECTED)}",
          flush=True)

    def read(p: str) -> str:
        return Path(p).read_text(encoding="utf-8", errors="replace")

    def work(job) -> None:
        row, kind = job
        dst = (out / "planted" / f"{row.case_id}__{row.report_id}__{kind}.json") if kind else \
            (out / "per_pair" / f"{row.case_id}__{row.report_id}.json")
        if dst.exists():
            try:
                if json.loads(dst.read_text())["meta"].get("error") is None:      # resume
                    return
            except (json.JSONDecodeError, KeyError):
                pass
        en, detail = read(row.en), None
        if kind:
            planted = plant(en, kind, random.Random(_h("plant", kind, row.case_id, row.report_id)))
            if planted is None:
                return                                  # this pair has no place for this kind of error
            en, detail = planted
        audit, meta = audit_one(a.server, a.model, read(row.it), en, mode)
        meta.update(case_id=row.case_id, report_id=row.report_id, half=row.half, kind=kind, detail=detail)
        tmp = dst.with_suffix(".json.tmp")              # atomic: a killed job never leaves a partial file
        tmp.write_text(json.dumps({"meta": meta, "audit": audit.model_dump(mode="json") if audit else None}))
        tmp.replace(dst)

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(work, jobs))

    # ---- tables: every per-pair file in the run dir (so a tune run extended by a held-out run gives one table)
    pair_rows, prob_rows, orig = [], [], {}
    for f in sorted((out / "per_pair").glob("*.json")):
        d = json.loads(f.read_text())
        m, rows = d["meta"], problem_rows(PairAudit.model_validate(d["audit"])) if d["audit"] else []
        key = {"case_id": m["case_id"], "report_id": m["report_id"], "half": m["half"]}
        prob_rows += [{**key, **r} for r in rows]
        mc = [r for r in rows if r["meaning_changed"]]
        orig[(m["case_id"], m["report_id"])] = len(mc) if d["audit"] else None
        pair_rows.append({**key, "reader": m["report_id"].split("_")[0], "n_it": m.get("n_it"), "n_en": m.get("n_en"),
                          "n_unchecked": m.get("n_unchecked"), "n_out_of_range": m.get("n_out_of_range"),
                          "n_problems": len(rows), "n_meaning_changed": len(mc),
                          "categories_meaning": ";".join(sorted({r["category"] for r in mc})),
                          "categories_wording": ";".join(sorted({r["category"] for r in rows if not r["meaning_changed"]})),
                          "teeth_meaning": ";".join(sorted({t for r in mc for t in r["tooth_ids"].split(";") if t})),
                          "it_segments_meaning": ";".join(str(r["segment"]) for r in mc if r["where"] == "it"),
                          "en_segments_added": ";".join(str(r["segment"]) for r in mc if r["where"] == "en"),
                          "error": m.get("error"), "seconds": m.get("seconds")})
    pdf = pd.DataFrame(pair_rows)
    pdf.to_csv(out / "pairs.csv", index=False)
    prob = pd.DataFrame(prob_rows, columns=["case_id", "report_id", "half", "where", "segment", "category",
                                            "meaning_changed", "tooth_ids"])
    prob["meaning_changed"] = prob["meaning_changed"].astype(bool)
    prob.to_csv(out / "problems.csv", index=False)
    ok = pdf[pdf["error"].isna()]
    flagged = ok[ok["n_meaning_changed"] > 0]
    summary: dict = {
        "out": str(out), "mode": mode, "n_pairs": len(pdf), "n_errors": int(pdf["error"].notna().sum()),
        "n_pairs_with_unchecked_segments": int((ok["n_unchecked"] > 0).sum()),
        "pairs_by_half": pdf["half"].value_counts().to_dict(),
        "flagged_meaning_changed_by_half": flagged["half"].value_counts().to_dict(),
        "flagged_meaning_changed_by_reader": flagged["reader"].value_counts().to_dict(),
        "problems_meaning_changed_by_category": prob[prob["meaning_changed"]]["category"].value_counts().to_dict(),
        "problems_wording_only_by_category": prob[~prob["meaning_changed"]]["category"].value_counts().to_dict(),
        "median_seconds": float(pd.to_numeric(pdf["seconds"], errors="coerce").median())}

    plant_rows = []
    for f in sorted((out / "planted").glob("*.json")):
        d = json.loads(f.read_text())
        m, n0 = d["meta"], orig.get((d["meta"]["case_id"], d["meta"]["report_id"]))
        n1 = n_meaning(d["audit"])
        cats = {r["category"] for r in problem_rows(PairAudit.model_validate(d["audit"])) if r["meaning_changed"]} \
            if d["audit"] else set()
        usable = n0 is not None and n1 is not None
        plant_rows.append({"case_id": m["case_id"], "report_id": m["report_id"], "half": m["half"], "kind": m["kind"],
                           "detail": m["detail"], "n_meaning_original": n0, "n_meaning_planted": n1, "usable": usable,
                           "detected": bool(usable and n1 > n0),
                           "expected_category": bool(usable and cats & EXPECTED[m["kind"]]), "error": m.get("error")})
    if plant_rows:
        pl = pd.DataFrame(plant_rows)
        pl.to_csv(out / "planted.csv", index=False)
        u = pl[pl["usable"]]
        rec = u.groupby("kind").agg(n=("detected", "size"), detected=("detected", "sum"),
                                    expected_category=("expected_category", "sum"))
        rec["recall"] = (rec["detected"] / rec["n"]).round(3)
        summary["planted_not_usable"] = int((~pl["usable"]).sum())
        summary["planted_recall"] = rec.reset_index().to_dict(orient="records")
        print("planted errors (detected = more meaning-changing problems than the unchanged pair):")
        print(rec.to_string())

    if a.known:
        kn = pd.read_csv(a.known, dtype=str)
        kn["known"] = kn["verdict"].map(known_class)
        kn = kn.merge(pdf[["case_id", "report_id", "n_meaning_changed", "categories_meaning", "error"]],
                      on=["case_id", "report_id"], how="left")
        kn["audited"] = kn["n_meaning_changed"].notna() & kn["error"].isna()
        kn["flagged"] = kn["n_meaning_changed"].fillna(0) > 0
        kn.drop(columns=["verdict"]).to_csv(out / "known_check.csv", index=False)   # verdict wording stays in the source file
        tab = kn[kn["audited"]].groupby(["known", "extent"])["flagged"].agg(n="size", flagged="sum").reset_index()
        summary["known_check"] = tab.to_dict(orient="records")
        summary["known_not_audited"] = int((~kn["audited"]).sum())
        print("against read verdicts (flagged = at least one meaning-changing problem):")
        print(tab.to_string(index=False))

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    write_provenance(out, extra={"manifest_run": str(a.manifest), "model": a.model, "model_path": a.model_path,
                                 "half": a.half, "plant": a.plant, "known": str(a.known) if a.known else None,
                                 "structured_output_mode": mode})
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
