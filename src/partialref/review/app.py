"""Report review app: read each report next to its parsed findings (explore by default, optional logging).

Runs ONLY on a compute node via jobs/review_app.sbatch (never on the login node), opened in the browser through
an SSH tunnel. Report text is shown on screen and never written anywhere. Optional logs hold IDs and codes only.
Blinded: Lumetti's labels and other parses are not shown. Access needs the token from the job's token file.

    REVIEW_TOKEN=... REVIEW_PARSE_RUN=results/extract/<run> REVIEW_MANIFEST=results/manifest/<run> \
        [REVIEW_LANG=en] [REVIEW_LOG_ROOT=results/review] [REVIEW_LIST=<review list json>] \
        streamlit run src/partialref/review/app.py

With REVIEW_LIST (partialref.review.sample) the app steps through that fixed list, one report at a time, with
logging on: "base" reports get a verdict for every finding plus the missed findings; "extra" reports only for
the finding types they were drawn for.
"""

from __future__ import annotations

import argparse
import hmac
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from partialref.extract.fewshot import SYSTEM_PROMPT
from partialref.extract.ontology import FindingType
from partialref.review.data import (
    REPORT_FLAGS,
    VERDICTS,
    case_order,
    done_reports,
    finding_rows,
    load_parse,
    load_reports,
    load_review_list,
    log_review,
    read_log,
    review_scope,
)

PERMANENT = [f"{q}{i}" for q in (1, 2, 3, 4) for i in range(1, 9)]
SLOT_CHOICES = PERMANENT + ["Left", "Right", "QuadrantI", "QuadrantII", "QuadrantIII", "QuadrantIV",
                            "UpperArch", "LowerArch"]
MISSED_ROWS = 6
GLOSSARY = """Finding types in this check (made-up example wording, not from the data):
- ToothAbsence: a tooth is missing ("36 missing", "edentulous upper jaw", "agenesis of 12")
- DentalImplant: metal screw replacing a tooth root ("implant in position 46")
- EndodonticTreatment: root canal treatment / root canal filling ("endodontic treatment of 26")
- ConservativeRestoration: a filling ("composite filling on 14", "amalgam restoration 37")
- ProstheticDevice: crown, bridge, post-and-core, denture ("crown on 47", "bridge from 34 to 36")
- Caries: tooth decay ("caries on 15", "carious lesion of 46")
- MandibularCanalCourse: course of the nerve canal in the lower jaw ("canals have a regular course")

Tooth numbers (FDI, patient's view): first digit 1 upper right, 2 upper left, 3 lower left, 4 lower right;
second digit 1 = front tooth ... 8 = wisdom tooth.

Coding rules of the parser (judge against these, not against your own habits):
- A bridge lists every tooth from the first to the last abutment; each pontic (replaced tooth) is ALSO a
  ToothAbsence.
- A residual root or an impacted / unerupted tooth is NEVER ToothAbsence (other types, not in this check).
- Crown + post-and-core on one tooth = two ProstheticDevice findings.
- An implant is DentalImplant only (no ToothAbsence for that position). Planned implants are not findings.
- "Not included in the scan" describes the scan, never ToothAbsence.

Verdict per finding: correct | wrong_slot (right finding, wrong tooth/side: pick the right one) |
not_in_report (the report does not say this) | unclear. Then add findings OF THESE TYPES that the report
states but the parser did not list."""


def args() -> argparse.Namespace:
    """Settings from environment variables (set by jobs/review_app.sbatch; tests set them directly)."""
    env = os.environ
    for k in ("REVIEW_PARSE_RUN", "REVIEW_MANIFEST"):
        if not env.get(k):
            st.error(f"{k} is not set: start the app with jobs/review_app.sbatch.")
            st.stop()
    return argparse.Namespace(parse_run=Path(env["REVIEW_PARSE_RUN"]), manifest=Path(env["REVIEW_MANIFEST"]),
                              lang=env.get("REVIEW_LANG", "en"), log_root=Path(env.get("REVIEW_LOG_ROOT", "results/review")),
                              review_list=Path(env["REVIEW_LIST"]) if env.get("REVIEW_LIST") else None)


def gate() -> None:
    """Token check (the job writes the token to an owner-only file); without a token set, refuse to run."""
    # own machine via partialref.review.local: only honoured when the server listens on loopback only
    if os.environ.get("REVIEW_LOCAL") == "1" and st.get_option("server.address") in ("127.0.0.1", "localhost"):
        return
    token = os.environ.get("REVIEW_TOKEN", "")
    if not token:
        st.error("REVIEW_TOKEN is not set: start the app with jobs/review_app.sbatch.")
        st.stop()
    if not st.session_state.get("ok"):
        entered = st.text_input("Access token", type="password")
        if entered and hmac.compare_digest(entered.encode(), token.encode()):
            st.session_state["ok"] = True
            st.rerun()
        st.stop()


def main() -> None:
    st.set_page_config(page_title="Report review", layout="wide")
    gate()
    a = args()
    reports = load_reports(a.manifest, a.lang)
    log_dir = a.log_root / a.parse_run.name
    entries = load_review_list(a.review_list, reports) if a.review_list else None
    scope = review_scope(a.review_list) if a.review_list else None

    with st.sidebar:
        st.markdown(f"**Parse run** `{a.parse_run.name}`  \n**Language** {a.lang}")
        if entries is not None:
            n_extra = sum(e["stratum"] == "extra" for e in entries)
            st.markdown(f"**Review list** {len(entries)} reports ({len(entries) - n_extra} base, {n_extra} extra)")
            if scope:
                st.markdown(f"**Scope** {len(scope)} finding types only (see the glossary)")
            order = [e["case_id"] for e in entries]
            order_key = ("list", str(a.review_list))
        else:
            half = st.selectbox("Cases", ["tune", "heldout", "all"],
                                help="tune: errors may change the prompt. heldout: only count, never tune on it.")
            centres = st.multiselect("Centres", ["P", "F", "S", "A"], default=["P", "F", "S"])
            only_multi = st.checkbox("Only cases with ≥ 2 reports", value=False)
            seed = st.number_input("Shuffle seed", value=0, step=1)
            order = case_order(reports, half, centres, int(seed), only_multi)
            order_key = (half, tuple(centres), seed, only_multi)
        if not order:
            st.warning("No cases for this filter.")
            st.stop()
        if "pos" not in st.session_state or st.session_state.get("order_key") != order_key:
            st.session_state.update(pos=0, order_key=order_key)

        def go_to() -> None:                       # applied once, when the box changes
            cid = st.session_state.get("jump", "").strip()
            st.session_state["jump_miss"] = bool(cid) and cid not in order
            if cid in order:
                st.session_state["pos"] = order.index(cid)

        st.text_input("Go to case ID (optional)", key="jump", on_change=go_to)
        if st.session_state.get("jump_miss"):
            st.caption("That case is not in the review list." if entries is not None else
                       "That case is not in the current filter (half / centres / multi-report).")
        logged = read_log(log_dir)
        done = done_reports(logged, scope) if entries is not None else \
            (set(map(tuple, logged[["case_id", "report_id"]].to_numpy())) if len(logged) else set())
        c1, c2 = st.columns(2)
        if c1.button("◀ Previous", width="stretch"):
            st.session_state["pos"] = max(0, st.session_state["pos"] - 1)
        if c2.button("Next ▶", width="stretch"):
            st.session_state["pos"] = min(len(order) - 1, st.session_state["pos"] + 1)
        if entries is not None and st.button("Next report without a saved decision", width="stretch"):
            todo = [i for i, e in enumerate(entries) if (e["case_id"], e["report_id"]) not in done]
            if todo:
                st.session_state["pos"] = todo[0]
        st.caption(f"{'Report' if entries is not None else 'Case'} {st.session_state['pos'] + 1} of {len(order)}")
        show_it = st.checkbox("Also show the Italian original", value=False)
        log_on = st.checkbox("Enable logging", value=entries is not None,
                             help="Off = explore only, nothing is saved. On = a log form appears under each report.")
        if entries is not None:
            n_done = sum((e["case_id"], e["report_id"]) in done for e in entries)
            st.caption(f"Saved so far: {n_done} of {len(entries)} listed reports")
        else:
            st.caption(f"Logged so far: {logged['report_id'].nunique() if len(logged) else 0} reports")

    case = order[st.session_state["pos"]]
    rep = reports[reports["case_id"] == case]
    judge_types = None
    if entries is not None:
        entry = entries[st.session_state["pos"]]
        rep = rep[rep["report_id"] == entry["report_id"]]
        judge_types = entry["judge_types"]
        saved = "  ·  decision saved" if (case, entry["report_id"]) in done else ""
        st.markdown(f"## {case}  ·  centre {case[0]}  ·  report {entry['report_id']}  ·  {entry['stratum']}{saved}")
        if judge_types:
            st.info("Extra report for: " + ", ".join(judge_types) + ". Judge only the findings of "
                    "this type; missed findings are not entered for extra reports.")
    else:
        st.markdown(f"## {case}  ·  centre {case[0]}  ·  {len(rep)} report(s)  ·  half: {rep['half'].iloc[0]}")
    if scope:
        with st.expander("Glossary: the finding types of this check", expanded=True):
            st.text(GLOSSARY)
    with st.expander("Finding type definitions and coding rules (the parser's own instructions)"):
        st.text(SYSTEM_PROMPT.split("- Use only these qualifier names")[0])

    for tab, r in zip(st.tabs([f"Report {x}" for x in rep["report_id"]]), rep.itertuples()):
        with tab:
            left, right = st.columns([1, 1])
            with left:
                st.markdown("**Report text**")
                st.text(Path(r.path).read_text(encoding="utf-8", errors="replace"))
                if show_it and a.lang == "en":
                    it_path = Path(r.path.replace("/reports_en/", "/reports_it/"))
                    if it_path.exists():
                        st.markdown("**Italian original**")
                        st.text(it_path.read_text(encoding="utf-8", errors="replace"))
            with right:
                d = load_parse(a.parse_run, case, r.report_id, a.lang)
                st.markdown("**Parsed findings**")
                if d is None:
                    st.warning("No parse file for this report in this run.")
                    continue
                m = d.get("meta", {})
                if d.get("extraction") is None:
                    st.error(f"Parse failed: {m.get('error')}")
                fr = finding_rows(d.get("extraction"))
                st.dataframe(fr.drop(columns="idx"), hide_index=True, width="stretch")
                na = (d.get("extraction") or {}).get("not_assessable") or []
                if na:
                    st.caption("Not assessable: " + ", ".join(na))
                flags = {k: m.get(k) for k in ("salvaged", "dropped_findings", "absence_conflicts", "attempts_made",
                                               "kept_attempt") if m.get(k)}
                if m.get("absence_conflict_details"):
                    flags["missing_and_present"] = "; ".join(f"{d['tooth']}+{d['with']}"
                                                            for d in m["absence_conflict_details"])
                if flags:
                    st.caption("Parser notes: " + ", ".join(f"{k}={v}" for k, v in flags.items()))
                if log_on:
                    log_form(log_dir, a, case, r.report_id, r.half, fr, judge_types, strict=entries is not None,
                             scope=scope)


def log_form(log_dir: Path, a: argparse.Namespace, case: str, report_id: str, half: str, fr: pd.DataFrame,
             judge_types: list[str] | None = None, strict: bool = False, scope: list[str] | None = None) -> None:
    """judge_types: only findings of these types get a verdict box and no missed findings are asked (extra
    reports of a review list). scope (scoped list): only these types get verdict boxes and only these can be
    entered as missed; the record carries the scope. strict (review list): warn when a finding to judge was
    left at '—'."""
    if judge_types:
        fr = fr[fr["finding_type"].isin(judge_types)]
    if scope:
        fr = fr[fr["finding_type"].isin(scope)]
    missed_types = scope or [x.value for x in FindingType]
    with st.form(key=f"log-{case}-{report_id}"):
        st.markdown("**Decision for every finding** — a finding left at '—' counts as not judged" if strict else
                    "**Log a decision (optional)** — leave a finding at '—' to skip it")
        verdicts = []
        for f in fr.itertuples():
            c1, c2 = st.columns([2, 1])
            v = c1.selectbox(f"{f.finding_type} · {f.anchor}", ["—"] + VERDICTS, key=f"v-{case}-{report_id}-{f.idx}")
            cs = c2.selectbox("correct slot (if wrong_slot)", ["—"] + SLOT_CHOICES, key=f"s-{case}-{report_id}-{f.idx}")
            if v != "—":
                verdicts.append({"idx": f.idx, "finding_type": f.finding_type, "anchor": f.anchor[:210],
                                 "verdict": v, "correct_slot": None if cs == "—" else cs})
        missed = []
        if not judge_types:
            st.markdown(f"Findings the parser missed (up to {MISSED_ROWS}; one row per finding type, several slots allowed)")
            for i in range(MISSED_ROWS):
                c1, c2 = st.columns([1, 2])
                t = c1.selectbox(f"type {i + 1}", ["—"] + missed_types, key=f"mt-{case}-{report_id}-{i}")
                s = c2.multiselect(f"slots {i + 1}", SLOT_CHOICES, key=f"ms-{case}-{report_id}-{i}")
                if t != "—":
                    missed.append({"finding_type": t, "slots": s})
        rflags = st.multiselect("Report flags", REPORT_FLAGS, key=f"rf-{case}-{report_id}")
        if st.form_submit_button("Save this report's decisions"):
            rec = {"parse_run": a.parse_run.name, "lang": a.lang, "case_id": case, "report_id": report_id,
                   "half": half, "verdicts": verdicts, "missed": missed, "flags": rflags}
            if scope:
                rec["scope"] = list(scope)
            f = log_review(log_dir, rec)
            st.success(f"Saved ({len(verdicts)} verdicts, {len(missed)} missed, {len(rflags)} flags) → {f.name}")
            if strict and len(verdicts) < len(fr):
                st.warning(f"{len(fr) - len(verdicts)} finding(s) have no verdict yet. Choose one for each and "
                           "save again; the last saved record of a report is the one that counts.")


main()
