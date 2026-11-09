"""Report review app: read each report next to its parsed findings (explore by default, optional logging).

Runs ONLY on a compute node via jobs/review_app.sbatch (never on the login node), opened in the browser through
an SSH tunnel. Report text is shown on screen and never written anywhere. Optional logs hold IDs and codes only.
Blinded: Lumetti's labels and other parses are not shown. Access needs the token from the job's token file.

    REVIEW_TOKEN=... REVIEW_PARSE_RUN=results/extract/<run> REVIEW_MANIFEST=results/manifest/<run> \
        [REVIEW_LANG=en] [REVIEW_LOG_ROOT=results/review] streamlit run src/partialref/review/app.py
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
    finding_rows,
    load_parse,
    load_reports,
    log_review,
    read_log,
)

PERMANENT = [f"{q}{i}" for q in (1, 2, 3, 4) for i in range(1, 9)]
SLOT_CHOICES = PERMANENT + ["Left", "Right", "QuadrantI", "QuadrantII", "QuadrantIII", "QuadrantIV",
                            "UpperArch", "LowerArch"]


def args() -> argparse.Namespace:
    """Settings from environment variables (set by jobs/review_app.sbatch; tests set them directly)."""
    env = os.environ
    for k in ("REVIEW_PARSE_RUN", "REVIEW_MANIFEST"):
        if not env.get(k):
            st.error(f"{k} is not set: start the app with jobs/review_app.sbatch.")
            st.stop()
    return argparse.Namespace(parse_run=Path(env["REVIEW_PARSE_RUN"]), manifest=Path(env["REVIEW_MANIFEST"]),
                              lang=env.get("REVIEW_LANG", "en"), log_root=Path(env.get("REVIEW_LOG_ROOT", "results/review")))


def gate() -> None:
    """Token check (the job writes the token to an owner-only file); without a token set, refuse to run."""
    if os.environ.get("REVIEW_LOCAL") == "1":      # own machine via partialref.review.local (127.0.0.1 only)
        return
    token = os.environ.get("REVIEW_TOKEN", "")
    if not token:
        st.error("REVIEW_TOKEN is not set: start the app with jobs/review_app.sbatch.")
        st.stop()
    if not st.session_state.get("ok"):
        entered = st.text_input("Access token", type="password")
        if entered and hmac.compare_digest(entered, token):
            st.session_state["ok"] = True
            st.rerun()
        st.stop()


def main() -> None:
    st.set_page_config(page_title="Report review", layout="wide")
    gate()
    a = args()
    reports = load_reports(a.manifest, a.lang)
    log_dir = a.log_root / a.parse_run.name

    with st.sidebar:
        st.markdown(f"**Parse run** `{a.parse_run.name}`  \n**Language** {a.lang}")
        half = st.selectbox("Cases", ["tune", "heldout", "all"],
                            help="tune: errors may change the prompt. heldout: only count, never tune on it.")
        centres = st.multiselect("Centres", ["P", "F", "S", "A"], default=["P", "F", "S"])
        only_multi = st.checkbox("Only cases with ≥ 2 reports", value=False)
        seed = st.number_input("Shuffle seed", value=0, step=1)
        order = case_order(reports, half, centres, int(seed), only_multi)
        if not order:
            st.warning("No cases for this filter.")
            st.stop()
        if "pos" not in st.session_state or st.session_state.get("order_key") != (half, tuple(centres), seed, only_multi):
            st.session_state.update(pos=0, order_key=(half, tuple(centres), seed, only_multi))

        def go_to() -> None:                       # applied once, when the box changes
            cid = st.session_state.get("jump", "").strip()
            if cid in order:
                st.session_state["pos"] = order.index(cid)

        st.text_input("Go to case ID (optional)", key="jump", on_change=go_to)
        c1, c2 = st.columns(2)
        if c1.button("◀ Previous", width="stretch"):
            st.session_state["pos"] = max(0, st.session_state["pos"] - 1)
        if c2.button("Next ▶", width="stretch"):
            st.session_state["pos"] = min(len(order) - 1, st.session_state["pos"] + 1)
        st.caption(f"Case {st.session_state['pos'] + 1} of {len(order)}")
        show_it = st.checkbox("Also show the Italian original", value=False)
        log_on = st.checkbox("Enable logging", value=False,
                             help="Off = explore only, nothing is saved. On = a log form appears under each report.")
        logged = read_log(log_dir)
        st.caption(f"Logged so far: {logged['report_id'].nunique() if len(logged) else 0} reports")

    case = order[st.session_state["pos"]]
    rep = reports[reports["case_id"] == case]
    st.markdown(f"## {case}  ·  centre {case[0]}  ·  {len(rep)} report(s)  ·  half: {rep['half'].iloc[0]}")
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
                flags = {k: m.get(k) for k in ("salvaged", "dropped_findings", "absence_conflicts", "attempts_made")
                         if m.get(k)}
                if flags:
                    st.caption("Parser notes: " + ", ".join(f"{k}={v}" for k, v in flags.items()))
                if log_on:
                    log_form(log_dir, a, case, r.report_id, rep["half"].iloc[0], fr)


def log_form(log_dir: Path, a: argparse.Namespace, case: str, report_id: str, half: str, fr: pd.DataFrame) -> None:
    with st.form(key=f"log-{case}-{report_id}"):
        st.markdown("**Log a decision (optional)** — leave a finding at '—' to skip it")
        verdicts = []
        for f in fr.itertuples():
            c1, c2 = st.columns([2, 1])
            v = c1.selectbox(f"{f.finding_type} · {f.anchor}", ["—"] + VERDICTS, key=f"v-{case}-{report_id}-{f.idx}")
            cs = c2.selectbox("correct slot (if wrong_slot)", ["—"] + SLOT_CHOICES, key=f"s-{case}-{report_id}-{f.idx}")
            if v != "—":
                verdicts.append({"idx": f.idx, "finding_type": f.finding_type, "anchor": f.anchor[:100],
                                 "verdict": v, "correct_slot": None if cs == "—" else cs})
        st.markdown("Findings the parser missed (up to 3)")
        missed = []
        for i in range(3):
            c1, c2 = st.columns([1, 2])
            t = c1.selectbox(f"type {i + 1}", ["—"] + [x.value for x in FindingType], key=f"mt-{case}-{report_id}-{i}")
            s = c2.multiselect(f"slots {i + 1}", SLOT_CHOICES, key=f"ms-{case}-{report_id}-{i}")
            if t != "—":
                missed.append({"finding_type": t, "slots": s})
        rflags = st.multiselect("Report flags", REPORT_FLAGS, key=f"rf-{case}-{report_id}")
        if st.form_submit_button("Save this report's decisions"):
            f = log_review(log_dir, {"parse_run": a.parse_run.name, "lang": a.lang, "case_id": case,
                                     "report_id": report_id, "half": half, "verdicts": verdicts,
                                     "missed": missed, "flags": rflags})
            st.success(f"Saved ({len(verdicts)} verdicts, {len(missed)} missed, {len(rflags)} flags) → {f.name}")


main()
