#!/usr/bin/env python3
"""
build_triage_sheet.py -- audit_report_facts.py's ERRORs, laid out to be read.

The audit prints what is mechanically wrong with stage 1. What it CANNOT say
is whether a finding is wrong, and for the not-in-text screen that gap is the
whole job: A018's report says "In the canine region of the 4th quadrant, the
presence of an endosseous implant is noted" and the extractor resolved that to
FDI 43, correctly, without the number ever being written. A083's "the upper
canine-incisor group" expanded to include 18, a molar, incorrectly. To a regex
those are the same finding. To a reader they are not, so this builds the page
a reader needs.

One row per finding: the FDI stage 1 asserted, the field it landed in, and the
sentences that most likely produced it -- for a not-in-text finding the tooth
number is by definition absent from the report, so the evidence to judge it by
is what the extractor was probably reading. Sentences are ordered by how close
their tooth numbers are to the asserted one, because a neighbour's sentence is
where a wrong number usually comes from, and the field's keyword and the FDIs
actually present are highlighted separately.

Decisions (keep / drop / unsure) are made in the browser and emitted as JSON;
nothing is written back to report_facts here. That is deliberate -- a drop
edits the ground truth, and it should be applied by a reviewed command, not by
a page that could be clicked shut.

Usage:
    python3 code/build_triage_sheet.py --split validate --out triage.html
    python3 code/build_triage_sheet.py --split training --out triage_train.html
"""
import argparse
import html
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from audit_report_facts import (audit_case, fdis_in_text,  # noqa: E402
                                report_text_for, positional_fdis)

# field -> what a sentence asserting it looks like. Measured against the corpus
# rather than guessed: fact_frequent_expression.txt tabulates how these 933
# reports actually phrase each finding, and the additions below are the
# sentences my first pass missed, counted over the real text --
#
#   periapical_lesions  210 sentences (radiolucen 88, radiopa 57,
#                       "post-extraction socket" 36, osteocondens 13,
#                       osteolytic 8). The corpus writes a lesion as a
#                       radiolucency far more often than as a "lesion", and
#                       its commonest negated form is "No definite osteolytic
#                       or osteocondensing lesions" (70x).
#   root_remnants       86 (post-extraction 82, "rr" 4). A remnant is usually
#                       described by how it got there, and "rr 15" is the
#                       corpus's own abbreviation (F014_2).
#
# fillings and caries needed nothing: "Conservative composite restoration" and
# "Destructive caries" were already covered.
FIELD_RE = {
    "crowns": r"prosthe|capsul|bridge|rehabilitat|cantilever|crown",
    "fillings": r"filling|restorat|conservative|amalgam|composite|reconstruct|black class",
    "caries": r"cari|decay|demineral",
    "endodontic": r"endodont|root canal|endocanal|obturat",
    "post_and_core": r"post[- ]and[- ]core|endocanal post|\bpost\b|stump|abutment",
    "root_remnants": r"remnant|residual|fragment|radicular|root retain|\brr\b"
                     r"|post-?extraction",
    "root_fractures": r"fractur",
    "periapical_lesions": r"periapical|granulom|apical|osteorarefaction|rarefact"
                          r"|lesion|cyst|osteoly|osteocondens|osteosclerot"
                          r"|radiolucen|radiopa|socket",
    "implants": r"implant|osseointegr|osteointegr|fixture",
    "unerupted": r"impact|inclus|unerupt|erupt|disodont|dysodont|mesiovert|semi",
    "primary_teeth": r"deciduous|primary|milk",
    "uncertain_teeth": r"cannot|unclear|not assessable|superimpos|doubt|artifact",
    "teeth_present": r"present|visible|in the arch|complete dentition",
    "teeth_absent": r"absen|missing|edentul|agenes",
    "canal_right": r"canal", "canal_left": r"canal",
    "sinus_right": r"sinus", "sinus_left": r"sinus",
    "bridges": r"bridge|pontic|span",
}

# 15% of the corpus's finding-shaped sentences DENY the finding -- "No evident
# carious lesions" (9x), "No definite osteolytic or osteocondensing lesions"
# (70x), "No prosthetic restorations or endosseous implants are present". A
# keyword match alone cannot tell those from an assertion, and a reviewer
# skimming for the keyword would read a denial as support. They are still shown
# -- a denial is evidence, and often the decisive kind -- but marked.
NEGATION_RE = re.compile(
    r"\bno\b|\bnor\b|\bnot\b|without|free of|negative for|absence of (?:evident|any)"
    r"|are (?:not )?appreciab|cannot be", re.I)
FIELD_LABEL = {
    "crowns": "crown", "fillings": "filling", "caries": "caries",
    "endodontic": "root canal", "post_and_core": "post & core",
    "root_remnants": "root remnant", "root_fractures": "root fracture",
    "periapical_lesions": "periapical lesion", "implants": "implant",
    "canal_right": "canal (right)", "canal_left": "canal (left)",
    "sinus_right": "sinus (right)", "sinus_left": "sinus (left)",
    # An arch-range row is not a finding on a tooth at all: a presence range
    # sweeps this position and teeth_present does not contain it. Naming it
    # after its field printed "arch range . arch range" and told the reader
    # nothing.
    "arch range": "inside a stated presence range",
    "present/absent": "in both presence lists",
}


def split_sentences(text):
    guarded = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", text)
    guarded = re.sub(r"\b([a-zA-Z])\.(?=[a-zA-Z]\.|\s*\d)", r"\1<DOT>", guarded)
    return [s.replace("<DOT>", ".").strip()
            for s in re.split(r"(?<=[.;:])\s+", guarded) if s.strip()]


def esc(t):
    return html.escape(str(t))


def highlight(sentence, fdi, pattern):
    out = esc(sentence)
    out = re.sub(rf"\b({fdi}|{str(fdi)[0]}\.{str(fdi)[1]})\b",
                 r'<mark class="fdi">\1</mark>', out)
    if pattern:
        out = re.sub(f"({pattern})", r'<mark class="kw">\1</mark>', out, flags=re.I)
    return out



def evidence_badges(case, arch, base, fdi, own_report, reports_dir, own_text):
    """What a reviewer would otherwise have to go and look up.

    Two facts settle most of these rows and neither is visible in the row
    itself: whether the tooth is named POSITIONALLY somewhere ("the first,
    second and third molars bilaterally", "the canine region of the 4th
    quadrant"), and whether ANOTHER radiologist named it in a sentence about
    the same finding. Both are computed here so the page carries them.
    """
    badges = []
    pattern = FIELD_RE.get(base, "")
    for s in split_sentences(own_text):
        if fdi in positional_fdis(s, arch):
            badges.append(("pos", "named by position", s))
            break
    for path in sorted(reports_dir.glob(f"{case}_*.txt")):
        if path.name == own_report:
            continue
        for s in split_sentences(path.read_text(encoding="utf-8")):
            flat = re.sub(r"(\d)\.(\d)", r"\1\2", s)
            named = bool(re.search(rf"\b{fdi}\b", flat)) or fdi in positional_fdis(s, arch)
            if named and (not pattern or re.search(pattern, s, re.I)):
                badges.append(("other", f"also in {path.name}", s))
                break
        if any(b[0] == "other" for b in badges):
            break
    return badges



def resolution_note(gt_dir, stem, arch, fdi):
    """How stage 2 currently resolves this position, in words.

    A contradiction row shows one tooth and the label "present/absent", which
    says the two lists disagree but not which one the reader is being asked
    about. resolve_presence has already picked a side -- an explicit absence
    outranks the presence list, a per-tooth finding outranks both -- so the row
    can say what the ground truth currently asserts and what changing it would
    change.
    """
    path = gt_dir / f"{stem}_gt.json"
    if not path.exists():
        return None
    gt = json.loads(path.read_text())
    value = ((gt.get("global", {}).get(f"dental_arch_findings_{arch}") or {})
             .get("findings") or {}).get(str(fdi))
    if value is None:
        return None
    if value == "absent":
        return ("absent", "The ground truth currently says this position is "
                          "ABSENT. Choosing present drops it from teeth_absent.")
    return ("present", f"The ground truth currently says this position is "
                       f"PRESENT and reads \u201c{value}\u201d. Choosing absent "
                       f"drops it from teeth_present and removes that finding.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validate")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    gt_dir = ROOT / f"dataset/{a.split}/outputs/ground_truth"
    reports_dir = ROOT / f"dataset/{a.split}/reports"

    by_case = defaultdict(list)
    reports_by_case = defaultdict(dict)
    counts = Counter()

    for path in sorted(gt_dir.glob("*_report_facts.json")):
        stem = path.name.replace("_report_facts.json", "")
        case = re.match(r"([A-Z]+\d+)", stem).group(1)
        # Every report the case has, because that is what the consensus file
        # answers to -- and because a reviewer judging "the report never names
        # this tooth" needs to see the report that does.
        text = report_text_for(stem, case, reports_dir)
        if not text:
            continue
        own = reports_dir / f"{stem}.txt"
        for rp in ([own] if own.exists() else sorted(reports_dir.glob(f"{case}_*.txt"))):
            reports_by_case[case][rp.name] = rp.read_text(encoding="utf-8")
        facts = json.loads(path.read_text())
        for level, screen, message in audit_case(
                case, facts, text,
                {"not-in-text", "contradiction", "arch-range"}, stem=stem):
            if level != "ERROR":
                continue
            rows = []
            m = re.search(r"^(\w+)\.([\w.]+) asserts (\d+)", message)
            m2 = re.search(r"^(\w+): (\d+) is in BOTH", message)
            m3 = re.search(r"^(\w+): \"(.+)\" sweeps the arch and teeth_present "
                           r"is missing \[([\d, ]+)\]", message)
            if m:
                rows = [(m.group(1), m.group(2), int(m.group(3)))]
            elif m2:
                rows = [(m2.group(1), "present/absent", int(m2.group(2)))]
            elif m3:
                # One row per tooth the range covers and the extraction lost:
                # keep says the stated range wins, drop says the absence does.
                rows = [(m3.group(1), "arch range", int(x))
                        for x in m3.group(3).split(",")]
            for arch, field, fdi in rows:
                base = field.split(".")[0]
                pattern = FIELD_RE.get(base, "")
                if field == "arch range":
                    # Quote the range sentence itself. Without this the row
                    # showed "No sentence in this report mentions this finding"
                    # under a heading built FROM a sentence in that report.
                    quoted = re.search(r'"(.+?)"', message)
                    if quoted:
                        pattern = re.escape(quoted.group(1)[:40])
                    # Denials are skipped, not shown. 15% of the corpus's
                # finding-shaped sentences rule the finding OUT ("No definite
                # osteolytic or osteocondensing lesions", 70x), and a sentence
                # that denies a finding is not evidence for the tooth number the
                # extractor chose -- it is noise in a list whose whole job is to
                # show what the extractor was probably reading.
                hits = [s for s in split_sentences(text)
                        if pattern and re.search(pattern, s, re.I)
                        and not NEGATION_RE.search(s)]
                # A sentence naming a NEIGHBOURING tooth is the usual source of a
                # wrong number, so surface the closest-numbered ones first.
                hits.sort(key=lambda s: min([abs(f - fdi) for f in fdis_in_text(s)] or [99]))
                by_case[case].append({
                    "id": f"{stem}-{arch}-{field}-{fdi}".replace(".", "_"),
                    "file": path.name, "screen": screen, "arch": arch,
                    "field": field, "base": base, "fdi": fdi,
                    "label": FIELD_LABEL.get(base, base),
                    "sentences": [highlight(s, fdi, pattern) for s in hits[:3]],
                    "badges": evidence_badges(case, arch, base, fdi, own.name
                                              if own.exists() else None,
                                              reports_dir, text),
                    "resolution": (resolution_note(gt_dir, stem, arch, fdi)
                                   if field == "present/absent" else None),
                })
                counts[screen] += 1
                counts[base] += 1

    total = sum(len(v) for v in by_case.values())
    nav = "".join(
        f'<a href="#{c}"><b>{esc(c)}</b><span>{len(by_case[c])}</span></a>'
        for c in sorted(by_case))

    cards = []
    for case in sorted(by_case):
        reports = "".join(
            f'<details class="rep"><summary>{esc(name)}</summary><p>{esc(text)}</p></details>'
            for name, text in sorted(reports_by_case[case].items()))
        items = []
        for it in sorted(by_case[case], key=lambda x: (x["arch"], x["base"], x["fdi"])):
            chip = {"contradiction": "contradiction",
                    "arch-range": "range vs absence"}.get(it["screen"], "not in text")
            cls = "bad" if it["screen"] == "contradiction" else "warn"
            badges = "".join(
                f'<p class="badge {kind}"><b>{esc(label)}</b> {highlight(s, it["fdi"], "")}</p>'
                for kind, label, s in it.get("badges", []))
            if it["field"] == "present/absent":
                # keep/drop cannot express this row: the question is which of
                # the two lists is wrong, and "keep" would mean keep the
                # contradiction. It asks for the answer instead.
                acts = ('<button data-v="present">present</button>'
                        '<button data-v="absent">absent</button>'
                        '<button data-v="unsure">unsure</button>')
            else:
                acts = ('<button data-v="keep">keep</button>'
                        '<button data-v="drop">drop</button>'
                        '<button data-v="unsure">unsure</button>'
                        '<button data-v="unlocate" title="the finding is real, '
                        'the region is known, the tooth is not">&rarr; unlocated</button>'
                        '<input class="to" type="number" min="11" max="85" '
                        'placeholder="&rarr; FDI" title="the finding is real but '
                        'on a different tooth">')
            res = it.get("resolution")
            res_html = (f'<p class="resolution {res[0]}">{esc(res[1])}</p>'
                        if res else "")
            ev = res_html + badges + "".join(f'<p class="sent">{s}</p>'
                                             for s in it["sentences"]) or \
                '<p class="nosent">No sentence in this report mentions this finding at all.</p>'
            items.append(f"""
    <div class="item" data-screen="{it['screen']}" data-id="{esc(it['id'])}" data-fdi="{it['fdi']}">
      <div class="itop">
        <b class="fdi">{it['fdi']}</b>
        <span class="chip {cls}">{chip}</span>
        <span class="field">{esc(it['arch'])}.{esc(it['field'])} &middot; {esc(it['label'])}</span>
        <div class="acts">{acts}</div>
      </div>
      {ev}
    </div>""")
        cards.append(f"""
<article class="case" id="{esc(case)}">
  <header class="chead"><h3>{esc(case)}</h3>
    <span class="cnt">{len(by_case[case])} finding(s)</span></header>
  <div class="reps">{reports}</div>
  {''.join(items)}
</article>""")

    page = f"""<title>Numbers the reports never wrote</title>
<style>
:root {{
  --bg:#f5f6f7; --surface:#fff; --surface-2:#eceef0; --line:#d7dbde;
  --ink:#111619; --muted:#556069; --faint:#87919a; --accent:#1c5cab;
  --ok:#2c6e4b; --ok-soft:#dcefe3; --warn:#8a5a00; --warn-soft:#f7edd7;
  --bad:#a3352f; --bad-soft:#f6e2e0; --kw:#d9e9fb; --fdi:#ffe9a8;
  --llm:#6b3fa0; --llm-soft:#ece4f6;
  --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  --serif:ui-serif,Georgia,"Iowan Old Style",serif;
}}
@media (prefers-color-scheme:dark) {{ :root:not([data-theme="light"]) {{
  --bg:#0d1114; --surface:#161b1f; --surface-2:#1e252b; --line:#2b343b;
  --ink:#e6ebef; --muted:#98a3ac; --faint:#6f7a83; --accent:#6da7ec;
  --ok:#6cc794; --ok-soft:#152f20; --warn:#dda94f; --warn-soft:#33270f;
  --bad:#e08079; --bad-soft:#331a18; --kw:#1c3550; --fdi:#4a3c12;
  --llm:#b69ae0; --llm-soft:#241a33;
}} }}
:root[data-theme="dark"] {{
  --bg:#0d1114; --surface:#161b1f; --surface-2:#1e252b; --line:#2b343b;
  --ink:#e6ebef; --muted:#98a3ac; --faint:#6f7a83; --accent:#6da7ec;
  --ok:#6cc794; --ok-soft:#152f20; --warn:#dda94f; --warn-soft:#33270f;
  --bad:#e08079; --bad-soft:#331a18; --kw:#1c3550; --fdi:#4a3c12;
  --llm:#b69ae0; --llm-soft:#241a33;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font-family:var(--sans);
  font-size:15px; line-height:1.55; }}
.wrap {{ max-width:1060px; margin:0 auto; padding:40px 24px 120px; }}
.eyebrow {{ font:600 .72rem/1 var(--mono); letter-spacing:.14em; text-transform:uppercase;
  color:var(--accent); margin:0 0 12px; }}
h1 {{ font-size:clamp(1.7rem,3.6vw,2.4rem); line-height:1.12; margin:0 0 10px;
  letter-spacing:-.02em; }}
.lede {{ color:var(--muted); max-width:68ch; margin:0 0 24px; }}
.stats {{ display:flex; flex-wrap:wrap; border:1px solid var(--line); border-radius:11px;
  overflow:hidden; background:var(--surface); margin:0 0 18px; }}
.stat {{ flex:1 1 120px; padding:13px 17px; border-right:1px solid var(--line); }}
.stat:last-child {{ border-right:0; }}
.stat b {{ display:block; font:600 1.45rem/1.1 var(--mono); font-variant-numeric:tabular-nums; }}
.stat span {{ font-size:.77rem; color:var(--muted); }}
.callout {{ border-left:3px solid var(--accent); background:var(--surface); padding:13px 17px;
  border-radius:0 8px 8px 0; margin:0 0 12px; }}
.callout h4 {{ margin:0 0 4px; font-size:.9rem; }}
.callout p {{ margin:0; color:var(--muted); font-size:.88rem; }}
.bar {{ position:sticky; top:0; z-index:5; background:var(--bg); padding:12px 0;
  border-bottom:1px solid var(--line); display:flex; gap:9px; flex-wrap:wrap;
  align-items:center; margin-bottom:20px; }}
.bar button {{ font:500 .8rem/1 var(--sans); padding:8px 13px; border-radius:99px;
  border:1px solid var(--line); background:var(--surface); color:var(--muted); cursor:pointer; }}
.bar button[aria-pressed="true"] {{ background:var(--accent); border-color:var(--accent);
  color:var(--bg); }}
#emit {{ background:var(--ink); color:var(--bg); border-color:var(--ink); }}
.bar .prog {{ margin-left:auto; font:600 .82rem/1 var(--mono); color:var(--muted);
  font-variant-numeric:tabular-nums; }}
nav.cases {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(94px,1fr)); gap:7px;
  margin:0 0 30px; }}
nav.cases a {{ display:flex; justify-content:space-between; padding:8px 11px; gap:6px;
  border:1px solid var(--line); border-radius:8px; background:var(--surface);
  text-decoration:none; color:var(--ink); font:500 .82rem/1 var(--mono); }}
nav.cases a span {{ color:var(--faint); }}
nav.cases a:hover {{ border-color:var(--accent); }}
.case {{ background:var(--surface); border:1px solid var(--line); border-radius:12px;
  padding:18px 20px; margin:0 0 16px; scroll-margin-top:64px; }}
.case[hidden] {{ display:none; }}
.chead {{ display:flex; align-items:baseline; gap:12px; border-bottom:2px solid var(--ink);
  padding-bottom:7px; margin-bottom:12px; }}
.chead h3 {{ margin:0; font:600 1.1rem/1 var(--mono); }}
.cnt {{ font-size:.78rem; color:var(--muted); margin-left:auto; }}
.rep summary {{ cursor:pointer; font:500 .78rem/1 var(--mono); color:var(--accent);
  padding:6px 0; }}
.rep p {{ font-family:var(--serif); font-size:.9rem; background:var(--surface-2);
  padding:12px 14px; border-radius:8px; margin:6px 0 0; white-space:pre-wrap; }}
.item {{ border-top:1px solid var(--line); padding:11px 0; }}
.item[hidden] {{ display:none; }}
.itop {{ display:flex; align-items:center; gap:9px; flex-wrap:wrap; }}
.fdi {{ font:600 .95rem/1 var(--mono); min-width:2.2em; }}
.field {{ font:400 .74rem/1.4 var(--mono); color:var(--faint); }}
.chip {{ font:600 .64rem/1 var(--mono); padding:4px 7px; border-radius:99px;
  letter-spacing:.05em; text-transform:uppercase; }}
.chip.warn {{ background:var(--warn-soft); color:var(--warn); }}
.chip.bad {{ background:var(--bad-soft); color:var(--bad); }}
.acts {{ margin-left:auto; display:flex; gap:4px; }}
.acts button {{ font:500 .7rem/1 var(--sans); padding:5px 9px; border-radius:6px;
  border:1px solid var(--line); background:var(--surface); color:var(--muted); cursor:pointer; }}
.acts .to {{ width:5.6em; font:500 .7rem/1 var(--mono); padding:5px 7px; border-radius:6px;
  border:1px solid var(--line); background:var(--surface); color:var(--ink); }}
.acts .to:not(:placeholder-shown) {{ background:var(--ok-soft); color:var(--ok);
  border-color:var(--ok); }}
.acts button[aria-pressed="true"][data-v="keep"] {{ background:var(--ok-soft); color:var(--ok);
  border-color:var(--ok); }}
.acts button[aria-pressed="true"][data-v="drop"] {{ background:var(--bad-soft); color:var(--bad);
  border-color:var(--bad); }}
.acts button[aria-pressed="true"][data-v="unsure"] {{ background:var(--warn-soft);
  color:var(--warn); border-color:var(--warn); }}
.acts button[aria-pressed="true"][data-v="present"] {{ background:var(--ok-soft);
  color:var(--ok); border-color:var(--ok); }}
.acts button[aria-pressed="true"][data-v="absent"] {{ background:var(--bad-soft);
  color:var(--bad); border-color:var(--bad); }}
.resolution {{ font-size:.85rem; margin:7px 0 0; padding:8px 11px; border-radius:8px;
  background:var(--surface-2); color:var(--muted); }}
.resolution b {{ color:var(--ink); }}
.acts button[aria-pressed="true"][data-v="unlocate"] {{ background:var(--llm-soft);
  color:var(--llm); border-color:var(--llm); }}
.sent {{ font-family:var(--serif); font-size:.9rem; margin:7px 0 0; }}
.badge {{ font-family:var(--serif); font-size:.88rem; margin:7px 0 0; padding:8px 11px;
  border-radius:8px; background:var(--ok-soft); }}
.badge b {{ font:600 .62rem/1 var(--mono); letter-spacing:.06em; text-transform:uppercase;
  color:var(--ok); margin-right:7px; }}
.badge.other {{ background:var(--llm-soft); }}
.badge.other b {{ color:var(--llm); }}
.nosent {{ color:var(--faint); font-size:.84rem; margin:7px 0 0; }}
.sent.neg {{ color:var(--muted); }}
.negtag {{ font:600 .62rem/1 var(--mono); letter-spacing:.06em; text-transform:uppercase;
  background:var(--bad-soft); color:var(--bad); padding:3px 6px; border-radius:99px;
  margin-right:6px; vertical-align:1px; }}
mark.kw {{ background:var(--kw); color:inherit; padding:0 2px; border-radius:3px; }}
mark.fdi {{ background:var(--fdi); color:inherit; padding:0 2px; border-radius:3px;
  font-weight:600; }}
dialog {{ border:1px solid var(--line); border-radius:12px; background:var(--surface);
  color:var(--ink); max-width:min(860px,92vw); padding:0; }}
dialog::backdrop {{ background:rgba(0,0,0,.5); }}
dialog header {{ padding:14px 18px; border-bottom:1px solid var(--line); display:flex;
  justify-content:space-between; align-items:center; }}
dialog pre {{ margin:0; padding:16px 18px; overflow:auto; max-height:60vh;
  font:400 .78rem/1.5 var(--mono); }}
button:focus-visible, a:focus-visible, summary:focus-visible {{ outline:2px solid var(--accent);
  outline-offset:2px; }}
</style>
<div class="wrap">
<p class="eyebrow">audit_report_facts &middot; {esc(a.split)} split &middot; stage 1</p>
<h1>Numbers the reports never wrote</h1>
<p class="lede">Every tooth number stage&nbsp;1 asserted that its report does not contain — after
ranges are expanded and after presence, pontics and unerupted teeth are excused, since a report
states those wholesale. What is left is a finding hung on a tooth nobody named. The sentences
below each row are the ones that most likely produced it: the field's own keyword highlighted
in <mark class="kw">blue</mark>, the tooth numbers actually present in
<mark class="fdi">yellow</mark>. The asserted FDI is by definition not among them.</p>

<div class="stats">
  <div class="stat"><b>{total}</b><span>findings to judge</span></div>
  <div class="stat"><b>{counts['not-in-text']}</b><span>unnamed tooth</span></div>
  <div class="stat"><b>{counts['contradiction']}</b><span>present AND absent</span></div>
  <div class="stat"><b>{counts['fillings']}</b><span>on fillings — the worst field</span></div>
  <div class="stat"><b>{len(by_case)}</b><span>cases affected</span></div>
  <div class="stat"><b id="done">0</b><span>decided here</span></div>
</div>

<div class="callout">
  <h4>What the three decisions mean</h4>
  <p><b>keep</b> — the report does support it, by a group name ("the upper canine-incisor group"),
  by a range, or by a phrasing the scan missed. <b>drop</b> — the extractor supplied a number the
  radiologist never did; it comes out of report_facts. <b>unsure</b> — the sentence is genuinely
  ambiguous, which is worth recording rather than guessing. <b>&rarr; FDI</b> — the finding is
  real but sits on a different tooth: A022's "root remnant in the canine region of the 2nd
  quadrant" is 23, and stage 1 wrote 24. That is the commonest shape here, and it is neither a
  keep nor a drop.</p>
</div>

<div class="bar">
  <button data-f="all" aria-pressed="true">All</button>
  <button data-f="undecided" aria-pressed="false">Undecided</button>
  <button data-f="not-in-text" aria-pressed="false">Unnamed tooth</button>
  <button data-f="contradiction" aria-pressed="false">Present &amp; absent</button>
  <button id="emit">Build the block</button>
  <span class="prog"><span id="pct">0</span>%</span>
</div>

<nav class="cases">{nav}</nav>
{''.join(cards)}
</div>
<dialog id="out"><header><h3>Decisions</h3>
  <button onclick="document.getElementById('out').close()">close</button></header>
  <pre id="outpre"></pre></dialog>
<script>
const items = [...document.querySelectorAll('.item')];
const decisions = {{}};
function refresh() {{
  const n = Object.keys(decisions).length;
  document.getElementById('done').textContent = n;
  document.getElementById('pct').textContent = Math.round(100 * n / items.length);
}}
// A present/absent row has no "-> FDI" input, and reading .to on it returns
// null. Setting a property on that null threw during BINDING, which killed the
// forEach for every row after it and the filter and emit handlers below with
// it -- the whole page went dead to clicks. Guard both uses.
items.forEach(it => {{
  const input = it.querySelector('.to');
  it.querySelectorAll('.acts button').forEach(b => {{
    b.onclick = () => {{
      const on = b.getAttribute('aria-pressed') === 'true';
      it.querySelectorAll('.acts button').forEach(o => o.setAttribute('aria-pressed', 'false'));
      if (input) input.value = '';
      if (on) {{ delete decisions[it.dataset.id]; }}
      else {{
        b.setAttribute('aria-pressed', 'true');
        decisions[it.dataset.id] = {{verdict: b.dataset.v, fdi: Number(it.dataset.fdi)}};
      }}
      refresh();
    }};
  }});
  // A relabel is its own verdict, not a keep with a note: the finding is real
  // and the tooth number is not, which is the commonest shape here (A022's
  // "root remnant in the canine region of the 2nd quadrant" -> 23, not 24).
  if (input) input.oninput = () => {{
    it.querySelectorAll('.acts button').forEach(o => o.setAttribute('aria-pressed', 'false'));
    const to = Number(input.value);
    if (to >= 11 && to <= 85) {{
      decisions[it.dataset.id] = {{verdict: 'relabel', fdi: Number(it.dataset.fdi), to}};
    }} else {{
      delete decisions[it.dataset.id];
    }}
    refresh();
  }};
}});
document.querySelectorAll('.bar button[data-f]').forEach(b => {{
  b.onclick = () => {{
    document.querySelectorAll('.bar button[data-f]').forEach(o =>
      o.setAttribute('aria-pressed', String(o === b)));
    const f = b.dataset.f;
    items.forEach(it => {{
      const undecided = !decisions[it.dataset.id];
      it.hidden = !(f === 'all' || (f === 'undecided' && undecided) || it.dataset.screen === f);
    }});
    document.querySelectorAll('.case').forEach(c =>
      c.hidden = ![...c.querySelectorAll('.item')].some(i => !i.hidden));
  }};
}});
document.getElementById('emit').onclick = () => {{
  const out = {{keep: [], drop: [], unsure: [], unlocate: [],
                present: [], absent: [], relabel: {{}}}};
  for (const [id, d] of Object.entries(decisions)) {{
    if (d.verdict === 'relabel') out.relabel[id] = {{from: d.fdi, to: d.to}};
    else out[d.verdict].push(id);
  }}
  document.getElementById('outpre').textContent = JSON.stringify(out, null, 2);
  document.getElementById('out').showModal();
}};
</script>
"""
    Path(a.out).write_text(page, encoding="utf-8")
    print(f"[OK] {a.out}  {total} findings across {len(by_case)} cases")


if __name__ == "__main__":
    main()
