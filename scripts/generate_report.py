#!/usr/bin/env python3
"""generate_report.py — Generate self-contained HTML report from progress.json + analysis.json.

Usage:
    python generate_report.py \
        --date          20260318_220000 \
        --commit        abc123 \
        --report-dir    _gh_pages/nightly/20260318_220000 \
        --html-output   report.html \
        --progress-json progress.json \
        --analysis-json analysis.json
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ──────────────────────────────────────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────────────────────────────────────

_CSS = """
:root{--green:#22c55e;--yellow:#f59e0b;--red:#ef4444;--blue:#3b82f6;--purple:#8b5cf6;
      --bg:#f9fafb;--card:#fff;--border:#e5e7eb;--text:#111827;--muted:#6b7280}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,sans-serif;background:var(--bg);color:var(--text);padding:1.5rem;max-width:1100px;margin:0 auto}
h1{font-size:1.5rem;font-weight:700;margin-bottom:.25rem}
h2{font-size:1.15rem;font-weight:600;margin:1.5rem 0 .75rem;border-bottom:2px solid var(--border);padding-bottom:.25rem}
h3{font-size:.95rem;font-weight:600;margin:.75rem 0 .25rem}
.meta{color:var(--muted);font-size:.82rem;margin-bottom:1.5rem}
.card{background:var(--card);border:1px solid var(--border);border-radius:.5rem;padding:1rem 1.25rem;margin-bottom:1rem}
/* Overall progress */
.overall-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:.75rem;margin:.75rem 0}
.stat-box{background:var(--card);border:1px solid var(--border);border-radius:.375rem;padding:.6rem;text-align:center}
.stat-box .num{font-size:1.4rem;font-weight:700;line-height:1}
.stat-box .lbl{font-size:.72rem;color:var(--muted);margin-top:.15rem}
/* Progress bars */
.bar-wrap{background:#e5e7eb;border-radius:9999px;height:.6rem;overflow:hidden;margin:.2rem 0;box-shadow:inset 0 1px 2px rgba(0,0,0,.06)}
.bar-fill{height:100%;border-radius:9999px;transition:width .6s ease;position:relative}
.bar-impl{background:linear-gradient(90deg,#60a5fa,#3b82f6)}
.bar-test{background:linear-gradient(90deg,#4ade80,#22c55e)}
.bar-bench{background:linear-gradient(90deg,#a78bfa,#8b5cf6)}
.bar-row{display:flex;align-items:center;gap:.5rem;font-size:.78rem;margin:.15rem 0}
.bar-label{width:50px;color:var(--muted);flex-shrink:0}
.bar-container{flex:1;min-width:80px}
.bar-count{width:55px;text-align:right;color:var(--muted);flex-shrink:0}
/* Category card */
.cat-card{display:grid;grid-template-columns:1fr auto;gap:.75rem;padding:.75rem 0;border-bottom:1px solid var(--border)}
.cat-card:last-child{border-bottom:none}
.cat-bars{min-width:0}
.cat-scores{display:flex;gap:.4rem;align-items:flex-start;flex-shrink:0}
.score-badge{display:inline-flex;flex-direction:column;align-items:center;padding:.25rem .4rem;
             border-radius:.25rem;font-size:.7rem;font-weight:600;min-width:42px}
.score-badge .val{font-size:1rem;line-height:1}
.score-perf{background:#ede9fe;color:#5b21b6}.score-func{background:#dcfce7;color:#166534}
.score-null{background:#f3f4f6;color:#9ca3af}
/* Detail section */
.detail-item{margin:.75rem 0;padding:.75rem;background:#fafafa;border-radius:.375rem;border:1px solid var(--border)}
.detail-item h3{margin:0 0 .35rem;font-size:.88rem}
.detail-issues{color:var(--red);font-size:.82rem;margin:.25rem 0}
.detail-eval{font-size:.82rem;color:var(--text);margin:.25rem 0}
/* Overall assessment */
.rec-list{margin:.5rem 0;padding-left:1.25rem}
.rec-list li{margin:.25rem 0;font-size:.85rem}
"""


# ──────────────────────────────────────────────────────────────────────────────
# HTML builders
# ──────────────────────────────────────────────────────────────────────────────

def _bar_html(pct: float, css_class: str) -> str:
    return (
        f'<div class="bar-wrap">'
        f'<div class="bar-fill {css_class}" style="width:{pct:.1f}%"></div>'
        f'</div>'
    )


def _score_badge(value, label: str, css_class: str) -> str:
    if value is None:
        return f'<div class="score-badge score-null"><span class="val">—</span>{label}</div>'
    return f'<div class="score-badge {css_class}"><span class="val">{value}</span>{label}</div>'


def build_html(args, prog: dict | None, analysis: dict | None) -> str:
    date_str = args.date
    commit   = args.commit
    gen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if analysis is None:
        analysis = {"categories": {}, "overall": {"summary": "No analysis available.", "recommendations": []}}

    parts = [
        "<!DOCTYPE html><html lang='zh-CN'><head>",
        "<meta charset='UTF-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        f"<title>TileOPs Report · {_esc(date_str)}</title>",
        f"<style>{_CSS}</style>",
        "</head><body>",
    ]

    # ── Section 1: Header ────────────────────────────────────────────────────
    parts += [
        "<h1>TileOPs Nightly Report</h1>",
        f"<p class='meta'>Date: {_esc(date_str)} &nbsp;|&nbsp; "
        f"Commit: <code>{_esc(commit)}</code> &nbsp;|&nbsp; "
        f"Generated: {gen_time}</p>",
    ]

    # ── Section 2: Overall progress ──────────────────────────────────────────
    if prog:
        total   = prog["total_ops"]
        impl    = prog["impl_ops"]
        tested  = prog["tested_ops"]
        benched = prog.get("benched_ops", 0)
        done    = prog["done_ops"]
        pct     = done / total * 100 if total else 0

        parts += [
            "<div class='card'>",
            "<h2 style='margin-top:0'>Overall Progress</h2>",
            "<div class='overall-grid'>",
            f"<div class='stat-box'><div class='num'>{total}</div><div class='lbl'>Total Ops</div></div>",
            f"<div class='stat-box'><div class='num' style='color:var(--blue)'>{impl}</div><div class='lbl'>Implemented</div></div>",
            f"<div class='stat-box'><div class='num' style='color:var(--green)'>{tested}</div><div class='lbl'>Tests Pass</div></div>",
            f"<div class='stat-box'><div class='num' style='color:var(--purple)'>{benched}</div><div class='lbl'>Bench Pass</div></div>",
            f"<div class='stat-box'><div class='num' style='font-size:1.1rem'>{done}/{total}</div><div class='lbl'>Done ({pct:.0f}%)</div></div>",
            "</div>",
            # Overall 3-bar
            "<div style='margin-top:.5rem'>",
        ]
        impl_pct  = impl / total * 100 if total else 0
        test_pct  = tested / total * 100 if total else 0
        bench_pct = benched / total * 100 if total else 0
        for label, p, cls, count in [
            ("Impl",  impl_pct,  "bar-impl",  f"{impl}/{total}"),
            ("Test",  test_pct,  "bar-test",  f"{tested}/{total}"),
            ("Bench", bench_pct, "bar-bench", f"{benched}/{total}"),
        ]:
            parts.append(
                f'<div class="bar-row"><span class="bar-label">{label}</span>'
                f'<div class="bar-container">{_bar_html(p, cls)}</div>'
                f'<span class="bar-count">{count}</span></div>'
            )
        parts += ["</div>", "</div>"]

    # ── Section 3: Per-category progress + scores ────────────────────────────
    parts += ["<div class='card'>", "<h2 style='margin-top:0'>Category Progress</h2>"]

    categories = prog.get("categories", []) if prog else []
    cat_analysis = analysis.get("categories", {})

    for cat in categories:
        name  = cat["name"]
        t     = cat["total_ops"]
        im    = cat["impl_ops"]
        te    = cat.get("tested_ops", 0)
        be    = cat.get("benched_ops", 0)
        ca    = cat_analysis.get(name, {})
        ps    = ca.get("perf_score")
        fs    = ca.get("func_score")

        impl_pct  = im / t * 100 if t else 0
        test_pct  = te / t * 100 if t else 0
        bench_pct = be / t * 100 if t else 0

        parts.append('<div class="cat-card">')
        parts.append('<div class="cat-bars">')
        parts.append(f'<h3 style="margin-bottom:.35rem">{_esc(name)}</h3>')
        for label, p, cls, count in [
            ("Impl",  impl_pct,  "bar-impl",  f"{im}/{t}"),
            ("Test",  test_pct,  "bar-test",  f"{te}/{t}"),
            ("Bench", bench_pct, "bar-bench", f"{be}/{t}"),
        ]:
            parts.append(
                f'<div class="bar-row"><span class="bar-label">{label}</span>'
                f'<div class="bar-container">{_bar_html(p, cls)}</div>'
                f'<span class="bar-count">{count}</span></div>'
            )
        parts.append('</div>')  # cat-bars
        parts.append('<div class="cat-scores">')
        parts.append(_score_badge(ps, "Perf", "score-perf"))
        parts.append(_score_badge(fs, "Func", "score-func"))
        parts.append('</div>')  # cat-scores
        parts.append('</div>')  # cat-card

    parts.append("</div>")  # card

    # ── Section 4: Per-category issues & evaluation ──────────────────────────
    parts += ["<div class='card'>", "<h2 style='margin-top:0'>Category Analysis</h2>"]

    for cat in categories:
        name = cat["name"]
        ca   = cat_analysis.get(name, {})
        issues = ca.get("issues", "")
        evaluation = ca.get("evaluation", "")

        if not issues and not evaluation:
            continue

        parts.append('<div class="detail-item">')
        parts.append(f'<h3>{_esc(name)}</h3>')
        if issues:
            parts.append(f'<div class="detail-issues"><strong>Issues:</strong> {_esc(issues)}</div>')
        if evaluation:
            parts.append(f'<div class="detail-eval"><strong>Evaluation:</strong> {_esc(evaluation)}</div>')
        parts.append('</div>')

    parts.append("</div>")  # card

    # ── Section 5: Overall assessment ────────────────────────────────────────
    overall = analysis.get("overall", {})
    summary = overall.get("summary", "")
    recs    = overall.get("recommendations", [])

    parts += ["<div class='card'>", "<h2 style='margin-top:0'>Overall Assessment</h2>"]
    if summary:
        parts.append(f"<p style='font-size:.88rem;line-height:1.6'>{_esc(summary)}</p>")
    if recs:
        parts.append("<h3>Recommendations</h3>")
        parts.append("<ol class='rec-list'>")
        for r in recs:
            parts.append(f"<li>{_esc(r)}</li>")
        parts.append("</ol>")
    parts.append("</div>")

    # ── Footer ───────────────────────────────────────────────────────────────
    parts += [
        "<p style='text-align:center;color:var(--muted);font-size:.72rem;margin-top:1.5rem'>",
        "Generated by Claude &mdash; review for accuracy before acting on recommendations.",
        "</p>",
        "</body></html>",
    ]

    return "\n".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TileOPs nightly HTML report")
    parser.add_argument("--date",          required=True, help="Run date timestamp")
    parser.add_argument("--commit",        required=True, help="Git commit hash")
    parser.add_argument("--report-dir",    required=True, help="Report directory")
    parser.add_argument("--html-output",   required=True, help="HTML output path")
    parser.add_argument("--progress-json", default=None,  help="progress.json path")
    parser.add_argument("--analysis-json", default=None,  help="analysis.json path")
    args = parser.parse_args()

    # Load progress.json
    prog = None
    if args.progress_json and os.path.exists(args.progress_json):
        try:
            prog = json.loads(Path(args.progress_json).read_text())
        except Exception:
            pass

    # Load analysis.json
    analysis = None
    if args.analysis_json and os.path.exists(args.analysis_json):
        try:
            analysis = json.loads(Path(args.analysis_json).read_text())
        except Exception:
            pass

    html = build_html(args, prog, analysis)
    Path(args.html_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.html_output).write_text(html)
    print(f"HTML report: {args.html_output}")


if __name__ == "__main__":
    main()
