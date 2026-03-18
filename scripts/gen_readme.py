#!/usr/bin/env python3
"""gen_readme.py — Generate README.md from op_registry.json + analysis.json."""

import argparse
import json
from pathlib import Path


def _bar(count: int, total: int, filled_ch: str = "\u2588", empty_ch: str = "\u2591", width: int = 15) -> str:
    """Unicode progress bar."""
    if total == 0:
        return f"`{empty_ch * width}` 0%"
    ratio = count / total
    filled = round(ratio * width)
    bar = filled_ch * filled + empty_ch * (width - filled)
    return f"`{bar}` {count}/{total}"


def _status_icon(impl: bool, tested: bool, test_failed: bool, bench_ok) -> str:
    if impl and tested and bench_ok is True:
        return "\u2705"   # green check
    if impl and tested:
        return "\U0001f7e1"  # yellow circle
    if impl and test_failed:
        return "\u274c"   # red x
    if impl:
        return "\U0001f7e6"  # blue square
    return "\u2b1c"       # white square


def _score_display(score) -> str:
    if score is None:
        return "\u2014"
    stars = "\u2b50" * score + "\u2606" * (5 - score)
    return f"{stars} ({score}/5)"


def build_readme(registry: dict, analysis: dict | None, pages_url: str) -> str:
    summary = registry.get("summary", {})
    ops = registry.get("ops", {})
    generated = registry.get("generated_at", "unknown")

    total = summary.get("total_ops", 0)
    impl = summary.get("impl_ops", 0)
    tested = summary.get("tested_ops", 0)
    benched = summary.get("benched_ops", 0)
    done = summary.get("done_ops", 0)
    pct = done * 100 // total if total else 0

    # Status counts
    ts = {"passed": 0, "failed": 0, "missing": 0}
    bs = {"qualified": 0, "passed": 0, "underperforming": 0, "failed": 0, "missing": 0}
    for op in ops.values():
        t = (op.get("test_status") or {}).get("status", "missing")
        b = (op.get("bench_status") or {}).get("status", "missing")
        ts[t] = ts.get(t, 0) + 1
        bs[b] = bs.get(b, 0) + 1
    bench_ok_count = bs["qualified"] + bs["passed"]

    cat_analysis = (analysis or {}).get("categories", {})
    overall = (analysis or {}).get("overall", {})

    lines = [
        "<div align=\"center\">",
        "",
        "# TileOPs Operator Tracking",
        "",
        f"**{done}/{total}** operators complete ({pct}%)",
        "",
        f"[![Report]({pages_url}report.html)]({pages_url})",
        "",
        f"Last updated: `{generated}`",
        "",
        "</div>",
        "",
        "---",
        "",
        "## Overview",
        "",
        f"| | Count | Bar |",
        f"|:--|------:|:----|",
        f"| Implemented | **{impl}**/{total} | {_bar(impl, total)} |",
        f"| Test Passed | **{tested}**/{total} | {_bar(tested, total)} |",
        f"| Bench Passed | **{benched}**/{total} | {_bar(benched, total)} |",
        f"| **Done** | **{done}**/{total} | {_bar(done, total)} |",
        "",
        "### Status Breakdown",
        "",
        "| Test | | Bench | |",
        "|:-----|----:|:------|----:|",
        f"| Passed | {ts['passed']} | Qualified (ratio >= 0.8) | {bs['qualified']} |",
        f"| Failed | {ts['failed']} | Passed (no ratio) | {bs['passed']} |",
        f"| Missing | {ts['missing']} | Underperforming | {bs['underperforming']} |",
        f"| | | Failed | {bs['failed']} |",
        f"| | | Missing | {bs['missing']} |",
        "",
    ]

    # Overall assessment from Claude
    if overall.get("summary"):
        lines += [
            "## Assessment",
            "",
            f"> {overall['summary']}",
            "",
        ]
        recs = overall.get("recommendations", [])
        if recs:
            lines.append("**Recommendations:**")
            lines.append("")
            for i, r in enumerate(recs, 1):
                lines.append(f"{i}. {r}")
            lines.append("")

    # Category progress
    lines += [
        "## Categories",
        "",
    ]

    for cat in summary.get("categories", []):
        name = cat["name"]
        t = cat["total_ops"]
        im = cat["impl_ops"]
        te = cat.get("tested_ops", 0)
        be = cat.get("benched_ops", 0)
        d = cat["done_ops"]

        ca = cat_analysis.get(name, {})
        ps = ca.get("perf_score")
        fs = ca.get("func_score")

        # Category header with scores
        score_text = ""
        if fs is not None or ps is not None:
            parts = []
            if fs is not None:
                parts.append(f"Func: {_score_display(fs)}")
            if ps is not None:
                parts.append(f"Perf: {_score_display(ps)}")
            score_text = " | " + " | ".join(parts)

        lines.append(f"### {name}{score_text}")
        lines.append("")
        lines.append(f"| | Progress | |")
        lines.append(f"|:--|:---------|:--|")
        lines.append(f"| Impl | {_bar(im, t)} | |")
        lines.append(f"| Test | {_bar(te, t)} | |")
        lines.append(f"| Bench | {_bar(be, t)} | |")
        lines.append("")

        # Issues & evaluation from Claude
        if ca.get("issues"):
            lines.append(f"> **Issues:** {ca['issues']}")
            lines.append("")
        if ca.get("evaluation"):
            lines.append(f"> **Evaluation:** {ca['evaluation']}")
            lines.append("")

        # Per-op details (collapsed)
        lines.append("<details>")
        lines.append(f"<summary>{d}/{t} done - click to expand</summary>")
        lines.append("")
        lines.append("| | Operator | Test | Bench | Ratio |")
        lines.append("|:--|:---------|:----:|:-----:|------:|")
        for op_s in cat.get("ops", []):
            op_id = op_s["id"]
            op_name = op_s["name"]
            op_reg = ops.get(op_id, {})
            impl_v = op_s.get("implemented", False)
            tested_v = op_s.get("tested", False)
            tf = op_s.get("test_failed", False)
            bo = op_s.get("bench_ok")
            icon = _status_icon(impl_v, tested_v, tf, bo)

            ts_s = (op_reg.get("test_status") or {}).get("status", "-")
            bs_s = (op_reg.get("bench_status") or {}).get("status", "-")
            ratio = (op_reg.get("bench_status") or {}).get("ratio")
            ratio_s = f"{ratio:.2f}" if ratio is not None else "-"

            lines.append(f"| {icon} | {op_name} | {ts_s} | {bs_s} | {ratio_s} |")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    lines += [
        "---",
        "",
        f"<sub>Auto-generated by the [TileOPs Report]({pages_url}) pipeline "
        f"from [`op_registry.json`](scripts/op_registry.json)</sub>",
    ]

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate README.md from registry + analysis")
    parser.add_argument("--registry", required=True, help="op_registry.json path")
    parser.add_argument("--analysis", default=None, help="analysis.json path")
    parser.add_argument("--output", default="README.md", help="Output path")
    parser.add_argument("--pages-url", default="https://superanggao.github.io/TileOPs-report/nightly/")
    args = parser.parse_args()

    registry = json.loads(Path(args.registry).read_text())

    analysis = None
    if args.analysis and Path(args.analysis).exists():
        try:
            analysis = json.loads(Path(args.analysis).read_text())
        except Exception:
            pass

    readme = build_readme(registry, analysis, args.pages_url)
    Path(args.output).write_text(readme)
    print(f"README written: {args.output}")


if __name__ == "__main__":
    main()
