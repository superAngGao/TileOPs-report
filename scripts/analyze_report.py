#!/usr/bin/env python3
"""analyze_report.py — Call Claude API to analyze nightly TileOPs report.

Reads progress.json, test_results.xml, and benchmark log; calls the Claude API
(supports a custom ANTHROPIC_BASE_URL proxy); writes analysis.md.

Environment variables:
  ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN  — required
  ANTHROPIC_BASE_URL                         — optional, for custom proxy
"""

import argparse
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def _load_progress(path: str | None) -> str:
    if not path or not Path(path).exists():
        return ""
    try:
        prog = json.loads(Path(path).read_text())
        done  = prog["done_ops"]
        total = prog["total_ops"]
        impl  = prog["impl_ops"]
        tested = prog["tested_ops"]
        pct   = done / total * 100 if total else 0

        lines = [
            "## Op Progress",
            f"- Total target ops: {total}",
            f"- Fully done (tested + bench): {done} ({pct:.1f}%)",
            f"- Implemented: {impl}",
            f"- Passing tests: {tested}",
            "",
            "### Category breakdown",
        ]
        for cat in prog.get("categories", []):
            d, t, im = cat["done_ops"], cat["total_ops"], cat["impl_ops"]
            lines.append(
                f"- {cat['name']}: {d}/{t} done, {im} implemented "
                f"({'✓' if d == t else '…'})"
            )
        return "\n".join(lines) + "\n"
    except Exception:
        return ""


def _load_test_failures(path: str | None) -> str:
    if not path or not Path(path).exists():
        return ""
    try:
        root = ET.parse(path).getroot()
        suites = list(root) if root.tag == "testsuites" else [root]
        total = errors = failures = skipped = 0
        failed: list[dict] = []
        for suite in suites:
            total    += int(suite.get("tests",    0))
            errors   += int(suite.get("errors",   0))
            failures += int(suite.get("failures", 0))
            skipped  += int(suite.get("skipped",  0))
            for tc in suite.iter("testcase"):
                fail_el = tc.find("failure")
                err_el  = tc.find("error")
                if fail_el is not None or err_el is not None:
                    cls  = tc.get("classname", "")
                    name = tc.get("name", "")
                    node = f"{cls}::{name}" if cls else name
                    elem = fail_el if fail_el is not None else err_el
                    msg  = (elem.get("message") or "")[:200]
                    failed.append({"node": node, "message": msg})

        passed = total - errors - failures - skipped
        lines = [
            "## Test Results",
            f"- Total: {total} | Passed: {passed} | "
            f"Failed: {failures + errors} | Skipped: {skipped}",
        ]
        if failed:
            lines += ["", f"### Failed tests ({len(failed)} failures)"]
            for t in failed[:25]:
                lines.append(f"- `{t['node']}`: {t['message'][:180]}")
            if len(failed) > 25:
                lines.append(f"... and {len(failed) - 25} more")
        else:
            lines.append("- All tests passed.")
        return "\n".join(lines) + "\n"
    except Exception:
        return ""


def _load_bench_log(path: str | None, tail: int = 50) -> str:
    if not path or not Path(path).exists():
        return ""
    try:
        log_lines = Path(path).read_text().splitlines()[-tail:]
        return (
            "## Benchmark Log (last 50 lines)\n"
            "```\n" + "\n".join(log_lines) + "\n```\n"
        )
    except Exception:
        return ""


def build_prompt(
    progress_json: str | None,
    test_xml: str | None,
    bench_log: str | None,
) -> str:
    sections = [
        "You are reviewing a nightly CI report for **TileOPs**, a high-performance "
        "LLM operator library built on TileLang. The goal is to implement all ops "
        "listed in the manifest with passing tests and passing benchmarks.\n",
        _load_progress(progress_json),
        _load_test_failures(test_xml),
        _load_bench_log(bench_log),
        "## Your Task\n"
        "Write a concise analysis (English, Markdown) covering:\n"
        "1. **Overall health** — Is the project on track?\n"
        "2. **Test failures** — Main failure patterns, any systemic issues?\n"
        "3. **Op progress** — Which categories are moving well? Any bottlenecks?\n"
        "4. **Top recommendations** — 2–3 concrete action items for the team.\n\n"
        "Be specific and actionable. Avoid generic advice. "
        "Use `## Section` headers and bullet lists.",
    ]
    return "\n".join(s for s in sections if s)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Call Claude API to analyse the nightly TileOPs report"
    )
    parser.add_argument("--progress-json", default=None, help="Path to progress.json")
    parser.add_argument("--test-xml",      default=None, help="Path to test_results.xml")
    parser.add_argument("--bench-log",     default=None, help="Path to tileops_benchmarks.log")
    parser.add_argument("--output",        required=True, help="Output path for analysis.md")
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-6",
        help="Claude model ID (default: claude-sonnet-4-6)",
    )
    args = parser.parse_args()

    api_key  = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    base_url = os.environ.get("ANTHROPIC_BASE_URL")

    if not api_key:
        print(
            "::warning::ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN not set — "
            "skipping Claude analysis.",
            file=sys.stderr,
        )
        sys.exit(0)

    try:
        import anthropic
    except ImportError:
        print(
            "::warning::anthropic package not installed — skipping Claude analysis.",
            file=sys.stderr,
        )
        sys.exit(0)

    prompt = build_prompt(args.progress_json, args.test_xml, args.bench_log)

    client_kwargs: dict = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    client = anthropic.Anthropic(**client_kwargs)

    try:
        message = client.messages.create(
            model=args.model,
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )
        analysis = message.content[0].text
    except Exception as exc:
        print(f"::warning::Claude API call failed: {exc}", file=sys.stderr)
        sys.exit(0)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(analysis)
    print(f"Analysis written: {args.output} ({len(analysis)} chars)")


if __name__ == "__main__":
    main()
