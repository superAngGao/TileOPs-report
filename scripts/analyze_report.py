#!/usr/bin/env python3
"""analyze_report.py — Call Claude API to analyze nightly TileOPs report.

Three-phase analysis:
  1. Performance analysis   — based solely on benchmark log
  2. Correctness analysis   — based solely on test_results.xml
  3. Composite scoring      — merges both dimensions into final scores

Reads progress.json (for context only), test_results.xml, and benchmark log;
calls the Claude API; writes analysis.md.

Environment variables:
  ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN  — required
  ANTHROPIC_BASE_URL                         — optional, for custom proxy
"""

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from claude_utils import (
    call_claude,
    call_claude_json,
    get_api_config,
    require_anthropic,
    SYSTEM_REPORT_ANALYZER,
)


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────────────

def _load_progress(path: str | None) -> str:
    """Load progress.json as context summary (not used for scoring)."""
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
            "## Op Progress (context only)",
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


def _load_test_data(path: str | None) -> str:
    """Load test_results.xml and return structured summary for correctness analysis."""
    if not path or not Path(path).exists():
        return ""
    try:
        root = ET.parse(path).getroot()
        suites = list(root) if root.tag == "testsuites" else [root]
        total = errors = failures = skipped = 0
        failed: list[dict] = []
        passed_tests: list[str] = []
        for suite in suites:
            total    += int(suite.get("tests",    0))
            errors   += int(suite.get("errors",   0))
            failures += int(suite.get("failures", 0))
            skipped  += int(suite.get("skipped",  0))
            for tc in suite.iter("testcase"):
                cls  = tc.get("classname", "")
                name = tc.get("name", "")
                node = f"{cls}::{name}" if cls else name
                fail_el = tc.find("failure")
                err_el  = tc.find("error")
                if fail_el is not None or err_el is not None:
                    elem = fail_el if fail_el is not None else err_el
                    msg  = (elem.get("message") or "")[:300]
                    body = (elem.text or "")[:300]
                    failed.append({"node": node, "message": msg, "trace": body})
                else:
                    passed_tests.append(node)

        passed = total - errors - failures - skipped
        lines = [
            "## Test Results (from JUnit XML)",
            f"- Total: {total} | Passed: {passed} | "
            f"Failed: {failures + errors} | Skipped: {skipped}",
        ]
        if failed:
            lines += ["", f"### Failed tests ({len(failed)} failures)"]
            for t in failed[:30]:
                lines.append(f"- `{t['node']}`")
                lines.append(f"  message: {t['message'][:250]}")
                if t["trace"].strip():
                    lines.append(f"  trace: {t['trace'][:250]}")
            if len(failed) > 30:
                lines.append(f"... and {len(failed) - 30} more")
        else:
            lines.append("- All tests passed.")

        if passed_tests:
            lines += ["", f"### Passed tests ({len(passed_tests)} total)"]
            for t in passed_tests[:50]:
                lines.append(f"- `{t}`")
            if len(passed_tests) > 50:
                lines.append(f"... and {len(passed_tests) - 50} more")

        return "\n".join(lines) + "\n"
    except Exception:
        return ""


def _load_bench_log(path: str | None) -> str:
    """Load full benchmark log for performance analysis."""
    if not path or not Path(path).exists():
        return ""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
        # Truncate to last 200 lines if very large
        log_lines = text.splitlines()
        if len(log_lines) > 200:
            log_lines = log_lines[-200:]
        return (
            "## Benchmark Log\n"
            "```\n" + "\n".join(log_lines) + "\n```\n"
        )
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Performance analysis (benchmark only)
# ─────────────────────────────────────────────────────────────────────────────

PERF_SYSTEM = (
    "You are a performance engineer reviewing benchmark results for TileOPs, "
    "a high-performance LLM operator library built on TileLang. "
    "Analyze ONLY performance data. Do not speculate about correctness or test results. "
    "Write concise, actionable analysis in English Markdown."
)


def build_perf_prompt(bench_log: str | None) -> str | None:
    bench_text = _load_bench_log(bench_log)
    if not bench_text:
        return None
    return (
        "You are reviewing benchmark results for **TileOPs**.\n\n"
        f"{bench_text}\n"
        "## Your Task\n"
        "Analyze the benchmark data above and provide:\n"
        "1. **Performance summary** — Overall performance level vs baselines.\n"
        "2. **Regressions** — Any ops with notably poor ratios or slowdowns.\n"
        "3. **Highlights** — Ops that exceed baselines or show strong performance.\n"
        "4. **perf_score** — An overall performance score (1-5) based on:\n"
        "   - 1 = <50% of baseline on average\n"
        "   - 2 = 50-75% of baseline\n"
        "   - 3 = 75-90% of baseline\n"
        "   - 4 = 90-100% of baseline\n"
        "   - 5 = >100% of baseline (exceeds)\n\n"
        "Use `## Section` headers and bullet lists. "
        "End with a line: `perf_score: N` where N is 1-5."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Correctness analysis (test XML only)
# ─────────────────────────────────────────────────────────────────────────────

CORRECTNESS_SYSTEM = (
    "You are a QA engineer reviewing test results for TileOPs, "
    "a high-performance LLM operator library built on TileLang. "
    "Analyze ONLY test correctness data. Do not speculate about performance or benchmarks. "
    "Write concise, actionable analysis in English Markdown."
)


def build_correctness_prompt(test_xml: str | None) -> str | None:
    test_text = _load_test_data(test_xml)
    if not test_text:
        return None
    return (
        "You are reviewing test results for **TileOPs**.\n\n"
        f"{test_text}\n"
        "## Your Task\n"
        "Analyze the test results above and provide:\n"
        "1. **Correctness summary** — Overall pass rate and test health.\n"
        "2. **Failure patterns** — Group failures by root cause or module. "
        "Are there systemic issues (e.g., import errors, shape mismatches, numerical precision)?\n"
        "3. **has_bugs** — true if there are meaningful test failures, false if all pass or "
        "failures are only flaky/infra-related.\n"
        "4. **func_score** — An overall functional completeness score (1-5) based on:\n"
        "   - 1 = Most tests fail, fundamental issues\n"
        "   - 2 = Many failures, major gaps in correctness\n"
        "   - 3 = Mixed results, some modules solid, others broken\n"
        "   - 4 = Mostly passing, only minor issues\n"
        "   - 5 = All or nearly all tests pass\n\n"
        "Use `## Section` headers and bullet lists. "
        "End with two lines:\n"
        "`func_score: N` where N is 1-5\n"
        "`has_bugs: true/false`"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Composite scoring
# ─────────────────────────────────────────────────────────────────────────────

COMPOSITE_SYSTEM = (
    "You are a senior CI/DevOps engineer producing the final nightly assessment for TileOPs. "
    "Synthesize the separate performance and correctness analyses into a unified report. "
    "Write concise English Markdown. Use ## Section headers and bullet lists. No greetings."
)


def build_composite_prompt(
    progress_json: str | None,
    perf_analysis: str | None,
    correctness_analysis: str | None,
) -> str:
    sections = [
        "You are producing the final nightly assessment for **TileOPs**, "
        "a high-performance LLM operator library built on TileLang.\n",
    ]

    progress_text = _load_progress(progress_json)
    if progress_text:
        sections.append(progress_text)

    if perf_analysis:
        sections.append(f"## Performance Analysis (from benchmarks)\n{perf_analysis}\n")
    else:
        sections.append("## Performance Analysis\nNo benchmark data available.\n")

    if correctness_analysis:
        sections.append(f"## Correctness Analysis (from tests)\n{correctness_analysis}\n")
    else:
        sections.append("## Correctness Analysis\nNo test data available.\n")

    sections.append(
        "## Your Task\n"
        "Based on the two separate analyses above, write a unified report covering:\n"
        "1. **Overall health** — Combine performance and correctness perspectives.\n"
        "2. **Key issues** — Top problems from either dimension.\n"
        "3. **Op progress** — Which categories are moving well? Any bottlenecks?\n"
        "4. **Composite scores** — Provide final scores:\n"
        "   - `perf_score`: 1-5 (from performance analysis, or null if no data)\n"
        "   - `func_score`: 1-5 (from correctness analysis, or null if no data)\n"
        "   - `has_bugs`: true/false (from correctness analysis, or null if no data)\n"
        "5. **Top recommendations** — 2-3 concrete action items.\n\n"
        "End with a scores block:\n"
        "```\n"
        "perf_score: N\n"
        "func_score: N\n"
        "has_bugs: true/false\n"
        "```"
    )
    return "\n".join(s for s in sections if s)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

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

    api_key, base_url = get_api_config()

    if not api_key:
        print(
            "::warning::ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN not set — "
            "skipping Claude analysis.",
            file=sys.stderr,
        )
        sys.exit(0)

    if require_anthropic() is None:
        sys.exit(0)

    # ── Phase 1: Performance analysis (benchmark only) ───────────────────────
    perf_analysis = None
    perf_prompt = build_perf_prompt(args.bench_log)
    if perf_prompt:
        print("Phase 1: Analyzing performance from benchmark log ...")
        try:
            perf_analysis = call_claude(
                perf_prompt, args.model, api_key, base_url,
                system=PERF_SYSTEM,
                max_tokens=1200,
                temperature=0.2,
            )
            print(f"  Performance analysis: {len(perf_analysis)} chars")
        except Exception as exc:
            print(f"::warning::Performance analysis failed: {exc}", file=sys.stderr)
    else:
        print("Phase 1: No benchmark log — skipping performance analysis.")

    # ── Phase 2: Correctness analysis (test XML only) ────────────────────────
    correctness_analysis = None
    correctness_prompt = build_correctness_prompt(args.test_xml)
    if correctness_prompt:
        print("Phase 2: Analyzing correctness from test XML ...")
        try:
            correctness_analysis = call_claude(
                correctness_prompt, args.model, api_key, base_url,
                system=CORRECTNESS_SYSTEM,
                max_tokens=1200,
                temperature=0.2,
            )
            print(f"  Correctness analysis: {len(correctness_analysis)} chars")
        except Exception as exc:
            print(f"::warning::Correctness analysis failed: {exc}", file=sys.stderr)
    else:
        print("Phase 2: No test XML — skipping correctness analysis.")

    # ── Phase 3: Composite scoring ───────────────────────────────────────────
    print("Phase 3: Generating composite analysis and scores ...")
    composite_prompt = build_composite_prompt(
        args.progress_json, perf_analysis, correctness_analysis,
    )
    try:
        analysis = call_claude(
            composite_prompt, args.model, api_key, base_url,
            system=COMPOSITE_SYSTEM,
            max_tokens=1500,
            temperature=0.3,
        )
    except Exception as exc:
        print(f"::warning::Composite analysis failed: {exc}", file=sys.stderr)
        # Fallback: concatenate whatever we have
        parts = []
        if perf_analysis:
            parts.append("## Performance Analysis\n" + perf_analysis)
        if correctness_analysis:
            parts.append("## Correctness Analysis\n" + correctness_analysis)
        analysis = "\n\n".join(parts) if parts else "Analysis unavailable."

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(analysis)
    print(f"Analysis written: {args.output} ({len(analysis)} chars)")


if __name__ == "__main__":
    main()
