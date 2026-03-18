#!/usr/bin/env python3
"""analyze_report.py — Call Claude API to analyze nightly TileOPs report.

Three-phase analysis:
  1. Performance analysis   — per-category scoring from benchmark log
  2. Correctness analysis   — per-category scoring from test_results.xml
  3. Composite scoring      — per-category composite scores + recommendations

Reads progress.json (for category structure), test_results.xml, and benchmark
log; calls the Claude API; writes analysis.md.

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

def _load_progress_detailed(path: str | None) -> tuple[dict | None, dict[str, str]]:
    """Load progress.json, return (full dict, test_fn→category_name map).

    The reverse map lets us group JUnit test results by operator category.
    """
    if not path or not Path(path).exists():
        return None, {}
    try:
        prog = json.loads(Path(path).read_text())
        fn_to_cat: dict[str, str] = {}
        for cat in prog.get("categories", []):
            cat_name = cat["name"]
            for op in cat.get("ops", []):
                # op dict from compute_progress may not carry test_fns,
                # but the name + sub can still be used for heuristic matching.
                # We store the op name itself as a key too.
                fn_to_cat[op["name"]] = cat_name
        return prog, fn_to_cat
    except Exception:
        return None, {}


def _progress_summary_text(prog: dict) -> str:
    """Format progress.json into a Markdown summary string."""
    done    = prog["done_ops"]
    total   = prog["total_ops"]
    impl    = prog["impl_ops"]
    tested  = prog["tested_ops"]
    benched = prog.get("benched_ops", 0)
    pct     = done / total * 100 if total else 0

    lines = [
        "## Op Progress",
        f"- Total target ops: {total}",
        f"- Fully done (impl + test + bench): {done} ({pct:.1f}%)",
        f"- Implemented: {impl}",
        f"- Passing tests: {tested}",
        f"- Passing benchmarks: {benched}",
        "",
        "### Category breakdown",
    ]
    for cat in prog.get("categories", []):
        d  = cat["done_ops"]
        t  = cat["total_ops"]
        im = cat["impl_ops"]
        te = cat.get("tested_ops", 0)
        be = cat.get("benched_ops", 0)
        lines.append(
            f"- **{cat['name']}**: {d}/{t} done, {im} impl, {te} tested, {be} benched "
            f"({'✓' if d == t else '…'})"
        )
    return "\n".join(lines) + "\n"


def _progress_category_detail(prog: dict) -> str:
    """Format per-category, per-op status from progress.json."""
    lines: list[str] = []
    for cat in prog.get("categories", []):
        d, t, im = cat["done_ops"], cat["total_ops"], cat["impl_ops"]
        te = cat.get("tested_ops", 0)
        lines += [
            f"### {cat['name']} ({d}/{t} done, {im} impl, {te} tested)",
        ]
        for op in cat.get("ops", []):
            impl_s = "✅" if op.get("implemented") else "🔲"
            if op.get("tested"):
                test_s = "✅"
            elif op.get("test_failed"):
                test_s = "❌"
            else:
                test_s = "—"
            bench_s = "✅" if op.get("bench_ok") is True else (
                "❌" if op.get("bench_ok") is False else "—"
            )
            lines.append(f"- `{op['name']}` [{op.get('sub','')}]  impl={impl_s}  test={test_s}  bench={bench_s}")
        lines.append("")
    return "\n".join(lines)


def _build_test_fn_to_cat(manifest_path: str | None) -> dict[str, str]:
    """Build test_fn → category_name map from op_manifest.json."""
    if not manifest_path or not Path(manifest_path).exists():
        return {}
    try:
        manifest = json.loads(Path(manifest_path).read_text())
        fn_to_cat: dict[str, str] = {}
        for cat in manifest.get("categories", []):
            cat_name = cat["name"]
            for op in cat.get("ops", []):
                for fn in op.get("test_fns", []):
                    fn_to_cat[fn] = cat_name
        return fn_to_cat
    except Exception:
        return {}


def _load_test_data_by_category(
    xml_path: str | None,
    fn_to_cat: dict[str, str],
) -> tuple[str, dict[str, dict]]:
    """Load test_results.xml and group results by category.

    Returns (overall_summary_text, {category_name: {passed, failed, failures_detail}}).
    """
    if not xml_path or not Path(xml_path).exists():
        return "", {}
    try:
        root = ET.parse(xml_path).getroot()
        suites = list(root) if root.tag == "testsuites" else [root]
        total = errors = failures = skipped = 0
        # Per-category accumulators
        cat_data: dict[str, dict] = {}
        uncategorized_passed: list[str] = []
        uncategorized_failed: list[dict] = []

        for suite in suites:
            total    += int(suite.get("tests",    0))
            errors   += int(suite.get("errors",   0))
            failures += int(suite.get("failures", 0))
            skipped  += int(suite.get("skipped",  0))
            for tc in suite.iter("testcase"):
                cls  = tc.get("classname", "")
                name = tc.get("name", "")
                node = f"{cls}::{name}" if cls else name
                # Extract bare function name (strip parametrize suffixes)
                bare_fn = name.split("[")[0]
                fail_el = tc.find("failure")
                err_el  = tc.find("error")
                is_fail = fail_el is not None or err_el is not None

                # Determine category
                cat_name = fn_to_cat.get(bare_fn, "")
                if not cat_name:
                    # Heuristic: try to match by test file path in classname
                    for fn, cn in fn_to_cat.items():
                        if fn in bare_fn:
                            cat_name = cn
                            break

                if is_fail:
                    elem = fail_el if fail_el is not None else err_el
                    msg  = (elem.get("message") or "")[:300]
                    body = (elem.text or "")[:300]
                    entry = {"node": node, "message": msg, "trace": body}
                    if cat_name:
                        cat_data.setdefault(cat_name, {"passed": [], "failed": []})
                        cat_data[cat_name]["failed"].append(entry)
                    else:
                        uncategorized_failed.append(entry)
                else:
                    if cat_name:
                        cat_data.setdefault(cat_name, {"passed": [], "failed": []})
                        cat_data[cat_name]["passed"].append(node)
                    else:
                        uncategorized_passed.append(node)

        passed = total - errors - failures - skipped

        # Build overall summary
        lines = [
            "## Test Results (from JUnit XML)",
            f"- Total: {total} | Passed: {passed} | "
            f"Failed: {failures + errors} | Skipped: {skipped}",
            "",
        ]

        # Per-category breakdown
        for cat_name in sorted(cat_data.keys()):
            cd = cat_data[cat_name]
            p_count = len(cd["passed"])
            f_count = len(cd["failed"])
            lines.append(f"### {cat_name} — {p_count} passed, {f_count} failed")
            if cd["failed"]:
                for t in cd["failed"][:10]:
                    lines.append(f"- ❌ `{t['node']}`")
                    lines.append(f"  message: {t['message'][:200]}")
                if len(cd["failed"]) > 10:
                    lines.append(f"  ... and {len(cd['failed']) - 10} more failures")
            if cd["passed"]:
                for t in cd["passed"][:5]:
                    lines.append(f"- ✅ `{t}`")
                if len(cd["passed"]) > 5:
                    lines.append(f"  ... and {len(cd['passed']) - 5} more passed")
            lines.append("")

        # Uncategorized
        if uncategorized_failed or uncategorized_passed:
            lines.append(f"### Uncategorized — {len(uncategorized_passed)} passed, {len(uncategorized_failed)} failed")
            for t in uncategorized_failed[:5]:
                lines.append(f"- ❌ `{t['node']}`: {t['message'][:150]}")
            for t in uncategorized_passed[:5]:
                lines.append(f"- ✅ `{t}`")
            lines.append("")

        return "\n".join(lines), cat_data
    except Exception:
        return "", {}


def _load_bench_log(path: str | None) -> str:
    """Load full benchmark log for performance analysis."""
    if not path or not Path(path).exists():
        return ""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
        log_lines = text.splitlines()
        if len(log_lines) > 300:
            log_lines = log_lines[-300:]
        return (
            "## Benchmark Log\n"
            "```\n" + "\n".join(log_lines) + "\n```\n"
        )
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Phase 0: Op completion progress (pure data, no Claude call)
# ─────────────────────────────────────────────────────────────────────────────

def build_progress_report(prog: dict | None) -> str:
    """Generate a per-category completion statistics section from progress.json.

    Returns ready-to-use Markdown text. No Claude API call needed.
    """
    if not prog:
        return (
            "## Op Completion Progress\n\n"
            "> No progress data available (progress.json missing or failed to generate).\n"
        )

    done    = prog["done_ops"]
    total   = prog["total_ops"]
    impl    = prog["impl_ops"]
    tested  = prog["tested_ops"]
    benched = prog.get("benched_ops", 0)
    pct     = done / total * 100 if total else 0

    lines = [
        "## Op Completion Progress",
        "",
        f"**Overall: {done} / {total} ops fully completed ({pct:.1f}%)**  ",
        f"Implemented: {impl} | Tests passing: {tested} | Bench passing: {benched}",
        "",
        "> An op is \"Done\" only when all three pass: implementation exists, "
        "tests pass, and benchmark passes.",
        "",
        "| # | Category | Total | Impl | Tested | Bench | Done | Completion | Status |",
        "|---|----------|------:|-----:|-------:|------:|-----:|-----------:|--------|",
    ]

    for cat in prog.get("categories", []):
        cid   = cat["id"]
        name  = cat["name"]
        t     = cat["total_ops"]
        im    = cat["impl_ops"]
        te    = cat.get("tested_ops", 0)
        be    = cat.get("benched_ops", 0)
        d     = cat["done_ops"]
        cp    = d / t * 100 if t else 0

        if d == t:
            status = "✅ Done"
        elif im > 0 or d > 0:
            status = "🔧 In Progress"
        else:
            status = "🔲 Not Started"

        bar = _progress_bar_text(cp)
        lines.append(
            f"| {cid} | **{name}** | {t} | {im} | {te} | {be} | {d} "
            f"| {bar} {cp:.0f}% | {status} |"
        )

    lines += [""]

    # Per-category op-level detail in collapsible sections
    lines.append("### Per-Op Detail")
    lines.append("")
    for cat in prog.get("categories", []):
        d, t = cat["done_ops"], cat["total_ops"]
        im   = cat["impl_ops"]
        te   = cat.get("tested_ops", 0)
        lines += [
            f"<details>",
            f"<summary><b>{cat['name']}</b> — {d}/{t} done, {im} impl, {te} tested</summary>",
            "",
            "| Op | Sub | Impl | Test | Bench |",
            "|---|---|:---:|:---:|:---:|",
        ]
        for op in cat.get("ops", []):
            impl_s  = "✅" if op.get("implemented") else "🔲"
            test_s  = "✅" if op.get("tested") else ("❌" if op.get("test_failed") else "—")
            bench_s = "✅" if op.get("bench_ok") is True else (
                "❌" if op.get("bench_ok") is False else "—"
            )
            lines.append(
                f"| `{op['name']}` | {op.get('sub', '')} "
                f"| {impl_s} | {test_s} | {bench_s} |"
            )
        lines += ["", "</details>", ""]

    return "\n".join(lines)


def _progress_bar_text(pct: float, width: int = 10) -> str:
    """Generate a simple text progress bar like [████░░░░░░]."""
    filled = round(pct / 100 * width)
    empty  = width - filled
    return f"[{'█' * filled}{'░' * empty}]"


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Performance analysis (benchmark only)
# ─────────────────────────────────────────────────────────────────────────────

PERF_SYSTEM = (
    "You are a performance engineer reviewing benchmark results for TileOPs, "
    "a high-performance LLM operator library built on TileLang. "
    "Analyze ONLY performance data. Do not speculate about correctness or test results. "
    "Write concise, actionable analysis in English Markdown. "
    "Score each operator category independently."
)

# Category names for reference in prompts
_CATEGORIES = [
    "Elementwise", "Reduce", "Norm", "Conv & Pooling", "GEMM",
    "Quantize", "Sampling", "Flash Attention", "MoE",
    "Linear Attention", "SSM",
]


def build_perf_prompt(
    bench_log: str | None,
    progress_detail: str | None = None,
) -> str | None:
    bench_text = _load_bench_log(bench_log)
    if not bench_text:
        return None

    cat_list = ", ".join(_CATEGORIES)

    sections = [
        "You are reviewing benchmark results for **TileOPs**.\n",
        f"The operator categories are: {cat_list}\n",
    ]
    if progress_detail:
        sections.append(
            "### Per-op implementation status (for mapping benchmarks to categories)\n"
            + progress_detail + "\n"
        )
    sections.append(bench_text)
    sections.append(
        "## Your Task\n"
        "Analyze the benchmark data above **per operator category** and provide:\n\n"
        "For EACH category that has benchmark data:\n"
        "1. **Summary** — Performance level vs baselines for this category.\n"
        "2. **Regressions** — Ops with notably poor ratios or slowdowns.\n"
        "3. **Highlights** — Ops that exceed baselines.\n"
        "4. **perf_score** — Category-level performance score (1-5):\n"
        "   - 1 = <50% of baseline on average\n"
        "   - 2 = 50-75% of baseline\n"
        "   - 3 = 75-90% of baseline\n"
        "   - 4 = 90-100% of baseline\n"
        "   - 5 = >100% of baseline (exceeds)\n\n"
        "Use `## Category Name` as section header for each category.\n"
        "For categories with no benchmark data, write: `perf_score: null`.\n\n"
        "End with a summary table in this exact format:\n"
        "```\n"
        "perf_scores:\n"
        + "".join(f"  {c}: N\n" for c in _CATEGORIES)
        + "```\n"
        "where N is 1-5 or null."
    )
    return "\n".join(sections)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Correctness analysis (test XML only)
# ─────────────────────────────────────────────────────────────────────────────

CORRECTNESS_SYSTEM = (
    "You are a QA engineer reviewing test results for TileOPs, "
    "a high-performance LLM operator library built on TileLang. "
    "Analyze ONLY test correctness data. Do not speculate about performance or benchmarks. "
    "Write concise, actionable analysis in English Markdown. "
    "Score each operator category independently."
)


def build_correctness_prompt(
    test_xml: str | None,
    fn_to_cat: dict[str, str] | None = None,
) -> str | None:
    if fn_to_cat is None:
        fn_to_cat = {}
    test_text, cat_data = _load_test_data_by_category(test_xml, fn_to_cat)
    if not test_text:
        return None

    cat_list = ", ".join(_CATEGORIES)

    return (
        "You are reviewing test results for **TileOPs**.\n\n"
        f"The operator categories are: {cat_list}\n\n"
        f"{test_text}\n"
        "## Your Task\n"
        "Analyze the test results above **per operator category** and provide:\n\n"
        "For EACH category:\n"
        "1. **Summary** — Pass rate and test health for this category.\n"
        "2. **Failure analysis** — Root causes of failures in this category "
        "(e.g., import errors, shape mismatches, numerical precision).\n"
        "3. **Recommendation** — 1-2 actionable items to improve this category.\n"
        "4. **func_score** — Category-level functional score (1-5):\n"
        "   - 1 = Most tests fail, fundamental issues\n"
        "   - 2 = Many failures, major gaps\n"
        "   - 3 = Mixed results, some ops broken\n"
        "   - 4 = Mostly passing, minor issues\n"
        "   - 5 = All or nearly all tests pass\n"
        "   For categories with no test data, use null.\n"
        "5. **has_bugs** — true/false for this category.\n\n"
        "Use `## Category Name` as section header for each category.\n\n"
        "End with a summary table in this exact format:\n"
        "```\n"
        "func_scores:\n"
        + "".join(f"  {c}: N\n" for c in _CATEGORIES)
        + "has_bugs:\n"
        + "".join(f"  {c}: true/false\n" for c in _CATEGORIES)
        + "```\n"
        "where N is 1-5 or null."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Composite scoring
# ─────────────────────────────────────────────────────────────────────────────

COMPOSITE_SYSTEM = (
    "You are a senior CI/DevOps engineer producing the final nightly assessment for TileOPs. "
    "Synthesize the separate performance and correctness analyses into a unified report. "
    "Write concise English Markdown. Use ## Section headers and bullet lists. No greetings. "
    "Provide per-category scores and targeted recommendations."
)


def build_composite_prompt(
    prog: dict | None,
    perf_analysis: str | None,
    correctness_analysis: str | None,
) -> str:
    cat_list = ", ".join(_CATEGORIES)

    sections = [
        "You are producing the final nightly assessment for **TileOPs**, "
        "a high-performance LLM operator library built on TileLang.\n",
        f"The operator categories are: {cat_list}\n",
    ]

    if prog:
        sections.append(_progress_summary_text(prog))
    else:
        sections.append("## Op Progress\nNo progress data available.\n")

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
        "Based on the two separate analyses above, write a unified report with:\n\n"
        "### 1. Overall Health\n"
        "A brief (2-3 sentence) project-level summary.\n\n"
        "### 2. Per-Category Assessment\n"
        "For EACH of the 11 categories, write a short paragraph covering:\n"
        "- Current status (implementation %, test health, perf level)\n"
        "- Key issues specific to this category\n"
        "- **1-2 concrete recommendations** to improve this category\n"
        "- Composite scores for this category\n\n"
        "Use `## Category Name` as the header for each category section.\n\n"
        "### 3. Top 3 Project-Wide Recommendations\n"
        "Actionable items that cut across categories.\n\n"
        "### 4. Score Summary\n"
        "End with a scores block in this exact format:\n"
        "```\n"
        "category_scores:\n"
        + "".join(
            f"  {c}:\n"
            f"    perf_score: N\n"
            f"    func_score: N\n"
            f"    has_bugs: true/false\n"
            for c in _CATEGORIES
        )
        + "```\n"
        "where N is 1-5 or null if no data is available for that dimension."
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
    parser.add_argument("--manifest",      default=None, help="Path to op_manifest.json")
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

    # ── Load category structure ──────────────────────────────────────────────
    prog, _ = _load_progress_detailed(args.progress_json)
    progress_detail = _progress_category_detail(prog) if prog else None

    # Build test_fn → category map from manifest
    fn_to_cat = _build_test_fn_to_cat(args.manifest)

    # ── Phase 0: Op completion progress (pure data) ──────────────────────────
    print("Phase 0: Generating per-category completion statistics ...")
    progress_report = build_progress_report(prog)
    print(f"  Progress report: {len(progress_report)} chars")

    # ── Phase 1: Performance analysis (benchmark only, per-category) ─────────
    perf_analysis = None
    perf_prompt = build_perf_prompt(args.bench_log, progress_detail)
    if perf_prompt:
        print("Phase 1: Analyzing performance per category from benchmark log ...")
        try:
            perf_analysis = call_claude(
                perf_prompt, args.model, api_key, base_url,
                system=PERF_SYSTEM,
                max_tokens=3000,
                temperature=0.2,
            )
            print(f"  Performance analysis: {len(perf_analysis)} chars")
        except Exception as exc:
            print(f"::warning::Performance analysis failed: {exc}", file=sys.stderr)
    else:
        print("Phase 1: No benchmark log — skipping performance analysis.")

    # ── Phase 2: Correctness analysis (test XML, per-category) ───────────────
    correctness_analysis = None
    correctness_prompt = build_correctness_prompt(args.test_xml, fn_to_cat)
    if correctness_prompt:
        print("Phase 2: Analyzing correctness per category from test XML ...")
        try:
            correctness_analysis = call_claude(
                correctness_prompt, args.model, api_key, base_url,
                system=CORRECTNESS_SYSTEM,
                max_tokens=3000,
                temperature=0.2,
            )
            print(f"  Correctness analysis: {len(correctness_analysis)} chars")
        except Exception as exc:
            print(f"::warning::Correctness analysis failed: {exc}", file=sys.stderr)
    else:
        print("Phase 2: No test XML — skipping correctness analysis.")

    # ── Phase 3: Composite per-category scoring ──────────────────────────────
    print("Phase 3: Generating per-category composite analysis and scores ...")
    composite_prompt = build_composite_prompt(
        prog, perf_analysis, correctness_analysis,
    )
    try:
        analysis = call_claude(
            composite_prompt, args.model, api_key, base_url,
            system=COMPOSITE_SYSTEM,
            max_tokens=4096,
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

    # Prepend Phase 0 progress report to final analysis
    full_analysis = progress_report + "\n---\n\n" + analysis

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(full_analysis)
    print(f"Analysis written: {args.output} ({len(full_analysis)} chars)")


if __name__ == "__main__":
    main()
