#!/usr/bin/env python3
"""analyze_report.py — Call Claude API to analyze nightly TileOPs report.

Single-call analysis: reads progress.json, test_results.xml, and benchmark
log; calls Claude once to produce structured analysis.json with per-category
scores and evaluations.

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
    call_claude_json,
    get_api_config,
    require_anthropic,
)

# Category names (must match op_manifest.json)
_CATEGORIES = [
    "Elementwise", "Reduce", "Norm", "Conv & Pooling", "GEMM",
    "Quantize", "Sampling", "Flash Attention", "MoE",
    "Linear Attention", "SSM",
]


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────────────

def _load_progress_detail(path: str | None) -> str:
    """Load progress.json and format per-category, per-op status for the prompt."""
    if not path or not Path(path).exists():
        return "No progress data available.\n"
    try:
        prog = json.loads(Path(path).read_text())
        lines = [
            f"Total ops: {prog['total_ops']} | "
            f"Implemented: {prog['impl_ops']} | "
            f"Tested: {prog['tested_ops']} | "
            f"Benched: {prog.get('benched_ops', 0)} | "
            f"Done: {prog['done_ops']}",
            "",
        ]
        for cat in prog.get("categories", []):
            t  = cat["total_ops"]
            im = cat["impl_ops"]
            te = cat.get("tested_ops", 0)
            be = cat.get("benched_ops", 0)
            d  = cat["done_ops"]
            lines.append(f"### {cat['name']} ({im}/{t} impl, {te}/{t} tested, {be}/{t} benched, {d}/{t} done)")
            for op in cat.get("ops", []):
                impl_s  = "Y" if op.get("implemented") else "N"
                test_s  = "pass" if op.get("tested") else ("FAIL" if op.get("test_failed") else "-")
                bench_s = "pass" if op.get("bench_ok") is True else ("FAIL" if op.get("bench_ok") is False else "-")
                lines.append(f"  {op['name']} [{op.get('sub','')}]: impl={impl_s} test={test_s} bench={bench_s}")
            lines.append("")
        return "\n".join(lines)
    except Exception:
        return "Failed to load progress data.\n"


def _build_test_fn_to_cat(manifest_path: str | None) -> dict[str, str]:
    """Build test_fn -> category_name map from op_manifest.json."""
    if not manifest_path or not Path(manifest_path).exists():
        return {}
    try:
        manifest = json.loads(Path(manifest_path).read_text())
        fn_to_cat: dict[str, str] = {}
        for cat in manifest.get("categories", []):
            for op in cat.get("ops", []):
                for fn in op.get("test_fns", []):
                    fn_to_cat[fn] = cat["name"]
        return fn_to_cat
    except Exception:
        return {}


def _load_test_summary(xml_path: str | None, fn_to_cat: dict[str, str]) -> str:
    """Load test_results.xml and return per-category test summary text."""
    if not xml_path or not Path(xml_path).exists():
        return "No test data available.\n"
    try:
        root = ET.parse(xml_path).getroot()
        suites = list(root) if root.tag == "testsuites" else [root]
        total = errors = failures = skipped = 0
        cat_data: dict[str, dict] = {}

        for suite in suites:
            total    += int(suite.get("tests",    0))
            errors   += int(suite.get("errors",   0))
            failures += int(suite.get("failures", 0))
            skipped  += int(suite.get("skipped",  0))
            for tc in suite.iter("testcase"):
                name = tc.get("name", "")
                bare_fn = name.split("[")[0]
                fail_el = tc.find("failure")
                err_el  = tc.find("error")
                is_fail = fail_el is not None or err_el is not None

                cat_name = fn_to_cat.get(bare_fn, "Other")
                cd = cat_data.setdefault(cat_name, {"passed": 0, "failed": 0, "errors": []})
                if is_fail:
                    cd["failed"] += 1
                    elem = fail_el if fail_el is not None else err_el
                    msg = (elem.get("message") or "")[:200]
                    if len(cd["errors"]) < 5:
                        cd["errors"].append(f"{name}: {msg}")
                else:
                    cd["passed"] += 1

        passed = total - errors - failures - skipped
        lines = [f"Total: {total} | Passed: {passed} | Failed: {failures + errors} | Skipped: {skipped}", ""]
        for cn in sorted(cat_data.keys()):
            cd = cat_data[cn]
            lines.append(f"### {cn}: {cd['passed']} passed, {cd['failed']} failed")
            for e in cd["errors"]:
                lines.append(f"  - {e}")
            lines.append("")
        return "\n".join(lines)
    except Exception:
        return "Failed to parse test data.\n"


def _load_bench_log(path: str | None) -> str:
    """Load benchmark log (last 300 lines)."""
    if not path or not Path(path).exists():
        return "No benchmark data available.\n"
    try:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
        log_lines = text.splitlines()
        if len(log_lines) > 300:
            log_lines = log_lines[-300:]
        return "\n".join(log_lines) + "\n"
    except Exception:
        return "Failed to load benchmark data.\n"


# ─────────────────────────────────────────────────────────────────────────────
# Prompt + System
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a senior CI/DevOps engineer analyzing nightly test results for TileOPs, "
    "a high-performance LLM operator library built on TileLang. "
    "You MUST respond with ONLY a valid JSON object. No markdown, no explanation, "
    "no text outside the JSON. The JSON must match the exact schema specified."
)

_CATEGORY_JSON_TEMPLATE = ",\n".join(
    f'    "{c}": {{"perf_score": <1-5 or null>, "func_score": <1-5 or null>, "issues": "<text>", "evaluation": "<text>"}}'
    for c in _CATEGORIES
)


def build_prompt(progress_text: str, test_text: str, bench_text: str) -> str:
    cat_list = ", ".join(_CATEGORIES)
    return f"""Analyze the nightly results for TileOPs.

## Operator Categories
{cat_list}

## Implementation Progress
{progress_text}

## Test Results
{test_text}

## Benchmark Log
{bench_text}

## Instructions

Score each category on TWO dimensions:

**perf_score** (performance, from benchmarks):
- 1 = <50% of baseline on average
- 2 = 50-75% of baseline
- 3 = 75-90% of baseline
- 4 = 90-100% of baseline
- 5 = >100% of baseline (exceeds)
- null = no benchmark data for this category

**func_score** (functional correctness, from tests):
- 1 = Most tests fail, fundamental issues
- 2 = Many failures, major gaps
- 3 = Mixed results, some ops broken
- 4 = Mostly passing, minor issues
- 5 = All or nearly all tests pass
- null = no test data for this category

For each category, provide:
- "perf_score": performance score 1-5 or null
- "func_score": functional score 1-5 or null
- "issues": concise description of problems (empty string if none)
- "evaluation": brief assessment of current status and 1-2 actionable recommendations

For the overall project, provide:
- "summary": 2-3 sentence project health assessment
- "recommendations": list of 3 top actionable items

Respond with ONLY this JSON (no other text):
{{
  "categories": {{
{_CATEGORY_JSON_TEMPLATE}
  }},
  "overall": {{
    "summary": "<text>",
    "recommendations": ["<item1>", "<item2>", "<item3>"]
  }}
}}"""


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Call Claude API to analyze nightly TileOPs report, output analysis.json"
    )
    parser.add_argument("--progress-json", default=None, help="Path to progress.json")
    parser.add_argument("--test-xml",      default=None, help="Path to test_results.xml")
    parser.add_argument("--bench-log",     default=None, help="Path to tileops_benchmarks.log")
    parser.add_argument("--manifest",      default=None, help="Path to op_manifest.json")
    parser.add_argument("--output",        required=True, help="Output path for analysis.json")
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

    # Load data
    progress_text = _load_progress_detail(args.progress_json)
    fn_to_cat = _build_test_fn_to_cat(args.manifest)
    test_text = _load_test_summary(args.test_xml, fn_to_cat)
    bench_text = _load_bench_log(args.bench_log)

    # Single Claude call → structured JSON
    prompt = build_prompt(progress_text, test_text, bench_text)
    print("Calling Claude for structured analysis ...")

    try:
        analysis = call_claude_json(
            prompt, args.model, api_key, base_url,
            system=SYSTEM_PROMPT,
            required_keys=["categories", "overall"],
            max_tokens=4096,
        )
    except Exception as exc:
        print(f"::warning::Claude analysis failed: {exc}", file=sys.stderr)
        # Fallback: empty structure
        analysis = {
            "categories": {c: {"perf_score": None, "func_score": None, "issues": "", "evaluation": ""} for c in _CATEGORIES},
            "overall": {"summary": "Analysis unavailable.", "recommendations": []},
        }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(analysis, ensure_ascii=False, indent=2))
    print(f"Analysis written: {args.output}")


if __name__ == "__main__":
    main()
