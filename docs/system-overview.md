# TileOPs Nightly Report System

## Overview

This system automatically tracks the development progress of [TileOPs](https://github.com/tile-ai/TileOPs), a high-performance LLM operator library built on TileLang. It runs nightly CI pipelines that collect test/benchmark results, call Claude for structured analysis, and deliver an HTML report via email and GitHub Pages.

---

## Architecture

```
tile-ai/TileOPs (upstream)
  │
  │  nightly.yml (18:00 UTC daily)
  │  ├── pytest → test_results.xml + tileops_op_test.log
  │  └── benchmark → bench_results.xml + tileops_benchmarks.log
  │        (uploaded as GitHub artifacts)
  ▼
superAngGao/TileOPs-report
  │
  │  report.yml (22:00 UTC daily, 4h after upstream)
  │
  │  Step 1: Download artifacts from upstream
  │  Step 2: fetch_progress.py     → progress.json
  │  Step 3: update_registry.py    → op_registry.json + op_data/*.yaml
  │  Step 4: analyze_report.py     → analysis.json  (Claude API)
  │  Step 5: generate_report.py    → report.html
  │  Step 6: send_report_email.py  → email to gaoang0125@163.com
  │  Step 7: Push to gh-pages branch
  │
  │  registry_rescan.yml (20:00 UTC every Sunday)
  │  └── bootstrap_registry.py    → full rescan of file mappings (Claude API)
```

---

## Workflows

### 1. Nightly Report (`report.yml`)

**Schedule**: Every day at 22:00 UTC
**Trigger**: Also supports `workflow_dispatch` with two options:
- `run_id`: Specify a TileOPs nightly run ID (default: latest)
- `registry_only`: Update registry + regenerate report from existing data (no new artifacts)

**Steps**:

| # | Step | Script | Input | Output |
|---|------|--------|-------|--------|
| 1 | Find nightly run | `gh run list` | — | `run_id`, `sha` |
| 2 | Download artifacts | `dawidd6/action-download-artifact@v3` | upstream artifacts | `test_results.xml`, `bench_results.xml`, `.log` files |
| 3 | Collect progress | `fetch_progress.py` | TileOPs source + test/bench XML + manifest | `progress.json` |
| 4 | Update registry | `update_registry.py` | merged PRs + test/bench data | `op_registry.json`, `op_data/*.yaml` |
| 5 | Claude analysis | `analyze_report.py` | `progress.json` + test XML + bench log + manifest | `analysis.json` |
| 6 | Generate report | `generate_report.py` | `progress.json` + `analysis.json` | `report.html` |
| 7 | Send email | `send_report_email.py` | `report.html` | email |
| 8 | Deploy | git push | all report files | gh-pages branch |

**Deduplication**: Each run stores `run_id.txt` in the report directory. If a run ID has already been processed, the pipeline skips.

**Report retention**: Reports older than 14 days are automatically pruned.

### 2. Weekly Registry Rescan (`registry_rescan.yml`)

**Schedule**: Every Sunday at 20:00 UTC (before nightly report)
**Purpose**: Full rescan of TileOPs codebase to rebuild operator-to-file mappings.

Uses `bootstrap_registry.py` which:
1. Auto-maps op classes and test functions via regex (no Claude needed)
2. Calls Claude to identify kernel implementation files (semantic matching)
3. Outputs `op_registry.json` + `op_data/*.yaml`

---

## Scripts

### `fetch_progress.py`

**Purpose**: Scan TileOPs codebase and test/benchmark results to determine per-operator completion status.

**Logic**:
- Scans `tileops/` directory for all Python class definitions
- Parses `test_results.xml` (JUnit) to find passed/failed test functions
- Parses `bench_results.xml` (JUnit) to find passed/failed benchmarks
- Cross-references with `op_manifest.json` to map classes/functions to operators

**Completion definition**: An operator is "done" only when ALL THREE pass:
1. Implementation exists (op class found in codebase)
2. Tests pass (at least one mapped test function passes)
3. Benchmark passes (at least one mapped benchmark passes)

**Output**: `progress.json`
```json
{
  "total_ops": 186,
  "done_ops": 5,
  "impl_ops": 122,
  "tested_ops": 30,
  "benched_ops": 8,
  "categories": [
    {
      "name": "Elementwise",
      "total_ops": 72,
      "impl_ops": 70,
      "tested_ops": 12,
      "benched_ops": 3,
      "done_ops": 2,
      "ops": [
        {
          "name": "add",
          "implemented": true,
          "tested": true,
          "test_failed": false,
          "bench_ok": true,
          "status": "done"
        }
      ]
    }
  ]
}
```

### `analyze_report.py`

**Purpose**: Call Claude API once to produce structured per-category analysis.

**Input**: `progress.json` + `test_results.xml` + `tileops_benchmarks.log` + `op_manifest.json`

**Claude interaction**:
- Single API call using `call_claude_json()` (temperature=0.0 for stable JSON output)
- System prompt: "Respond with ONLY a valid JSON object"
- Model: `claude-sonnet-4-6` (configurable via `--model`)

**Output**: `analysis.json`
```json
{
  "categories": {
    "Elementwise": {
      "perf_score": 4,
      "func_score": 3,
      "issues": "Numerical precision issues in exp/log",
      "evaluation": "Strong impl coverage. Prioritize test coverage."
    },
    "Reduce": { ... },
    ...
  },
  "overall": {
    "summary": "Project health assessment...",
    "recommendations": ["item1", "item2", "item3"]
  }
}
```

**Scoring rubrics**:

| Score | perf_score (Performance) | func_score (Correctness) |
|-------|--------------------------|--------------------------|
| 1 | <50% of baseline | Most tests fail |
| 2 | 50-75% of baseline | Many failures, major gaps |
| 3 | 75-90% of baseline | Mixed results |
| 4 | 90-100% of baseline | Mostly passing |
| 5 | >100% of baseline | All tests pass |
| null | No benchmark data | No test data |

### `generate_report.py`

**Purpose**: Generate self-contained HTML report from structured data.

**Input**: `progress.json` + `analysis.json` + date/commit metadata

**Output**: `report.html` with 4 sections:
1. **Header** — Date, commit hash, generation timestamp
2. **Overall Progress** — Summary stats (total/impl/tested/benched/done) + 3 progress bars (Impl/Test/Bench)
3. **Category Progress** — For each of 11 categories:
   - 3 gradient progress bars (blue=Impl, green=Test, purple=Bench)
   - 2 score badges (Perf, Func) from Claude analysis
4. **Category Analysis** — Per-category issues and evaluation text
5. **Overall Assessment** — Project summary + numbered recommendations

No markdown-to-HTML conversion. Pure template filling from JSON data.

### `send_report_email.py`

**Purpose**: Send the HTML report as an email.

**Configuration** (via GitHub Actions secrets):

| Secret | Example | Description |
|--------|---------|-------------|
| `MAIL_SMTP_SERVER` | `smtp.163.com` | SMTP server |
| `MAIL_SMTP_PORT` | `465` | Port (465=SSL, 587=STARTTLS) |
| `MAIL_USERNAME` | `user@163.com` | Sender email |
| `MAIL_PASSWORD` | `(authorization code)` | 163 mail authorization code |

Recipient: `gaoang0125@163.com` (hardcoded in workflow)

### `bootstrap_registry.py`

**Purpose**: Initial setup and weekly full rescan of operator file mappings.

**Process**:
1. Scan `tileops/ops/` for op class definitions → auto-map to operators
2. Scan `tests/` for test function definitions → auto-map to operators
3. Read bench file paths from manifest
4. Call Claude to identify kernel files in `tileops/kernels/` (semantic matching, batched by category)

**Output**: `op_registry.json` (compact summary) + `op_data/<id>.yaml` (per-operator detail)

### `update_registry.py`

**Purpose**: Daily incremental registry update from merged PRs and test results.

**Process**:
1. Scan GitHub PRs merged since last update (via `gh` CLI)
2. Match changed file paths to affected operators
3. Update `kernel_last_pr_*` metadata for affected ops
4. Parse test results + benchmark logs for affected ops
5. Call Claude to re-evaluate `func_score`, `perf_score`, `has_bugs` (batched, 20 ops per call)

### `claude_utils.py`

**Purpose**: Shared Claude API utility module.

**Functions**:
- `get_api_config()` — Read API key + base URL from environment
- `call_claude()` — Call Claude Messages API with retry logic (2 retries, exponential backoff)
- `call_claude_json()` — Call + extract JSON + validate required keys
- `extract_json()` — Parse JSON from Claude response (handles ```json blocks, bare JSON, outermost `{}`)

**Predefined system prompts**: `SYSTEM_REPORT_ANALYZER`, `SYSTEM_SCORE_EVALUATOR`, `SYSTEM_KERNEL_MAPPER`

---

## Data Files

### `op_manifest.json`

Static manifest defining all 186 target operators across 11 categories. Each operator entry includes:
- `op_classes`: Class names to search for in code (implementation check)
- `test_fns`: Test function names to match in JUnit XML (test check)
- `bench_file`: Benchmark file path (bench check)

### `op_registry.json`

Runtime registry mapping each operator to its actual files in the TileOPs codebase:
```json
{
  "ops": {
    "ew.binary_arith.add": {
      "files": {
        "kernel": ["tileops/kernels/elementwise.py"],
        "op": ["tileops/ops/elementwise.py"],
        "tests": ["tests/ops/test_binary_arith.py::test_add_broadcast"],
        "bench": ["benchmarks/ops/bench_binary_arith.py"]
      },
      "func_score": 4,
      "perf_score": 3,
      "has_bugs": false,
      "kernel_last_pr_author": "...",
      "kernel_last_pr_title": "..."
    }
  }
}
```

### `op_data/*.yaml`

Per-operator YAML files (186 total) with detailed metadata including score notes, PR history, test status, and timestamps.

---

## Operator Categories

| # | Category | Ops | Difficulty |
|---|----------|----:|:----------:|
| 1 | Elementwise | 72 | ⭐ |
| 2 | Reduce | 20 | ⭐ |
| 3 | Norm | 10 | ⭐ |
| 4 | Conv & Pooling | 16 | ⭐ |
| 5 | GEMM | 19 | ⭐⭐ |
| 6 | Quantize | 10 | ⭐⭐ |
| 7 | Sampling | 7 | ⭐⭐ |
| 8 | Flash Attention | 16 | ⭐⭐⭐ |
| 9 | MoE | 6 | ⭐⭐⭐ |
| 10 | Linear Attention | 8 | ⭐⭐⭐ |
| 11 | SSM | 2 | ⭐⭐⭐ |

**Total: 186 operators**

---

## GitHub Actions Secrets Required

| Secret | Used by | Purpose |
|--------|---------|---------|
| `ANTHROPIC_API_KEY` | `analyze_report.py`, `update_registry.py`, `bootstrap_registry.py` | Claude API authentication |
| `ANTHROPIC_BASE_URL` | (optional) | Custom API proxy |
| `GITHUB_TOKEN` | (auto-provided) | Access upstream repo artifacts and PRs |
| `MAIL_SMTP_SERVER` | `send_report_email.py` | SMTP server address |
| `MAIL_SMTP_PORT` | `send_report_email.py` | SMTP port |
| `MAIL_USERNAME` | `send_report_email.py` | Sender email |
| `MAIL_PASSWORD` | `send_report_email.py` | SMTP authorization code |

---

## Deployment

Reports are deployed to the `gh-pages` branch using orphan commits (force push). Each report lives in `nightly/<YYYYMMDD_HHMMSS>/` with an auto-generated `index.html` listing all reports.

GitHub Pages URL: `https://superanggao.github.io/TileOPs-report/nightly/`
