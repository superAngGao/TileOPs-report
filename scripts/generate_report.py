#!/usr/bin/env python3
"""generate_report.py — 将 pytest junit XML + benchmark markdown + 开发进度合并为综合报告。

生成两种格式：
  - <output>        — Markdown 报告
  - <html-output>   — 自包含 HTML 报告（可直接用浏览器打开或由 HTTP server 提供）
"""

import argparse
import json
import os
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# 一、测试结果解析
# ──────────────────────────────────────────────────────────────────────────────

def parse_junit(xml_path: str) -> dict:
    if not os.path.exists(xml_path):
        return {"error": "测试结果文件未找到"}
    tree = ET.parse(xml_path)
    root = tree.getroot()
    suites = list(root) if root.tag == "testsuites" else [root]
    total = errors = failures = skipped = 0
    failed_tests = []
    for suite in suites:
        total    += int(suite.get("tests",    0))
        errors   += int(suite.get("errors",   0))
        failures += int(suite.get("failures", 0))
        skipped  += int(suite.get("skipped",  0))
        for tc in suite.iter("testcase"):
            failure = tc.find("failure")
            error   = tc.find("error")
            if failure is not None or error is not None:
                classname = tc.get("classname", "")
                name = tc.get("name", "")
                node = f"{classname}::{name}" if classname else name
                elem = failure if failure is not None else error
                message = (elem.get("message") or "")[:300]
                text    = (elem.text or "").strip()[:500]
                failed_tests.append({"node": node, "message": message, "text": text})
    passed = total - errors - failures - skipped
    return {
        "total": total, "passed": passed,
        "failed": failures + errors, "skipped": skipped,
        "failed_tests": failed_tests,
    }


def read_file_tail(path: Path, n: int = 50) -> list[str]:
    if not path.exists():
        return []
    return path.read_text().splitlines()[-n:]


# ──────────────────────────────────────────────────────────────────────────────
# 二、开发进度节（Markdown）
# ──────────────────────────────────────────────────────────────────────────────

_STATUS_ICON = {
    "done":      "✅",
    "partial":   "🔧",
    "impl_only": "🔧",
    "missing":   "🔲",
}

_DIFFICULTY_STAR = {1: "⭐", 2: "⭐⭐", 3: "⭐⭐⭐"}


def _cat_status_label(cat: dict) -> str:
    done  = cat["done_ops"]
    total = cat["total_ops"]
    impl  = cat["impl_ops"]
    if done == total:
        return "✅ Done"
    if impl > 0 or done > 0:
        return f"🔧 In Progress ({done}/{total})"
    return "🔲 Not Started"


def build_progress_section(progress_json_path: str | None) -> list[str]:
    if not progress_json_path or not os.path.exists(progress_json_path):
        return [
            "## 开发进度",
            "",
            "> 进度数据未生成（fetch_progress.py 未运行或运行失败）。",
            "",
        ]

    prog = json.loads(Path(progress_json_path).read_text())
    done  = prog["done_ops"]
    total = prog["total_ops"]
    impl  = prog["impl_ops"]
    pct   = done / total * 100 if total else 0

    lines: list[str] = [
        "## 开发进度",
        "",
        f"> 数据来源：[issue #407](https://github.com/tile-ai/TileOPs/issues/407)  ",
        f"> 更新时间：{prog['generated_at']}",
        "",
        f"**总进度：{done} / {total} ops 已完成（{pct:.1f}%）**  "
        f"（已实现: {impl}，已通过测试: {prog['tested_ops']}）",
        "",
        "| # | Category | Ops | Difficulty | Issue | 状态 |",
        "|---|---|---|---|---|---|",
    ]

    for cat in prog["categories"]:
        diff = _DIFFICULTY_STAR.get(cat["difficulty"], "")
        status = _cat_status_label(cat)
        lines.append(
            f"| {cat['id']} | {cat['name']} | {cat['total_ops']} "
            f"| {diff} | #{cat['issue']} | {status} |"
        )

    lines += ["", "### 各分类 op 详情", ""]
    for cat in prog["categories"]:
        done_c  = cat["done_ops"]
        total_c = cat["total_ops"]
        pct_c   = done_c / total_c * 100 if total_c else 0
        lines += [
            f"<details>",
            f"<summary><b>{cat['name']}</b> — {done_c}/{total_c} ops "
            f"({pct_c:.0f}%) [{_cat_status_label(cat)}]</summary>",
            "",
            "| Op | Sub-category | 实现 | 测试 | Benchmark |",
            "|---|---|:---:|:---:|:---:|",
        ]
        for op in cat["ops"]:
            impl_icon  = "✅" if op["implemented"] else "🔲"
            test_icon  = "✅" if op["tested"] else ("❌" if op["test_failed"] else "—")
            bench_icon = "✅" if op["bench_ok"] is True else ("❌" if op["bench_ok"] is False else "—")
            lines.append(
                f"| `{op['name']}` | {op['sub']} | {impl_icon} | {test_icon} | {bench_icon} |"
            )
        lines += ["", "</details>", ""]

    return lines


# ──────────────────────────────────────────────────────────────────────────────
# 三、Markdown 报告组装
# ──────────────────────────────────────────────────────────────────────────────

def build_report(args) -> str:
    report_dir = Path(args.report_dir)
    lines: list[str] = []

    # 标题
    lines += [
        "# TileOPs 夜测综合报告",
        "",
        f"- **日期**: {args.date}",
        f"- **Commit**: `{args.commit}`",
        f"- **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    # 总体状态
    test_ok  = int(args.test_exit)  == 0
    bench_ok = int(args.bench_exit) == 0
    overall  = "✅ PASS" if (test_ok and bench_ok) else "❌ FAIL"
    lines += [
        "## 总体状态",
        "",
        "| 项目 | 状态 |",
        "|------|------|",
        f"| 测试（pytest） | {'✅ PASS' if test_ok else '❌ FAIL'} |",
        f"| Benchmark | {'✅ PASS' if bench_ok else '❌ FAIL（或已跳过）'} |",
        f"| **综合** | **{overall}** |",
        "",
    ]

    # 开发进度节
    lines += build_progress_section(getattr(args, "progress_json", None))

    # 测试结果
    lines += ["## 测试结果", ""]
    stats = parse_junit(str(report_dir / "test_results.xml"))
    if "error" in stats:
        lines.append(f"> {stats['error']}")
    else:
        pass_rate = (
            f"{stats['passed'] / stats['total'] * 100:.1f}%"
            if stats["total"] > 0 else "N/A"
        )
        lines += [
            "| 总计 | 通过 | 失败 | 跳过 | 通过率 |",
            "|------|------|------|------|--------|",
            f"| {stats['total']} | {stats['passed']} | {stats['failed']} "
            f"| {stats['skipped']} | {pass_rate} |",
            "",
        ]
        if stats["failed_tests"]:
            lines += ["### 失败用例", ""]
            for t in stats["failed_tests"]:
                lines.append(f"#### `{t['node']}`")
                if t["message"]:
                    lines += ["```", t["message"], "```"]
                if t["text"] and t["text"] != t["message"]:
                    lines += [
                        "<details><summary>详细信息</summary>",
                        "", "```", t["text"], "```", "</details>",
                    ]
                lines.append("")
        else:
            lines += ["**所有用例均通过。**", ""]

    # Benchmark 结果
    lines += ["## Benchmark 结果", ""]
    bench_path = report_dir / "benchmark_report.md"
    if bench_path.exists():
        lines += (report_dir / "benchmark_report.md").read_text().splitlines()[2:]
    else:
        lines.append("> Benchmark 报告未生成（可能已跳过或运行失败）。")
    lines.append("")

    # 附录：测试输出末 50 行
    test_log = report_dir / "test_output.log"
    if test_log.exists():
        lines += [
            "## 附录：测试输出（末 50 行）",
            "", "```",
            *read_file_tail(test_log, 50),
            "```", "",
        ]

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# 四、HTML 报告生成
# ──────────────────────────────────────────────────────────────────────────────

_HTML_CSS = """
:root{--green:#22c55e;--yellow:#f59e0b;--red:#ef4444;--gray:#6b7280;--blue:#3b82f6;
      --bg:#f9fafb;--card:#fff;--border:#e5e7eb;--text:#111827;--muted:#6b7280}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,sans-serif;background:var(--bg);color:var(--text);padding:1rem}
h1{font-size:1.6rem;font-weight:700;margin-bottom:.25rem}
h2{font-size:1.1rem;font-weight:600;margin:1.5rem 0 .5rem;border-bottom:2px solid var(--border);padding-bottom:.25rem}
h3{font-size:.95rem;font-weight:600;margin:.75rem 0 .25rem}
.meta{color:var(--muted);font-size:.8rem;margin-bottom:1rem}
.card{background:var(--card);border:1px solid var(--border);border-radius:.5rem;padding:1rem;margin-bottom:1rem}
.badge{display:inline-block;padding:.15rem .5rem;border-radius:.25rem;font-size:.75rem;font-weight:600}
.badge-pass{background:#dcfce7;color:#166534}.badge-fail{background:#fee2e2;color:#991b1b}
.badge-skip{background:#fef3c7;color:#92400e}.badge-gray{background:#f3f4f6;color:#374151}
table{width:100%;border-collapse:collapse;font-size:.82rem;margin:.5rem 0}
th{background:#f3f4f6;text-align:left;padding:.4rem .6rem;border:1px solid var(--border);font-weight:600}
td{padding:.35rem .6rem;border:1px solid var(--border);vertical-align:middle}
tr:nth-child(even){background:#fafafa}
.prog-wrap{background:#e5e7eb;border-radius:9999px;height:.75rem;min-width:80px;overflow:hidden}
.prog-bar{height:100%;border-radius:9999px;transition:width .3s}
.prog-done{background:var(--green)}.prog-impl{background:var(--yellow)}.prog-miss{background:#d1d5db}
.icon-done{color:var(--green)}.icon-part{color:var(--yellow)}.icon-miss{color:var(--gray)}
details>summary{cursor:pointer;user-select:none;padding:.35rem .5rem;border-radius:.25rem;font-size:.85rem}
details>summary:hover{background:#f3f4f6}
details[open]>summary{font-weight:600}
.op-table td:nth-child(3),.op-table td:nth-child(4),.op-table td:nth-child(5){text-align:center;width:60px}
pre{background:#1e1e1e;color:#d4d4d4;padding:.75rem;border-radius:.375rem;overflow-x:auto;font-size:.78rem;line-height:1.5}
.summary-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:.75rem;margin:.75rem 0}
.stat-box{background:var(--card);border:1px solid var(--border);border-radius:.375rem;padding:.75rem;text-align:center}
.stat-box .num{font-size:1.6rem;font-weight:700;line-height:1}.stat-box .lbl{font-size:.72rem;color:var(--muted);margin-top:.2rem}
"""

_HTML_JS = """
function toggle(id){var el=document.getElementById(id);el.open=!el.open}
"""


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _progress_bar_html(done: int, impl: int, total: int) -> str:
    if total == 0:
        return ""
    pct_done = done / total * 100
    pct_impl = (impl - done) / total * 100
    return (
        f'<div class="prog-wrap" title="{done}/{total} done, {impl} impl">'
        f'<div class="prog-bar prog-done" style="width:{pct_done:.1f}%;float:left"></div>'
        f'<div class="prog-bar prog-impl" style="width:{pct_impl:.1f}%;float:left"></div>'
        f'</div>'
    )


def _cat_status_html(cat: dict) -> str:
    done  = cat["done_ops"]
    total = cat["total_ops"]
    impl  = cat["impl_ops"]
    if done == total:
        return '<span class="icon-done">✅ Done</span>'
    if impl > 0 or done > 0:
        return f'<span class="icon-part">🔧 In Progress ({done}/{total})</span>'
    return '<span class="icon-miss">🔲 Not Started</span>'


def _op_icon(flag: bool | None, fail: bool = False) -> str:
    if flag is True:
        return '<span style="color:var(--green)">✅</span>'
    if fail:
        return '<span style="color:var(--red)">❌</span>'
    if flag is False:
        return '<span style="color:var(--red)">❌</span>'
    return '<span style="color:var(--gray)">—</span>'


def build_html_report(args, prog: dict | None, test_stats: dict) -> str:
    date_str  = args.date
    commit    = args.commit
    gen_time  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    test_ok   = int(args.test_exit)  == 0
    bench_ok  = int(args.bench_exit) == 0
    overall   = "PASS" if (test_ok and bench_ok) else "FAIL"
    ov_cls    = "badge-pass" if overall == "PASS" else "badge-fail"

    # ── 进度摘要数字 ──────────────────────────────────────────────────────────
    if prog:
        done_ops   = prog["done_ops"]
        total_ops  = prog["total_ops"]
        impl_ops   = prog["impl_ops"]
        tested_ops = prog["tested_ops"]
        pct        = done_ops / total_ops * 100 if total_ops else 0
        prog_ts    = prog.get("generated_at", "")
    else:
        done_ops = impl_ops = tested_ops = total_ops = 0
        pct = 0.0
        prog_ts = ""

    # ── HTML 骨架 ─────────────────────────────────────────────────────────────
    html_parts = [
        "<!DOCTYPE html><html lang='zh-CN'><head>",
        "<meta charset='UTF-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        f"<title>TileOPs 夜测报告 · {date_str}</title>",
        f"<style>{_HTML_CSS}</style>",
        f"<script>{_HTML_JS}</script>",
        "</head><body>",
        f"<h1>TileOPs 夜测综合报告</h1>",
        f"<p class='meta'>日期: {_esc(date_str)} &nbsp;|&nbsp; Commit: <code>{_esc(commit)}</code>"
        f" &nbsp;|&nbsp; 生成时间: {gen_time}</p>",
    ]

    # ── 总体状态卡 ────────────────────────────────────────────────────────────
    html_parts += [
        "<div class='card'>",
        "<h2 style='margin-top:0'>总体状态</h2>",
        "<div class='summary-grid'>",
        f"<div class='stat-box'><div class='num'><span class='badge {ov_cls}'>{overall}</span></div>"
        f"<div class='lbl'>综合结果</div></div>",
        f"<div class='stat-box'><div class='num'><span class='badge {'badge-pass' if test_ok else 'badge-fail'}'>"
        f"{'PASS' if test_ok else 'FAIL'}</span></div><div class='lbl'>测试 (pytest)</div></div>",
        f"<div class='stat-box'><div class='num'><span class='badge {'badge-pass' if bench_ok else 'badge-skip'}'>"
        f"{'PASS' if bench_ok else 'FAIL/SKIP'}</span></div><div class='lbl'>Benchmark</div></div>",
    ]
    if not ("error" in test_stats):
        prate = f"{test_stats['passed']/test_stats['total']*100:.1f}%" if test_stats["total"] else "N/A"
        html_parts += [
            f"<div class='stat-box'><div class='num'>{test_stats['total']}</div><div class='lbl'>总测试数</div></div>",
            f"<div class='stat-box'><div class='num' style='color:var(--green)'>{test_stats['passed']}</div><div class='lbl'>通过</div></div>",
            f"<div class='stat-box'><div class='num' style='color:var(--red)'>{test_stats['failed']}</div><div class='lbl'>失败</div></div>",
            f"<div class='stat-box'><div class='num'>{prate}</div><div class='lbl'>通过率</div></div>",
        ]
    html_parts += ["</div></div>"]  # summary-grid + card

    # ── 开发进度节 ────────────────────────────────────────────────────────────
    if prog:
        html_parts += [
            "<div class='card'>",
            "<h2 style='margin-top:0'>开发进度</h2>",
            f"<p style='font-size:.82rem;color:var(--muted)'>数据来源: "
            f"<a href='https://github.com/tile-ai/TileOPs/issues/407'>#407</a>"
            f" &nbsp;·&nbsp; 更新: {_esc(prog_ts)}</p>",
            "<div class='summary-grid' style='margin-top:.75rem'>",
            f"<div class='stat-box'><div class='num'>{done_ops}/{total_ops}</div><div class='lbl'>已完成 ops</div></div>",
            f"<div class='stat-box'><div class='num' style='color:var(--yellow)'>{impl_ops}</div><div class='lbl'>已实现 (含未测)</div></div>",
            f"<div class='stat-box'><div class='num' style='color:var(--blue)'>{tested_ops}</div><div class='lbl'>测试通过</div></div>",
            f"<div class='stat-box'><div class='num'>{pct:.1f}%</div><div class='lbl'>完成率</div></div>",
            "</div>",
            # 全局进度条
            "<div style='margin:.75rem 0'>",
            _progress_bar_html(done_ops, impl_ops, total_ops),
            f"<div style='font-size:.75rem;color:var(--muted);margin-top:.25rem'>"
            f"<span style='color:var(--green)'>■</span> 完成 &nbsp;"
            f"<span style='color:var(--yellow)'>■</span> 已实现未达标 &nbsp;"
            f"<span style='color:#d1d5db'>■</span> 未开始</div>",
            "</div>",
            # 各分类表格
            "<table>",
            "<thead><tr><th>#</th><th>Category</th><th>Ops</th>"
            "<th>Difficulty</th><th>Issue</th><th>进度</th><th>状态</th></tr></thead>",
            "<tbody>",
        ]
        diff_map = {1: "⭐", 2: "⭐⭐", 3: "⭐⭐⭐"}
        for cat in prog["categories"]:
            bar = _progress_bar_html(cat["done_ops"], cat["impl_ops"], cat["total_ops"])
            html_parts.append(
                f"<tr><td>{cat['id']}</td><td><b>{_esc(cat['name'])}</b></td>"
                f"<td>{cat['total_ops']}</td>"
                f"<td>{diff_map.get(cat['difficulty'],'')}</td>"
                f"<td><a href='https://github.com/tile-ai/TileOPs/issues/{cat['issue']}'>#{cat['issue']}</a></td>"
                f"<td style='min-width:120px'>{bar}"
                f"<div style='font-size:.72rem;color:var(--muted)'>{cat['done_ops']}/{cat['total_ops']}</div></td>"
                f"<td>{_cat_status_html(cat)}</td></tr>"
            )
        html_parts += ["</tbody></table>", ""]

        # 各分类 op 明细（折叠）
        html_parts.append("<h3 style='margin-top:1rem'>各分类 op 明细</h3>")
        for cat in prog["categories"]:
            done_c  = cat["done_ops"]
            total_c = cat["total_ops"]
            pct_c   = done_c / total_c * 100 if total_c else 0
            html_parts += [
                f"<details>",
                f"<summary>{_esc(cat['name'])} — {done_c}/{total_c} ops ({pct_c:.0f}%)"
                f" {_cat_status_html(cat)}</summary>",
                "<table class='op-table'>",
                "<thead><tr><th>Op</th><th>Sub-category</th>"
                "<th>实现</th><th>测试</th><th>Bench</th></tr></thead>",
                "<tbody>",
            ]
            for op in cat["ops"]:
                impl_icon  = _op_icon(op["implemented"])
                test_icon  = _op_icon(op["tested"], fail=op["test_failed"])
                bench_icon = _op_icon(op["bench_ok"])
                html_parts.append(
                    f"<tr><td><code>{_esc(op['name'])}</code></td>"
                    f"<td>{_esc(op['sub'])}</td>"
                    f"<td>{impl_icon}</td><td>{test_icon}</td><td>{bench_icon}</td></tr>"
                )
            html_parts += ["</tbody></table></details>"]

        html_parts.append("</div>")  # close card
    else:
        html_parts += [
            "<div class='card'><h2 style='margin-top:0'>开发进度</h2>",
            "<p>进度数据未生成（fetch_progress.py 未运行或运行失败）。</p></div>",
        ]

    # ── 测试结果节 ────────────────────────────────────────────────────────────
    html_parts += ["<div class='card'>", "<h2 style='margin-top:0'>测试结果</h2>"]
    if "error" in test_stats:
        html_parts.append(f"<p>{_esc(test_stats['error'])}</p>")
    else:
        prate = f"{test_stats['passed']/test_stats['total']*100:.1f}%" if test_stats["total"] else "N/A"
        html_parts += [
            "<table><thead><tr><th>总计</th><th>通过</th><th>失败</th><th>跳过</th><th>通过率</th></tr></thead>",
            f"<tbody><tr><td>{test_stats['total']}</td>"
            f"<td style='color:var(--green)'>{test_stats['passed']}</td>"
            f"<td style='color:var(--red)'>{test_stats['failed']}</td>"
            f"<td>{test_stats['skipped']}</td>"
            f"<td><b>{prate}</b></td></tr></tbody></table>",
        ]
        if test_stats["failed_tests"]:
            html_parts.append("<h3>失败用例</h3>")
            for t in test_stats["failed_tests"]:
                html_parts += [
                    f"<details><summary><code>{_esc(t['node'])}</code></summary>",
                    f"<pre>{_esc(t['message'])}</pre>",
                ]
                if t["text"] and t["text"] != t["message"]:
                    html_parts.append(f"<pre>{_esc(t['text'])}</pre>")
                html_parts.append("</details>")
        else:
            html_parts.append("<p style='color:var(--green);font-weight:600'>所有用例均通过。</p>")
    html_parts.append("</div>")

    # ── Benchmark 结果节 ──────────────────────────────────────────────────────
    report_dir = Path(args.report_dir)
    bench_path = report_dir / "benchmark_report.md"
    html_parts += ["<div class='card'>", "<h2 style='margin-top:0'>Benchmark 结果</h2>"]
    if bench_path.exists():
        bench_lines = bench_path.read_text().splitlines()[2:]
        html_parts.append("<pre>" + _esc("\n".join(bench_lines)) + "</pre>")
    else:
        html_parts.append("<p>Benchmark 报告未生成（可能已跳过或运行失败）。</p>")
    html_parts.append("</div>")

    # ── 附录：测试日志末 50 行 ─────────────────────────────────────────────────
    test_log = report_dir / "test_output.log"
    if test_log.exists():
        tail = read_file_tail(test_log, 50)
        html_parts += [
            "<div class='card'>",
            "<h2 style='margin-top:0'>附录：测试输出（末 50 行）</h2>",
            "<pre>" + _esc("\n".join(tail)) + "</pre>",
            "</div>",
        ]

    html_parts += ["</body></html>"]
    return "\n".join(html_parts)


# ──────────────────────────────────────────────────────────────────────────────
# 五、入口
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="生成 TileOPs 夜测综合报告")
    parser.add_argument("--date",          required=True, help="运行日期时间戳")
    parser.add_argument("--commit",        required=True, help="Git commit hash")
    parser.add_argument("--report-dir",    required=True, help="报告目录")
    parser.add_argument("--test-exit",     default="0",   help="pytest 退出码")
    parser.add_argument("--bench-exit",    default="0",   help="benchmark 退出码")
    parser.add_argument("--output",        required=True, help="Markdown 报告输出路径")
    parser.add_argument("--progress-json", default=None,  help="progress.json 路径（可选）")
    parser.add_argument("--html-output",   default=None,  help="HTML 报告输出路径（可选）")
    args = parser.parse_args()

    # Markdown
    md_content = build_report(args)
    Path(args.output).write_text(md_content)
    print(f"Markdown 报告已保存：{args.output}")

    # HTML（如果指定了路径）
    if args.html_output:
        prog = None
        if args.progress_json and os.path.exists(args.progress_json):
            prog = json.loads(Path(args.progress_json).read_text())
        test_stats = parse_junit(str(Path(args.report_dir) / "test_results.xml"))
        html_content = build_html_report(args, prog, test_stats)
        Path(args.html_output).write_text(html_content)
        print(f"HTML  报告已保存：{args.html_output}")


if __name__ == "__main__":
    main()
