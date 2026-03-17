#!/usr/bin/env python3
"""update_registry.py — 每日更新算子注册表。

执行流程：
  1. 扫描 TileOPs 中当日（或指定日期后）merge 的 PR
  2. 将 PR 涉及的文件匹配到对应算子，更新 kernel 变更元数据
  3. 解析 test_results.xml，更新各算子的测试状态
  4. 解析 benchmark 日志，提取性能数据
  5. 调用 Claude 对有变化的算子重新评估 func_score / perf_score / has_bugs
  6. 将更新写回 op_registry.json 和 op_data/*.yaml

用法：
    python scripts/update_registry.py \\
        --registry    scripts/op_registry.json \\
        --op-data-dir scripts/op_data/ \\
        --test-xml    path/to/test_results.xml \\
        --bench-log   path/to/tileops_benchmarks.log \\
        --tileops-repo tile-ai/TileOPs \\
        [--since-date YYYY-MM-DD]   # 默认昨天

环境变量：
    ANTHROPIC_API_KEY 或 ANTHROPIC_AUTH_TOKEN  必需（用于 Claude 评估）
    ANTHROPIC_BASE_URL                          可选
    GH_TOKEN                                    必需（用于 gh CLI 扫描 PR）
"""

import argparse
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from claude_utils import (
    call_claude_json,
    get_api_config,
    require_anthropic,
    SYSTEM_SCORE_EVALUATOR,
)

try:
    import yaml as _yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

NOW_ISO  = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
NOW_DATE = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
TODAY    = datetime.now(tz=timezone.utc).date()


# ─────────────────────────────────────────────────────────────────────────────
# 1. 注册表 I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_registry(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        print(f"⚠ op_registry.json 不存在: {path}", file=sys.stderr)
        return {"schema_version": "2.0", "ops": {}}
    return json.loads(p.read_text())


def save_registry(registry: dict, path: str) -> None:
    registry["generated_at"] = NOW_ISO
    Path(path).write_text(
        json.dumps(registry, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_op_yaml(op_data_dir: str, op_id: str) -> dict | None:
    path = Path(op_data_dir) / f"{op_id}.yaml"
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    if HAS_YAML:
        return _yaml.safe_load(text) or {}
    # 降级：尝试 JSON 解析（bootstrap 可能写的是 JSON）
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def save_op_yaml(op_data_dir: str, op_id: str, data: dict) -> None:
    path = Path(op_data_dir) / f"{op_id}.yaml"
    if not path.exists():
        return  # 不创建新文件，bootstrap 负责创建
    if HAS_YAML:
        path.write_text(
            _yaml.dump(data, allow_unicode=True, default_flow_style=False,
                       sort_keys=False, width=120),
            encoding="utf-8",
        )
    else:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def deep_update(base: dict, updates: dict) -> dict:
    """递归合并 updates 到 base（返回 base 的引用）。"""
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


# ─────────────────────────────────────────────────────────────────────────────
# 2. GitHub PR 扫描
# ─────────────────────────────────────────────────────────────────────────────

def _gh(args_list: list[str], timeout: int = 60) -> str | None:
    """运行 gh CLI 命令，返回 stdout 或 None（失败时）。"""
    try:
        result = subprocess.run(
            ["gh"] + args_list,
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            print(f"  gh 命令失败: {' '.join(args_list[:4])}: {result.stderr[:200]}",
                  file=sys.stderr)
            return None
        return result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        print(f"  gh CLI 不可用: {exc}", file=sys.stderr)
        return None


def get_merged_prs(tileops_repo: str, since_date: str) -> list[dict]:
    """返回 since_date 之后 merge 进 main 的 PR 列表。"""
    out = _gh([
        "pr", "list",
        "--repo",   tileops_repo,
        "--state",  "merged",
        "--base",   "main",
        "--search", f"merged:>={since_date}",
        "--limit",  "100",
        "--json",   "number,title,mergedAt,author,url",
    ])
    if not out:
        return []
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return []


def get_pr_files(tileops_repo: str, pr_number: int) -> list[str]:
    """返回 PR 修改的文件路径列表。"""
    out = _gh([
        "pr", "view", str(pr_number),
        "--repo", tileops_repo,
        "--json", "files",
        "-q",     ".files[].path",
    ], timeout=30)
    if not out:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def _path_matches(changed: str, mapped: str) -> bool:
    """判断 PR 变更路径是否匹配注册表中的文件路径（去掉 ::fn 后缀后比较）。"""
    mapped_path = mapped.split("::")[0]
    # 完全匹配 or 其中一个是另一个的后缀
    return (changed == mapped_path
            or changed.endswith("/" + mapped_path)
            or mapped_path.endswith("/" + changed))


def match_pr_to_ops(registry: dict, changed_files: list[str]) -> set[str]:
    """找出 PR 涉及的算子 ID 集合。"""
    affected: set[str] = set()
    for op_id, op_data in registry.get("ops", {}).items():
        for file_list in op_data.get("files", {}).values():
            for mapped in file_list:
                if any(_path_matches(cf, mapped) for cf in changed_files):
                    affected.add(op_id)
                    break
            else:
                continue
            break
    return affected


def is_kernel_file(op_data: dict, filepath: str) -> bool:
    """判断 filepath 是否属于该算子的 kernel 文件。"""
    for mapped in op_data.get("files", {}).get("kernel", []):
        if _path_matches(filepath, mapped):
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# 3. 测试结果解析
# ─────────────────────────────────────────────────────────────────────────────

def parse_test_xml(test_xml: str | None) -> dict[str, dict]:
    """
    解析 JUnit XML，返回 {fn_name: {passed: bool, errors: [str]}}。
    fn_name 取 testcase.name 中 '[' 之前的部分。
    """
    results: dict[str, dict] = {}
    if not test_xml or not Path(test_xml).exists():
        return results
    try:
        root = ET.parse(test_xml).getroot()
    except ET.ParseError:
        return results

    for tc in root.iter("testcase"):
        fn = tc.get("name", "").split("[")[0]
        if not fn:
            continue
        fail_el = tc.find("failure")
        err_el  = tc.find("error")

        if fn not in results:
            results[fn] = {"passed": True, "errors": []}

        if fail_el is not None or err_el is not None:
            results[fn]["passed"] = False
            elem = fail_el if fail_el is not None else err_el
            msg  = (elem.get("message") or "")
            body = (elem.text or "")
            combined = (msg + "\n" + body).strip()[:600]
            results[fn]["errors"].append(combined)

    return results


def get_op_test_status(op_data: dict, test_results: dict) -> dict | None:
    """
    计算算子的测试状态。
    返回 {last_run, passed, errors} 或 None（无测试映射时）。
    """
    test_entries = op_data.get("files", {}).get("tests", [])
    if not test_entries:
        return None

    all_errors: list[str] = []
    all_passed = True
    any_found  = False

    for entry in test_entries:
        fn = entry.split("::")[-1] if "::" in entry else entry
        if fn in test_results:
            any_found = True
            r = test_results[fn]
            if not r["passed"]:
                all_passed = False
                all_errors.extend(r["errors"])

    if not any_found:
        return None

    return {
        "last_run": NOW_DATE,
        "passed":   all_passed,
        "errors":   all_errors[:5],   # 最多保留 5 条错误
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Benchmark 日志解析
# ─────────────────────────────────────────────────────────────────────────────

def parse_bench_log(bench_log: str | None) -> dict[str, dict]:
    """
    从 benchmark 日志中提取性能数据。
    支持两种常见格式：
      A) "bench_xxx | ... | TileOps: 1.23 ms | Baseline: 1.45 ms | Ratio: 0.85"
      B) Markdown 表格行（从 gen_index.py 生成的报告）

    返回 {bench_fn_stem: {summary, ratio, raw_line}}。
    """
    results: dict[str, dict] = {}
    if not bench_log or not Path(bench_log).exists():
        return results
    try:
        text = Path(bench_log).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return results

    # 格式 A
    pat_a = re.compile(
        r"(bench_\w+).*?TileOps\s*[:|]\s*([\d.]+\s*\w+).*?Baseline\s*[:|]\s*([\d.]+\s*\w+)"
        r"(?:.*?Ratio\s*[:|]\s*([\d.]+))?",
        re.IGNORECASE,
    )
    for m in pat_a.finditer(text):
        fn    = m.group(1).lower()
        tileops  = m.group(2).strip()
        baseline = m.group(3).strip()
        ratio    = float(m.group(4)) if m.group(4) else None
        results[fn] = {
            "tileops":  tileops,
            "baseline": baseline,
            "ratio":    ratio,
            "summary":  f"TileOps {tileops} vs Baseline {baseline}"
                        + (f" (ratio {ratio:.2f})" if ratio else ""),
        }

    return results


def get_op_bench_data(op_data: dict, bench_results: dict) -> dict | None:
    """提取算子对应的 benchmark 数据。"""
    bench_entries = op_data.get("files", {}).get("bench", [])
    found = {}
    for entry in bench_entries:
        # 取文件 stem，如 benchmarks/ops/bench_binary_arith.py -> bench_binary_arith
        stem = Path(entry.split("::")[0]).stem.lower()
        for key, data in bench_results.items():
            if key.startswith(stem) or stem.startswith(key.split("_")[0]):
                found[key] = data
    return found if found else None


# ─────────────────────────────────────────────────────────────────────────────
# 5. Claude 评估
# ─────────────────────────────────────────────────────────────────────────────

def build_score_prompt(ops_batch: list[dict]) -> str:
    return f"""你是 TileOPs 项目的代码审查专家。请根据以下信息为每个算子打分。

## 评分标准

### func_score（功能完整性，1-5）
- 1 = 骨架/占位符，基本不可用
- 2 = 基础功能，仅支持最简单的形状/类型
- 3 = 主要功能实现，但缺少边界情况处理
- 4 = 接近完整，仅有次要功能缺失
- 5 = 完整实现，覆盖所有预期场景

### perf_score（性能，相对 baseline，1-5）
- 1 = <50% baseline
- 2 = 50-75%
- 3 = 75-90%
- 4 = 90-100%
- 5 = >100%（超越 baseline）

### has_bugs
- 如果测试失败或 benchmark 异常，通常为 true

## 算子状态数据
```json
{json.dumps(ops_batch, ensure_ascii=False, indent=2)}
```

## 输出格式
严格输出 JSON，不含其他文字：
{{
  "updates": [
    {{
      "id": "ew.binary_arith.add",
      "func_score": 4,
      "func_score_notes": "前向传播完整，缺少 inplace 操作和复数类型",
      "perf_score": 4,
      "has_bugs": false,
      "bugs": []
    }}
  ]
}}

注意：
- 信息不足时某字段可设为 null（不强行猜测）
- 只输出**有实质变化**的算子（如当前分数已正确，可不输出）
- func_score_notes 限 1-2 句，说明原因或缺失点
- bugs 为 string 列表，简短描述已知问题"""


# ─────────────────────────────────────────────────────────────────────────────
# 6. 主逻辑
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="每日更新 TileOPs 算子注册表")
    parser.add_argument("--registry",     required=True, help="op_registry.json 路径")
    parser.add_argument("--op-data-dir",  required=True, help="op_data/ 目录路径")
    parser.add_argument("--test-xml",     default=None,  help="test_results.xml 路径")
    parser.add_argument("--bench-log",    default=None,  help="tileops_benchmarks.log 路径")
    parser.add_argument("--tileops-repo", default="tile-ai/TileOPs",
                        help="TileOPs GitHub 仓库（owner/name）")
    parser.add_argument("--since-date",   default=None,
                        help="扫描 PR 的起始日期 YYYY-MM-DD（默认昨天）")
    parser.add_argument("--model",        default="claude-sonnet-4-6")
    parser.add_argument("--skip-claude",  action="store_true",
                        help="跳过 Claude 评估步骤")
    args = parser.parse_args()

    api_key, base_url = get_api_config()

    since_date = args.since_date or str(TODAY - timedelta(days=1))

    # ── 加载注册表 ───────────────────────────────────────────────────────────
    print(f"加载注册表: {args.registry}")
    registry = load_registry(args.registry)
    n_ops = len(registry.get("ops", {}))
    print(f"  {n_ops} 个算子")
    if n_ops == 0:
        print("  ⚠ 注册表为空，请先运行 bootstrap_registry.py", file=sys.stderr)

    # ── PR 扫描 ──────────────────────────────────────────────────────────────
    print(f"\n扫描 {args.tileops_repo} 中 {since_date} 之后的 merged PR ...")
    prs = get_merged_prs(args.tileops_repo, since_date)
    print(f"  找到 {len(prs)} 个 PR")

    # pr_kernel_updates: op_id -> 最新 PR 信息（只记录影响 kernel 文件的 PR）
    pr_kernel_updates: dict[str, dict] = {}
    # pr_any_updates: op_id -> 最新 PR 信息（影响任意文件）
    pr_any_updates: dict[str, dict] = {}

    for pr in prs:
        pr_num    = pr["number"]
        pr_author = pr.get("author", {}).get("login", "unknown")
        pr_merged = pr.get("mergedAt", NOW_ISO)
        pr_url    = pr.get("url", "")
        pr_title  = pr.get("title", "")

        changed_files = get_pr_files(args.tileops_repo, pr_num)
        if not changed_files:
            continue

        affected_ops = match_pr_to_ops(registry, changed_files)
        if affected_ops:
            print(f"  PR #{pr_num} ({pr_title[:50]}): 影响 {len(affected_ops)} 个算子")

        for op_id in affected_ops:
            op_data = registry["ops"].get(op_id, {})
            pr_info = {
                "number":   pr_num,
                "author":   pr_author,
                "mergedAt": pr_merged,
                "url":      pr_url,
                "title":    pr_title,
            }
            # 更新最新 PR（按 mergedAt 时间）
            existing = pr_any_updates.get(op_id, {})
            if not existing or pr_merged > existing.get("mergedAt", ""):
                pr_any_updates[op_id] = pr_info

            # 只有 kernel 文件变化时才更新 kernel_last_*
            if any(is_kernel_file(op_data, cf) for cf in changed_files):
                existing_k = pr_kernel_updates.get(op_id, {})
                if not existing_k or pr_merged > existing_k.get("mergedAt", ""):
                    pr_kernel_updates[op_id] = pr_info

    print(f"  {len(pr_any_updates)} 个算子受 PR 影响, "
          f"其中 {len(pr_kernel_updates)} 个有 kernel 变更")

    # ── 测试结果 ─────────────────────────────────────────────────────────────
    print(f"\n解析测试结果: {args.test_xml or '(未提供)'}")
    test_results = parse_test_xml(args.test_xml)
    print(f"  {len(test_results)} 个测试函数")

    # ── Benchmark 日志 ───────────────────────────────────────────────────────
    print(f"解析 benchmark 日志: {args.bench_log or '(未提供)'}")
    bench_results = parse_bench_log(args.bench_log)
    print(f"  {len(bench_results)} 个 benchmark 有数据")

    # ── 更新每个算子 ─────────────────────────────────────────────────────────
    print("\n更新算子记录 ...")
    ops_to_score: list[dict] = []   # 需要 Claude 重新评分的算子

    for op_id, op_data in registry.get("ops", {}).items():
        registry_updates: dict = {}   # 更新到 op_registry.json
        yaml_updates: dict     = {}   # 更新到 YAML 文件
        needs_score = False

        # PR kernel 元数据
        if op_id in pr_kernel_updates:
            pk = pr_kernel_updates[op_id]
            meta = {
                "kernel_last_updated":   pk["mergedAt"],
                "kernel_last_pr_number": pk["number"],
                "kernel_last_pr_url":    pk["url"],
                "kernel_last_pr_author": pk["author"],
                "kernel_last_pr_title":  pk["title"],
            }
            registry_updates["kernel_last_updated"]   = pk["mergedAt"]
            registry_updates["kernel_last_pr_author"] = pk["author"]
            yaml_updates.update(meta)
            needs_score = True

        # 测试状态
        test_status = get_op_test_status(op_data, test_results)
        if test_status is not None:
            yaml_updates["test_status"] = test_status
            # 有测试失败时初步标记 has_bugs
            if not test_status["passed"] and op_data.get("has_bugs") is None:
                registry_updates["has_bugs"] = True
                yaml_updates["has_bugs"]     = True
            needs_score = True

        # Benchmark 数据（写入 YAML perf_details.results，供 Claude 参考）
        bench_data = get_op_bench_data(op_data, bench_results)
        if bench_data:
            results_list = [
                {"bench_fn": fn, **v}
                for fn, v in bench_data.items()
            ]
            yaml_updates.setdefault("perf_details", {})["results"] = results_list
            yaml_updates.setdefault("perf_details", {})["last_run"] = NOW_DATE
            needs_score = True

        # 收集需要评分的算子信息
        if needs_score or op_id in pr_any_updates:
            ops_to_score.append({
                "id":                  op_id,
                "name":                op_data.get("name", ""),
                "category":            op_data.get("category", ""),
                "current_func_score":  op_data.get("func_score"),
                "current_perf_score":  op_data.get("perf_score"),
                "current_has_bugs":    op_data.get("has_bugs"),
                "test_status":         test_status,
                "pr_title":            pr_any_updates.get(op_id, {}).get("title"),
                "bench_data":          bench_data,
            })

        # 写回注册表
        if registry_updates:
            op_data.update(registry_updates)
            op_data["last_updated"] = NOW_ISO

        # 写回 YAML
        if yaml_updates:
            yaml_data = load_op_yaml(args.op_data_dir, op_id)
            if yaml_data is not None:
                deep_update(yaml_data, yaml_updates)
                yaml_data["last_updated"] = NOW_ISO
                yaml_data["updated_by"]   = "nightly-auto"
                save_op_yaml(args.op_data_dir, op_id, yaml_data)

    print(f"  {len(ops_to_score)} 个算子待 Claude 评分")

    # ── Claude 评分 ──────────────────────────────────────────────────────────
    if ops_to_score and not args.skip_claude and api_key:
        if require_anthropic() is None:
            args.skip_claude = True

    if ops_to_score and not args.skip_claude and api_key:
        batch_size = 20
        total_updated = 0
        for i in range(0, len(ops_to_score), batch_size):
            batch = ops_to_score[i : i + batch_size]
            print(f"  Claude 评分批次 {i // batch_size + 1}/{(len(ops_to_score) - 1) // batch_size + 1}"
                  f" ({len(batch)} 个算子) ...")
            try:
                prompt = build_score_prompt(batch)
                data   = call_claude_json(
                    prompt, args.model, api_key, base_url,
                    system=SYSTEM_SCORE_EVALUATOR,
                    required_keys=["updates"],
                )

                for item in data.get("updates", []):
                    op_id = item.get("id")
                    if not op_id or op_id not in registry.get("ops", {}):
                        continue

                    # 更新 registry JSON
                    op = registry["ops"][op_id]
                    for field in ("func_score", "perf_score", "has_bugs"):
                        if field in item and item[field] is not None:
                            op[field] = item[field]
                    op["last_updated"] = NOW_ISO

                    # 更新 YAML
                    yaml_data = load_op_yaml(args.op_data_dir, op_id)
                    if yaml_data is not None:
                        for field in ("func_score", "func_score_notes",
                                      "perf_score", "has_bugs", "bugs"):
                            if field in item:
                                yaml_data[field] = item[field]
                        if "func_score" in item:
                            yaml_data["func_score_updated"] = NOW_DATE
                        if "perf_score" in item:
                            yaml_data["perf_score_updated"] = NOW_DATE
                        yaml_data["last_updated"] = NOW_ISO
                        yaml_data["updated_by"]   = "claude-nightly"
                        save_op_yaml(args.op_data_dir, op_id, yaml_data)

                    total_updated += 1

                print(f"    更新了 {len(data.get('updates', []))} 个算子")

            except Exception as exc:
                print(f"  ⚠ Claude 评分失败: {exc}", file=sys.stderr)

        print(f"  Claude 共更新 {total_updated} 个算子")
    elif ops_to_score and not api_key:
        print("  ⚠ 未设置 ANTHROPIC_API_KEY，跳过 Claude 评分", file=sys.stderr)

    # ── 保存注册表 ───────────────────────────────────────────────────────────
    save_registry(registry, args.registry)
    print(f"\n注册表已保存: {args.registry}")
    print("完成！")


if __name__ == "__main__":
    main()
