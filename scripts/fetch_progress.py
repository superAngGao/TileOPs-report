#!/usr/bin/env python3
"""fetch_progress.py — 扫描代码库 + 解析测试/benchmark 结果，输出 progress.json。

用法：
    python fetch_progress.py \\
        --repo-dir  <TileOPs 根目录> \\
        --manifest  <op_manifest.json 路径> \\
        --test-xml  <test_results.xml 路径> \\
        --bench-xml <bench_results.xml 路径（可选）> \\
        --output    <progress.json 输出路径>
"""

import argparse
import json
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path


# ── 1. 扫描代码库中已定义的 class 名 ────────────────────────────────────────

def scan_classes(repo_dir: str) -> set[str]:
    """递归扫描 tileops/ 下所有 .py 文件，返回所有 class 名的集合。"""
    classes: set[str] = set()
    tileops_dir = Path(repo_dir) / "tileops"
    if not tileops_dir.exists():
        return classes
    for py_file in tileops_dir.rglob("*.py"):
        try:
            text = py_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for m in re.finditer(r"^class\s+(\w+)", text, re.MULTILINE):
            classes.add(m.group(1))
    return classes


# ── 2. 解析 JUnit XML，收集通过/失败的测试函数名 ────────────────────────────

def _parse_junit(xml_path: str | None) -> tuple[set[str], set[str]]:
    """
    返回 (passed_fns, failed_fns)。
    fn 格式为 test_xxx（不含参数括号）。
    """
    passed: set[str] = set()
    failed: set[str] = set()
    if not xml_path or not os.path.exists(xml_path):
        return passed, failed

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError:
        return passed, failed

    for tc in root.iter("testcase"):
        raw_name: str = tc.get("name", "")
        fn = raw_name.split("[")[0]          # 去掉参数部分
        has_failure = tc.find("failure") is not None or tc.find("error") is not None
        if has_failure:
            failed.add(fn)
        else:
            passed.add(fn)
    return passed, failed


def parse_passed_tests(xml_path: str | None) -> tuple[set[str], set[str]]:
    return _parse_junit(xml_path)


def parse_passed_benches(bench_xml_path: str | None) -> tuple[set[str], set[str]]:
    return _parse_junit(bench_xml_path)


# ── 3. 逐 op 判断状态 ────────────────────────────────────────────────────────

def check_op(
    op: dict,
    all_classes: set[str],
    passed_tests: set[str],
    failed_tests: set[str],
    passed_benches: set[str],
    failed_benches: set[str],
) -> dict:
    """
    返回单个 op 的状态字典：
      implemented  : bool  — op_classes 中至少一个类存在于代码库
      tested       : bool  — test_fns 中至少一个函数通过了测试
      test_failed  : bool  — test_fns 中至少一个函数失败
      bench_ok     : bool | None — benchmark 运行通过（None=未映射/无 bench_xml）
      status       : "done" | "impl_only" | "partial" | "missing"
    """
    op_classes: list[str] = op.get("op_classes", [])
    test_fns: list[str] = op.get("test_fns", [])
    bench_file: str | None = op.get("bench_file")

    implemented = bool(op_classes) and any(c in all_classes for c in op_classes)
    tested = bool(test_fns) and any(fn in passed_tests for fn in test_fns)
    test_failed = bool(test_fns) and any(fn in failed_tests for fn in test_fns)

    # benchmark: 只有传入了 bench_xml 才做判断
    if bench_file and (passed_benches or failed_benches):
        # 根据 bench_file 的文件名前缀匹配 bench 函数名
        bench_stem = Path(bench_file).stem  # e.g. "bench_binary_arith"
        bench_ok: bool | None = any(
            fn.startswith("bench_") and bench_stem.lstrip("bench_") in fn
            for fn in passed_benches
        ) or any(fn.startswith(bench_stem) for fn in passed_benches)
        # 更简单的策略：只要该 bench_file 对应的任意 bench 函数通过，即认为 bench_ok
        # 具体做法：bench_xml 中 classname 包含 bench_file stem
    else:
        bench_ok = None

    if implemented and tested and bench_ok is True:
        status = "done"        # 三项全部通过
    elif implemented and tested:
        status = "tested"      # 实现+测试通过，但 benchmark 未通过或无数据
    elif implemented and not tested and not test_fns:
        status = "impl_only"   # 有实现，但 manifest 未映射测试函数
    elif implemented:
        status = "partial"     # 有实现，但测试缺失或失败
    else:
        status = "missing"     # 完全未实现

    return {
        "implemented": implemented,
        "tested": tested,
        "test_failed": test_failed,
        "bench_ok": bench_ok,
        "status": status,
    }


# ── 4. 主逻辑 ────────────────────────────────────────────────────────────────

def compute_progress(
    manifest: dict,
    all_classes: set[str],
    passed_tests: set[str],
    failed_tests: set[str],
    passed_benches: set[str],
    failed_benches: set[str],
) -> dict:
    now_iso = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    total_ops = manifest["total_ops"]

    categories_out = []
    grand_done = 0
    grand_impl = 0
    grand_tested = 0

    grand_benched = 0

    for cat in manifest["categories"]:
        cat_done = 0
        cat_impl = 0
        cat_tested = 0
        cat_benched = 0
        ops_out = []

        for op in cat["ops"]:
            st = check_op(
                op, all_classes,
                passed_tests, failed_tests,
                passed_benches, failed_benches,
            )
            if st["implemented"]:
                cat_impl += 1
            if st["tested"]:
                cat_tested += 1
            if st["bench_ok"] is True:
                cat_benched += 1
            if st["status"] == "done":
                cat_done += 1

            ops_out.append({
                "id": op["id"],
                "name": op["name"],
                "sub": op.get("sub", ""),
                **st,
            })

        cat_total = len(cat["ops"])
        grand_done += cat_done
        grand_impl += cat_impl
        grand_tested += cat_tested
        grand_benched += cat_benched

        categories_out.append({
            "id": cat["id"],
            "name": cat["name"],
            "issue": cat["issue"],
            "difficulty": cat["difficulty"],
            "total_ops": cat_total,
            "done_ops": cat_done,
            "impl_ops": cat_impl,
            "tested_ops": cat_tested,
            "benched_ops": cat_benched,
            "ops": ops_out,
        })

    return {
        "generated_at": now_iso,
        "total_ops": total_ops,
        "done_ops": grand_done,
        "impl_ops": grand_impl,
        "tested_ops": grand_tested,
        "benched_ops": grand_benched,
        "categories": categories_out,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="TileOPs 开发进度采集工具")
    parser.add_argument("--repo-dir",   required=True, help="TileOPs 仓库根目录")
    parser.add_argument("--manifest",   required=True, help="op_manifest.json 路径")
    parser.add_argument("--test-xml",   default=None,  help="test_results.xml 路径")
    parser.add_argument("--bench-xml",  default=None,  help="bench_results.xml 路径（可选）")
    parser.add_argument("--output",     required=True, help="progress.json 输出路径")
    args = parser.parse_args()

    # 加载 manifest
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"错误：manifest 文件不存在: {args.manifest}", flush=True)
        raise SystemExit(1)
    manifest = json.loads(manifest_path.read_text())

    # 扫描代码库
    print(f"  扫描代码库: {args.repo_dir}", flush=True)
    all_classes = scan_classes(args.repo_dir)
    print(f"  找到 {len(all_classes)} 个 class 定义", flush=True)

    # 解析测试结果
    passed_tests, failed_tests = parse_passed_tests(args.test_xml)
    print(f"  测试结果: {len(passed_tests)} passed, {len(failed_tests)} failed", flush=True)

    # 解析 benchmark 结果
    passed_benches, failed_benches = parse_passed_benches(args.bench_xml)
    if args.bench_xml:
        print(f"  Benchmark 结果: {len(passed_benches)} passed, {len(failed_benches)} failed", flush=True)
    else:
        print("  Benchmark XML 未提供，跳过 benchmark 状态检查", flush=True)

    # 计算进度
    progress = compute_progress(
        manifest, all_classes,
        passed_tests, failed_tests,
        passed_benches, failed_benches,
    )

    # 输出
    out_path = Path(args.output)
    out_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2))

    done = progress["done_ops"]
    total = progress["total_ops"]
    pct = done / total * 100 if total else 0
    print(f"  进度: {done}/{total} ops 完成 ({pct:.1f}%)", flush=True)
    print(f"  进度数据已保存: {args.output}", flush=True)


if __name__ == "__main__":
    main()
