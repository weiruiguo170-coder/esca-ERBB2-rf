#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import List


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")


def run_cmd(cmd: List[str]) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def python_cmd(py_exe: str, script_name: str, *args: str) -> List[str]:
    return [py_exe, os.path.join(SRC, script_name), *args]


def run_submission(py_exe: str) -> None:
    for step in ["table1", "table2", "figure1", "figure3", "figure4", "readme"]:
        run_cmd(python_cmd(py_exe, "submission_builder.py", "--step", step))


def run_paper_replace(py_exe: str) -> None:
    for step in ["table1", "table2", "figure1", "figure3", "figure4", "summary"]:
        run_cmd(python_cmd(py_exe, "paper_replace_builder.py", "--step", step))


def run_teaching_figure4(py_exe: str) -> None:
    for step in ["matched", "matrices", "preview", "qc"]:
        run_cmd(python_cmd(py_exe, "generate_teaching_figure4_dataset.py", "--step", step))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="论文复现项目统一入口（包装已有脚本，不做额外重训练）。"
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "full",
            "quick",
            "quick_lite",
            "submission",
            "paper_replace",
            "teaching_fig1",
            "teaching_fig3",
            "teaching_fig4",
            "figure4_sensitivity",
            "deliverables_only",
        ],
        help="执行模式。",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python 可执行文件路径，默认使用当前解释器。",
    )
    args = parser.parse_args()

    py_exe = args.python

    if args.mode == "full":
        run_cmd(python_cmd(py_exe, "reproduce_core.py"))
        return
    if args.mode == "quick":
        run_cmd(python_cmd(py_exe, "reproduce_quick.py"))
        return
    if args.mode == "quick_lite":
        run_cmd(python_cmd(py_exe, "reproduce_quick_lite.py"))
        return
    if args.mode == "submission":
        run_submission(py_exe)
        return
    if args.mode == "paper_replace":
        run_paper_replace(py_exe)
        return
    if args.mode == "teaching_fig1":
        run_cmd(python_cmd(py_exe, "generate_teaching_figure1_dataset.py"))
        return
    if args.mode == "teaching_fig3":
        run_cmd(python_cmd(py_exe, "generate_teaching_figure3_dataset.py"))
        return
    if args.mode == "teaching_fig4":
        run_teaching_figure4(py_exe)
        return
    if args.mode == "figure4_sensitivity":
        run_cmd(python_cmd(py_exe, "build_figure4_sensitivity.py"))
        return
    if args.mode == "deliverables_only":
        # 推荐给 GitHub 用户：不触发长训练，只构建展示与投稿交付物。
        run_submission(py_exe)
        run_cmd(python_cmd(py_exe, "build_figure4_sensitivity.py"))
        return

    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
