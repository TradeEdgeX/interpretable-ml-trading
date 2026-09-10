#!/usr/bin/env python3
"""
检查是否存在未来函数（look-ahead bias）问题

主要检查点：
1. 特征计算是否在划分训练/测试集之前使用了fit=True
2. 波动率模型训练是否使用了未来信息
3. 回测评估时是否使用了测试集数据来训练模型
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np


def check_feature_fitting_order(script_path: str) -> dict:
    """检查特征计算和数据集划分的顺序"""
    issues = []

    try:
        with open(script_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 检查是否在划分训练/测试集之前使用了fit=True
        lines = content.split("\n")
        fit_before_split = False
        split_after_fit = False

        for i, line in enumerate(lines):
            if "fit=True" in line and "run_feature_pipeline" in line:
                fit_line = i
                fit_before_split = True
            if "split_idx" in line and "int(len(df_features)" in line:
                split_line = i
                if fit_before_split and fit_line < split_line:
                    issues.append(
                        {
                            "type": "feature_fit_before_split",
                            "severity": "HIGH",
                            "description": f"特征计算在划分训练/测试集之前使用了fit=True（第{fit_line+1}行），这会导致数据泄漏",
                            "line": fit_line + 1,
                        }
                    )
                    split_after_fit = True

        return {
            "has_issues": len(issues) > 0,
            "issues": issues,
        }
    except Exception as e:
        return {
            "has_issues": True,
            "issues": [
                {
                    "type": "error",
                    "severity": "HIGH",
                    "description": f"无法读取脚本: {e}",
                }
            ],
        }


def check_volatility_model_training(script_path: str) -> dict:
    """检查波动率模型训练是否使用了未来信息"""
    issues = []

    try:
        with open(script_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 检查是否在训练波动率模型时使用了未来波动率标签
        if "future_volatility_label" in content:
            # 检查是否在训练集上计算未来波动率
            if 'df_features["future_volatility"]' in content:
                # 检查是否在完整的df_features上计算（包括测试集）
                lines = content.split("\n")
                for i, line in enumerate(lines):
                    if "future_volatility_label" in line and "df_features" in line:
                        issues.append(
                            {
                                "type": "future_vol_in_training",
                                "severity": "INFO",
                                "description": f"在完整的df_features上计算未来波动率标签（第{i+1}行），这是正确的，因为标签可以使用未来信息",
                                "line": i + 1,
                            }
                        )

        return {
            "has_issues": len(issues) > 0,
            "issues": issues,
        }
    except Exception as e:
        return {
            "has_issues": True,
            "issues": [
                {
                    "type": "error",
                    "severity": "HIGH",
                    "description": f"无法读取脚本: {e}",
                }
            ],
        }


def check_test_set_usage(script_path: str) -> dict:
    """检查测试集是否被用于训练"""
    issues = []

    try:
        with open(script_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 检查是否在评估时使用了测试集来训练模型
        lines = content.split("\n")
        in_evaluate_function = False
        train_calls_in_evaluate = []

        for i, line in enumerate(lines):
            if "def evaluate" in line:
                in_evaluate_function = True
            if in_evaluate_function and "def " in line and "def evaluate" not in line:
                in_evaluate_function = False
            if in_evaluate_function and (
                "train" in line.lower() or "fit" in line.lower()
            ):
                if "X_test" in line or "df_test" in line:
                    train_calls_in_evaluate.append(
                        {
                            "line": i + 1,
                            "content": line.strip(),
                        }
                    )

        if train_calls_in_evaluate:
            for call in train_calls_in_evaluate:
                issues.append(
                    {
                        "type": "train_on_test",
                        "severity": "CRITICAL",
                        "description": f'在评估函数中使用了测试集来训练模型（第{call["line"]}行）',
                        "line": call["line"],
                        "content": call["content"],
                    }
                )

        return {
            "has_issues": len(issues) > 0,
            "issues": issues,
        }
    except Exception as e:
        return {
            "has_issues": True,
            "issues": [
                {
                    "type": "error",
                    "severity": "HIGH",
                    "description": f"无法读取脚本: {e}",
                }
            ],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check for look-ahead bias issues")
    parser.add_argument(
        "--script",
        type=str,
        default="src/diagnostics/sr_reversal_model_comparison.py",
        help="Path to script to check",
    )
    args = parser.parse_args()

    script_path = Path(args.script)
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return

    print("🔍 Checking for look-ahead bias issues...")
    print("=" * 60)

    # Check 1: Feature fitting order
    print("\n1️⃣ Checking feature fitting order...")
    fit_check = check_feature_fitting_order(str(script_path))
    if fit_check["has_issues"]:
        print("   ⚠️  ISSUES FOUND:")
        for issue in fit_check["issues"]:
            print(f"      [{issue['severity']}] {issue['description']}")
    else:
        print("   ✅ No issues found")

    # Check 2: Volatility model training
    print("\n2️⃣ Checking volatility model training...")
    vol_check = check_volatility_model_training(str(script_path))
    if vol_check["has_issues"]:
        print("   ℹ️  INFO:")
        for issue in vol_check["issues"]:
            print(f"      [{issue['severity']}] {issue['description']}")
    else:
        print("   ✅ No issues found")

    # Check 3: Test set usage
    print("\n3️⃣ Checking test set usage in evaluation...")
    test_check = check_test_set_usage(str(script_path))
    if test_check["has_issues"]:
        print("   ⚠️  ISSUES FOUND:")
        for issue in test_check["issues"]:
            print(f"      [{issue['severity']}] {issue['description']}")
            if "content" in issue:
                print(f"         Code: {issue['content']}")
    else:
        print("   ✅ No issues found")

    print("\n" + "=" * 60)
    print("📊 Summary:")

    all_issues = fit_check["issues"] + vol_check["issues"] + test_check["issues"]
    critical_issues = [i for i in all_issues if i.get("severity") == "CRITICAL"]
    high_issues = [i for i in all_issues if i.get("severity") == "HIGH"]

    if critical_issues:
        print(f"   ❌ CRITICAL issues: {len(critical_issues)}")
    if high_issues:
        print(f"   ⚠️  HIGH severity issues: {len(high_issues)}")
    if not critical_issues and not high_issues:
        print("   ✅ No critical or high severity issues found")

    if critical_issues or high_issues:
        print("\n💡 Recommendations:")
        print("   1. 特征计算应该在划分训练/测试集之后进行")
        print("   2. 在训练集上使用fit=True，在测试集上使用fit=False")
        print("   3. 确保评估函数中不使用测试集来训练模型")


if __name__ == "__main__":
    main()
