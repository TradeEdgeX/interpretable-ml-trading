"""
检查全局归一化使用的 pytest 测试

注意：这个测试检查代码中是否有使用全局归一化（window=None）的情况。
如果你的代码都使用滚动归一化（指定了 window 参数），这个测试可能不会发现任何问题。

全局归一化在时序数据中会导致未来信息泄露，应该避免使用。
推荐做法：
- 使用滚动归一化：normalize_by_group(..., window=252)  # 日频数据
- 每个标的每个特征自己归一化：normalize_by_group(..., group_col="_symbol", window=252)

如果你确认所有归一化都使用了滚动窗口，这个测试可以跳过或者标记为可选。
"""

import pytest
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def src_python_files():
    """获取所有 src 目录下的 Python 文件"""
    src_dir = PROJECT_ROOT / "src"
    python_files = []
    for path in src_dir.rglob("*.py"):
        # 排除 __pycache__ 和虚拟环境
        if "__pycache__" in str(path) or ".venv" in str(path) or "venv" in str(path):
            continue
        python_files.append(path)
    return python_files


class TestGlobalNormalization:
    """全局归一化检查测试类"""

    def test_no_global_normalization(self, src_python_files):
        """
        测试没有使用全局归一化

        注意：如果你的代码都使用滚动归一化（指定了 window 参数），
        这个测试可能不会发现任何问题。可以跳过或标记为可选。
        """
        issues = []

        for file_path in src_python_files:
            file_issues = self._check_global_normalization(file_path)
            if file_issues:
                issues.extend([(file_path, issue) for issue in file_issues])

        if issues:
            # 打印问题
            print("\n⚠️  发现以下文件可能使用了全局归一化：")
            for file_path, (line_num, issue) in issues:
                rel_path = file_path.relative_to(PROJECT_ROOT)
                print(f"   📄 {rel_path}:{line_num}: {issue}")

            # 对于明确的 window=None，应该失败测试
            critical_issues = [
                (fp, (ln, iss))
                for fp, (ln, iss) in issues
                if "window=None" in iss or "明确使用全局归一化" in iss
            ]

            if critical_issues:
                error_msg = f"发现 {len(critical_issues)} 个明确的全局归一化使用:\n"
                for file_path, (line_num, issue) in critical_issues[:5]:
                    rel_path = file_path.relative_to(PROJECT_ROOT)
                    error_msg += f"  {rel_path}:{line_num}: {issue}\n"
                pytest.fail(error_msg)
            else:
                # 只是警告，不失败测试
                print(f"\n⚠️  发现 {len(issues)} 个潜在问题（需要人工检查）")
        else:
            print("✅ 未发现使用全局归一化的情况（所有归一化都使用了滚动窗口）")

    def test_no_explicit_window_none_in_src(self):
        """
        快速检查：确保 src 代码中没有显式的 window=None 调用。

        排除 utils_normalization.py（该文件包含警示文案）。
        """
        src_dir = PROJECT_ROOT / "src"
        allowlist = set()
        offenders = []

        for path in src_dir.rglob("*.py"):
            if "utils_normalization.py" in str(path):
                continue
            if "__pycache__" in str(path):
                continue
            text = path.read_text(encoding="utf-8")
            if "window=None" in text:
                rel = path.relative_to(PROJECT_ROOT)
                if rel in allowlist:
                    continue
                offenders.append(rel)

        assert not offenders, (
            "检测到显式 window=None，请确认是否需要改为滚动归一化：" f" {offenders}"
        )

    def _check_global_normalization(self, file_path: Path) -> List[Tuple[int, str]]:
        """检查文件中是否有使用全局归一化的情况"""
        issues = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                lines = content.split("\n")

            # 检查是否有调用 normalize_by_group 或 normalize_dataframe
            for i, line in enumerate(lines, 1):
                # 检查是否有调用这些函数
                if "normalize_by_group" in line or "normalize_dataframe" in line:
                    # 检查是否明确指定了 window=None 或没有指定 window 参数
                    # 如果指定了 window=数字，则没问题

                    # 情况1: window=None (明确指定)
                    if "window=None" in line:
                        issues.append((i, f"明确使用全局归一化: {line.strip()}"))

                    # 情况2: 调用函数但没有指定 window 参数（默认就是 None）
                    # 排除函数定义（def 开头的行）
                    if line.strip().startswith("def "):
                        continue

                    # 检查是否是函数调用
                    if "normalize_by_group(" in line or "normalize_dataframe(" in line:
                        # 检查这一行和后续几行是否有 window= 参数
                        window_found = False
                        check_lines = lines[
                            i - 1 : min(i + 5, len(lines))
                        ]  # 检查当前行和后续5行
                        for check_line in check_lines:
                            if "window=" in check_line:
                                window_found = True
                                # 如果 window 是数字，则没问题
                                if "window=" in check_line and any(
                                    c.isdigit()
                                    for c in check_line.split("window=")[1]
                                    .split(",")[0]
                                    .split(")")[0]
                                ):
                                    break
                                # 如果 window=None，已经在上面检查了
                                break

                        # 如果没有找到 window 参数，可能是使用默认值（None）
                        if not window_found:
                            # 检查函数调用是否完整（在同一行）
                            if ")" in line:
                                issues.append(
                                    (
                                        i,
                                        f"可能使用默认全局归一化（未指定window参数）: {line.strip()}",
                                    )
                                )
                            else:
                                # 多行调用，标记为需要检查
                                issues.append(
                                    (
                                        i,
                                        f"多行调用，需要检查是否指定window参数: {line.strip()}",
                                    )
                                )

        except Exception as e:
            issues.append((0, f"解析文件时出错: {e}"))

        return issues
