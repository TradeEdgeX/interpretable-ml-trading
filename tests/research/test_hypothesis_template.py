from __future__ import annotations

from pathlib import Path

from src.research.hypothesis_template import validate_decision_path, validate_template_text

REPO = Path(__file__).resolve().parents[2]


def test_showcase_template_passes() -> None:
    path = REPO / "config/experiments/20260910_ma50_ma200_cross/DECISION.md"
    report = validate_decision_path(path)
    assert report.ok, (report.missing, report.issues)
    assert report.slots["sociology"]
    assert report.slots["math"]
    assert report.slots["stats"]


def test_empty_scaffold_fails() -> None:
    path = REPO / "config/experiments/_template/DECISION.md"
    text = path.read_text(encoding="utf-8").replace("{{TOPIC}}", "demo")
    report = validate_template_text(text)
    assert report.ok is False
    assert "sociology" in report.missing or "claim" in report.missing


def test_five_boxes_alone_are_not_enough() -> None:
    text = """
## 原句
价格上穿均线就做多，跌破就走。预期趋势年赚钱。

## 五格
| 格 | 内容 |
| 机制 | 上穿做多 |
| 预期市况 | 趋势年 |
| 合同 | 跌破离场 |
| 证伪条件 | 年化为负 |
| 落地 | 同一句 |
"""
    report = validate_template_text(text)
    assert report.ok is False
    assert "sociology" in report.missing
    assert "math" in report.missing
