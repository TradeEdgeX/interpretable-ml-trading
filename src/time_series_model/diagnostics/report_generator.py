"""
统一报告生成器

从execution_log.jsonl生成统一格式报告，支持backtest和live两种模式。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.time_series_model.diagnostics.execution_log_aggregate import (
    aggregate_stage_logs,
)


def generate_unified_report(
    records: List[Dict[str, Any]],
    source: str = "pipeline",
    output_formats: List[str] = ["json", "markdown"],
) -> Dict[str, Any]:
    """
    生成统一格式报告

    Args:
        records: Execution log记录列表
        source: 数据源（"pipeline"或"live_trading"）
        output_formats: 输出格式列表

    Returns:
        统一格式的报告字典
    """
    positions = []
    operations = []

    # 从records中提取positions和operations
    # 这里简化处理，实际应该跟踪position状态
    for rec in records:
        execution = rec.get("execution") or {}
        gate = rec.get("gate") or {}
        returns_data = rec.get("returns") or {}
        features = rec.get("features") or {}

        if not execution.get("intent", False):
            continue

        symbol = rec.get("symbol")
        timestamp = rec.get("timestamp")
        position_id = f"{symbol}_{timestamp}"

        # 提取archetype和regime
        archetype = gate.get("archetype") or execution.get("archetype")
        regime = rec.get("router", {}).get("mode") or "UNKNOWN"

        # 提取returns
        ret_mean = returns_data.get("ret_mean")
        ret_trend = returns_data.get("ret_trend")
        ret = ret_mean if ret_mean is not None else ret_trend

        # 确定side
        side = "LONG" if ret and ret > 0 else "SHORT"

        # 创建position
        position = {
            "position_id": position_id,
            "symbol": symbol,
            "side": side,
            "entry_time": timestamp,
            "exit_time": None,
            "entry_price": None,  # 需要从OHLC数据中获取
            "exit_price": None,
            "initial_size": None,
            "current_size": None,
            "total_cost": None,
            "total_value": None,
            "unrealized_pnl": None,
            "realized_pnl": ret if ret else 0.0,
            "status": "OPEN",
            "stop_loss_price": None,
            "take_profit_price": None,
            "trailing_stop_config": None,
            "exit_reason": None,
            "strategy_id": None,
            "notes": None,
            # 扩展字段
            "archetype": archetype,
            "regime": regime,
            "gate_decision": "allow" if not gate.get("blocked") else "veto",
            "gate_reasons": gate.get("reasons", {}),
            "reflexivity_features": {
                "ofci_pct": features.get("ofci_pct"),
                "shd_pct": features.get("shd_pct"),
                "lfi_pct": features.get("lfi_pct"),
            },
            "execution_intent": execution.get("intent", False),
            "execution_submit_order": execution.get("submit_order", False),
        }

        positions.append(position)

        # 创建operation
        operation = {
            "operation_id": f"{position_id}_entry",
            "position_id": position_id,
            "operation_type": "entry",
            "operation_time": timestamp,
            "size": None,
            "price": None,
            "pnl": None,
            "cumulative_pnl": None,
            "stop_loss_price": None,
            "take_profit_price": None,
            "reason": f"{archetype}_{regime}",
            "order_id": None,
            "notes": None,
        }

        operations.append(operation)

    return {
        "source": source,
        "generated_at": datetime.now().isoformat(),
        "positions": positions,
        "operations": operations,
    }


def save_report(
    report: Dict[str, Any],
    output_path: Path,
    formats: List[str] = ["json", "markdown"],
) -> None:
    """保存报告到文件"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if "json" in formats:
        json_path = output_path.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"✅ JSON报告已保存: {json_path}")

    if "markdown" in formats:
        md_path = output_path.with_suffix(".md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# 统一交易报告\n\n")
            f.write(f"**数据源**: {report['source']}\n")
            f.write(f"**生成时间**: {report['generated_at']}\n\n")

            f.write(f"## 仓位统计\n\n")
            f.write(f"- 总仓位数: {len(report['positions'])}\n")
            f.write(f"- 总操作数: {len(report['operations'])}\n\n")

            f.write("## 仓位详情\n\n")
            for pos in report["positions"][:10]:  # 只显示前10个
                f.write(f"### {pos['position_id']}\n")
                f.write(f"- Symbol: {pos['symbol']}\n")
                f.write(f"- Side: {pos['side']}\n")
                f.write(f"- Entry Time: {pos['entry_time']}\n")
                f.write(f"- Archetype: {pos.get('archetype', 'N/A')}\n")
                f.write(f"- Regime: {pos.get('regime', 'N/A')}\n")
                f.write(f"- Realized PnL: {pos.get('realized_pnl', 0):.4f}\n\n")

            if len(report["positions"]) > 10:
                f.write(f"... 还有 {len(report['positions']) - 10} 个仓位\n\n")

        print(f"✅ Markdown报告已保存: {md_path}")


def generate_grafana_json(
    report: Dict[str, Any],
    output_path: Path,
) -> None:
    """生成Grafana格式的JSON"""
    grafana_data = []

    for op in report["operations"]:
        grafana_data.append(
            {
                "time": op["operation_time"],
                "symbol": (
                    report["positions"][0]["symbol"]
                    if report["positions"]
                    else "UNKNOWN"
                ),
                "position_id": op["position_id"],
                "operation_type": op["operation_type"],
                "price": op["price"],
                "size": op["size"],
                "pnl": op["pnl"],
                "reason": op["reason"],
            }
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(grafana_data, f, indent=2, default=str)

    print(f"✅ Grafana JSON已保存: {output_path}")
