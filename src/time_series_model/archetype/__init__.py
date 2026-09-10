"""
Archetype module - 三层配置加载器 (Gate / Evidence / Execution)

配置结构：
  config/strategies/{strategy}/archetypes/
    gate.yaml       # Gate 规则 (hard_gates / guardrails / system_safety)
    evidence.yaml   # Evidence 规则 (quantile_mapping / affects)
    execution.yaml  # Execution 约束 (RR / holding_bars / direction_policy)

用法：
    from src.time_series_model.archetype import load_strategy_archetype

    archetype = load_strategy_archetype("bpc")
    # archetype.gate_config, archetype.evidence_config, archetype.execution_config
"""

from src.time_series_model.archetype.loader import (
    StrategyArchetype,
    GateConfig,
    GateRule,
    EvidenceConfig,
    EvidenceFeature,
    ExecutionConfig,
    load_strategy_archetype,
    load_all_strategy_archetypes,
)

__all__ = [
    "StrategyArchetype",
    "GateConfig",
    "GateRule",
    "EvidenceConfig",
    "EvidenceFeature",
    "ExecutionConfig",
    "load_strategy_archetype",
    "load_all_strategy_archetypes",
]
