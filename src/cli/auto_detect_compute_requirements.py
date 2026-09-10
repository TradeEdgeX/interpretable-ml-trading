"""
自动推导计算需求：从gate rules和regime配置中提取特征并映射到optional blocks

实施方案3：自动推导计算需求（最智能）
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set, Any
import yaml
import sys

# 获取项目根目录
def get_project_root() -> Path:
    """获取项目根目录"""
    current = Path(__file__).resolve()
    while current != current.parent:
        if (current / "pyproject.toml").exists() or (current / ".git").exists():
            return current
        current = current.parent
    return Path(__file__).parent.parent.parent

PROJECT_ROOT = get_project_root()


# Reserved keys in when-then "when" blocks (not feature names)
_WHEN_RESERVED = frozenset({"any_of", "all_of", "min_matches", "min_matches_any"})


def _collect_features_from_when(when: Any, out: Set[str]) -> None:
    """Recursively collect feature names from a when-then 'when' block."""
    if not isinstance(when, dict):
        return
    for key, val in when.items():
        key = str(key).strip()
        if key in _WHEN_RESERVED:
            if isinstance(val, list):
                for item in val:
                    _collect_features_from_when(item, out)
            continue
        # Key is a feature name (e.g. shd_pct, cvd_change_5); val is the condition (quantile_gt etc.)
        if isinstance(val, dict) and val:
            out.add(key)
            # Do not recurse into condition dict (quantile_gt, quantile_lt are not feature names)
        elif isinstance(val, list):
            for item in val:
                _collect_features_from_when(item, out)


def extract_required_features_from_execution_archetypes(
    execution_archetypes_path: str | Path = "config/strategies/tpc/archetypes/gate.yaml",
) -> Set[str]:
    """
    从 gate / execution_archetypes 配置中提取所有 gate rules 需要的特征。

    支持两种格式:
      - BPC gate.yaml: hard_gates[] / soft_gates[] / guardrails[] 各有 when 字段
      - Legacy execution_archetypes.yaml: archetypes[].when_then_rules[].when

    Returns:
        需要的特征名称集合
    """
    path = Path(execution_archetypes_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    
    if not path.exists():
        return set()
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
    except Exception:
        return set()
    
    required_features: Set[str] = set()
    
    # Format 1: BPC gate.yaml (hard_gates / soft_gates / guardrails)
    for section_key in ('hard_gates', 'soft_gates', 'guardrails'):
        rules = config.get(section_key)
        if not isinstance(rules, list):
            continue
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            when = rule.get('when')
            if when:
                _collect_features_from_when(when, required_features)
    
    # Format 2: Legacy execution_archetypes.yaml (archetypes[].when_then_rules[])
    for arch_name, arch_data in config.get('archetypes', {}).items():
        if not isinstance(arch_data, dict):
            continue
        for rule in arch_data.get('when_then_rules', []) or []:
            if not isinstance(rule, dict):
                continue
            when = rule.get('when')
            if when:
                _collect_features_from_when(when, required_features)
    
    return required_features


def resolve_feature_dependencies(
    nodes: Set[str],
    feature_dependencies: Dict[str, Any] | None,
) -> Set[str]:
    """
    递归解析feature nodes的依赖关系
    
    Args:
        nodes: 初始的feature nodes集合
        feature_dependencies: feature dependencies配置（可以为None）
    
    Returns:
        包含所有依赖的feature nodes集合
    """
    if feature_dependencies is None:
        return set(nodes)
    
    all_nodes = set(nodes)
    feats = feature_dependencies.get('features', {}) or {}
    to_check = list(nodes)
    checked = set()
    
    while to_check:
        node = to_check.pop(0)
        if node in checked:
            continue
        checked.add(node)
        
        node_cfg = feats.get(node)
        if isinstance(node_cfg, dict):
            node_deps = node_cfg.get('dependencies', [])
            if isinstance(node_deps, list):
                for dep in node_deps:
                    if dep not in all_nodes:
                        all_nodes.add(dep)
                        to_check.append(dep)
    
    return all_nodes


def map_features_to_tier_nodes(
    features: Set[str],
    feature_dependencies: Dict[str, Any],
) -> Set[str]:
    """
    将gate规则需要的特征映射到对应的feature nodes
    
    Args:
        features: gate规则需要的特征列名集合（如path_efficiency_pct, jump_risk_pct）
        feature_dependencies: feature dependencies配置
    
    Returns:
        需要的feature nodes集合（如path_efficiency_pct_f, jump_risk_pct_f）
    """
    required_nodes: Set[str] = set()
    
    if not feature_dependencies:
        return required_nodes
    
    feats = feature_dependencies.get('features', {}) or {}
    
    # 构建特征列名 -> feature node的映射
    feature_to_nodes: Dict[str, List[str]] = {}
    for node_name, node_cfg in feats.items():
        if not isinstance(node_cfg, dict):
            continue
        output_cols = node_cfg.get('output_columns', [])
        if isinstance(output_cols, list):
            for col in output_cols:
                if col not in feature_to_nodes:
                    feature_to_nodes[col] = []
                feature_to_nodes[col].append(node_name)
    
    # 查找每个特征对应的nodes
    for feat in features:
        nodes = feature_to_nodes.get(feat, [])
        if nodes:
            # 通常一个特征只对应一个node，但如果有多个，都添加
            required_nodes.update(nodes)
    
    return required_nodes


def map_features_to_optional_blocks(
    features: Set[str],
    optional_blocks_library: Dict[str, List[str]],
    feature_dependencies: Dict[str, any] | None = None,
) -> Set[str]:
    """
    将特征映射到optional blocks
    
    Args:
        features: 需要的特征名称集合
        optional_blocks_library: optional blocks定义（block_name -> feature_nodes）
        feature_dependencies: feature dependencies配置（用于查找特征属于哪个node）
    
    Returns:
        需要的optional blocks集合
    """
    required_blocks: Set[str] = set()
    
    # 特征到block的简单映射规则
    feature_to_block_patterns = {
        'vpin_block': ['vpin'],
        'volume_profile_block': ['vp_', 'volume_profile', 'vpvr_', 'vpvr'],
        'trade_cluster_block': ['trade_cluster'],
    }
    
    # 直接模式匹配
    for block_name, patterns in feature_to_block_patterns.items():
        if block_name not in optional_blocks_library:
            continue
        for feat in features:
            for pattern in patterns:
                if pattern.lower() in feat.lower():
                    required_blocks.add(block_name)
                    break
    
    # 通过feature_dependencies查找（如果提供）
    if feature_dependencies:
        feats = feature_dependencies.get('features', {}) or {}
        
        # 构建特征到node的映射
        feature_to_nodes: Dict[str, List[str]] = {}
        for node_name, node_cfg in feats.items():
            if not isinstance(node_cfg, dict):
                continue
            output_cols = node_cfg.get('output_columns', [])
            if isinstance(output_cols, list):
                for col in output_cols:
                    if col not in feature_to_nodes:
                        feature_to_nodes[col] = []
                    feature_to_nodes[col].append(node_name)
        
        # 查找特征属于哪个node，然后查找node属于哪个block
        for feat in features:
            nodes = feature_to_nodes.get(feat, [])
            for node in nodes:
                for block_name, block_nodes in optional_blocks_library.items():
                    if node in block_nodes:
                        required_blocks.add(block_name)
                        break
    
    return required_blocks


def ensure_tier_features(
    required_nodes: Set[str],
    tier_file_path: Path,
    feature_dependencies: Dict[str, Any] | None = None,
) -> List[str]:
    """
    确保feature nodes在tier文件中，如果不在则添加
    
    Args:
        required_nodes: 需要的feature nodes集合
        tier_file_path: tier文件路径（如features_tier0.yaml）
        feature_dependencies: feature dependencies配置（用于解析依赖）
    
    Returns:
        添加的nodes列表（如果已经存在则返回空列表）
    """
    if not tier_file_path.exists():
        return []
    
    # 读取tier文件
    with open(tier_file_path, 'r', encoding='utf-8') as f:
        tier_content = f.read()
        tier_list = yaml.safe_load(tier_content) or []
    
    if not isinstance(tier_list, list):
        return []
    
    # 转换为集合以便快速查找
    existing_nodes = {str(node).strip() for node in tier_list if str(node).strip()}
    
    # 找出缺失的nodes
    missing_nodes = required_nodes - existing_nodes
    
    if not missing_nodes:
        return []
    
    # 处理依赖关系：如果某个node有依赖，确保依赖的nodes也被添加
    nodes_to_add = set(missing_nodes)
    if feature_dependencies:
        feats = feature_dependencies.get('features', {}) or {}
        for node in list(missing_nodes):
            node_cfg = feats.get(node)
            if isinstance(node_cfg, dict):
                deps = node_cfg.get('dependencies', [])
                if isinstance(deps, list):
                    for dep in deps:
                        if dep not in existing_nodes and dep not in nodes_to_add:
                            nodes_to_add.add(dep)
    
    # 添加缺失的nodes到tier文件
    # 保持原有格式，在文件末尾添加（如果有注释区域，在注释后添加）
    new_nodes = sorted(nodes_to_add)
    
    # 检查文件末尾是否有注释区域
    lines = tier_content.split('\n')
    comment_section_end = len(lines)
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i].strip()
        if line and not line.startswith('#'):
            comment_section_end = i + 1
            break
    
    # 构建新的tier文件内容
    # 保留原有内容，在末尾添加新nodes
    new_lines = lines[:comment_section_end]
    if new_lines and new_lines[-1].strip():
        new_lines.append('')
    
    # 添加新nodes
    for node in new_nodes:
        new_lines.append(f"- {node}")
    
    # 写入文件
    with open(tier_file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(new_lines))
        if not new_lines[-1].endswith('\n'):
            f.write('\n')
    
    return sorted(new_nodes)


def auto_detect_compute_requirements(
    task_spec_path: str | Path,
    execution_archetypes_path: str | Path = "config/strategies/tpc/archetypes/gate.yaml",
    feature_dependencies_path: str | Path = "config/feature_dependencies.yaml",
) -> Set[str]:
    """
    自动推导计算需求：从gate rules和regime配置中提取特征并映射到optional blocks
    
    Args:
        task_spec_path: TaskSpec文件路径
        execution_archetypes_path: execution_archetypes.yaml路径
        feature_dependencies_path: feature_dependencies.yaml路径
    
    Returns:
        需要的optional blocks集合
    """
    # 1. 提取需要的特征
    required_features = extract_required_features_from_execution_archetypes(
        execution_archetypes_path
    )
    
    if not required_features:
        return set()
    
    # 2. 读取TaskSpec获取optional_blocks_library
    ts_path = Path(task_spec_path)
    if not ts_path.is_absolute():
        ts_path = PROJECT_ROOT / ts_path
    
    if not ts_path.exists():
        return set()
    
    with open(ts_path, 'r', encoding='utf-8') as f:
        ts_obj = yaml.safe_load(f) or {}
    
    # 获取feature_plan
    fp: Dict[str, any] = {}
    plan_ref = str(ts_obj.get("feature_plan_ref") or "").strip()
    if plan_ref:
        p = Path(plan_ref)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if p.exists():
            objp = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            if isinstance(objp, dict):
                fp = objp.get("feature_plan") or {}
    
    # Apply overrides
    fp_over = ts_obj.get("feature_plan_overrides") or {}
    if isinstance(fp_over, dict) and fp_over:
        fp = {**fp, **fp_over}
    
    optional_blocks_library = fp.get("optional_blocks_library", {}) or {}
    
    # 3. 读取feature_dependencies（可选）
    feature_dependencies = None
    deps_path = Path(feature_dependencies_path)
    if not deps_path.is_absolute():
        deps_path = PROJECT_ROOT / deps_path
    if deps_path.exists():
        with open(deps_path, 'r', encoding='utf-8') as f:
            feature_dependencies = yaml.safe_load(f) or {}
    
    # 4. 映射特征到blocks
    required_blocks = map_features_to_optional_blocks(
        required_features,
        optional_blocks_library,
        feature_dependencies,
    )
    
    return required_blocks


def auto_detect_tier_features(
    task_spec_path: str | Path,
    base_config_dir: str | Path,
    execution_archetypes_path: str | Path = "config/strategies/tpc/archetypes/gate.yaml",
    feature_dependencies_path: str | Path = "config/feature_dependencies.yaml",
) -> List[str]:
    """
    自动检测并确保gate规则需要的tier features被包含在tier文件中
    
    Args:
        task_spec_path: TaskSpec文件路径
        base_config_dir: base config目录（用于定位tier文件）
        execution_archetypes_path: execution_archetypes.yaml路径
        feature_dependencies_path: feature_dependencies.yaml路径
    
    Returns:
        添加的feature nodes列表
    """
    # 1. 提取gate规则需要的特征
    required_features = extract_required_features_from_execution_archetypes(
        execution_archetypes_path
    )
    
    if not required_features:
        return []
    
    # 2. 读取feature_dependencies
    deps_path = Path(feature_dependencies_path)
    if not deps_path.is_absolute():
        deps_path = PROJECT_ROOT / deps_path
    if not deps_path.exists():
        return []
    
    with open(deps_path, 'r', encoding='utf-8') as f:
        feature_dependencies = yaml.safe_load(f) or {}
    
    # 3. 映射特征到feature nodes
    required_nodes = map_features_to_tier_nodes(required_features, feature_dependencies)
    
    if not required_nodes:
        return []
    
    # 4. 确定tier文件路径
    # 从TaskSpec读取tier_feature_files配置
    ts_path = Path(task_spec_path)
    if not ts_path.is_absolute():
        ts_path = PROJECT_ROOT / ts_path
    
    if not ts_path.exists():
        return []
    
    with open(ts_path, 'r', encoding='utf-8') as f:
        ts_obj = yaml.safe_load(f) or {}
    
    # 获取feature_plan
    fp: Dict[str, Any] = {}
    plan_ref = str(ts_obj.get("feature_plan_ref") or "").strip()
    if plan_ref:
        p = Path(plan_ref)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if p.exists():
            objp = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            if isinstance(objp, dict):
                fp = objp.get("feature_plan") or {}
    
    # Apply overrides
    fp_over = ts_obj.get("feature_plan_overrides") or {}
    if isinstance(fp_over, dict) and fp_over:
        fp = {**fp, **fp_over}
    
    tier_feature_files = fp.get("tier_feature_files", {}) or {}
    tiers_enabled = fp.get("tiers_enabled", []) or []
    
    # 默认使用tier0（如果启用）
    tier_file_path = None
    if "tier0" in tiers_enabled:
        tier0_path = tier_feature_files.get("tier0")
        if tier0_path:
            tier_file_path = Path(tier0_path)
            if not tier_file_path.is_absolute():
                tier_file_path = PROJECT_ROOT / tier_file_path
    elif base_config_dir:
        # 回退：尝试从base_config_dir推断tier0路径
        base_dir = Path(base_config_dir)
        if not base_dir.is_absolute():
            base_dir = PROJECT_ROOT / base_dir
        # 常见的tier0文件路径
        possible_paths = [
            base_dir / "path_primitives_4h_80h_min" / "features_tier0.yaml",
            base_dir / "features_tier0.yaml",
        ]
        for p in possible_paths:
            if p.exists():
                tier_file_path = p
                break
    
    if not tier_file_path or not tier_file_path.exists():
        return []
    
    # 5. 确保tier文件中包含这些nodes
    added_nodes = ensure_tier_features(required_nodes, tier_file_path, feature_dependencies)
    
    return added_nodes


if __name__ == "__main__":
    # 测试
    required_blocks = auto_detect_compute_requirements(
        "config/tasks/task_spec_highcap6_2024_202510.yaml"
    )
    print(f"自动推导的required blocks: {sorted(required_blocks)}")
