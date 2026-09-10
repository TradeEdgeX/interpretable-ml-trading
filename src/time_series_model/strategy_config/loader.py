"""Loader for per-strategy configuration directories."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def _load_yaml_file(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data


@dataclass
class ModuleFunctionConfig:
    module: str
    function: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeaturePipelineConfig:
    requested_features: List[str] = field(default_factory=list)
    # Optional: feature module names (e.g. "oi_scene_semantic_scores_f") removed from
    # effective requested_features at load time even if duplicated in requested_features.
    forbidden_requested_features: List[str] = field(default_factory=list)
    # Optional: output-column names to multiply by -1 BEFORE training/inference.
    # Used to align negative-direction factors into a consistent "higher = more bullish" convention.
    invert_features: List[str] = field(default_factory=list)
    # Optional: columns to exclude from the model input columns (feature_cols).
    #
    # Important:
    # - This does NOT prevent the column from being computed / present in the feature dataframe.
    # - This is used to keep label/backtest-required columns (e.g. raw price-unit `atr`)
    #   available while preventing them from being fed into the model as inputs.
    exclude_columns: List[str] = field(default_factory=list)
    post_processors: List[ModuleFunctionConfig] = field(default_factory=list)
    selector: Optional[ModuleFunctionConfig] = None
    ensure_signal: Optional[Dict[str, Any]] = None


@dataclass
class LabelConfig:
    target_column: str
    generator: ModuleFunctionConfig
    filters: List[Dict[str, Any]] = field(default_factory=list)
    post_label_filters: List[Dict[str, Any]] = field(default_factory=list)
    # Optional: model_hints to override model.yaml defaults (e.g., task_type, objective)
    # Useful for labels like labels_return_tree.yaml that need regression instead of binary
    model_hints: Dict[str, Any] = field(default_factory=dict)
    # Optional: KPI definitions for specialized evaluation (e.g., Return Tree ranking KPIs)
    kpi_definition: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VolatilityModelConfig:
    """Configuration for volatility model training."""

    enabled: bool = False
    config_path: Optional[str] = (
        None  # Path to volatility_model.yaml, None = use default
    )
    target_column: str = "future_volatility"  # Column name for volatility labels


@dataclass
class ModelConfig:
    trainer: ModuleFunctionConfig
    prediction: Dict[str, Any] = field(default_factory=dict)
    output: Dict[str, Any] = field(default_factory=dict)
    volatility_model: Optional[VolatilityModelConfig] = None


@dataclass
class EvaluationConfig:
    metrics: List[Dict[str, Any]] = field(default_factory=list)
    save_details: bool = False
    results_file: Optional[str] = None


@dataclass
class BacktestConfig:
    enabled: bool = False
    class_path: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyConfig:
    name: str
    path: Path
    features: FeaturePipelineConfig
    labels: LabelConfig
    model: ModelConfig
    evaluation: EvaluationConfig
    backtest: BacktestConfig
    meta: Dict[str, Any] = field(default_factory=dict)


class StrategyConfigLoader:
    """Load strategy configuration files from a directory."""

    REQUIRED_FILES = ("features.yaml", "labels.yaml", "model.yaml")
    OPTIONAL_FILES = ("evaluation.yaml", "backtest.yaml", "meta.yaml")

    def __init__(
        self,
        config_dir: Path | str,
        *,
        strict_name_match: bool = False,
        labels_override: Optional[Path | str] = None,
        features_override: Optional[Path | str] = None,
    ) -> None:
        self.config_dir = Path(config_dir)
        self.strict_name_match = bool(strict_name_match)
        self.labels_override = Path(labels_override) if labels_override else None
        self.features_override = Path(features_override) if features_override else None
        if not self.config_dir.exists() or not self.config_dir.is_dir():
            raise FileNotFoundError(f"Config directory not found: {self.config_dir}")
        if self.labels_override and not self.labels_override.exists():
            raise FileNotFoundError(
                f"Labels override file not found: {self.labels_override}"
            )
        if self.features_override and not self.features_override.exists():
            raise FileNotFoundError(
                f"Features override file not found: {self.features_override}"
            )

    def load(self) -> StrategyConfig:
        missing_required = [
            fname
            for fname in self.REQUIRED_FILES
            if not (self.config_dir / fname).exists()
        ]
        if missing_required:
            raise FileNotFoundError(
                f"Config directory {self.config_dir} is missing required files: {missing_required}"
            )

        missing_optional = [
            fname
            for fname in self.OPTIONAL_FILES
            if not (self.config_dir / fname).exists()
        ]
        # Only warn if in verbose mode or if this is a tree model config (not nnmultihead)
        # nnmultihead configs don't need these optional files, so suppress warnings
        is_nnmultihead = "nnmultihead" in str(self.config_dir)
        if missing_optional and not is_nnmultihead:
            for fname in missing_optional:
                print(f"   ⚠️  Optional config '{fname}' not found in {self.config_dir}")

        # Support --features override: use specified file instead of default features.yaml
        features_path = self.features_override or (self.config_dir / "features.yaml")
        features_data = _load_yaml_file(features_path)
        if self.features_override:
            print(f"   🔧 Using features override: {self.features_override}")
        # Support --labels override: use specified file instead of default labels.yaml
        labels_path = self.labels_override or (self.config_dir / "labels.yaml")
        labels_data = _load_yaml_file(labels_path)
        if self.labels_override:
            print(f"   📋 Using labels override: {self.labels_override}")
        model_data = _load_yaml_file(self.config_dir / "model.yaml")

        evaluation_data = (
            _load_yaml_file(self.config_dir / "evaluation.yaml")
            if (self.config_dir / "evaluation.yaml").exists()
            else {}
        )
        backtest_data = (
            _load_yaml_file(self.config_dir / "backtest.yaml")
            if (self.config_dir / "backtest.yaml").exists()
            else {}
        )
        meta_data = (
            _load_yaml_file(self.config_dir / "meta.yaml")
            if (self.config_dir / "meta.yaml").exists()
            else {}
        )

        # ------------------------------------------------------------------
        # Strategy ID convention:
        # - The strategy "ID" is ALWAYS the directory name: config/strategies/<dir_name>
        # - YAML `name:` fields are treated as optional "declared name" (display-only).
        #   If declared_name != dir_name, we warn (or error in strict mode) to catch drift.
        # This avoids having to keep `name:` duplicated across features/labels/model YAMLs.
        # ------------------------------------------------------------------
        dir_name = self.config_dir.name
        declared_name = (
            features_data.get("name")
            or labels_data.get("name")
            or model_data.get("name")
        )
        if declared_name and declared_name != dir_name:
            msg = (
                f"Strategy name mismatch: directory='{dir_name}' but YAML declared name='{declared_name}'. "
                f"Convention: strategy id is the directory name. "
                f"Either rename the directory or update YAML name fields (or remove them)."
            )
            if self.strict_name_match:
                raise ValueError(msg)
            print(f"   ⚠️  {msg}")

        name = dir_name

        feature_cfg = self._parse_feature_config(features_data)
        label_cfg = self._parse_label_config(labels_data)
        model_cfg = self._parse_model_config(model_data)
        evaluation_cfg = self._parse_evaluation_config(evaluation_data)
        backtest_cfg = self._parse_backtest_config(backtest_data)

        meta = meta_data.get("strategy", meta_data)
        if declared_name and declared_name != dir_name:
            # Preserve for debugging / UI display if needed.
            try:
                meta = dict(meta)
            except Exception:
                meta = {"meta": meta}
            meta["declared_name"] = declared_name

        return StrategyConfig(
            name=name,
            path=self.config_dir,
            features=feature_cfg,
            labels=label_cfg,
            model=model_cfg,
            evaluation=evaluation_cfg,
            backtest=backtest_cfg,
            meta=meta,
        )

    def _parse_feature_config(self, data: Dict[str, Any]) -> FeaturePipelineConfig:
        pipeline = data.get("feature_pipeline", {})
        requested = pipeline.get("requested_features", []) or []

        # Support new structured format: requested_features = {required: [...], optional_blocks: {...}}
        # Flatten to a single list for backward compatibility with tree models
        if isinstance(requested, dict):
            required_list = requested.get("required") or []
            optional_blocks_dict = requested.get("optional_blocks") or {}
            # Flatten: combine required + all optional_blocks features into a single list
            flattened = list(required_list)
            for block_features in optional_blocks_dict.values():
                if isinstance(block_features, list):
                    flattened.extend(block_features)
            requested = flattened

        forbidden_raw = pipeline.get("forbidden_requested_features", []) or []
        forbidden_list = [
            str(x).strip()
            for x in forbidden_raw
            if isinstance(x, (str, int)) and str(x).strip()
        ]
        forbidden_set = set(forbidden_list)

        invert_features = pipeline.get("invert_features", []) or []
        exclude_columns = pipeline.get("exclude_columns", []) or []
        post_processors = [
            self._parse_module_function(entry)
            for entry in pipeline.get("post_processors", []) or []
        ]
        selector = (
            self._parse_module_function(pipeline["selector"])
            if pipeline.get("selector")
            else None
        )
        ensure_signal = pipeline.get("ensure_signal_column")
        if forbidden_set:
            requested = [
                str(r).strip()
                for r in requested
                if str(r).strip() and str(r).strip() not in forbidden_set
            ]
        return FeaturePipelineConfig(
            requested_features=requested,
            forbidden_requested_features=forbidden_list,
            invert_features=(
                invert_features if isinstance(invert_features, list) else []
            ),
            exclude_columns=(
                [str(x).strip() for x in exclude_columns if str(x).strip()]
                if isinstance(exclude_columns, list)
                else []
            ),
            post_processors=post_processors,
            selector=selector,
            ensure_signal=ensure_signal,
        )

    def _parse_label_config(self, data: Dict[str, Any]) -> LabelConfig:
        target = data.get("target_column", "label")
        generator_cfg = data.get("label_generator") or data.get("generator") or {}
        generator = self._parse_module_function(generator_cfg)
        filters = data.get("filters", []) or []
        post_filters = data.get("post_label_filters", []) or []
        model_hints = data.get("model_hints", {}) or {}
        kpi_definition = data.get("kpi_definition", {}) or {}
        return LabelConfig(
            target_column=target,
            generator=generator,
            filters=filters,
            post_label_filters=post_filters,
            model_hints=model_hints,
            kpi_definition=kpi_definition,
        )

    def _parse_model_config(self, data: Dict[str, Any]) -> ModelConfig:
        trainer = self._parse_module_function(data.get("trainer", {}))
        prediction = data.get("prediction", {})
        output = data.get("output", {})

        # Parse volatility model config (optional)
        vol_model_data = data.get("volatility_model", {})
        volatility_model = None
        if vol_model_data and vol_model_data.get("enabled", False):
            volatility_model = VolatilityModelConfig(
                enabled=True,
                config_path=vol_model_data.get("config_path"),
                target_column=vol_model_data.get("target_column", "future_volatility"),
            )

        return ModelConfig(
            trainer=trainer,
            prediction=prediction,
            output=output,
            volatility_model=volatility_model,
        )

    def _parse_evaluation_config(self, data: Dict[str, Any]) -> EvaluationConfig:
        evaluation = data.get("evaluation", {})
        metrics = evaluation.get("metrics", []) or []
        save_details = evaluation.get("save_details", False)
        results_file = evaluation.get("results_file")
        return EvaluationConfig(
            metrics=metrics, save_details=save_details, results_file=results_file
        )

    def _parse_backtest_config(self, data: Dict[str, Any]) -> BacktestConfig:
        backtest = data.get("backtest", {})
        if not backtest:
            return BacktestConfig()
        return BacktestConfig(
            enabled=backtest.get("enabled", False),
            class_path=backtest.get("class"),
            params=backtest.get("params", {}),
        )

    def _parse_module_function(
        self, entry: Optional[Dict[str, Any]]
    ) -> ModuleFunctionConfig:
        if not entry:
            raise ValueError("Module/function configuration is missing")
        module = entry.get("module")
        function = entry.get("function")
        if not module or not function:
            raise ValueError(f"Incomplete module/function config: {entry}")
        params = entry.get("params", {}) or {}
        return ModuleFunctionConfig(module=module, function=function, params=params)
