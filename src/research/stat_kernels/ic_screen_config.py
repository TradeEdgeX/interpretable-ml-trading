"""Load per-strategy holdout IC screen rules from ``ic_screen.yaml``."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

TREE_STRATEGIES_ROOT = Path("config/strategies/tree_strategies")


@dataclass
class ICScreenWriteback:
    mode: str = "columns"
    top_n_columns: int | None = 20
    top_n_nodes: int | None = None
    always_include: list[str] = field(default_factory=lambda: ["atr_f"])
    invert_mode: str = "none"


@dataclass
class ICScreenConfig:
    """Holdout IC feature-selection rules (see strategy ``ic_screen.yaml``)."""

    target: str = "label"
    horizons: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    max_lag: int = 5
    allowed_best_lags: list[int] | None = None
    min_ic: float = 0.02
    min_n: int = 200
    reject_peak_at: int | None = None
    writeback: ICScreenWriteback = field(default_factory=ICScreenWriteback)
    label_horizon_bars: int | None = None
    config_dir: Path | None = None

    def horizons_csv(self) -> str:
        return ",".join(str(h) for h in self.horizons)

    def allowed_best_lags_csv(self) -> str | None:
        if not self.allowed_best_lags:
            return None
        return ",".join(str(h) for h in self.allowed_best_lags)

    def summary_line(self) -> str:
        lag_rule = (
            f"peak ∈ {{{self.allowed_best_lags_csv()}}}"
            if self.allowed_best_lags
            else f"peak ≤ {self.max_lag}"
        )
        parts = [
            f"target={self.target}",
            lag_rule,
            f"|IC|≥{self.min_ic}",
            f"horizons={self.horizons_csv()}",
        ]
        if self.label_horizon_bars is not None:
            parts.insert(1, f"label_H={self.label_horizon_bars}")
        if self.reject_peak_at is not None:
            parts.append(f"reject_peak@{self.reject_peak_at}")
        return ", ".join(parts)

    def to_ic_prune_kwargs(self) -> dict[str, Any]:
        wb = self.writeback
        out: dict[str, Any] = {
            "target": self.target,
            "horizons": self.horizons_csv(),
            "max_lag": self.max_lag,
            "min_ic": self.min_ic,
            "min_n": self.min_n,
            "writeback_mode": wb.mode,
            "top_n_columns": wb.top_n_columns,
            "top_n_nodes": wb.top_n_nodes,
            "invert_mode": wb.invert_mode,
            "always_include": list(wb.always_include),
        }
        if self.allowed_best_lags:
            out["allowed_best_lags"] = self.allowed_best_lags_csv()
        if self.reject_peak_at is not None:
            out["reject_peak_at"] = self.reject_peak_at
        return out

    def default_writeback_paths(self) -> dict[str, Path]:
        if self.config_dir is None:
            return {}
        root = self.config_dir
        return {
            "write_features_yaml": root / "features.yaml",
            "write_model_features_yaml": root / "archetypes" / "model_features.yaml",
        }


def resolve_strategy_config_dir(
    *,
    strategy: str | None = None,
    config_dir: str | Path | None = None,
    project_root: Path | None = None,
) -> Path | None:
    root = project_root or Path(__file__).resolve().parents[3]
    if config_dir is not None:
        path = Path(config_dir)
        if not path.is_absolute():
            path = (root / path).resolve()
        return path if path.is_dir() else None
    if not strategy:
        return None
    for candidate in (
        root / TREE_STRATEGIES_ROOT / strategy,
        root / "config/strategies" / strategy,
    ):
        if candidate.is_dir():
            return candidate.resolve()
    return None


def _read_label_horizon(config_dir: Path) -> int | None:
    labels_path = config_dir / "labels.yaml"
    if not labels_path.is_file():
        return None
    data = yaml.safe_load(labels_path.read_text(encoding="utf-8")) or {}
    gen = data.get("label_generator") or data.get("generator") or {}
    params = gen.get("params") or {}
    horizon = params.get("horizon")
    if horizon is None:
        meta = data.get("label_meta") or {}
        raw = meta.get("horizon")
        if isinstance(raw, str) and raw.split():
            try:
                horizon = int(str(raw).split()[0])
            except ValueError:
                horizon = None
    return int(horizon) if horizon is not None else None


def _parse_int_list(raw: Any) -> list[int] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        return [int(p) for p in parts] if parts else None
    if isinstance(raw, (list, tuple)):
        return [int(x) for x in raw]
    return None


def load_ic_screen(config_dir: Path | str) -> ICScreenConfig:
    """Load ``ic_screen.yaml`` from a strategy config directory."""
    path = Path(config_dir)
    ic_path = path / "ic_screen.yaml"
    if not ic_path.is_file():
        raise FileNotFoundError(f"Missing ic_screen.yaml in {path}")

    data = yaml.safe_load(ic_path.read_text(encoding="utf-8")) or {}
    block = data.get("ic_screen") or data
    if not isinstance(block, dict):
        raise ValueError(f"ic_screen.yaml must contain mapping 'ic_screen': {ic_path}")

    wb_raw = block.get("writeback") or {}
    if not isinstance(wb_raw, dict):
        wb_raw = {}
    always = wb_raw.get("always_include") or ["atr_f"]
    writeback = ICScreenWriteback(
        mode=str(wb_raw.get("mode", "columns")),
        top_n_columns=wb_raw.get("top_n_columns", 20),
        top_n_nodes=wb_raw.get("top_n_nodes"),
        always_include=[str(x) for x in always] if isinstance(always, list) else ["atr_f"],
        invert_mode=str(wb_raw.get("invert_mode", "none")),
    )

    label_h = block.get("label_horizon_bars")
    if label_h is None:
        label_h = _read_label_horizon(path)

    cfg = ICScreenConfig(
        target=str(block.get("target", "label")),
        horizons=_parse_int_list(block.get("horizons")) or [1, 2, 3, 4, 5],
        max_lag=int(block.get("max_lag", 5)),
        allowed_best_lags=_parse_int_list(block.get("allowed_best_lags")),
        min_ic=float(block.get("min_ic", 0.02)),
        min_n=int(block.get("min_n", 200)),
        reject_peak_at=(
            int(block["reject_peak_at"]) if block.get("reject_peak_at") is not None else None
        ),
        writeback=writeback,
        label_horizon_bars=int(label_h) if label_h is not None else None,
        config_dir=path.resolve(),
    )
    return cfg


def load_ic_screen_optional(
    *,
    strategy: str | None = None,
    config_dir: str | Path | None = None,
    project_root: Path | None = None,
) -> ICScreenConfig | None:
    resolved = resolve_strategy_config_dir(
        strategy=strategy, config_dir=config_dir, project_root=project_root
    )
    if resolved is None:
        return None
    try:
        return load_ic_screen(resolved)
    except FileNotFoundError:
        return None


def ic_prune_params_to_argv(params: dict[str, Any]) -> list[str]:
    """Convert merged ic-prune params to mlbot CLI argv fragments."""
    argv: list[str] = []
    scalar_flags = [
        ("holdout_start", "--holdout-start"),
        ("holdout_end", "--holdout-end"),
        ("horizons", "--horizons"),
        ("max_lag", "--max-lag"),
        ("allowed_best_lags", "--allowed-best-lags"),
        ("reject_peak_at", "--reject-peak-at"),
        ("min_ic", "--min-ic"),
        ("min_n", "--min-n"),
        ("target", "--target"),
        ("writeback_mode", "--writeback-mode"),
        ("top_n_columns", "--top-n-columns"),
        ("top_n_nodes", "--top-n-nodes"),
        ("invert_mode", "--invert-mode"),
        ("always_include", "--always-include"),
        ("intersect_features_yaml", "--intersect-features-yaml"),
        ("emit_monotone_constraints", "--emit-monotone-constraints"),
    ]
    for key, flag in scalar_flags:
        val = params.get(key)
        if val is None:
            continue
        argv += [flag, str(val)]

    write_yaml = params.get("write_features_yaml")
    if write_yaml is False:
        argv.append("--no-write-features-yaml")
    elif write_yaml:
        argv += ["--write-features-yaml", str(write_yaml)]

    wmf = params.get("write_model_features_yaml")
    if wmf is False:
        argv.append("--no-write-model-features-yaml")
    elif wmf:
        argv += ["--write-model-features-yaml", str(wmf)]
    return argv


IC_PRUNE_OVERRIDE_KEYS = frozenset(
    {
        "target",
        "horizons",
        "max_lag",
        "allowed_best_lags",
        "min_ic",
        "min_n",
        "reject_peak_at",
        "writeback_mode",
        "top_n_columns",
        "top_n_nodes",
        "invert_mode",
        "always_include",
        "intersect_features_yaml",
        "write_features_yaml",
        "write_model_features_yaml",
        "holdout_start",
        "holdout_end",
        "emit_monotone_constraints",
    }
)


def resolve_ic_prune_params(
    *,
    strategy: str | None = None,
    config_dir: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Merge ic_screen.yaml with explicit step/CLI overrides (overrides win)."""
    screen = load_ic_screen_optional(
        strategy=strategy, config_dir=config_dir, project_root=project_root
    )
    merged: dict[str, Any] = {}
    if screen is not None:
        merged.update(screen.to_ic_prune_kwargs())
        merged["_ic_screen_summary"] = screen.summary_line()
        merged["_strategy_config_dir"] = str(screen.config_dir)
        for key, val in screen.default_writeback_paths().items():
            merged.setdefault(key, str(val))
    else:
        merged.update(
            {
                "target": "label",
                "horizons": "1,2,3,4,5",
                "max_lag": 5,
                "min_ic": 0.02,
                "min_n": 200,
                "writeback_mode": "columns",
                "top_n_columns": 20,
                "invert_mode": "none",
                "always_include": ["atr_f"],
            }
        )

    for key, val in (overrides or {}).items():
        if val is None or key not in IC_PRUNE_OVERRIDE_KEYS:
            continue
        if key == "always_include" and isinstance(val, str):
            merged[key] = [x.strip() for x in val.split(",") if x.strip()]
        elif key in ("write_features_yaml", "write_model_features_yaml", "intersect_features_yaml"):
            merged[key] = val
        else:
            merged[key] = val

    if isinstance(merged.get("always_include"), list):
        merged["always_include"] = ",".join(str(x) for x in merged["always_include"])
    return merged
