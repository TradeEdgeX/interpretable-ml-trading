from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def _binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(int)
    mask = np.isfinite(y_score) & np.isfinite(y_true)
    y_true = y_true[mask]
    y_score = y_score[mask]
    if y_true.size == 0:
        return 0.5
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = pd.Series(y_score).rank(method="average").to_numpy()
    sum_pos_ranks = float(ranks[pos].sum())
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(max(0.0, min(1.0, auc)))


def _average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(int)
    mask = np.isfinite(y_score) & np.isfinite(y_true)
    y_true = y_true[mask]
    y_score = y_score[mask]
    if y_true.size == 0:
        return 0.0
    order = np.argsort(-y_score)
    y = y_true[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    denom = tp + fp
    prec = np.where(denom > 0, tp / denom, 0.0)
    # AP = sum(precision@k for positive k) / n_pos
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        return 0.0
    ap = float(np.sum(prec[y == 1]) / n_pos)
    return float(max(0.0, min(1.0, ap)))


def _brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = y_true.astype(float)
    mask = np.isfinite(y_prob) & np.isfinite(y_true)
    y_true = y_true[mask]
    y_prob = y_prob[mask]
    if y_true.size == 0:
        return 0.0
    return float(np.mean((y_prob - y_true) ** 2))


def _calibration_bins(
    y_true: np.ndarray, y_prob: np.ndarray, *, n_bins: int = 12
) -> List[Dict[str, float]]:
    y_true = y_true.astype(float)
    y_prob = np.clip(y_prob.astype(float), 0.0, 1.0)
    mask = np.isfinite(y_prob) & np.isfinite(y_true)
    y_true = y_true[mask]
    y_prob = y_prob[mask]
    if y_true.size == 0:
        return []
    bins = []
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    for i in range(len(edges) - 1):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        m = (y_prob >= lo) & (y_prob < hi if i < len(edges) - 2 else y_prob <= hi)
        if not np.any(m):
            continue
        p_mean = float(np.mean(y_prob[m]))
        y_mean = float(np.mean(y_true[m]))
        bins.append(
            {
                "lo": lo,
                "hi": hi,
                "p_mean": p_mean,
                "y_mean": y_mean,
                "n": float(np.sum(m)),
            }
        )
    return bins


def _ece_from_bins(bins: List[Dict[str, float]]) -> float:
    if not bins:
        return 0.0
    n = float(sum(b.get("n", 0.0) for b in bins))
    if n <= 0:
        return 0.0
    e = 0.0
    for b in bins:
        w = float(b.get("n", 0.0)) / n
        e += w * abs(float(b.get("p_mean", 0.0)) - float(b.get("y_mean", 0.0)))
    return float(e)


class SurvivalMLP(nn.Module):
    def __init__(
        self, *, in_dim: int, hidden: int = 128, depth: int = 2, dropout: float = 0.1
    ):
        super().__init__()
        layers: List[nn.Module] = []
        d = int(in_dim)
        for _ in range(max(1, int(depth))):
            layers.append(nn.Linear(d, int(hidden)))
            layers.append(nn.ReLU())
            if float(dropout) > 0:
                layers.append(nn.Dropout(float(dropout)))
            d = int(hidden)
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass(frozen=True)
class SurvivalHeadTrainConfig:
    symbol_col: str = "symbol"
    timestamp_col: str = "timestamp"
    mode_col: str = "mode"
    label_col: str = "y_surv"

    feature_cols: Tuple[str, ...] = (
        "head_dir_score",
        "head_mfe_atr",
        "head_mae_atr",
        "head_t_to_mfe",
        "drawdown",
    )
    include_mode_onehot: bool = True

    train_ratio: float = 0.7
    val_ratio_within_train: float = 0.15

    seed: int = 0
    device: str = "cpu"
    epochs: int = 5
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 0.0
    hidden: int = 128
    depth: int = 2
    dropout: float = 0.1

    pos_weight: float = 1.0
    n_calibration_bins: int = 12


def _time_split_indices(
    n: int, train_ratio: float, val_ratio_within_train: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(n)
    n_train = max(1, int(n * float(train_ratio)))
    n_train = min(n_train, n)
    n_val = (
        int(max(1, int(n_train * float(val_ratio_within_train)))) if n_train >= 3 else 0
    )
    n_tr = max(1, n_train - n_val) if n_train >= 2 else n_train
    idx = np.arange(n, dtype=int)
    tr = idx[:n_tr]
    va = idx[n_tr:n_train] if n_val > 0 else np.asarray([], dtype=int)
    te = idx[n_train:]
    return tr, va, te


def _build_X(
    df: pd.DataFrame,
    *,
    cfg: SurvivalHeadTrainConfig,
) -> Tuple[np.ndarray, List[str]]:
    cols = list(cfg.feature_cols)
    X = []
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"Missing feature col: {c}")
        X.append(
            pd.to_numeric(df[c], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        )
    feat_names = cols[:]

    if cfg.include_mode_onehot:
        m = (
            df[cfg.mode_col].astype(str).str.upper()
            if cfg.mode_col in df.columns
            else pd.Series(["NO_TRADE"] * len(df))
        )
        for name in ["NO_TRADE", "MEAN", "TREND"]:
            X.append((m == name).to_numpy(dtype=float))
            feat_names.append(f"mode_is_{name}")

    X2 = np.stack(X, axis=1) if X else np.zeros((len(df), 0), dtype=float)
    return X2, feat_names


def train_survival_head(
    df_logs: pd.DataFrame,
    df_labels: pd.DataFrame,
    *,
    cfg: SurvivalHeadTrainConfig,
) -> Tuple[Dict[str, Any], pd.DataFrame, Dict[str, Any], bytes, bytes, bytes]:
    """
    Returns:
      - metrics dict
      - preds_df: symbol,timestamp,split,survival_prob,y_true
      - curves dict
      - roc_png/pr_png/cal_png bytes (for report embedding)
    """
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))

    df = df_logs.copy()
    lab = df_labels.copy()

    for c in [cfg.symbol_col, cfg.timestamp_col]:
        if c not in df.columns:
            raise ValueError(f"Missing logs col: {c}")
        if c not in lab.columns:
            raise ValueError(f"Missing labels col: {c}")
    if cfg.label_col not in lab.columns:
        raise ValueError(f"Missing label col: {cfg.label_col}")

    df[cfg.timestamp_col] = pd.to_datetime(
        df[cfg.timestamp_col], utc=True, errors="coerce"
    )
    lab[cfg.timestamp_col] = pd.to_datetime(
        lab[cfg.timestamp_col], utc=True, errors="coerce"
    )

    m = df.merge(
        lab[[cfg.symbol_col, cfg.timestamp_col, cfg.label_col]],
        on=[cfg.symbol_col, cfg.timestamp_col],
        how="inner",
    )
    m = m.sort_values([cfg.symbol_col, cfg.timestamp_col]).reset_index(drop=True)
    if len(m) < 50:
        raise ValueError(f"Not enough samples for survival head training: {len(m)}")

    X_all, feat_names = _build_X(m, cfg=cfg)
    y_all = (
        pd.to_numeric(m[cfg.label_col], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    y_all = (y_all > 0.5).astype(np.float32)

    # Build split masks per symbol (time-ordered)
    split = np.array(["test"] * len(m), dtype=object)
    tr_idx_all = []
    va_idx_all = []
    te_idx_all = []
    for sym, g in m.groupby(cfg.symbol_col, sort=False):
        idx = g.index.to_numpy(dtype=int)
        tr, va, te = _time_split_indices(
            len(idx), cfg.train_ratio, cfg.val_ratio_within_train
        )
        tr_idx = idx[tr]
        va_idx = idx[va] if len(va) else np.asarray([], dtype=int)
        te_idx = idx[te] if len(te) else np.asarray([], dtype=int)
        split[tr_idx] = "train"
        split[va_idx] = "val"
        split[te_idx] = "test"
        tr_idx_all.append(tr_idx)
        va_idx_all.append(va_idx)
        te_idx_all.append(te_idx)

    tr_idx = np.concatenate(tr_idx_all) if tr_idx_all else np.asarray([], dtype=int)
    va_idx = np.concatenate(va_idx_all) if va_idx_all else np.asarray([], dtype=int)
    te_idx = np.concatenate(te_idx_all) if te_idx_all else np.asarray([], dtype=int)

    device = torch.device(str(cfg.device))
    model = SurvivalMLP(
        in_dim=int(X_all.shape[1]),
        hidden=int(cfg.hidden),
        depth=int(cfg.depth),
        dropout=float(cfg.dropout),
    ).to(device)

    # Weighted BCE
    pos_weight = torch.tensor([float(cfg.pos_weight)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(
        model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay)
    )

    def _batch_iter(idxs: np.ndarray, batch_size: int):
        idxs = np.asarray(idxs, dtype=int)
        if idxs.size == 0:
            return
        order = np.random.permutation(idxs)
        for i in range(0, len(order), int(batch_size)):
            yield order[i : i + int(batch_size)]

    X_t = torch.tensor(X_all, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_all, dtype=torch.float32, device=device)

    best_val = None
    best_state = None
    for _epoch in range(int(cfg.epochs)):
        model.train()
        for b in _batch_iter(tr_idx, cfg.batch_size):
            logits = model(X_t[b])
            loss = loss_fn(logits, y_t[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if va_idx.size:
            model.eval()
            with torch.no_grad():
                v = loss_fn(model(X_t[va_idx]), y_t[va_idx]).item()
            if best_val is None or v < best_val:
                best_val = float(v)
                best_state = {
                    k: v.detach().cpu() for k, v in model.state_dict().items()
                }

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    model.eval()
    with torch.no_grad():
        logits = model(X_t).detach().cpu().numpy().astype(float)
    prob = 1.0 / (1.0 + np.exp(-logits))

    # Metrics on test split
    y_te = y_all[te_idx] if te_idx.size else y_all
    p_te = prob[te_idx] if te_idx.size else prob
    auc = _binary_auc(y_te, p_te)
    ap = _average_precision(y_te, p_te)
    br = _brier(y_te, p_te)
    bins = _calibration_bins(y_te, p_te, n_bins=int(cfg.n_calibration_bins))
    ece = _ece_from_bins(bins)

    curves = {
        "calibration_bins": bins,
    }
    metrics = {
        "n": int(len(m)),
        "n_train": int(tr_idx.size),
        "n_val": int(va_idx.size),
        "n_test": int(te_idx.size),
        "pos_rate_test": float(np.mean(y_te)) if y_te.size else 0.0,
        "auc_test": float(auc),
        "ap_test": float(ap),
        "brier_test": float(br),
        "ece_test": float(ece),
        "feature_cols": feat_names,
    }

    preds_df = pd.DataFrame(
        {
            cfg.symbol_col: m[cfg.symbol_col].values,
            cfg.timestamp_col: m[cfg.timestamp_col].values,
            "split": split,
            "survival_prob": prob.astype(float),
            "y_true": y_all.astype(int),
        }
    )

    # Build simple matplotlib plots (optional dependency)
    roc_png = b""
    pr_png = b""
    cal_png = b""
    try:
        import matplotlib.pyplot as plt  # type: ignore

        # ROC curve
        order = np.argsort(-p_te)
        y_sorted = y_te[order]
        tp = np.cumsum(y_sorted == 1)
        fp = np.cumsum(y_sorted == 0)
        n_pos = max(1, int((y_te == 1).sum()))
        n_neg = max(1, int((y_te == 0).sum()))
        tpr = tp / n_pos
        fpr = fp / n_neg
        fig = plt.figure(figsize=(4.2, 3.2))
        plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
        plt.plot([0, 1], [0, 1], "--", color="#94a3b8")
        plt.title("Survival Head ROC (test)")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend(loc="lower right", fontsize=8)
        buf = BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", dpi=140)
        plt.close(fig)
        roc_png = buf.getvalue()

        # PR curve
        order = np.argsort(-p_te)
        y_sorted = y_te[order]
        tp = np.cumsum(y_sorted == 1)
        fp = np.cumsum(y_sorted == 0)
        denom = tp + fp
        prec = np.where(denom > 0, tp / denom, 0.0)
        rec = tp / max(1, int((y_te == 1).sum()))
        fig = plt.figure(figsize=(4.2, 3.2))
        plt.plot(rec, prec, label=f"AP={ap:.3f}")
        plt.title("Survival Head PR (test)")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(loc="lower left", fontsize=8)
        buf = BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", dpi=140)
        plt.close(fig)
        pr_png = buf.getvalue()

        # Calibration
        fig = plt.figure(figsize=(4.2, 3.2))
        xs = [b["p_mean"] for b in bins]
        ys = [b["y_mean"] for b in bins]
        plt.plot(xs, ys, marker="o")
        plt.plot([0, 1], [0, 1], "--", color="#94a3b8")
        plt.title(f"Calibration (ECE={ece:.3f})")
        plt.xlabel("Predicted prob")
        plt.ylabel("Empirical freq")
        buf = BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", dpi=140)
        plt.close(fig)
        cal_png = buf.getvalue()
    except Exception:
        pass

    # Attach minimal model payload for saving
    payload = {
        "state_dict": model.state_dict(),
        "meta": {
            "feature_cols": feat_names,
            "include_mode_onehot": bool(cfg.include_mode_onehot),
            "cfg": {
                "train_ratio": float(cfg.train_ratio),
                "val_ratio_within_train": float(cfg.val_ratio_within_train),
                "hidden": int(cfg.hidden),
                "depth": int(cfg.depth),
                "dropout": float(cfg.dropout),
            },
        },
    }
    metrics["_model_payload"] = payload  # internal use by save helper
    return metrics, preds_df, curves, roc_png, pr_png, cal_png


def save_survival_head_artifacts(
    *,
    out_dir: str | Path,
    metrics: Dict[str, Any],
    preds_df: pd.DataFrame,
    curves: Dict[str, Any],
    roc_png: bytes,
    pr_png: bytes,
    cal_png: bytes,
) -> None:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)

    payload = metrics.pop("_model_payload", None)
    if isinstance(payload, dict):
        torch.save(payload, p / "model.pt")

    (p / "metrics.json").write_text(
        json.dumps(
            {k: v for k, v in metrics.items() if not k.startswith("_")},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (p / "curves.json").write_text(
        json.dumps(curves, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    preds_df.to_parquet(p / "survival_preds.parquet", index=False)

    # Save images + HTML (embed as base64 to keep report portable)
    def _img(b: bytes) -> str:
        if not b:
            return ""
        return f"<img style='max-width:100%;border:1px solid #e5e7eb;border-radius:12px' src='data:image/png;base64,{base64.b64encode(b).decode('ascii')}'/>"

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Survival Head Report</title>
<style>
body{{font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial;margin:24px;color:#0f172a;background:#fafafa}}
code{{background:#f1f5f9;padding:2px 6px;border-radius:8px}}
table{{border-collapse:collapse;width:100%;font-size:12px;background:#fff;border-radius:12px;overflow:hidden}}
th,td{{border-bottom:1px solid #eef2f7;text-align:left;padding:8px 10px;vertical-align:top}}
th{{background:#f8fafc;font-weight:600}}
.card{{border:1px solid #e5e7eb;border-radius:16px;padding:16px 18px;background:#fff;box-shadow:0 1px 2px rgba(0,0,0,0.04);margin-top:16px}}
.muted{{color:#64748b}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
</style></head>
<body>
<div class="card">
  <h2 style="margin:0 0 10px;">Survival Head (MLP) — evaluation</h2>
  <p class="muted" style="margin-top:0;">只评估“保命信号”的可用性：AUC/AP/Calibration + 曲线。它不负责赚钱，不直接决定方向。</p>
  <table><thead><tr><th>metric</th><th>value</th></tr></thead><tbody>
    <tr><td><code>auc_test</code></td><td>{metrics.get('auc_test')}</td></tr>
    <tr><td><code>ap_test</code></td><td>{metrics.get('ap_test')}</td></tr>
    <tr><td><code>brier_test</code></td><td>{metrics.get('brier_test')}</td></tr>
    <tr><td><code>ece_test</code></td><td>{metrics.get('ece_test')}</td></tr>
    <tr><td><code>pos_rate_test</code></td><td>{metrics.get('pos_rate_test')}</td></tr>
    <tr><td><code>n_test</code></td><td>{metrics.get('n_test')}</td></tr>
  </tbody></table>
</div>
<div class="grid">
  <div class="card">{_img(roc_png)}</div>
  <div class="card">{_img(pr_png)}</div>
</div>
<div class="card">{_img(cal_png)}</div>
</body></html>
"""
    (p / "report.html").write_text(html, encoding="utf-8")
