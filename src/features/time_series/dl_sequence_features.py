"""Deep Learning Sequence Feature Extraction with Mamba/Transformer.

支持 FP16 + FlashAttention / Mamba 提取序列特征
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, List, Literal
import warnings

from src.features.registry import register_feature

warnings.filterwarnings("ignore")

# Check for deep learning dependencies
DL_BACKEND = None
try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
    print("✓ PyTorch available")

    # Try Mamba (preferred for efficiency)
    try:
        from mamba_ssm import Mamba

        DL_BACKEND = "mamba"
        print("✓ Mamba available (O(n) complexity)")
    except ImportError:
        print("⚠️  Mamba not available, will use Transformer")

        # Try FlashAttention (preferred for Transformer)
        try:
            from flash_attn import flash_attn_qkvpacked_func

            DL_BACKEND = "flash_attention"
            FLASH_ATTN_AVAILABLE = True
            print("✓ FlashAttention available (2-4x speedup)")
        except ImportError:
            print("⚠️  FlashAttention not available, using standard Transformer")
            DL_BACKEND = "transformer"
            FLASH_ATTN_AVAILABLE = False
            flash_attn_qkvpacked_func = None  # Define as None for safe checking

except ImportError:
    TORCH_AVAILABLE = False
    # Provide a dummy 'nn' module so class definitions (nn.Module) don't crash
    import types
    nn = types.SimpleNamespace(Module=object, Linear=None, LayerNorm=None,
                               Parameter=None, Embedding=None,
                               TransformerEncoderLayer=None,
                               TransformerEncoder=None, Sequential=None,
                               GELU=None, Dropout=None)
    torch = types.SimpleNamespace(randn=None, arange=None, matmul=None,
                                  softmax=None)
    print("❌ PyTorch not installed")
    print("   Install with: pip install torch")


class MambaSequenceEncoder(nn.Module):  # type: ignore[misc]
    """Mamba-based sequence encoder (O(n) complexity, memory efficient)."""

    def __init__(self, input_dim=5, d_model=64, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Mamba blocks
        self.mamba1 = Mamba(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
        )
        self.mamba2 = Mamba(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
        )

        # Layer norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.feature_dim = d_model

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            features: (batch, d_model)
        """
        # Input projection
        x = self.input_proj(x)  # (B, L, d_model)

        # Mamba block 1
        x = x + self.mamba1(self.norm1(x))

        # Mamba block 2
        x = x + self.mamba2(self.norm2(x))

        # Global average pooling
        features = x.mean(dim=1)  # (B, d_model)

        return features


class FlashAttentionEncoder(nn.Module):  # type: ignore[misc]
    """Flash Attention-based Transformer encoder (2-4x speedup)."""

    def __init__(self, input_dim=5, d_model=64, nhead=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding (learned)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, d_model) * 0.01)

        # QKV projections
        self.qkv_proj = nn.Linear(d_model, d_model * 3)

        # Layer norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Feed forward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

        self.num_layers = num_layers
        self.feature_dim = d_model

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            features: (batch, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Input projection
        x = self.input_proj(x)  # (B, L, d_model)

        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]

        # Multiple transformer layers
        for _ in range(self.num_layers):
            # Self-attention with Flash Attention
            residual = x
            x = self.norm1(x)

            # QKV projection
            qkv = self.qkv_proj(x)  # (B, L, 3*d_model)
            qkv = qkv.reshape(
                batch_size, seq_len, 3, self.nhead, self.d_model // self.nhead
            )
            qkv = qkv.permute(0, 2, 3, 1, 4)  # (B, 3, nhead, L, head_dim)

            # Flash attention expects (B, L, 3, nhead, head_dim)
            qkv = qkv.permute(0, 3, 1, 2, 4)  # (B, L, 3, nhead, head_dim)

            # Use flash attention if available, otherwise use standard attention
            use_flash = False
            if FLASH_ATTN_AVAILABLE and flash_attn_qkvpacked_func is not None:
                try:
                    attn_out = flash_attn_qkvpacked_func(qkv.half())  # FP16
                    attn_out = attn_out.float()  # Back to FP32
                    use_flash = True
                except Exception:
                    pass

            if not use_flash:
                # Fallback to standard attention
                q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
                q = q.reshape(batch_size, seq_len, self.d_model)
                k = k.reshape(batch_size, seq_len, self.d_model)
                v = v.reshape(batch_size, seq_len, self.d_model)

                attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (
                    self.d_model**0.5
                )
                attn_weights = torch.softmax(attn_weights, dim=-1)
                attn_out = torch.matmul(attn_weights, v)

            x = residual + attn_out

            # Feed forward
            residual = x
            x = self.norm2(x)
            x = residual + self.ffn(x)

        # Global average pooling
        features = x.mean(dim=1)  # (B, d_model)

        return features


class StandardTransformerEncoder(nn.Module):  # type: ignore[misc]
    """Standard Transformer encoder (fallback)."""

    def __init__(self, input_dim=5, d_model=64, nhead=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = nn.Embedding(1000, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.feature_dim = d_model

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            features: (batch, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Input projection
        x = self.input_proj(x)

        # Add positional encoding
        positions = (
            torch.arange(seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)
        )
        x = x + self.pos_encoder(positions)

        # Transformer
        x = self.transformer(x)

        # Global average pooling
        features = x.mean(dim=1)

        return features



class DeepLearningSequenceExtractor:
    """Extract sequence features using deep learning models."""

    def __init__(
        self,
        backend: Literal["mamba", "flash_attention", "transformer", "auto"] = "auto",
        seq_length: int = 120,
        d_model: int = 64,
        use_fp16: bool = False,  # 默认关闭 FP16 提升稳定性
        device: Optional[str] = None,
        normalization_method: Literal["ema"] = "ema",  # 仅支持 EMA（因果安全）
        output_normalization: Literal["tanh", "none"] = "tanh",
        seed: int = 42,
    ):
        """
        Args:
            backend: 'mamba', 'flash_attention', 'transformer', or 'auto'
            seq_length: Sequence length (default: 120 bars = 10 hours for 5min)
            d_model: Model dimension (default: 64)
            use_fp16: Use FP16 mixed precision (default: True)
            device: 'cuda', 'cpu', or None (auto-detect)
            normalization_method: Normalization strategy ('global', 'rolling', 'ema', 'adaptive')
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required. Install with: pip install torch")

        self.seq_length = seq_length
        self.d_model = d_model
        self.use_fp16 = use_fp16
        self.normalization_method = normalization_method
        self.output_normalization = output_normalization
        self.seed = seed

        # Auto-detect device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            # If explicitly specified, try to use it, but fallback to CPU if CUDA not available
            if device == "cuda" and not torch.cuda.is_available():
                print(f"   ⚠️  CUDA requested but not available, falling back to CPU")
                self.device = torch.device("cpu")
            else:
                self.device = torch.device(device)

        # Select backend
        if backend == "auto":
            backend = DL_BACKEND or "transformer"

        self.backend = backend
        self.model = None
        self.is_fitted = False

        # Normalization parameters
        self.scaler_mean = None
        self.scaler_std = None
        self.ema_mean = None
        self.ema_var = None
        self.alpha = 0.01  # EMA smoothing parameter

        print(f"\n🔷 DeepLearningSequenceExtractor (LEAK-FREE MODE)")
        print(f"   Backend: {self.backend}")
        print(f"   Device: {self.device} (CUDA available: {torch.cuda.is_available()})")
        print(f"   Sequence length: {seq_length}")
        print(f"   Output dimension: {d_model}")
        print(f"   FP16: {use_fp16}")
        print(f"   Normalization: causal EMA (α={self.alpha})")
        print(f"   Output normalization: {self.output_normalization}")

    def _create_model(self, input_dim: int):
        """Create the appropriate model based on backend."""
        # Make model init deterministic across calls so repeated feature computation
        # (train/test splits, research/backtest/live) doesn't drift due to random weights.
        rng_state = torch.random.get_rng_state()
        try:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

            if self.backend == "mamba":
                model = MambaSequenceEncoder(
                    input_dim=input_dim,
                    d_model=self.d_model,
                    d_state=16,
                    d_conv=4,
                    expand=2,
                )
            elif self.backend == "flash_attention":
                model = FlashAttentionEncoder(
                    input_dim=input_dim,
                    d_model=self.d_model,
                    nhead=8,
                    num_layers=2,
                    dropout=0.1,
                )
            else:  # transformer
                model = StandardTransformerEncoder(
                    input_dim=input_dim,
                    d_model=self.d_model,
                    nhead=8,
                    num_layers=2,
                    dropout=0.1,
                )
        finally:
            torch.random.set_rng_state(rng_state)

        model = model.to(self.device)
        model.eval()  # Inference mode

        # Use FP16 if requested and on GPU
        if self.use_fp16 and self.device.type == "cuda":
            model = model.half()
            print(f"   ✓ Model converted to FP16")

        return model

    def _prepare_sequences(self, df: pd.DataFrame, columns: List[str]) -> np.ndarray:
        """
        Prepare sliding window sequences with STRICTLY CAUSAL EMA normalization.

        【关键修复】：使用严格因果的 EMA 归一化，杜绝数据泄漏。
        - 不使用全局统计量
        - EMA 每次 transform 调用时从头开始
        - 确保每个时间点 t 的特征只基于 [0, t] 的数据
        """
        data = df[columns].values.astype(np.float32)
        n = len(data)

        if n < self.seq_length:
            raise ValueError(f"Data too short ({n}) for seq_length={self.seq_length}")

        # 初始化 EMA（仅用前 seq_length 个点，确保因果性）
        init_data = data[: self.seq_length]
        ema_mean = np.mean(init_data, axis=0, keepdims=True)
        ema_var = np.var(init_data, axis=0, keepdims=True)

        normalized_data = np.zeros_like(data)
        sequences = []

        # 逐点处理，确保严格因果性
        for t in range(n):
            x = data[t : t + 1]  # 当前时刻数据 shape: (1, d)

            # 更新 EMA（仅使用历史信息，完全因果）
            ema_mean = self.alpha * x + (1 - self.alpha) * ema_mean
            ema_var = self.alpha * (x - ema_mean) ** 2 + (1 - self.alpha) * ema_var
            ema_std = np.sqrt(ema_var) + 1e-8

            # 归一化当前点
            normalized_data[t] = (x - ema_mean) / ema_std

            # 当有足够的历史时，构建窗口 [t-seq_len+1, t]
            if t >= self.seq_length - 1:
                start_idx = t - self.seq_length + 1
                seq = normalized_data[start_idx : t + 1]  # shape: (seq_len, d)
                sequences.append(seq)

        return np.array(sequences, dtype=np.float32)

    def fit(self, df: pd.DataFrame, feature_columns: Optional[List[str]] = None):
        """
        Initialize model. Does NOT access data to prevent leakage.

        【关键修复】：fit() 不再接触数据，只初始化模型结构。
        所有归一化统计量在 transform() 时从头计算，确保因果性。
        """
        if feature_columns is None:
            feature_columns = ["open", "high", "low", "close", "volume"]

        missing = [col for col in feature_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        self.feature_columns = feature_columns
        input_dim = len(feature_columns)

        # Create model (only structure, no data access)
        self.model = self._create_model(input_dim)

        # Reset normalization state (will be computed fresh in each transform)
        self.scaler_mean = None
        self.scaler_std = None
        self.ema_mean = None
        self.ema_var = None

        self.is_fitted = True
        print(
            f"   ✓ Model initialized (leak-free): {input_dim} input -> {self.d_model} output"
        )

        return self

    def transform(self, df: pd.DataFrame, batch_size: int = 64) -> np.ndarray:
        """
        Extract sequence features with NO DATA LEAKAGE.

        【关键修复】：每次 transform 调用时，EMA 从头开始计算，确保完全因果。
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before transform()")

        # Prepare sequences (EMA reset internally per call, ensuring causality)
        sequences = self._prepare_sequences(df, self.feature_columns)
        num_samples = len(sequences)

        if num_samples == 0:
            raise ValueError("No valid sequences generated")

        # Extract features in batches
        all_features = []

        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                batch = sequences[i : i + batch_size]

                # Convert to tensor
                if self.use_fp16 and self.device.type == "cuda":
                    batch_tensor = torch.from_numpy(batch).half().to(self.device)
                else:
                    batch_tensor = torch.from_numpy(batch).to(self.device)

                # Extract features
                features = self.model(batch_tensor)

                # Convert back to numpy
                if self.use_fp16:
                    features = features.float()
                all_features.append(features.cpu().numpy())

        # Concatenate all batches
        dl_features = np.vstack(all_features)

        # Normalize outputs to a stable, unitless range for cross-asset comparability.
        # This is per-sample (no future leakage).
        if self.output_normalization == "tanh":
            dl_features = np.tanh(dl_features)

        print(f"   ✓ Extracted {len(dl_features)} leak-free sequence features")

        return dl_features

    def fit_transform(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        batch_size: int = 64,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit and transform in one step."""
        self.fit(df, feature_columns)
        features = self.transform(df, batch_size)

        # Valid indices
        valid_indices = np.arange(self.seq_length - 1, len(df))

        return features, valid_indices

    def add_to_dataframe(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        batch_size: int = 64,
        prefix: str = "dl_seq",
    ) -> pd.DataFrame:
        """Add deep learning sequence features to dataframe."""
        try:
            # Extract features
            if not self.is_fitted:
                features, valid_indices = self.fit_transform(
                    df, feature_columns, batch_size
                )
            else:
                features = self.transform(df, batch_size)
                valid_indices = np.arange(self.seq_length - 1, len(df))

            # Create feature names
            feature_names = [f"{prefix}_f{i}" for i in range(self.d_model)]

            # Create DataFrame
            features_df = pd.DataFrame(
                features, columns=feature_names, index=df.index[valid_indices]
            )

            # Merge
            df_with_features = df.join(features_df, how="left")

            print(f"   ✓ Added {len(feature_names)} DL sequence features")
            print(f"   ✓ Valid samples: {len(valid_indices)} / {len(df)}")

            return df_with_features

        except Exception as e:
            print(f"   ⚠️  DL feature extraction failed: {e}")
            print(f"   ⚠️  Returning original dataframe")
            return df


def add_dl_sequence_features(
    df: pd.DataFrame,
    backend: str = "auto",
    seq_length: int = 120,
    d_model: int = 64,
    feature_columns: Optional[List[str]] = None,
    use_fp16: bool = False,  # 默认关闭 FP16 提升稳定性
    normalization_method: str = "ema",  # 强制使用 EMA（因果安全）
    output_normalization: str = "tanh",
    device: Optional[str] = None,  # 'cuda', 'cpu', or None (auto-detect)
    seed: int = 42,
) -> pd.DataFrame:
    """Convenience function to add DL sequence features.

    Args:
        df: DataFrame with OHLCV data
        backend: 'mamba', 'flash_attention', 'transformer', or 'auto'
        seq_length: Sequence length (default: 120 = 10 hours for 5min bars)
        d_model: Output dimension (default: 64)
        feature_columns: Input columns (default: OHLCV)
        use_fp16: Use FP16 mixed precision (default: True)
        normalization_method: Normalization strategy ('global', 'rolling', 'ema', 'adaptive')

    Returns:
        DataFrame with DL sequence features added
    """
    if not TORCH_AVAILABLE:
        print("⚠️  PyTorch not available, skipping DL features")
        return df

    print(f"\n🔷 Extracting Leak-Free Deep Learning Sequence Features...")

    if feature_columns is None:
        feature_columns = ["open", "high", "low", "close", "volume"]

    # Cache extractor per config to avoid repeated model re-inits and keep outputs consistent.
    global _DL_EXTRACTOR_CACHE
    try:
        _DL_EXTRACTOR_CACHE
    except NameError:
        _DL_EXTRACTOR_CACHE = {}

    cache_key = (
        backend,
        seq_length,
        d_model,
        use_fp16,
        str(device) if device is not None else None,
        tuple(feature_columns),
        normalization_method,
        output_normalization,
        seed,
    )

    extractor = _DL_EXTRACTOR_CACHE.get(cache_key)
    if extractor is None:
        extractor = DeepLearningSequenceExtractor(
            backend=backend,
            seq_length=seq_length,
            d_model=d_model,
            use_fp16=use_fp16,
            normalization_method=normalization_method,
            output_normalization=output_normalization,  # stable/bounded embeddings
            device=device,
            seed=seed,
        )
        _DL_EXTRACTOR_CACHE[cache_key] = extractor

    df_with_features = extractor.add_to_dataframe(df, feature_columns)

    return df_with_features


# For backward compatibility
add_transformer_features = add_dl_sequence_features


# =============================================================================
# Registered feature functions for config-driven pipeline
# =============================================================================


@register_feature("compute_dl_sequence_features", category="dl")
def compute_dl_sequence_features(
    df: pd.DataFrame,
    backend: str = "auto",
    seq_length: int = 120,
    d_model: int = 64,
    feature_columns: Optional[List[str]] = None,
    use_fp16: bool = False,
    prefix: str = "dl_seq",
    device: Optional[str] = None,
    output_normalization: str = "tanh",
) -> pd.DataFrame:
    """
    Compute leak-free deep learning sequence embeddings (Mamba/Transformer).

    Args:
        df: Input dataframe.
        backend: 'mamba', 'flash_attention', 'transformer', or 'auto'.
        seq_length: Sliding window length.
        d_model: Output embedding dimension.
        feature_columns: Columns used as model inputs (defaults to OHLCV).
        use_fp16: Whether to enable FP16 during inference (GPU only).
        prefix: Feature column prefix (default: dl_seq).
        device: 'cuda', 'cpu', or None (auto-detect)

    Returns:
        DataFrame with DL sequence features appended.
    """
    result = add_dl_sequence_features(
        df.copy(),
        backend=backend,
        seq_length=seq_length,
        d_model=d_model,
        feature_columns=feature_columns,
        use_fp16=use_fp16,
        output_normalization=output_normalization,
        device=device,
    )

    # Ensure expected columns exist even if DL backend is unavailable
    expected_cols = [f"{prefix}_f{i}" for i in range(d_model)]
    for col in expected_cols:
        if col not in result.columns:
            result[col] = 0.0

    return result


@register_feature("compute_dl_sequence_features_from_series", category="dl")
def compute_dl_sequence_features_from_series(
    *,
    open: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    backend: str = "auto",
    seq_length: int = 120,
    d_model: int = 64,
    feature_columns: Optional[List[str]] = None,
    use_fp16: bool = False,
    prefix: str = "dl_seq",
    device: Optional[str] = None,
    output_normalization: str = "tanh",
) -> pd.DataFrame:
    """
    Narrow-IO entrypoint for DL sequence embeddings (Series-in, DataFrame-out).
    Keeps pipeline from passing a wide DF into heavy DL code.
    """
    df = pd.DataFrame(
        {"open": open, "high": high, "low": low, "close": close, "volume": volume}
    )
    return compute_dl_sequence_features(
        df,
        backend=backend,
        seq_length=seq_length,
        d_model=d_model,
        feature_columns=feature_columns,
        use_fp16=use_fp16,
        prefix=prefix,
        device=device,
        output_normalization=output_normalization,
    )
