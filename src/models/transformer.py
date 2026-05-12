"""
transformer.py — Temporal Fusion Transformer for pump-and-dump detection.

Architecture highlights:
  - Variable Selection Networks (VSN) per modality — learned per-feature importance
  - Gated Residual Networks (GRN) with proper GLU gating
  - Bidirectional cross-modal attention (social ↔ market)
  - Fused encoder with progressive head compression + LayerNorm
  - SDPA (flash-attention) via scaled_dot_product_attention when available (PyTorch ≥ 2.0)
  - Output keys: 'manipulation_probability' (canonical) + 'manipulation_prob' (alias, compat)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ── Flash / SDPA attention ─────────────────────────────────────────────────────
# PyTorch ≥ 2.0 exposes F.scaled_dot_product_attention which uses flash-attention
# kernels automatically on CUDA and falls back to an optimised math impl on CPU.
_SDPA_AVAILABLE = hasattr(F, 'scaled_dot_product_attention')


# ── Positional Encoding ────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1), :])


# ── GLU & GRN ─────────────────────────────────────────────────────────────────

class GLU(nn.Module):
    """Gated Linear Unit — dynamically suppresses irrelevant features."""
    def __init__(self, input_size: int):
        super().__init__()
        self.fc_value = nn.Linear(input_size, input_size)
        self.fc_gate  = nn.Linear(input_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_value(x) * torch.sigmoid(self.fc_gate(x))


class GatedResidualNetwork(nn.Module):
    """
    GRN block from the TFT paper.
    ELU non-linearity → dropout → GLU gating → LayerNorm residual.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 dropout: float = 0.1):
        super().__init__()
        self.fc1     = nn.Linear(input_size, hidden_size)
        self.elu     = nn.ELU()
        self.fc2     = nn.Linear(hidden_size, output_size)
        self.glu     = GLU(output_size)
        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(output_size)
        self.skip    = (
            nn.Linear(input_size, output_size)
            if input_size != output_size else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        h = self.elu(self.fc1(x))
        h = self.dropout(self.fc2(h))
        h = self.glu(h)
        return self.norm(residual + h)


# ── Variable Selection Network ─────────────────────────────────────────────────

class VariableSelectionNetwork(nn.Module):
    """
    Learned per-feature soft-selection gate.
    Each input feature gets its own GRN; a shared softmax weight
    controls how much each feature contributes to the combined embedding.

    Input:  (B, T, n_features)
    Output: (B, T, d_model)
    """
    def __init__(self, n_features: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        # One lightweight GRN per feature (projects raw scalar → d_model)
        self.feature_grns = nn.ModuleList([
            GatedResidualNetwork(1, d_model, d_model, dropout)
            for _ in range(n_features)
        ])
        # Shared selection network produces softmax weights over all features
        self.selection_grn = GatedResidualNetwork(n_features, d_model, n_features, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, n_features)
        B, T, C = x.shape

        # Per-feature embeddings: list of (B*T, d_model)
        per_feature = [
            self.feature_grns[i](x[..., i:i+1].reshape(B * T, 1))
            for i in range(C)
        ]  # each: (B*T, d_model)
        per_feature = torch.stack(per_feature, dim=1)  # (B*T, C, d_model)

        # Compute selection weights
        flat_x = x.reshape(B * T, C)                          # (B*T, C)
        weights = torch.softmax(
            self.selection_grn(flat_x), dim=-1
        ).unsqueeze(-1)                                        # (B*T, C, 1)

        # Weighted sum over features → combined embedding
        combined = (per_feature * weights).sum(dim=1)          # (B*T, d_model)
        return combined.reshape(B, T, -1)                      # (B, T, d_model)


# ── Cross-Modal Attention ──────────────────────────────────────────────────────

class MultiModalAttention(nn.Module):
    """
    Cross-modal multi-head attention with residual + LayerNorm.
    Uses F.scaled_dot_product_attention (flash-attention) when available.
    """
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        self.nhead    = nhead
        self.d_model  = d_model
        self.head_dim = d_model // nhead
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = dropout
        self.norm = nn.LayerNorm(d_model)

    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.nhead, self.head_dim).transpose(1, 2)

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor) -> tuple[torch.Tensor, None]:
        B, T, _ = query.shape
        q = self._reshape(self.q_proj(query))
        k = self._reshape(self.k_proj(key))
        v = self._reshape(self.v_proj(value))

        if _SDPA_AVAILABLE:
            drop = self.attn_drop if self.training else 0.0
            attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop)
        else:
            scale = self.head_dim ** -0.5
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            if self.training and self.attn_drop > 0:
                scores = F.dropout(scores, p=self.attn_drop)
            attn_out = torch.matmul(F.softmax(scores, dim=-1), v)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        attn_out = self.out_proj(attn_out)
        return self.norm(query + attn_out), None  # None = no explicit weights returned


# ── Main Model ─────────────────────────────────────────────────────────────────

class TemporalFusionTransformer(nn.Module):
    """
    Multi-modal TFT for stock manipulation detection.

    Inputs:
        social_features  (B, T, social_dim)
        market_features  (B, T, market_dim)

    Outputs (dict):
        manipulation_probability  (B, 1)  — primary detection head
        manipulation_prob         (B, 1)  — alias (backwards-compat)
        lead_lag_score            (B, 1)
        volume_anomaly            (B, 1)
        sentiment_surge           (B, 1)
    """

    def __init__(self,
                 social_features: int = 6,
                 market_features: int = 13,
                 d_model: int = 128,
                 nhead: int = 8,
                 num_layers: int = 4,
                 dropout: float = 0.1):
        super().__init__()

        # ── Per-modality Variable Selection Networks ───────────────────────────
        self.social_vsn  = VariableSelectionNetwork(social_features, d_model, dropout)
        self.market_vsn  = VariableSelectionNetwork(market_features, d_model, dropout)

        # ── Positional encoding (shared) ───────────────────────────────────────
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout * 0.5)

        # ── Per-modality temporal encoders ─────────────────────────────────────
        enc_layer_cfg = dict(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,          # Pre-LN: more stable training
        )
        self.social_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(**enc_layer_cfg),
            num_layers=max(1, num_layers // 2),
            enable_nested_tensor=False,
        )
        self.market_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(**enc_layer_cfg),
            num_layers=max(1, num_layers // 2),
            enable_nested_tensor=False,
        )

        # ── Bidirectional cross-modal attention ────────────────────────────────
        self.cross_attn_sm = MultiModalAttention(d_model, nhead, dropout)  # social queries market
        self.cross_attn_ms = MultiModalAttention(d_model, nhead, dropout)  # market queries social

        # ── Fusion encoder (operates on concatenated [social ‖ market]) ─────────
        fused_dim = d_model * 2
        fused_layer_cfg = dict(
            d_model=fused_dim, nhead=nhead,
            dim_feedforward=fused_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.fusion_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(**fused_layer_cfg),
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        # ── Output heads ──────────────────────────────────────────────────────
        half   = fused_dim // 2
        quarter = fused_dim // 4

        # Primary detection head (progressive compression + LayerNorm)
        self.detection_head = nn.Sequential(
            nn.Linear(fused_dim, half),
            nn.LayerNorm(half),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(half, quarter),
            nn.LayerNorm(quarter),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(quarter, 1),
            nn.Sigmoid(),
        )

        # Auxiliary heads
        self.lead_lag_head = nn.Sequential(
            nn.Linear(fused_dim, half),
            nn.LayerNorm(half),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(half, 1),
            nn.Tanh(),
        )
        self.volume_anomaly_head = nn.Sequential(
            nn.Linear(fused_dim, quarter),
            nn.GELU(),
            nn.Linear(quarter, 1),
            nn.Sigmoid(),
        )
        self.sentiment_surge_head = nn.Sequential(
            nn.Linear(fused_dim, quarter),
            nn.GELU(),
            nn.Linear(quarter, 1),
            nn.Sigmoid(),
        )

        # Weight initialisation
        self._init_weights()

    # ── Weight initialisation ──────────────────────────────────────────────────
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(self, social_features: torch.Tensor,
                market_features: torch.Tensor,
                return_attention: bool = False) -> dict:
        # Variable selection
        social_embed = self.pos_encoder(self.social_vsn(social_features))
        market_embed = self.pos_encoder(self.market_vsn(market_features))

        # Temporal encoding
        social_enc = self.social_encoder(social_embed)
        market_enc = self.market_encoder(market_embed)

        # Bidirectional cross-modal attention
        social_att, _ = self.cross_attn_sm(social_enc, market_enc, market_enc)
        market_att, _ = self.cross_attn_ms(market_enc, social_enc, social_enc)

        # Fusion & global pooling
        fused   = torch.cat([social_att, market_att], dim=-1)
        fused   = self.fusion_encoder(fused)
        pooled  = fused.mean(dim=1)

        manip_prob = self.detection_head(pooled)

        outputs = {
            # Canonical key (used everywhere in eval, inference)
            'manipulation_probability': manip_prob,
            # Backwards-compatible alias (used by training loop, rule-based fallback)
            'manipulation_prob':        manip_prob,
            'lead_lag_score':           self.lead_lag_head(pooled),
            'volume_anomaly':           self.volume_anomaly_head(pooled),
            'sentiment_surge':          self.sentiment_surge_head(pooled),
        }

        # Attention weights not returned in SDPA mode (always None now, kept for API compat)
        if return_attention:
            outputs['attention_social_to_market'] = None
            outputs['attention_market_to_social'] = None

        return outputs
