"""
v12_model.py — Causal Graph Transformer for ADIA Lab Challenge.

Architecture:
  1. EdgeCNNEncoder   — same 8ch conv tower as v8b, applied to ALL n*(n-1) edges
  2. ExtraEdgeMLP     — projects scalar edge features (LiNGAM, PC) to d_edge
  3. NodeMLP          — embeds per-node MI/CMI/stat features
  4. NodeTypeEmbed    — learned token for X, Y, other
  5. GraphTransformerLayer × n_layers
       • Multi-head self-attention over all n variables
       • Edge embedding as additive attention bias (Graphormer-style)
       • FFN + pre-norm
  6. OutputHead       — 8-class logits for each non-X,Y node

Key difference from v8b:
  v8b: 4 edges per node → local node head
  v12: ALL edges → full graph transformer → joint classification
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Constants ────────────────────────────────────────────────────────────────
N_CHANNELS    = 8   # sorted_u, v_sorted_by_u, 3×kernel, 3×ANM
N_EXTRA_EDGE  = 3   # lingam_B, pc_skel, pc_dir  (from GraphData.get_extra_edge_features)
N_NODE_FEATS  = 7   # MI, CMI×2, variance, skewness, kurtosis
N_NODE_TYPES  = 3   # 0=X, 1=Y, 2=other
N_CLASSES     = 8


# ─── Edge CNN Encoder (identical tower to v8b) ────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, d: int, ks: int = 3, n_groups: int = 8):
        super().__init__()
        self.conv = nn.Conv1d(d, d, ks, padding=ks // 2, bias=False)
        self.norm = nn.GroupNorm(n_groups, d)
        self.act  = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.norm(self.conv(x)))


class EdgeCNNEncoder(nn.Module):
    """
    [B_e, N_CHANNELS, L] → [B_e, d_edge]

    Same architecture as v8b stem+convs+avgpool.
    Accepts variable-length L (AvgPool collapses the time dim).
    """
    def __init__(self, in_channels: int = N_CHANNELS, d_edge: int = 64, n_conv: int = 5):
        super().__init__()
        self.stem = nn.Linear(in_channels, d_edge)  # channel expansion
        self.convs = nn.Sequential(*[ConvBlock(d_edge) for _ in range(n_conv)])
        self.pool  = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B_e, C, L]
        x = self.stem(x.permute(0, 2, 1)).permute(0, 2, 1)  # [B_e, d_edge, L]
        x = self.convs(x)                                     # [B_e, d_edge, L]
        x = self.pool(x).squeeze(-1)                          # [B_e, d_edge]
        return x


# ─── Edge feature aggregation ─────────────────────────────────────────────────

class EdgeFeatureProjector(nn.Module):
    """
    Combines CNN embedding [B, n, n, d_edge] with scalar extras [B, n, n, N_EXTRA_EDGE]
    → [B, n, n, d_edge].
    """
    def __init__(self, d_edge: int = 64, n_extra: int = N_EXTRA_EDGE):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_edge + n_extra, d_edge),
            nn.LayerNorm(d_edge),
            nn.GELU(),
        )

    def forward(self, cnn_emb: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
        # cnn_emb: [B, n, n, d_edge], extra: [B, n, n, N_EXTRA_EDGE]
        return self.proj(torch.cat([cnn_emb, extra], dim=-1))


# ─── Graph Transformer Layer ──────────────────────────────────────────────────

class EdgeBiasProjector(nn.Module):
    """Maps edge embeddings to per-head attention biases."""
    def __init__(self, d_edge: int, n_heads: int):
        super().__init__()
        self.linear = nn.Linear(d_edge, n_heads)

    def forward(self, edge_emb: torch.Tensor) -> torch.Tensor:
        # edge_emb: [B, n, n, d_edge] → [B, n_heads, n, n]
        return self.linear(edge_emb).permute(0, 3, 1, 2)


class GraphTransformerLayer(nn.Module):
    """
    One layer of Graphormer-style attention:
      attn(i→j) = softmax( (Q_i · K_j)/√d_k + edge_bias(i,j) ) · V_j
    Followed by FFN. Pre-norm (layer norm before each sub-layer).
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_edge: int,
        d_ff: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        d_ff = d_ff or 4 * d_model

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model)

        self.edge_bias = EdgeBiasProjector(d_edge, n_heads)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,           # [B, n, d_model]
        edge_emb: torch.Tensor,    # [B, n, n, d_edge]
        mask: torch.Tensor,        # [B, n] bool, True=valid
    ) -> torch.Tensor:
        B, n, D = x.shape
        H, d_k = self.n_heads, self.d_k

        # ── Attention ────────────────────────────────────────────────────────
        residual = x
        x_norm = self.norm1(x)

        Q = self.W_Q(x_norm).view(B, n, H, d_k).transpose(1, 2)  # [B, H, n, d_k]
        K = self.W_K(x_norm).view(B, n, H, d_k).transpose(1, 2)
        V = self.W_V(x_norm).view(B, n, H, d_k).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # [B, H, n, n]
        attn = attn + self.edge_bias(edge_emb)                          # add edge bias

        # Mask padding nodes (they should not be attended to)
        if mask is not None:
            pad_mask = ~mask  # [B, n], True=padding
            attn = attn.masked_fill(pad_mask[:, None, None, :], float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = attn.nan_to_num(0.0)   # handle all-padding edge case
        attn = self.drop(attn)

        out = torch.matmul(attn, V)                          # [B, H, n, d_k]
        out = out.transpose(1, 2).contiguous().view(B, n, D) # [B, n, D]
        out = self.W_O(out)
        x   = residual + self.drop(out)

        # ── FFN ──────────────────────────────────────────────────────────────
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


# ─── Full Model ───────────────────────────────────────────────────────────────

class V12Model(nn.Module):
    """
    Full Graph Transformer for causal role classification.

    Input per graph:
      edge_seqs   [B, n, n, C, L]   — 8-channel sorted obs sequences
      extra_edges [B, n, n, E_extra] — scalar edge features (LiNGAM, PC)
      node_feats  [B, n, F_node]    — per-node MI/CMI/stat features
      node_types  [B, n]  int       — 0=X, 1=Y, 2=other
      mask        [B, n]  bool      — True=valid node

    Output:
      logits  [B, n, 8]  — class logits for ALL nodes
                           (caller applies loss only to "other" nodes)
    """
    def __init__(
        self,
        in_channels: int    = N_CHANNELS,
        n_extra_edge: int   = N_EXTRA_EDGE,
        f_node: int         = N_NODE_FEATS,
        d_edge: int         = 64,
        d_model: int        = 256,
        n_heads: int        = 8,
        n_layers: int       = 6,
        n_classes: int      = N_CLASSES,
        dropout: float      = 0.1,
    ):
        super().__init__()
        self.d_edge = d_edge

        # Edge branch
        self.edge_cnn  = EdgeCNNEncoder(in_channels, d_edge)
        self.edge_proj = EdgeFeatureProjector(d_edge, n_extra_edge)

        # Node branch
        self.node_mlp = nn.Sequential(
            nn.Linear(f_node, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.node_type_emb = nn.Embedding(N_NODE_TYPES, d_model)

        # Transformer
        self.layers = nn.ModuleList([
            GraphTransformerLayer(d_model, n_heads, d_edge, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def encode_edges(
        self,
        edge_seqs: torch.Tensor,     # [B, n, n, C, L]
        extra_edges: torch.Tensor,   # [B, n, n, E_extra]
    ) -> torch.Tensor:
        """Returns [B, n, n, d_edge] edge embeddings."""
        B, n, _, C, L = edge_seqs.shape
        # Flatten to [B*n*n, C, L], run CNN, reshape back
        flat = edge_seqs.view(B * n * n, C, L)
        cnn_out = self.edge_cnn(flat).view(B, n, n, self.d_edge)  # [B, n, n, d_edge]
        return self.edge_proj(cnn_out, extra_edges)                 # [B, n, n, d_edge]

    def forward(
        self,
        edge_seqs: torch.Tensor,     # [B, n, n, C, L]
        extra_edges: torch.Tensor,   # [B, n, n, E_extra]
        node_feats: torch.Tensor,    # [B, n, F_node]
        node_types: torch.Tensor,    # [B, n] int
        mask: torch.Tensor,          # [B, n] bool
    ) -> torch.Tensor:               # [B, n, 8]

        # 1. Encode all edges → [B, n, n, d_edge]
        edge_emb = self.encode_edges(edge_seqs, extra_edges)

        # 2. Build node embeddings: feature MLP + type embedding
        x = self.node_mlp(node_feats) + self.node_type_emb(node_types)  # [B, n, d_model]

        # 3. Transformer layers
        for layer in self.layers:
            x = layer(x, edge_emb, mask)

        # 4. Output logits for all nodes
        x = self.final_norm(x)
        return self.output_head(x)  # [B, n, 8]


# ─── Convenience factory ──────────────────────────────────────────────────────

def build_v12_model(
    d_edge: int   = 64,
    d_model: int  = 256,
    n_heads: int  = 8,
    n_layers: int = 6,
    dropout: float = 0.1,
) -> V12Model:
    return V12Model(
        in_channels=N_CHANNELS,
        n_extra_edge=N_EXTRA_EDGE,
        f_node=N_NODE_FEATS,
        d_edge=d_edge,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        n_classes=N_CLASSES,
        dropout=dropout,
    )


if __name__ == "__main__":
    # Quick smoke test
    B, n, C, L = 2, 6, N_CHANNELS, 256
    E = N_EXTRA_EDGE

    edge_seqs   = torch.randn(B, n, n, C, L)
    extra_edges = torch.randn(B, n, n, E)
    node_feats  = torch.randn(B, n, N_NODE_FEATS)
    node_types  = torch.randint(0, 3, (B, n))
    node_types[:, 0] = 0   # X
    node_types[:, 1] = 1   # Y
    mask        = torch.ones(B, n, dtype=torch.bool)

    model = build_v12_model()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"V12Model — {n_params:,} trainable parameters")

    logits = model(edge_seqs, extra_edges, node_feats, node_types, mask)
    print(f"Output shape: {logits.shape}")   # [2, 6, 8]
    assert logits.shape == (B, n, N_CLASSES)
    print("Smoke test passed.")
