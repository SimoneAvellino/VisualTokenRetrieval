"""xLSTM-based token compressor.

Compresses a variable-length context token sequence ``[B, T, D]`` into a fixed
set of learned query vectors ``Z [B, num_queries, D]`` via an xLSTM block stack
followed by cross-attention from learned queries to the xLSTM hidden states.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

from xlstm import (
    FeedForwardConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
)


def build_xlstm(
    embed_dim: int,
    num_heads: int,
    max_seq_len: int,
    num_blocks: int,
    slstm_at: Sequence[int],
) -> xLSTMBlockStack:
    """Build an xLSTM block stack with mLSTM blocks and sLSTM at ``slstm_at``."""
    return xLSTMBlockStack(
        xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    num_heads=num_heads, conv1d_kernel_size=4, qkv_proj_blocksize=4
                )
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend="vanilla", num_heads=4, bias_init="powerlaw_blockdependent"
                ),
                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
            ),
            context_length=max_seq_len,
            num_blocks=num_blocks,
            embedding_dim=embed_dim,
            slstm_at=list(slstm_at),
        )
    )


class TokenCompressor(nn.Module):
    """``x [B,T,D]`` -> ``Z [B,num_queries,D]``, ``M [B,D]``, optional ``H [B,T,D]``."""

    def __init__(
        self,
        embed_dim: int,
        num_queries: int,
        num_heads: int,
        max_seq_len: int,
        num_blocks: int,
        slstm_at: Sequence[int],
    ) -> None:
        super().__init__()
        self.xlstm = build_xlstm(embed_dim, num_heads, max_seq_len, num_blocks, slstm_at)
        self.global_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.LayerNorm(embed_dim)
        )
        self.q_tokens = nn.Parameter(torch.randn(num_queries, embed_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_h: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        H = self.xlstm(x)
        mf = mask.unsqueeze(-1).to(H.dtype) if mask is not None else None
        M = ((H * mf).sum(1) / mf.sum(1).clamp(1e-6)) if mf is not None else H.mean(1)
        M = self.global_proj(M)
        Q = self.q_tokens.unsqueeze(0).expand(x.shape[0], -1, -1) + M.unsqueeze(1)
        Z, _ = self.cross_attn(
            Q,
            H,
            H,
            key_padding_mask=(~mask if mask is not None else None),
            need_weights=False,
        )
        return self.ln(Z), M, (H if return_h else None)
