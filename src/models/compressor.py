"""xLSTM-based token compressor.

Compresses a variable-length context token sequence ``[B, T, D]`` into a fixed
set of learned query vectors ``Z [B, num_queries, D]``. The xLSTM block stack
provides a linear-in-``T`` sequence backbone; a bottleneck head then distils its
hidden states into the query set.

Two bottleneck heads are available:

    - **single cross-attention** (default, legacy): learned queries cross-attend
      the hidden states once.
    - **Perceiver resampler** (``use_perceiver=True``): ``L`` latents iteratively
      cross-attend the hidden states over ``depth`` blocks, cost
      ``O(depth * L * T)`` (linear in ``T``). More latents at negligible extra
      cost relieves the 16-slot information bottleneck. With ``multiscale_strides``
      of length > 1 the latents additionally attend a multi-resolution view of the
      sequence (the hidden states pooled at several temporal strides), so both
      rapid actions (stride 1) and global context (large stride) are captured.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

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


def _masked_avgpool_time(
    H: torch.Tensor, mask: Optional[torch.Tensor], stride: int
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Average-pool ``H [B, T, D]`` over time in non-overlapping windows of ``stride``.

    Padded positions (``mask == False``) are excluded from each window's mean. A
    pooled position is valid iff its window held at least one real token. Returns
    ``(H_pooled [B, T', D], mask_pooled [B, T'] | None)``.
    """
    if stride <= 1:
        return H, mask
    B, T, D = H.shape
    pad = (stride - T % stride) % stride
    if pad:
        H = F.pad(H, (0, 0, 0, pad))
        if mask is not None:
            mask = F.pad(mask, (0, pad), value=False)
    Tp = H.shape[1] // stride
    Hw = H.reshape(B, Tp, stride, D)
    if mask is None:
        return Hw.mean(2), None
    mw = mask.reshape(B, Tp, stride).to(H.dtype).unsqueeze(-1)  # [B, Tp, stride, 1]
    pooled = (Hw * mw).sum(2) / mw.sum(2).clamp_min(1e-6)
    pooled_mask = mw.squeeze(-1).sum(2) > 0
    return pooled, pooled_mask


class PerceiverResampler(nn.Module):
    """Iterative latent bottleneck (Perceiver-style cross-attention resampler).

    ``L`` learned latents cross-attend the (multi-scale) hidden states over
    ``depth`` blocks, each block a cross-attention + feed-forward with pre-norm
    residuals. Cost is ``O(depth * L * T_total)`` — linear in sequence length, so
    raising ``L`` (e.g. 16 -> 64) is cheap relative to ``O(T^2)`` self-attention.
    """

    def __init__(
        self,
        dim: int,
        n_latents: int,
        depth: int,
        heads: int,
        strides: Sequence[int] = (1,),
    ) -> None:
        super().__init__()
        self.strides = list(strides)
        self.latents = nn.Parameter(torch.randn(n_latents, dim) * 0.02)
        self.blocks = nn.ModuleList(
            nn.ModuleDict(
                {
                    "ca_ln_q": nn.LayerNorm(dim),
                    "ca_ln_kv": nn.LayerNorm(dim),
                    "ca": nn.MultiheadAttention(dim, heads, batch_first=True),
                    "ff_ln": nn.LayerNorm(dim),
                    "ff": nn.Sequential(
                        nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim)
                    ),
                }
            )
            for _ in range(depth)
        )

    def _multiscale_kv(
        self, H: torch.Tensor, vmask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Concatenate ``H`` pooled at each stride into a single key/value bank."""
        if self.strides == [1]:
            return H, (~vmask if vmask is not None else None)
        feats: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []
        any_mask = vmask is not None
        for s in self.strides:
            hs, ms = _masked_avgpool_time(H, vmask, s)
            feats.append(hs)
            if any_mask:
                masks.append(ms if ms is not None else hs.new_ones(hs.shape[:2], dtype=torch.bool))
        kv = torch.cat(feats, dim=1)
        key_padding_mask = ~torch.cat(masks, dim=1) if any_mask else None
        return kv, key_padding_mask

    def forward(self, H: torch.Tensor, vmask: Optional[torch.Tensor] = None) -> torch.Tensor:
        kv, key_padding_mask = self._multiscale_kv(H, vmask)
        z = self.latents.unsqueeze(0).expand(H.shape[0], -1, -1)
        for blk in self.blocks:
            a, _ = blk["ca"](
                blk["ca_ln_q"](z),
                blk["ca_ln_kv"](kv),
                blk["ca_ln_kv"](kv),
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            z = z + a
            z = z + blk["ff"](blk["ff_ln"](z))
        return z


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
        use_perceiver: bool = False,
        perceiver_depth: int = 2,
        multiscale_strides: Sequence[int] = (1,),
    ) -> None:
        super().__init__()
        self.xlstm = build_xlstm(embed_dim, num_heads, max_seq_len, num_blocks, slstm_at)
        self.global_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.LayerNorm(embed_dim)
        )
        self.use_perceiver = use_perceiver
        if use_perceiver:
            self.resampler = PerceiverResampler(
                embed_dim, num_queries, perceiver_depth, num_heads, multiscale_strides
            )
        else:
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
        if self.use_perceiver:
            Z = self.resampler(H, vmask=mask)
        else:
            Q = self.q_tokens.unsqueeze(0).expand(x.shape[0], -1, -1) + M.unsqueeze(1)
            Z, _ = self.cross_attn(
                Q,
                H,
                H,
                key_padding_mask=(~mask if mask is not None else None),
                need_weights=False,
            )
        return self.ln(Z), M, (H if return_h else None)
