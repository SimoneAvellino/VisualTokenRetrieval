"""Query-conditioned GT visual-token reconstructor.

Pipeline::

    context  -> TokenCompressor (xLSTM)  -> Z (+ M, optional H)
    question -> CLIPTextEncoder          -> Q
    Q cross-attends memory -> Transformer Decoder -> pred_tokens [B, L, D]

``memory_mode`` selects what the decoder attends over: the compressed queries
``z``, queries + global vector ``z_m``, the full hidden states ``h``, or their
concatenation ``h_z``.

``decode_mode`` selects how many tokens are produced:

    - ``"fixed"`` (legacy): exactly ``out_len`` tokens; the target is time-warped
      to ``out_len`` upstream, which destroys the mapping to real timestamps.
    - ``"length_aware"``: up to ``max_out_len`` tokens plus a scalar length head
      that regresses the *true* token count. The target keeps its native length
      (no warp), so output slot ``t`` retains its real-time meaning; the loss is
      masked beyond the true length. Compute stays bounded at ``max_out_len``.

``forward`` always returns ``(pred, aux)`` where ``aux`` is a dict. In
length-aware mode ``aux["pred_len"]`` holds the predicted token count ``[B]``.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .compressor import TokenCompressor
from .text_encoder import CLIPTextEncoder


class QueryConditionedGTReconstructor(nn.Module):
    """Query-conditioned GT token reconstructor."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        decoder_layers: int,
        num_queries: int,
        out_len: int,
        max_seq_len: int,
        dropout: float,
        num_blocks: int,
        slstm_at: Sequence[int],
        clip_model_name: str = "openai/clip-vit-large-patch14",
        memory_mode: str = "z",
        decode_mode: str = "fixed",
        max_out_len: int = 64,
        use_perceiver: bool = False,
        perceiver_depth: int = 2,
        multiscale_strides: Sequence[int] = (1,),
        model_dim: int = 0,
    ) -> None:
        super().__init__()
        assert memory_mode in {"z", "z_m", "h", "h_z"}
        assert decode_mode in {"fixed", "length_aware"}
        self.embed_dim = embed_dim
        self.out_len = out_len
        self.memory_mode = memory_mode
        self.decode_mode = decode_mode
        # number of decoder output slots: fixed -> out_len; length_aware -> cap
        self.num_slots = out_len if decode_mode == "fixed" else max_out_len

        # Internal working dimension. ``model_dim<=0`` (or == embed_dim) keeps the
        # original behaviour exactly: in/out projections become Identity, so the
        # whole core runs at embed_dim. A smaller ``model_dim`` adds linear in/out
        # projections and runs every submodule at the smaller dim, which is where
        # the parameter count lives (matrices scale with dim^2).
        d = embed_dim if model_dim <= 0 else model_dim
        self.model_dim = d
        self.in_proj = nn.Identity() if d == embed_dim else nn.Linear(embed_dim, d)
        self.out_proj = nn.Identity() if d == embed_dim else nn.Linear(d, embed_dim)

        self.compressor = TokenCompressor(
            d,
            num_queries,
            num_heads,
            max_seq_len,
            num_blocks,
            slstm_at,
            use_perceiver=use_perceiver,
            perceiver_depth=perceiver_depth,
            multiscale_strides=multiscale_strides,
        )
        self.text_encoder = CLIPTextEncoder(d, clip_model_name)
        self.q2mem = nn.MultiheadAttention(
            d, num_heads, dropout=dropout, batch_first=True
        )
        self.q2mem_ln = nn.LayerNorm(d)
        self.out_queries = nn.Parameter(torch.randn(self.num_slots, d) * 0.02)
        self.out_pos = nn.Parameter(torch.randn(self.num_slots, d) * 0.02)
        self.q_summary_proj = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.LayerNorm(d)
        )
        dec_layer = nn.TransformerDecoderLayer(
            d,
            num_heads,
            d * 2,
            dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(
            dec_layer, decoder_layers, norm=nn.LayerNorm(d)
        )
        self.out_ln = nn.LayerNorm(d)
        # length head: regresses the (positive) true token count from a pooled
        # decoder summary. softplus keeps the prediction >= 0.
        self.length_head = (
            nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
            if decode_mode == "length_aware"
            else None
        )
        # span head: regress the moment's (start, end) as fractions of query_time,
        # directly from a pooled query-conditioned summary. This is the temporal-
        # grounding output — localization that does not rely on token matching.
        self.span_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2))

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(-1).to(x.dtype)
        return (x * m).sum(1) / m.sum(1).clamp(1e-6)

    def _select_memory(
        self,
        Z: torch.Tensor,
        M: torch.Tensor,
        H: Optional[torch.Tensor],
        vmask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B = Z.shape[0]
        if self.memory_mode == "z":
            return Z, None
        if self.memory_mode == "z_m":
            return torch.cat([Z, M.unsqueeze(1)], 1), None
        if H is None:
            raise RuntimeError("memory_mode h/h_z requires return_h=True")
        if self.memory_mode == "h":
            return H, ~vmask
        zp = torch.zeros(B, Z.shape[1], dtype=torch.bool, device=Z.device)
        return torch.cat([Z, H], 1), torch.cat([zp, ~vmask], 1)

    def forward(
        self,
        video_tokens: torch.Tensor,
        video_mask: torch.Tensor,
        question_ids: torch.Tensor,
        question_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        need_h = self.memory_mode in {"h", "h_z"}
        # project tokens to the internal working dim (Identity when model_dim==embed_dim)
        video_tokens = self.in_proj(video_tokens)
        Z, M, H = self.compressor(video_tokens, mask=video_mask, return_h=need_h)
        Q = self.text_encoder(question_ids, question_mask)
        mem, mpad = self._select_memory(Z, M, H, video_mask)
        C, _ = self.q2mem(Q, mem, mem, key_padding_mask=mpad, need_weights=False)
        C = self.q2mem_ln(C + Q)
        B = video_tokens.shape[0]
        qs = self._masked_mean(Q, question_mask)
        qs = self.q_summary_proj(qs)
        tgt = (
            self.out_queries.unsqueeze(0).expand(B, -1, -1)
            + self.out_pos.unsqueeze(0)
            + qs.unsqueeze(1)
        )
        pred = self.decoder(tgt, C, memory_key_padding_mask=~question_mask)
        pred = self.out_ln(pred)

        aux: Dict[str, torch.Tensor] = {}
        pooled = pred.mean(1)  # [B, d] query-conditioned summary (internal dim)
        if self.length_head is not None:
            # length head runs at the internal dim, before projecting pred back out
            # softplus -> positive token-count prediction
            aux["pred_len"] = F.softplus(self.length_head(pooled).squeeze(-1))

        # span head -> (start, end) in [0, 1], with end >= start by construction.
        raw = self.span_head(pooled)                       # [B, 2]
        s = torch.sigmoid(raw[:, 0])
        e = s + (1.0 - s) * torch.sigmoid(raw[:, 1])
        aux["pred_span"] = torch.stack([s, e], dim=-1)     # [B, 2]

        # project predictions back to embed_dim so the loss compares in token space
        pred = self.out_proj(pred)
        return pred, aux
