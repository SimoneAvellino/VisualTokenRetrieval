"""Simplified query-conditioned mean-token reconstructor.

A deliberately stripped-down counterpart to
:class:`~src.models.reconstructor.QueryConditionedGTReconstructor`. It predicts a
*single* pooled vector (trained against the mean of the GT window tokens), but —
crucially — lets the **question drive** that prediction by attending over the
*full* video sequence, instead of collapsing the video to one vector first.

Pipeline::

    video    -> in_proj (embed_dim -> core) -> xLSTM -> H   [B, T, core]   (kept!)
    question -> frozen CLIP sentence embedding (pooled [EOS])              [B, core]
    tgt      = q_proj(sentence_emb)                                        [B, 1, core]
    pred     = TransformerDecoder(tgt, memory=H)   # the query localises the moment
               -> out_proj (core -> embed_dim)                            [B, 1, embed_dim]

Design notes (motivated by the query-usage ablation, which showed the previous
single-video-vector design *ignored* the question):

    - **Video kept as a sequence.** ``H`` (all timesteps) is the decoder memory, so
      the question-derived query can attend to — and pick out — the relevant moment.
      The output size is fixed (one query slot) regardless of ``T``: cross-attention
      over variable-length keys yields one vector per query (Perceiver/DETR style).
    - **Question drives the prediction.** The decoder query ``tgt`` is derived from
      the question's CLIP sentence embedding (no dominant learnable query), so the
      output is genuinely a function of the question.
    - **CLIP sentence embedding.** The pooled [EOS] vector (what CLIP aligns with
      images) is a stronger text summary than mean-pooling the raw token states.

``forward`` returns ``(pred [B, 1, embed_dim], aux={})`` to match the training
loop's calling convention.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn

from .compressor import build_xlstm
from .text_encoder import CLIPTextEncoder


class MeanReconstructor(nn.Module):
    """Predict the mean GT token from (video context, question)."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        decoder_layers: int,
        dropout: float,
        num_blocks: int,
        slstm_at: Sequence[int],
        max_seq_len: int,
        num_queries: int = 1,
        clip_model_name: str = "openai/clip-vit-large-patch14",
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_queries = num_queries

        # Text encoder runs raw (no projection); the core dim is therefore the
        # CLIP text hidden size. Build it first so every other submodule can be
        # sized to that dim.
        self.text_encoder = CLIPTextEncoder(embed_dim, clip_model_name, project=False)
        d = self.text_encoder.out_dim
        self.core_dim = d

        # video tokens (embed_dim) <-> core dim
        self.in_proj = nn.Linear(embed_dim, d)
        self.out_proj = nn.Linear(d, embed_dim)

        self.xlstm = build_xlstm(d, num_heads, max_seq_len, num_blocks, slstm_at)

        # N output query slots (fixed, independent of the video length), each
        # conditioned on the question (Fix 2 + 3): a learnable per-slot embedding
        # plus the question's CLIP sentence embedding. More slots give the decoder
        # room to attend different parts of the moment before they are pooled.
        self.out_queries = nn.Parameter(torch.randn(num_queries, d) * 0.02)
        self.q_proj = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.LayerNorm(d)
        )

        # the queries attend over ALL video timesteps (Fix 1): cross-attention picks
        # the queried moment out of the full sequence H.
        dec_layer = nn.TransformerDecoderLayer(
            d, num_heads, d * 2, dropout, batch_first=True, norm_first=True, activation="gelu"
        )
        self.decoder = nn.TransformerDecoder(dec_layer, decoder_layers, norm=nn.LayerNorm(d))

    def forward(
        self,
        video_tokens: torch.Tensor,
        video_mask: torch.Tensor,
        question_ids: torch.Tensor,
        question_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        H = self.xlstm(self.in_proj(video_tokens))     # [B, T, core]  full sequence

        # question -> CLIP sentence embedding -> per-slot output queries
        q = self.q_proj(self.text_encoder.pooled(question_ids, question_mask))  # [B, core]
        tgt = self.out_queries.unsqueeze(0) + q.unsqueeze(1)   # [B, N, core]

        # the queries attend the video timesteps to localise the moment.
        pred = self.decoder(tgt, H, memory_key_padding_mask=~video_mask)  # [B, N, core]
        pred = self.out_proj(pred)                     # [B, N, embed_dim]
        return pred, {}
