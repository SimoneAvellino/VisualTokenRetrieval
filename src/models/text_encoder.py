"""Frozen CLIP text encoder with a trainable projection head."""

from __future__ import annotations

import torch
import torch.nn as nn

from transformers import CLIPTextModel, CLIPTokenizer


class CLIPTextEncoder(nn.Module):
    """Frozen CLIP backbone + trainable LayerNorm-Linear-LayerNorm projection.

    Maps the per-token CLIP text states into the model's ``embed_dim`` space.
    Only the projection head is trained; the CLIP backbone is frozen.
    """

    def __init__(self, embed_dim: int, clip_model_name: str) -> None:
        super().__init__()
        print(f"Loading CLIP: {clip_model_name}")
        self.encoder = CLIPTextModel.from_pretrained(clip_model_name)
        for p in self.encoder.parameters():
            p.requires_grad = False
        clip_dim = self.encoder.config.hidden_size
        self.proj = nn.Sequential(
            nn.LayerNorm(clip_dim),
            nn.Linear(clip_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        print(f"CLIP {clip_dim}d -> proj -> {embed_dim}d")

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.proj(out.last_hidden_state.to(self.proj[0].weight.dtype))


def build_clip_tokenizer(name: str) -> CLIPTokenizer:
    """Load the CLIP tokenizer matching the text encoder."""
    print(f"Loading CLIP tokenizer: {name}")
    return CLIPTokenizer.from_pretrained(name)
