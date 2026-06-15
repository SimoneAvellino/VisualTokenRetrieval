"""Frozen CLIP text encoder with a trainable projection head."""

from __future__ import annotations

import torch
import torch.nn as nn

from transformers import CLIPTextModel, CLIPTokenizer


class CLIPTextEncoder(nn.Module):
    """Frozen CLIP backbone, optionally + a trainable projection head.

    With ``project=True`` (default) a trainable LayerNorm-Linear-LayerNorm head
    maps the per-token CLIP text states into ``embed_dim``; only that head is
    trained, the CLIP backbone is frozen. With ``project=False`` the raw CLIP
    states are returned unchanged (no parameters added): the consuming model must
    then run at the CLIP hidden size, exposed as :attr:`out_dim`.
    """

    def __init__(
        self, embed_dim: int, clip_model_name: str, project: bool = True
    ) -> None:
        super().__init__()
        print(f"Loading CLIP: {clip_model_name}")
        self.encoder = CLIPTextModel.from_pretrained(clip_model_name)
        for p in self.encoder.parameters():
            p.requires_grad = False
        clip_dim = self.encoder.config.hidden_size
        self.clip_dim = clip_dim
        self.project = project
        if project:
            self.proj = nn.Sequential(
                nn.LayerNorm(clip_dim),
                nn.Linear(clip_dim, embed_dim),
                nn.LayerNorm(embed_dim),
            )
            self.out_dim = embed_dim
            print(f"CLIP {clip_dim}d -> proj -> {embed_dim}d")
        else:
            self.proj = None
            self.out_dim = clip_dim
            print(f"CLIP {clip_dim}d -> raw (no projection)")

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if self.proj is None:
            # raw CLIP states (already in the model's dtype: the frozen backbone
            # is moved with the rest of the module).
            return out.last_hidden_state
        return self.proj(out.last_hidden_state.to(self.proj[0].weight.dtype))

    def pooled(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Return CLIP's sentence embedding (the pooled [EOS] state) ``[B, out_dim]``.

        This is the representation CLIP aligns with images — a stronger single-vector
        summary of the question than mean-pooling the per-token states.
        """
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.pooler_output
        if self.proj is None:
            return pooled
        return self.proj(pooled.to(self.proj[0].weight.dtype))


def build_clip_tokenizer(name: str) -> CLIPTokenizer:
    """Load the CLIP tokenizer matching the text encoder."""
    print(f"Loading CLIP tokenizer: {name}")
    return CLIPTokenizer.from_pretrained(name)
