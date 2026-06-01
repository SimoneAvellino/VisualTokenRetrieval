"""Qwen3-VL vision-tower loading and visual-token encoding.

Wraps a frozen Qwen3-VL model so that only its vision stack (ViT + patch merger
+ projector) is run on a list of frames, returning projected visual tokens
already in the language model's embedding space.

Supports offline / air-gapped clusters: it resolves a local HF snapshot from the
cache when present and shims ``huggingface_hub.is_offline_mode`` for older
transformers versions.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image

import huggingface_hub
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


# huggingface_hub offline-mode compatibility shim
if not hasattr(huggingface_hub, "is_offline_mode"):
    from huggingface_hub.constants import HF_HUB_OFFLINE

    def _is_offline_mode() -> bool:
        return HF_HUB_OFFLINE

    huggingface_hub.is_offline_mode = _is_offline_mode


def resolve_local_hf_snapshot(model_name: str, hf_cache_dir: str) -> Optional[str]:
    """Return a local snapshot path for ``model_name`` if cached, else None."""
    if Path(model_name).exists():
        return model_name
    try:
        org, repo = model_name.split("/", 1)
    except ValueError:
        return None

    hub_root = Path(hf_cache_dir) / "hub"
    model_root = hub_root / f"models--{org}--{repo}"
    if not model_root.exists():
        return None

    ref_file = model_root / "refs" / "main"
    snapshot_id = (
        ref_file.read_text(encoding="utf-8").strip() if ref_file.exists() else None
    )
    if snapshot_id:
        snapshot_path = model_root / "snapshots" / snapshot_id
        if snapshot_path.exists():
            return str(snapshot_path)

    snapshots_dir = model_root / "snapshots"
    if snapshots_dir.exists():
        snapshots = sorted([p for p in snapshots_dir.iterdir() if p.is_dir()])
        if snapshots:
            return str(snapshots[-1])

    return None


def load_model(
    model_id: str,
    hf_cache_dir: str,
    target_resolution: Tuple[int, int],
    attn_implementation: Optional[str] = "flash_attention_2",
) -> Tuple[Qwen3VLForConditionalGeneration, AutoProcessor]:
    """Load the Qwen3-VL model and processor.

    Falls back to default attention if the requested ``attn_implementation``
    (e.g. flash-attn 2) is unavailable, which is common on local machines.
    """
    model_source = resolve_local_hf_snapshot(model_id, hf_cache_dir) or model_id
    local_only = Path(model_source).exists() or bool(os.environ.get("HF_HUB_OFFLINE", ""))

    print(f">>> model_id     = {model_id}", flush=True)
    print(f">>> model_source = {model_source}", flush=True)
    print(f">>> cache_dir    = {hf_cache_dir}", flush=True)
    print(f">>> resolution   = {target_resolution} (W x H)", flush=True)

    processor = AutoProcessor.from_pretrained(
        model_source,
        trust_remote_code=False,
        local_files_only=local_only,
        cache_dir=hf_cache_dir,
    )

    common = dict(
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=False,
        local_files_only=local_only,
        cache_dir=hf_cache_dir,
    )
    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_source, attn_implementation=attn_implementation, **common
        )
        print(f">>> Loaded with {attn_implementation}.", flush=True)
    except Exception as e:
        print(
            f">>> {attn_implementation} unavailable ({e}). Falling back to default attention.",
            flush=True,
        )
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_source, **common)

    model.eval()
    return model, processor


@torch.inference_mode()
def encode_segment(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    pil_frames: List[Image.Image],
    save_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Run only the Qwen3-VL vision tower on frames; return projected tokens.

    Returns:
        Tensor ``[num_visual_tokens, hidden_size]`` in ``save_dtype`` on CPU.
        The Qwen3-VL ``visual`` module returns ``(hidden_states, deepstack)``;
        ``hidden_states`` is already projected into the LM embedding space and
        ``deepstack`` is dropped.
    """
    vi = processor.image_processor(images=pil_frames, return_tensors="pt")
    vi = vi.to(model.device)

    pixel_values = vi.pixel_values.to(torch.bfloat16)
    grid_thw = vi.image_grid_thw

    out = model.visual(pixel_values, grid_thw=grid_thw)
    projected = out[0] if isinstance(out, (tuple, list)) else out

    return projected.detach().to("cpu", dtype=save_dtype).contiguous()
