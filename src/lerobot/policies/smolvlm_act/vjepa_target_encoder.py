"""Frozen V-JEPA2 Target Encoder for Stage 2 training.

Wraps the pretrained V-JEPA2 encoder and disentangle post-trained
ProjectionHead (task_head) to produce target latent representations for future frames.

Input: video [B, 3, T, H, W] -> output: [B, proj_dim]
"""

import os
import sys

import torch
import torch.nn as nn

# Add vjepa2 to path for imports.
# Try: VJEPA2_ROOT env var, then Spurious-Correlation-Bye-Bye/vjepa2, then ~/vjepa2
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CANDIDATE_ROOTS = [
    os.environ.get("VJEPA2_ROOT", ""),
    os.path.normpath(os.path.join(_THIS_DIR, "..", "..", "..", "..", "..", "vjepa2")),
    os.path.normpath(os.path.join(_THIS_DIR, "..", "..", "..", "..", "..", "..", "Spurious-Correlation-Bye-Bye", "vjepa2")),
    os.path.expanduser("~/Spurious-Correlation-Bye-Bye/vjepa2"),
    os.path.expanduser("~/vjepa2"),
]
VJEPA2_ROOT = None
for _root in _CANDIDATE_ROOTS:
    if _root and os.path.isdir(os.path.join(_root, "src", "models")):
        VJEPA2_ROOT = _root
        break
if VJEPA2_ROOT is None:
    raise RuntimeError(
        "Cannot find vjepa2 directory. Set VJEPA2_ROOT env var or place vjepa2/ "
        "in ~/Spurious-Correlation-Bye-Bye/vjepa2 or ~/vjepa2"
    )
if VJEPA2_ROOT not in sys.path:
    sys.path.insert(0, VJEPA2_ROOT)

from src.models import vision_transformer as vit_module
from src.models.projection_head import ProjectionHead


# V-JEPA2 architecture configs: (encoder_fn_name, embed_dim, num_heads)
VJEPA2_ARCH_CONFIGS = {
    "vit_large": ("vit_large", 1024, 16),
    "vit_huge": ("vit_huge", 1280, 16),
    "vit_giant": ("vit_giant_xformers_rope", 1408, 22),
    "vit_giant_384": ("vit_giant_xformers_rope", 1408, 22),
}


class VJEPATargetEncoder(nn.Module):
    """Frozen V-JEPA2 encoder + pretrained task-biased ProjectionHead.

    Produces target latent representations for future frames.
    All parameters are frozen (no gradients).

    Args:
        checkpoint_path: path to disentangle post-training checkpoint
        arch: V-JEPA2 architecture name
        img_size: spatial resolution
        num_frames: number of frames for video input (default 8)
        proj_dim: ProjectionHead output dimension
        pooler_depth: AttentivePooler depth
    """

    def __init__(
        self,
        checkpoint_path: str,
        arch: str = "vit_large",
        img_size: int = 256,
        num_frames: int = 8,
        proj_dim: int = 256,
        pooler_depth: int = 1,
    ):
        super().__init__()

        if arch not in VJEPA2_ARCH_CONFIGS:
            raise ValueError(f"Unknown arch '{arch}'. Choose from: {list(VJEPA2_ARCH_CONFIGS.keys())}")

        encoder_fn_name, embed_dim, num_heads = VJEPA2_ARCH_CONFIGS[arch]
        self.embed_dim = embed_dim
        self.proj_dim = proj_dim
        self.num_frames = num_frames

        # Build encoder
        encoder_fn = getattr(vit_module, encoder_fn_name)
        encoder_kwargs = dict(
            img_size=(img_size, img_size),
            num_frames=num_frames,
            tubelet_size=2,
            patch_size=16,
        )
        if "rope" in encoder_fn_name or "xformers" in encoder_fn_name:
            encoder_kwargs.update(
                use_sdpa=True,
                use_SiLU=False,
                wide_SiLU=True,
                uniform_power=False,
                use_rope=True,
            )
        self.encoder = encoder_fn(**encoder_kwargs)

        # Build ProjectionHead (AttentivePooler + Linear)
        self.task_head = ProjectionHead(
            embed_dim=embed_dim,
            proj_dim=proj_dim,
            num_heads=num_heads,
            pooler_depth=pooler_depth,
        )

        # Load pretrained weights
        self._load_checkpoint(checkpoint_path)

        # Freeze all parameters
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

        print(f"[VJEPATargetEncoder] arch={arch}, embed_dim={embed_dim}, "
              f"num_frames={num_frames}, proj_dim={proj_dim}")

    def _load_checkpoint(self, checkpoint_path: str):
        """Load encoder and task_head weights from disentangle checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        # Clean 'module.' and 'backbone.' prefixes from encoder state dict
        encoder_sd = {}
        for k, v in ckpt["encoder"].items():
            k = k.replace("module.", "").replace("backbone.", "")
            encoder_sd[k] = v
        self.encoder.load_state_dict(encoder_sd, strict=False)

        # Load task_head
        task_head_sd = {}
        for k, v in ckpt["task_head"].items():
            k = k.replace("module.", "")
            task_head_sd[k] = v
        self.task_head.load_state_dict(task_head_sd)

        print(f"[VJEPATargetEncoder] Loaded checkpoint from {checkpoint_path}")

    @torch.no_grad()
    def forward(self, video):
        """
        Args:
            video: [B, 3, T, H, W] - future frames as video clip

        Returns:
            z: [B, proj_dim] - projected spatiotemporal latent representation
        """
        patch_features = self.encoder(video)  # [B, T'*H'*W', embed_dim]
        z = self.task_head(patch_features)  # [B, proj_dim]
        return z
