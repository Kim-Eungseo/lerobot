"""Action heads for SmolVLM-Act.

All heads take a single hidden state vector (B, hidden_dim) from the learnable
action token and output an action chunk (B, chunk_size, action_dim).

Available heads:
  - MLPActionHead: simple sequential MLP (original SmolVLM-Act default)
  - ResNetActionHead: MLPResNet with residual blocks (adapted from OpenVLA-OFT L1Regression)
  - DiffusionActionHead: DDIM-based denoising diffusion (adapted from OpenVLA-OFT)

Adapted from: openvla-oft/prismatic/models/action_heads.py
"""

import math

import torch
import torch.nn as nn
from torch import Tensor


# ─── Building blocks ────────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal timestep embedding for diffusion."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        # x: (B,)
        device = x.device
        half_dim = self.dim // 2
        exponent = torch.arange(half_dim, device=device) * -math.log(10000) / (half_dim - 1)
        emb = x[:, None] * torch.exp(exponent)[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)  # (B, dim)


class MLPResNetBlock(nn.Module):
    """Pre-LayerNorm residual MLP block."""

    def __init__(self, dim: int):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.ffn(x)


class MLPResNet(nn.Module):
    """MLP with residual blocks: LN -> FC -> ReLU -> ResBlocks -> LN -> FC."""

    def __init__(self, num_blocks: int, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.blocks = nn.ModuleList([MLPResNetBlock(hidden_dim) for _ in range(num_blocks)])
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.relu(self.fc1(self.layer_norm1(x)))
        for block in self.blocks:
            x = block(x)
        return self.fc2(self.layer_norm2(x))


# ─── Action Heads ───────────────────────────────────────────────────

class MLPActionHead(nn.Module):
    """Simple sequential MLP action head (original SmolVLM-Act default).

    Architecture: LayerNorm -> (Linear -> GELU) x N -> Linear -> reshape
    """

    def __init__(self, hidden_dim: int, action_dim: int, chunk_size: int,
                 mlp_hidden: int = 2048, num_layers: int = 3):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size

        layers = [nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, mlp_hidden), nn.GELU()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(mlp_hidden, mlp_hidden), nn.GELU()]
        layers.append(nn.Linear(mlp_hidden, chunk_size * action_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, hidden_state: Tensor) -> Tensor:
        """(B, hidden_dim) -> (B, chunk_size, action_dim)"""
        return self.mlp(hidden_state).reshape(-1, self.chunk_size, self.action_dim)


class ResNetActionHead(nn.Module):
    """MLPResNet-based action head with residual blocks.

    Adapted from OpenVLA-OFT L1RegressionActionHead. Uses residual connections
    for better gradient flow compared to the simple sequential MLP.

    Architecture: LN -> FC -> ReLU -> ResBlock x N -> LN -> FC -> reshape
    """

    def __init__(self, hidden_dim: int, action_dim: int, chunk_size: int,
                 mlp_hidden: int = 2048, num_blocks: int = 2):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.mlp = MLPResNet(
            num_blocks=num_blocks,
            input_dim=hidden_dim,
            hidden_dim=mlp_hidden,
            output_dim=chunk_size * action_dim,
        )

    def forward(self, hidden_state: Tensor) -> Tensor:
        """(B, hidden_dim) -> (B, chunk_size, action_dim)"""
        return self.mlp(hidden_state).reshape(-1, self.chunk_size, self.action_dim)


class DiffusionActionHead(nn.Module):
    """DDIM-based diffusion action head.

    Adapted from OpenVLA-OFT DiffusionActionHead. Generates actions via
    iterative denoising from the action token hidden state.

    Training: samples noise, adds to GT actions, predicts noise conditioned on
    hidden state + noisy actions + timestep embedding.

    Inference: starts from pure noise, iteratively denoises using DDIM scheduler.
    """

    def __init__(self, hidden_dim: int, action_dim: int, chunk_size: int,
                 mlp_hidden: int = 2048, num_blocks: int = 2,
                 num_train_steps: int = 50, num_infer_steps: int = 10):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.num_train_steps = num_train_steps
        self.num_infer_steps = num_infer_steps

        # Noise prediction network: conditioned on [hidden_state; noisy_action; timestep_emb]
        # Per-step: hidden_dim + action_dim + hidden_dim -> action_dim
        self.noise_predictor = MLPResNet(
            num_blocks=num_blocks,
            input_dim=hidden_dim + action_dim + hidden_dim,
            hidden_dim=mlp_hidden,
            output_dim=action_dim,
        )

        self.time_encoder = SinusoidalPositionalEncoding(dim=hidden_dim)

        # DDIM scheduler
        from diffusers.schedulers.scheduling_ddim import DDIMScheduler
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_train_steps,
            beta_schedule="squaredcos_cap_v2",
        )

    def compute_loss(self, hidden_state: Tensor, gt_actions: Tensor) -> Tensor:
        """Training forward: compute MSE loss on noise prediction.

        Args:
            hidden_state: (B, hidden_dim) from action token
            gt_actions: (B, chunk_size, action_dim)

        Returns:
            loss: scalar MSE loss
        """
        B, C, A = gt_actions.shape
        device = gt_actions.device

        # Sample random noise
        noise = torch.randn_like(gt_actions)

        # Sample random timesteps (one per sample)
        timesteps = torch.randint(0, self.num_train_steps, (B,), device=device)

        # Forward diffusion: add noise to GT actions
        noisy_actions = self.noise_scheduler.add_noise(gt_actions, noise, timesteps)  # (B, C, A)

        # Timestep embedding
        t_emb = self.time_encoder(timesteps.float()).to(dtype=hidden_state.dtype)  # (B, hidden_dim)

        # Predict noise per chunk step
        # Expand hidden_state and t_emb to match chunk: (B, hidden_dim) -> (B, C, hidden_dim)
        h_expanded = hidden_state.unsqueeze(1).expand(-1, C, -1)
        t_expanded = t_emb.unsqueeze(1).expand(-1, C, -1)

        # Concatenate: [hidden; noisy_action; timestep]
        cond = torch.cat([h_expanded, noisy_actions, t_expanded], dim=-1)  # (B, C, hidden+A+hidden)
        noise_pred = self.noise_predictor(cond)  # (B, C, A)

        return torch.nn.functional.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def forward(self, hidden_state: Tensor) -> Tensor:
        """Inference: generate actions via DDIM denoising.

        Args:
            hidden_state: (B, hidden_dim) from action token

        Returns:
            actions: (B, chunk_size, action_dim)
        """
        B = hidden_state.shape[0]
        device = hidden_state.device
        dtype = hidden_state.dtype

        # Start from pure noise
        actions = torch.randn(B, self.chunk_size, self.action_dim, device=device, dtype=dtype)

        # Set inference timesteps
        self.noise_scheduler.set_timesteps(self.num_infer_steps, device=device)

        for t in self.noise_scheduler.timesteps:
            t_batch = t.expand(B).to(device)
            t_emb = self.time_encoder(t_batch.float()).to(dtype=dtype)

            h_expanded = hidden_state.unsqueeze(1).expand(-1, self.chunk_size, -1)
            t_expanded = t_emb.unsqueeze(1).expand(-1, self.chunk_size, -1)

            cond = torch.cat([h_expanded, actions, t_expanded], dim=-1)
            noise_pred = self.noise_predictor(cond)

            actions = self.noise_scheduler.step(noise_pred, t, actions).prev_sample

        return actions


# ─── Factory ────────────────────────────────────────────────────────

ACTION_HEAD_REGISTRY = {
    "mlp": MLPActionHead,
    "resnet": ResNetActionHead,
    "diffusion": DiffusionActionHead,
}


def build_action_head(
    head_type: str,
    hidden_dim: int,
    action_dim: int,
    chunk_size: int,
    mlp_hidden: int = 2048,
    **kwargs,
) -> nn.Module:
    """Factory to build an action head by name.

    Args:
        head_type: "mlp", "resnet", or "diffusion"
        hidden_dim: VLM hidden state dimension
        action_dim: action vector dimension
        chunk_size: number of actions in a chunk
        mlp_hidden: MLP hidden layer dimension
        **kwargs: extra args passed to the head constructor

    Returns:
        An action head module.
    """
    if head_type not in ACTION_HEAD_REGISTRY:
        raise ValueError(f"Unknown action head '{head_type}'. Choose from: {list(ACTION_HEAD_REGISTRY.keys())}")

    cls = ACTION_HEAD_REGISTRY[head_type]
    return cls(
        hidden_dim=hidden_dim,
        action_dim=action_dim,
        chunk_size=chunk_size,
        mlp_hidden=mlp_hidden,
        **kwargs,
    )
