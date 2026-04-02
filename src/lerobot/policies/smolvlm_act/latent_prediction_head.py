"""Latent Prediction Head for Stage 2 training.

Maps the learnable action token's hidden state to V-JEPA2 latent space,
matching the output of the frozen V-JEPA2 task encoder + AttentivePooler.
"""

import torch.nn as nn


class MLPResNetBlock(nn.Module):
    """Pre-LayerNorm residual MLP block."""

    def __init__(self, dim: int):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return x + self.ffn(x)


class LatentPredictionHead(nn.Module):
    """Projects action token hidden state to V-JEPA2 latent space.

    Architecture: LayerNorm -> Linear -> ReLU -> ResBlock x N -> LayerNorm -> Linear

    Args:
        input_dim: VLM hidden state dimension (e.g., 1536 for SmolVLM2-500M)
        hidden_dim: MLP hidden dimension
        proj_dim: output dimension (must match V-JEPA2 task_head proj_dim, typically 256)
        num_blocks: number of residual MLP blocks
    """

    def __init__(self, input_dim: int = 1536, hidden_dim: int = 1536, proj_dim: int = 256, num_blocks: int = 2):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.blocks = nn.ModuleList([MLPResNetBlock(hidden_dim) for _ in range(num_blocks)])
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, proj_dim)

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: [B, input_dim] - action token hidden state from VLM

        Returns:
            z_pred: [B, proj_dim] - predicted V-JEPA2 latent representation
        """
        x = self.layer_norm1(hidden_states)
        x = self.relu(self.fc1(x))
        for block in self.blocks:
            x = block(x)
        x = self.layer_norm2(x)
        return self.fc2(x)
