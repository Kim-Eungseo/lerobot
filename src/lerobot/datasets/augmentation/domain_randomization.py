"""Online domain randomization augmentations for visual inputs.

Ported from vjepa2's augmentation pipeline. Each augmentation operates on
uint8 (T, H, W, 3) numpy arrays with temporal consistency (same params for all frames).
"""

import random

import cv2
import numpy as np


class LightingAugmentation:
    """Per-channel RGB gain to simulate lighting / color temperature changes."""

    def __init__(self, gain_range: tuple[float, float] = (0.3, 2.0)):
        self.gain_range = gain_range

    def __call__(self, frames: np.ndarray, gain: np.ndarray | None = None) -> np.ndarray:
        """
        Args:
            frames: (T, H, W, 3) uint8 RGB
            gain: optional (3,) float array for reproducibility
        Returns:
            (T, H, W, 3) uint8
        """
        if gain is None:
            lo, hi = self.gain_range
            gain = np.random.uniform(lo, hi, size=3).astype(np.float32)
        result = frames.astype(np.float32) / 255.0
        result = result * gain.reshape(1, 1, 1, 3)
        return (result.clip(0, 1) * 255).astype(np.uint8)


# ISO noise model parameters
_ISO_RANGES = {
    1: (200, 400),
    2: (400, 800),
    3: (800, 1600),
    4: (1600, 3200),
    5: (3200, 6400),
}


class SensorNoiseAugmentation:
    """Physics-based camera sensor noise (shot + read noise)."""

    def __init__(
        self,
        iso_level_range: tuple[int, int] = (1, 5),
        gamma_range: tuple[float, float] = (1.8, 2.6),
    ):
        self.iso_level_range = iso_level_range
        self.gamma_range = gamma_range

    def __call__(self, frames: np.ndarray, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random

        iso_level = rng.randint(self.iso_level_range[0], self.iso_level_range[1] + 1)
        iso_lo, iso_hi = _ISO_RANGES[iso_level]
        iso = rng.uniform(iso_lo, iso_hi)

        # Noise coefficients scale with ISO
        shot_coeff = iso / 6400.0 * 0.02
        read_coeff = iso / 6400.0 * 0.005
        gamma = rng.uniform(*self.gamma_range)
        channel_gain = rng.uniform(0.85, 1.15, size=3).astype(np.float32)

        result = frames.astype(np.float32) / 255.0

        # Shared noise seed for temporal consistency
        frame_seed = rng.randint(0, 2**31)

        for i in range(len(result)):
            rs = np.random.RandomState(frame_seed + i)
            linear = np.power(result[i], gamma)
            noise_var = shot_coeff * linear + read_coeff**2
            noise_std = np.sqrt(np.maximum(noise_var, 1e-10))
            noise = rs.normal(0.0, noise_std).astype(np.float32)
            noisy = linear + noise * channel_gain.reshape(1, 1, 3)
            result[i] = np.power(np.clip(noisy, 0, 1), 1.0 / gamma)

        return (result.clip(0, 1) * 255).astype(np.uint8)


# Adjacent corner combinations for crop
_ADJACENT_CORNERS = [
    ("top", "left"),
    ("top", "right"),
    ("bottom", "left"),
    ("bottom", "right"),
]


class PerEdgeCropAugmentation:
    """Random corner crop with resize back to original size."""

    def __init__(self, crop_ratios: list[float] | None = None):
        self.crop_ratios = crop_ratios or [0.05, 0.10]

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        T, H, W, C = frames.shape
        corner = _ADJACENT_CORNERS[np.random.randint(len(_ADJACENT_CORNERS))]
        r1 = np.random.choice(self.crop_ratios)
        r2 = np.random.choice(self.crop_ratios)

        top = int(H * r1) if "top" in corner else 0
        bot = int(H * r1) if "bottom" in corner else 0
        lft = int(W * r2) if "left" in corner else 0
        rgt = int(W * r2) if "right" in corner else 0

        bot_idx = H - bot if bot > 0 else H
        rgt_idx = W - rgt if rgt > 0 else W
        cropped = frames[:, top:bot_idx, lft:rgt_idx]

        result = np.empty((T, H, W, C), dtype=np.uint8)
        for i in range(T):
            result[i] = cv2.resize(cropped[i], (W, H))
        return result


class DomainRandomization:
    """Composite domain randomization pipeline.

    Randomly selects one augmentation per sample call, providing natural
    domain diversity within each training batch.

    Args:
        enable_lighting: Enable RGB gain augmentation
        enable_noise: Enable sensor noise augmentation
        enable_crop: Enable per-edge crop augmentation
        p: Probability of applying any augmentation per sample
        lighting_gain_range: (lo, hi) for RGB channel gain
        noise_iso_range: (lo_level, hi_level) ISO noise levels 1-5
    """

    def __init__(
        self,
        enable_lighting: bool = True,
        enable_noise: bool = True,
        enable_crop: bool = True,
        p: float = 0.8,
        lighting_gain_range: tuple[float, float] = (0.3, 2.0),
        noise_iso_range: tuple[int, int] = (1, 5),
        crop_ratios: list[float] | None = None,
    ):
        self.p = p
        self._augmentations: list[tuple[str, object]] = []
        if enable_lighting:
            self._augmentations.append(("lighting", LightingAugmentation(lighting_gain_range)))
        if enable_noise:
            self._augmentations.append(("noise", SensorNoiseAugmentation(noise_iso_range)))
        if enable_crop:
            self._augmentations.append(("crop", PerEdgeCropAugmentation(crop_ratios)))

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        """Apply a random augmentation to frames.

        Args:
            frames: (T, H, W, 3) uint8 RGB
        Returns:
            (T, H, W, 3) uint8
        """
        if not self._augmentations or random.random() > self.p:
            return frames

        name, aug = random.choice(self._augmentations)
        return aug(frames)
