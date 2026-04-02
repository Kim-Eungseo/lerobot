"""Stage 2 Dataset utilities for SmolVLM-Act + V-JEPA2 alignment training.

Provides:
  - build_stage2_delta_timestamps: extends delta_timestamps to load N context + K future frames
  - Stage2DatasetWrapper: splits loaded frames into context (SmolVLM-Act) and future (V-JEPA2)
  - build_vjepa_transform: V-JEPA2 ImageNet normalization transform
"""

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

# V-JEPA2 uses ImageNet normalization
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def build_vjepa_transform(img_size: int = 256):
    """Build V-JEPA2 image transform: resize + center crop + ImageNet normalization.

    Expects input tensors in [0, 1] float range (as returned by LeRobot video decoding).
    """
    return T.Compose([
        T.Resize(img_size, interpolation=T.InterpolationMode.BILINEAR, antialias=True),
        T.CenterCrop(img_size),
        T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])


def build_stage2_delta_timestamps(
    action_key: str,
    fps: int,
    chunk_size: int = 50,
    num_future_frames: int = 8,
):
    """Build delta_timestamps that load 1 context + K future frames.

    The observation image key will load [0, 1/fps, 2/fps, ..., K/fps] timestamps,
    giving 1+K consecutive frames. The first frame is the context frame, the rest
    are future frames for V-JEPA2 target computation.

    Args:
        action_key: action key in dataset (e.g., "action.ee_delta_pose")
        fps: dataset frames per second
        chunk_size: action chunk size
        num_future_frames: K, number of future frames for V-JEPA2
    """
    dt = {}
    # 1 context + K future frames
    dt["observation.images.anchor"] = [i / fps for i in range(1 + num_future_frames)]
    # Actions (same as standard training)
    dt[action_key] = [i / fps for i in range(chunk_size)]
    return dt


class Stage2DatasetWrapper(Dataset):
    """Wraps a LeRobot dataset to split frames into context and future for Stage 2.

    The base dataset should be loaded with delta_timestamps from
    build_stage2_delta_timestamps, so observation.images.anchor contains
    [1+K, C, H, W] frames (or after camera key remapping, observation.images.camera1).

    This wrapper:
    1. Keeps the first frame as context (for SmolVLM-Act, unchanged)
    2. Extracts remaining K frames and applies V-JEPA2 transform as future_pixel_values
    """

    def __init__(self, base_dataset, num_future_frames: int = 8, vjepa_img_size: int = 256):
        self._ds = base_dataset
        self.num_future_frames = num_future_frames
        self.vjepa_transform = build_vjepa_transform(vjepa_img_size)

    @property
    def num_frames(self):
        return self._ds.num_frames

    @property
    def num_episodes(self):
        return self._ds.num_episodes

    @property
    def meta(self):
        return self._ds.meta

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        item = self._ds[idx]

        # Find the camera key that has multi-frame data
        cam_key = None
        for key in list(item.keys()):
            if key.startswith("observation.images.") and not key.endswith("_is_pad"):
                if isinstance(item[key], torch.Tensor) and item[key].ndim == 4:
                    cam_key = key
                    break

        if cam_key is None:
            raise ValueError(
                "No multi-frame camera key found. Ensure delta_timestamps loads 1+K frames. "
                f"Available keys: {list(item.keys())}"
            )

        all_frames = item[cam_key]  # [1+K, C, H, W]
        expected = 1 + self.num_future_frames
        if all_frames.shape[0] < expected:
            raise ValueError(
                f"Expected {expected} frames but got {all_frames.shape[0]}. "
                f"Check delta_timestamps for {cam_key}."
            )

        # Split: context (first frame) and future (remaining K frames)
        item[cam_key] = all_frames[0]  # [C, H, W] - context for SmolVLM-Act

        # Future frames -> V-JEPA2 transform (resize to vjepa_img_size, ImageNet norm)
        future_frames = all_frames[1:expected]  # [K, C, H, W], float [0, 1]
        future_transformed = torch.stack([
            self.vjepa_transform(f) for f in future_frames
        ])  # [K, 3, vjepa_img_size, vjepa_img_size]
        item["future_pixel_values"] = future_transformed

        # Handle is_pad key if present
        pad_key = f"{cam_key}_is_pad"
        if pad_key in item:
            pad_vals = item[pad_key]
            if isinstance(pad_vals, torch.Tensor) and pad_vals.ndim >= 1:
                item["future_is_pad"] = pad_vals[1:expected]
                item[pad_key] = pad_vals[0]

        return item
