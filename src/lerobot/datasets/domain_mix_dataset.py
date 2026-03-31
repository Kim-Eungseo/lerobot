"""Multi-domain dataset that mixes samples from heterogeneous LeRobot datasets.

Handles feature alignment (e.g. different action key names), per-dataset
augmentation configs, and weighted domain sampling for natural domain
randomization within each training batch.
"""

import logging
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import datasets
import torch
import torch.utils.data

from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import VideoFrame

logger = logging.getLogger(__name__)


@dataclass
class DomainDatasetConfig:
    """Config for one domain in the mix."""

    name: str
    root: str
    action_key: str = "action"  # key to rename → "action"
    weight: float = 1.0  # sampling weight
    bg_augment_enable: bool = False
    episodes: list[int] | None = None


class DomainMixDataset(torch.utils.data.Dataset):
    """Wraps multiple LeRobotDatasets with feature alignment and domain mixing.

    Each sample is drawn from a random domain (proportional to weights),
    with features remapped to a common schema. Background augmentation is
    applied per-domain based on mask availability.
    """

    def __init__(
        self,
        domain_configs: list[DomainDatasetConfig],
        image_transforms: Callable | None = None,
        bg_augment=None,
        domain_randomization=None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
        video_backend: str | None = None,
    ):
        super().__init__()
        self._domains: list[dict] = []
        self._datasets: list[LeRobotDataset] = []
        self._action_keys: list[str] = []
        self._weights: list[float] = []

        for dc in domain_configs:
            root = Path(dc.root)
            # Per-domain bg_augment: only enable if both global config and domain flag are set
            ds_bg_augment = bg_augment if dc.bg_augment_enable else None

            ds = LeRobotDataset(
                repo_id=dc.name,
                root=str(root),
                episodes=dc.episodes,
                image_transforms=image_transforms,
                bg_augment=ds_bg_augment,
                domain_randomization=domain_randomization,
                delta_timestamps=delta_timestamps,
                tolerance_s=tolerance_s,
                video_backend=video_backend,
            )
            self._datasets.append(ds)
            self._action_keys.append(dc.action_key)
            self._weights.append(dc.weight)
            self._domains.append({
                "name": dc.name,
                "root": str(root),
                "action_key": dc.action_key,
                "num_frames": ds.num_frames,
            })
            logger.info(
                f"Loaded domain '{dc.name}': {ds.num_frames} frames, "
                f"{ds.num_episodes} episodes, action_key='{dc.action_key}'"
            )

        # Build cumulative lengths for index mapping
        self._cum_lengths = []
        total = 0
        for ds in self._datasets:
            total += ds.num_frames
            self._cum_lengths.append(total)

        # Normalize weights for weighted sampling
        total_weight = sum(self._weights)
        self._norm_weights = [w / total_weight for w in self._weights]

        # Aggregate stats (using first common action key stats as "action")
        all_stats = []
        for ds, action_key in zip(self._datasets, self._action_keys):
            stats = dict(ds.meta.stats)
            if action_key != "action" and action_key in stats:
                stats["action"] = stats.pop(action_key)
            all_stats.append(stats)
        self.stats = aggregate_stats(all_stats)

    @property
    def meta(self):
        """Return first dataset's meta for compatibility (camera_keys, fps, etc)."""
        return self._datasets[0].meta

    @property
    def fps(self) -> int:
        return self._datasets[0].meta.fps

    @property
    def num_frames(self) -> int:
        return self._cum_lengths[-1] if self._cum_lengths else 0

    @property
    def num_episodes(self) -> int:
        return sum(ds.num_episodes for ds in self._datasets)

    @property
    def features(self):
        return self._datasets[0].features

    @property
    def camera_keys(self) -> list[str]:
        return self._datasets[0].meta.camera_keys

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Map flat index to dataset + local index
        ds_idx = 0
        for i, cum_len in enumerate(self._cum_lengths):
            if idx < cum_len:
                ds_idx = i
                break
        local_idx = idx - (self._cum_lengths[ds_idx - 1] if ds_idx > 0 else 0)

        item = self._datasets[ds_idx][local_idx]

        # Remap action key if needed
        action_key = self._action_keys[ds_idx]
        if action_key != "action" and action_key in item:
            item["action"] = item.pop(action_key)
            # Also remap padding key
            pad_key = f"{action_key}_is_pad"
            if pad_key in item:
                item["action_is_pad"] = item.pop(pad_key)

        # Remove extra action keys from sim datasets
        for key in list(item.keys()):
            if key.startswith("action.") and key != "action":
                del item[key]
            if key.endswith("_is_pad") and key.startswith("action."):
                del item[key]

        item["dataset_index"] = torch.tensor(ds_idx)
        return item

    def __repr__(self):
        lines = [f"{self.__class__.__name__}("]
        for d in self._domains:
            lines.append(f"  {d['name']}: {d['num_frames']} frames, action_key='{d['action_key']}'")
        lines.append(f"  Total: {self.num_frames} frames, {self.num_episodes} episodes")
        lines.append(")")
        return "\n".join(lines)
