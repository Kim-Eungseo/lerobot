#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Private reader component for LeRobotDataset. Handles random-access reading (HF dataset, delta indices, video decoding)."""

from collections.abc import Callable
from pathlib import Path

import datasets
import numpy as np
import torch

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.feature_utils import (
    check_delta_timestamps,
    get_delta_indices,
    get_hf_features_from_features,
)
from lerobot.datasets.io_utils import (
    hf_transform_to_torch,
    load_nested_dataset,
)
from lerobot.datasets.video_utils import decode_video_frames


class DatasetReader:
    """Encapsulates read-side state and methods for LeRobotDataset.

    Owns: hf_dataset, _absolute_to_relative_idx, delta_indices.
    """

    def __init__(
        self,
        meta: LeRobotDatasetMetadata,
        root: Path,
        episodes: list[int] | None,
        tolerance_s: float,
        video_backend: str,
        delta_timestamps: dict[str, list[float]] | None,
        image_transforms: Callable | None,
        bg_augment=None,
        domain_randomization=None,
    ):
        """Initialize the reader with metadata, filtering, and transform config.

        The HF dataset is not loaded here — call :meth:`try_load` or
        :meth:`load_and_activate` afterward.

        Args:
            meta: Dataset metadata instance.
            root: Local dataset root directory.
            episodes: Optional list of episode indices to select. ``None``
                means all episodes.
            tolerance_s: Timestamp synchronization tolerance in seconds.
            video_backend: Video decoding backend identifier.
            delta_timestamps: Optional dict mapping feature keys to lists of
                relative timestamp offsets for temporal context windows.
            image_transforms: Optional torchvision v2 transform applied to
                visual features.
            bg_augment: Optional BgAugmentConfig for online background compositing.
            domain_randomization: Optional DomainRandomizationConfig for online augmentations.
        """
        self._meta = meta
        self.root = root
        self.episodes = episodes
        self._tolerance_s = tolerance_s
        self._video_backend = video_backend
        self._image_transforms = image_transforms

        # BgCompositor initialization (textures cached once at init time)
        self._bg_compositor = None
        self._bg_mask_subdir = "masks"
        if bg_augment is not None and bg_augment.enable:
            from lerobot.datasets.augmentation.bg_compositing import BgCompositor

            self._bg_compositor = BgCompositor(
                texture_dir=bg_augment.texture_dir,
                bg_mode=bg_augment.bg_mode,
                p_bg=bg_augment.p_bg,
                resolution=bg_augment.texture_resolution,
            )
            self._bg_mask_subdir = bg_augment.mask_subdir

        # Domain randomization initialization
        self._domain_randomization = None
        if domain_randomization is not None and domain_randomization.enable:
            from lerobot.datasets.augmentation.domain_randomization import DomainRandomization

            self._domain_randomization = DomainRandomization(
                enable_lighting=domain_randomization.enable_lighting,
                enable_noise=domain_randomization.enable_noise,
                enable_crop=domain_randomization.enable_crop,
                p=domain_randomization.p,
                lighting_gain_range=domain_randomization.lighting_gain_range,
                noise_iso_range=domain_randomization.noise_iso_range,
                crop_ratios=domain_randomization.crop_ratios,
            )

        self.hf_dataset: datasets.Dataset | None = None
        self._absolute_to_relative_idx: dict[int, int] | None = None

        # Setup delta_indices (doesn't depend on hf_dataset)
        self.delta_indices = None
        if delta_timestamps is not None:
            check_delta_timestamps(delta_timestamps, meta.fps, tolerance_s)
            self.delta_indices = get_delta_indices(delta_timestamps, meta.fps)

    def try_load(self) -> bool:
        """Attempt to load from local cache. Returns True if data is sufficient."""
        try:
            self.hf_dataset = self._load_hf_dataset()
        except (FileNotFoundError, NotADirectoryError):
            self.hf_dataset = None
            return False
        if not self._check_cached_episodes_sufficient():
            self.hf_dataset = None
            return False
        self._build_index_mapping()
        return True

    def load_and_activate(self) -> None:
        """Load HF dataset from disk and build index mapping. Call after data is on disk."""
        self.hf_dataset = self._load_hf_dataset()
        self._build_index_mapping()

    def _build_index_mapping(self) -> None:
        """Build absolute-to-relative index mapping from loaded hf_dataset."""
        self._absolute_to_relative_idx = None
        if self.episodes is not None and self.hf_dataset is not None:
            self._absolute_to_relative_idx = {
                abs_idx.item() if isinstance(abs_idx, torch.Tensor) else abs_idx: rel_idx
                for rel_idx, abs_idx in enumerate(self.hf_dataset["index"])
            }

    @property
    def num_frames(self) -> int:
        """Number of frames in selected episodes."""
        if self.episodes is not None and self.hf_dataset is not None:
            return len(self.hf_dataset)
        return self._meta.total_frames

    @property
    def num_episodes(self) -> int:
        """Number of episodes selected."""
        return len(self.episodes) if self.episodes is not None else self._meta.total_episodes

    def _load_hf_dataset(self) -> datasets.Dataset:
        """hf_dataset contains all the observations, states, actions, rewards, etc."""
        features = get_hf_features_from_features(self._meta.features)
        hf_dataset = load_nested_dataset(self.root / "data", features=features, episodes=self.episodes)
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    def _check_cached_episodes_sufficient(self) -> bool:
        """Check if the cached dataset contains all requested episodes and their video files."""
        if self.hf_dataset is None or len(self.hf_dataset) == 0:
            return False

        available_episodes = {
            ep_idx.item() if isinstance(ep_idx, torch.Tensor) else ep_idx
            for ep_idx in self.hf_dataset.unique("episode_index")
        }

        if self.episodes is None:
            requested_episodes = set(range(self._meta.total_episodes))
        else:
            requested_episodes = set(self.episodes)

        if not requested_episodes.issubset(available_episodes):
            return False

        if len(self._meta.video_keys) > 0:
            for ep_idx in requested_episodes:
                for vid_key in self._meta.video_keys:
                    video_path = self.root / self._meta.get_video_file_path(ep_idx, vid_key)
                    if not video_path.exists():
                        return False

        return True

    def get_episodes_file_paths(self) -> list[Path]:
        """Return deduplicated file paths (data + video) for selected episodes.

        Used to build the ``allow_patterns`` list for ``snapshot_download``.
        """
        episodes = self.episodes if self.episodes is not None else list(range(self._meta.total_episodes))
        fpaths = [str(self._meta.get_data_file_path(ep_idx)) for ep_idx in episodes]
        if len(self._meta.video_keys) > 0:
            video_files = [
                str(self._meta.get_video_file_path(ep_idx, vid_key))
                for vid_key in self._meta.video_keys
                for ep_idx in episodes
            ]
            fpaths += video_files
        # episodes are stored in the same files, so we return unique paths only
        fpaths = list(set(fpaths))
        return fpaths

    def _get_query_indices(
        self, abs_idx: int, ep_idx: int
    ) -> tuple[dict[str, list[int]], dict[str, torch.Tensor]]:
        """Compute query indices for delta timestamps."""
        ep = self._meta.episodes[ep_idx]
        ep_start = ep["dataset_from_index"]
        ep_end = ep["dataset_to_index"]
        query_indices = {
            key: [max(ep_start, min(ep_end - 1, abs_idx + delta)) for delta in delta_idx]
            for key, delta_idx in self.delta_indices.items()
        }
        padding = {
            f"{key}_is_pad": torch.BoolTensor(
                [(abs_idx + delta < ep_start) | (abs_idx + delta >= ep_end) for delta in delta_idx]
            )
            for key, delta_idx in self.delta_indices.items()
        }
        return query_indices, padding

    def _get_query_timestamps(
        self,
        current_ts: float,
        query_indices: dict[str, list[int]] | None = None,
    ) -> dict[str, list[float]]:
        query_timestamps = {}
        for key in self._meta.video_keys:
            if query_indices is not None and key in query_indices:
                if self._absolute_to_relative_idx is not None:
                    relative_indices = [self._absolute_to_relative_idx[idx] for idx in query_indices[key]]
                    timestamps = self.hf_dataset[relative_indices]["timestamp"]
                else:
                    timestamps = self.hf_dataset[query_indices[key]]["timestamp"]
                query_timestamps[key] = torch.stack(timestamps).tolist()
            else:
                query_timestamps[key] = [current_ts]

        return query_timestamps

    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
        """Query dataset for indices across keys, skipping video keys."""
        result: dict = {}
        for key, q_idx in query_indices.items():
            if key in self._meta.video_keys:
                continue
            relative_indices = (
                q_idx
                if self._absolute_to_relative_idx is None
                else [self._absolute_to_relative_idx[idx] for idx in q_idx]
            )
            try:
                result[key] = torch.stack(self.hf_dataset[key][relative_indices])
            except (KeyError, TypeError, IndexError):
                result[key] = torch.stack(self.hf_dataset[relative_indices][key])
        return result

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict[str, torch.Tensor]:
        """Note: When using data workers (e.g. DataLoader with num_workers>0), do not call this function
        in the main process (e.g. by using a second Dataloader with num_workers=0). It will result in a
        Segmentation Fault.
        """
        ep = self._meta.episodes[ep_idx]
        item = {}
        for vid_key, query_ts in query_timestamps.items():
            from_timestamp = ep[f"videos/{vid_key}/from_timestamp"]
            shifted_query_ts = [from_timestamp + ts for ts in query_ts]

            video_path = self.root / self._meta.get_video_file_path(ep_idx, vid_key)
            frames = decode_video_frames(video_path, shifted_query_ts, self._tolerance_s, self._video_backend)
            item[vid_key] = frames.squeeze(0)

        return item

    def _get_mask_path(self, ep_idx: int) -> Path:
        """Return masks.npz path for the given episode index."""
        return self.root / self._bg_mask_subdir / f"ep{ep_idx:06d}" / "masks.npz"

    def _get_frame_indices(self, query_ts: list[float], ep_idx: int) -> list[int]:
        """Convert query timestamps to masks.npz frame indices."""
        ep = self._meta.episodes[ep_idx]
        first_vid_key = self._meta.video_keys[0]
        from_ts = ep[f"videos/{first_vid_key}/from_timestamp"]
        fps = self._meta.fps
        return [max(0, round((ts - from_ts) * fps)) for ts in query_ts]

    def _load_masks(
        self,
        mask_path: Path,
        frame_indices: list[int],
        expected_T: int,
    ) -> dict[str, np.ndarray]:
        """Load and slice masks from masks.npz for given frame indices.

        Returns:
            {"table": (T,H,W), "floor": (T,H,W), "wall": (T,H,W)} uint8.
            Only includes regions present in the file. Returns {} on error.
        """
        try:
            data = np.load(mask_path)
        except Exception:
            return {}

        result = {}
        for region in ("table", "floor", "wall"):
            if region not in data:
                continue
            arr = data[region]  # (T_full, H, W)
            T_max = arr.shape[0]
            safe_indices = [min(i, T_max - 1) for i in frame_indices]
            result[region] = arr[safe_indices]  # (T, H, W)
        return result

    def get_item(self, idx) -> dict:
        """Core __getitem__ logic. Assumes hf_dataset is loaded.

        ``idx`` is a *relative* index into the (possibly episode-filtered)
        HF dataset, **not** the absolute frame index stored in the ``index``
        column.  The absolute index is retrieved from the row itself.
        """
        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()
        abs_idx = item["index"].item()

        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(abs_idx, ep_idx)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        if len(self._meta.video_keys) > 0:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            video_frames = self._query_videos(query_timestamps, ep_idx)

            # ── step 3.5: online_bg compositing ──
            if self._bg_compositor is not None:
                mask_path = self._get_mask_path(ep_idx)
                if mask_path.exists():
                    for cam in self._meta.camera_keys:
                        if cam not in video_frames:
                            continue
                        frames_t = video_frames[cam]  # float32 [0,1]
                        single = frames_t.ndim == 3  # (C,H,W) single frame
                        if single:
                            frames_t = frames_t.unsqueeze(0)  # (1,C,H,W)

                        T = frames_t.shape[0]
                        frame_indices = self._get_frame_indices(query_timestamps[cam], ep_idx)
                        masks = self._load_masks(mask_path, frame_indices, T)
                        if masks:
                            # float [0,1] (T,C,H,W) → uint8 (T,H,W,C)
                            frames_u8 = (
                                frames_t.permute(0, 2, 3, 1).mul(255).byte().numpy()
                            )
                            frames_u8 = self._bg_compositor.apply(frames_u8, masks)
                            # uint8 (T,H,W,C) → float [0,1] (T,C,H,W)
                            frames_t = (
                                torch.from_numpy(frames_u8).permute(0, 3, 1, 2).float().div(255.0)
                            )
                            if single:
                                frames_t = frames_t.squeeze(0)
                            video_frames[cam] = frames_t

            # ── step 3.6: online domain randomization (lighting/noise/crop) ──
            if self._domain_randomization is not None:
                for cam in self._meta.camera_keys:
                    if cam not in video_frames:
                        continue
                    frames_t = video_frames[cam]
                    single = frames_t.ndim == 3
                    if single:
                        frames_t = frames_t.unsqueeze(0)
                    frames_u8 = frames_t.permute(0, 2, 3, 1).mul(255).byte().numpy()
                    frames_u8 = self._domain_randomization(frames_u8)
                    frames_t = torch.from_numpy(frames_u8).permute(0, 3, 1, 2).float().div(255.0)
                    if single:
                        frames_t = frames_t.squeeze(0)
                    video_frames[cam] = frames_t

            item = {**video_frames, **item}

        if self._image_transforms is not None:
            image_keys = self._meta.camera_keys
            for cam in image_keys:
                item[cam] = self._image_transforms(item[cam])

        # Add task as a string
        task_idx = item["task_index"].item()
        item["task"] = self._meta.tasks.iloc[task_idx].name

        # add subtask information if available
        if "subtask_index" in self._meta.features and self._meta.subtasks is not None:
            subtask_idx = item["subtask_index"].item()
            item["subtask"] = self._meta.subtasks.iloc[subtask_idx].name

        return item
