#!/usr/bin/env python3
"""Domain-mix training for SmolVLA with 4 heterogeneous datasets.

Combines:
- 2 simulation datasets (maniskill-franka, maniskill-xarm) with bg_augment
- 2 real-world datasets (oxe-bridge, oxe-fractal) without bg_augment
- Online domain randomization (lighting, noise, crop) for all

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/train_domain_mix.py
"""

import dataclasses
import logging
import time
from contextlib import nullcontext
from pathlib import Path
from pprint import pformat

import torch
from tqdm import tqdm

# Register all policy plugins before importing anything else
from lerobot.utils.import_utils import register_third_party_plugins
register_third_party_plugins()

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.domain_mix_dataset import DomainDatasetConfig
from lerobot.datasets.factory import IMAGENET_STATS
from lerobot.datasets.transforms import (
    BgAugmentConfig,
    DomainRandomizationConfig,
    ImageTransforms,
    ImageTransformsConfig,
)
from lerobot.datasets.utils import cycle
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.utils.logging_utils import AverageMeter
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import format_big_number, has_method, init_logging


# ─── Configuration ──────────────────────────────────────────────────
DATA_ROOT = "/data/lerobot"
TEXTURE_DIR = "/home/ngseo/vjepa2/bggen/outputs"  # wall/, table/, floor/ subdirs

DOMAINS = [
    # Simulation (masks available → bg_augment on)
    # 6:1:2:1 ratio
    DomainDatasetConfig(
        name="maniskill-franka",
        root=f"{DATA_ROOT}/maniskill-franka",
        action_key="action.ee_delta_pose",
        weight=6.0,
        bg_augment_enable=True,
    ),
    DomainDatasetConfig(
        name="maniskill-xarm",
        root=f"{DATA_ROOT}/maniskill-xarm",
        action_key="action.ee_delta_pose",
        weight=1.0,
        bg_augment_enable=True,
    ),
    DomainDatasetConfig(
        name="oxe-bridge",
        root=f"{DATA_ROOT}/oxe-bridge",
        action_key="action",
        weight=2.0,
        bg_augment_enable=False,
    ),
    DomainDatasetConfig(
        name="oxe-fractal",
        root=f"{DATA_ROOT}/oxe-fractal",
        action_key="action",
        weight=1.0,
        bg_augment_enable=False,
    ),
]

# Training hyperparams
POLICY_PATH = "lerobot/smolvla_base"
BATCH_SIZE = 8
STEPS = 50_000
NUM_WORKERS = 4
SEED = 42
LOG_FREQ = 50
SAVE_FREQ = 10_000
LR = 5e-5
OUTPUT_DIR = Path("outputs/train/domain_mix_smolvla")


def build_delta_timestamps(action_key: str, fps: int, chunk_size: int = 50) -> dict[str, list[float]]:
    """Build delta_timestamps with the correct action key for each dataset."""
    dt = {}
    # observation: single current frame
    dt["observation.images.anchor"] = [0.0]
    # action: chunk_size future steps
    dt[action_key] = [i / fps for i in range(chunk_size)]
    return dt


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    device = torch.device("cuda")

    if SEED is not None:
        torch.manual_seed(SEED)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # ── Augmentation configs ──
    image_transforms_cfg = ImageTransformsConfig(
        enable=True,
        max_num_transforms=2,
    )
    image_transforms = ImageTransforms(image_transforms_cfg)

    bg_augment = BgAugmentConfig(
        enable=True,
        texture_dir=TEXTURE_DIR,
        p_bg=0.8,
        bg_mode="both",
        mask_subdir="masks",
        texture_resolution=256,
    )

    domain_randomization = DomainRandomizationConfig(
        enable=True,
        p=0.7,
        enable_lighting=True,
        enable_noise=True,
        enable_crop=True,
        lighting_gain_range=(0.3, 2.0),
        noise_iso_range=(1, 4),
    )

    # ── Load policy config to get chunk_size ──
    import lerobot.policies.smolvla  # noqa: F401 - register smolvla
    policy_cfg = PreTrainedConfig.from_pretrained(POLICY_PATH)
    # Train from scratch (pretrained weights cause NaN with different input schema)
    policy_cfg.pretrained_path = None
    policy_cfg.load_vlm_weights = True  # Still load VLM backbone weights
    chunk_size = policy_cfg.chunk_size

    # Configure for our data: 1 camera, no state, 7-dim action
    policy_cfg.empty_cameras = 2  # Pad 2 missing camera slots
    policy_cfg.train_state_proj = False  # No state in our data

    # ── Build per-domain datasets ──
    logging.info("Loading datasets...")
    datasets_list = []
    all_stats = []

    for dc in DOMAINS:
        from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

        root = Path(dc.root)
        meta = LeRobotDatasetMetadata(dc.name, root=str(root))
        fps = meta.fps

        # Per-domain delta_timestamps with correct action key
        delta_timestamps = build_delta_timestamps(dc.action_key, fps, chunk_size)

        # Per-domain bg_augment
        ds_bg_augment = bg_augment if dc.bg_augment_enable else None

        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        ds = LeRobotDataset(
            repo_id=dc.name,
            root=str(root),
            episodes=dc.episodes,
            image_transforms=image_transforms,
            bg_augment=ds_bg_augment,
            domain_randomization=domain_randomization,
            delta_timestamps=delta_timestamps,
            tolerance_s=0.02,  # Slightly generous for floating point timestamp precision
            video_backend="pyav",
        )
        datasets_list.append((dc, ds))

        # Remap stats action key
        stats = dict(ds.meta.stats)
        if dc.action_key != "action" and dc.action_key in stats:
            stats["action"] = stats[dc.action_key]
        all_stats.append(stats)

        logging.info(
            f"  {dc.name}: {ds.num_frames} frames, {ds.num_episodes} eps, "
            f"fps={fps}, action_key='{dc.action_key}', bg_augment={dc.bg_augment_enable}"
        )

    # ── Build combined dataset ──
    class CombinedDomainDataset(torch.utils.data.Dataset):
        """Concatenates multiple LeRobotDatasets with feature alignment."""

        def __init__(self, domain_datasets):
            self._items = domain_datasets  # list of (DomainDatasetConfig, LeRobotDataset)
            self._cum_lengths = []
            total = 0
            for _, ds in self._items:
                total += ds.num_frames
                self._cum_lengths.append(total)

        @property
        def num_frames(self):
            return self._cum_lengths[-1] if self._cum_lengths else 0

        @property
        def num_episodes(self):
            return sum(ds.num_episodes for _, ds in self._items)

        @property
        def meta(self):
            return self._items[0][1].meta

        def __len__(self):
            return self.num_frames

        def __getitem__(self, idx):
            ds_idx = 0
            for i, cum_len in enumerate(self._cum_lengths):
                if idx < cum_len:
                    ds_idx = i
                    break
            local_idx = idx - (self._cum_lengths[ds_idx - 1] if ds_idx > 0 else 0)

            dc, ds = self._items[ds_idx]
            item = ds[local_idx]

            # Remap action key
            if dc.action_key != "action" and dc.action_key in item:
                item["action"] = item.pop(dc.action_key)
                pad_key = f"{dc.action_key}_is_pad"
                if pad_key in item:
                    item["action_is_pad"] = item.pop(pad_key)

            # Remove extra action keys
            for key in list(item.keys()):
                if key.startswith("action."):
                    del item[key]
                elif key.endswith("_is_pad") and "action." in key:
                    del item[key]

            item["dataset_index"] = torch.tensor(ds_idx)

            # Replace NaN in action padding with zeros
            action_key_out = "action" if "action" in item else dc.action_key
            if action_key_out in item and isinstance(item[action_key_out], torch.Tensor):
                item[action_key_out] = torch.nan_to_num(item[action_key_out], nan=0.0)

            # Rename camera key to match SmolVLA expectations
            if "observation.images.anchor" in item:
                item["observation.images.camera1"] = item.pop("observation.images.anchor")
            if "observation.images.anchor_is_pad" in item:
                item["observation.images.camera1_is_pad"] = item.pop("observation.images.anchor_is_pad")

            # Inject dummy observation.state if missing (SmolVLA expects it)
            if "observation.state" not in item:
                item["observation.state"] = torch.zeros(7)

            # Ensure task is a string for tokenizer
            if "task" in item and not isinstance(item["task"], str):
                # OXE datasets have int task indices; look up task description
                task_idx = item["task"]
                dc_ds = self._items[ds_idx][1]
                tasks_df = dc_ds.meta.tasks
                if task_idx < len(tasks_df) and "task" in tasks_df.columns:
                    task_str = tasks_df.iloc[task_idx]["task"]
                    item["task"] = str(task_str) if task_str else "manipulation"
                else:
                    item["task"] = "manipulation"

            # Remove non-tensor fields that can't be collated, except 'task' (string)
            for key in list(item.keys()):
                if not isinstance(item[key], (torch.Tensor, str)):
                    del item[key]

            return item

    dataset = CombinedDomainDataset(datasets_list)

    # Aggregate stats manually (built-in aggregate_stats is strict about shapes)
    # Compute conservative merged action stats across all domains
    agg_stats = {}
    action_means = []
    action_stds = []
    action_mins = []
    action_maxs = []
    for stats in all_stats:
        if "action" in stats:
            s = stats["action"]
            action_means.append(s["mean"])
            action_stds.append(s["std"])
            action_mins.append(s["min"])
            action_maxs.append(s["max"])
    if action_means:
        # Convert to tensors if needed
        def to_tensor(x):
            return x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)

        means_t = torch.stack([to_tensor(m) for m in action_means])
        stds_t = torch.stack([to_tensor(s) for s in action_stds])
        mins_t = torch.stack([to_tensor(m) for m in action_mins])
        maxs_t = torch.stack([to_tensor(m) for m in action_maxs])

        # Use grand mean/std that covers all domains
        agg_stats["action"] = {
            "mean": means_t.mean(dim=0),
            "std": stds_t.max(dim=0).values.clamp(min=0.01),  # max std for safety
            "min": mins_t.min(dim=0).values,
            "max": maxs_t.max(dim=0).values,
        }

    # Camera keys: use ImageNet stats (overwritten below anyway)
    first_meta = datasets_list[0][1].meta
    for key in first_meta.camera_keys:
        for stats_type, stats_val in IMAGENET_STATS.items():
            agg_stats.setdefault(key, {})[stats_type] = torch.tensor(stats_val, dtype=torch.float32)

    # Copy over any other shared scalar stats (reward, etc)
    for stats in all_stats:
        for key, val in stats.items():
            if key not in agg_stats and not key.startswith("action"):
                agg_stats[key] = val

    # Add dummy state stats (our datasets don't have observation.state)
    agg_stats["observation.state"] = {
        "min": torch.zeros(7),
        "max": torch.ones(7),
        "mean": torch.zeros(7),
        "std": torch.ones(7),
    }

    # Patch meta.stats with aggregated stats
    first_meta._stats = agg_stats

    logging.info(f"Combined dataset: {dataset.num_frames} frames, {dataset.num_episodes} episodes")

    # ── Create policy ──
    logging.info("Creating policy...")

    # Map dataset camera name to what SmolVLA base expects
    rename_map = {
        "observation.images.anchor": "observation.images.camera1",
    }

    # Set features BEFORE make_policy (which will use them if input_features is set)
    from lerobot.configs.types import FeatureType, PolicyFeature
    policy_cfg.input_features = {
        "observation.images.camera1": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
    }

    policy = make_policy(
        cfg=policy_cfg,
        ds_meta=first_meta,
        rename_map=rename_map,
    )

    # Override output_features AFTER make_policy (make_policy overwrites them from ds_meta)
    policy.config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
    }

    # ── Create processors ──
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=None,
        dataset_stats=agg_stats,
    )

    # ── Optimizer ──
    logging.info("Creating optimizer...")
    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=LR,
        betas=(0.9, 0.95),
        weight_decay=1e-10,
    )
    # Simple cosine warmup scheduler
    warmup_steps = 1000
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return 0.5 * (1 + torch.cos(torch.tensor(3.14159 * (step - warmup_steps) / max(1, STEPS - warmup_steps)))).item()
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Dummy train_cfg for checkpoint saving
    train_cfg = None

    # ── Dataloader ──
    def collate_fn(batch):
        """Custom collate that handles string fields (task) alongside tensors."""
        result = {}
        for key in batch[0]:
            values = [item[key] for item in batch]
            if isinstance(values[0], torch.Tensor):
                result[key] = torch.stack(values)
            else:
                result[key] = values  # Keep strings as list
        return result

    # Build weighted sampler so domain ratios are controlled by weight config
    sample_weights = []
    offset = 0
    for dc, ds in datasets_list:
        n = ds.num_frames
        w = dc.weight / n  # per-sample weight = domain_weight / domain_size
        sample_weights.extend([w] * n)
        logging.info(f"  Sampler: {dc.name} weight={dc.weight} → {n} frames × {w:.2e}/sample")
        offset += n
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=BATCH_SIZE * STEPS,  # total samples for the run
        replacement=True,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=NUM_WORKERS,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        pin_memory=False,
        drop_last=True,
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
        collate_fn=collate_fn,
    )

    dl_iter = cycle(dataloader)

    # ── Training loop ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    num_learnable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total = sum(p.numel() for p in policy.parameters())
    logging.info(f"Output dir: {OUTPUT_DIR}")
    logging.info(f"Steps: {STEPS}")
    logging.info(f"Batch size: {BATCH_SIZE}")
    logging.info(f"Learnable params: {format_big_number(num_learnable)} / {format_big_number(num_total)}")

    grad_clip_norm = policy_cfg.optimizer_grad_clip_norm
    policy.train()

    progbar = tqdm(total=STEPS, desc="Training", unit="step")
    running_loss = 0.0

    for step in range(1, STEPS + 1):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        batch = preprocessor(batch)
        data_time = time.perf_counter() - start_time

        t0 = time.perf_counter()

        loss, output_dict = policy.forward(batch)

        # Debug first step
        if step == 1:
            logging.info(f"Step 1 debug: loss={loss.item()}, isnan={loss.isnan().item()}")
            for k, v in sorted(batch.items()):
                if isinstance(v, torch.Tensor) and v.is_floating_point():
                    logging.info(f"  {k}: shape={v.shape} nan={v.isnan().any().item()} inf={v.isinf().any().item()} min={v.min():.4f} max={v.max():.4f} device={v.device}")
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm)

        optimizer.step()
        optimizer.zero_grad()
        if lr_scheduler is not None:
            lr_scheduler.step()

        if has_method(policy, "update"):
            policy.update()

        update_time = time.perf_counter() - t0
        running_loss = 0.95 * running_loss + 0.05 * loss.item() if step > 1 else loss.item()

        progbar.update(1)
        progbar.set_postfix(loss=f"{running_loss:.4f}", gn=f"{grad_norm.item():.2f}", lr=f"{optimizer.param_groups[0]['lr']:.1e}")

        if step % LOG_FREQ == 0:
            logging.info(
                f"step:{step} loss:{running_loss:.4f} grdn:{grad_norm.item():.3f} "
                f"lr:{optimizer.param_groups[0]['lr']:.1e} updt_s:{update_time:.3f} data_s:{data_time:.3f}"
            )

        if step % SAVE_FREQ == 0 or step == STEPS:
            logging.info(f"Saving checkpoint at step {step}")
            checkpoint_dir = get_step_checkpoint_dir(OUTPUT_DIR, STEPS, step)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "step": step,
                "model_state_dict": policy.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": running_loss,
            }, checkpoint_dir / "checkpoint.pt")
            logging.info(f"Saved to {checkpoint_dir}")

    progbar.close()
    logging.info("Training complete!")


if __name__ == "__main__":
    main()
