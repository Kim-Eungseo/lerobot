#!/usr/bin/env python3
"""Single-dataset training for SmolVLM2 + Single Learnable Action Token.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/train_smolvlm_act.py \
        --dataset_name maniskill-franka \
        --dataset_root /data/lerobot/maniskill-franka \
        --action_key action.ee_delta_pose

    CUDA_VISIBLE_DEVICES=0 python scripts/train_smolvlm_act.py \
        --dataset_name oxe-bridge \
        --dataset_root /data/lerobot/oxe-bridge \
        --action_key action
"""

import argparse
import logging
import time
from pathlib import Path

import torch
from tqdm import tqdm

from lerobot.datasets.factory import IMAGENET_STATS
from lerobot.datasets.transforms import (
    DomainRandomizationConfig,
    ImageTransforms,
    ImageTransformsConfig,
)
from lerobot.datasets.utils import cycle
from lerobot.policies.smolvlm_act.configuration_smolvlm_act import SmolVLMActConfig
from lerobot.policies.smolvlm_act.modeling_smolvlm_act import SmolVLMActPolicy
from lerobot.policies.smolvlm_act.processor_smolvlm_act import make_smolvlm_act_pre_post_processors
from lerobot.utils.train_utils import get_step_checkpoint_dir
from lerobot.utils.utils import format_big_number, has_method


# ─── Default Configuration ──────────────────────────────────────────
VLM_MODEL = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
BATCH_SIZE = 8
STEPS = 50_000
NUM_WORKERS = 4
SEED = 42
LOG_FREQ = 50
SAVE_FREQ = 10_000
LR = 5e-5
OUTPUT_DIR = Path("outputs/train/smolvlm_act")


def parse_args():
    parser = argparse.ArgumentParser(description="Train SmolVLM2-Act on a single dataset")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset repo_id (e.g. maniskill-franka)")
    parser.add_argument("--dataset_root", type=str, required=True, help="Path to dataset root directory")
    parser.add_argument("--action_key", type=str, default="action", help="Action key in dataset (default: action)")
    parser.add_argument("--vlm_model", type=str, default=VLM_MODEL, help="SmolVLM2 model name")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output_dir", type=str, default=None, help="Output dir (default: outputs/train/smolvlm_act_{dataset_name})")
    parser.add_argument("--save_freq", type=int, default=SAVE_FREQ)
    parser.add_argument("--log_freq", type=int, default=LOG_FREQ)
    parser.add_argument("--no_lora", action="store_true", help="Disable LoRA")
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=50)
    return parser.parse_args()


def build_delta_timestamps(action_key: str, fps: int, chunk_size: int = 50) -> dict[str, list[float]]:
    dt = {}
    dt["observation.images.anchor"] = [0.0]
    dt[action_key] = [i / fps for i in range(chunk_size)]
    return dt


def main():
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    output_dir = Path(args.output_dir) if args.output_dir else Path(f"outputs/train/smolvlm_act_{args.dataset_name}")

    device = torch.device("cuda")

    if args.seed is not None:
        torch.manual_seed(args.seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # ── Augmentation configs ──
    image_transforms_cfg = ImageTransformsConfig(enable=True, max_num_transforms=2)
    image_transforms = ImageTransforms(image_transforms_cfg)

    domain_randomization = DomainRandomizationConfig(
        enable=True,
        p=0.7,
        enable_lighting=True,
        enable_noise=True,
        enable_crop=True,
        lighting_gain_range=(0.3, 2.0),
        noise_iso_range=(1, 4),
    )

    # ── Create policy config ──
    policy_cfg = SmolVLMActConfig(
        vlm_model_name=args.vlm_model,
        load_vlm_weights=True,
        freeze_vision_encoder=True,
        use_lora=not args.no_lora,
        lora_rank=args.lora_rank,
        chunk_size=args.chunk_size,
        n_action_steps=args.chunk_size,
        train_state_proj=False,
        empty_cameras=2,
        optimizer_lr=args.lr,
    )
    chunk_size = policy_cfg.chunk_size

    # ── Load dataset ──
    logging.info(f"Loading dataset: {args.dataset_name} from {args.dataset_root}")

    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    root = Path(args.dataset_root)
    meta = LeRobotDatasetMetadata(args.dataset_name, root=str(root))
    fps = meta.fps

    delta_timestamps = build_delta_timestamps(args.action_key, fps, chunk_size)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(
        repo_id=args.dataset_name,
        root=str(root),
        image_transforms=image_transforms,
        domain_randomization=domain_randomization,
        delta_timestamps=delta_timestamps,
        tolerance_s=0.02,
        video_backend="pyav",
    )

    logging.info(f"  {args.dataset_name}: {dataset.num_frames} frames, {dataset.num_episodes} eps, fps={fps}")

    # ── Wrap dataset for key remapping ──
    class SingleDatasetWrapper(torch.utils.data.Dataset):
        def __init__(self, ds, action_key):
            self._ds = ds
            self._action_key = action_key

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
            return self._ds.num_frames

        def __getitem__(self, idx):
            item = self._ds[idx]

            # Remap action key
            if self._action_key != "action" and self._action_key in item:
                item["action"] = item.pop(self._action_key)
                pad_key = f"{self._action_key}_is_pad"
                if pad_key in item:
                    item["action_is_pad"] = item.pop(pad_key)

            # Remove extra action keys
            for key in list(item.keys()):
                if key.startswith("action."):
                    del item[key]
                elif key.endswith("_is_pad") and "action." in key:
                    del item[key]

            # Replace NaN
            if "action" in item and isinstance(item["action"], torch.Tensor):
                item["action"] = torch.nan_to_num(item["action"], nan=0.0)

            # Rename camera key
            if "observation.images.anchor" in item:
                item["observation.images.camera1"] = item.pop("observation.images.anchor")
            if "observation.images.anchor_is_pad" in item:
                item["observation.images.camera1_is_pad"] = item.pop("observation.images.anchor_is_pad")

            # Inject dummy state
            if "observation.state" not in item:
                item["observation.state"] = torch.zeros(7)

            # Ensure task is a string
            if "task" in item and not isinstance(item["task"], str):
                task_idx = item["task"]
                tasks_df = self._ds.meta.tasks
                if task_idx < len(tasks_df) and "task" in tasks_df.columns:
                    task_str = tasks_df.iloc[task_idx]["task"]
                    item["task"] = str(task_str) if task_str else "manipulation"
                else:
                    item["task"] = "manipulation"

            # Remove non-tensor, non-string fields
            for key in list(item.keys()):
                if not isinstance(item[key], (torch.Tensor, str)):
                    del item[key]

            return item

    wrapped_dataset = SingleDatasetWrapper(dataset, args.action_key)

    # ── Build stats ──
    stats = dict(dataset.meta.stats)
    if args.action_key != "action" and args.action_key in stats:
        stats["action"] = stats[args.action_key]

    agg_stats = {}
    if "action" in stats:
        agg_stats["action"] = stats["action"]

    for key in meta.camera_keys:
        for stats_type, stats_val in IMAGENET_STATS.items():
            agg_stats.setdefault(key, {})[stats_type] = torch.tensor(stats_val, dtype=torch.float32)

    agg_stats["observation.state"] = {
        "min": torch.zeros(7),
        "max": torch.ones(7),
        "mean": torch.zeros(7),
        "std": torch.ones(7),
    }

    meta._stats = agg_stats

    # ── Create policy ──
    logging.info("Creating SmolVLM-Act policy...")

    from lerobot.configs.types import FeatureType, PolicyFeature

    policy_cfg.input_features = {
        "observation.images.camera1": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
    }
    policy_cfg.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
    }

    policy = SmolVLMActPolicy(config=policy_cfg)

    # ── Create processors ──
    preprocessor, postprocessor = make_smolvlm_act_pre_post_processors(
        config=policy_cfg,
        dataset_stats=agg_stats,
    )

    # ── Optimizer ──
    logging.info("Creating optimizer...")
    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=1e-10,
    )
    warmup_steps = 1000
    total_steps = args.steps
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return 0.5 * (1 + torch.cos(torch.tensor(3.14159 * (step - warmup_steps) / max(1, total_steps - warmup_steps)))).item()
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Dataloader ──
    def collate_fn(batch):
        result = {}
        for key in batch[0]:
            values = [item[key] for item in batch]
            if isinstance(values[0], torch.Tensor):
                result[key] = torch.stack(values)
            else:
                result[key] = values
        return result

    dataloader = torch.utils.data.DataLoader(
        wrapped_dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=False,
        drop_last=True,
        prefetch_factor=2 if args.num_workers > 0 else None,
        collate_fn=collate_fn,
    )

    dl_iter = cycle(dataloader)

    # ── Training loop ──
    output_dir.mkdir(parents=True, exist_ok=True)
    num_learnable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total = sum(p.numel() for p in policy.parameters())
    logging.info(f"Output dir: {output_dir}")
    logging.info(f"Steps: {total_steps}")
    logging.info(f"Batch size: {args.batch_size}")
    logging.info(f"Learnable params: {format_big_number(num_learnable)} / {format_big_number(num_total)}")

    grad_clip_norm = policy_cfg.optimizer_grad_clip_norm
    policy.train()

    progbar = tqdm(total=total_steps, desc="Training", unit="step")
    running_loss = 0.0

    for step in range(1, total_steps + 1):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        batch = preprocessor(batch)
        data_time = time.perf_counter() - start_time

        t0 = time.perf_counter()

        loss, output_dict = policy.forward(batch)

        if step == 1:
            logging.info(f"Step 1 debug: loss={loss.item()}, isnan={loss.isnan().item()}")
            for k, v in sorted(batch.items()):
                if isinstance(v, torch.Tensor) and v.is_floating_point():
                    logging.info(
                        f"  {k}: shape={v.shape} nan={v.isnan().any().item()} "
                        f"inf={v.isinf().any().item()} min={v.min():.4f} max={v.max():.4f} device={v.device}"
                    )

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip_norm)
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()

        if has_method(policy, "update"):
            policy.update()

        update_time = time.perf_counter() - t0
        running_loss = 0.95 * running_loss + 0.05 * loss.item() if step > 1 else loss.item()

        progbar.update(1)
        progbar.set_postfix(loss=f"{running_loss:.4f}", gn=f"{grad_norm.item():.2f}", lr=f"{optimizer.param_groups[0]['lr']:.1e}")

        if step % args.log_freq == 0:
            logging.info(
                f"step:{step} loss:{running_loss:.4f} grdn:{grad_norm.item():.3f} "
                f"lr:{optimizer.param_groups[0]['lr']:.1e} updt_s:{update_time:.3f} data_s:{data_time:.3f}"
            )

        if step % args.save_freq == 0 or step == total_steps:
            logging.info(f"Saving checkpoint at step {step}")
            checkpoint_dir = get_step_checkpoint_dir(output_dir, total_steps, step)
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
