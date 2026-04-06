#!/usr/bin/env python3
"""Record a single demo video for SmolVLM-Act on LIBERO.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/record_demo_video.py \
        --checkpoint path/to/checkpoint.pt \
        --task_suite libero_spatial \
        --action_head_type mlp \
        --output_path demo.mp4
"""

import argparse
import logging
import math
import os
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

import logging as _logging
_orig_fh = _logging.FileHandler
class _SafeFH(_logging.FileHandler):
    def __init__(self, fn, *a, **kw):
        if fn == "/tmp/robosuite.log":
            fn = os.path.expanduser("~/.robosuite.log")
        super().__init__(fn, *a, **kw)
_logging.FileHandler = _SafeFH

from lerobot.datasets.factory import IMAGENET_STATS
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvlm_act.configuration_smolvlm_act import SmolVLMActConfig
from lerobot.policies.smolvlm_act.modeling_smolvlm_act import SmolVLMActPolicy
from lerobot.policies.smolvlm_act.processor_smolvlm_act import make_smolvlm_act_pre_post_processors

LIBERO_DATASETS = {
    "libero_spatial": "lerobot/libero_spatial_image",
    "libero_object": "lerobot/libero_object_image",
    "libero_goal": "lerobot/libero_goal_image",
}

TASK_MAX_STEPS = {
    "libero_spatial": 280,
    "libero_object": 280,
    "libero_goal": 300,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--task_suite", type=str, default="libero_spatial")
    parser.add_argument("--action_head_type", type=str, default="mlp")
    parser.add_argument("--no_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--pooling_type", type=str, default="action_token", choices=["action_token", "attentive"])
    parser.add_argument("--attentive_pooling_heads", type=int, default=8)
    parser.add_argument("--chunk_size", type=int, default=10)
    parser.add_argument("--task_id", type=int, default=None, help="Specific task id (0-9). If None, pick best from results.")
    parser.add_argument("--episode_id", type=int, default=0)
    parser.add_argument("--output_path", type=str, default="demo.mp4")
    parser.add_argument("--vlm_model", type=str, default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float32)


def quat2axisangle(quat):
    if quat[3] > 1.0: quat[3] = 1.0
    elif quat[3] < -1.0: quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load policy
    from lerobot.configs.types import FeatureType, PolicyFeature

    cfg = SmolVLMActConfig(
        vlm_model_name=args.vlm_model, load_vlm_weights=True,
        freeze_vision_encoder=True, use_lora=not args.no_lora, lora_rank=args.lora_rank,
        chunk_size=args.chunk_size, n_action_steps=args.chunk_size,
        action_head_type=args.action_head_type,
        pooling_type=args.pooling_type,
        attentive_pooling_heads=args.attentive_pooling_heads,
        train_state_proj=False, empty_cameras=0, max_state_dim=32, max_action_dim=32,
        resize_imgs_with_padding=(256, 256),
    )
    cfg.input_features = {
        "observation.images.camera1": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
    }
    cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))}

    policy = SmolVLMActPolicy(config=cfg)
    policy.to(device=device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    policy.load_state_dict(ckpt["model_state_dict"])
    policy.eval()
    logging.info(f"Loaded checkpoint step {ckpt.get('step', '?')}")

    # Load dataset stats
    dataset_name = LIBERO_DATASETS.get(args.task_suite)
    ds = LeRobotDataset(repo_id=dataset_name, episodes=[0])
    stats = dict(ds.meta.stats)
    agg_stats = {"action": {k: torch.tensor(to_np(v)) for k, v in stats["action"].items()}}
    for st, sv in IMAGENET_STATS.items():
        agg_stats.setdefault("observation.images.image", {})[st] = torch.tensor(sv, dtype=torch.float32)
    if "observation.state" in stats:
        agg_stats["observation.state"] = {k: torch.tensor(to_np(v)) for k, v in stats["observation.state"].items()}
    else:
        agg_stats["observation.state"] = {"mean": torch.zeros(8), "std": torch.ones(8), "min": torch.zeros(8), "max": torch.ones(8)}

    preprocessor, _ = make_smolvlm_act_pre_post_processors(config=cfg, dataset_stats=agg_stats)
    action_mean = to_np(agg_stats["action"]["mean"])
    action_std = to_np(agg_stats["action"]["std"])

    # Load LIBERO env (original 10 tasks)
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    LIBERO_SPATIAL_TASKS = [
        "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
    ]

    max_steps = TASK_MAX_STEPS[args.task_suite]
    task_id = args.task_id if args.task_id is not None else 3  # default: next to cookie box
    task_name = LIBERO_SPATIAL_TASKS[task_id]
    task_description = task_name.replace("_", " ")

    bddl_dir = Path(get_libero_path("bddl_files")) / args.task_suite
    init_dir = Path(get_libero_path("init_states")) / args.task_suite

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_dir / f"{task_name}.bddl"),
        camera_heights=256, camera_widths=256,
    )
    env.seed(0)

    init_states = torch.load(init_dir / f"{task_name}.pruned_init", weights_only=False)

    logging.info(f"Task {task_id}: {task_description}")
    logging.info(f"Recording episode {args.episode_id}...")

    # Run episode and record frames
    env.reset()
    obs = env.set_init_state(init_states[args.episode_id])
    for _ in range(10):
        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])

    frames = []
    action_queue = deque()
    success = False

    for t in range(max_steps):
        # Record frame
        img = obs["agentview_image"][::-1, ::-1].copy()
        frames.append(img)

        if len(action_queue) == 0:
            img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            state = np.concatenate([
                obs["robot0_eef_pos"],
                quat2axisangle(obs["robot0_eef_quat"].copy()),
                obs["robot0_gripper_qpos"]
            ]).astype(np.float32)

            batch = {
                "observation.images.camera1": img_t.unsqueeze(0).to(device),
                "observation.state": torch.from_numpy(state).unsqueeze(0).to(device),
                "task": [task_description],
            }
            batch = preprocessor(batch)
            policy.reset()

            with torch.autocast("cuda", dtype=torch.bfloat16):
                images, img_masks = policy.prepare_images(batch)
                obs_state = policy.prepare_state(batch)
                ac = policy.model.predict_actions(
                    images, img_masks,
                    batch["observation.language.tokens"],
                    batch["observation.language.attention_mask"],
                    obs_state
                )
            ac = ac[0].detach().float().cpu().numpy()[:, :7]
            ac = ac * (action_std[:7] + 1e-8) + action_mean[:7]
            for a in ac:
                action_queue.append(a)

        action = action_queue.popleft()
        action[-1] = 1.0 if action[-1] >= 0.0 else -1.0
        obs, _, done, _ = env.step(action.tolist())
        if done:
            # Record a few more frames after success
            for _ in range(10):
                img = obs["agentview_image"][::-1, ::-1].copy()
                frames.append(img)
                obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
            success = True
            break

    env.close()

    # Write video
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, args.fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()

    status = "SUCCESS" if success else "FAIL"
    logging.info(f"[{status}] Saved {len(frames)} frames to {output_path} ({len(frames)/args.fps:.1f}s)")


if __name__ == "__main__":
    main()
