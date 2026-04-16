#!/usr/bin/env python3
"""Evaluate SmolVLM-Act on original LIBERO benchmark (10 tasks per suite).

Unlike LIBERO-plus (2000+ tasks), this uses the original 10 tasks per suite
with their original init state files. No dependency on LIBERO-plus benchmark system.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_libero_original.py \
        --checkpoint outputs/train/smolvlm_act_libero_spatial/checkpoints/030000/checkpoint.pt \
        --task_suite libero_spatial \
        --num_episodes 50

    # Diffusion head:
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_libero_original.py \
        --checkpoint outputs/train/smolvlm_act_libero_spatial_diffusion/checkpoints/030000/checkpoint.pt \
        --task_suite libero_spatial \
        --action_head_type diffusion \
        --num_episodes 50
"""

import argparse
import json
import logging
import math
import os
from collections import deque
from pathlib import Path

import numpy as np
import torch

# Monkey-patch robosuite log
import logging as _logging
_orig_fh = _logging.FileHandler
class _SafeFH(_logging.FileHandler):
    def __init__(self, fn, *a, **kw):
        if fn == "/tmp/robosuite.log":
            fn = os.path.expanduser("~/.robosuite.log")
        super().__init__(fn, *a, **kw)
_logging.FileHandler = _SafeFH

from lerobot.datasets.factory import IMAGENET_STATS
from lerobot.policies.smolvlm_act.configuration_smolvlm_act import SmolVLMActConfig
from lerobot.policies.smolvlm_act.modeling_smolvlm_act import SmolVLMActPolicy
from lerobot.policies.smolvlm_act.processor_smolvlm_act import make_smolvlm_act_pre_post_processors


# ─── Original LIBERO task definitions (10 tasks per suite) ──────

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

LIBERO_OBJECT_TASKS = [
    "pick_up_the_alphabet_soup_and_place_it_in_the_basket",
    "pick_up_the_butter_and_place_it_in_the_basket",
    "pick_up_the_cream_cheese_and_place_it_in_the_basket",
    "pick_up_the_ketchup_and_place_it_in_the_basket",
    "pick_up_the_milk_and_place_it_in_the_basket",
    "pick_up_the_orange_juice_and_place_it_in_the_basket",
    "pick_up_the_salad_dressing_and_place_it_in_the_basket",
    "pick_up_the_bbq_sauce_and_place_it_in_the_basket",
    "pick_up_the_tomato_sauce_and_place_it_in_the_basket",
    "pick_up_the_chocolate_pudding_and_place_it_in_the_basket",
]

LIBERO_GOAL_TASKS = [
    "open_the_bottom_drawer_of_the_cabinet",
    "open_the_bottom_drawer_of_the_cabinet_and_put_the_bowl_in_it",
    "open_the_top_drawer_and_put_the_bowl_inside",
    "push_the_plate_to_the_front_of_the_stove",
    "put_the_bowl_on_the_plate",
    "put_the_bowl_on_the_stove",
    "put_the_bowl_on_top_of_the_cabinet",
    "put_the_cream_cheese_in_the_bowl",
    "put_the_wine_bottle_on_the_rack",
    "turn_on_the_stove",
]

LIBERO_10_TASKS = [
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
    "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
    "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
    "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
    "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
    "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
    "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
    "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
]

LIBERO_TASKS = {
    "libero_spatial": LIBERO_SPATIAL_TASKS,
    "libero_object": LIBERO_OBJECT_TASKS,
    "libero_goal": LIBERO_GOAL_TASKS,
    "libero_10": LIBERO_10_TASKS,
}

TASK_MAX_STEPS = {
    "libero_spatial": 280,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Eval SmolVLM-Act on original LIBERO (10 tasks)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--task_suite", type=str, default="libero_spatial",
                        choices=list(LIBERO_TASKS.keys()))
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--vlm_model", type=str, default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    parser.add_argument("--action_head_type", type=str, default="mlp", choices=["mlp", "resnet", "diffusion"])
    parser.add_argument("--chunk_size", type=int, default=10)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--no_lora", action="store_true", help="Disable LoRA (for full finetune checkpoints)")
    parser.add_argument("--pooling_type", type=str, default="action_token", choices=["action_token", "attentive"])
    parser.add_argument("--attentive_pooling_heads", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=7)
    # Temporal ensemble
    parser.add_argument("--n_execute", type=int, default=None, help="Actions to execute before re-predict (default=chunk_size)")
    parser.add_argument("--temporal_ensemble", action="store_true", help="Enable temporal ensemble (weighted avg of overlapping chunks)")
    parser.add_argument("--ensemble_exp_weight", type=float, default=0.0, help="Exponential decay weight for ensemble (0=uniform)")
    return parser.parse_args()


def quat2axisangle(quat):
    if quat[3] > 1.0: quat[3] = 1.0
    elif quat[3] < -1.0: quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


LIBERO_DATASETS = {
    "libero_spatial": "lerobot/libero_spatial_image",
    "libero_object": "lerobot/libero_object_image",
    "libero_goal": "lerobot/libero_goal_image",
    "libero_10": "lerobot/libero_10_image",
}


def load_policy(args, device):
    """Load trained SmolVLM-Act policy from checkpoint."""
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
    cfg.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
    }

    policy = SmolVLMActPolicy(config=cfg)
    policy.to(device=device)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    policy.load_state_dict(ckpt["model_state_dict"])
    logging.info(f"Loaded checkpoint from {args.checkpoint} (step {ckpt.get('step', '?')})")

    # Load real dataset stats for proper unnormalization
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset_name = LIBERO_DATASETS.get(args.task_suite, LIBERO_DATASETS["libero_spatial"])
    logging.info(f"Loading dataset stats from {dataset_name}...")
    dataset = LeRobotDataset(repo_id=dataset_name, episodes=[0])
    stats = dict(dataset.meta.stats)

    agg_stats = {}
    if "action" in stats:
        agg_stats["action"] = stats["action"]
    else:
        agg_stats["action"] = {"mean": torch.zeros(7), "std": torch.ones(7), "min": -torch.ones(7), "max": torch.ones(7)}
    for stats_type, stats_val in IMAGENET_STATS.items():
        agg_stats.setdefault("observation.images.image", {})[stats_type] = torch.tensor(stats_val, dtype=torch.float32)
    if "observation.state" in stats:
        agg_stats["observation.state"] = stats["observation.state"]
    else:
        agg_stats["observation.state"] = {"mean": torch.zeros(8), "std": torch.ones(8), "min": torch.zeros(8), "max": torch.ones(8)}

    # Extract action stats for manual unnormalization
    action_mean = agg_stats["action"]["mean"].numpy() if hasattr(agg_stats["action"]["mean"], "numpy") else np.asarray(agg_stats["action"]["mean"], dtype=np.float32)
    action_std = agg_stats["action"]["std"].numpy() if hasattr(agg_stats["action"]["std"], "numpy") else np.asarray(agg_stats["action"]["std"], dtype=np.float32)

    preprocessor, _ = make_smolvlm_act_pre_post_processors(
        config=cfg,
        dataset_stats=agg_stats,
    )
    return policy, preprocessor, cfg, action_mean, action_std


def _predict_chunk(policy, preprocessor, obs, task_description, device, action_mean, action_std):
    """Single forward pass: obs -> unnormalized action chunk (chunk_size, 7)."""
    img = obs["agentview_image"][::-1, ::-1].copy()
    img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

    eef_pos = obs["robot0_eef_pos"]
    eef_quat = obs["robot0_eef_quat"]
    gripper = obs["robot0_gripper_qpos"]
    state = np.concatenate([eef_pos, quat2axisangle(eef_quat.copy()), gripper]).astype(np.float32)

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
        lang_tokens = batch["observation.language.tokens"]
        lang_masks = batch["observation.language.attention_mask"]
        action_chunk = policy.model.predict_actions(
            images, img_masks, lang_tokens, lang_masks, obs_state
        )
    actions = action_chunk[0].float().cpu().numpy()[:, :7]

    if action_mean is not None and action_std is not None:
        actions = actions * (action_std[:7] + 1e-8) + action_mean[:7]

    return actions


@torch.no_grad()
def run_episode(policy, preprocessor, env, task_description, max_steps, chunk_size, device,
                action_mean=None, action_std=None, n_execute=None, temporal_ensemble=False,
                ensemble_exp_weight=0.0):
    """Run one episode, return success bool.

    Args:
        n_execute: Number of actions to execute before re-predicting. Default=chunk_size (no overlap).
        temporal_ensemble: If True, average overlapping action predictions with exponential weighting.
        ensemble_exp_weight: Exponential weight for temporal ensemble (0=uniform, higher=favor newer).
    """
    if n_execute is None:
        n_execute = chunk_size

    obs = env._last_obs

    if not temporal_ensemble:
        # Simple re-prediction: execute n_execute steps, discard rest, re-predict
        action_queue = deque()
        for t in range(max_steps):
            if len(action_queue) == 0:
                actions = _predict_chunk(policy, preprocessor, obs, task_description, device,
                                         action_mean, action_std)
                for a in actions[:n_execute]:
                    action_queue.append(a)

            action = action_queue.popleft()
            action[-1] = 1.0 if action[-1] >= 0.0 else -1.0
            obs, reward, done, info = env.step(action.tolist())
            if done:
                return True
    else:
        # Temporal ensemble: accumulate overlapping predictions, weighted average
        # action_buffer[t] = list of (prediction, weight) for timestep t
        action_buffer = {}
        next_predict_t = 0

        for t in range(max_steps):
            # Predict new chunk if needed
            if t >= next_predict_t:
                actions = _predict_chunk(policy, preprocessor, obs, task_description, device,
                                         action_mean, action_std)
                for i, a in enumerate(actions):
                    future_t = t + i
                    if future_t not in action_buffer:
                        action_buffer[future_t] = []
                    weight = np.exp(-ensemble_exp_weight * i)
                    action_buffer[future_t].append((a, weight))
                next_predict_t = t + n_execute

            # Weighted average of all predictions for this timestep
            if t in action_buffer:
                preds = action_buffer[t]
                total_weight = sum(w for _, w in preds)
                action = sum(a * w for a, w in preds) / total_weight
                del action_buffer[t]
            else:
                # Fallback: should not happen
                action = np.zeros(7)

            action[-1] = 1.0 if action[-1] >= 0.0 else -1.0
            obs, reward, done, info = env.step(action.tolist())
            if done:
                return True

    return False


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir) if args.output_dir else Path(
        f"outputs/eval/{args.task_suite}_{args.action_head_type}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load policy
    policy, preprocessor, cfg, action_mean, action_std = load_policy(args, device)
    policy.eval()

    # Load LIBERO env
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_names = LIBERO_TASKS[args.task_suite]
    max_steps = TASK_MAX_STEPS[args.task_suite]
    init_dir = Path(get_libero_path("init_states")) / args.task_suite
    bddl_dir = Path(get_libero_path("bddl_files")) / args.task_suite

    logging.info(f"Evaluating on {args.task_suite} ({len(task_names)} tasks, {args.num_episodes} episodes each)")
    logging.info(f"Action head: {args.action_head_type}, Checkpoint: {args.checkpoint}")

    total_successes = 0
    total_episodes = 0
    task_results = {}

    for task_idx, task_name in enumerate(task_names):
        task_description = task_name.replace("_", " ")
        bddl_file = bddl_dir / f"{task_name}.bddl"
        init_file = init_dir / f"{task_name}.pruned_init"

        if not bddl_file.exists():
            logging.warning(f"  Skip task {task_idx}: bddl not found: {bddl_file}")
            continue
        if not init_file.exists():
            logging.warning(f"  Skip task {task_idx}: init states not found: {init_file}")
            continue

        init_states = torch.load(init_file, weights_only=False)
        num_eps = min(args.num_episodes, len(init_states))

        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl_file),
            camera_heights=256,
            camera_widths=256,
        )
        env.seed(0)

        task_successes = 0
        for ep_idx in range(num_eps):
            env.reset()
            obs = env.set_init_state(init_states[ep_idx])

            # Wait for objects to settle
            for _ in range(10):
                obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
            env._last_obs = obs

            success = run_episode(policy, preprocessor, env, task_description, max_steps, args.chunk_size, device,
                                 action_mean=action_mean, action_std=action_std,
                                 n_execute=args.n_execute, temporal_ensemble=args.temporal_ensemble,
                                 ensemble_exp_weight=args.ensemble_exp_weight)
            task_successes += int(success)
            total_successes += int(success)
            total_episodes += 1

            if (ep_idx + 1) % 10 == 0:
                logging.info(f"  Task {task_idx} [{task_description[:50]}] ep {ep_idx+1}/{num_eps}: "
                             f"{task_successes}/{ep_idx+1} ({task_successes/(ep_idx+1)*100:.0f}%)")

        task_sr = task_successes / num_eps
        task_results[task_description] = {"success_rate": task_sr, "successes": task_successes, "episodes": num_eps}
        logging.info(f"Task {task_idx}: {task_description[:60]}  SR={task_sr:.2f} ({task_successes}/{num_eps})")

        env.close()

    # Final results
    overall_sr = total_successes / total_episodes if total_episodes > 0 else 0
    logging.info(f"\n{'='*60}")
    logging.info(f"RESULTS: {args.task_suite} ({args.action_head_type} head)")
    logging.info(f"{'='*60}")
    logging.info(f"Overall Success Rate: {overall_sr:.4f} ({overall_sr*100:.1f}%)")
    logging.info(f"Total: {total_successes}/{total_episodes}")
    logging.info(f"{'='*60}")
    for desc, res in task_results.items():
        logging.info(f"  {desc[:60]:60s}  {res['success_rate']:.2f}")

    # Save results
    results = {
        "task_suite": args.task_suite,
        "action_head_type": args.action_head_type,
        "checkpoint": args.checkpoint,
        "overall_success_rate": overall_sr,
        "total_successes": total_successes,
        "total_episodes": total_episodes,
        "per_task": task_results,
    }
    results_path = output_dir / f"results_{args.task_suite}_{args.action_head_type}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
