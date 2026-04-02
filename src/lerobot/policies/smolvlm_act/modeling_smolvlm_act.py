"""
SmolVLM2 + Single Learnable Action Token Policy.

Architecture:
  [image_tokens] [text_tokens] [<learnable_action_token>]
      → SmolVLM2 forward (frozen/LoRA)
      → extract last hidden state at <ACTION> position
      → simple MLP action head
      → (B, chunk_size, action_dim)

Usage:
    from lerobot.policies.smolvlm_act.modeling_smolvlm_act import SmolVLMActPolicy
    policy = SmolVLMActPolicy(config)
"""

import math
from collections import deque

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.smolvlm_act.configuration_smolvlm_act import SmolVLMActConfig
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE


# ─── Learnable Action Token ──────────────────────────────────────────

class LearnableActionToken(nn.Module):
    """Single learnable embedding appended to the VLM sequence."""

    def __init__(self, hidden_dim: int, init_std: float = 0.02):
        super().__init__()
        self.token_embedding = nn.Parameter(torch.randn(1, 1, hidden_dim) * init_std)

    def expand(self, batch_size: int) -> Tensor:
        return self.token_embedding.expand(batch_size, -1, -1)


# ─── Action Head ─────────────────────────────────────────────────────

class ActionHead(nn.Module):
    """Maps a single hidden state vector to a full action chunk."""

    def __init__(self, hidden_dim: int, action_dim: int, chunk_size: int, mlp_hidden: int = 2048, num_layers: int = 3):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size

        layers = [nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, mlp_hidden), nn.GELU()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(mlp_hidden, mlp_hidden), nn.GELU()]
        layers.append(nn.Linear(mlp_hidden, chunk_size * action_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, hidden_state: Tensor) -> Tensor:
        # hidden_state: (B, hidden_dim)
        out = self.mlp(hidden_state)  # (B, chunk_size * action_dim)
        return out.reshape(-1, self.chunk_size, self.action_dim)


# ─── Helper functions ────────────────────────────────────────────────

def resize_with_pad(img, width, height, pad_value=-1):
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")
    cur_height, cur_width = img.shape[2:]
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(img, size=(resized_height, resized_width), mode="bilinear", align_corners=False)
    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))
    return F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)


def pad_vector(vector, new_dim):
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector


# ─── Core VLM + Action Token Model ──────────────────────────────────

class SmolVLM2WithActionToken(nn.Module):
    """SmolVLM2 backbone + learnable action token + action head."""

    def __init__(self, config: SmolVLMActConfig):
        super().__init__()
        self.config = config

        # Load VLM
        if config.load_vlm_weights:
            self.vlm = AutoModelForImageTextToText.from_pretrained(
                config.vlm_model_name,
                torch_dtype="bfloat16",
                low_cpu_mem_usage=True,
            )
        else:
            vlm_config = AutoConfig.from_pretrained(config.vlm_model_name)
            from transformers import SmolVLMForConditionalGeneration
            self.vlm = SmolVLMForConditionalGeneration(config=vlm_config)

        self.processor = AutoProcessor.from_pretrained(config.vlm_model_name)

        # Get hidden dim from loaded model
        hidden_dim = self.vlm.config.text_config.hidden_size
        self.hidden_dim = hidden_dim

        # Freeze vision encoder
        if config.freeze_vision_encoder:
            for p in self.vlm.model.vision_model.parameters():
                p.requires_grad = False

        # Learnable action token
        self.action_token = LearnableActionToken(hidden_dim, init_std=config.action_token_init_std)

        # State projection
        self.state_proj = nn.Linear(config.max_state_dim, hidden_dim)
        if not config.train_state_proj:
            for p in self.state_proj.parameters():
                p.requires_grad = False

        # Action head (supports multiple types via config.action_head_type)
        from lerobot.policies.smolvlm_act.action_heads import build_action_head
        original_action_dim = config.max_action_dim
        head_type = getattr(config, "action_head_type", "mlp")
        head_kwargs = {}
        if head_type == "mlp":
            head_kwargs["num_layers"] = config.action_head_num_layers
        elif head_type == "resnet":
            head_kwargs["num_blocks"] = getattr(config, "action_head_num_blocks", 2)
        elif head_type == "diffusion":
            head_kwargs["num_blocks"] = getattr(config, "action_head_num_blocks", 2)
            head_kwargs["num_train_steps"] = getattr(config, "diffusion_num_train_steps", 50)
            head_kwargs["num_infer_steps"] = getattr(config, "diffusion_num_infer_steps", 10)
        self.action_head = build_action_head(
            head_type=head_type,
            hidden_dim=hidden_dim,
            action_dim=original_action_dim,
            chunk_size=config.chunk_size,
            mlp_hidden=config.action_head_hidden_dim,
            **head_kwargs,
        )
        self._action_head_type = head_type

        # Token IDs for image processing
        self.fake_image_token = self.processor.tokenizer.fake_image_token_id
        self.global_image_token = self.processor.tokenizer.global_image_token_id

        # Apply LoRA if requested
        if config.use_lora:
            self._apply_lora(config)

    def _apply_lora(self, config: SmolVLMActConfig):
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=0.0,
            bias="none",
        )
        self.vlm = get_peft_model(self.vlm, lora_config)

    def get_language_model(self):
        vlm = self.vlm.base_model.model if hasattr(self.vlm, "base_model") else self.vlm
        return vlm.model.text_model

    def embed_language_tokens(self, tokens: Tensor) -> Tensor:
        vlm = self.vlm.base_model.model if hasattr(self.vlm, "base_model") else self.vlm
        return vlm.model.text_model.embed_tokens(tokens)

    def embed_image(self, pixel_values: Tensor) -> Tensor:
        vlm = self.vlm.base_model.model if hasattr(self.vlm, "base_model") else self.vlm
        image_features = vlm.model.vision_model(pixel_values.to(dtype=vlm.dtype))
        image_features = image_features.last_hidden_state
        image_features = vlm.model.connector(image_features)
        return image_features

    def build_prefix_embeds(self, images, img_masks, lang_tokens, lang_masks, state):
        """Build the prefix embeddings: [image_embs] [lang_embs] [state_emb]"""
        embs = []
        pad_masks = []

        # Image embeddings
        for img, img_mask in zip(images, img_masks, strict=False):
            img_emb = self.embed_image(img)
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * math.sqrt(img_emb_dim)

            bsize, num_img_embs = img_emb.shape[:2]
            expanded_mask = img_mask[:, None].expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(expanded_mask)

        # Language embeddings
        lang_emb = self.embed_language_tokens(lang_tokens)
        lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # State embedding
        state_emb = self.state_proj(state)
        if state_emb.ndim == 2:
            state_emb = state_emb[:, None, :]
        bsize = state_emb.shape[0]
        device = state_emb.device
        state_mask = torch.ones(bsize, state_emb.shape[1], dtype=torch.bool, device=device)
        embs.append(state_emb)
        pad_masks.append(state_mask)

        prefix_embs = torch.cat(embs, dim=1)
        prefix_mask = torch.cat(pad_masks, dim=1)
        return prefix_embs, prefix_mask

    def forward_with_hidden(self, images, img_masks, lang_tokens, lang_masks, state):
        """Full forward: prefix + action token -> VLM -> last hidden at action position."""
        prefix_embs, prefix_mask = self.build_prefix_embeds(
            images, img_masks, lang_tokens, lang_masks, state
        )
        B = prefix_embs.shape[0]
        device = prefix_embs.device

        # Append learnable action token
        action_emb = self.action_token.expand(B).to(dtype=prefix_embs.dtype, device=device)
        full_embs = torch.cat([prefix_embs, action_emb], dim=1)  # (B, seq+1, D)

        action_mask = torch.ones(B, 1, dtype=torch.bool, device=device)
        full_mask = torch.cat([prefix_mask, action_mask], dim=1)

        # Causal attention mask
        seq_len = full_embs.shape[1]
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
        # Apply padding mask
        causal_mask = causal_mask[None, None, :, :] & full_mask[:, None, None, :]

        # Forward through language model
        lm = self.get_language_model()
        hidden_states = lm(
            inputs_embeds=full_embs,
            attention_mask=causal_mask,
        ).last_hidden_state  # (B, seq+1, D)

        # Extract the last position (action token)
        action_hidden = hidden_states[:, -1, :]  # (B, D)
        return action_hidden

    def predict_actions(self, images, img_masks, lang_tokens, lang_masks, state):
        """Inference: get action chunk from single forward pass (or DDIM denoising for diffusion)."""
        action_hidden = self.forward_with_hidden(images, img_masks, lang_tokens, lang_masks, state)
        return self.action_head(action_hidden.float())  # (B, chunk_size, action_dim)

    def _compute_action_loss(self, action_hidden, gt_actions):
        """Compute action loss, dispatching to the right method based on head type."""
        if self._action_head_type == "diffusion":
            return self.action_head.compute_loss(action_hidden.float(), gt_actions)
        else:
            predicted = self.action_head(action_hidden.float())  # (B, chunk_size, action_dim)
            return F.l1_loss(predicted, gt_actions, reduction="mean")

    def compute_loss(self, images, img_masks, lang_tokens, lang_masks, state, gt_actions):
        """Training: compute action loss (L1 for mlp/resnet, MSE noise for diffusion)."""
        action_hidden = self.forward_with_hidden(images, img_masks, lang_tokens, lang_masks, state)
        if self._action_head_type == "diffusion":
            # Diffusion returns scalar MSE loss on noise prediction
            return self.action_head.compute_loss(action_hidden.float(), gt_actions).unsqueeze(0)
        else:
            predicted = self.action_head(action_hidden.float())
            return F.l1_loss(predicted, gt_actions, reduction="none")

    def compute_stage2_loss(self, images, img_masks, lang_tokens, lang_masks,
                            state, gt_actions, z_target, pred_head,
                            lambda_action=1.0, latent_loss_type="mse"):
        """Stage 2: joint latent alignment + action prediction.

        Args:
            images, img_masks, lang_tokens, lang_masks, state: standard VLM inputs
            gt_actions: [B, chunk_size, action_dim] ground truth actions
            z_target: [B, proj_dim] frozen V-JEPA2 target latent (detached)
            pred_head: LatentPredictionHead module
            lambda_action: weight for action loss
            latent_loss_type: "mse" or "l1"

        Returns:
            loss: scalar total loss
            metrics: dict with l_latent, l_action, cosine_sim
        """
        action_hidden = self.forward_with_hidden(images, img_masks, lang_tokens, lang_masks, state)

        # Latent prediction -> align with V-JEPA2
        z_pred = pred_head(action_hidden)  # (B, proj_dim)
        z_target = z_target.detach()
        if latent_loss_type == "l1":
            l_latent = F.l1_loss(z_pred, z_target)
        else:
            l_latent = F.mse_loss(z_pred, z_target)

        # Action prediction (dispatches correctly for diffusion vs mlp/resnet)
        l_action = self._compute_action_loss(action_hidden, gt_actions)

        loss = l_latent + lambda_action * l_action

        with torch.no_grad():
            cos_sim = F.cosine_similarity(z_pred, z_target, dim=-1).mean()

        metrics = {
            "l_latent": l_latent.item(),
            "l_action": l_action.item(),
            "cosine_sim": cos_sim.item(),
        }
        return loss, metrics


# ─── Policy Wrapper ──────────────────────────────────────────────────

class SmolVLMActPolicy(PreTrainedPolicy):
    """LeRobot policy wrapper for SmolVLM2 + Learnable Action Token."""

    config_class = SmolVLMActConfig
    name = "smolvlm_act"

    def __init__(self, config: SmolVLMActConfig, **kwargs):
        super().__init__(config)
        self.config = config
        self.model = SmolVLM2WithActionToken(config)
        self.reset()

    def reset(self):
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    def prepare_images(self, batch):
        images = []
        img_masks = []
        present_img_keys = [key for key in self.config.image_features if key in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features missing from batch. (batch keys: {batch.keys()}) "
                f"(image_features: {self.config.image_features})"
            )

        for key in present_img_keys:
            img = batch[key][:, -1, :, :, :] if batch[key].ndim == 5 else batch[key]
            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)
            # Normalize [0,1] -> [-1,1] for SigLIP
            img = img * 2.0 - 1.0

            bsize = img.shape[0]
            device = img.device
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        # Pad missing cameras
        missing_img_keys = [key for key in self.config.image_features if key not in batch]
        for i in range(min(len(missing_img_keys), self.config.empty_cameras)):
            img = torch.ones_like(images[-1]) * -1
            mask = torch.zeros(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        return images, img_masks

    def prepare_state(self, batch):
        state = batch[OBS_STATE][:, -1, :] if batch[OBS_STATE].ndim > 2 else batch[OBS_STATE]
        return pad_vector(state, self.config.max_state_dim)

    def prepare_action(self, batch):
        return pad_vector(batch[ACTION], self.config.max_action_dim)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        self.eval()
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        if len(self._queues[ACTION]) == 0:
            actions = self._get_action_chunk(batch)
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])

        return self._queues[ACTION].popleft()

    def _get_action_chunk(self, batch):
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        actions = self.model.predict_actions(images, img_masks, lang_tokens, lang_masks, state)

        # Unpad
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]
        return actions

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("action_is_pad")

        losses = self.model.compute_loss(images, img_masks, lang_tokens, lang_masks, state, actions)

        # Mask padded actions
        original_action_dim = self.config.action_feature.shape[0]
        losses = losses[:, :, :original_action_dim]

        loss_dict = {}
        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)

        if reduction == "none":
            per_sample_loss = losses.mean(dim=(1, 2))
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict
        else:
            loss = losses.mean()
            loss_dict["loss"] = loss.item()
            return loss, loss_dict
