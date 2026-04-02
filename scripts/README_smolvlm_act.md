# SmolVLM-Act: SmolVLM2 + Learnable Action Token for Robot Control

## 아키텍처

```
Input Sequence:
  [image_1] [image_2] ... [language_tokens] [state_emb] [<ACTION>]
       |         |              |               |            |
       v         v              v               v            v
  ┌──────────────────────────────────────────────────────────────┐
  │              SmolVLM2 Language Model (LoRA)                  │
  │  (vision encoder는 frozen, LM의 q_proj/v_proj에 LoRA 적용)  │
  └──────────────────────────────────────────────────────────────┘
                                                          |
                                                   action_hidden
                                                    (B, 1536)
                                                          |
                                              ┌───────────┴───────────┐
                                              │     Action Head       │
                                              │  (MLP/ResNet/Diffusion)│
                                              └───────────┬───────────┘
                                                          |
                                                   action_chunk
                                                (B, chunk_size, action_dim)
```

### 핵심 아이디어

OpenVLA-OFT가 action token 여러 개를 시퀀스에 넣고 각각의 hidden state를 합쳐서 action을 예측하는 것과 달리,
SmolVLM-Act는 **단 하나의 learnable action token** `<ACTION>`을 시퀀스 맨 끝에 추가한다.
이 토큰의 hidden state가 전체 시퀀스 (이미지 + 언어 + state)의 정보를 causal attention으로 집약하므로,
하나의 벡터 `(B, hidden_dim)`만으로 전체 action chunk를 예측할 수 있다.

SmolVLA가 Expert Layer + Flow Matching을 사용하는 것과 비교하면, SmolVLM-Act는
별도의 Expert 모듈 없이 VLM의 LM backbone만으로 action을 생성하므로 구조가 훨씬 단순하고 inference가 빠르다.


## 파일 구조

```
lerobot/
├── src/lerobot/policies/smolvlm_act/
│   ├── __init__.py
│   ├── configuration_smolvlm_act.py   # SmolVLMActConfig (모든 하이퍼파라미터)
│   ├── modeling_smolvlm_act.py        # 핵심 모델 + LeRobot Policy wrapper
│   ├── processor_smolvlm_act.py       # 전처리 파이프라인 (tokenizer, normalizer)
│   ├── action_heads.py                # 3종 action head (MLP, ResNet, Diffusion)
│   ├── latent_prediction_head.py      # Stage 2: V-JEPA2 latent alignment head
│   ├── vjepa_target_encoder.py        # Stage 2: frozen V-JEPA2 encoder wrapper
│   └── stage2_dataset.py              # Stage 2: context + future frame dataset
│
├── scripts/
│   ├── train_smolvlm_act.py                  # 단일 데이터셋 action 학습
│   ├── train_domain_mix_smolvlm.py           # 멀티 도메인 action 학습
│   ├── train_stage2_smolvlm_act.py           # 단일 데이터셋 Stage 2 (V-JEPA2 alignment + action)
│   ├── train_stage2_domain_mix_smolvlm.py    # 멀티 도메인 Stage 2
│   └── train_eval_libero_smolvlm_act.py      # LIBERO 학습 + 시뮬레이션 평가
```


## 모델 구성요소 상세

### 1. SmolVLM2WithActionToken (`modeling_smolvlm_act.py`)

메인 모델. 다음 서브모듈로 구성:

| 모듈 | 역할 | 파라미터 |
|------|------|----------|
| `vlm` | SmolVLM2-500M (SigLIP vision + LLaMA LM) | ~500M (대부분 frozen) |
| `action_token` | Learnable `<ACTION>` embedding `nn.Parameter(1, 1, 1536)` | 1,536 |
| `state_proj` | Robot state → LM hidden dim `Linear(32, 1536)` | 49K (frozen by default) |
| `action_head` | Hidden state → action chunk (타입 선택 가능) | 8~15M |
| LoRA adapters | LM의 q_proj, v_proj에 rank-32 LoRA | ~2.4M |

**Forward 흐름:**
1. `embed_image(img)` → SigLIP vision encoder → connector → image embeddings `(B, num_patches, 1536)`
2. `embed_language_tokens(tokens)` → LM embed_tokens → language embeddings `(B, seq_len, 1536)`
3. `state_proj(state)` → state embedding `(B, 1, 1536)`
4. `action_token.expand(B)` → action embedding `(B, 1, 1536)`
5. Concat: `[image_embs | lang_embs | state_emb | action_emb]` → `(B, total_seq, 1536)`
6. Causal attention mask 생성 (padding 고려)
7. LM forward → hidden_states `(B, total_seq, 1536)`
8. `hidden_states[:, -1, :]` → action token의 hidden state `(B, 1536)`
9. `action_head(hidden_state)` → `(B, chunk_size, action_dim)`


### 2. Action Heads (`action_heads.py`)

세 가지 action head를 지원하며, `action_head_type` config로 선택:

#### MLPActionHead (`"mlp"`, default)
```
LayerNorm → Linear(1536, 2048) → GELU → Linear(2048, 2048) → GELU → Linear(2048, chunk*action_dim) → reshape
```
- 가장 단순, inference 빠름
- Training loss: L1 (per-element)
- Params: ~8.1M

#### ResNetActionHead (`"resnet"`)
```
LayerNorm → Linear(1536, 2048) → ReLU → [ResBlock × 2] → LayerNorm → Linear(2048, chunk*action_dim) → reshape
```
- Residual connection으로 gradient flow 개선
- OpenVLA-OFT의 L1RegressionActionHead와 동일한 구조
- Training loss: L1
- Params: ~12.3M

#### DiffusionActionHead (`"diffusion"`)
```
Training:
  1. GT actions에 noise 추가 (DDIM forward process)
  2. [hidden_state; noisy_action; timestep_emb] → MLPResNet → noise prediction
  3. Loss: MSE(predicted_noise, actual_noise)

Inference:
  1. Pure noise에서 시작
  2. DDIM denoising (기본 10 steps) → clean action chunk
```
- 다중 모드 action 분포 모델링 가능
- Inference 시 iterative denoising 필요 (느림)
- OpenVLA-OFT의 DiffusionActionHead 구조 적용
- Params: ~14.7M


### 3. LatentPredictionHead (`latent_prediction_head.py`)

Stage 2 학습에서 사용. Action token hidden state를 V-JEPA2 latent space로 projection:
```
LayerNorm → Linear(1536, 1536) → ReLU → [ResBlock × 2] → LayerNorm → Linear(1536, 256)
```
- Input: `(B, 1536)` — action token hidden state
- Output: `(B, 256)` — V-JEPA2 task encoder와 동일한 차원
- Params: ~7.5M


### 4. VJEPATargetEncoder (`vjepa_target_encoder.py`)

Stage 2 학습의 frozen target encoder:
```
future_video [B, 3, K, H, W]
    → V-JEPA2 ViT-Large encoder [B, 2048, 1024]
    → AttentivePooler (learnable query cross-attention) [B, 1024]
    → Linear projection [B, 256]
```
- Disentangle post-training에서 학습된 task-biased encoder + task_head 사용
- 모든 파라미터 frozen (no grad)
- Domain-invariant task representation 제공


## 학습 모드

### Mode 1: Action Prediction Only (기본)
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_smolvlm_act.py \
    --dataset_name maniskill-franka \
    --dataset_root /data/lerobot/maniskill-franka \
    --action_key action.ee_delta_pose
```
- Loss: `L1(predicted_actions, gt_actions)`
- action_head만 학습 + LoRA adapters

### Mode 2: Stage 2 Joint Training (V-JEPA2 alignment + action)
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_stage2_smolvlm_act.py \
    --dataset_name maniskill-franka \
    --dataset_root /data/lerobot/maniskill-franka \
    --action_key action.ee_delta_pose \
    --vjepa_checkpoint /path/to/disentangle_checkpoint.pt \
    --lambda_action 1.0
```
- Loss: `L_latent(z_pred, z_target) + lambda * L_action(pred_actions, gt_actions)`
- action_head + LoRA + LatentPredictionHead 학습
- frozen V-JEPA2가 future frame target 제공

### Mode 3: LIBERO Benchmark (학습 + 시뮬레이션 평가)
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_eval_libero_smolvlm_act.py \
    --task_suite libero_spatial \
    --steps 30000 \
    --action_head_type diffusion \
    --num_eval_episodes 20
```
- HuggingFace Hub에서 `lerobot/libero_spatial_image` 자동 다운로드
- 학습 완료 후 LIBERO 시뮬레이션에서 success rate 측정


## 주요 Config 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `vlm_model_name` | `SmolVLM2-500M-Video-Instruct` | VLM backbone. 256M/500M/2.2B 선택 가능 |
| `chunk_size` | 50 | Action chunk 길이 (한 번에 예측하는 action 수) |
| `n_action_steps` | 50 | 실제 실행하는 action 수 (≤ chunk_size) |
| `action_head_type` | `"mlp"` | `"mlp"`, `"resnet"`, `"diffusion"` |
| `use_lora` | True | LoRA 사용 여부 |
| `lora_rank` | 32 | LoRA rank |
| `lora_target_modules` | `["q_proj", "v_proj"]` | LoRA 적용 모듈 |
| `freeze_vision_encoder` | True | SigLIP vision encoder freeze |
| `resize_imgs_with_padding` | (512, 512) | 입력 이미지 크기 (aspect ratio 유지 padding) |
| `max_state_dim` | 32 | Robot state 최대 차원 (부족하면 zero-padding) |
| `max_action_dim` | 32 | Action 최대 차원 (부족하면 zero-padding) |


## SmolVLA vs SmolVLM-Act 비교

| | SmolVLA | SmolVLM-Act |
|---|---------|-------------|
| **Action 생성** | Expert Layer (cross-attention) + Flow Matching | Single Action Token + MLP/ResNet/Diffusion |
| **추가 모듈** | Expert LM (separate transformer) | Action head (MLP only) |
| **Inference** | N-step denoising (flow matching) | 1 forward pass (MLP/ResNet) or N-step DDIM (Diffusion) |
| **Trainable params** | Expert layer + action projectors | LoRA + action token + action head |
| **복잡도** | 높음 (Expert + Flow Matching) | 낮음 (하나의 learnable token) |
| **Stage 2 지원** | 미지원 | V-JEPA2 latent alignment joint training |


## VLM 모델 옵션

| 모델 | HuggingFace ID | Hidden Dim | 총 파라미터 |
|------|----------------|------------|-------------|
| SmolVLM2-256M | `HuggingFaceTB/SmolVLM2-256M-Video-Instruct` | 960 | 256M |
| SmolVLM2-500M | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | 1536 | 500M |
| SmolVLM2-2.2B | `HuggingFaceTB/SmolVLM2-2.2B-Instruct` | 2048 | 2.2B |

기본값은 500M. `--vlm_model` 인자로 변경 가능:
```bash
python scripts/train_smolvlm_act.py --vlm_model HuggingFaceTB/SmolVLM2-2.2B-Instruct ...
```


## Stage 2: V-JEPA2 Latent Alignment

```
                     SmolVLM-Act                           V-JEPA2 (frozen)
              ┌─────────────────────┐              ┌─────────────────────┐
context frame │  [img][text][state] │ future frames│  video [B,3,K,H,W] │
    (t=0)     │  [<ACTION>]         │    (t=1..K)  │                     │
              │       │             │              │       │             │
              │  action_hidden      │              │  ViT-Large encoder  │
              │  (B, 1536)          │              │  + AttentivePooler  │
              │       │             │              │  + Linear proj      │
              │  ┌────┴────┐        │              │       │             │
              │  │LatentPred│        │              │  z_target (B, 256) │
              │  │  Head    │        │              └───────┬─────────────┘
              │  └────┬────┘        │                      │
              │  z_pred (B, 256)    │                      │
              └───────┬─────────────┘                      │
                      │              L_latent               │
                      └──────────── MSE/L1 ────────────────┘

              action_hidden ──→ ActionHead ──→ L_action (L1 vs GT)

              L_total = L_latent + lambda * L_action
```

Stage 2는 action token의 hidden state가 V-JEPA2의 domain-invariant task representation과
align되도록 학습한다. 이를 통해:
1. Action token이 "미래에 어떤 일이 일어날지"를 encoding하게 됨
2. Domain-invariant representation을 학습하여 sim-to-real transfer에 유리
3. Action prediction과 jointly 학습하여 action 성능 유지
