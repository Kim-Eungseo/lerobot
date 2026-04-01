# SmolVLM2-Act: Single Learnable Action Token Policy

SmolVLM2를 VLM 백본으로 사용하고, 시퀀스 끝에 **learnable token 1개**를 추가하여 해당 hidden state로부터 전체 action chunk를 예측하는 정책.

## 아키텍처

```
[image_tokens] [language_tokens] [state_emb] [<ACTION>]
                                                 |
                    SmolVLM2 (frozen/LoRA)        |
                                                 v
                              last hidden state (B, hidden_dim)
                                                 |
                              MLP Action Head     |
                                                 v
                            actions (B, chunk_size, action_dim)
```

- `<ACTION>`: 학습 가능한 `nn.Parameter` 1개 (VLM hidden_dim 크기)
- Action Head: `LayerNorm -> Linear -> GELU -> Linear -> GELU -> Linear`
- Loss: L1
- 학습 파라미터: learnable token + action head + LoRA adapter

## 파일 구조

```
src/lerobot/policies/smolvlm_act/
    __init__.py
    configuration_smolvlm_act.py   # SmolVLMActConfig
    modeling_smolvlm_act.py        # SmolVLMActPolicy, SmolVLM2WithActionToken, ActionHead, LearnableActionToken
    processor_smolvlm_act.py       # pre/post processor

scripts/
    train_smolvlm_act.py           # 단일 데이터셋 학습
    train_domain_mix_smolvlm.py    # 4개 도메인 혼합 학습
```

## 실행 방법

### 단일 데이터셋 학습

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_smolvlm_act.py \
    --dataset_name maniskill-franka \
    --dataset_root /data/lerobot/maniskill-franka \
    --action_key action.ee_delta_pose
```

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_smolvlm_act.py \
    --dataset_name oxe-bridge \
    --dataset_root /data/lerobot/oxe-bridge \
    --action_key action
```

### 도메인 혼합 학습

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_domain_mix_smolvlm.py
```

도메인 구성 (스크립트 상단에서 수정):

| 도메인 | action_key | 비율 | bg_augment |
|--------|------------|------|------------|
| maniskill-franka | `action.ee_delta_pose` | 6 | O |
| maniskill-xarm | `action.ee_delta_pose` | 1 | O |
| oxe-bridge | `action` | 2 | X |
| oxe-fractal | `action` | 1 | X |

## 주요 인자 (train_smolvlm_act.py)

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--dataset_name` | (필수) | 데이터셋 이름 |
| `--dataset_root` | (필수) | 데이터셋 경로 |
| `--action_key` | `action` | 데이터셋의 action 키 |
| `--vlm_model` | `SmolVLM2-500M-Video-Instruct` | VLM 백본 |
| `--batch_size` | 8 | |
| `--steps` | 50,000 | |
| `--lr` | 5e-5 | |
| `--lora_rank` | 32 | |
| `--chunk_size` | 50 | action chunk 길이 |
| `--no_lora` | false | LoRA 비활성화 |
| `--output_dir` | auto | 체크포인트 저장 경로 |
| `--save_freq` | 10,000 | 체크포인트 저장 간격 |

## VLM 모델 옵션

```bash
# 256M (가볍고 빠름)
--vlm_model HuggingFaceTB/SmolVLM2-256M-Video-Instruct

# 500M (기본)
--vlm_model HuggingFaceTB/SmolVLM2-500M-Video-Instruct

# 2.2B (성능 우선)
--vlm_model HuggingFaceTB/SmolVLM2-2.2B-Instruct
```

hidden_dim은 모델에서 자동으로 읽어오므로 별도 설정 불필요.

## 기존 SmolVLA와의 차이

| | SmolVLA | SmolVLM2-Act |
|---|---------|-------------|
| Action 생성 | Expert Layer + Flow Matching | Learnable token 1개 + MLP |
| 추론 | N-step denoising | Single forward pass |
| 추가 모듈 | Expert LM (cross-attn) | Action head (3-layer MLP) |
| Loss | MSE (flow matching) | L1 |
