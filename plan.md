# SmolVLM2 + Single Learnable Action Token: 구현 계획

## 목표

SmolVLM2를 직접 백본으로 사용하되, OFT처럼 다수의 action token을 bidirectional decoding하는 대신 **learnable token 1개만 시퀀스 끝에 추가**하고, 해당 토큰의 hidden state에 간단한 action head를 붙여 action chunk 전체를 예측한다. 학습은 **다중 도메인 혼합(Domain-Mix Training)** 으로 수행한다.

> **OFT와의 차이:** OFT는 `chunk_len x action_dim`개의 action token을 넣고 bidirectional decoding으로 hidden states를 추출하지만, 본 계획은 **단일 learnable token** 1개의 hidden state만으로 전체 action chunk를 한번에 예측한다.
>
> **SmolVLA와의 차이:** SmolVLA는 SmolVLM + Expert Layer + Flow Matching 구조를 사용하지만, 본 계획은 SmolVLM2를 직접 사용하며 단일 토큰 hidden state + 간단한 MLP action head를 사용한다.

---

## 1. 아키텍처 개요

```
Image + Language Instruction
        |
+--------------------------------------+
|  SmolVLM2 (frozen or LoRA)           |
|  - SigLIP Vision Encoder             |
|  - SmolLM2 LLM backbone              |
|  - 256M: hidden_dim = 960            |
|  - 500M: hidden_dim = 1536           |
|  - 2.2B: hidden_dim = 2048           |
+--------------------------------------+
|  input: [img_tokens] [text_tokens] [<ACTION>]   <- learnable token 1개
+----------+---------------------------+
           | last hidden state at <ACTION> position
           | shape: (B, 1, hidden_dim)
           v
+--------------------------------------+
|  Action Head (simple MLP)            |
|  Linear(hidden_dim, mlp_hidden)      |
|  -> ReLU -> Linear(mlp_hidden,       |
|                    mlp_hidden)        |
|  -> ReLU -> Linear(mlp_hidden,       |
|        chunk_size * action_dim)      |
+----------+---------------------------+
           v
   Continuous Actions (B, chunk_size, action_dim)
```

**SmolVLM2 모델 옵션:**

| 모델 | HuggingFace ID | hidden_dim | 파라미터 |
|------|----------------|------------|---------|
| SmolVLM2-256M | `HuggingFaceTB/SmolVLM2-256M-Video-Instruct` | 960 | 256M |
| SmolVLM2-500M | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | 1536 | 500M |
| SmolVLM2-2.2B | `HuggingFaceTB/SmolVLM2-2.2B-Instruct` | 2048 | 2.2B |

**핵심 아이디어:**
- SmolVLM2 시퀀스 끝에 learnable embedding 1개(`<ACTION>` token)를 concat
- VLM이 image + text context를 attend한 뒤, `<ACTION>` 위치의 hidden state가 전체 장면/명령어 정보를 압축
- 그 single hidden vector `(B, hidden_dim)`을 간단한 MLP에 넣어 `(B, chunk_size * action_dim)`을 출력하고 reshape

---

## 2. 핵심 컴포넌트

### 2-1. Learnable Action Token

```python
class LearnableActionToken(nn.Module):
    """VLM hidden_dim과 같은 크기의 learnable embedding 1개"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        # 학습 가능한 action query 토큰 (1, hidden_dim)
        self.token_embedding = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

    def expand(self, batch_size: int):
        """(1, 1, D) -> (B, 1, D)"""
        return self.token_embedding.expand(batch_size, -1, -1)
```

**왜 learnable token인가:**
- 기존 vocab의 어떤 토큰도 "action을 예측해라"라는 의미를 가지지 않음
- Learnable embedding은 학습 과정에서 "이 위치에서 action-relevant 정보를 집약하라"는 신호를 VLM에 전달하도록 최적화됨
- 추론 시에도 동일한 학습된 embedding을 사용하므로 별도의 masking이 불필요

### 2-2. Action Head (Simple MLP)

```python
class ActionHead(nn.Module):
    """단일 hidden state -> 전체 action chunk를 예측하는 간단한 MLP"""

    def __init__(self, hidden_dim: int, action_dim: int, chunk_size: int, mlp_hidden: int = 2048):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, chunk_size * action_dim),
        )

    def forward(self, hidden_state):
        # hidden_state: (B, hidden_dim) -- <ACTION> 토큰의 last hidden state
        out = self.mlp(hidden_state)  # (B, chunk_size * action_dim)
        return out.reshape(-1, self.chunk_size, self.action_dim)  # (B, chunk_size, action_dim)
```

**OFT의 MLPResNet 대비 단순화:**
- OFT: 여러 토큰의 hidden states를 concat -> ResNet blocks -> per-step output
- 본 계획: 단일 hidden state -> 3-layer MLP -> 전체 chunk를 한번에 출력
- 입력이 1개 벡터이므로 residual block 없이도 충분

### 2-3. 학습 시 Forward Pass

```python
# 1) 이미지 + 텍스트를 SmolVLM2 processor로 토큰화
inputs = processor(images=image, text=instruction, return_tensors="pt")
input_embeds = vlm.get_input_embeddings()(inputs["input_ids"])  # (B, seq_len, D)

# 2) Learnable action token을 시퀀스 끝에 concat
action_token_embed = action_token.expand(B)  # (B, 1, D)
full_embeds = torch.cat([input_embeds, action_token_embed], dim=1)  # (B, seq_len+1, D)

# attention mask도 1 추가
full_attn_mask = torch.cat([
    inputs["attention_mask"],
    torch.ones(B, 1, device=device)
], dim=1)

# 3) VLM forward (inputs_embeds 사용 -- input_ids 대신)
output = vlm(
    inputs_embeds=full_embeds,
    attention_mask=full_attn_mask,
    pixel_values=inputs["pixel_values"],
    output_hidden_states=True,
)

# 4) 마지막 토큰(<ACTION>) 위치의 hidden state 추출
last_hidden = output.hidden_states[-1]  # (B, seq_len+1, D)
action_hidden = last_hidden[:, -1, :]   # (B, D) -- 항상 마지막 위치

# 5) Action head로 전체 chunk 예측 & L1 loss
predicted_actions = action_head(action_hidden)  # (B, chunk_size, action_dim)
loss = F.l1_loss(predicted_actions, gt_actions)
```

### 2-4. 추론 시

```python
@torch.no_grad()
def predict_action(self, image, instruction):
    inputs = self.processor(images=image, text=instruction, return_tensors="pt")
    input_embeds = self.vlm.get_input_embeddings()(inputs["input_ids"])

    # Learnable token concat (학습된 embedding 그대로 사용)
    action_token_embed = self.action_token.expand(1)
    full_embeds = torch.cat([input_embeds, action_token_embed], dim=1)
    full_attn_mask = torch.cat([
        inputs["attention_mask"],
        torch.ones(1, 1, device=device)
    ], dim=1)

    output = self.vlm(
        inputs_embeds=full_embeds,
        attention_mask=full_attn_mask,
        pixel_values=inputs["pixel_values"],
        output_hidden_states=True,
    )

    action_hidden = output.hidden_states[-1][:, -1, :]  # (1, D)
    normalized_actions = self.action_head(action_hidden)  # (1, chunk_size, action_dim)
    return self._unnormalize(normalized_actions)
```

**OFT 추론과의 차이:**
- OFT: action token embedding을 0으로 마스킹해서 정보 누출 방지 필요
- 본 계획: learnable token은 어차피 action 정보를 포함하지 않으므로 마스킹 불필요. 학습/추론 시 동일한 embedding 사용

---

## 3. SmolVLM2 + Domain-Mix Training 이식 계획

### 3-1. 리포 구조 (LeRobot fork 기준)

```
lerobot/
+-- lerobot/
|   +-- common/
|   |   +-- policies/
|   |   |   +-- smolvlm_act/
|   |   |       +-- __init__.py
|   |   |       +-- configuration_smolvlm_act.py  # (A) 설정
|   |   |       +-- modeling_smolvlm_act.py        # (B) 메인 policy 모델
|   |   |       +-- action_head.py                 # (C) Simple MLP action head
|   |   |       +-- learnable_token.py             # (D) Learnable action token
|   |   +-- ...
|   +-- configs/
|       +-- policy/
|           +-- smolvlm_act.yaml                   # (E) 학습 config
+-- scripts/
|   +-- train_domain_mix_smolvlm.py                # (F) 도메인 혼합 학습 스크립트
+-- ...
```

### 3-2. 도메인 혼합 데이터셋 구성

`train_domain_mix.py`의 도메인 혼합 전략을 그대로 채택:

```python
DOMAINS = [
    # 시뮬레이션 (마스크 있음 -> bg_augment ON)
    DomainDatasetConfig(name="maniskill-franka", weight=6.0, bg_augment_enable=True),
    DomainDatasetConfig(name="maniskill-xarm",   weight=1.0, bg_augment_enable=True),
    # 실제 로봇 (마스크 없음 -> bg_augment OFF)
    DomainDatasetConfig(name="oxe-bridge",       weight=2.0, bg_augment_enable=False),
    DomainDatasetConfig(name="oxe-fractal",      weight=1.0, bg_augment_enable=False),
]
```

| 도메인 | 타입 | Action Key | 비율 | bg_augment |
|--------|------|------------|------|------------|
| maniskill-franka | 시뮬레이션 | `action.ee_delta_pose` | 6 | O |
| maniskill-xarm | 시뮬레이션 | `action.ee_delta_pose` | 1 | O |
| oxe-bridge | 실제 | `action` | 2 | X |
| oxe-fractal | 실제 | `action` | 1 | X |

### 3-3. 구현 Step-by-Step

#### Step 1: Learnable Action Token 모듈

`learnable_token.py` 생성:
- `nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)` -- 작은 값으로 초기화
- `expand(batch_size)` 메서드 제공

#### Step 2: Action Head 모듈

`action_head.py` 생성:
- 3-layer MLP: `LayerNorm -> Linear -> ReLU -> Linear -> ReLU -> Linear`
- 입력: `(B, hidden_dim)` -- 단일 토큰의 hidden state
- 출력: `(B, chunk_size, action_dim)` -- 전체 action chunk

#### Step 3: SmolVLM-Act Policy 모델 (`modeling_smolvlm_act.py`)

```python
class SmolVLMActPolicy(nn.Module):
    """SmolVLM2 + Learnable Action Token + Simple MLP Action Head"""

    def __init__(self, config):
        super().__init__()
        # 1) SmolVLM2 직접 로드
        self.vlm = AutoModelForVision2Seq.from_pretrained(config.vlm_model_name)
        self.processor = AutoProcessor.from_pretrained(config.vlm_model_name)

        # hidden_dim은 모델 크기에 따라 자동 결정 (256M=960, 500M=1536, 2.2B=2048)
        hidden_dim = self.vlm.config.text_config.hidden_size

        # 2) Learnable action token (1개)
        self.action_token = LearnableActionToken(hidden_dim)

        # 3) Action head (simple MLP)
        self.action_head = ActionHead(
            hidden_dim=hidden_dim,
            action_dim=config.action_dim,
            chunk_size=config.chunk_size,
            mlp_hidden=config.action_head_hidden_dim,  # 2048
        )

        # 4) LoRA 적용
        if config.use_lora:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(r=config.lora_rank, target_modules=["q_proj", "v_proj"])
            self.vlm = get_peft_model(self.vlm, lora_config)

    def forward(self, batch):
        """학습: image + text -> VLM -> <ACTION> hidden -> MLP -> actions"""
        B = batch["pixel_values"].shape[0]
        input_embeds = self.vlm.get_input_embeddings()(batch["input_ids"])

        # Learnable token concat
        action_embed = self.action_token.expand(B)
        full_embeds = torch.cat([input_embeds, action_embed], dim=1)
        full_mask = torch.cat([
            batch["attention_mask"],
            torch.ones(B, 1, device=full_embeds.device)
        ], dim=1)

        output = self.vlm(
            inputs_embeds=full_embeds,
            attention_mask=full_mask,
            pixel_values=batch["pixel_values"],
            output_hidden_states=True,
        )

        # 마지막 토큰의 hidden state
        action_hidden = output.hidden_states[-1][:, -1, :]  # (B, D)
        predicted = self.action_head(action_hidden)  # (B, chunk_size, action_dim)
        loss = F.l1_loss(predicted, batch["action"])
        return loss
```

#### Step 4: 도메인 혼합 데이터셋

`train_domain_mix.py`의 `CombinedDomainDataset` 패턴을 그대로 차용:
- `WeightedRandomSampler`로 도메인 비율 제어 (6:1:2:1)
- 도메인별 `bg_augment` (시뮬레이션만)
- `DomainRandomizationConfig`로 온라인 augmentation (전체)
- Action key 리매핑 (`action.ee_delta_pose` -> `action`)
- NaN padding을 0으로 대체
- 도메인별 통계 집계 (`agg_stats`)

**`train_domain_mix.py`와의 차이점 (데이터 부분):**
- SmolVLM2 processor로 직접 이미지/텍스트 전처리 (SmolVLA의 preprocessor 미사용)
- `input_ids` 생성 후 learnable token은 model forward 시 concat (dataset에서는 안함)
- action mask 불필요 (위치가 항상 마지막 1개이므로)

#### Step 5: Data Augmentation

`train_domain_mix.py`와 동일:

```python
image_transforms_cfg = ImageTransformsConfig(enable=True, max_num_transforms=2)

bg_augment = BgAugmentConfig(
    enable=True, texture_dir=TEXTURE_DIR,
    p_bg=0.8, bg_mode="both", mask_subdir="masks", texture_resolution=256,
)

domain_randomization = DomainRandomizationConfig(
    enable=True, p=0.7,
    enable_lighting=True, enable_noise=True, enable_crop=True,
    lighting_gain_range=(0.3, 2.0), noise_iso_range=(1, 4),
)
```

#### Step 6: 학습 Config

```python
VLM_MODEL = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"  # 또는 256M, 2.2B 변형
BATCH_SIZE = 8
STEPS = 50_000
LR = 5e-5
CHUNK_SIZE = 50
ACTION_DIM = 7
ACTION_HEAD_HIDDEN = 2048
USE_LORA = True
LORA_RANK = 32
OUTPUT_DIR = Path("outputs/train/domain_mix_smolvlm_act")
```

#### Step 7: 학습 루프

```python
for step in range(1, STEPS + 1):
    batch = next(dl_iter)

    # Forward: VLM에 learnable token 붙여서 통과 -> 마지막 hidden -> action head
    loss = policy.forward(batch)

    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in policy.parameters() if p.requires_grad],
        grad_clip_norm,
    )
    optimizer.step()
    optimizer.zero_grad()
    lr_scheduler.step()
```

**학습되는 파라미터:**
1. `action_token.token_embedding` -- learnable token 1개 (hidden_dim floats)
2. `action_head` 전체 -- MLP 파라미터
3. VLM의 LoRA adapter -- q_proj, v_proj

---

## 4. 접근법 비교

| 항목 | OFT (다수 action token) | 본 계획 (single learnable token) | SmolVLA (flow matching) |
|------|------------------------|----------------------------------|------------------------|
| **Action token 수** | chunk_len x action_dim (350개) | **1개** | 없음 |
| **Token 종류** | Special tokens (고정 embedding) | **Learnable nn.Parameter** | N/A |
| **Decoding** | Bidirectional (전체 동시) | **Autoregressive (마지막 1개)** | Iterative denoising |
| **Hidden state 추출** | 여러 위치에서 추출 + concat | **마지막 위치 1개만** | Expert layer output |
| **Action head** | MLPResNet (2 residual blocks) | **Simple 3-layer MLP** | Flow matching decoder |
| **추론 시 마스킹** | Action token을 0으로 마스킹 필요 | **불필요 (learnable이므로)** | N/A |
| **Loss** | L1 | **L1** | MSE |
| **추론 속도** | 1 forward pass | **1 forward pass** | N denoising steps |
| **구현 복잡도** | 중간 (token 관리, mask 생성) | **낮음** | 높음 |

---

## 5. 주의사항 & 결정 포인트

| 항목 | 고려사항 |
|------|---------|
| **inputs_embeds 사용** | learnable token을 concat하려면 `input_ids`가 아닌 `inputs_embeds`로 VLM에 입력해야 함. SmolVLM2가 `inputs_embeds` + `pixel_values` 동시 입력을 지원하는지 확인 필요 |
| **Hidden dim** | SmolVLM2 모델 크기에 따라 다름 (256M=960, 500M=1536, 2.2B=2048). `vlm.config.text_config.hidden_size`에서 자동 추출 |
| **Action head 크기** | hidden_dim 1536에서 chunk_size(50) x action_dim(7) = 350차원을 출력. 병목이 될 수 있으므로 중간 hidden을 2048 정도로 설정 |
| **Learnable token 초기화** | 너무 크면 VLM hidden state 분포를 교란. `N(0, 0.02)` 정도로 작게 초기화 |
| **LoRA vs Full fine-tune** | 1개 토큰의 hidden state에 정보를 집약하려면 VLM backbone도 어느 정도 adaptation 필요. LoRA rank 32~64 권장 |
| **모델 크기 선택** | SmolVLM2-256M(빠른 실험), 500M(기본), 2.2B(성능 우선). `config.vlm_model_name`으로 전환 가능 |
| **Chunk size 50** | 단일 벡터에서 350차원(50x7)을 예측하는 건 부담이 될 수 있음. chunk_size를 줄이거나 action head를 키우는 실험 고려 |
| **Normalization** | 도메인별 통계 집계 필요 (`train_domain_mix.py`의 `agg_stats` 참조) |
| **Camera/Action key** | 도메인별 리매핑 필요 (기존 `train_domain_mix.py` 로직 재사용) |

---

## 6. 구현 우선순위

1. **Learnable action token** 모듈 구현 (가장 단순, 독립적)
2. **Action head (MLP)** 구현 & 단위 테스트
3. **SmolVLM2 forward pass 검증** -- `inputs_embeds` + `pixel_values` 동시 입력, `output_hidden_states=True` 확인
4. **Policy wrapper** -- learnable token concat -> VLM forward -> hidden 추출 -> action head 연결
5. **도메인 혼합 데이터 파이프라인** -- `train_domain_mix.py`의 `CombinedDomainDataset` + `WeightedRandomSampler` 차용
6. **학습 스크립트** (`train_domain_mix_smolvlm.py`) -- 학습 루프 통합
7. **추론 / 평가** -- unnormalize + action chunk selection

---

## 7. 핵심 요약

> **OpenVLA-OFT가 하는 것:** VLM input sequence 끝에 `chunk_len x action_dim`개의 action token을 넣고, bidirectional decoding으로 각 위치의 hidden states를 뽑아 MLPResNet에 통과시켜 연속 action을 예측한다.
>
> **우리가 할 것:** SmolVLM2 input sequence 끝에 **learnable token 1개만** 추가하고, 해당 위치의 single hidden state를 간단한 3-layer MLP에 넣어 **전체 action chunk `(chunk_size, action_dim)`을 한번에** 예측한다. 학습은 LoRA + learnable token + action head 파라미터를 업데이트하며, `train_domain_mix.py`의 도메인 혼합 전략(4개 데이터셋, WeightedRandomSampler, bg_augment, domain_randomization)을 활용한다. OFT 대비 구현이 훨씬 단순하고, 추론 시 별도의 token masking도 불필요하다.
