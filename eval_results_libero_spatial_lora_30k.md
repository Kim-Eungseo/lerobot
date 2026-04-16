# SmolVLM-Act LIBERO Spatial Evaluation Results

## Best Results Summary (Original LIBERO, 10 tasks × 50 episodes)

| Rank | Model | Pooling | Head | Best Step | **Overall SR** |
| ---- | ----- | ------- | ---- | --------- | -------------- |
| 1 | **2.2B Attentive + MLP** | attentive | MLP | 190k | **84.2%** |
| 2 | 2.2B AT + MLP (run2) | action_token | MLP | 190k | 83.8% |
| 3 | 2.2B AT + MLP (run2 130k) | action_token | MLP | 130k/140k | 83.4% |
| 4 | 2.2B AT + MLP (run1) | action_token | MLP | 110k | 81.4% |
| 5 | 500M Attentive + MLP | attentive | MLP | 90k | 80.4% |
| 6 | 500M AT + MLP | action_token | MLP | 110k | 78.4% |
| 7 | 500M Diffusion (mislabeled) | action_token | MLP run2 | 120k | 72.2% |
| 8 | 500M Full FT | action_token | MLP | 40k | 59.6% |
| 9 | 2.2B Attentive + Diffusion | attentive | Diffusion | 120k | 45.2% |
| 10 | 2.2B AT + Diffusion | action_token | Diffusion | 120k | 34.8% |

## 2.2B Step Scaling

### Attentive + MLP (Best so far)

| Step | SR |
| ---- | -- |
| 90k  | 80.6% |
| 100k | 81.0% |
| 110k | 82.2% |
| 120k | 81.6% |
| 130k | 82.6% |
| 150k | 82.0% |
| 170k | 81.0% |
| **190k** | **84.2%** |
| 200k | 80.8% |

### Action Token + MLP — Run2

| Step | SR |
| ---- | -- |
| 90k  | 83.2% |
| 120k | 82.8% |
| 130k | 83.4% |
| 140k | 83.4% |
| 150k | 78.0% |
| 170k | 80.2% |
| **190k** | **83.8%** |
| 200k | 83.0% |

### Action Token + MLP — Run1 (different seed)

| Step | SR |
| ---- | -- |
| 60k  | 77.8% |
| 90k  | 81.2% |
| **110k** | **81.4%** |
| 120k | 77.2% |

### Attentive + MLP, chunk_size=20

| Step | SR |
| ---- | -- |
| 90k  | 74.8% |
| **110k** | **79.0%** |
| 120k | 64.2% |

→ chunk_size=20 → chunk_size=10보다 일관되게 낮음 (open-loop 누적 에러)

### Action Token + Diffusion (real)

| Step | SR |
| ---- | -- |
| 90k  | 20.2% |
| **120k** | **34.8%** |
| 150k | 27.4% |

### Attentive + Diffusion

| Step | SR |
| ---- | -- |
| 90k  | 25.8% |
| **120k** | **45.2%** |
| 150k | 32.4% |

### Temporal Ensemble Experiments (run2 130k checkpoint)

| 설정 | SR | vs Baseline |
| ---- | -- | ----------- |
| **Baseline (exec=10)** | **83.4%** | - |
| exec=5 (no ensemble) | 81.0% | -2.4 |
| ensemble uniform (exec=5) | 82.6% | -0.8 |
| ensemble exp=0.5 (exec=5) | 79.4% | -4.0 |

→ Temporal ensemble은 도움 안 됨. 이 모델은 chunk=10 그대로 실행이 최적.

## 500M Step Scaling

### Action Token + MLP

| Step | SR |
| ---- | -- |
| 30k  | 57.0% |
| 50k  | 68.4% |
| 70k  | 70.4% |
| 80k  | 72.2% |
| 90k  | 73.6% |
| 100k | 72.6% |
| **110k** | **78.4%** |
| 120k | 73.6% |

### Attentive + MLP

| Step | SR |
| ---- | -- |
| **90k** | **80.4%** |
| 100k | 78.4% |
| 110k | 79.4% |
| 120k | 79.0% |

### "Diffusion" (actually MLP run2)

| Step | SR |
| ---- | -- |
| 90k  | 69.2% |
| 100k | 71.4% |
| 110k | 71.2% |
| **120k** | **72.2%** |

### 500M LoRA vs Full FT (40k)

| Task                             | LoRA 40k | Full FT 40k |
| -------------------------------- | -------- | ----------- |
| **Overall**                      | 57.0%    | **59.6%**   |

## Key Findings

1. **Attentive pooling > Action token**: 500M (+2%p), 2.2B (+0.4%p) — 일관된 개선
2. **2.2B > 500M**: ~5%p 차이, 2.2B는 더 빨리 수렴
3. **MLP head >> Diffusion head**: Diffusion은 35-45%로 MLP의 절반 수준 (이 데이터 규모에서 수렴 못함)
4. **chunk_size=10 > chunk_size=20**: open-loop 길수록 누적 에러
5. **Temporal ensemble 효과 없음**: 재예측 빈도 증가가 오히려 약간의 손실
6. **장기 학습 효과**: 130k 이후 큰 개선 없이 수렴 (130k~200k 구간 ~83-84% 변동)
7. **"bowl on ramekin"이 가장 어려움**: 모든 모델에서 8-30%

## Config

| | 500M | 2.2B |
|---|---|---|
| VLM | SmolVLM2-500M-Video-Instruct | SmolVLM2-2.2B-Instruct |
| LoRA rank | 32 | 32 |
| Chunk Size | 10 | 10 (or 20) |
| LR | 5e-5 | 5e-5 |
| Batch Size | 8 | 8 |
| Dataset | lerobot/libero_spatial_image | lerobot/libero_spatial_image |

## Checkpoints

| Experiment | Location |
| ---------- | -------- |
| 500M MLP LoRA (30k-120k) | `/home/ngseo/lerobot/outputs/train/smolvlm_act_libero_spatial_lora_mlp/` |
| 500M Full FT (40k) | `/home/ngseo/lerobot/outputs/train/smolvlm_act_libero_spatial_fullft_mlp_40k/` |
| 500M Attentive (90k best) | `/data/smolvlm_outputs/lora_mlp_attentive/` |
| 500M "Diffusion" = MLP run2 (120k) | `/data/smolvlm_outputs/lora_diffusion_120k/` |
| 2.2B AT MLP run1 | `/home/ngseo/lerobot/outputs/train/smolvlm_act_libero_spatial_lora_mlp_2B/` + `/data/smolvlm_outputs/lora_mlp_2B/` |
| 2.2B AT MLP run2 (130k-140k) | `/data/smolvlm_outputs/lora_diffusion_2B_150k/` (mislabeled name) |
| 2.2B AT MLP run2 200k (150k-200k) | `/data/smolvlm_outputs/lora_mlp_2B_200k/` |
| 2.2B Attentive 120k | `/data/smolvlm_outputs/lora_mlp_attentive_2B/` |
| 2.2B Attentive 200k (120k-200k) | `/data/smolvlm_outputs/lora_mlp_attentive_2B_200k/` |
| 2.2B Attentive chunk=20 | `/data/smolvlm_outputs/lora_attentive_mlp_2B_chunk20/` |
| 2.2B AT + Diffusion (real) 150k | `/data/smolvlm_outputs/lora_diffusion_2B_150k_real/` |
| 2.2B Attentive + Diffusion 150k | `/data/smolvlm_outputs/lora_attentive_diffusion_2B_150k/` |
| Eval results | `/data/smolvlm_outputs/eval/` and `/home/ngseo/lerobot/outputs/eval/` |
