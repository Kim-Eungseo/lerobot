# SmolVLM-Act LIBERO Spatial Evaluation Results

## Config


|                | MLP LoRA 30k                 | MLP LoRA 40k                 | MLP Full FT 40k              | Diffusion LoRA 30k           |
| -------------- | ---------------------------- | ---------------------------- | ---------------------------- | ---------------------------- |
| VLM            | SmolVLM2-500M-Video-Instruct | SmolVLM2-500M-Video-Instruct | SmolVLM2-500M-Video-Instruct | SmolVLM2-500M-Video-Instruct |
| Action Head    | MLP                          | MLP                          | MLP                          | Diffusion (DDIM 10 steps)    |
| LoRA           | rank 32                      | rank 32                      | No (full finetune)           | rank 32                      |
| Training Steps | 30,000                       | 40,000                       | 40,000                       | 30,000                       |
| Chunk Size     | 10                           | 10                           | 10                           | 10                           |
| LR             | 5e-5                         | 5e-5                         | 5e-5                         | 5e-5                         |
| Batch Size     | 8                            | 8                            | 8                            | 8                            |
| Dataset        | lerobot/libero_spatial_image | lerobot/libero_spatial_image | lerobot/libero_spatial_image | lerobot/libero_spatial_image |


## Overall Results (LIBERO-plus, 258 task variants)


| Model          | Overall SR | Successes | Total Episodes | Total Tasks |
| -------------- | ---------- | --------- | -------------- | ----------- |
| **MLP LoRA**   | **5.8%**   | 298       | 5,160          | 258         |
| Diffusion LoRA | 2.4%       | 125       | 5,160          | 258         |


- 20 episodes per task, 280 max steps per episode

## Per Base Task Breakdown


| Base Task                                        | MLP avg | MLP max | Diff avg | Diff max | # variants |
| ------------------------------------------------ | ------- | ------- | -------- | -------- | ---------- |
| between the plate and the ramekin -> plate       | 6.2%    | 30%     | 0.9%     | 15%      | 29         |
| from table center -> plate                       | 4.0%    | 60%     | 9.6%     | 50%      | 46         |
| in the top drawer of the wooden cabinet -> plate | 2.7%    | 20%     | 0.2%     | 5%       | 26         |
| next to the cookie box -> plate                  | 12.3%   | 25%     | 0.0%     | 0%       | 20         |
| next to the plate -> plate                       | 0.0%    | 0%      | 0.0%     | 0%       | 6          |
| next to the ramekin -> plate                     | 0.5%    | 5%      | 0.0%     | 0%       | 10         |
| on the cookie box -> plate                       | 24.4%   | 75%     | 4.5%     | 15%      | 31         |
| on the ramekin -> plate                          | 1.1%    | 10%     | 0.4%     | 10%      | 35         |
| on the stove -> plate                            | 0.0%    | 0%      | 0.0%     | 0%       | 41         |
| on the wooden cabinet -> plate                   | 0.7%    | 5%      | 0.0%     | 0%       | 14         |


## Original LIBERO Results (10 tasks, 50 episodes/task)

### MLP LoRA bs8 — Step Scaling

| Task                             | 30k   | 40k   | 50k    | 60k   | 70k   | **80k**  |
| -------------------------------- | ----- | ----- | ------ | ----- | ----- | -------- |
| between plate & ramekin -> plate | 84%   | 84%   | **100%** | 98% | 86%   | 80%      |
| from table center -> plate       | 40%   | 40%   | **82%** | 66%  | 76%   | 74%      |
| in top drawer -> plate           | 72%   | 72%   | 74%    | 74%   | 80%   | **86%**  |
| next to cookie box -> plate      | **98%** | **98%** | 94%  | **98%** | 96% | 96%      |
| next to plate -> plate           | **74%** | **74%** | 68%  | 66%   | **74%** | 72%   |
| next to ramekin -> plate         | 72%   | 72%   | 74%    | 60%   | **78%** | 72%    |
| on cookie box -> plate           | 72%   | 72%   | 84%    | 74%   | 80%   | **90%**  |
| on ramekin -> plate              | 24%   | 24%   | 20%    | 10%   | 26%   | **30%**  |
| on stove -> plate                | 4%    | 4%    | 34%    | 26%   | 40%   | **52%**  |
| on wooden cabinet -> plate       | 30%   | 30%   | 54%    | **70%** | 68% | **70%**  |
| **Overall**                      | 57.0% | 57.0% | 68.4%  | 64.2% | 70.4% | **72.2%** |

### MLP LoRA vs Full FT comparison (40k)

| Task                             | LoRA 40k | Full FT 40k |
| -------------------------------- | -------- | ----------- |
| next to cookie box -> plate      | 92%      | **94%**     |
| between plate & ramekin -> plate | **86%**  | 74%         |
| from table center -> plate       | 70%      | **78%**     |
| in top drawer -> plate           | 70%      | **84%**     |
| next to plate -> plate           | **66%**  | 60%         |
| next to ramekin -> plate         | 58%      | **86%**     |
| on cookie box -> plate           | **78%**  | 58%         |
| on wooden cabinet -> plate       | 28%      | **38%**     |
| on ramekin -> plate              | 12%      | **24%**     |
| on stove -> plate                | **10%**  | 0%          |
| **Overall**                      | 57.0%    | **59.6%**   |

### LIBERO-plus Overall SR

| Model              | LIBERO-plus SR |
| ------------------ | -------------- |
| LoRA bs8 30k       | 5.8%           |
| LoRA bs8 80k       | **15.6%**      |
| Full FT 40k        | 10.1%          |
| Diffusion LoRA 30k | 2.4%           |

## Notes

- Original LIBERO: 10 tasks, 50 episodes per task, 280 max steps per episode
- LIBERO-plus: 258 task variants with different table configurations, 20 episodes per task
- Checkpoints at `outputs/train/smolvlm_act_libero_spatial_{lora_mlp,lora_diffusion,fullft_mlp_40k}/`
- MLP head significantly outperforms diffusion head
- Longer training consistently improves: 30k (57%) -> 50k (68%) -> 80k (**72%**)
- "bowl on stove" improved dramatically with more training: 4% (30k) -> 52% (80k)
- bs32 80k experiment in progress on GPU 2

