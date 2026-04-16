# ACT Experiment: `beat_block_hammer` on `demo_clean`

## Summary

This report documents a RoboTwin **Action Chunking Transformer (ACT)** run on the `beat_block_hammer` task. The goal of the run was to establish a machine-safe ACT baseline on the same task and dataset used for the DP baseline so that the two policies could be compared directly.

## Headline Result

| Item | Value |
| --- | --- |
| Policy | `ACT` |
| Task | `beat_block_hammer` |
| Training setting | `demo_clean` |
| Expert demonstrations | `50` |
| Main checkpoint used for eval | `policy_best.ckpt` |
| Evaluation setting | `demo_clean` |
| Instruction type | `unseen` |
| Success rate | `32/100 = 32.0%` |

## Experimental Context

The ACT pipeline in RoboTwin was trained on the same `beat_block_hammer` `demo_clean` dataset used for the DP baseline. To fit the local **8 GB RTX 4060 Ti** safely and avoid overwriting an older ACT run, training was launched into a separate checkpoint folder with `batch_size=1`.

This makes the run comparable to the DP baseline while remaining practical on local hardware.

## Setup

- Framework: RoboTwin `policy/ACT`
- GPU: `NVIDIA GeForce RTX 4060 Ti`
- Task alias: `sim-beat_block_hammer-demo_clean-50`
- Training epochs: `500`
- Checkpoint directory: `policy/ACT/act_ckpt/act-beat_block_hammer/demo_clean-50-b1`
- Main evaluation checkpoint: `policy_best.ckpt`
- Evaluation episodes: `100`
- Temporal aggregation: `true` during evaluation

## Workflow

### 1. Preprocess

```bash
conda activate robotwin
cd /home/kai/robotwin/policy/ACT
bash process_data.sh beat_block_hammer demo_clean 50
```

### 2. Train

```bash
conda activate robotwin
cd /home/kai/robotwin/policy/ACT

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 imitate_episodes.py \
  --task_name sim-beat_block_hammer-demo_clean-50 \
  --ckpt_dir ./act_ckpt/act-beat_block_hammer/demo_clean-50-b1 \
  --policy_class ACT \
  --kl_weight 10 \
  --chunk_size 50 \
  --hidden_dim 512 \
  --batch_size 1 \
  --dim_feedforward 3200 \
  --num_epochs 500 \
  --lr 1e-5 \
  --save_freq 2000 \
  --state_dim 14 \
  --seed 0
```

### 3. Evaluate

The stock `policy/ACT/eval.sh` appends `-${expert_data_num}` to the checkpoint-setting folder, which does not work for a custom folder like `demo_clean-50-b1`. The evaluation was therefore launched directly with the exact checkpoint directory.

```bash
conda activate robotwin
cd /home/kai/robotwin
CUDA_VISIBLE_DEVICES=0 PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py \
  --config policy/ACT/deploy_policy.yml \
  --overrides \
  --task_name beat_block_hammer \
  --task_config demo_clean \
  --ckpt_setting demo_clean-50-b1 \
  --ckpt_dir policy/ACT/act_ckpt/act-beat_block_hammer/demo_clean-50-b1 \
  --seed 0 \
  --temporal_agg true
```

## Interpretation

A `32.0%` success rate on `100` evaluation episodes puts ACT very close to the DP baseline on this task and dataset scale. On this machine and data regime, ACT did not clearly outperform DP, but it produced a credible baseline and a useful direct comparison point.

## Resource-Constrained Notes

This ACT run was scoped around the same `RTX 4060 Ti 8GB` limit as the DP baseline. To keep the run stable and to avoid overwriting an earlier ACT directory, training used a dedicated checkpoint folder and `batch_size=1`. The project intent here is to show that the ACT pipeline can be trained, evaluated, and documented cleanly under consumer-GPU limits.

## What This Run Demonstrates

- end-to-end ACT preprocessing, training, checkpointing, and evaluation on RoboTwin
- machine-safe reproduction of ACT under limited VRAM
- a directly comparable ACT baseline against the DP run on the same task and data scale

## Artifacts

- Raw result file: [results/act/beat_block_hammer/demo_clean_50_b1/result.txt](../../results/act/beat_block_hammer/demo_clean_50_b1/result.txt)
- Local eval path: `eval_result/beat_block_hammer/ACT/demo_clean/demo_clean-50-b1/2026-04-15 17:39:05/_result.txt`
- Local checkpoint dir: `policy/ACT/act_ckpt/act-beat_block_hammer/demo_clean-50-b1`
- Rollout videos: `episode0.mp4` through `episode99.mp4` in the eval folder

## Next Work

- compare ACT and DP in a dedicated summary page
- add selected rollout media for qualitative inspection
- test whether different ACT hyperparameters improve robustness on unseen instructions
