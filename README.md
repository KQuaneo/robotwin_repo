# RoboTwin Experiment Report

## Overview

This repository records a complete baseline experiment for **RoboTwin Diffusion Policy (DP)** on the `beat_block_hammer` task. The intent is to document a reproducible engineering run rather than post an isolated metric without context. The page now captures the task, training conditions, machine constraints, commands, evaluation protocol, and result interpretation in one place.

## Headline Result

| Item | Value |
| --- | --- |
| Policy | `DP` |
| Task | `beat_block_hammer` |
| Training setting | `demo_clean` |
| Expert demonstrations | `50` |
| Final checkpoint | `600.ckpt` |
| Evaluation setting | `demo_clean` |
| Instruction type | `unseen` |
| Success rate | `33/100 = 33.0%` |

## Why This Run Matters

This experiment establishes a concrete RoboTwin DP baseline under resource constraints that are common on individual workstations. The original RoboTwin DP configuration exceeded the memory budget of the local GPU, so the run was adapted to fit an **8 GB RTX 4060 Ti** by lowering both training and validation batch size to `8`. The core DP workflow, task definition, preprocessing path, training duration, and evaluation entrypoint remained aligned with the standard RoboTwin pipeline.

That makes the result useful as a reproducible baseline and a reference point for later comparisons against ACT or stronger policies.

## Experimental Setup

- Framework: RoboTwin `policy/DP`
- GPU: `NVIDIA GeForce RTX 4060 Ti`
- Policy config: `robot_dp_14.yaml`
- Training epochs: `600`
- Checkpoints saved: `300.ckpt`, `600.ckpt`
- Logging mode: `offline`
- Evaluation episodes: `100`

## Methodology

### Dataset Preparation

The DP model was trained on the processed Zarr dataset generated from `50` expert demonstrations collected for `beat_block_hammer` under the `demo_clean` setting.

### Training

The model was trained for `600` epochs using the RoboTwin DP training entrypoint. Because the default configuration OOMed on the local GPU, batch sizes were reduced to make the run stable.

### Evaluation

The final `600.ckpt` checkpoint was evaluated on `100` episodes with instruction type `unseen`, producing a final success rate of `33.0%`.

## Reproducible Commands

### 1. Preprocess

```bash
conda activate robotwin
cd /home/kai/robotwin/policy/DP
bash process_data.sh beat_block_hammer demo_clean 50
```

### 2. Train

```bash
conda activate robotwin
cd /home/kai/robotwin/policy/DP

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train.py \
  --config-name=robot_dp_14.yaml \
  task.name=beat_block_hammer \
  task.dataset.zarr_path=data/beat_block_hammer-demo_clean-50.zarr \
  training.debug=False \
  training.seed=0 \
  training.device=cuda:0 \
  exp_name=beat_block_hammer-robot_dp-train \
  logging.mode=offline \
  setting=demo_clean \
  expert_data_num=50 \
  head_camera_type=D435 \
  dataloader.batch_size=8 \
  val_dataloader.batch_size=8
```

### 3. Evaluate

```bash
conda activate robotwin
cd /home/kai/robotwin/policy/DP
bash eval.sh beat_block_hammer demo_clean demo_clean 50 0 0
```

## Interpretation

A `33.0%` success rate on `100` evaluation episodes shows that the DP policy captured part of the task structure from a modest expert dataset, but still has significant room to improve in robustness and generalization under unseen language instructions. This is a serious baseline, not a polished final result.

## Stored Artifacts

- Raw evaluation output: `results/beat_block_hammer_dp_demo_clean/result.txt`
- Local eval path: `eval_result/beat_block_hammer/DP/demo_clean/demo_clean/2026-04-15 11:36:32/_result.txt`
- Local checkpoints: `policy/DP/checkpoints/beat_block_hammer-demo_clean-50-0/300.ckpt`, `policy/DP/checkpoints/beat_block_hammer-demo_clean-50-0/600.ckpt`

## Next Upgrades

- Add a training-loss curve exported from `logs.json.txt`
- Add one successful and one failed rollout video or GIF
- Compare DP against ACT on the same task and split
- Scale the number of demonstrations and report trend lines
