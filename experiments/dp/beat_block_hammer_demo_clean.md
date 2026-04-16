# DP Experiment: `beat_block_hammer` on `demo_clean`

## Summary

This report documents a RoboTwin **Diffusion Policy (DP)** run on the `beat_block_hammer` task. The purpose of the run was to establish a reproducible local baseline under realistic hardware limits rather than to maximize leaderboard performance.

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

## Experimental Context

The standard RoboTwin DP workflow was used for preprocessing, training, and evaluation. The only practical adaptation was to reduce batch size so that the run would fit on an **8 GB RTX 4060 Ti**. The default DP configuration exceeded local VRAM, so both training and validation batch size were lowered to `8`.

## Setup

- Framework: RoboTwin `policy/DP`
- GPU: `NVIDIA GeForce RTX 4060 Ti`
- Config: `robot_dp_14.yaml`
- Training epochs: `600`
- Checkpoints observed: `300.ckpt`, `600.ckpt`
- Logging mode: `offline`
- Evaluation episodes: `100`

## Workflow

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

A `33.0%` success rate on `100` evaluation episodes shows that DP learned part of the task from a modest expert dataset, but the policy remains far from robust under unseen instructions. This makes the run a useful baseline and comparison anchor for later RoboTwin experiments.

## Resource-Constrained Notes

This run was carried out on an `RTX 4060 Ti 8GB`, so the stock RoboTwin DP setting had to be adjusted to avoid OOM. Reducing batch size was an engineering necessity rather than an algorithmic change. That distinction matters because the purpose of this run is to document a realistic local baseline, not to present an idealized large-GPU result.

## What This Run Demonstrates

- end-to-end DP preprocessing, training, checkpointing, and evaluation on RoboTwin
- stable adaptation of the pipeline to consumer-GPU hardware
- a baseline result that can be compared directly against ACT on the same task and data scale

## Evidence

- A DP training-loss curve was generated locally from `logs.json.txt`.
- Rollout videos were generated during evaluation in the local RoboTwin eval directory recorded below.

## Artifacts

- Raw result file: [results/dp/beat_block_hammer/demo_clean/result.txt](../../results/dp/beat_block_hammer/demo_clean/result.txt)
- Legacy raw result path preserved: `results/beat_block_hammer_dp_demo_clean/result.txt`
- Local eval path: `eval_result/beat_block_hammer/DP/demo_clean/demo_clean/2026-04-15 11:36:32/_result.txt`
- Local checkpoints: `policy/DP/checkpoints/beat_block_hammer-demo_clean-50-0/300.ckpt`, `policy/DP/checkpoints/beat_block_hammer-demo_clean-50-0/600.ckpt`

## Next Work

- Compare against ACT on the same task and split
- Repeat with more demonstrations and summarize scaling behavior