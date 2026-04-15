# RoboTwin Results

## DP on `beat_block_hammer`

- Policy: `DP`
- Task: `beat_block_hammer`
- Training setting: `demo_clean`
- Expert demonstrations: `50`
- Final checkpoint: `600.ckpt`
- Evaluation setting: `demo_clean`
- Instruction type: `unseen`
- Success rate: `33/100 = 33.0%`

## Reproduction

### Preprocess

```bash
conda activate robotwin
cd /home/kai/robotwin/policy/DP
bash process_data.sh beat_block_hammer demo_clean 50
```

### Train

This machine used reduced batch sizes to fit an 8 GB GPU.

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

### Evaluate

```bash
conda activate robotwin
cd /home/kai/robotwin/policy/DP
bash eval.sh beat_block_hammer demo_clean demo_clean 50 0 0
```

## Artifacts

- Checkpoints observed locally: `300.ckpt`, `600.ckpt`
- Eval result: `eval_result/beat_block_hammer/DP/demo_clean/demo_clean/2026-04-15 11:36:32/_result.txt`
