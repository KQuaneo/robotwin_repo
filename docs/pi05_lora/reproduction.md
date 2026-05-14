# Reproduction Notes

These notes describe the high-level workflow used for the experiment. Paths refer to the original RoboTwin repository layout.

## Remote Workspace

Use a data disk, not the small system disk:

```bash
cd /root/autodl-tmp/robotwin
```

## Preflight

```bash
bash policy/pi05/tools/preflight_5090_32g.sh --smoke
```

On RTX 5090 / Blackwell, CUDA 12.8-compatible JAX and PyTorch wheels are important. Older CUDA 12.4 wheels may fail with `sm_120 is not compatible` or `no kernel image is available`.

## Data Audit

```bash
python script/audit_robotwin_dataset.py audit \
  --task beat_block_hammer \
  --config demo_clean
```

## Data Conversion

```bash
cd policy/pi05
bash process_data_pi05.sh beat_block_hammer demo_clean 50
bash generate.sh \
  processed_data/beat_block_hammer-demo_clean-50 \
  beat_block_hammer-demo_clean-50
```

## Training

The final training run resumed from checkpoint 4000 and trained to 8000:

```bash
cd /root/autodl-tmp/robotwin/policy/pi05
export WANDB_MODE=offline
export HF_HOME=/root/autodl-tmp/hf-cache
export OPENPI_DATA_HOME=/root/autodl-tmp/openpi-cache
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda-12.8
export TORCH_CUDA_ARCH_LIST=12.0
export CUDA_VISIBLE_DEVICES=0,1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

./.venv/bin/python scripts/train.py pi05_base_aloha_lora \
  --exp-name=beat_block_hammer_demo_clean_50_lora \
  --resume \
  --num-train-steps=8000
```

Checkpoint output:

```text
policy/pi05/checkpoints/pi05_base_aloha_lora/beat_block_hammer_demo_clean_50_lora/8000
```

## Evaluation

Evaluation used `script/eval_policy.py` with raw seed override support:

```bash
./policy/pi05/.venv/bin/python script/eval_policy.py \
  --config policy/pi05/deploy_policy.yml \
  --overrides \
  --task_name beat_block_hammer \
  --task_config demo_clean \
  --train_config_name pi05_base_aloha_lora \
  --model_name beat_block_hammer_demo_clean_50_lora \
  --ckpt_setting beat_block_hammer_demo_clean_50_lora_ckpt8000_seed0_first50 \
  --seed 0 \
  --start_seed_raw 100000 \
  --policy_name pi05 \
  --checkpoint_id 8000 \
  --instruction_type unseen \
  --pi0_step 50 \
  --test_num 50
```

For two-GPU evaluation, two independent screen sessions were launched:

```bash
screen -dmS pi05_ckpt8000_eval_first50  bash -lc '<eval on GPU 0, start_seed_raw=100000, test_num=50>'
screen -dmS pi05_ckpt8000_eval_cont50   bash -lc '<eval on GPU 1, start_seed_raw=100067, test_num=50>'
```

## Artifacts

Evaluation logs and videos were stored on the remote machine:

```text
/root/autodl-tmp/robotwin_logs/
/root/autodl-tmp/robotwin/eval_result/beat_block_hammer/pi05/demo_clean/
```

Large artifacts are intentionally not committed to this repository.
