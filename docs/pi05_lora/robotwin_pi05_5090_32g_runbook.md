# RoboTwin pi0.5 on RTX 5090 32G Runbook

这份清单面向租一台 RTX 5090 32G 机器后跑 `policy/pi05`。它只基于当前 RoboTwin 仓库里的入口整理，不假设平台商已经装好了正确环境。

## 先看结论

- 优先跑 `pi05_base_aloha_lora`，不要一上来跑 `pi05_aloha_full_base`。32G 对 full fine-tune 很紧，LoRA 更像第一版可复现路线。
- 租机镜像要选 Linux、Python 3.11、NVIDIA driver + CUDA 12.8 或更新、可访问 Hugging Face/GCS/S3/PyPI。
- 当前 `policy/pi05/uv.lock` 里锁到了 `jax==0.5.0`、CUDA 12.4 相关 wheel、`torch==2.6.0`。RTX 5090 是 Blackwell/sm_120，官方 NVIDIA 架构矩阵显示 Blackwell 首个 CUDA Toolkit 支持是 CUDA 12.8；PyTorch 官方安装页也已经提供 CUDA 12.8 wheel。因此租机后的第一件事是跑 smoke test，确认 JAX/Torch 没有 `no kernel image is available` 或 `sm_120 is not compatible`。
- 如果 smoke test 失败，先换 CUDA 12.8+ 的基础镜像或升级 pi0.5 环境里的 CUDA wheel，再开始花钱训练。

参考：

- NVIDIA CUDA Toolkit/Driver/Architecture Matrix: https://docs.nvidia.com/datacenter/tesla/drivers/cuda-toolkit-driver-and-architecture-matrix.html
- JAX installation docs: https://docs.jax.dev/en/latest/installation.html
- PyTorch start locally: https://pytorch.org/get-started/locally/
- PyTorch previous versions with `cu128`: https://pytorch.org/get-started/previous-versions/

## 租机规格

最低建议：

- GPU: RTX 5090 32G, 单卡即可
- CPU/RAM: 16 vCPU / 64G RAM 起步，数据转换和视频解码会吃 CPU 与内存
- Disk: 300G 起步；如果同时放 raw data、processed data、LeRobot 数据和 checkpoint，建议 500G+
- OS: Ubuntu 22.04/24.04
- Driver/CUDA: driver 能支持 CUDA 12.8+；容器内也用 CUDA 12.8+ runtime
- Python: 3.11，和 `policy/pi05/.python-version` 保持一致

不建议：

- CUDA 12.4/12.6 的旧 PyTorch/JAX 镜像
- Windows 裸环境
- 没公网或不能访问模型权重下载源的机器

## 仓库入口

当前 pi0.5 路径：

- 依赖工程: `policy/pi05/pyproject.toml`
- 数据预处理: `policy/pi05/process_data_pi05.sh`
- 转 LeRobot/ALOHA: `policy/pi05/generate.sh`
- 训练: `policy/pi05/finetune.sh`
- 评测: `policy/pi05/eval.sh`
- 配置定义: `policy/pi05/src/openpi/training/config.py`
- RoboTwin 策略加载: `policy/pi05/pi_model.py`

推荐第一版训练配置：

- `pi05_base_aloha_lora`
- 默认 `batch_size=32`
- 默认 `num_train_steps=30000`
- 默认预训练权重: `s3://openpi-assets/checkpoints/pi05_base/params`

## 上机后 10 分钟检查

在仓库根目录执行：

```bash
bash policy/pi05/tools/preflight_5090_32g.sh
```

如果已经装好 `uv` 和 pi0.5 环境，可以进一步跑 GPU smoke test：

```bash
bash policy/pi05/tools/preflight_5090_32g.sh --smoke
```

要检查某个任务数据：

```bash
bash policy/pi05/tools/preflight_5090_32g.sh \
  --task beat_block_hammer \
  --config demo_clean \
  --episodes 50
```

## 安装顺序

从干净机器开始：

```bash
git clone <your_robotwin_repo_url> robotwin
cd robotwin

python3 --version
nvidia-smi

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

先装/校验 RoboTwin 基础环境，再进 pi0.5：

```bash
# RoboTwin 仿真依赖，具体仍以 README 和官方文档为准
pip install -r script/requirements.txt

cd policy/pi05
uv python pin 3.11
uv sync
```

如果 `uv sync` 后的 smoke test 在 5090 上报 CUDA 架构不兼容，说明当前锁文件拉到了旧 CUDA wheel。优先处理环境，不要继续训练：

```bash
cd policy/pi05
uv run python - <<'PY'
import jax
print(jax.devices())
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
PY
```

期望看到 GPU 设备，Torch capability 应该是 `(12, 0)`。如果失败，换成平台提供的 CUDA 12.8+ 深度学习镜像，或在单独分支里升级 JAX/Torch CUDA 依赖后重新 lock。

## 数据准备流程

如果还没有 raw RoboTwin 数据，先采集：

```bash
bash collect_data.sh beat_block_hammer demo_clean 0
```

建议先审计数据：

```bash
python script/audit_robotwin_dataset.py audit \
  --task beat_block_hammer \
  --config demo_clean
```

转换为 pi0.5 预处理格式：

```bash
cd policy/pi05
bash process_data_pi05.sh beat_block_hammer demo_clean 50
```

这会从 `../../data/beat_block_hammer/demo_clean` 读取，写到：

```text
policy/pi05/processed_data/beat_block_hammer-demo_clean-50
```

再转成 LeRobot/ALOHA 风格数据：

```bash
cd policy/pi05
bash generate.sh \
  processed_data/beat_block_hammer-demo_clean-50 \
  beat_block_hammer-demo_clean-50
```

注意：训练配置里的 `repo_id` 默认还是 `"your_repo_id"`。如果转换脚本把数据落在本地 LeRobot cache，需要确认 `repo_id` 和实际数据目录一致；不一致时要改 `policy/pi05/src/openpi/training/config.py` 或新增自己的 config。

## 训练

第一版推荐命名：

```bash
cd policy/pi05
export WANDB_MODE=offline
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

bash finetune.sh \
  pi05_base_aloha_lora \
  beat_block_hammer_demo_clean_50_lora \
  0
```

checkpoint 默认写到：

```text
policy/pi05/checkpoints/pi05_base_aloha_lora/beat_block_hammer_demo_clean_50_lora/
```

32G 显存建议：

- 先保留 `batch_size=32`。
- 如果 OOM，把 `batch_size` 降到 16，并保持 `fsdp_devices=1`。
- 不建议单卡 32G 直接 full fine-tune；需要 full 时先短跑 100-500 steps 验证显存。

## 评测

示例：

```bash
cd policy/pi05
bash eval.sh \
  beat_block_hammer \
  demo_clean \
  pi05_base_aloha_lora \
  beat_block_hammer_demo_clean_50_lora \
  0 \
  0
```

默认 `deploy_policy.yml` 里：

- `checkpoint_id: 30000`
- `pi0_step: 50`
- `instruction_type: unseen`

如果训练步数不是 30000，评测前改 `checkpoint_id` 或通过 override 传入。

## 常见失败点

- `sm_120 is not compatible`: Torch/JAX wheel 太旧，换 CUDA 12.8+ wheel 或镜像。
- `no kernel image is available`: 也是 CUDA 架构不匹配，先别训练。
- `checkpoint/assets` 找不到: 评测要求 checkpoint 目录下有 norm stats/assets，确认训练成功保存了对应 step。
- `repo_id=your_repo_id`: 数据 config 没改到真实数据集 ID。
- GCS/S3 下载失败: 预训练权重下载源不可达，换有公网的机器或提前缓存 checkpoint。
- 显存 OOM: LoRA 降 batch；full 改 LoRA 或换更大显存。

## 建议保存的产物

每次训练至少保存：

- 本次数据目录名和 episode 数
- `policy/pi05/src/openpi/training/config.py` 中实际使用的 config
- `policy/pi05/checkpoints/<train_config>/<exp_name>`
- `wandb` 或本地训练日志
- `eval` 的 seed、task_config、checkpoint_id 和成功率

