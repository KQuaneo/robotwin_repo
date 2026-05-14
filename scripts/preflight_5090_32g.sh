#!/usr/bin/env bash
set -euo pipefail

TASK_NAME="beat_block_hammer"
TASK_CONFIG="demo_clean"
EPISODES="50"
TRAIN_CONFIG="pi05_base_aloha_lora"
GPU_ID="0"
RUN_SMOKE="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task)
      TASK_NAME="$2"
      shift 2
      ;;
    --config)
      TASK_CONFIG="$2"
      shift 2
      ;;
    --episodes)
      EPISODES="$2"
      shift 2
      ;;
    --train-config)
      TRAIN_CONFIG="$2"
      shift 2
      ;;
    --gpu)
      GPU_ID="$2"
      shift 2
      ;;
    --smoke)
      RUN_SMOKE="1"
      shift
      ;;
    -h|--help)
      cat <<'EOF'
Usage: bash policy/pi05/tools/preflight_5090_32g.sh [options]

Options:
  --task NAME          RoboTwin task name. Default: beat_block_hammer
  --config NAME        RoboTwin task config. Default: demo_clean
  --episodes N         Expected episode count. Default: 50
  --train-config NAME  openpi train config. Default: pi05_base_aloha_lora
  --gpu ID             CUDA_VISIBLE_DEVICES for smoke test. Default: 0
  --smoke              Run uv/JAX/Torch GPU smoke test. May trigger uv sync/downloads.
EOF
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PI05_DIR="${ROOT_DIR}/policy/pi05"
RAW_DATA_DIR="${ROOT_DIR}/data/${TASK_NAME}/${TASK_CONFIG}"
PROCESSED_DIR="${PI05_DIR}/processed_data/${TASK_NAME}-${TASK_CONFIG}-${EPISODES}"

ok() {
  printf '[OK] %s\n' "$1"
}

warn() {
  printf '[WARN] %s\n' "$1"
}

fail() {
  printf '[FAIL] %s\n' "$1"
}

section() {
  printf '\n== %s ==\n' "$1"
}

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

section "Repository"
if [[ -f "${ROOT_DIR}/README.md" && -d "${PI05_DIR}" ]]; then
  ok "RoboTwin root: ${ROOT_DIR}"
else
  fail "Cannot locate RoboTwin root from script path"
  exit 1
fi

if [[ -f "${PI05_DIR}/.python-version" ]]; then
  ok "pi0.5 Python pin: $(tr -d '\n' < "${PI05_DIR}/.python-version")"
else
  warn "Missing ${PI05_DIR}/.python-version"
fi

section "GPU"
if have_cmd nvidia-smi; then
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader || true
else
  warn "nvidia-smi not found"
fi

section "Host Tools"
if have_cmd python3; then
  ok "python3: $(python3 --version 2>&1)"
else
  fail "python3 not found"
fi

if have_cmd uv; then
  ok "uv: $(uv --version 2>&1)"
else
  warn "uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
fi

if have_cmd git; then
  ok "git: $(git --version 2>&1)"
else
  warn "git not found"
fi

section "Python Runtime Dependencies"
if have_cmd python3; then
  python3 - <<'PY'
missing = []
for name in ("h5py", "cv2", "yaml", "numpy"):
    try:
        __import__(name)
    except Exception:
        missing.append(name)

if missing:
    print("[WARN] Missing Python modules for RoboTwin data/audit scripts:", ", ".join(missing))
    print("[WARN] Install RoboTwin runtime deps inside the intended env, e.g. pip install -r script/requirements.txt")
else:
    print("[OK] RoboTwin data/audit Python modules are importable")
PY
fi

section "pi0.5 Config"
if grep -R "name=\"${TRAIN_CONFIG}\"" "${PI05_DIR}/src/openpi/training/config.py" >/dev/null 2>&1; then
  ok "train config exists: ${TRAIN_CONFIG}"
else
  fail "train config not found: ${TRAIN_CONFIG}"
fi

if grep -q 'jax.*==0.5.0' "${PI05_DIR}/pyproject.toml"; then
  warn "pyproject pins jax[cuda12]==0.5.0; verify this wheel on RTX 5090 before training"
fi

if grep -q 'name = "torch"' "${PI05_DIR}/uv.lock" && grep -A3 'name = "torch"' "${PI05_DIR}/uv.lock" | grep -q 'version = "2.6.0"'; then
  warn "uv.lock pins torch 2.6.0. RTX 5090 usually needs CUDA 12.8+ builds; run --smoke"
fi

if grep -q 'nvidia_cuda_runtime_cu12-12.4' "${PI05_DIR}/uv.lock"; then
  warn "uv.lock contains CUDA 12.4 runtime wheels. Blackwell/sm_120 should be verified with CUDA 12.8+"
fi

section "Data"
if [[ -d "${RAW_DATA_DIR}" ]]; then
  ok "raw data dir exists: ${RAW_DATA_DIR}"
  RAW_COUNT="$(find "${RAW_DATA_DIR}/data" -maxdepth 1 -name 'episode*.hdf5' 2>/dev/null | wc -l | tr -d ' ')"
  if [[ "${RAW_COUNT}" -ge "${EPISODES}" ]]; then
    ok "raw HDF5 episodes: ${RAW_COUNT}"
  else
    warn "raw HDF5 episodes: ${RAW_COUNT}; expected at least ${EPISODES}"
  fi
else
  warn "raw data dir not found: ${RAW_DATA_DIR}"
fi

if [[ -d "${PROCESSED_DIR}" ]]; then
  ok "processed pi0.5 dir exists: ${PROCESSED_DIR}"
  PROC_COUNT="$(find "${PROCESSED_DIR}" -maxdepth 1 -type d -name 'episode_*' 2>/dev/null | wc -l | tr -d ' ')"
  ok "processed episodes: ${PROC_COUNT}"
else
  warn "processed pi0.5 dir not found yet: ${PROCESSED_DIR}"
fi

section "Suggested Commands"
cat <<EOF
# Audit raw RoboTwin data
python script/audit_robotwin_dataset.py audit --task ${TASK_NAME} --config ${TASK_CONFIG}

# Convert raw RoboTwin data for pi0.5
cd policy/pi05
bash process_data_pi05.sh ${TASK_NAME} ${TASK_CONFIG} ${EPISODES}

# Convert processed data to LeRobot/ALOHA style
bash generate.sh processed_data/${TASK_NAME}-${TASK_CONFIG}-${EPISODES} ${TASK_NAME}-${TASK_CONFIG}-${EPISODES}

# First 5090 32G training attempt
export WANDB_MODE=offline
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
bash finetune.sh ${TRAIN_CONFIG} ${TASK_NAME}_${TASK_CONFIG}_${EPISODES}_lora ${GPU_ID}
EOF

if [[ "${RUN_SMOKE}" == "1" ]]; then
  section "GPU Smoke Test"
  if ! have_cmd uv; then
    fail "uv is required for --smoke"
    exit 1
  fi

  cd "${PI05_DIR}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run python - <<'PY'
import os

print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

try:
    import jax
    import jax.numpy as jnp

    print("jax:", jax.__version__)
    print("jax devices:", jax.devices())
    x = jnp.ones((128, 128), dtype=jnp.bfloat16)
    y = x @ x
    print("jax matmul:", y.shape, y.dtype)
except Exception as exc:
    print("JAX_SMOKE_FAILED:", repr(exc))
    raise

try:
    import torch

    print("torch:", torch.__version__, "cuda:", torch.version.cuda)
    print("torch cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("torch device:", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
        x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        y = x @ x
        torch.cuda.synchronize()
        print("torch matmul:", tuple(y.shape), y.dtype)
except Exception as exc:
    print("TORCH_SMOKE_FAILED:", repr(exc))
    raise
PY
fi

section "Done"
ok "Preflight complete"
