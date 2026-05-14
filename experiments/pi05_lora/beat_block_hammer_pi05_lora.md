# pi0.5 LoRA Experiment Report

## Objective

Evaluate whether pi0.5 can be adapted to RoboTwin's `beat_block_hammer` task using LoRA fine-tuning on 50 demonstrations, and identify whether more training steps or language-instruction generalization dominate performance.

## Setup

- Task: `beat_block_hammer`
- Task config: `demo_clean`
- Dataset: 50 demonstrations
- Model config: `pi05_base_aloha_lora`
- Training hardware: 2x RTX 5090 32GB
- Fine-tuning strategy: LoRA, not full fine-tuning
- Training target: 8000 steps
- Evaluation policy step horizon: `pi0_step=50`
- Evaluation videos: enabled through RoboTwin `eval_video_log`

## Evaluation Design

The experiment compares two checkpoints:

- `checkpoint-5000`
- `checkpoint-8000`

Two instruction regimes were tested:

- `seen`: language templates seen during training/data generation.
- `unseen`: paraphrased instruction templates.

The unseen split used 100 rollouts per checkpoint. The seen split used 50 rollouts per checkpoint as a quick language-gap probe.

## Results

| checkpoint | instruction split | success | trials | success rate |
|---|---:|---:|---:|---:|
| 5000 | unseen | 34 | 100 | 34.0% |
| 8000 | unseen | 45 | 100 | 45.0% |
| 5000 | seen | 15 | 50 | 30.0% |
| 8000 | seen | 27 | 50 | 54.0% |

## Interpretation

The 8000-step checkpoint is meaningfully better than the 5000-step checkpoint:

- Unseen instructions: `34% -> 45%`
- Seen instructions: `30% -> 54%`

This suggests that the model was still undertrained around 5000 steps and benefited from additional LoRA updates. However, the final 8000-step unseen success rate is still only 45%, so the model is far from saturated on this task.

The seen/unseen gap at 8000 steps suggests language phrasing still matters. Since the task dynamics are unchanged, this likely reflects instruction grounding or language-conditioned action selection instability rather than pure low-level control failure alone.

## Important Caveat

The `ckpt5000 unseen` first-half run encountered a stuck evaluation process after 17 episodes. It was stopped and resumed from the next raw seed for the remaining 33 episodes. The final 100-trial result is therefore composed of:

- First segment: 8/17
- Resume segment: 10/33
- Second GPU segment: 16/50
- Total: 34/100

No finished episode was discarded.

## Failure Analysis Plan

Rollout videos were saved for every episode. The next useful step is to classify failed episodes into categories such as:

- Failed to approach hammer.
- Grasped wrong part of hammer.
- Grasp was unstable or dropped hammer.
- Correct grasp but wrong hitting trajectory.
- Reached block but insufficient impact.
- Language-conditioned arm/handle confusion.
- Simulator/control instability.

This classification is more informative than only reporting aggregate success rate, because it shows where pi0.5 adaptation fails: visual grounding, language grounding, grasping, or long-horizon execution.

## Summary

This experiment establishes a reproducible pi0.5 LoRA baseline on RoboTwin with RTX 5090 hardware. The result is not merely a training run: it is a controlled checkpoint and instruction-generalization study with saved videos for failure diagnosis.
