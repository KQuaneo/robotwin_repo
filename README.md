# RoboTwin Training Archive

This repository is a structured record of my **RoboTwin training, evaluation, and comparison work**. It is not intended to present a single policy as the whole project. Instead, it is organized as a growing experiment archive where each run can be documented with its own setup, commands, artifacts, and observations.

## Repository Scope

This repo is for:
- training records across multiple RoboTwin policies
- evaluation outputs and result summaries
- reproducible command logs
- future cross-policy comparisons such as `DP`, `ACT`, and later baselines

## Structure

```text
experiments/
  dp/
  act/
results/
  dp/
  act/
```

## Experiment Index

| Policy | Task | Setting | Checkpoint | Result | Report |
| --- | --- | --- | --- | --- | --- |
| `DP` | `beat_block_hammer` | `demo_clean`, 50 demos | `600.ckpt` | `33.0%` | [Report](experiments/dp/beat_block_hammer_demo_clean.md) |
| `ACT` | `beat_block_hammer` | `demo_clean`, 50 demos | `policy_best.ckpt` | `32.0%` | [Report](experiments/act/beat_block_hammer_demo_clean_b1.md) |

## Current Comparison

| Policy | Success Rate | Notes |
| --- | --- | --- |
| `DP` | `33.0%` | Diffusion Policy baseline, batch size reduced to fit 8 GB GPU |
| `ACT` | `32.0%` | Action Chunking Transformer run, machine-safe batch size 1 |

At the moment, both baselines are close on this task and data scale. The purpose of the archive is to keep those runs reproducible and directly comparable as more policies and settings are added.

## Current Highlight

The repo now contains two completed baseline runs on `beat_block_hammer` with `50` `demo_clean` demonstrations:
- `DP`: `33/100 = 33.0%`
- `ACT`: `32/100 = 32.0%`

These are baseline reference points, not final performance claims.

## Planned Additions

- more ACT runs with different settings
- side-by-side policy comparison tables
- training-loss figures
- rollout media for qualitative evaluation
- notes on machine constraints and reproducibility

## Notes

As more experiments are added, the top-level README will remain an index page, while detailed run descriptions will live under `experiments/` and raw outputs will be grouped under `results/`.
