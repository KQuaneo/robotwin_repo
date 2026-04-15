# RoboTwin Training Archive

This repository is a structured record of my **RoboTwin training, evaluation, and comparison work**. It is not intended to present a single policy as the whole project. Instead, it is organized as a growing experiment archive where each run can be documented with its own setup, commands, artifacts, and observations.

## Repository Scope

This repo is for:
- training records across multiple RoboTwin policies
- evaluation outputs and result summaries
- reproducible command logs
- future cross-policy comparisons such as `DP`, `ACT`, and later baselines

This repo is not limited to Diffusion Policy. `DP` is simply the first experiment documented here.

## Structure

```text
experiments/
  dp/
results/
  dp/
```

## Experiment Index

| Policy | Task | Setting | Checkpoint | Result | Report |
| --- | --- | --- | --- | --- | --- |
| `DP` | `beat_block_hammer` | `demo_clean`, 50 demos | `600.ckpt` | `33.0%` | [Report](experiments/dp/beat_block_hammer_demo_clean.md) |

## Current Highlight

The current documented run is a RoboTwin **DP baseline** on `beat_block_hammer`, trained on `50` demonstrations under `demo_clean` and evaluated on `100` episodes with `unseen` instructions. The final observed success rate was **33.0%**.

## Planned Additions

- ACT training records
- side-by-side policy comparison tables
- training-loss figures
- rollout media for qualitative evaluation
- notes on machine constraints and reproducibility

## Notes

As more experiments are added, the top-level README will remain an index page, while detailed run descriptions will live under `experiments/` and raw outputs will be grouped under `results/`.
