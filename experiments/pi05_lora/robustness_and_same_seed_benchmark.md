# Robustness and Same-Seed Benchmark

This report extends the pi0.5 LoRA experiment with two follow-up evaluations on RoboTwin `beat_block_hammer`:

- domain-randomization robustness tests for the pi0.5 LoRA checkpoint
- a same-task, same-seed comparison against DP and ACT baselines

## Setup

| item | value |
|---|---|
| Task | `beat_block_hammer` |
| Data | `demo_clean`, 50 demonstrations |
| pi0.5 policy | `pi05_base_aloha_lora` |
| pi0.5 checkpoint | `checkpoint-8000` |
| Hardware | 2x RTX 5090 32GB |
| Evaluation seed start | `100000` |
| Primary evaluation split | unseen instructions |

## pi0.5 Robustness Results

The robustness tests used the same pi0.5 LoRA checkpoint and changed only the RoboTwin task configuration.

| evaluation config | change from clean eval | success | trials | success rate |
|---|---|---:|---:|---:|
| `demo_clean` | no added randomization | 45 | 100 | 45.0% |
| `demo_light_only` | randomized lighting | 25 | 50 | 50.0% |
| `demo_bg_only` | randomized background/table textures | 19 | 50 | 38.0% |
| `demo_clutter_only` | cluttered table objects | 26 | 50 | 52.0% |
| `demo_all_randomized` | background + lighting + clutter | 15 | 50 | 30.0% |

## Robustness Interpretation

The model was comparatively tolerant to lighting and clutter in this run, but background/table texture changes reduced success noticeably. The all-randomized setting was the hardest case, ending at `30.0%`.

A separate practical finding is that full randomization was difficult to evaluate reliably: the run needed `3275` setup retries to collect 50 valid rollouts. This indicates that the randomized scene generator itself can become a bottleneck when measuring robustness, not just the policy.

## Same-Seed Benchmark

The benchmark compares pi0.5 LoRA, Diffusion Policy, and ACT on the same RoboTwin task and seed sequence. The pi0.5 and DP results are complete. ACT was paused after 46 completed or timed-out trials and can be resumed from the saved supervisor state.

| policy | checkpoint / model | seed start | completed trials | success | success rate | status |
|---|---|---:|---:|---:|---:|---|
| pi0.5 LoRA | `checkpoint-8000` | `100000` | 100 | 45 | 45.0% | complete |
| DP | `600.ckpt` | `100000` | 100 | 26 | 26.0% | complete |
| ACT | `policy_last.ckpt` | `100000` / `100069` split | 46 | 13 | 28.3% | paused, partial |

The ACT partial count includes `13` successes, `20` normal rollout failures, and `13` timeout failures. Timeout failures are reported separately because they indicate evaluator/runtime instability rather than a clean policy failure. The saved remote artifacts are:

- `/root/autodl-tmp/robotwin_logs/ACT_same_seed100_saved_partial_summary.json`
- `/root/autodl-tmp/robotwin_logs/ACT_same_seed100_saved_partial_results.tsv`

## Engineering Notes

- The RTX 5090 environment needed CUDA/Vulkan-aware setup for SAPIEN rendering.
- The baseline runner initially failed with `Render Error` until it inherited the same Vulkan/SAPIEN environment variables used by the pi0.5 runner, especially `VK_ICD_FILENAMES` and the pi0.5 virtualenv path.
- ACT rollout was unstable in this environment, with repeated single-episode stalls around fixed simulation steps. A watchdog supervisor was added so stalled seeds are recorded as timeout failures and the benchmark can be resumed instead of losing the whole run.
- All large artifacts, model weights, raw rollouts, and SSH credentials are intentionally excluded from this public experiment repository.
