# Resume and Portfolio Material

## Project Title

RoboTwin VLA Fine-Tuning and Policy Benchmarking on RTX 5090

## Short Portfolio Summary

Fine-tuned and evaluated a pi0.5 vision-language-action model on RoboTwin `beat_block_hammer` using 50 demonstrations, then benchmarked the resulting LoRA policy against DP and ACT baselines under fixed-seed rollout evaluation. The project covered data auditing, pi0.5/LeRobot conversion, CUDA 12.8 and SAPIEN setup on RTX 5090, LoRA training, checkpoint selection, robustness testing, and video-backed failure analysis.

## Resume Bullets

- Fine-tuned pi0.5 VLA with LoRA on RoboTwin manipulation data and improved unseen-instruction success from `34.0%` at checkpoint 5000 to `45.0%` at checkpoint 8000.
- Built an end-to-end robotics ML evaluation pipeline covering dataset audit, pi0.5/LeRobot data conversion, training, checkpoint management, rollout evaluation, and failure-video manifest generation.
- Evaluated robustness under lighting, background, clutter, and combined domain randomization; identified background texture shifts and full-randomized scene instability as key bottlenecks.
- Debugged RTX 5090 deployment issues across CUDA 12.8, PyTorch/JAX, SAPIEN Vulkan rendering, and cuRobo JIT compilation to run large-scale rollout evaluation on 2x 32GB GPUs.
- Ran a same-seed benchmark comparing pi0.5 LoRA, Diffusion Policy, and ACT on the same RoboTwin task; added watchdog logging for ACT rollout stalls to preserve resumable partial results.

## Interview Talking Points

- Why LoRA matters here: the work is not just "small fine-tuning"; it demonstrates adapting a generalist VLA model into a simulator-specific manipulation stack while preserving a reproducible evaluation protocol.
- Why the benchmark matters: comparing pi0.5 against DP and ACT under the same seeds makes the result more credible than isolated success rates.
- Why robustness matters: the clean success rate alone hides sensitivity to visual distribution shift; background and all-randomized settings revealed policy and environment-generation failure modes.
- Engineering depth: the project required making modern Blackwell GPUs work with robotics simulation, JAX/Torch, SAPIEN Vulkan, and policy-specific dependencies.

## Final Metrics To Fill

| metric | value |
|---|---|
| pi0.5 clean same-seed success | `45/100 = 45.0%` |
| pi0.5 light-only robustness | `25/50 = 50.0%` |
| pi0.5 background-only robustness | `19/50 = 38.0%` |
| pi0.5 clutter-only robustness | `26/50 = 52.0%` |
| pi0.5 all-random robustness | `15/50 = 30.0%` |
| DP same-seed success | `26/100 = 26.0%` |
| ACT same-seed success | partial: `13/46 = 28.3%`, including `13` timeout failures |
