# Failure Analysis

This note summarizes the failure modes observed during the RoboTwin `beat_block_hammer` pi0.5 LoRA evaluation and the follow-up baseline benchmark.

## Scope

| item | value |
|---|---|
| Task | `beat_block_hammer` |
| Demonstrations | `demo_clean`, 50 episodes |
| Main pi0.5 checkpoint | `checkpoint-8000` |
| Main pi0.5 clean eval | `45/100 = 45.0%` |
| Baselines | Diffusion Policy, ACT |
| Hardware | 2x RTX 5090 32GB |

## Observed Failure Categories

| category | description | likely root cause | evidence / signal |
|---|---|---|---|
| Approach pose error | The end effector reaches the object area but does not arrive at a useful pre-contact pose. | Limited demonstrations and narrow task-specific adaptation; visual-language prior does not fully solve simulator contact geometry. | Clean pi0.5 eval succeeds on fewer than half of unseen trials. |
| Contact / hammering error | The policy reaches the hammer or block but fails to generate the correct contact direction or force sequence. | Tool-use tasks require precise contact timing; LoRA adaptation may improve task intent without fully solving low-level dynamics. | `beat_block_hammer` remains below 50% even after training to 8000 steps. |
| Visual distribution shift | Background/table texture changes reduce success relative to lighting or clutter changes. | The policy is more sensitive to scene appearance than to moderate lighting or object clutter in this setup. | Background-only robustness: `19/50 = 38.0%`; all-randomized: `15/50 = 30.0%`. |
| Scene generation instability | Randomized evaluation sometimes fails before policy quality is measured. | RoboTwin randomization can produce invalid or hard-to-initialize scenes. | Full randomization required `3275` setup retries to collect 50 valid rollouts. |
| ACT rollout stall | ACT evaluation can stall inside a single rollout and stop producing episode results. | Runtime/evaluator instability in this CUDA/SAPIEN/cuRobo stack, not just policy quality. | ACT partial benchmark includes `13` timeout failures among 46 completed or timed-out trials. |

## Benchmark Interpretation

The strongest clean result in this run is pi0.5 LoRA at `45/100 = 45.0%`, compared with DP at `26/100 = 26.0%`. ACT is not reported as a complete 100-seed result because rollout stability was insufficient; the saved partial result is `13/46 = 28.3%`.

The result should be described as a systems and evaluation project, not only as a model fine-tuning run. The main engineering contribution is the reproducible pipeline: data audit, conversion, RTX 5090 compatibility work, LoRA training, checkpoint comparison, robustness testing, same-seed benchmarking, and watchdog-based handling of evaluator stalls.

## Actionable Follow-Ups

1. Label 20-30 failed rollout videos manually into approach, grasp/contact, task sequencing, and simulator/runtime buckets.
2. Run a data scaling experiment with 100 or 200 demonstrations to test whether pi0.5 success improves with more task data.
3. Add one or two more RoboTwin tasks to test whether the pipeline generalizes beyond `beat_block_hammer`.
4. Resume ACT only if a full three-policy table is required; otherwise keep the partial ACT result as evidence of baseline runtime instability.
