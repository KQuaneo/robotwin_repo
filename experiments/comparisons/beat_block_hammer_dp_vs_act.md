# DP vs ACT on `beat_block_hammer`

## Goal

This note compares two RoboTwin baselines trained and evaluated on the same task and data scale:

- `DP` on `beat_block_hammer`
- `ACT` on `beat_block_hammer`

Both runs used:

- task setting: `demo_clean`
- expert demonstrations: `50`
- evaluation episodes: `100`
- instruction type: `unseen`
- local hardware: `RTX 4060 Ti 8GB`

## Result Table

| Policy | Checkpoint Used | Success Rate | Notes |
| --- | --- | --- | --- |
| `DP` | `600.ckpt` | `33.0%` | batch size reduced to `8` |
| `ACT` | `policy_best.ckpt` | `32.0%` | batch size reduced to `1` |

## Practical Interpretation

The main conclusion from these two runs is not that one method clearly wins. The observed gap is very small, and both results should be treated as **baseline reference points** on a modest-data, consumer-GPU setup.

The stronger value of the comparison is:

- both policies were actually trained and evaluated end-to-end
- both were adapted to the same hardware limit
- both are documented with reproducible commands and result artifacts

## Engineering Differences That Mattered

### DP

- preprocessing output: `.zarr`
- training outputs: JSON logs and run folders
- evaluation path was relatively straightforward once checkpoints were located correctly
- memory pressure was high enough that the stock configuration had to be reduced

### ACT

- preprocessing output: processed episode `.hdf5`
- training outputs: checkpoints plus loss-curve PNGs
- evaluation required a direct invocation because the stock wrapper did not match the custom checkpoint folder name
- machine-safe training used `batch_size=1`

## Resource-Constrained Reproduction Notes

These experiments were intentionally scoped to methods that could be run reliably on an `RTX 4060 Ti 8GB`.

That constraint shaped several decisions:

- limiting the policy set to `DP` and `ACT`
- using smaller batch sizes than stock defaults
- focusing on one task and one demonstration scale before expanding outward

This is a feature of the project rather than a defect: the repository is designed to show realistic reproduction and comparison under limited hardware, not just idealized large-GPU experimentation.

## What This Comparison Supports

This comparison supports the claim that the repository contains:

- real baseline reproduction work
- comparative experiment organization
- reproducible embodied-policy engineering

It does **not** yet support stronger claims such as:

- robust cross-task generalization
- broad policy benchmarking
- algorithmic novelty

## Next Steps

- add rollout media showing one success and one failure from each policy
- add training-loss figures directly to the experiment pages
- repeat the comparison on another task or another demo count
- record failure-mode observations once media review is complete
