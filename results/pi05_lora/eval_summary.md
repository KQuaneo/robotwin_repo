# pi0.5 ckpt5000/8000 Eval Summary

| split | success | total | rate |
|---|---:|---:|---:|
| ckpt8000_unseen_100 | 45 | 100 | 45.0% |
| ckpt5000_unseen_100 | 34 | 100 | 34.0% |
| ckpt8000_seen_50 | 27 | 50 | 54.0% |
| ckpt5000_seen_50 | 15 | 50 | 30.0% |

## Notes

- `ckpt8000_unseen_100` was run as two 50-episode chunks on two GPUs.
- `ckpt5000_unseen_100` includes one resumed chunk because the first GPU process stopped advancing after 17 finished episodes.
- All finished episodes were counted; no success or failure was discarded.
- Rollout videos were saved by RoboTwin under `eval_result/.../episode*.mp4`.

## Result Files on the Remote Machine

```text
/root/autodl-tmp/robotwin_logs/pi05_eval_summary_ckpt5000_8000.md
/root/autodl-tmp/robotwin_logs/pi05_eval_failure_manifest_ckpt5000_8000.json
```
