#!/usr/bin/env python3
"""Audit, clean-plan, and visualize RoboTwin demonstration datasets."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None
try:
    import h5py
except ModuleNotFoundError:
    h5py = None
try:
    import numpy as np
except ModuleNotFoundError:
    np = None


CAMERAS = ("head_camera", "left_camera", "right_camera")
RAW_IMAGE_KEY = "/observation/{camera}/rgb"
RAW_VECTOR_KEY = "/joint_action/vector"
RAW_ACTION_KEYS = (
    "/joint_action/left_arm",
    "/joint_action/left_gripper",
    "/joint_action/right_arm",
    "/joint_action/right_gripper",
)


def require_runtime_dependencies() -> None:
    missing = []
    if cv2 is None:
        missing.append("opencv-python")
    if h5py is None:
        missing.append("h5py")
    if np is None:
        missing.append("numpy")
    if missing:
        raise SystemExit(
            "Missing runtime dependencies for dataset auditing: "
            + ", ".join(missing)
            + ". Activate the RoboTwin conda environment or install script/requirements.txt."
        )


def episode_index(path: Path) -> int:
    match = re.search(r"episode_?(\d+)", path.stem)
    if match is None:
        raise ValueError(f"Cannot parse episode index from {path}")
    return int(match.group(1))


def load_seed_count(seed_path: Path) -> int | None:
    if not seed_path.exists():
        return None
    text = seed_path.read_text(encoding="utf-8").strip()
    if not text:
        return 0
    return len(text.split())


def load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def decode_image(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray) and value.ndim == 3:
        return value
    if isinstance(value, np.void):
        value = value.tobytes()
    if isinstance(value, np.ndarray):
        if value.dtype == np.uint8:
            buf = value
        else:
            buf = np.frombuffer(value.tobytes(), dtype=np.uint8)
    elif isinstance(value, (bytes, bytearray)):
        buf = np.frombuffer(value, dtype=np.uint8)
    else:
        raise TypeError(f"Unsupported image value type: {type(value)}")
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cv2.imdecode returned None")
    return img


def sample_indices(length: int) -> list[int]:
    if length <= 0:
        return []
    indices = {0, length // 2, length - 1}
    return sorted(indices)


def finite_stats(array: np.ndarray) -> dict[str, float | bool]:
    finite = np.isfinite(array)
    result: dict[str, float | bool] = {
        "all_finite": bool(finite.all()),
        "nan_count": int(np.isnan(array).sum()),
        "inf_count": int(np.isinf(array).sum()),
    }
    if finite.any():
        values = array[finite]
        result.update({
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "std": float(values.std()),
        })
    return result


def inspect_episode(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    issues: list[str] = []
    info: dict[str, Any] = {
        "episode": episode_index(path),
        "path": str(path),
        "issues": issues,
    }

    try:
        with h5py.File(path, "r") as f:
            missing = [key for key in (RAW_VECTOR_KEY, *RAW_ACTION_KEYS) if key not in f]
            for camera in CAMERAS:
                key = RAW_IMAGE_KEY.format(camera=camera)
                if key not in f:
                    missing.append(key)
            if missing:
                issues.extend(f"missing_key:{key}" for key in missing)
                info["ok"] = False
                return info

            vector = f[RAW_VECTOR_KEY][()]
            info["vector_shape"] = list(vector.shape)
            info["vector_stats"] = finite_stats(vector)
            if vector.ndim != 2:
                issues.append(f"bad_vector_ndim:{vector.ndim}")
            if vector.ndim == 2 and vector.shape[1] != args.expected_action_dim:
                issues.append(f"bad_action_dim:{vector.shape[1]}")
            if not np.isfinite(vector).all():
                issues.append("non_finite_vector")

            action_delta = np.diff(vector, axis=0) if vector.ndim == 2 and len(vector) > 1 else np.empty((0,))
            if action_delta.size:
                per_step_l2 = np.linalg.norm(action_delta, axis=1)
                max_l2 = float(per_step_l2.max())
                max_abs = float(np.abs(action_delta).max())
                info["action_delta"] = {
                    "max_l2": max_l2,
                    "mean_l2": float(per_step_l2.mean()),
                    "max_abs": max_abs,
                }
                if max_l2 > args.max_l2_jump:
                    issues.append(f"large_l2_jump:{max_l2:.6f}")
                if max_abs > args.max_abs_jump:
                    issues.append(f"large_abs_jump:{max_abs:.6f}")

            lengths: dict[str, int] = {"joint_action/vector": int(vector.shape[0])}
            camera_shapes: dict[str, list[int]] = {}
            for camera in CAMERAS:
                ds = f[RAW_IMAGE_KEY.format(camera=camera)]
                lengths[f"observation/{camera}/rgb"] = int(ds.shape[0])
                for idx in sample_indices(int(ds.shape[0])):
                    try:
                        img = decode_image(ds[idx])
                    except Exception as exc:  # noqa: BLE001 - keep audit running
                        issues.append(f"decode_failed:{camera}:{idx}:{exc}")
                        continue
                    camera_shapes[camera] = list(img.shape)
                    if img.ndim != 3 or img.shape[2] != 3:
                        issues.append(f"bad_image_shape:{camera}:{img.shape}")

            for key in RAW_ACTION_KEYS:
                lengths[key.lstrip("/")] = int(f[key].shape[0])

            info["lengths"] = lengths
            info["camera_shapes"] = camera_shapes
            unique_lengths = set(lengths.values())
            if len(unique_lengths) != 1:
                issues.append(f"length_mismatch:{sorted(unique_lengths)}")
            length = int(vector.shape[0])
            if length < args.min_length:
                issues.append(f"too_short:{length}")
            if args.max_length is not None and length > args.max_length:
                issues.append(f"too_long:{length}")
    except Exception as exc:  # noqa: BLE001 - report corrupt files
        issues.append(f"open_failed:{exc}")

    info["ok"] = len(issues) == 0
    return info


def collect_dataset_report(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root)
    dataset_dir = root / "data" / args.task / args.config
    data_dir = dataset_dir / "data"
    instruction_dir = dataset_dir / "instructions"
    video_dir = dataset_dir / "video"
    seed_path = dataset_dir / "seed.txt"
    scene_info_path = dataset_dir / "scene_info.json"

    files = sorted(data_dir.glob("episode*.hdf5"), key=episode_index)
    if args.episodes is not None:
        files = files[:args.episodes]

    report: dict[str, Any] = {
        "task": args.task,
        "config": args.config,
        "dataset_dir": str(dataset_dir),
        "episode_count": len(files),
        "seed_count": load_seed_count(seed_path),
        "scene_info_count": None,
        "instruction_count": len(list(instruction_dir.glob("episode*.json"))) if instruction_dir.exists() else 0,
        "video_count": len(list(video_dir.glob("episode*.mp4"))) if video_dir.exists() else 0,
        "episodes": [],
        "bad_episodes": [],
        "missing_episode_indices": [],
        "dataset_issues": [],
    }

    scene_info = load_json(scene_info_path)
    if isinstance(scene_info, dict):
        report["scene_info_count"] = len(scene_info)

    indices = [episode_index(path) for path in files]
    if indices:
        expected = set(range(min(indices), max(indices) + 1))
        missing = sorted(expected - set(indices))
        report["missing_episode_indices"] = missing
        if missing:
            report["dataset_issues"].append(f"missing_hdf5_indices:{missing}")

    for label, count in (
        ("seed_count", report["seed_count"]),
        ("instruction_count", report["instruction_count"]),
        ("video_count", report["video_count"]),
        ("scene_info_count", report["scene_info_count"]),
    ):
        if count is not None and count < len(files):
            report["dataset_issues"].append(f"{label}_less_than_hdf5:{count}<{len(files)}")

    bad = []
    for path in files:
        episode_report = inspect_episode(path, args)
        report["episodes"].append(episode_report)
        if not episode_report["ok"]:
            bad.append({
                "episode": episode_report["episode"],
                "path": episode_report["path"],
                "issues": episode_report["issues"],
            })
    report["bad_episodes"] = bad
    report["ok"] = not report["dataset_issues"] and not bad
    return report


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def print_audit_summary(report: dict[str, Any]) -> None:
    print(f"Dataset: {report['task']}/{report['config']}")
    print(f"Episodes: {report['episode_count']}")
    print(
        "Side files: "
        f"seeds={report['seed_count']}, "
        f"instructions={report['instruction_count']}, "
        f"videos={report['video_count']}, "
        f"scene_info={report['scene_info_count']}"
    )
    print(f"Dataset issues: {len(report['dataset_issues'])}")
    print(f"Bad episodes: {len(report['bad_episodes'])}")
    if report["dataset_issues"]:
        print("Dataset issue list:")
        for issue in report["dataset_issues"]:
            print(f"  - {issue}")
    if report["bad_episodes"]:
        print("Bad episode list:")
        for item in report["bad_episodes"][:20]:
            print(f"  - episode {item['episode']}: {', '.join(item['issues'])}")
        if len(report["bad_episodes"]) > 20:
            print(f"  ... {len(report['bad_episodes']) - 20} more")


def load_episode_for_visualization(path: Path) -> tuple[np.ndarray, dict[str, list[np.ndarray]]]:
    with h5py.File(path, "r") as f:
        vector = f[RAW_VECTOR_KEY][()]
        images: dict[str, list[np.ndarray]] = {}
        for camera in CAMERAS:
            ds = f[RAW_IMAGE_KEY.format(camera=camera)]
            images[camera] = [decode_image(ds[idx]) for idx in range(ds.shape[0])]
    return vector, images


def save_contact_sheet(images: dict[str, list[np.ndarray]], out_path: Path) -> None:
    length = min(len(frames) for frames in images.values())
    indices = sample_indices(length)
    rows = []
    for idx in indices:
        row = []
        for camera in CAMERAS:
            img = images[camera][idx]
            img = cv2.resize(img, (320, 240), interpolation=cv2.INTER_AREA)
            label = f"{camera} frame {idx}"
            cv2.putText(img, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
            row.append(img)
        rows.append(np.concatenate(row, axis=1))
    sheet = np.concatenate(rows, axis=0)
    cv2.imwrite(str(out_path), sheet)


def save_debug_video(images: dict[str, list[np.ndarray]], out_path: Path, max_frames: int, fps: int) -> None:
    length = min(len(frames) for frames in images.values())
    if length == 0:
        return
    stride = max(1, math.ceil(length / max_frames))
    first = cv2.resize(images[CAMERAS[0]][0], (320, 240), interpolation=cv2.INTER_AREA)
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * len(CAMERAS), height))
    for idx in range(0, length, stride):
        row = []
        for camera in CAMERAS:
            img = cv2.resize(images[camera][idx], (320, 240), interpolation=cv2.INTER_AREA)
            cv2.putText(img, f"{camera} {idx}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
            row.append(img)
        writer.write(np.concatenate(row, axis=1))
    writer.release()


def save_action_plots(vector: np.ndarray, out_dir: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    timesteps = np.arange(vector.shape[0])
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(timesteps, vector)
    ax.set_title("Joint action vector")
    ax.set_xlabel("frame")
    ax.set_ylabel("qpos/action value")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "joint_action_vector.png", dpi=160)
    plt.close(fig)

    if vector.shape[0] > 1:
        delta = np.diff(vector, axis=0)
        l2 = np.linalg.norm(delta, axis=1)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(np.arange(len(l2)), l2)
        ax.set_title("Per-step action L2 delta")
        ax.set_xlabel("frame")
        ax.set_ylabel("L2 delta")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "action_delta_l2.png", dpi=160)
        plt.close(fig)


def command_audit(args: argparse.Namespace) -> int:
    require_runtime_dependencies()
    report = collect_dataset_report(args)
    out_path = Path(args.out) if args.out else Path(args.root) / "debug" / f"{args.task}_{args.config}_audit.json"
    write_json(out_path, report)
    print_audit_summary(report)
    print(f"Report: {out_path}")
    return 0 if report["ok"] else 1


def command_clean_plan(args: argparse.Namespace) -> int:
    require_runtime_dependencies()
    report = collect_dataset_report(args)
    plan = {
        "task": report["task"],
        "config": report["config"],
        "dataset_issues": report["dataset_issues"],
        "bad_episodes": report["bad_episodes"],
        "note": "Non-mutating clean plan only. Review before deleting or replacing any files.",
    }
    out_path = Path(args.out) if args.out else Path(args.root) / "debug" / f"{args.task}_{args.config}_bad_episodes.json"
    write_json(out_path, plan)
    print(f"Bad episodes: {len(plan['bad_episodes'])}")
    print(f"Clean plan: {out_path}")
    return 0


def command_visualize(args: argparse.Namespace) -> int:
    require_runtime_dependencies()
    root = Path(args.root)
    episode_path = root / "data" / args.task / args.config / "data" / f"episode{args.episode}.hdf5"
    if not episode_path.exists():
        raise FileNotFoundError(f"Episode file not found: {episode_path}")

    out_dir = Path(args.out) if args.out else root / "debug" / f"{args.task}_{args.config}_episode{args.episode}"
    out_dir.mkdir(parents=True, exist_ok=True)

    vector, images = load_episode_for_visualization(episode_path)
    save_contact_sheet(images, out_dir / "contact_sheet.jpg")
    save_debug_video(images, out_dir / "rollout_preview.mp4", args.max_video_frames, args.fps)
    save_action_plots(vector, out_dir)

    summary = {
        "episode": args.episode,
        "episode_path": str(episode_path),
        "frames": int(vector.shape[0]),
        "action_dim": int(vector.shape[1]) if vector.ndim == 2 else None,
        "vector_stats": finite_stats(vector),
        "outputs": {
            "contact_sheet": str(out_dir / "contact_sheet.jpg"),
            "rollout_preview": str(out_dir / "rollout_preview.mp4"),
            "joint_action_vector": str(out_dir / "joint_action_vector.png"),
            "action_delta_l2": str(out_dir / "action_delta_l2.png"),
        },
    }
    write_json(out_dir / "summary.json", summary)
    print(f"Visualization written to: {out_dir}")
    return 0


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", default=".", help="Repository root. Default: current directory.")
    parser.add_argument("--task", required=True, help="Task name, e.g. beat_block_hammer.")
    parser.add_argument("--config", required=True, help="Task config, e.g. demo_clean.")
    parser.add_argument("--episodes", type=int, default=None, help="Limit to first N episodes.")
    parser.add_argument("--expected-action-dim", type=int, default=14)
    parser.add_argument("--min-length", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--max-l2-jump", type=float, default=10.0)
    parser.add_argument("--max-abs-jump", type=float, default=3.0)
    parser.add_argument("--out", default=None, help="Output JSON path or visualization directory.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="Audit raw RoboTwin dataset consistency.")
    add_common_args(audit)
    audit.set_defaults(func=command_audit)

    clean = subparsers.add_parser("clean-plan", help="Write a non-mutating bad episode manifest.")
    add_common_args(clean)
    clean.set_defaults(func=command_clean_plan)

    visualize = subparsers.add_parser("visualize", help="Visualize one raw RoboTwin episode.")
    add_common_args(visualize)
    visualize.add_argument("--episode", type=int, required=True)
    visualize.add_argument("--max-video-frames", type=int, default=200)
    visualize.add_argument("--fps", type=int, default=10)
    visualize.set_defaults(func=command_visualize)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
