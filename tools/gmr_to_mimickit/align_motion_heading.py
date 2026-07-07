#!/usr/bin/env python3
"""Rotate a MimicKit motion so its net XY displacement points to +X."""

from __future__ import annotations

import argparse
import math
import os
import pickle
import sys

import torch

sys.path.append("mimickit")
import util.torch_util as torch_util  # noqa: E402


def _yaw_quat(yaw: float, device: torch.device) -> torch.Tensor:
    half = 0.5 * yaw
    return torch.tensor(
        [0.0, 0.0, math.sin(half), math.cos(half)],
        dtype=torch.float32,
        device=device,
    )


def align_motion(input_file: str, output_file: str, target_heading: float) -> None:
    with open(input_file, "rb") as f:
        motion = pickle.load(f)

    frames = torch.tensor(motion["frames"], dtype=torch.float32)
    if frames.ndim != 2 or frames.shape[-1] < 6:
        raise ValueError(f"Invalid motion frames shape: {tuple(frames.shape)}")

    disp = frames[-1, 0:2] - frames[0, 0:2]
    source_heading = math.atan2(float(disp[1]), float(disp[0]))
    yaw_delta = target_heading - source_heading

    cos_yaw = math.cos(yaw_delta)
    sin_yaw = math.sin(yaw_delta)
    xy = frames[:, 0:2].clone()
    frames[:, 0] = cos_yaw * xy[:, 0] - sin_yaw * xy[:, 1]
    frames[:, 1] = sin_yaw * xy[:, 0] + cos_yaw * xy[:, 1]

    yaw_q = _yaw_quat(yaw_delta, frames.device).expand(frames.shape[0], -1)
    root_q = torch_util.exp_map_to_quat(frames[:, 3:6])
    aligned_root_q = torch_util.quat_mul(yaw_q, root_q)
    frames[:, 3:6] = torch_util.quat_to_exp_map(aligned_root_q)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    aligned_motion = dict(motion)
    aligned_motion["frames"] = frames.tolist()
    with open(output_file, "wb") as f:
        pickle.dump(aligned_motion, f)

    new_disp = frames[-1, 0:2] - frames[0, 0:2]
    new_heading = math.atan2(float(new_disp[1]), float(new_disp[0]))
    print(
        f"Wrote {output_file}: "
        f"source_heading={math.degrees(source_heading):.2f}deg, "
        f"target_heading={math.degrees(target_heading):.2f}deg, "
        f"new_heading={math.degrees(new_heading):.2f}deg"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--target_heading_deg", type=float, default=0.0)
    args = parser.parse_args()

    align_motion(
        input_file=args.input_file,
        output_file=args.output_file,
        target_heading=math.radians(args.target_heading_deg),
    )


if __name__ == "__main__":
    main()
