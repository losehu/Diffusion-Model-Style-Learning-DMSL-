"""Convert APEX Go2 CSV retargeted motions to MimicKit motion pickles."""

import argparse
import csv
import math
import os
import pickle

import numpy as np


CSV_TO_MIMICKIT_JOINT_ORDER = [
    "base2", "shoulder2", "elbow2",  # FL
    "base1", "shoulder1", "elbow1",  # FR
    "base4", "shoulder4", "elbow4",  # RL
    "base3", "shoulder3", "elbow3",  # RR
]


def _quat_xyzw_to_exp_map(quat):
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    quat = quat / np.clip(norm, 1e-8, None)

    neg_w = quat[..., 3:4] < 0.0
    quat = np.where(neg_w, -quat, quat)

    xyz = quat[..., :3]
    w = np.clip(quat[..., 3], -1.0, 1.0)
    xyz_norm = np.linalg.norm(xyz, axis=-1)
    angle = 2.0 * np.arctan2(xyz_norm, w)

    exp_map = np.zeros_like(xyz)
    small = xyz_norm < 1e-8
    exp_map[small] = 2.0 * xyz[small]
    exp_map[~small] = xyz[~small] / xyz_norm[~small, None] * angle[~small, None]
    return exp_map.astype(np.float32)


def _read_csv(input_file):
    with open(input_file, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Empty CSV: {input_file}")

    data = {
        key: np.array([float(row[key]) for row in rows], dtype=np.float32)
        for key in rows[0].keys()
    }
    return data


def _drop_final_wrap_frame(frames, threshold):
    final_step = np.linalg.norm(frames[-1, 0:2] - frames[-2, 0:2])
    if final_step > threshold:
        frames = frames[:-1]
        print(f"Dropped final wrap frame: xy_step={final_step:.3f}m > {threshold:.3f}m")
    return frames


def convert_file(input_file, output_file, fps, loop_mode, rebase_xy, drop_final_wrap_threshold):
    data = _read_csv(input_file)

    root_pos = np.stack([data["com_x"], data["com_y"], data["height"]], axis=-1)
    if rebase_xy:
        root_pos[:, 0:2] -= root_pos[0:1, 0:2]

    quat = np.stack(
        [data["quat_x"], data["quat_y"], data["quat_z"], data["quat_w"]],
        axis=-1,
    )
    root_rot = _quat_xyzw_to_exp_map(quat)

    joint_dof = np.stack([data[name] for name in CSV_TO_MIMICKIT_JOINT_ORDER], axis=-1)
    frames = np.concatenate([root_pos, root_rot, joint_dof], axis=-1).astype(np.float32)
    if drop_final_wrap_threshold is not None:
        frames = _drop_final_wrap_frame(frames, drop_final_wrap_threshold)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "wb") as f:
        pickle.dump(
            {
                "loop_mode": int(loop_mode),
                "fps": int(fps),
                "frames": frames.tolist(),
            },
            f,
        )

    duration = (frames.shape[0] - 1) / float(fps)
    print(f"Wrote {output_file}: frames={frames.shape[0]}, fps={fps}, duration={duration:.2f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--loop_mode", type=int, default=1, choices=[0, 1])
    parser.add_argument("--no_rebase_xy", action="store_true")
    parser.add_argument(
        "--drop_final_wrap_threshold",
        type=float,
        default=None,
        help="Drop the final frame if its XY step from the previous frame exceeds this many meters.",
    )
    args = parser.parse_args()

    convert_file(
        input_file=args.input_file,
        output_file=args.output_file,
        fps=args.fps,
        loop_mode=args.loop_mode,
        rebase_xy=not args.no_rebase_xy,
        drop_final_wrap_threshold=args.drop_final_wrap_threshold,
    )


if __name__ == "__main__":
    main()
