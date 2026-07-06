#!/usr/bin/env python3
"""Summarize Go2 SMP training logs and flag common reward failures."""

from __future__ import annotations

import argparse
from pathlib import Path


KEYS = [
    "Iteration",
    "Test_Return",
    "Test_Episode_Length",
    "Task_Reward_Mean",
    "Smp_Reward_Mean",
    "Combined_Reward_Mean",
    "Sds_Loss_Mean",
    "Critic_Loss",
    "Actor_Loss",
    "Clip_Frac",
]


def _load_last_row(log_file: Path) -> dict[str, float | str]:
    text = log_file.read_text().replace("\r", "\n").strip().splitlines()
    rows = [line.split() for line in text if line.split()]
    if len(rows) < 2:
        raise ValueError(f"No data rows found in {log_file}")

    header = rows[0]
    row = rows[-1]
    data: dict[str, float | str] = dict(zip(header, row))
    for key, value in list(data.items()):
        try:
            data[key] = float(value)
        except ValueError:
            pass
    return data


def _print_summary(data: dict[str, float | str]) -> None:
    for key in KEYS:
        if key in data:
            print(f"{key}: {data[key]}")

    task_reward = float(data.get("Task_Reward_Mean", -1.0))
    smp_reward = float(data.get("Smp_Reward_Mean", -1.0))
    sds_loss = float(data.get("Sds_Loss_Mean", -1.0))
    ep_len = float(data.get("Test_Episode_Length", -1.0))

    print("\nDiagnosis:")
    if ep_len > 550 and task_reward > 0.9 and smp_reward < 0.05:
        print("- 速度任务已经学会了，但 SMP 风格基本塌了；视觉上通常会很丑或钻空子。")
        print("- 优先收窄速度范围，或提高 SMP_REWARD_WEIGHT / 降低 TASK_REWARD_WEIGHT 后重训 policy。")
    elif ep_len < 300:
        print("- episode 很短，说明稳定性还没学会或提前终止太多。")
    elif smp_reward > 0.3 and task_reward > 0.8:
        print("- 速度和风格都在工作，这个 run 大概率可以看可视化。")
    else:
        print("- 指标没有明显单点结论，建议结合 TensorBoard 曲线和可视化看。")

    if sds_loss > 2.0 and smp_reward < 0.05:
        print("- SDS loss 很高，policy 已经远离 prior 分布。")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="训练输出目录或 log.txt 路径")
    args = parser.parse_args()

    path = Path(args.path)
    log_file = path / "log.txt" if path.is_dir() else path
    data = _load_last_row(log_file)
    print(f"Log: {log_file}")
    _print_summary(data)


if __name__ == "__main__":
    main()
