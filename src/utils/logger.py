from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List

from src.types import JointCommand, PoseFrame


@dataclass
class CsvRunLogger:
    run_dir: Path
    _pose_writer: csv.DictWriter | None = field(default=None, init=False)
    _joint_writer: csv.DictWriter | None = field(default=None, init=False)
    _pose_fp: object | None = field(default=None, init=False)
    _joint_fp: object | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def _open_pose_writer(self, landmark_names: Iterable[str]) -> None:
        pose_path = self.run_dir / "pose_keypoints.csv"
        self._pose_fp = pose_path.open("w", newline="", encoding="utf-8")
        fieldnames: List[str] = ["frame_index", "timestamp_s"]
        for name in landmark_names:
            fieldnames.extend([f"{name}_x", f"{name}_y", f"{name}_z", f"{name}_visibility"])
        self._pose_writer = csv.DictWriter(self._pose_fp, fieldnames=fieldnames)
        self._pose_writer.writeheader()

    def _open_joint_writer(self, joint_names: Iterable[str]) -> None:
        joint_path = self.run_dir / "joint_targets.csv"
        self._joint_fp = joint_path.open("w", newline="", encoding="utf-8")
        fieldnames = ["frame_index", "timestamp_s", *joint_names]
        self._joint_writer = csv.DictWriter(self._joint_fp, fieldnames=fieldnames)
        self._joint_writer.writeheader()

    def log_pose(self, pose: PoseFrame) -> None:
        if self._pose_writer is None:
            self._open_pose_writer(sorted(pose.keypoints.keys()))
        row: Dict[str, float | int] = {
            "frame_index": pose.frame_index,
            "timestamp_s": pose.timestamp_s,
        }
        for name, keypoint in sorted(pose.keypoints.items()):
            row[f"{name}_x"] = keypoint.x
            row[f"{name}_y"] = keypoint.y
            row[f"{name}_z"] = keypoint.z
            row[f"{name}_visibility"] = keypoint.visibility
        self._pose_writer.writerow(row)

    def log_joint_command(self, command: JointCommand) -> None:
        if self._joint_writer is None:
            self._open_joint_writer(sorted(command.joint_angles_rad.keys()))
        row: Dict[str, float | int] = {
            "frame_index": command.frame_index,
            "timestamp_s": command.timestamp_s,
        }
        for name, angle in sorted(command.joint_angles_rad.items()):
            row[name] = angle
        self._joint_writer.writerow(row)

    def close(self) -> None:
        if self._pose_fp is not None:
            self._pose_fp.close()
        if self._joint_fp is not None:
            self._joint_fp.close()
