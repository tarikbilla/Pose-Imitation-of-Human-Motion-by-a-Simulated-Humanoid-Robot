# Product Requirements Document (PRD)

## Project: Pose Imitation of Human Motion by a Simulated Humanoid Robot

| Field | Value |
|---|---|
| Course | CPSM 2026S |
| Supervisor | M.Sc. Severin Stahl (THM, Campus Friedberg, IEM) |
| Document Version | 1.0 |
| Date | 2026-05-05 |
| Status | Draft |
| Primary Tech Stack | Python, Webots, OpenCV, MediaPipe / a learned pose estimator |
| Capture Hardware | Sony A7 III (tripod-mounted), 1920×1080 @ 25–100 FPS (adaptive) |
| Webots Project Root | `main/` (contains `worlds/Pose-Imitation-of-Human-Motion-by-a-Simulated-Humanoid-Robot.wbt`) |

---

## 1. Background

Humanoid robots are a topic of strong scientific interest because they promise to operate in environments designed for humans without costly modifications. However, bipedal locomotion and whole-body action require intelligent control to maintain balance. Recent research increasingly leverages machine learning and human-motion imitation to address this challenge ([1], [2]).

This project explores **markerless visual teleoperation** of a humanoid robot inside the **Webots** simulation environment. A single human is recorded by a tripod-mounted RGB camera; the captured video is processed to extract 2D/3D human pose, mapped to the robot's joint space, and used to drive the simulated humanoid's actuators in (near) real time.

References:
- [1] Ze et al., *TWIST: Teleoperated Whole-Body Imitation System*, CoRL 2025. https://doi.org/10.48550/arXiv.2505.02833
- [2] Tao et al., *Visual Perception Method Based on Human Pose Estimation for Humanoid Robot Imitating Human Motions*, CCRIS '21, pp. 54–61. https://doi.org/10.1145/3483845.3483867

---

## 2. Project Goal

Build an end-to-end pipeline in which a simulated humanoid robot in Webots imitates the live or recorded motion of a single human, using only a single monocular video stream as input.

### 2.1 In Scope
- Single human subject, full body in frame, tripod-mounted **Sony A7 III** camera (static viewpoint).
- Monocular RGB input at **1920×1080** with an **adaptive frame rate of 25–100 FPS** depending on system resources and pipeline load.
- Live capture (HDMI capture card / USB streaming) and/or pre-recorded video file.
- Pose estimation in Python.
- Retargeting of human pose to the humanoid robot's joint structure.
- Webots controller (Python) that drives the robot's joints.
- Basic balance handling sufficient for upper-body and slow lower-body imitation.
- Quantitative + qualitative evaluation of imitation fidelity and latency.

### 2.2 Out of Scope (initial release)
- Multi-person tracking.
- Dynamic camera or moving viewpoint.
- Hardware deployment on a physical humanoid robot.
- Highly dynamic motions (running, jumping, acrobatics).
- Hand/finger-level dexterity; facial expression imitation.
- Reinforcement-learning-based whole-body controller (may be future work).

---

## 3. Stakeholders

| Role | Responsibility |
|---|---|
| Student / Developer | Implementation, evaluation, documentation. |
| Supervisor (S. Stahl) | Requirements clarification, scientific guidance, grading. |
| End user (demo) | Stands in front of the camera; observes robot imitation. |

---

## 4. User Stories

- **US-1:** As a user, I can stand in front of a webcam and see the simulated humanoid mirror my upper-body motions in Webots in near real time.
- **US-2:** As a developer, I can replay a recorded video and have the robot imitate the same motion deterministically for evaluation.
- **US-3:** As a researcher, I can log joint trajectories (human vs. robot) and compute imitation-fidelity metrics offline.
- **US-4:** As a user, the robot should not fall over during normal standing/upper-body imitation.

---

## 5. System Architecture

```
+----------------+     +-------------------+     +------------------+     +----------------------+
|  Camera /      | --> | Pose Estimation   | --> | Retargeting /    | --> | Webots Controller    |
|  Video file    |     | (2D/3D keypoints) |     | IK to robot DoF  |     | (Python, joint cmds) |
+----------------+     +-------------------+     +------------------+     +----------------------+
                                                                                 |
                                                                                 v
                                                                       +-------------------+
                                                                       | Webots Simulation |
                                                                       | (humanoid robot)  |
                                                                       +-------------------+
                                                                                 |
                                                                                 v
                                                                       +-------------------+
                                                                       | Logging / Metrics |
                                                                       +-------------------+
```

### 5.1 Components

1. **Video Input Module**
   - Primary source: **Sony A7 III** mounted on a tripod, captured into the host PC via an HDMI-to-USB capture device (e.g., Elgato Cam Link 4K) or the camera's USB streaming mode, exposed to the OS as a UVC video device.
   - Secondary source: pre-recorded video file (MP4/MOV) recorded with the same camera for deterministic replay.
   - Capture parameters: **1920×1080 (Full HD)**, color, progressive scan.
   - **Adaptive frame rate: 25–100 FPS**, automatically selected based on:
     - measured pose-pipeline throughput (rolling average),
     - CPU/GPU utilization,
     - end-to-end latency budget (NFR-1).
   - Frame-rate controller: starts at a conservative 25 FPS, ramps up toward 100 FPS while the pipeline keeps up; backs off when latency exceeds the budget or frames are being dropped.
   - Library: OpenCV `VideoCapture` (with `CAP_AVFOUNDATION` on macOS, `CAP_V4L2` on Linux).
   - Each frame is timestamped at capture time and passed to the pose estimator.

2. **Pose Estimation Module**
   - Library candidates (Python): **MediaPipe Pose**, **MMPose**, or **Ultralytics YOLOv8-Pose**.
   - Output: per-frame keypoints (33 landmarks for MediaPipe) with 3D world coordinates and visibility.
   - Smoothing: One-Euro filter or exponential smoothing to reduce jitter.

   #### 5.1.2 MediaPipe Pose Landmark Set (33 landmarks)

   The system SHALL consume all 33 MediaPipe Pose landmarks. Each landmark provides `(x, y, z, visibility)` where `x, y` are normalized image coordinates, `z` is depth relative to the hips (in normalized units), and `visibility ∈ [0, 1]`.

   | ID | Name | ID | Name | ID | Name |
   |----|------|----|------|----|------|
   | 0  | nose                | 11 | left_shoulder        | 22 | right_thumb          |
   | 1  | left_eye_inner      | 12 | right_shoulder       | 23 | left_hip             |
   | 2  | left_eye            | 13 | left_elbow           | 24 | right_hip            |
   | 3  | left_eye_outer      | 14 | right_elbow          | 25 | left_knee            |
   | 4  | right_eye_inner     | 15 | left_wrist           | 26 | right_knee           |
   | 5  | right_eye           | 16 | right_wrist          | 27 | left_ankle           |
   | 6  | right_eye_outer     | 17 | left_pinky           | 28 | right_ankle          |
   | 7  | left_ear            | 18 | right_pinky          | 29 | left_heel            |
   | 8  | right_ear           | 19 | left_index           | 30 | right_heel           |
   | 9  | mouth_left          | 20 | right_index          | 31 | left_foot_index      |
   | 10 | mouth_right         | 21 | left_thumb           | 32 | right_foot_index     |

   Notes:
   - Reference figure: `docs/` (MediaPipe Pose skeleton diagram).
   - Naming follows MediaPipe’s `mp.solutions.pose.PoseLandmark` enum (uppercase form, e.g. `LEFT_SHOULDER`).
   - The retargeting module currently uses a curated subset relevant to humanoid joints (shoulders, elbows, wrists, hips, knees, ankles, head reference). The remaining landmarks (face, fingers, feet indices) are still **logged** for downstream metrics, gesture extensions, and future work.

3. **Retargeting Module**
   - Convert human keypoints into joint angles compatible with the Webots humanoid (e.g., NAO, Atlas, or a custom URDF/PROTO).
   - Approach:
     - Compute joint angles directly from vector geometry between keypoints (shoulder, elbow, hip, knee, etc.).
     - Optional analytical/numerical IK for end-effector targets.
   - Apply joint limits and rate limits to ensure mechanically feasible commands.

4. **Webots Controller (Python)**
   - Communicates with Webots via the `controller` Python API.
   - Reads target joint angles from the retargeting module (shared queue / IPC / socket).
   - Sends `setPosition()` commands to the relevant motors at each control step.
   - Optional: simple balance assist (fixed feet, or PD-stabilized torso) for initial milestones.

5. **Logging & Evaluation**
   - Log: input frames (optional), human keypoints, target joint angles, achieved joint angles, timestamps.
   - Metrics: per-joint MAE, end-to-end latency, dropped frames, simulation-vs-real-time ratio.

### 5.2 Communication Between Pose Pipeline and Webots
- **Option A (preferred):** Pose pipeline runs inside the Webots Python controller process when performance allows.
- **Option B:** Pose pipeline runs as a separate process and streams joint targets to the controller via local socket / ZeroMQ / shared memory. Chosen automatically based on performance benchmarks.

---

## 6. Functional Requirements

| ID | Requirement |
|---|---|
| FR-1 | The system SHALL accept a live feed from a Sony A7 III (via UVC / HDMI capture) and a recorded video file as input sources, selectable via configuration. |
| FR-1a | The system SHALL capture video at 1920×1080 resolution. |
| FR-1b | The system SHALL operate at an adaptive frame rate between 25 FPS and 100 FPS, automatically scaling based on pipeline throughput and latency. |
| FR-2 | The system SHALL extract human body keypoints from each input frame. |
| FR-3 | The system SHALL map estimated keypoints to the simulated humanoid's joint space. |
| FR-4 | The Webots controller SHALL command the humanoid's motors to follow the mapped joint trajectory each simulation step. |
| FR-5 | The system SHALL clip joint commands to the robot's mechanical limits. |
| FR-6 | The system SHALL temporally smooth keypoints / joint commands to prevent unsafe oscillations. |
| FR-7 | The system SHALL log human keypoints and robot joint states with timestamps to disk. |
| FR-8 | The system SHALL be configurable via a single config file (YAML/TOML). |
| FR-9 | The system SHALL provide a runnable demo via a single command (e.g., `python run.py`). |

---

## 7. Non-Functional Requirements

| ID | Requirement | Target |
|---|---|---|
| NFR-1 | End-to-end latency (camera frame → joint command) | ≤ 150 ms (goal), ≤ 300 ms (acceptable) |
| NFR-2 | Pose pipeline throughput | ≥ 25 FPS sustained (lower bound), target 60+ FPS, up to 100 FPS when resources allow |
| NFR-3 | Imitation fidelity (upper body, MAE per joint) | ≤ 10° on representative test motions |
| NFR-4 | Robot stability | No falls during standing upper-body imitation in baseline scenario |
| NFR-5 | Reproducibility | Fixed random seeds; pinned dependency versions |
| NFR-6 | Code quality | Type hints, linting (ruff), formatting (black), unit tests for retargeting math |
| NFR-7 | Portability | Runs on macOS and Linux with Webots R2023b or newer |

---

## 8. Tooling & Dependencies

- **Simulator:** Webots R2023b+. The `main/` directory is the Webots project root (contains `worlds/Pose-Imitation-of-Human-Motion-by-a-Simulated-Humanoid-Robot.wbt`, plus `controllers/`, `protos/`, `plugins/` as needed). Webots is launched against this project directory.
- **Capture hardware:** Sony A7 III (tripod-mounted) + HDMI-to-USB capture device (UVC) or USB streaming, providing 1920×1080 @ 25–100 FPS to the host.
- **Language:** Python 3.10+.
- **Key libraries:** `opencv-python`, `mediapipe` (or `ultralytics`), `numpy`, `scipy`, `pyyaml`, `matplotlib` (analysis), `pytest`.
- **Webots API:** `controller` Python module shipped with Webots.
- **Optional:** `pyzmq` for inter-process messaging.

---

## 9. Proposed Repository Structure

```
.
├── docs/
│   ├── CSPM 2026S Stahl Project.pdf
│   └── PRD.md
├── main/                       # Webots project root
│   ├── worlds/
│   │   └── Pose-Imitation-of-Human-Motion-by-a-Simulated-Humanoid-Robot.wbt
│   ├── controllers/            # Webots Python controllers (one folder per controller)
│   ├── protos/                 # custom PROTO models (if any)
│   └── plugins/                # optional Webots plugins
├── src/
│   ├── perception/             # camera capture (Sony A7 III) + pose estimation
│   ├── retargeting/            # keypoints -> joint angles
│   ├── utils/                  # logging, filters, adaptive FPS controller, config
│   └── run.py                  # entry point (launches pose pipeline + Webots)
├── tests/
├── configs/
│   └── default.yaml
├── requirements.txt
├── README.md
└── PRD.md  (top-level pointer to docs/PRD.md)
```

---

## 10. Milestones & Timeline

| # | Milestone | Deliverable | Target |
|---|---|---|---|
| M1 | Environment setup | Webots world loads; humanoid robot present; Python controller stub commands one motor. | Week 1–2 |
| M2 | Pose estimation prototype | Standalone script extracts and visualizes keypoints from webcam/video. | Week 3 |
| M3 | Retargeting v1 (upper body) | Shoulder/elbow joint angles drive the simulated robot's arms. | Week 4–5 |
| M4 | Real-time integration | Live webcam → live robot imitation in Webots, end-to-end. | Week 6 |
| M5 | Lower body + stability | Hip/knee/ankle imitation with feet pinned or PD-stabilized torso. | Week 7–8 |
| M6 | Evaluation & logging | Metrics, plots, recorded demo videos. | Week 9 |
| M7 | Final report & cleanup | Documentation, reproducible demo, code review. | Week 10 |

---

## 11. Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Monocular 3D pose is ambiguous (depth, self-occlusion). | Inaccurate retargeting. | Use MediaPipe World Landmarks; constrain via joint limits; consider multi-view extension later. |
| Direct joint mapping causes the robot to fall. | Demo failure. | Start with feet fixed; add PD torso stabilization; restrict CoM-affecting joints. |
| Latency too high for real-time feel. | Poor UX. | Run pose estimation in a separate process; downscale frames; use GPU model. |
| Webots humanoid model joint structure differs from human skeleton. | Mapping errors. | Build an explicit, documented mapping table; validate per-joint with test motions. |
| Library/version drift (Webots, MediaPipe). | Reproducibility issues. | Pin versions in `requirements.txt`; document Webots release. |
| Sony A7 III capture chain (HDMI/UVC) introduces extra latency or frame drops. | Higher end-to-end latency, jitter. | Benchmark capture latency; prefer hardware capture device with low-latency UVC mode; expose driver settings in config. |
| Sustaining 100 FPS not feasible on target hardware. | Underutilized capture rate. | Adaptive FPS controller (FR-1b) gracefully downscales toward 25 FPS without breaking the pipeline. |

---

## 12. Acceptance Criteria

The project is considered complete when:
1. Running a single command launches Webots and the Python pipeline together.
2. With a live webcam, the simulated humanoid visibly imitates the user's upper-body motion in real time without falling.
3. Replaying a recorded reference video reproduces equivalent robot motion deterministically.
4. Quantitative metrics (latency, per-joint MAE) are reported in `docs/` along with plots.
5. The repository contains documentation sufficient for a new developer to reproduce the results.

---

## 13. Open Questions

1. **Robot model:** Should we use a built-in Webots humanoid (e.g., NAO, Atlas) or a custom PROTO model? Any preference from the supervisor?
2. **Evaluation motions:** Is there a required set of reference motions/poses to evaluate against?
3. **Real-time vs. offline:** Is real-time imitation a hard requirement, or is offline replay of recorded video acceptable for grading?
4. **Hardware:** Will a GPU be available for pose estimation, or must the pipeline run on CPU only?
5. **Multi-view extension:** Is a stretch goal of multi-camera input of interest, or strictly single camera as in [2]?

> Please confirm or clarify the items above so they can be locked into v1.1 of this PRD.
