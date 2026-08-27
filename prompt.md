Repo: https://github.com/tarikbilla/Pose-Imitation-of-Human-Motion-by-a-Simulated-Humanoid-Robot
Branch: A2-MeTRAbs

This branch replaces MediaPipe with MeTRAbs (GPU-accelerated 3D pose estimation) across the whole pipeline, including a full rewrite of main/libraries/nao_retarget.py around real 3D geometry (a per-frame torso-local basis + swing-twist joint decomposition). It was written entirely on a machine with no Python interpreter and no GPU, so none of it has actually been executed — only manually derived and algebraically verified. Your job is to verify it on this machine, which does have a GPU, and fix whatever doesn't hold up.

Please do the following, in order:

1. Set up the environment per docs/RUN_INSTRUCTIONS.md (conda env, TensorFlow + TensorFlow-Hub with a GPU build matching this machine's CUDA/cuDNN driver). Confirm with:
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   It must print a non-empty GPU list.

2. Run scripts/inspect_metrabs_skeleton.py. It downloads the MeTRAbs model and checks the real joint names it reports against src/perception/landmarks.py's CANONICAL_TO_RAW_ALIASES table (which is my best-effort guess, not verified against a live model). It should print "OK: every canonical landmark matched, no leftover raw names." If it instead lists MISSING canonical landmarks, fix CANONICAL_TO_RAW_ALIASES using the raw names it prints — every downstream joint lookup depends on this being correct, so don't skip this step. pose_estimator.py will refuse to start at all if the mapping is wrong, so this is a hard blocker, not a nice-to-have.

3. Run the full test suite: pytest tests/ -v
   Pay special attention to tests/test_nao_retarget.py — it round-trip-tests the new swing-twist leg/arm geometry (project known NAO angles into synthetic 3D landmarks, recover them, assert they match almost exactly). I verified this algebra by hand, but confirm it actually passes. If it doesn't, the bug is most likely in nao_retarget.py's _torso_frame, _swing_twist, _arm, or _leg functions — check sign conventions there first before touching the test.

4. Run the pipeline against a live webcam (python run.py --no-webots first, then with Webots) and specifically verify the left/right anatomical labeling convention MeTRAbs uses. The yaw computation in src/perception/gait_cues.py and the mirroring logic in main/libraries/nao_retarget.py both assume MeTRAbs' coco_19 skeleton follows the same anatomical-left/right convention MediaPipe used (left_shoulder = anatomical left, appearing on the image's right when facing the camera) — this was measured empirically on MediaPipe data and carried over as an assumption, not verified against real MeTRAbs output. Concretely: stand facing the camera square-on and confirm the gait command's body_yaw_rad reads close to 0 degrees. If it reads close to +/-180 degrees instead, the left/right convention is inverted — see the sign-convention notes in gait_cues.py's module docstring for where to fix it.

5. More generally: watch for any other place where I explicitly flagged an assumption or an unverified claim in a comment or docstring (search for "unverified", "not been re-measured", "best-effort", "written without GPU access") and confirm or correct each one against real behavior on this machine.

Report back concretely: what passed as-is, what you had to fix, and what (if anything) still doesn't work.
