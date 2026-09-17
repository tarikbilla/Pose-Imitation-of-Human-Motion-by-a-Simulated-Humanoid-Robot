"""MeTRAbs skeleton landmark definitions.

Reference: https://github.com/isarandi/metrabs, docs/API.md ("Skeleton
Conventions"). The 19 body landmarks match MeTRAbs' ``coco_19`` convention
(shoulders, elbows, wrists, hips, knees, ankles, face points) without SMPL's
extra spine/collar joints.

Hands
-----
``coco_19`` stops at the wrist, which leaves NAO's ``ElbowYaw``, ``WristYaw``
and its finger motors with nothing to track -- a roll joint rotates about the
axis its own bone lies along, so no accuracy on shoulder/elbow/wrist reveals
it. :data:`HAND_LANDMARKS` adds a thumb and a finger marker per side, which is
enough to pin the hand's own frame and so to solve all three.

Those points exist only in MeTRAbs' **full 122-joint superset**
(``pose.skeleton: ""``). Asking for the superset costs nothing: every named
skeleton is a plain index gather out of it -- measured at 152.1 ms/call for the
superset against 152.4 ms for ``coco_19`` -- so the argument picks a selection,
not a different inference.

THE ALIAS TRAP, AND IT IS SILENT
--------------------------------
The superset carries *two* joints for most limbs: a bare SMPL joint (``lwri``,
index 19) and a CMU-Panoptic surface marker (``lwri_cmu_panoptic``, index 64).
They are different points -- SMPL's sits medially, inside the body. ``coco_19``
REPORTS the bare names but SELECTS the ``_cmu_panoptic`` indices, verified
index by index against ``metrabs_eff2s_y4``:

    coco_19 'lwri' -> superset [ 64] lwri_cmu_panoptic
    coco_19 'lsho' -> superset [ 56] lsho_cmu_panoptic       (all 19 like this)

So a name-ordered alias tuple that happens to try ``lwri`` first resolves to
the SMPL joint the moment the skeleton is switched to the superset, and every
arm and leg landmark quietly moves inside the body -- with no error, no warning
and a skeleton that still looks plausible. :data:`SUPERSET_NAMES` is therefore
authoritative and is prepended to each alias tuple below, so the superset
reproduces ``coco_19`` exactly. Pinned by ``tests/test_landmarks.py``.

The abbreviated aliases below (``lsho``, ``lelb``, ``pelv`` ...) are the names
a live ``metrabs_eff2s_y4`` model actually reports for ``coco_19``; they are
pinned by ``tests/test_landmarks.py`` so a future model whose naming drifts
fails in CI rather than at the first frame on the GPU machine.

This list is NOT hard-relied upon for correctness: at runtime, ``pose_estimator.py``
reads the actual joint names for the loaded model
(``metrabs_model.skeleton_info(...)``) and normalizes them against
``CANONICAL_TO_RAW_ALIASES`` below, logging a loud warning (and refusing to
start unless ``pose.allow_synthetic_fallback`` is set) for any canonical name
it cannot match. Run ``scripts/inspect_metrabs_skeleton.py`` once on the
target GPU machine to print the model's actual names/edges and correct this
file (and the alias table) if they differ.
"""
from __future__ import annotations

# Canonical joint names used throughout this codebase (dict keys on
# PoseFrame.keypoints). Order is not semantically meaningful -- lookups are by
# name -- but is kept stable for iteration/logging.
POSE_LANDMARKS: list[str] = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "neck",     # coco_19 extension over plain 17-point COCO: shoulder midpoint
    "pelvis",   # coco_19 extension over plain 17-point COCO: hip midpoint
]

# Hand detail, present only in the 122-joint superset (pose.skeleton: "").
# OPTIONAL: running coco_19 simply omits them and the hand joints go untracked,
# which is what the whole project did before these were added.
#
# All three markers on a side come from the SAME dataset convention (H36M), and
# that is deliberate: the hand solve only ever uses DIFFERENCES between them
# (wrist->finger for the hand's long axis, wrist->thumb for which way the palm
# faces), so what matters is that they are mutually consistent, not that they
# agree with the cmu_panoptic wrist the arm chain uses. Mixing an SMPL hand
# centre with an H36M thumb would put a fictitious twist in the palm frame.
HAND_LANDMARKS: list[str] = [
    "left_hand_root",    # H36M wrist marker: the origin of the hand frame
    "left_thumb",
    "left_finger",
    "right_hand_root",
    "right_thumb",
    "right_finger",
]

# Landmarks the pipeline cannot start without. The hand ones are not among
# them: they are absent from every skeleton but the superset.
REQUIRED_LANDMARKS: list[str] = list(POSE_LANDMARKS)
OPTIONAL_LANDMARKS: frozenset[str] = frozenset(HAND_LANDMARKS)

POSE_LANDMARKS = POSE_LANDMARKS + HAND_LANDMARKS

NUM_LANDMARKS: int = len(POSE_LANDMARKS)

# The superset joint name each canonical landmark MUST resolve to. Read off a
# live ``metrabs_eff2s_y4`` (``per_skeleton_indices``/``per_skeleton_joint_names``)
# rather than guessed -- see "THE ALIAS TRAP" in the module docstring for what
# happens without it.
SUPERSET_NAMES: dict[str, str] = {
    "nose": "nose_cmu_panoptic",
    "left_eye": "leye_cmu_panoptic",
    "right_eye": "reye_cmu_panoptic",
    "left_ear": "lear_cmu_panoptic",
    "right_ear": "rear_cmu_panoptic",
    "left_shoulder": "lsho_cmu_panoptic",
    "right_shoulder": "rsho_cmu_panoptic",
    "left_elbow": "lelb_cmu_panoptic",
    "right_elbow": "relb_cmu_panoptic",
    "left_wrist": "lwri_cmu_panoptic",
    "right_wrist": "rwri_cmu_panoptic",
    "left_hip": "lhip_cmu_panoptic",
    "right_hip": "rhip_cmu_panoptic",
    "left_knee": "lkne_cmu_panoptic",
    "right_knee": "rkne_cmu_panoptic",
    "left_ankle": "lank_cmu_panoptic",
    "right_ankle": "rank_cmu_panoptic",
    "neck": "neck_cmu_panoptic",
    "pelvis": "pelv_cmu_panoptic",
    "left_hand_root": "lwri_h36m",
    "left_thumb": "lthu_h36m",
    "left_finger": "lfin_h36m",
    "right_hand_root": "rwri_h36m",
    "right_thumb": "rthu_h36m",
    "right_finger": "rfin_h36m",
}

# Best-effort mapping from OUR canonical name to the raw name(s) MeTRAbs might
# use for the same joint, tried in order, case-insensitively. Matching also
# falls back to normalizing the raw name (lowercasing, "l_"/"r_" -> "left_"/
# "right_", stripping underscores) before comparing, so small spelling
# differences (e.g. "lshoulder" vs "left_shoulder") still resolve.
CANONICAL_TO_RAW_ALIASES: dict[str, tuple[str, ...]] = {
    "nose": ("nose",),
    "left_eye": ("left_eye", "leye", "l_eye"),
    "right_eye": ("right_eye", "reye", "r_eye"),
    "left_ear": ("left_ear", "lear", "l_ear"),
    "right_ear": ("right_ear", "rear", "r_ear"),
    "left_shoulder": ("left_shoulder", "lshoulder", "l_shoulder", "lsho"),
    "right_shoulder": ("right_shoulder", "rshoulder", "r_shoulder", "rsho"),
    "left_elbow": ("left_elbow", "lelbow", "l_elbow", "lelb"),
    "right_elbow": ("right_elbow", "relbow", "r_elbow", "relb"),
    "left_wrist": ("left_wrist", "lwrist", "l_wrist", "lwri"),
    "right_wrist": ("right_wrist", "rwrist", "r_wrist", "rwri"),
    "left_hip": ("left_hip", "lhip", "l_hip"),
    "right_hip": ("right_hip", "rhip", "r_hip"),
    "left_knee": ("left_knee", "lknee", "l_knee", "lkne"),
    "right_knee": ("right_knee", "rknee", "r_knee", "rkne"),
    "left_ankle": ("left_ankle", "lankle", "l_ankle", "lank"),
    "right_ankle": ("right_ankle", "rankle", "r_ankle", "rank"),
    "neck": ("neck",),
    "pelvis": ("pelvis", "pelv", "root", "hip"),
    # Hand markers. No short bare aliases (``lthu``/``lfin``) are listed ahead
    # of the superset name for the same reason as above -- h36m_25 reports the
    # bare form, the superset reports the suffixed one, and only one of them is
    # the marker we measured.
    "left_hand_root": ("lwri_h36m", "left_hand_root"),
    "left_thumb": ("lthu_h36m", "left_thumb", "lthu"),
    "left_finger": ("lfin_h36m", "left_finger", "lfin"),
    "right_hand_root": ("rwri_h36m", "right_hand_root"),
    "right_thumb": ("rthu_h36m", "right_thumb", "rthu"),
    "right_finger": ("rfin_h36m", "right_finger", "rfin"),
}

# Put the authoritative superset name at the FRONT of every alias tuple, so a
# first-match-wins lookup against the 122-joint superset lands on the joint
# coco_19 would have selected rather than on its same-named SMPL neighbour.
for _canonical, _raw in SUPERSET_NAMES.items():
    _aliases = CANONICAL_TO_RAW_ALIASES.get(_canonical, ())
    if _raw not in _aliases[:1]:
        CANONICAL_TO_RAW_ALIASES[_canonical] = (_raw,) + tuple(
            a for a in _aliases if a != _raw
        )
del _canonical, _raw, _aliases

# Bone connections for the skeleton overlay, as pairs of canonical names. Used
# as a fallback when the live model's own edge list (via
# ``metrabs_model.skeleton_info``) is unavailable (e.g. offline unit tests).
POSE_CONNECTIONS: tuple[tuple[str, str], ...] = (
    ("left_ear", "left_eye"),
    ("left_eye", "nose"),
    ("nose", "right_eye"),
    ("right_eye", "right_ear"),
    ("neck", "nose"),
    ("neck", "left_shoulder"),
    ("neck", "right_shoulder"),
    ("neck", "pelvis"),
    ("pelvis", "left_hip"),
    ("pelvis", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    # Hand detail. Drawn from the WRIST rather than from left_hand_root so the
    # overlay shows the hand attached to the arm the robot is actually driving;
    # the solve itself uses left_hand_root (see HAND_LANDMARKS).
    ("left_wrist", "left_finger"),
    ("left_wrist", "left_thumb"),
    ("right_wrist", "right_finger"),
    ("right_wrist", "right_thumb"),
)


def _normalize(name: str) -> str:
    """Lower-case a joint name and expand an ``l_``/``r_`` side prefix.

    Deliberately does NOT expand a bare leading ``l``/``r``: "lear" would
    become "left_ear" but "lelb" would become "left_elb", and "neck"/"nose"
    would not survive the same rule at all. Abbreviations are handled by
    listing them explicitly in :data:`CANONICAL_TO_RAW_ALIASES` instead.
    """
    n = name.strip().lower().replace("-", "_")
    if n.startswith("l_"):
        return "left_" + n[2:]
    if n.startswith("r_"):
        return "right_" + n[2:]
    return n


def build_raw_to_canonical_map(raw_names: list[str]) -> dict[str, str]:
    """Match a live model's raw joint names to our canonical names.

    Returns ``{raw_name: canonical_name}`` for every raw name that could be
    matched. Names that cannot be matched are omitted -- callers should treat
    an incomplete match against ``POSE_LANDMARKS`` as a loud, fatal error
    (see ``pose_estimator.PoseEstimator``), not something to silently ignore.
    """
    normalized_raw = {_normalize(r): r for r in raw_names}
    result: dict[str, str] = {}
    for canonical, aliases in CANONICAL_TO_RAW_ALIASES.items():
        for alias in aliases:
            if alias in normalized_raw:
                result[normalized_raw[alias]] = canonical
                break
            norm_alias = _normalize(alias)
            if norm_alias in normalized_raw:
                result[normalized_raw[norm_alias]] = canonical
                break
    return result


def landmark_id(name: str) -> int:
    """Index of ``name`` in the canonical :data:`POSE_LANDMARKS` order."""
    return POSE_LANDMARKS.index(name)


def enumerate_landmarks() -> list[tuple[int, str]]:
    return list(enumerate(POSE_LANDMARKS))
