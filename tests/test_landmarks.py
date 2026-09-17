from __future__ import annotations

from src.perception.landmarks import (
    CANONICAL_TO_RAW_ALIASES,
    HAND_LANDMARKS,
    NUM_LANDMARKS,
    OPTIONAL_LANDMARKS,
    POSE_LANDMARKS,
    REQUIRED_LANDMARKS,
    SUPERSET_NAMES,
    build_raw_to_canonical_map,
    landmark_id,
)


def test_the_body_skeleton_is_still_the_nineteen_coco_19_joints() -> None:
    """Hands were ADDED, not substituted. Everything that decides where a limb
    is still reads the same nineteen points it always did."""
    assert len(REQUIRED_LANDMARKS) == 19
    assert set(REQUIRED_LANDMARKS).isdisjoint(HAND_LANDMARKS)


def test_landmark_count_covers_body_plus_hands() -> None:
    assert NUM_LANDMARKS == 25
    assert len(POSE_LANDMARKS) == 25
    assert POSE_LANDMARKS[:19] == REQUIRED_LANDMARKS
    assert OPTIONAL_LANDMARKS == frozenset(HAND_LANDMARKS)


def test_landmark_id_round_trip() -> None:
    for idx, name in enumerate(POSE_LANDMARKS):
        assert landmark_id(name) == idx


def test_key_anatomy_present() -> None:
    expected = {
        "nose", "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle",
    }
    assert expected.issubset(set(POSE_LANDMARKS))


def test_every_canonical_landmark_has_an_alias() -> None:
    assert set(CANONICAL_TO_RAW_ALIASES.keys()) == set(POSE_LANDMARKS)


def test_build_raw_to_canonical_map_matches_exact_names() -> None:
    # The common case: a live model reports exactly our canonical names.
    mapping = build_raw_to_canonical_map(list(POSE_LANDMARKS))
    assert set(mapping.values()) == set(POSE_LANDMARKS)


def test_build_raw_to_canonical_map_matches_common_abbreviations() -> None:
    raw = ["lsho", "rsho", "lhip", "rhip", "nose"]
    mapping = build_raw_to_canonical_map(raw)
    assert mapping["lsho"] == "left_shoulder"
    assert mapping["rsho"] == "right_shoulder"
    assert mapping["lhip"] == "left_hip"
    assert mapping["rhip"] == "right_hip"
    assert mapping["nose"] == "nose"


def test_build_raw_to_canonical_map_ignores_unknown_names() -> None:
    mapping = build_raw_to_canonical_map(["left_shoulder", "some_unknown_joint"])
    assert mapping == {"left_shoulder": "left_shoulder"}


# The joint names a live metrabs_eff2s_y4 model reports for `coco_19`, in the
# model's own order (verified against the loaded SavedModel, and printable with
# scripts/inspect_metrabs_skeleton.py). Pinned here because the alias table was
# originally written by guesswork: nine of these matched nothing, so the
# pipeline refused to start on the very first frame it ever saw a GPU.
METRABS_COCO_19_RAW = [
    "neck", "nose", "pelv",
    "lsho", "lelb", "lwri", "lhip", "lkne", "lank",
    "rsho", "relb", "rwri", "rhip", "rkne", "rank",
    "leye", "lear", "reye", "rear",
]


def test_every_real_metrabs_joint_maps_to_a_canonical_name() -> None:
    mapping = build_raw_to_canonical_map(METRABS_COCO_19_RAW)
    unmatched = [r for r in METRABS_COCO_19_RAW if r not in mapping]
    assert not unmatched, f"raw MeTRAbs names with no canonical match: {unmatched}"


def test_the_real_metrabs_skeleton_covers_every_required_landmark() -> None:
    """The check pose_estimator.py makes at startup, run offline."""
    mapping = build_raw_to_canonical_map(METRABS_COCO_19_RAW)
    missing = [n for n in REQUIRED_LANDMARKS if n not in set(mapping.values())]
    assert not missing, f"canonical landmarks the model cannot supply: {missing}"


def test_coco_19_supplies_no_hands_and_that_is_not_an_error() -> None:
    """coco_19 stops at the wrist. Running it must leave the hand landmarks
    unmatched WITHOUT tripping the startup check -- the hand joints simply go
    untracked, as they did before hands existed."""
    mapping = build_raw_to_canonical_map(METRABS_COCO_19_RAW)
    matched = set(mapping.values())
    assert not (matched & set(HAND_LANDMARKS))
    assert all(name in matched for name in REQUIRED_LANDMARKS)


# ---------------------------------------------------------------------------
# The alias trap. Read off a live metrabs_eff2s_y4: every skeleton is an index
# gather out of ONE 122-joint superset, and that superset holds two different
# joints for most limbs -- a bare SMPL joint and a `_cmu_panoptic` surface
# marker several centimetres away. coco_19 REPORTS the bare names but SELECTS
# the cmu_panoptic indices, so a first-match-wins lookup that tries the bare
# name first silently moves every landmark the moment the superset is used.
# ---------------------------------------------------------------------------
SUPERSET_SAMPLE = [
    # The bare SMPL joints, listed FIRST so a naive lookup would find them first.
    "lsho", "lelb", "lwri", "lhip", "lkne", "lank", "lhan",
    "rsho", "relb", "rwri", "rhip", "rkne", "rank", "rhan",
    "neck", "nose", "pelv", "leye", "lear", "reye", "rear",
    # The cmu_panoptic surface markers coco_19 actually selects.
    "lsho_cmu_panoptic", "lelb_cmu_panoptic", "lwri_cmu_panoptic",
    "lhip_cmu_panoptic", "lkne_cmu_panoptic", "lank_cmu_panoptic",
    "rsho_cmu_panoptic", "relb_cmu_panoptic", "rwri_cmu_panoptic",
    "rhip_cmu_panoptic", "rkne_cmu_panoptic", "rank_cmu_panoptic",
    "neck_cmu_panoptic", "nose_cmu_panoptic", "pelv_cmu_panoptic",
    "leye_cmu_panoptic", "lear_cmu_panoptic",
    "reye_cmu_panoptic", "rear_cmu_panoptic",
    # The H36M hand markers, which exist only here.
    "lwri_h36m", "lthu_h36m", "lfin_h36m",
    "rwri_h36m", "rthu_h36m", "rfin_h36m",
]


def test_the_superset_resolves_the_marker_not_the_smpl_joint() -> None:
    """The trap, pinned. Without SUPERSET_NAMES fourteen of the nineteen body
    landmarks bind to the SMPL joint -- which sits medially, inside the body --
    and the skeleton still looks plausible enough that nothing complains."""
    mapping = build_raw_to_canonical_map(SUPERSET_SAMPLE)
    raw_for = {canonical: raw for raw, canonical in mapping.items()}
    wrong = {
        canonical: raw_for.get(canonical)
        for canonical, expected in SUPERSET_NAMES.items()
        if raw_for.get(canonical) != expected
    }
    assert not wrong, f"landmarks bound to the wrong superset joint: {wrong}"


def test_the_superset_supplies_the_hands() -> None:
    mapping = build_raw_to_canonical_map(SUPERSET_SAMPLE)
    matched = set(mapping.values())
    assert set(HAND_LANDMARKS) <= matched
    assert set(REQUIRED_LANDMARKS) <= matched


def test_switching_to_the_superset_does_not_move_a_single_body_landmark() -> None:
    """coco_19 and the superset must name the SAME physical point for every
    body landmark, or the retargeting constants (all of them ratios of measured
    segment lengths) silently change meaning."""
    coco = build_raw_to_canonical_map(METRABS_COCO_19_RAW)
    superset = build_raw_to_canonical_map(SUPERSET_SAMPLE)
    coco_raw = {c: r for r, c in coco.items()}
    super_raw = {c: r for r, c in superset.items()}
    for name in REQUIRED_LANDMARKS:
        # coco_19 reports the bare name; the superset reports the suffixed one.
        # Same joint, and that is exactly what has to stay true.
        assert super_raw[name] == f"{coco_raw[name]}_cmu_panoptic", (
            f"{name}: coco_19 gives {coco_raw[name]!r} but the superset "
            f"resolves to {super_raw[name]!r}")


def test_no_two_raw_names_collapse_onto_the_same_canonical_joint() -> None:
    """A sloppy alias (e.g. a bare 'l' prefix rule) can silently make 'lear'
    and 'lelb' fight over one slot; the loser is then read as a missing joint."""
    mapping = build_raw_to_canonical_map(METRABS_COCO_19_RAW)
    assert len(set(mapping.values())) == len(mapping)


def test_left_and_right_are_not_swapped() -> None:
    mapping = build_raw_to_canonical_map(METRABS_COCO_19_RAW)
    for raw, canonical in mapping.items():
        if raw.startswith("l") and canonical != "neck":
            assert canonical.startswith("left_"), f"{raw} -> {canonical}"
        elif raw.startswith("r"):
            assert canonical.startswith("right_"), f"{raw} -> {canonical}"
