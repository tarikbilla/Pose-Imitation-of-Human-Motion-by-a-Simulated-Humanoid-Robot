import cv2

from .skeleton import A3_INDEX as HALPE26_INDEX
from .skeleton import HAND_JOINTS
from .skeleton import SKELETON_LINKS as SKELETON

COLOR_LEFT = (90, 200, 255)
COLOR_RIGHT = (255, 170, 80)
COLOR_TRUNK = (200, 200, 200)
COLOR_HAND = (140, 255, 170)
COLOR_LOW = (80, 80, 90)
COLOR_TEXT = (240, 240, 240)
COLOR_PANEL = (28, 28, 32)

SCORE_THRESHOLD = 0.35


def _limb_color(a, b):
    if a in HAND_JOINTS or b in HAND_JOINTS:
        return COLOR_HAND
    if a.startswith("left") or b.startswith("left"):
        return COLOR_LEFT
    if a.startswith("right") or b.startswith("right"):
        return COLOR_RIGHT
    return COLOR_TRUNK


def draw_skeleton(frame, keypoints, scores, threshold=SCORE_THRESHOLD):
    if keypoints is None:
        return frame

    for start, end in SKELETON:
        i, j = HALPE26_INDEX[start], HALPE26_INDEX[end]
        if scores[i] < threshold or scores[j] < threshold:
            continue
        p1 = (int(keypoints[i][0]), int(keypoints[i][1]))
        p2 = (int(keypoints[j][0]), int(keypoints[j][1]))
        cv2.line(frame, p1, p2, _limb_color(start, end), 2, cv2.LINE_AA)

    for index, (x, y) in enumerate(keypoints):
        confident = scores[index] >= threshold
        cv2.circle(
            frame,
            (int(x), int(y)),
            4 if confident else 2,
            COLOR_TEXT if confident else COLOR_LOW,
            -1,
            cv2.LINE_AA,
        )
    return frame


def draw_hud(frame, lines, origin=(12, 12), width=260):
    height = 20 * len(lines) + 14
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        origin,
        (origin[0] + width, origin[1] + height),
        COLOR_PANEL,
        -1,
    )
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (origin[0] + 10, origin[1] + 24 + index * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            COLOR_TEXT,
            1,
            cv2.LINE_AA,
        )
    return frame


def draw_orientation_hint(frame, text, ok):
    height, width = frame.shape[:2]
    color = (120, 220, 120) if ok else (110, 110, 230)
    cv2.rectangle(frame, (0, height - 42), (width, height), COLOR_PANEL, -1)
    cv2.putText(
        frame,
        text,
        (14, height - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )
    return frame
