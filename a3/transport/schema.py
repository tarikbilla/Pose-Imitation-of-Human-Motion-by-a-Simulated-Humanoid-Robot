import json

PROTOCOL = "a3-upper"
VERSION = 2

FIELDS = (
    "seq",
    "t",
    "valid",
    "left_upper_arm",
    "left_fore_arm",
    "left_hand_normal",
    "right_upper_arm",
    "right_fore_arm",
    "right_hand_normal",
    "torso_yaw",
    "torso_pitch",
    "torso_roll",
    "head_yaw",
    "head_pitch",
    "confidence",
    "hip_height",
    "stance_width",
    "left_foot_lift",
    "right_foot_lift",
    "com_offset_x",
    "com_offset_y",
    "lower_body_valid",
)

DIRECTION_FIELDS = (
    "left_upper_arm",
    "left_fore_arm",
    "left_hand_normal",
    "right_upper_arm",
    "right_fore_arm",
    "right_hand_normal",
)

SCALAR_FIELDS = (
    "torso_yaw",
    "torso_pitch",
    "torso_roll",
    "head_yaw",
    "head_pitch",
    "confidence",
)

LOWER_FIELDS = (
    "hip_height",
    "stance_width",
    "left_foot_lift",
    "right_foot_lift",
    "com_offset_x",
    "com_offset_y",
)


def empty(seq=0, t=0.0):
    packet = {
        "p": PROTOCOL,
        "v": VERSION,
        "seq": seq,
        "t": round(t, 4),
        "valid": False,
        "confidence": 0.0,
    }
    for name in DIRECTION_FIELDS:
        packet[name] = [0.0, 0.0, 0.0]
    for name in SCALAR_FIELDS:
        packet.setdefault(name, 0.0)
    for name in LOWER_FIELDS:
        packet[name] = 0.0
    packet["lower_body_valid"] = False
    return packet


def build(seq, t, directions, scalars, confidence, valid=True, lower=None):
    packet = {
        "p": PROTOCOL,
        "v": VERSION,
        "seq": seq,
        "t": round(t, 4),
        "valid": bool(valid),
        "confidence": round(float(confidence), 4),
    }
    for name in DIRECTION_FIELDS:
        vector = directions.get(name)
        packet[name] = (
            [round(float(c), 5) for c in vector] if vector is not None else [0.0, 0.0, 0.0]
        )
    for name in SCALAR_FIELDS:
        if name == "confidence":
            continue
        packet[name] = round(float(scalars.get(name, 0.0)), 5)
    for name in LOWER_FIELDS:
        packet[name] = round(float((lower or {}).get(name, 0.0)), 5)
    packet["lower_body_valid"] = bool(lower)
    return packet


def encode(packet):
    return json.dumps(packet, separators=(",", ":")).encode("utf-8")


def decode(payload):
    try:
        packet = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if packet.get("p") != PROTOCOL:
        return None
    if packet.get("v") != VERSION:
        return None
    return packet
