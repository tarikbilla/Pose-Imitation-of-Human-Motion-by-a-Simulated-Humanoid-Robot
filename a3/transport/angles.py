import json

PROTOCOL = "a3-angles"
VERSION = 1


def build(seq, timestamp, angles, forward=0.0, lateral=0.0, valid=True,
          confidence=0.0):
    return {
        "p": PROTOCOL,
        "v": VERSION,
        "seq": int(seq),
        "t": round(float(timestamp), 4),
        "valid": bool(valid),
        "confidence": round(float(confidence), 4),
        "forward": round(float(forward), 5),
        "lateral": round(float(lateral), 5),
        "a": {name: round(float(value), 5) for name, value in angles.items()},
    }


def empty(seq=0, timestamp=0.0):
    return build(seq, timestamp, {}, valid=False)


def encode(packet):
    return json.dumps(packet, separators=(",", ":")).encode("utf-8")


def decode(payload):
    try:
        packet = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if packet.get("p") != PROTOCOL or packet.get("v") != VERSION:
        return None
    return packet
