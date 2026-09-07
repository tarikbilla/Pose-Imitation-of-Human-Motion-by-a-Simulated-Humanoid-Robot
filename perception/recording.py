import json
import os
import time

import numpy as np

from .skeleton import A3_JOINTS

FORMAT_VERSION = 2
SCHEMA = "a3-29"


class Recorder:
    def __init__(self, path, metadata=None):
        self.path = path
        self.metadata = metadata or {}
        self.handle = None
        self.frames = 0
        self.started = None

    def open(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self.handle = open(self.path, "w", encoding="utf-8")
        self.started = time.perf_counter()
        header = {
            "format": "a3-keypoints",
            "version": FORMAT_VERSION,
            "schema": SCHEMA,
            "keypoint_names": list(A3_JOINTS),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        header.update(self.metadata)
        self.handle.write(json.dumps(header) + "\n")
        return self

    def write(self, keypoints, scores, timestamp=None, extra=None):
        if self.handle is None:
            raise RuntimeError("recorder not opened")
        entry = {
            "i": self.frames,
            "t": round(
                timestamp if timestamp is not None else time.perf_counter() - self.started,
                5,
            ),
        }
        if keypoints is None:
            entry["kp"] = None
            entry["sc"] = None
        else:
            entry["kp"] = [[round(float(x), 2), round(float(y), 2)] for x, y in keypoints]
            entry["sc"] = [round(float(s), 4) for s in scores]
        if extra:
            entry.update(extra)
        self.handle.write(json.dumps(entry) + "\n")
        self.frames += 1

    def close(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *args):
        self.close()


class Replay:
    def __init__(self, path, loop=False, realtime=False):
        self.path = path
        self.loop = loop
        self.realtime = realtime
        self.header = {}
        self.entries = []
        self.index = 0
        self._start_wall = None

    def load(self):
        with open(self.path, "r", encoding="utf-8") as handle:
            lines = [line for line in handle if line.strip()]
        if not lines:
            raise ValueError(f"empty recording {self.path}")
        self.header = json.loads(lines[0])
        if self.header.get("format") != "a3-keypoints":
            raise ValueError(f"not an a3 recording: {self.path}")
        self.entries = [json.loads(line) for line in lines[1:]]
        self.index = 0
        self._start_wall = None
        return self

    def __len__(self):
        return len(self.entries)

    @property
    def duration(self):
        return self.entries[-1]["t"] if self.entries else 0.0

    def read(self):
        if self.index >= len(self.entries):
            if not self.loop:
                return None
            self.index = 0
            self._start_wall = None

        entry = self.entries[self.index]
        self.index += 1

        if self.realtime:
            now = time.perf_counter()
            if self._start_wall is None:
                self._start_wall = now - entry["t"]
            else:
                remaining = (self._start_wall + entry["t"]) - now
                if remaining > 0:
                    time.sleep(remaining)

        keypoints = None
        scores = None
        if entry.get("kp") is not None:
            keypoints = np.asarray(entry["kp"], dtype=np.float32)
            scores = np.asarray(entry["sc"], dtype=np.float32)
        return entry, keypoints, scores

    def rewind(self):
        self.index = 0
        self._start_wall = None

    def describe(self):
        return {
            "kind": "replay",
            "path": self.path,
            "frames": len(self.entries),
            "duration_s": round(self.duration, 2),
            "source": self.header.get("source"),
            "frame_size": self.header.get("frame_size"),
        }

    def __enter__(self):
        return self.load()

    def __exit__(self, *args):
        pass
