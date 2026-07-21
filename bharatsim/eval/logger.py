"""Imitation-learning data logger.

Records (BEV, future-occupancy, route features, speed, command) -> expert
control at each tick to a compressed .npz shard, ready for behavior cloning /
DAgger. Pair with ExpertAutopilot to auto-generate training data.
"""
import os
import numpy as np


class DataLogger:
    def __init__(self, out_dir="dataset", shard_size=2000):
        self.dir = out_dir
        self.shard_size = shard_size
        os.makedirs(out_dir, exist_ok=True)
        self._buf = []
        self._shard = 0

    def record(self, obs, ctrl):
        self._buf.append({
            "bev": obs.bev.astype(np.uint8),
            "future_occ": obs.future_occ.astype(np.uint8),
            "speed": np.float32(obs.speed),
            "command": np.int64(obs.command),
            "heading_errors": np.array(obs.route["heading_errors"], np.float32),
            "lateral_offset": np.float32(obs.route["lateral_offset"]),
            "steer": np.float32(ctrl.steer),
            "throttle": np.float32(ctrl.throttle),
            "brake": np.float32(ctrl.brake),
        })
        if len(self._buf) >= self.shard_size:
            self.flush()

    def flush(self):
        if not self._buf:
            return
        keys = self._buf[0].keys()
        arrs = {k: np.stack([b[k] for b in self._buf]) for k in keys}
        path = os.path.join(self.dir, f"shard_{self._shard:04d}.npz")
        np.savez_compressed(path, **arrs)
        self._shard += 1
        self._buf = []
        return path
