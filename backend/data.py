import numpy as np
import torch
from torch.utils.data import Dataset

from config import DRIVERS, N_DRIVERS, N_FEATURES, SEQ_LEN


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def generate_snapshot(rng: np.random.Generator):
    n, t, f = N_DRIVERS, SEQ_LEN, N_FEATURES
    order = rng.permutation(n)
    pace = np.array([DRIVERS[i]["pace"] for i in range(n)], dtype=np.float32)
    pace = pace + rng.normal(0, 0.02, n).astype(np.float32)
    tire = rng.uniform(0.15, 0.92, n).astype(np.float32)
    ers = rng.uniform(0.1, 1.0, n).astype(np.float32)
    base_gap = np.abs(rng.normal(0.55, 0.35, n)).astype(np.float32)
    base_gap = np.clip(base_gap, 0.04, 2.4)
    track_pos = np.linspace(0, 0.92, n, endpoint=False)[np.argsort(np.argsort(order))]
    track_pos = (track_pos + rng.uniform(0, 0.03, n)) % 1.0

    x = np.zeros((n, t, f), dtype=np.float32)
    corner_wave = 0.5 + 0.5 * np.sin(np.linspace(0, 3 * np.pi, t))

    for i in range(n):
        speed = np.clip(pace[i] + rng.normal(0, 0.03, t) - 0.18 * corner_wave, 0.35, 1.0)
        throttle = np.clip(0.55 + 0.4 * (1 - corner_wave) + rng.normal(0, 0.05, t), 0, 1)
        brake = np.clip(0.65 * corner_wave + rng.normal(0, 0.04, t), 0, 1)
        steering = np.clip(corner_wave * rng.choice([-1, 1]) + rng.normal(0, 0.08, t), -1, 1)
        drs = np.where(corner_wave < 0.35, rng.random(t) < 0.55, 0.0).astype(np.float32)
        gap_ahead = np.clip(base_gap[i] + rng.normal(0, 0.05, t), 0.02, 3.0) / 3.0
        gap_behind = np.clip(base_gap[(i + 1) % n] + rng.normal(0, 0.05, t), 0.02, 3.0) / 3.0
        x[i, :, 0] = speed
        x[i, :, 1] = throttle
        x[i, :, 2] = brake
        x[i, :, 3] = np.abs(steering)
        x[i, :, 4] = drs
        x[i, :, 5] = tire[i]
        x[i, :, 6] = gap_ahead
        x[i, :, 7] = gap_behind
        x[i, :, 8] = ers[i]
        x[i, :, 9] = ((track_pos[i] + np.linspace(0, 0.04, t)) % 1.0)
        x[i, :, 10] = corner_wave
        x[i, :, 11] = np.clip((pace[i] - 0.88) / 0.15 + 0.5, 0, 1)

    pos = np.argsort(-track_pos)
    adj = np.eye(n, dtype=np.float32)
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            circ = min(abs(track_pos[a] - track_pos[b]), 1 - abs(track_pos[a] - track_pos[b]))
            if circ < 0.08:
                adj[a, b] = 1.0
    for k in range(n - 1):
        adj[pos[k], pos[k + 1]] = 1.0
        adj[pos[k + 1], pos[k]] = 1.0

    pairs = []
    labels = []
    pack = adj.sum(axis=1)
    for k in range(n - 1):
        defender = int(pos[k])
        attacker = int(pos[k + 1])
        gap = float(x[attacker, -1, 6]) * 3.0
        pace_delta = pace[attacker] - pace[defender]
        tire_delta = tire[defender] - tire[attacker]
        drs_a = float(x[attacker, -8:, 4].mean())
        drs_d = float(x[defender, -8:, 4].mean())
        corner = float(x[attacker, -1, 10])
        neighbors_a = int(pack[attacker])
        neighbors_d = int(pack[defender])
        jammed = 1.0 if neighbors_a + neighbors_d >= 8 else 0.0
        train = 0.0
        if k + 2 < n:
            train = float(x[int(pos[k + 2]), -8:, 4].mean())
        blocked_ahead = 0.0
        if k > 0:
            gap2 = float(x[defender, -1, 6]) * 3.0
            blocked_ahead = 1.0 if gap2 < 0.35 else 0.0
        ers_a = ers[attacker]
        z = (
            1.55 * pace_delta * 14
            + 1.35 * (0.48 - gap)
            + 1.05 * tire_delta
            + 0.85 * (drs_a - 0.6 * drs_d)
            + 0.55 * ers_a
            + 0.40 * train
            - 1.35 * corner
            - 0.95 * jammed
            - 0.85 * blocked_ahead
            - 0.12 * neighbors_a
            + rng.normal(0, 0.12)
        )
        y = 1 if rng.random() < _sigmoid(z) else 0
        pairs.append([attacker, defender])
        labels.append(y)

    return {
        "x": x,
        "adj": adj,
        "pairs": np.array(pairs, dtype=np.int64),
        "y": np.array(labels, dtype=np.float32),
        "order": pos.astype(np.int64),
        "track_pos": track_pos.astype(np.float32),
        "pace": pace,
        "tire": tire,
    }


class OvertakeDataset(Dataset):
    def __init__(self, n_samples: int, seed: int):
        rng = np.random.default_rng(seed)
        self.samples = [generate_snapshot(rng) for _ in range(n_samples)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "x": torch.from_numpy(s["x"]),
            "adj": torch.from_numpy(s["adj"]),
            "pairs": torch.from_numpy(s["pairs"]),
            "y": torch.from_numpy(s["y"]),
        }


def collate(batch):
    return {
        "x": torch.stack([b["x"] for b in batch]),
        "adj": torch.stack([b["adj"] for b in batch]),
        "pairs": torch.stack([b["pairs"] for b in batch]),
        "y": torch.stack([b["y"] for b in batch]),
    }
