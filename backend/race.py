"""
Live "race replay" dashboard backend.

IMPORTANT — what is real here and what is a visualization aid:
  REAL:  which race is being replayed, every driver's real grid slot, real
         final classification, real pit-stop lap numbers, real season form
         going into that race. These come straight out of real_data.py /
         the bundled f1db CSVs.
  INTERPOLATED (disclosed, not measured): there is no public source of
         lap-by-lap car position this project can reach (see
         real_data.py's module docstring), so the on-screen car positions
         between the real start (grid) and real end (final classification)
         are produced by a smooth interpolation with a bit of deterministic
         "jitter" so the animation doesn't look like a boring straight
         line. It is a visualization convenience, not a telemetry
         reconstruction, and the position graphic should not be read as
         "this is where the car really was on lap 14."
  MODEL PREDICTIONS: the probabilities shown are the actual trained
         CNN+GAT model's output on that race's real pre-race features
         (grid, quali gap, season form, pace, pit count), evaluated against
         whichever two cars are currently adjacent in the (interpolated)
         running order.
"""

import numpy as np
import torch

from config import N_DRIVERS, SEQ_LEN
from logger import get_logger
import real_data

logger = get_logger("race")

TRACKS = [
    {"id": "monaco", "name": "Monaco", "circuit_id": "monaco", "length_km": 3.337, "corners": 19},
    {"id": "spa", "name": "Spa-Francorchamps", "circuit_id": "spa-francorchamps", "length_km": 7.004, "corners": 19},
    {"id": "silverstone", "name": "Silverstone", "circuit_id": "silverstone", "length_km": 5.891, "corners": 18},
    {"id": "monza", "name": "Monza", "circuit_id": "monza", "length_km": 5.793, "corners": 11},
    {"id": "singapore", "name": "Singapore", "circuit_id": "marina-bay", "length_km": 4.940, "corners": 19},
]


class RaceSim:
    """Replays one real historical race per (track, seed) pick, scoring
    currently-adjacent cars with the model trained in train.py."""

    def __init__(self, model, device, track_id="spa", seed=21):
        self.model = model
        self.device = device
        self.track = next((t for t in TRACKS if t["id"] == track_id), TRACKS[1])
        self.rng = np.random.default_rng(seed)
        self.reset()

    def _pick_race(self, seed):
        races = [r for r in real_data.races_for_circuit(self.track["circuit_id"]) if r["laps"] >= 10]
        if not races:
            raise RuntimeError(f"No usable real races found for circuit {self.track['circuit_id']!r}")
        rng = np.random.default_rng(seed)
        pool = races[: min(len(races), 6)]  # bias toward recent seasons
        while pool:
            choice = pool.pop(int(rng.integers(0, len(pool))))
            rec = real_data.get_race_record(choice["race_id"])
            if rec is not None:
                self.race_meta = choice
                return rec
        raise RuntimeError("No usable real race record for this circuit")

    def reset(self, track_id=None, seed=None):
        if track_id:
            self.track = next((t for t in TRACKS if t["id"] == track_id), self.track)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.rec = self._pick_race(int(self.rng.integers(0, 1_000_000)))
        logger.info(
            "loaded real race %s at %s: %d drivers, %d laps",
            self.race_meta, self.track["circuit_id"], self.rec["n_active"], self.rec["total_laps"],
        )
        n = self.rec["n_active"]
        self.n = n
        self.driver_meta = [real_data.driver_display(did) for did in self.rec["driver_ids"]]
        self.constructor_meta = [real_data.constructor_display(cid) for cid in self.rec["constructor_ids"]]

        grid_pos = self.rec["grid_pos"][:n]
        finish_pos = self.rec["finish_order"][:n]
        self.grid_rank = np.argsort(np.argsort(grid_pos))          # 0 = pole
        self.finish_rank = np.argsort(np.argsort(finish_pos))      # 0 = winner
        self.phase = self.rng.uniform(0, 2 * np.pi, n)
        self.jitter_amp = self.rng.uniform(0.15, 0.55, n)
        pit_counts = {i: len(v) for i, v in self.rec["pit_laps"].items()}
        self.total_pit_stops = np.array([max(pit_counts.get(i, 0), 1) for i in range(n)], dtype=np.float32)

        self.total_laps = self.rec["total_laps"]
        self.lap = 1
        self.events = []
        self.history = []
        self.overtakes = []
        self.static_feat = self.rec["x"][:n].copy()
        self._record_standings()
        return self.state()

    def _current_rank(self, lap):
        """Interpolated running order at a given lap: real grid -> real
        finish, plus deterministic jitter so mid-race lead changes are
        visible. See module docstring: this is an animation aid, not a
        telemetry reconstruction."""
        t = np.clip(lap / max(self.total_laps, 1), 0, 1)
        base = self.grid_rank + (self.finish_rank - self.grid_rank) * t
        wobble = self.jitter_amp * np.sin(2 * np.pi * (2.5 * t) + self.phase) * (1 - t) * 1.2
        posval = base + wobble
        return np.argsort(posval)  # order[0] = current leader's index

    def _dynamic_features(self, lap):
        """Leakage-safe: all model inputs are grid-time-known; per-lap pit
        info is used for DISPLAY only, never written into model features."""
        return self.static_feat.copy()

    def _build_adj_and_pairs(self, order):
        n = N_DRIVERS
        adj = np.eye(n, dtype=np.float32)
        pairs = []
        for k in range(len(order) - 1):
            a, b = int(order[k]), int(order[k + 1])
            adj[a, b] = 1.0
            adj[b, a] = 1.0
            pairs.append([b, a])  # b (behind) attacking a (ahead)
        return adj, np.array(pairs, dtype=np.int64)

    def predict(self):
        order = self._current_rank(self.lap)  # index 0 = leader ... index n-1 = last
        feat = self._dynamic_features(self.lap)
        adj, pairs = self._build_adj_and_pairs(order)

        x_full = np.zeros((N_DRIVERS, SEQ_LEN, feat.shape[-1]), dtype=np.float32)
        x_full[: self.n] = real_data.tile_to_sequence(feat, SEQ_LEN, self.rec.get("hist_seq"))

        xt = torch.from_numpy(x_full).unsqueeze(0).to(self.device)
        adjt = torch.from_numpy(adj).unsqueeze(0).to(self.device)
        pairst = torch.from_numpy(pairs).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logit = self.model(xt, adjt, pairst)
            probs = torch.sigmoid(logit).cpu().numpy().ravel()

        preds = []
        for k, (att, dfd) in enumerate(pairs):
            att, dfd = int(att), int(dfd)
            p = float(np.clip(probs[k], 0.01, 0.99))
            preds.append(
                {
                    "attacker": att,
                    "defender": dfd,
                    "attacker_code": self.driver_meta[att]["code"],
                    "defender_code": self.driver_meta[dfd]["code"],
                    "probability": round(p * 100, 1),
                    "will_overtake": p >= 0.5,
                    "horizon": "by race end (pre-race model, real f1db features)",
                }
            )
        preds.sort(key=lambda d: -d["probability"])
        self._last_order = order
        self._last_feat = feat
        return preds

    def step(self):
        if self.lap >= self.total_laps:
            return self.state()
        prev_order = list(self._current_rank(self.lap))
        preds = self.predict()
        self.lap += 1
        new_order = list(self._current_rank(self.lap))

        happened = []
        for pos_now, idx in enumerate(new_order):
            pos_prev = prev_order.index(idx)
            if pos_now < pos_prev:  # actually gained places (matches the real eventual outcome)
                behind_idx = prev_order[pos_prev - 1] if pos_prev > 0 else None
                if behind_idx is not None:
                    match = next((p for p in preds if p["attacker"] == idx and p["defender"] == behind_idx), None)
                    evt = {
                        "lap": self.lap,
                        "attacker": self.driver_meta[idx]["code"],
                        "defender": self.driver_meta[behind_idx]["code"],
                        "probability": match["probability"] if match else None,
                    }
                    happened.append(evt)
                    self.overtakes.append(evt)
                    prob_txt = f" (model said {evt['probability']:.0f}%)" if evt["probability"] is not None else ""
                    self.events.append(
                        f"Lap {self.lap}: {evt['attacker']} moves ahead of {evt['defender']}{prob_txt}"
                    )
        for i in range(self.n):
            if self.lap in self.rec["pit_laps"].get(i, []):
                self.events.append(f"Lap {self.lap}: {self.driver_meta[i]['code']} pits (real pit stop)")

        self._record_standings()
        st = self.state()
        st["last_overtakes"] = happened
        return st

    def _record_standings(self):
        order = self._current_rank(self.lap)
        self.history.append({"lap": self.lap, "order": [self.driver_meta[i]["code"] for i in order]})

    def state(self):
        preds = self.predict()
        order = [int(i) for i in self._last_order]
        feat = self._last_feat
        field = []
        for pos, idx in enumerate(order, start=1):
            dm = self.driver_meta[idx]
            cm = self.constructor_meta[idx]
            t = np.clip(self.lap / max(self.total_laps, 1), 0, 1)
            track_pos = ((self.n - pos) / self.n + 0.15 * t) % 1.0
            field.append(
                {
                    "pos": pos,
                    "idx": idx,
                    "code": dm["code"],
                    "name": dm["name"],
                    "team": cm["name"],
                    "color": cm["color"],
                    "num": dm["num"],
                    "track_pos": float(track_pos),
                    "tire": 0.0,
                    "pit_progress": 0.0,
                    "grid_position": int(self.grid_rank[idx]) + 1,
                    "real_final_position": int(self.finish_rank[idx]) + 1,
                }
            )
        graph = {
            "nodes": [
                {"id": i, "code": self.driver_meta[i]["code"], "color": self.constructor_meta[i]["color"]}
                for i in range(self.n)
            ],
            "edges": [{"source": int(order[k]), "target": int(order[k + 1])} for k in range(len(order) - 1)],
        }
        return {
            "lap": self.lap,
            "total_laps": self.total_laps,
            "track": self.track,
            "race_name": self.race_meta["name"],
            "race_year": self.race_meta["year"],
            "field": field,
            "predictions": preds[:12],
            "events": self.events[-12:],
            "overtakes": self.overtakes[-20:],
            "graph": graph,
            "finished": self.lap >= self.total_laps,
        }
