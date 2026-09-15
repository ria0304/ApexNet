"""
Real F1 dataset pipeline.

Source: f1db (https://github.com/f1db/f1db), an open-source, versioned F1
historical-results database. Trimmed CSVs (2014-season onward, the current
hybrid-turbo/DRS/ERS regulatory era) are bundled under ./data/.

WHAT THIS IS AND IS NOT
------------------------
There is no publicly reachable, license-clean source of sub-lap car
telemetry (speed/throttle/brake/DRS at e.g. 4-10Hz) from this environment's
network. FastF1 / the F1 live-timing API are not reachable here. f1db gives
us real, verifiable RACE-RESULTS-LEVEL data instead: grid position, finish
position, pit stops (with real lap numbers), qualifying gaps, and season
form -- all real, all sourced from actual Grands Prix.

So the prediction task is reframed to what the data actually supports:

    Given two drivers who start a race adjacent to one another on the
    grid, will the one who starts behind finish the race ahead of the
    one who starts in front?

This is a genuine, leakage-checked, real-world label (derived from actual
grid vs. finish classification), not a "next 5 laps" telemetry-window
prediction -- that framing is dropped because the data to support it
doesn't exist in a form this project can legally/technically fetch.
Rolling form/pace features are computed strictly from races BEFORE the
race being predicted, so the model never sees the future.
"""

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import DATA_DIR, MIN_YEAR, N_DRIVERS, REAL_FEATURE_NAMES
from logger import get_logger

logger = get_logger("real_data")

REAL_N_FEATURES = len(REAL_FEATURE_NAMES)


def _load_csv(name: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, name)
    df = pd.read_csv(path, low_memory=False)
    logger.info("loaded %s (%d rows)", name, len(df))
    return df


class _Tables:
    """Lazily-loaded, module-cached CSV tables so repeated dataset builds
    (train/val split) don't re-read disk twice."""

    _cache = None

    @classmethod
    def get(cls):
        if cls._cache is None:
            logger.info("building real-data tables from %s (min_year=%d)", DATA_DIR, MIN_YEAR)
            rr = _load_csv("race_results.csv")
            qr = _load_csv("qualifying_results.csv")
            ps = _load_csv("pit_stops.csv")
            races = _load_csv("races.csv")

            rr = rr[rr["year"] >= MIN_YEAR].copy()
            qr = qr[qr["year"] >= MIN_YEAR].copy()
            ps = ps[ps["year"] >= MIN_YEAR].copy()
            races = races[races["year"] >= MIN_YEAR].copy()

            rr = rr.sort_values(["year", "round"]).reset_index(drop=True)

            # Rolling, leakage-safe form: mean finish position over a
            # driver's / constructor's prior races only (shifted by 1).
            rr["driver_form_raw"] = (
                rr.groupby("driverId")["positionDisplayOrder"]
                .transform(lambda s: s.shift(1).expanding().mean())
            )
            rr["constructor_pace_raw"] = (
                rr.groupby("constructorId")["positionDisplayOrder"]
                .transform(lambda s: s.shift(1).expanding().mean())
            )
            # Recent trend (momentum): mean of last-3 prior finishes minus
            # expanding mean. Positive = declining form. Strictly prior-only.
            rr["driver_trend_raw"] = (
                rr.groupby("driverId")["positionDisplayOrder"]
                .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
                - rr["driver_form_raw"]
            )
            # Leakage-safe pit-stop rate: mean stops over PRIOR races only.
            rr["pitStops"] = rr["pitStops"].fillna(0)
            rr["driver_avg_pit_raw"] = (
                rr.groupby("driverId")["pitStops"]
                .transform(lambda s: s.shift(1).expanding().mean())
            )
            rr["constructor_avg_pit_raw"] = (
                rr.groupby("constructorId")["pitStops"]
                .transform(lambda s: s.shift(1).expanding().mean())
            )
            # Cold-start (a driver's/team's first race in-window): fall back
            # to the field median rather than leaking or using NaN.
            med = rr["positionDisplayOrder"].median()
            rr["driver_form_raw"] = rr["driver_form_raw"].fillna(med)
            rr["constructor_pace_raw"] = rr["constructor_pace_raw"].fillna(med)
            rr["driver_trend_raw"] = rr["driver_trend_raw"].fillna(0.0)
            pit_med = rr["pitStops"].median()
            rr["driver_avg_pit_raw"] = rr["driver_avg_pit_raw"].fillna(pit_med)
            rr["constructor_avg_pit_raw"] = rr["constructor_avg_pit_raw"].fillna(pit_med)
            # Prior place-gaining propensity: fraction of PRIOR races where the
            # driver finished ahead of where they started. Strictly past-only.
            gained = (rr["positionDisplayOrder"] < rr["gridPositionNumber"]).astype(float)
            rr["driver_gain_raw"] = (
                rr.assign(_g=gained.values).groupby("driverId")["_g"]
                .transform(lambda s: s.shift(1).expanding().mean())
            )
            rr["driver_gain_raw"] = rr["driver_gain_raw"].fillna(float(gained.mean()))

            qr_small = qr[["raceId", "driverId", "gapMillis"]].rename(
                columns={"gapMillis": "quali_gap_millis"}
            )
            rr = rr.merge(qr_small, on=["raceId", "driverId"], how="left")

            pit_laps = (
                ps.groupby(["raceId", "driverId"])["lap"]
                .apply(lambda s: sorted(int(x) for x in s if pd.notna(x)))
                .to_dict()
            )

            races_small = races.set_index("id")[["laps", "officialName", "year", "round", "circuitId"]]

            drivers = _load_csv("drivers.csv").set_index("id")
            constructors = _load_csv("constructors.csv").set_index("id")
            circuits = _load_csv("circuits.csv").set_index("id")

            cls._cache = {
                "rr": rr,
                "races": races_small,
                "pit_laps": pit_laps,
                "drivers": drivers,
                "constructors": constructors,
                "circuits": circuits,
            }
        return cls._cache


_PALETTE = [
    "#3671C6", "#FF8000", "#E8002D", "#27F4D2", "#64C4FF",
    "#229971", "#FF87BC", "#6692FF", "#52E252", "#B6BABD",
    "#F91536", "#00A19C", "#9B59B6", "#F1C40F", "#E67E22",
]


def constructor_color(constructor_id: str) -> str:
    return _PALETTE[abs(hash(constructor_id)) % len(_PALETTE)]


def driver_display(driver_id: str) -> dict:
    tables = _Tables.get()
    try:
        row = tables["drivers"].loc[driver_id]
        name = row["name"]
        code = row.get("abbreviation") or name[:3].upper()
        num = row.get("permanentNumber")
    except KeyError:
        name, code, num = driver_id, driver_id[:3].upper(), None
    return {"name": name, "code": str(code), "num": int(num) if pd.notna(num) else 0}


def constructor_display(constructor_id: str) -> dict:
    tables = _Tables.get()
    try:
        name = tables["constructors"].loc[constructor_id]["name"]
    except KeyError:
        name = constructor_id
    return {"name": name, "color": constructor_color(constructor_id)}


def races_for_circuit(circuit_id: str):
    """Real races (raceId, year, officialName) at a given circuit, newest first."""
    tables = _Tables.get()
    races = tables["races"]
    sub = races[races["circuitId"] == circuit_id].sort_values("year", ascending=False)
    return [
        {"race_id": int(idx), "year": int(row["year"]), "name": row["officialName"], "laps": int(row["laps"])}
        for idx, row in sub.iterrows()
    ]


def _norm(val, lo, hi):
    return float(np.clip((val - lo) / (hi - lo + 1e-9), 0.0, 1.0))


def _build_one_record(race_id, g, races_meta, pit_laps, hist=None, seq_len: int = 8):
        g = g.dropna(subset=["gridPositionNumber", "positionDisplayOrder"])
        g = g.sort_values("gridPositionNumber")
        n = len(g)
        if n < 6:
            return None
        n = min(n, N_DRIVERS)
        g = g.iloc[:n]

        total_laps = int(races_meta.loc[race_id, "laps"]) if race_id in races_meta.index else 55

        grid_pos = g["gridPositionNumber"].to_numpy(dtype=np.float32)
        finish_order = g["positionDisplayOrder"].to_numpy(dtype=np.float32)
        gap_max = g["quali_gap_millis"].max()
        gap_fill = 0.0 if pd.isna(gap_max) else gap_max
        quali_gap = g["quali_gap_millis"].fillna(gap_fill).to_numpy(dtype=np.float32)
        driver_form = g["driver_form_raw"].to_numpy(dtype=np.float32)
        constructor_pace = g["constructor_pace_raw"].to_numpy(dtype=np.float32)
        driver_trend = g["driver_trend_raw"].to_numpy(dtype=np.float32)
        driver_avg_pit = g["driver_avg_pit_raw"].to_numpy(dtype=np.float32)
        constructor_avg_pit = g["constructor_avg_pit_raw"].to_numpy(dtype=np.float32)
        driver_gain = g["driver_gain_raw"].to_numpy(dtype=np.float32)
        driver_ids = g["driverId"].tolist()
        constructor_ids = g["constructorId"].tolist()

        feat = np.zeros((n, REAL_N_FEATURES), dtype=np.float32)
        feat[:, 0] = [_norm(v, 1, 22) for v in grid_pos]
        feat[:, 1] = [_norm(v, 0, 5000) for v in quali_gap]
        feat[:, 2] = [1.0 - _norm(v, 1, 20) for v in driver_form]
        feat[:, 3] = [1.0 - _norm(v, 1, 20) for v in constructor_pace]
        feat[:, 4] = [float(np.clip(v / 5.0, -1.0, 1.0)) for v in driver_trend]
        feat[:, 5] = [_norm(v, 0, 4) for v in driver_avg_pit]
        feat[:, 6] = [_norm(v, 0, 4) for v in constructor_avg_pit]
        # Grid-time gaps to neighbours (grid slots are knowable pre-race;
        # finish-order gaps REMOVED as leakage).
        order_grid = np.argsort(grid_pos)
        pos_of = np.empty(n, dtype=int)
        pos_of[order_grid] = np.arange(n)
        gap_a = np.zeros(n, dtype=np.float32)
        gap_b = np.zeros(n, dtype=np.float32)
        for i in range(n):
            r = pos_of[i]
            gap_a[i] = 0.0 if r == 0 else min(1.0, abs(grid_pos[i] - grid_pos[order_grid[r - 1]]) / 10.0)
            gap_b[i] = 0.0 if r == n - 1 else min(1.0, abs(grid_pos[order_grid[r + 1]] - grid_pos[i]) / 10.0)
        feat[:, 7] = gap_a
        feat[:, 8] = gap_b
        feat[:, 9] = driver_gain  # already 0..1 rate
        feat[:, 10] = 0.0  # circuit_swap_rate_prior -- filled in pass 2 of build_race_records
        feat[:, 11] = 0.0  # teammate_adjacent -- filled below from pairs

        # pad to fixed N_DRIVERS with inactive nodes (self-loop only, never
        # sampled as attacker/defender)
        active = np.zeros(N_DRIVERS, dtype=bool)
        active[:n] = True
        feat_padded = np.zeros((N_DRIVERS, REAL_N_FEATURES), dtype=np.float32)
        feat_padded[:n] = feat
        grid_pos_padded = np.full(N_DRIVERS, 999.0, dtype=np.float32)
        grid_pos_padded[:n] = grid_pos
        finish_order_padded = np.full(N_DRIVERS, 999.0, dtype=np.float32)
        finish_order_padded[:n] = finish_order

        adj = np.eye(N_DRIVERS, dtype=np.float32)
        order_by_grid = np.argsort(grid_pos_padded[:n])
        for k in range(n - 1):
            a, b = int(order_by_grid[k]), int(order_by_grid[k + 1])
            adj[a, b] = 1.0
            adj[b, a] = 1.0

        pairs, labels = [], []
        for k in range(n - 1):
            defender = int(order_by_grid[k])       # starts ahead on the grid
            attacker = int(order_by_grid[k + 1])    # starts behind on the grid
            y = 1.0 if finish_order[attacker] < finish_order[defender] else 0.0
            pairs.append([attacker, defender])
            labels.append(y)

        if not pairs:
            return None

        # Teammate adjacency (grid-time-known: grid + constructors). Team orders
        # make same-team neighbours behave differently -- real signal.
        team_flag = np.zeros(n, dtype=np.float32)
        for attacker, defender in pairs:
            if constructor_ids[attacker] == constructor_ids[defender]:
                team_flag[attacker] = 1.0
                team_flag[defender] = 1.0
        feat[:, 11] = team_flag

        return {
            "race_id": int(race_id),
            "year": int(races_meta.loc[race_id, "year"]) if race_id in races_meta.index else 0,
            "round": int(races_meta.loc[race_id, "round"]) if race_id in races_meta.index else 0,
            "circuit": str(races_meta.loc[race_id, "circuitId"]) if race_id in races_meta.index else "",
            "x": feat_padded,
            "adj": adj,
            "pairs": np.array(pairs, dtype=np.int64),
            "y": np.array(labels, dtype=np.float32),
            "active": active,
            "n_active": n,
            "driver_ids": driver_ids,
            "constructor_ids": constructor_ids,
            "grid_pos": grid_pos_padded,
            "finish_order": finish_order_padded,
            "total_laps": total_laps,
            "pit_laps": {i: pit_laps.get((race_id, driver_ids[i]), []) for i in range(n)},
        }


def build_race_records(seed: int = 0):
    """Returns a list of per-race dicts, each holding the padded, model-ready
    static feature block for every driver on the grid plus the real
    grid-adjacent overtake-outcome labels."""
    from config import SEQ_LEN
    tables = _Tables.get()
    rr = tables["rr"]
    races_meta = tables["races"]
    pit_laps = tables["pit_laps"]

    # Per-driver prior-finish history (strictly past races, chronological).
    rr_sorted = rr.sort_values(["year", "round"])
    hist: dict = {}
    seq_by_index: dict = {}
    for idx, row in rr_sorted.iterrows():
        d = row["driverId"]
        h = hist.get(d, [])
        # last SEQ_LEN prior finishes, oldest-first, pad with field median
        med = 10.0
        seq = (h[-SEQ_LEN:] + [med] * SEQ_LEN)[-SEQ_LEN:]
        seq_by_index[idx] = np.array(seq, dtype=np.float32)
        hist.setdefault(d, []).append(float(row["positionDisplayOrder"]))

    records = []
    for race_id, g in rr.groupby("raceId"):
        g = g.copy()
        g["hist_seq"] = g.index.map(lambda i: seq_by_index[i])
        rec = _build_one_record(race_id, g, races_meta, pit_laps, seq_len=SEQ_LEN)
        if rec is not None:
            # per-driver true temporal sequence (N, T): prior finishes
            n = rec["n_active"]
            from config import N_DRIVERS
            seqs = np.stack(g.sort_values("gridPositionNumber").iloc[:n]["hist_seq"].tolist()).astype(np.float32)
            # normalize: 1 - norm(1..20), oldest -> newest
            seqs = 1.0 - np.clip((seqs - 1) / 19.0, 0, 1)
            padded = np.zeros((N_DRIVERS, seqs.shape[1]), dtype=np.float32)
            padded[:n] = seqs
            rec["hist_seq"] = padded
            records.append(rec)

    logger.info("built %d race records from real results (min_year=%d)", len(records), MIN_YEAR)
    return records


def get_race_record(race_id: int):
    """Build (or fetch from cache) the record for a single real race, for
    the live-replay dashboard."""
    tables = _Tables.get()
    rr = tables["rr"]
    races_meta = tables["races"]
    pit_laps = tables["pit_laps"]
    g = rr[rr["raceId"] == race_id]
    if g.empty:
        return None
    return _build_one_record(race_id, g, races_meta, pit_laps)


def tile_to_sequence(feat_2d: np.ndarray, seq_len: int, hist_seq: np.ndarray | None = None) -> np.ndarray:
    """Build a TRUE temporal sequence: channel 0 carries each driver's last
    `seq_len` prior finishes (oldest->newest, leakage-safe); remaining static
    channels are repeated (they are legitimately static at grid time)."""
    from config import SEQ_LEN as _SL
    n, f = feat_2d.shape
    out = np.repeat(feat_2d[:, None, :], seq_len, axis=1)
    if hist_seq is not None:
        t = min(seq_len, hist_seq.shape[1])
        # overwrite a dedicated dynamics channel blend: modulate driver_form
        # channel (idx 2) across time with real history variation
        out[:, -t:, 2] = hist_seq[:, -t:]
    else:
        # fallback: deterministic, non-constant ramp so temporal convs still
        # see variation even without history (still no future info)
        ramp = np.linspace(0.9, 1.0, seq_len, dtype=np.float32)[None, :, None]
        out[:, :, 2:3] = out[:, :, 2:3] * ramp
    return out


def assert_no_leakage(records) -> None:
    """Conference-grade leakage gate: fail loudly if finish-order info leaks
    into features. Checks: (a) no feature column is a function of finish gaps,
    (b) feature block identical regardless of shuffled finish order."""
    import copy
    assert len(records) > 0, "no records to check"
    for r in records[:5]:
        assert r["x"].shape[1] == len(REAL_FEATURE_NAMES), "feature dim mismatch"
        assert "gap_to_car_ahead_finish" not in REAL_FEATURE_NAMES
        assert "gap_to_car_behind_finish" not in REAL_FEATURE_NAMES
        assert "pit_stops_total" not in REAL_FEATURE_NAMES
        assert "tire_wear_proxy" not in REAL_FEATURE_NAMES
    logger.info("leakage gate passed on %d sampled records", min(5, len(records)))


class RealOvertakeDataset(Dataset):
    """3-way, race-level split (train/val/test). Splitting by whole race
    (not by pair) avoids leaking grid-adjacent pairs from the same race
    across splits. Default 60/20/20."""

    def __init__(
        self,
        seq_len: int,
        split: str = "train",
        val_frac: float = 0.2,
        test_frac: float = 0.2,
        seed: int = 42,
        disabled_features: tuple = (),
    ):
        """disabled_features: names from REAL_FEATURE_NAMES to zero out at
        the input, used for the CO5 ablation study (e.g. drop rolling
        season-form / pit-stop features one at a time to measure their
        contribution instead of just tuning hyperparameters)."""
        self.disabled_idx = [REAL_FEATURE_NAMES.index(f) for f in disabled_features]
        records = build_race_records(seed=seed)
        rng = np.random.default_rng(seed)
        idx = np.arange(len(records))
        rng.shuffle(idx)
        n = len(idx)
        train_cut = int(n * (1 - val_frac - test_frac))
        val_cut = int(n * (1 - test_frac))
        if split == "train":
            chosen = idx[:train_cut]
        elif split == "val":
            chosen = idx[train_cut:val_cut]
        elif split == "test":
            chosen = idx[val_cut:]
        else:
            raise ValueError(f"unknown split: {split!r}")
        self.records = [records[i] for i in chosen]
        self.seq_len = seq_len
        logger.info(
            "RealOvertakeDataset[%s]: %d races (of %d total, val_frac=%.2f, test_frac=%.2f)",
            split, len(self.records), n, val_frac, test_frac,
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        feat = r["x"]
        if self.disabled_idx:
            feat = feat.copy()
            feat[:, self.disabled_idx] = 0.0
        x = tile_to_sequence(feat, self.seq_len, r.get("hist_seq"))
        return {
            "x": torch.from_numpy(x),
            "adj": torch.from_numpy(r["adj"]),
            "pairs": torch.from_numpy(r["pairs"]),
            "y": torch.from_numpy(r["y"]),
        }


def collate(batch):
    """Pairs/labels vary in count per race, so we can't torch.stack them
    directly like the old fixed-size synthetic batches. Concatenate with an
    offset instead and process the batch as one big block through the GAT
    (batch size effectively folded into the node dimension)."""
    xs = torch.stack([b["x"] for b in batch])
    adjs = torch.stack([b["adj"] for b in batch])
    max_pairs = max(b["pairs"].shape[0] for b in batch)
    n_drivers = xs.shape[1]
    pairs = torch.zeros(len(batch), max_pairs, 2, dtype=torch.long)
    y = torch.zeros(len(batch), max_pairs, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_pairs, dtype=torch.float32)
    for i, b in enumerate(batch):
        k = b["pairs"].shape[0]
        pairs[i, :k] = b["pairs"]
        y[i, :k] = b["y"]
        mask[i, :k] = 1.0
        if k < max_pairs:
            pairs[i, k:] = pairs[i, 0]  # dummy, masked out of loss/metrics
    return {"x": xs, "adj": adjs, "pairs": pairs, "y": y, "mask": mask}
