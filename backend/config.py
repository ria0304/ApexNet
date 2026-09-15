import os

SEQ_LEN = 8  # true history length: last 8 prior races per driver (no tiling)
N_DRIVERS = 22  # padded to the largest real grid size we see (2014+ grids run 18-22 cars)
HIDDEN = 48
GAT_HEADS = 4
GAT_LAYERS = 2
DROPOUT = 0.15
LR = 1e-3
EPOCHS = 30
BATCH_SIZE = 8
SEED = 42
N_SEEDS = 3  # multi-seed reporting: [42, 43, 44]
SEEDS = [42, 43, 44]
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"

# --- Real dataset (f1db, https://github.com/f1db/f1db) ---
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MIN_YEAR = 2014  # hybrid-turbo/DRS/ERS era: consistent overtaking dynamics with today's rules

REAL_FEATURE_NAMES = [
    # All knowable at grid time. NO finish-derived gaps, NO full-race pit totals.
    "grid_position",
    "qualifying_gap_to_pole",
    "driver_season_form",       # rolling mean finish, strictly prior races
    "constructor_season_pace",  # rolling mean finish, strictly prior races
    "driver_recent_trend",      # mean(last3 prior) - expanding mean: momentum signal
    "driver_avg_pit_prior",     # mean pit-stop count over prior races only
    "constructor_avg_pit_prior",
    "grid_gap_ahead",           # grid-slot gap to car ahead (knowable at grid time)
    "grid_gap_behind",
    "driver_gain_rate_prior",   # prior fraction of races where driver gained >=1 place
    "circuit_swap_rate_prior",  # prior grid-adjacent overtake rate at this circuit
    "teammate_adjacent",        # 1 if a grid-neighbour is the driver's teammate
]


