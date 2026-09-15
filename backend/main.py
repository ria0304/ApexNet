import json
import os
import time
from typing import Optional

import torch
from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import sys

sys.path.insert(0, os.path.dirname(__file__))

from config import CHECKPOINT_DIR, SEQ_LEN, HIDDEN
from logger import get_logger
from model import build_model
from race import TRACKS, RaceSim
from real_data import REAL_N_FEATURES

log = get_logger("main")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CKPT_V2 = os.path.join(ROOT, CHECKPOINT_DIR, "cnn_gat_v2.pt")
CKPT = os.path.join(ROOT, CHECKPOINT_DIR, "cnn_gat.pt")
METRICS = os.path.join(ROOT, CHECKPOINT_DIR, "metrics.json")
STATIC = os.path.join(ROOT, "frontend")

device = torch.device("cpu")
_MODEL_NAME = "cnn_gat_v2" if os.path.exists(CKPT_V2) else "cnn_gat"
_CKPT = CKPT_V2 if os.path.exists(CKPT_V2) else CKPT
model = build_model(_MODEL_NAME, REAL_N_FEATURES, SEQ_LEN, HIDDEN)
if os.path.exists(_CKPT):
    blob = torch.load(_CKPT, map_location=device, weights_only=False)
    try:
        model.load_state_dict(blob["state_dict"])
    except RuntimeError:
        log.warning("checkpoint %s incompatible with %s (feature/model change) -- serving UNTRAINED %s. Re-run train.py.", _CKPT, _MODEL_NAME, _MODEL_NAME)
    log.info("loaded checkpoint %s (val metrics: %s)", _CKPT, blob.get("metrics"))
else:
    log.warning("no checkpoint found at %s -- serving an UNTRAINED model. Run `python train.py` first.", _CKPT)
model.eval()

sim = RaceSim(model, device, track_id="spa", seed=21)
log.info("RaceSim initialized on track=%s", sim.track["id"])

app = FastAPI(title="F1 Overtaking Prediction CNN+GAT (real f1db data)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    log.info(
        "%s %s -> %d (%.1fms)",
        request.method, request.url.path, response.status_code, (time.time() - t0) * 1000,
    )
    return response


class ResetBody(BaseModel):
    track_id: Optional[str] = "spa"
    seed: Optional[int] = 21


@app.get("/api/health")
def health():
    return {"ok": True, "model": "cnn_gat", "checkpoint": os.path.exists(CKPT)}


@app.get("/api/tracks")
def tracks():
    return TRACKS


@app.get("/api/metrics")
def metrics():
    if os.path.exists(METRICS):
        with open(METRICS) as f:
            return json.load(f)
    return {}


@app.get("/api/state")
def state():
    return sim.state()


@app.post("/api/reset")
def reset(body: ResetBody):
    log.info("reset requested: track_id=%s seed=%s", body.track_id, body.seed)
    st = sim.reset(track_id=body.track_id, seed=body.seed)
    log.info("reset -> replaying real race %r (%s)", st.get("race_name"), st.get("race_year"))
    return st


@app.post("/api/step")
def step():
    st = sim.step()
    if st.get("last_overtakes"):
        log.info("lap %d: %d real position change(s)", st["lap"], len(st["last_overtakes"]))
    return st


@app.post("/api/run")
def run(laps: int = Query(5, ge=1, le=20)):
    log.info("run requested: laps=%d", laps)
    last = None
    for _ in range(laps):
        last = sim.step()
        if last.get("finished"):
            log.info("race finished at lap %d", last["lap"])
            break
    return last


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


if os.path.isdir(STATIC):
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
