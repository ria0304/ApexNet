#!/bin/bash
export PYTHONPATH="/workspace/backend:${PYTHONPATH}"
cd /workspace
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
