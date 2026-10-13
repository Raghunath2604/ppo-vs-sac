"""A small FastAPI service that serves a trained policy.

    PPO_MODEL_PATH=runs/default/model.pt uvicorn --factory algo_compare.serve:create_app --port 8000

    curl -X POST localhost:8000/act -H 'content-type: application/json' \
         -d '{"observation": [0.0, 0.0, 0.0, 0.0]}'
"""

from __future__ import annotations

import os
import time

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .export import build_deployable


class ActRequest(BaseModel):
    observation: list[float]


class ActResponse(BaseModel):
    action: list[float]
    latency_ms: float


def create_app(model_path: str | None = None) -> FastAPI:
    model_path = model_path or os.environ.get("PPO_MODEL_PATH", "runs/default/model.pt")
    policy, meta = build_deployable(model_path)
    torch.set_num_threads(1)
    app = FastAPI(title="PPO policy server", version="0.1.0")
    state = {"requests": 0}

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", **meta, "requests_served": state["requests"]}

    @app.post("/act", response_model=ActResponse)
    def act(req: ActRequest) -> ActResponse:
        if len(req.observation) != meta["obs_dim"]:
            raise HTTPException(
                status_code=422,
                detail=f"expected observation of length {meta['obs_dim']}, got {len(req.observation)}",
            )
        t0 = time.perf_counter()
        with torch.no_grad():
            action = policy(torch.tensor([req.observation], dtype=torch.float32))[0].tolist()
        state["requests"] += 1
        return ActResponse(action=action, latency_ms=(time.perf_counter() - t0) * 1000)

    return app
