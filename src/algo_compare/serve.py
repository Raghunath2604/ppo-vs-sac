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
from fastapi.middleware.cors import CORSMiddleware
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
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    state = {"requests": 0, "policy": policy, "meta": meta}

    @app.get("/health")
    def health() -> dict:
        m = state["meta"]
        return {"status": "ok", **m, "requests_served": state["requests"]}

    @app.post("/switch")
    def switch_model(algo: str) -> dict:
        candidates = [
            f"runs/algos_preview/{algo}/seed_1/model.pt",
            f"runs/algos_quick/{algo}/seed_1/model.pt",
            f"export/{algo}/model.pt",
        ]
        found = next((c for c in candidates if os.path.exists(c)), None)
        if not found:
            raise HTTPException(status_code=404, detail=f"model for {algo} not found")
        p, m = build_deployable(found)
        state["policy"] = p
        state["meta"] = m
        return {"status": "ok", "switched_to": algo, **m}

    @app.post("/act", response_model=ActResponse)
    def act(req: ActRequest) -> ActResponse:
        m = state["meta"]
        p = state["policy"]
        if len(req.observation) != m["obs_dim"]:
            raise HTTPException(
                status_code=422,
                detail=f"expected observation of length {m['obs_dim']}, got {len(req.observation)}",
            )
        t0 = time.perf_counter()
        with torch.no_grad():
            action = p(torch.tensor([req.observation], dtype=torch.float32))[0].tolist()
        state["requests"] += 1
        return ActResponse(action=action, latency_ms=(time.perf_counter() - t0) * 1000)

    return app
