"""API FastAPI — Point d'entrée REST + WebSocket.

Endpoints :
    GET  /             — Health check
    GET  /stream       — WebSocket pour streaming temps réel
    POST /attack/start — Démarrer une attaque
    POST /attack/stop  — Arrêter l'attaque
    GET  /ids/status   — État de l'IDS
    GET  /metrics      — Métriques de performance
    POST /sim/start    — Démarrer la simulation
    POST /sim/stop     — Arrêter la simulation
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from canlab.config import API_HOST, API_PORT, LOG_LEVEL
from canlab.sim.bus import CANBusManager
from canlab.attack.injector import CANInjector, AttackMode
from canlab.ids.rules import RuleBasedIDS
from canlab.ids.cusum import CUSUMIDS

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# État global de l'application
# ──────────────────────────────────────────────
class AppState:
    """État partagé de l'application."""

    def __init__(self) -> None:
        self.sim_manager = CANBusManager()
        self.injector = CANInjector()
        self.rule_ids = RuleBasedIDS()
        self.cusum_ids = CUSUMIDS()

        self.sim_running = False
        self.attack_running = False
        self.sim_task: asyncio.Task | None = None
        self.attack_task: asyncio.Task | None = None

        # Métriques
        self.metrics = {
            "frames_total": 0,
            "frames_attack": 0,
            "alerts_total": 0,
            "start_time": time.time(),
        }

        # WebSocket clients
        self.ws_clients: list[WebSocket] = []


state = AppState()


# ──────────────────────────────────────────────
# Lifespan
# ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Setup / teardown de l'application."""
    logger.info("🚀 CAN-Stealth-Attack-AI API démarrée")
    state.rule_ids.activate()
    state.cusum_ids.activate()
    yield
    # Cleanup
    if state.sim_running:
        state.sim_manager.stop_bus()
    if state.attack_running:
        state.injector.stop()
    logger.info("👋 API arrêtée")


# ──────────────────────────────────────────────
# App FastAPI
# ──────────────────────────────────────────────
app = FastAPI(
    title="CAN-Stealth-Attack-AI Lab",
    description="API pour le laboratoire de cybersécurité CAN",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Modèles Pydantic
# ──────────────────────────────────────────────
class AttackRequest(BaseModel):
    mode: str = "naive"  # "naive" ou "stealth"
    target_id: str = "0x130"
    target_speed: float = 200.0
    duration: float = 30.0


class SimRequest(BaseModel):
    duration: float = 60.0
    virtual: bool = True


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────
@app.get("/")
async def root():
    """Health check."""
    return {
        "status": "ok",
        "service": "CAN-Stealth-Attack-AI Lab",
        "uptime_s": time.time() - state.metrics["start_time"],
    }


@app.post("/sim/start")
async def start_simulation(req: SimRequest):
    """Démarre la simulation CAN."""
    if state.sim_running:
        return {"status": "already_running"}

    if req.virtual:
        import canlab.config as cfg
        cfg.USE_VIRTUAL_BUS = True
        cfg.CAN_BUSTYPE = "virtual"

    state.sim_manager = CANBusManager()

    async def _run_sim():
        state.sim_running = True
        try:
            await state.sim_manager.run_async(duration_s=req.duration)
        finally:
            state.sim_running = False

    state.sim_task = asyncio.create_task(_run_sim())
    return {"status": "started", "duration": req.duration}


@app.post("/sim/stop")
async def stop_simulation():
    """Arrête la simulation CAN."""
    state.sim_manager.stop_bus()
    state.sim_running = False
    if state.sim_task and not state.sim_task.done():
        state.sim_task.cancel()
    return {"status": "stopped", "frames_sent": state.sim_manager.frame_count}


@app.post("/attack/start")
async def start_attack(req: AttackRequest):
    """Démarre une attaque CAN."""
    if state.attack_running:
        return {"status": "already_running"}

    target_id = int(req.target_id, 16) if isinstance(req.target_id, str) else req.target_id

    async def _run_attack():
        state.attack_running = True
        try:
            result = await state.injector.run_attack(
                mode=AttackMode(req.mode),
                target_id=target_id,
                target_speed=req.target_speed,
                duration_s=req.duration,
            )
            state.metrics["frames_attack"] += result.get("frames_injected", 0)
        finally:
            state.attack_running = False

    state.attack_task = asyncio.create_task(_run_attack())
    return {
        "status": "started",
        "mode": req.mode,
        "target_id": req.target_id,
        "target_speed": req.target_speed,
    }


@app.post("/attack/stop")
async def stop_attack():
    """Arrête l'attaque en cours."""
    state.injector.stop()
    state.attack_running = False
    if state.attack_task and not state.attack_task.done():
        state.attack_task.cancel()
    return {"status": "stopped"}


@app.get("/ids/status")
async def ids_status():
    """État de tous les systèmes IDS."""
    return {
        "rules": state.rule_ids.get_summary(),
        "cusum": state.cusum_ids.get_summary(),
        "sim_running": state.sim_running,
        "attack_running": state.attack_running,
    }


@app.get("/metrics")
async def get_metrics():
    """Métriques de performance."""
    uptime = time.time() - state.metrics["start_time"]
    return {
        "uptime_s": uptime,
        "frames_total": state.sim_manager.frame_count,
        "frames_attack": state.metrics["frames_attack"],
        "alerts_total": len(state.rule_ids.alerts),
        "cusum_score": state.cusum_ids.S,
        "cusum_alerts": state.cusum_ids._alert_count,
        "rule_ids_summary": state.rule_ids.get_summary(),
    }


# ──────────────────────────────────────────────
# WebSocket — Streaming temps réel
# ──────────────────────────────────────────────
@app.websocket("/stream")
async def websocket_stream(websocket: WebSocket):
    """Stream temps réel des événements CAN + IDS."""
    await websocket.accept()
    state.ws_clients.append(websocket)
    logger.info("📡 Client WebSocket connecté (%d total)", len(state.ws_clients))

    try:
        while True:
            # Envoyer un snapshot périodique
            snapshot = {
                "type": "snapshot",
                "timestamp": time.time(),
                "sim_running": state.sim_running,
                "attack_running": state.attack_running,
                "frames_total": state.sim_manager.frame_count,
                "rule_score": state.rule_ids.get_score(),
                "cusum_score": state.cusum_ids.S,
                "alerts": len(state.rule_ids.alerts),
            }

            # Ajouter les données moteur si la simulation tourne
            if state.sim_running:
                snapshot["engine_rpm"] = state.sim_manager.engine.rpm
                snapshot["vehicle_speed"] = state.sim_manager.engine.vehicle_speed_kmh
                snapshot["steering_angle"] = state.sim_manager.steering.angle_deg
                snapshot["displayed_speed"] = state.sim_manager.cluster.displayed_speed_kmh

            await websocket.send_text(json.dumps(snapshot))
            await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        state.ws_clients.remove(websocket)
        logger.info("📡 Client WebSocket déconnecté (%d restants)", len(state.ws_clients))
    except Exception as e:
        logger.error("WebSocket error: %s", e)
        if websocket in state.ws_clients:
            state.ws_clients.remove(websocket)


# ──────────────────────────────────────────────
# Point d'entrée
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=API_HOST, port=API_PORT)
