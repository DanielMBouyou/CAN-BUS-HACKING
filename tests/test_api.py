"""Tests pour l'API FastAPI."""

import pytest
from httpx import ASGITransport, AsyncClient

from canlab.api.main import app


@pytest.fixture
async def client():
    """Client HTTP async pour tester l'API."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_root(client):
    """Test health check."""
    resp = await client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "uptime_s" in data


@pytest.mark.asyncio
async def test_ids_status(client):
    """Test endpoint IDS status."""
    resp = await client.get("/ids/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "rules" in data
    assert "cusum" in data


@pytest.mark.asyncio
async def test_metrics(client):
    """Test endpoint métriques."""
    resp = await client.get("/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "uptime_s" in data
    assert "frames_total" in data


@pytest.mark.asyncio
async def test_attack_start_stop(client):
    """Test démarrer et arrêter une attaque."""
    # Start
    resp = await client.post(
        "/attack/start",
        json={
            "mode": "naive",
            "target_id": "0x130",
            "target_speed": 200.0,
            "duration": 2.0,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "started"

    # Stop
    resp = await client.post("/attack/stop")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_sim_start_stop(client):
    """Test démarrer et arrêter la simulation."""
    resp = await client.post(
        "/sim/start",
        json={"duration": 2.0, "virtual": True},
    )
    assert resp.status_code == 200

    resp = await client.post("/sim/stop")
    assert resp.status_code == 200
