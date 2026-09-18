import pytest
from fastapi.testclient import TestClient

import server.main as main


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "DB_PATH", str(tmp_path / "test.db"))
    return TestClient(main.app)


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "consent" in r.text.lower()


def test_create_session_requires_consent(client):
    r = client.post(
        "/api/sessions",
        json={"screen_w": 1280, "screen_h": 800, "dpr": 1.0, "consent": False},
    )
    assert r.status_code == 422
    r = client.post(
        "/api/sessions",
        json={"screen_w": 1280, "screen_h": 800, "dpr": 1.0},
    )
    assert r.status_code == 422


def test_session_and_chunk_flow(client):
    r = client.post(
        "/api/sessions",
        json={
            "screen_w": 1280,
            "screen_h": 800,
            "dpr": 1.0,
            "tasks": ["fitts"],
            "consent": True,
        },
    )
    assert r.status_code == 200
    sid = r.json()["id"]

    samples = [
        {"t": 0.0, "x": 0.0, "y": 0.0, "type": "move"},
        {"t": 100.0, "x": 10.0, "y": 5.0, "type": "move"},
        {"t": 200.0, "x": 20.0, "y": 10.0, "type": "down"},
        {"t": 300.0, "x": 20.0, "y": 10.0, "type": "up"},
    ]
    r = client.post(f"/api/sessions/{sid}/chunks", json={"samples": samples})
    assert r.status_code == 200
    assert r.json()["stored"] == 4

    # unknown session rejected
    r = client.post("/api/sessions/nope/chunks", json={"samples": samples})
    assert r.status_code == 404

    # malformed samples cleaned but not rejected
    r = client.post(
        f"/api/sessions/{sid}/chunks",
        json={"samples": [{"t": 1.0, "type": "bogus"}]},
    )
    assert r.status_code == 200
    assert r.json()["stored"] == 0

    r = client.get("/api/stats")
    assert r.json() == {"sessions": 1, "samples": 4}


def test_chunk_size_limit(client):
    r = client.post(
        "/api/sessions",
        json={"screen_w": 1280, "screen_h": 800, "dpr": 1.0, "consent": True},
    )
    sid = r.json()["id"]
    big = [{"t": float(i), "x": 0.0, "y": 0.0, "type": "move"} for i in range(5001)]
    r = client.post(f"/api/sessions/{sid}/chunks", json={"samples": big})
    assert r.status_code == 422
