"""
FastAPI route and integration tests for API endpoints, healthcheck, SPA serving, and history management.
"""

import json
from unittest.mock import patch, MagicMock
import pytest


def test_health_endpoints(client):
    """Tests /health and /api/health monitoring endpoints."""
    for path in ["/health", "/api/health"]:
        response = client.get(path)
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


def test_spa_frontend_routes(client):
    """Verifies that Single Page Application routes return 200 and HTML content."""
    for route in ["/", "/login", "/signup", "/history"]:
        response = client.get(route)
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "<html" in response.text.lower() or "<!doctype html" in response.text.lower()


def test_history_endpoints_unauthorized(client):
    """Ensures unauthenticated users cannot access or delete caption history."""
    res_get = client.get("/api/history")
    assert res_get.status_code == 401

    res_del = client.delete("/api/history/1")
    assert res_del.status_code == 401


def test_history_crud_flow(client, registered_user):
    """Tests getting empty history, inserting history via caption generation, and deleting an item."""
    token = registered_user["token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 1. Initial history should be empty
    res_empty = client.get("/api/history", headers=headers)
    assert res_empty.status_code == 200
    assert res_empty.json()["history"] == []


@patch("api.index.genai.Client")
def test_generate_captions_success_with_mocked_gemini(mock_genai_client, client, sample_image_bytes, registered_user):
    """Tests end-to-end /api/generate pipeline with a mocked Gemini Vision and LLM client."""
    # Setup mock Gemini client
    mock_instance = MagicMock()
    mock_genai_client.return_value = mock_instance

    # Mock Vision response
    vision_resp = MagicMock()
    vision_resp.text = "A serene sunset over a mountain lake with pine trees."

    # Mock Structured Caption response
    caption_resp = MagicMock()
    mock_data = {
        "instagram": "Chasing golden hour in the mountains. ✨ #Sunset #Nature #Travel #MountainLife #Photography",
        "x": "Mountain views never get old. 🌄 #Nature #Travel"
    }
    caption_resp.parsed = mock_data
    caption_resp.text = json.dumps(mock_data)

    # First call is vision, second call is caption generation
    mock_instance.models.generate_content.side_effect = [vision_resp, caption_resp]

    # Perform POST request to /api/generate
    form_data = {
        "platforms": "instagram,x",
        "tone": "Inspirational",
        "context": "Sunset hike adventure",
        "api_key": "AIzaSyDummyKeyForTestingAPIWithPytest12345"
    }
    files = {
        "file": ("test.jpg", sample_image_bytes, "image/jpeg")
    }
    headers = {"Authorization": f"Bearer {registered_user['token']}"}

    response = client.post("/api/generate", data=form_data, files=files, headers=headers)
    assert response.status_code == 200
    data = response.json()

    assert data["description"] == "A serene sunset over a mountain lake with pine trees."
    assert "instagram" in data["captions"]
    assert "x" in data["captions"]
    assert data["saved_to_history"] is True
    assert data["user"] == registered_user["name"]

    # Verify history now contains the generated caption
    hist_resp = client.get("/api/history", headers=headers)
    assert hist_resp.status_code == 200
    history_items = hist_resp.json()["history"]
    assert len(history_items) == 1
    assert history_items[0]["description"] == "A serene sunset over a mountain lake with pine trees."

    # Test deleting that history item
    item_id = history_items[0]["id"]
    del_resp = client.delete(f"/api/history/{item_id}", headers=headers)
    assert del_resp.status_code == 200
    assert del_resp.json()["success"] is True

    # Confirm history is now empty again
    hist_empty_resp = client.get("/api/history", headers=headers)
    assert len(hist_empty_resp.json()["history"]) == 0


def test_generate_captions_missing_api_key(client, sample_image_bytes, monkeypatch):
    """Ensures 400 Bad Request is returned when no API key is provided and none in environment."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    form_data = {
        "platforms": "instagram",
        "tone": "Casual",
        "api_key": ""
    }
    files = {
        "file": ("test.jpg", sample_image_bytes, "image/jpeg")
    }
    response = client.post("/api/generate", data=form_data, files=files)
    assert response.status_code == 400
    assert "API key is missing" in response.json()["detail"]


def test_generate_captions_invalid_api_key(client, sample_image_bytes, monkeypatch):
    """Ensures 400 Bad Request is returned when an invalid short API key is supplied."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    form_data = {
        "platforms": "instagram",
        "tone": "Casual",
        "api_key": "short_key_invalid"
    }
    files = {
        "file": ("test.jpg", sample_image_bytes, "image/jpeg")
    }
    response = client.post("/api/generate", data=form_data, files=files)
    assert response.status_code == 400
    assert "Invalid API key format" in response.json()["detail"]
