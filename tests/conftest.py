"""
Pytest configuration and fixtures for testing Social Caption API and authentication.
"""

import os
import io
import sys
import tempfile
import pytest
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi.testclient import TestClient

@pytest.fixture(autouse=True)
def isolated_test_db(tmp_path, monkeypatch):
    """Sets up an isolated SQLite database for each test run."""
    test_db_path = str(tmp_path / "test_auth.db")
    monkeypatch.setenv("DATABASE_PATH", test_db_path)
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyTestMockKeyForPytestRunning12345")
    
    # Re-import / initialize database tables
    from api import index
    index.init_db()
    yield test_db_path

@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    from api.index import app
    return TestClient(app)

@pytest.fixture
def sample_image_bytes():
    """Generates a small RGB test image in memory as JPEG bytes."""
    img = Image.new("RGB", (100, 100), color=(73, 109, 137))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.getvalue()

@pytest.fixture
def registered_user(client):
    """Registers a sample user and returns their credentials, response and token."""
    user_data = {
        "name": "Jane Doe",
        "email": "jane.doe@example.com",
        "password": "SecurePassword123!"
    }
    resp = client.post("/api/auth/register", json=user_data)
    data = resp.json()
    return {
        **user_data,
        "id": data["user"]["id"],
        "token": data["token"]
    }
