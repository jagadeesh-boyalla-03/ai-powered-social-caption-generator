"""
Unit and integration tests for Authentication, Hashing, Sessions, and User Management.
"""

import pytest
from datetime import datetime, timedelta
from api.index import (
    hash_password,
    verify_password,
    create_session,
    get_user_from_token,
    get_db,
)


def test_password_hashing_and_verification():
    """Verifies that PBKDF2 hashing generates unique salts and validates accurately."""
    pwd = "MySuperSecretPassword@2026"
    hash1 = hash_password(pwd)
    hash2 = hash_password(pwd)

    # Salts must be unique
    assert hash1 != hash2
    assert ":" in hash1

    # Verification must succeed for the right password
    assert verify_password(pwd, hash1) is True
    assert verify_password(pwd, hash2) is True

    # Verification must fail for incorrect passwords
    assert verify_password("WrongPassword123", hash1) is False
    assert verify_password("", hash1) is False
    assert verify_password(pwd, "invalid_hash_format") is False


def test_register_user_success(client):
    """Tests successful user registration returning user data and active session token."""
    payload = {
        "name": "Alice Developer",
        "email": "alice@company.org",
        "password": "StrongPassword789"
    }
    response = client.post("/api/auth/register", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["token"]
    assert data["user"]["email"] == "alice@company.org"
    assert data["user"]["name"] == "Alice Developer"
    assert "id" in data["user"]


@pytest.mark.parametrize("invalid_payload,expected_err", [
    ({"name": "A", "email": "valid@email.com", "password": "pass1234"}, "minimum 2 characters"),
    ({"name": "Alice", "email": "invalid-email", "password": "pass1234"}, "valid email address"),
    ({"name": "Alice", "email": "valid@email.com", "password": "123"}, "at least 6 characters"),
])
def test_register_validation_errors(client, invalid_payload, expected_err):
    """Ensures input validation rejects invalid names, malformed emails, and short passwords."""
    response = client.post("/api/auth/register", json=invalid_payload)
    assert response.status_code == 400
    assert expected_err in response.json()["detail"]


def test_register_duplicate_email_rejected(client):
    """Verifies that duplicate registration attempts for the same email are rejected."""
    payload = {
        "name": "Bob Smith",
        "email": "bob@example.com",
        "password": "Password123!"
    }
    # First registration should succeed
    res1 = client.post("/api/auth/register", json=payload)
    assert res1.status_code == 200

    # Duplicate registration (even with different casing) should fail with 400
    payload_upper = {
        "name": "Bob Smith Duplicate",
        "email": "BOB@EXAMPLE.COM",
        "password": "Password456!"
    }
    res2 = client.post("/api/auth/register", json=payload_upper)
    assert res2.status_code == 400
    assert "already exists" in res2.json()["detail"]


def test_login_user_success(client, registered_user):
    """Verifies that registered users can log in successfully."""
    login_payload = {
        "email": registered_user["email"],
        "password": registered_user["password"]
    }
    response = client.post("/api/auth/login", json=login_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["token"]
    assert data["user"]["email"] == registered_user["email"]


def test_login_user_invalid_credentials(client, registered_user):
    """Verifies that wrong passwords and unregistered emails receive a 401 Unauthorized."""
    # Wrong password
    res_wrong_pw = client.post("/api/auth/login", json={
        "email": registered_user["email"],
        "password": "IncorrectPassword"
    })
    assert res_wrong_pw.status_code == 401

    # Nonexistent user
    res_no_user = client.post("/api/auth/login", json={
        "email": "nonexistent@example.com",
        "password": "SomePassword123"
    })
    assert res_no_user.status_code == 401


def test_get_current_user_profile(client, registered_user):
    """Tests the /api/auth/me profile endpoint with valid and invalid Authorization headers."""
    # Authorized request
    headers = {"Authorization": f"Bearer {registered_user['token']}"}
    response = client.get("/api/auth/me", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["authenticated"] is True
    assert data["user"]["email"] == registered_user["email"]
    assert "stats" in data
    assert data["stats"]["total_captions_generated"] >= 0

    # Unauthorized request (no token)
    res_unauth = client.get("/api/auth/me")
    assert res_unauth.status_code == 401

    # Invalid token
    res_invalid = client.get("/api/auth/me", headers={"Authorization": "Bearer fake-token-12345"})
    assert res_invalid.status_code == 401


def test_logout_user(client, registered_user):
    """Verifies that logout invalidates the session token."""
    token = registered_user["token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Verify session works prior to logout
    res_before = client.get("/api/auth/me", headers=headers)
    assert res_before.status_code == 200

    # Logout
    res_logout = client.post("/api/auth/logout", headers=headers)
    assert res_logout.status_code == 200
    assert res_logout.json()["success"] is True

    # Verify session is no longer active
    res_after = client.get("/api/auth/me", headers=headers)
    assert res_after.status_code == 401


def test_expired_session_handling(registered_user):
    """Verifies that expired sessions are automatically purged."""
    user_id = registered_user["id"]
    token = create_session(user_id)

    # Manually update the expiration date to the past in SQLite
    past_date = (datetime.utcnow() - timedelta(days=1)).isoformat()
    with get_db() as conn:
        conn.execute("UPDATE sessions SET expires_at = ? WHERE token = ?", (past_date, token))
        conn.commit()

    # Attempting to fetch user should return None and delete token
    user = get_user_from_token(f"Bearer {token}")
    assert user is None

    # Check that the token was removed from the database
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sessions WHERE token = ?", (token,))
        row = cursor.fetchone()
        assert row is None
