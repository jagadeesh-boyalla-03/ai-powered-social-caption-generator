"""
Unit tests for social media caption generation logic, clamping, fallbacks, and retry mechanism.
"""

from unittest.mock import MagicMock
import pytest
from api.index import (
    looks_like_gemini_api_key,
    clamp_caption,
    build_fallback_caption,
    generate_gemini_content_with_retry,
    PLATFORM_RULES,
    GEMINI_MODEL_FALLBACKS,
)


def test_looks_like_gemini_api_key():
    """Validates the API key format checking function."""
    assert looks_like_gemini_api_key("AIzaSyD-abc1234567890123456") is True
    assert looks_like_gemini_api_key("12345678901234567890") is True
    assert looks_like_gemini_api_key("short_key") is False
    assert looks_like_gemini_api_key("") is False
    assert looks_like_gemini_api_key(None) is False


def test_clamp_caption():
    """Verifies that clamp_caption correctly truncates text over the limit and preserves under-limit text."""
    short_text = "This is a short caption #ai"
    assert clamp_caption(short_text, 100) == short_text

    long_text = "A" * 300
    clamped = clamp_caption(long_text, 280)
    assert len(clamped) <= 280
    assert clamped.endswith("...")

    # Multiple blank lines should be collapsed to at most two newlines
    whitespace_text = "Line 1\n\n\n\n\nLine 2"
    clamped_ws = clamp_caption(whitespace_text, 100)
    assert clamped_ws == "Line 1\n\nLine 2"


def test_platform_rules_integrity():
    """Verifies that all supported platforms have valid labels, max_chars, and style strings."""
    expected_platforms = ["instagram", "x", "facebook", "linkedin", "pinterest", "youtube", "kick"]
    for platform in expected_platforms:
        assert platform in PLATFORM_RULES
        rule = PLATFORM_RULES[platform]
        assert "label" in rule and len(rule["label"]) > 0
        assert "max_chars" in rule and rule["max_chars"] > 0
        assert "style" in rule and len(rule["style"]) > 0

    # Specific check on X/Twitter strict 280-char limit
    assert PLATFORM_RULES["x"]["max_chars"] == 280


def test_build_fallback_caption_for_all_platforms():
    """Ensures fallback captions are successfully generated within character limits for all platforms."""
    description = "A coffee cup on a wooden table beside an open laptop"
    tone = "Inspiring"
    context = "Morning productivity routine"

    for platform_key, rule in PLATFORM_RULES.items():
        caption = build_fallback_caption(platform_key, description, tone, context)
        assert caption is not None
        assert len(caption) > 0
        assert len(caption) <= rule["max_chars"]
        assert "#" in caption  # Must contain hashtags


def test_generate_gemini_content_with_retry_success():
    """Tests that retry helper returns successful response on the first try."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Generated caption response"
    mock_client.models.generate_content.return_value = mock_response

    result = generate_gemini_content_with_retry(mock_client, "Test prompt", max_retries=2)
    assert result.text == "Generated caption response"
    assert mock_client.models.generate_content.call_count == 1


def test_generate_gemini_content_with_retry_transient_error():
    """Tests that retry helper retries on transient errors (e.g. 503, 429) and succeeds on second try."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Success after retry"

    # First call throws 503 UNAVAILABLE, second succeeds
    mock_client.models.generate_content.side_effect = [
        Exception("503 UNAVAILABLE - Service temporarily overloaded"),
        mock_response
    ]

    result = generate_gemini_content_with_retry(mock_client, "Test prompt", max_retries=3)
    assert result.text == "Success after retry"
    assert mock_client.models.generate_content.call_count == 2


def test_generate_gemini_content_with_retry_model_fallback():
    """Tests that retry helper falls back to subsequent models when one model fails."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Success on fallback model"

    # Fail all attempts for model 1, succeed on model 2
    def side_effect(**kwargs):
        if kwargs.get("model") == GEMINI_MODEL_FALLBACKS[0]:
            raise Exception("404 Model not found or unavailable")
        return mock_response

    mock_client.models.generate_content.side_effect = side_effect

    result = generate_gemini_content_with_retry(mock_client, "Test prompt", max_retries=1)
    assert result.text == "Success on fallback model"


def test_generate_gemini_content_with_retry_persistent_failure():
    """Tests that persistent exceptions are properly raised when all fallback models fail."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("Persistent API Failure")

    with pytest.raises(Exception) as exc_info:
        generate_gemini_content_with_retry(mock_client, "Test prompt", max_retries=1)
    assert "Persistent API Failure" in str(exc_info.value)
