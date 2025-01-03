"""Unit tests for network functionality and rate limiting."""
import time
from unittest.mock import Mock, patch

import pytest

from evolia.utils.network_logging import RequestsClient


@pytest.fixture
def requests_client():
    """Create a requests client for testing."""
    return RequestsClient(rate_limit=2, rate_window=1, domain_whitelist=["example.com"])


def test_request_validation(requests_client):
    """Test request validation and rate limiting."""
    # Mock the session's request method
    mock_response = Mock()
    mock_response.status_code = 200

    with patch.object(requests_client.session, "request", return_value=mock_response):
        # Test successful request
        response = requests_client.get("https://example.com/api")
        assert response.status_code == 200

        # Test domain validation
        with pytest.raises(PermissionError):
            requests_client.get("https://blocked.com/api")

        # Test rate limiting
        response = requests_client.get("http://example.com/api")  # Second request
        assert response.status_code == 200

        # Third request should fail due to rate limit
        with pytest.raises(Exception, match="Rate limit exceeded"):
            requests_client.get("http://example.com/api")

        # Wait for rate limit window to reset
        time.sleep(1.1)

        # Fourth request should succeed after waiting
        response = requests_client.get("http://example.com/api")
        assert response.status_code == 200
