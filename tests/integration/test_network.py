"""Tests for network functionality"""
import pytest
import requests
import time
from unittest.mock import patch

from evolia.utils.network_logging import LoggedRequests

@pytest.fixture
def requests_client():
    """Create a LoggedRequests client with basic config."""
    config = {
        'network': {
            'whitelist_domains': ['example.com'],
            'rate_limit': {
                'enabled': True,
                'max_calls': 2,
                'period': 1
            },
            'access': {
                'allow_http': False,
                'allow_external': False
            }
        }
    }
    return LoggedRequests(config)

def test_request_validation(requests_client):
    """Test request validation (whitelist, protocol, external access)."""
    # Test allowed domain
    with patch('requests.Session.request') as mock_request:
        requests_client.get('https://example.com/api')
        mock_request.assert_called_once_with('GET', 'https://example.com/api')
    
    # Test blocked domain
    with pytest.raises(PermissionError, match="not allowed"):
        requests_client.get('https://blocked.com/api')
    
    # Test HTTP blocked
    with pytest.raises(PermissionError, match="HTTP requests are not allowed"):
        requests_client.get('http://example.com/api')

def test_rate_limiting(requests_client):
    """Test rate limiting functionality."""
    with patch('requests.Session.request'):
        # First two calls should succeed
        requests_client.get('https://example.com/api')
        requests_client.get('https://example.com/api')
        
        # Third call should fail
        with pytest.raises(Exception, match="Rate limit exceeded"):
            requests_client.get('https://example.com/api')
        
        # Wait for rate limit to reset
        time.sleep(1.1)
        
        # Should work again
        requests_client.get('https://example.com/api')

def test_request_logging(caplog):
    """Test request logging functionality."""
    config = {
        'network': {
            'logging': {'enabled': True},
            'rate_limit': {'enabled': False}
        }
    }
    client = LoggedRequests(config)
    
    with patch('requests.Session.request'):
        client.post(
            'https://example.com/api',
            headers={'Authorization': 'Bearer token'},
            json={'data': 'test'}
        )
    
    # Verify basic logging
    assert 'Making POST request to https://example.com/api' in caplog.text
    
    # Verify payload logging
    for record in caplog.records:
        if hasattr(record, 'payload'):
            payload = record.payload
            if isinstance(payload, dict):
                assert payload.get('headers', {}).get('Authorization') == 'Bearer token'
                break 