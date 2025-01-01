"""Tests for network functionality"""
import pytest
import requests
import time
import json
from unittest.mock import patch, Mock

from evolia.utils.network_logging import LoggedRequests

def test_logged_requests_whitelist():
    """Test that LoggedRequests correctly enforces domain whitelisting"""
    config = {
        'network': {
            'whitelist_domains': ['example.com'],
            'rate_limit': {'enabled': False}
        }
    }
    requests_client = LoggedRequests(config)
    
    # Test allowed domain
    with patch('requests.Session.request') as mock_request:
        requests_client.get('https://example.com/api')
        mock_request.assert_called_once_with('GET', 'https://example.com/api')
        
    # Test blocked domain
    with pytest.raises(PermissionError) as exc:
        requests_client.get('https://blocked.com/api')
    assert "not allowed" in str(exc.value)
    
def test_logged_requests_rate_limit():
    """Test that LoggedRequests correctly applies rate limiting"""
    config = {
        'network': {
            'rate_limit': {
                'enabled': True,
                'max_calls': 2,
                'period': 1
            }
        }
    }
    requests_client = LoggedRequests(config)
    
    # First two calls should succeed
    with patch('requests.Session.request') as mock_request:
        requests_client.get('https://example.com/api')
        requests_client.get('https://example.com/api')
        assert mock_request.call_count == 2
        
    # Third call should raise an exception
    with pytest.raises(Exception) as exc:
        requests_client.get('https://example.com/api')
    assert "Rate limit exceeded" in str(exc.value)
    
    # Wait for the period to expire
    time.sleep(1.1)
    
    # Should be able to call again
    with patch('requests.Session.request') as mock_request:
        requests_client.get('https://example.com/api')
        mock_request.assert_called_once_with('GET', 'https://example.com/api')
        
def test_logged_requests_logging(caplog):
    """Test that LoggedRequests correctly logs requests"""
    config = {
        'network': {
            'logging': {
                'enabled': True,
                'log_headers': True,
                'log_params': True,
                'log_data': True,
                'log_json': True
            },
            'rate_limit': {'enabled': False}
        }
    }
    requests_client = LoggedRequests(config)
    
    # Make a request and check the logs
    with patch('requests.Session.request'):
        requests_client.get('https://example.com/api', 
                    headers={'Authorization': 'Bearer token'},
                    params={'key': 'value'},
                    json={'data': 'test'})
        
    # Check basic request logging
    assert 'Making GET request to https://example.com/api' in caplog.text
    
    # Check structured logging payload
    found_payload = False
    for record in caplog.records:
        if hasattr(record, 'payload'):
            payload = record.payload
            if isinstance(payload, dict):
                if payload.get('headers', {}).get('Authorization') == 'Bearer token':
                    found_payload = True
                    break
    assert found_payload, "Structured logging payload not found"

def test_logged_requests_methods():
    """Test that all HTTP methods are supported and logged"""
    config = {
        'network': {
            'rate_limit': {'enabled': False}
        }
    }
    requests_client = LoggedRequests(config)
    methods = ['get', 'post', 'put', 'delete', 'patch', 'head', 'options']
    
    for method in methods:
        with patch('requests.Session.request') as mock_request:
            getattr(requests_client, method)('https://example.com/api')
            mock_request.assert_called_once_with(method.upper(), 'https://example.com/api')
            
def test_logged_requests_no_whitelist():
    """Test that LoggedRequests allows all domains when no whitelist is specified"""
    config = {
        'network': {
            'rate_limit': {'enabled': False}
        }
    }
    requests_client = LoggedRequests(config)
    
    # Should be able to call any domain
    with patch('requests.Session.request') as mock_request:
        requests_client.get('https://any-domain.com/api')
        mock_request.assert_called_once_with('GET', 'https://any-domain.com/api')
        
def test_logged_requests_http_disabled():
    """Test that HTTP requests are blocked when configured"""
    config = {
        'network': {
            'access': {
                'allow_http': False
            },
            'rate_limit': {'enabled': False}
        }
    }
    requests_client = LoggedRequests(config)
    
    # HTTPS should work
    with patch('requests.Session.request') as mock_request:
        requests_client.get('https://example.com/api')
        mock_request.assert_called_once_with('GET', 'https://example.com/api')
        
    # HTTP should be blocked
    with pytest.raises(PermissionError) as exc:
        requests_client.get('http://example.com/api')
    assert "HTTP requests are not allowed" in str(exc.value)
    
def test_logged_requests_external_disabled():
    """Test that external requests are blocked when configured"""
    config = {
        'network': {
            'access': {
                'allow_external': False
            },
            'whitelist_domains': ['example.com'],
            'rate_limit': {'enabled': False}
        }
    }
    requests_client = LoggedRequests(config)
    
    # Whitelisted domain should work
    with patch('requests.Session.request') as mock_request:
        requests_client.get('https://example.com/api')
        mock_request.assert_called_once_with('GET', 'https://example.com/api')
        
    # External domain should be blocked
    with pytest.raises(PermissionError) as exc:
        requests_client.get('https://external-domain.com/api')
    assert "Domain external-domain.com not allowed" in str(exc.value)
    
def test_logged_requests_ssl_verification():
    """Test that SSL verification settings are respected"""
    config = {
        'network': {
            'security': {
                'verify_ssl': False
            },
            'rate_limit': {'enabled': False}
        }
    }
    requests_client = LoggedRequests(config)
    
    # SSL verification should be disabled
    assert not requests_client.session.verify
    
def test_logged_requests_redirects():
    """Test that redirect settings are respected"""
    config = {
        'network': {
            'security': {
                'allow_redirects': False
            },
            'rate_limit': {'enabled': False}
        }
    }
    requests_client = LoggedRequests(config)
    
    # Redirects should be disabled
    assert not requests_client.session.allow_redirects 