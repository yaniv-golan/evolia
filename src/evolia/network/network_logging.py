"""Network request logging and control functionality"""
import logging
import requests
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse

from .logger import setup_logger
from .network_rate_limiter import RateLimiter

logger = setup_logger()

# Error messages for network validation
ERROR_MESSAGES = {
    'http_not_allowed': "HTTP requests are not allowed. Use HTTPS instead.",
    'external_not_allowed': "External requests are not allowed",
    'domain_not_whitelisted': "Request to {url} is not allowed"
}

# Warning messages for logging
WARNING_MESSAGES = {
    'http_warning': "HTTP request to {url} is not allowed",
    'external_warning': "External request to {url} is not allowed",
    'whitelist_warning': "Request to {url} is not whitelisted"
}

class LoggedRequests:
    """Wrapper around requests library that logs all network requests"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize LoggedRequests
        
        Args:
            config: Network configuration dictionary
        """
        self.config = config.get('network', {})
        self.session = requests.Session()
        
        # Set up domain whitelisting
        self.whitelist_domains = self.config.get('whitelist_domains', [])
        
        # Set up rate limiting
        rate_limit = self.config.get('rate_limit', {})
        if rate_limit.get('enabled', True):
            self.rate_limiter = RateLimiter(
                max_calls=rate_limit.get('max_calls', 100),
                period=rate_limit.get('period', 60)
            )
        else:
            self.rate_limiter = None
            
        # Set up session defaults
        security = self.config.get('security', {})
        self.session.verify = security.get('verify_ssl', True)
        self.session.allow_redirects = security.get('allow_redirects', True)
        
    def is_whitelisted(self, url: str) -> bool:
        """Check if a URL's domain is in the whitelist"""
        if not self.whitelist_domains:
            return True  # If no whitelist is specified, allow all domains
            
        domain = urlparse(url).netloc
        return domain in self.whitelist_domains
        
    def validate_url(self, url: str):
        """Validate a URL against security settings
        
        Args:
            url: URL to validate
            
        Raises:
            PermissionError: If the URL is not allowed
        """
        parsed = urlparse(url)
        
        # Check HTTP/HTTPS
        if not self.config.get('access', {}).get('allow_http', True) and parsed.scheme == 'http':
            logger.warning(WARNING_MESSAGES['http_warning'].format(url=url), extra={
                'payload': {
                    'url': url,
                    'component': 'network',
                    'operation': 'validate_url'
                }
            })
            raise PermissionError(ERROR_MESSAGES['http_not_allowed'])
            
        # Check external access
        if not self.config.get('access', {}).get('allow_external', True):
            if parsed.netloc not in self.whitelist_domains:
                logger.warning(WARNING_MESSAGES['external_warning'].format(url=url), extra={
                    'payload': {
                        'url': url,
                        'component': 'network',
                        'operation': 'validate_url'
                    }
                })
                raise PermissionError(ERROR_MESSAGES['external_not_allowed'])
                
        # Check domain whitelist
        if not self.is_whitelisted(url):
            logger.warning(WARNING_MESSAGES['whitelist_warning'].format(url=url), extra={
                'payload': {
                    'url': url,
                    'component': 'network',
                    'operation': 'validate_url'
                }
            })
            raise PermissionError(ERROR_MESSAGES['domain_not_whitelisted'].format(url=url))
        
    def _extract_log_data(self, method: str, url: str, **kwargs) -> Dict[str, Any]:
        """Extract log data based on configuration.
        
        Args:
            method: HTTP method
            url: Request URL
            **kwargs: Additional request arguments
            
        Returns:
            Dictionary containing log data based on config
        """
        log_data = {
            'method': method,
            'url': url,
            'component': 'network',
            'operation': 'http_request'
        }
        
        # Add optional logging data based on config
        logging_config = self.config.get('logging', {})
        
        # Request details
        if logging_config.get('log_headers', True):
            log_data['headers'] = kwargs.get('headers', {})
        if logging_config.get('log_params', True):
            log_data['params'] = kwargs.get('params', {})
        if logging_config.get('log_data', False):
            log_data['data'] = kwargs.get('data', None)
        if logging_config.get('log_json', False):
            log_data['json'] = kwargs.get('json', None)
            
        # Security details
        if logging_config.get('log_security', True):
            log_data['security'] = {
                'whitelisted': self.is_whitelisted(url),
                'verify_ssl': self.session.verify,
                'allow_redirects': self.session.allow_redirects
            }
            
        # Rate limiting details
        if logging_config.get('log_rate_limit', True) and self.rate_limiter is not None:
            log_data['rate_limit'] = {
                'max_calls': self.rate_limiter.max_calls,
                'period': self.rate_limiter.period,
                'current_calls': len(self.rate_limiter.call_times)
            }
            
        return log_data
        
    def log_request(self, method: str, url: str, **kwargs):
        """Log details about an HTTP request"""
        if not self.config.get('logging', {}).get('enabled', True):
            return
            
        log_data = self._extract_log_data(method, url, **kwargs)
        logger.info(f"Making {method} request to {url}", extra={
            'payload': log_data
        })
        
    def _make_request(self, method: str, url: str, **kwargs):
        """Make an HTTP request with validation and logging
        
        Args:
            method: HTTP method to use
            url: URL to request
            **kwargs: Additional arguments to pass to requests
            
        Returns:
            requests.Response: Response from the request
            
        Raises:
            PermissionError: If the request is not allowed
            Exception: If rate limit is exceeded
        """
        # Validate URL
        self.validate_url(url)
        
        # Check rate limit
        if self.rate_limiter is not None:
            if not self.rate_limiter.check_rate_limit():
                logger.warning("Rate limit exceeded", extra={
                    'payload': self._extract_log_data(method, url, **kwargs)
                })
                raise Exception(f"Rate limit exceeded: {self.rate_limiter.max_calls} calls per {self.rate_limiter.period} seconds")
            self.rate_limiter.record_call()
            
        # Log request
        self.log_request(method, url, **kwargs)
        
        # Make request
        return getattr(self.session, method.lower())(url, **kwargs)
        
    def get(self, url: str, **kwargs):
        """Make a GET request with logging"""
        return self._make_request('GET', url, **kwargs)
        
    def post(self, url: str, **kwargs):
        """Make a POST request with logging"""
        return self._make_request('POST', url, **kwargs)
        
    def put(self, url: str, **kwargs):
        """Make a PUT request with logging"""
        return self._make_request('PUT', url, **kwargs)
        
    def delete(self, url: str, **kwargs):
        """Make a DELETE request with logging"""
        return self._make_request('DELETE', url, **kwargs)
        
    def patch(self, url: str, **kwargs):
        """Make a PATCH request with logging"""
        return self._make_request('PATCH', url, **kwargs)
        
    def head(self, url: str, **kwargs):
        """Make a HEAD request with logging"""
        return self._make_request('HEAD', url, **kwargs)
        
    def options(self, url: str, **kwargs):
        """Make an OPTIONS request with logging"""
        return self._make_request('OPTIONS', url, **kwargs) 