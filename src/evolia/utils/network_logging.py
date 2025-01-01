"""Network logging functionality"""
import requests
import time
from typing import Optional, Dict, Any
from urllib.parse import urlparse
from contextlib import contextmanager

from .logger import setup_logger

logger = setup_logger()

@contextmanager
def network_request_context(method: str, url: str):
    """Context manager for network request logging"""
    logger.info(f"Making {method} request to {url}")
    try:
        yield
        logger.info(f"Request complete: {method} {url}")
    except Exception as e:
        logger.error(f"Request failed: {method} {url}", exc_info=True)
        raise

class LoggedRequests:
    """Wrapper around requests library that adds logging and security controls"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('network', {})
        self.session = requests.Session()
        
        # Configure SSL verification
        if not self.config.get('security', {}).get('verify_ssl', True):
            self.session.verify = False
            logger.warning("SSL verification disabled")
            
        # Configure redirects
        if not self.config.get('security', {}).get('allow_redirects', True):
            self.session.allow_redirects = False
            logger.warning("HTTP redirects disabled")
            
        # Initialize rate limiter if enabled
        self.rate_limit = self.config.get('rate_limit', {})
        if self.rate_limit.get('enabled', False):
            self.calls = []
            self.max_calls = self.rate_limit.get('max_calls', 10)
            self.period = self.rate_limit.get('period', 60)
            logger.debug(f"Rate limiting enabled: {self.max_calls} calls per {self.period}s")
            
        # Get domain whitelist
        self.whitelist_domains = self.config.get('whitelist_domains', [])
        if self.whitelist_domains:
            logger.debug(f"Domain whitelist: {self.whitelist_domains}")
            
    def _check_rate_limit(self):
        """Check if rate limit has been exceeded"""
        if not self.rate_limit.get('enabled', False):
            return
            
        # Remove old calls
        now = time.time()
        self.calls = [t for t in self.calls if now - t < self.period]
        
        # Check limit
        if len(self.calls) >= self.max_calls:
            logger.error(f"Rate limit exceeded: {self.max_calls} calls per {self.period}s")
            raise Exception("Rate limit exceeded")
            
        # Add new call
        self.calls.append(now)
        
    def _check_domain(self, url: str):
        """Check if domain is allowed"""
        if not self.whitelist_domains:
            return
            
        domain = urlparse(url).netloc
        if domain not in self.whitelist_domains:
            logger.error(f"Domain not allowed: {domain}")
            raise PermissionError(f"Domain {domain} not allowed")
            
    def _check_http(self, url: str):
        """Check if HTTP is allowed"""
        if not self.config.get('access', {}).get('allow_http', True):
            if url.startswith('http://'):
                logger.error("HTTP request blocked: HTTPS required")
                raise PermissionError("HTTP requests are not allowed")
                
    def _check_external(self, url: str):
        """Check if external requests are allowed"""
        if not self.config.get('access', {}).get('allow_external', True):
            domain = urlparse(url).netloc
            if domain not in self.whitelist_domains:
                logger.error(f"External request blocked: {domain}")
                raise PermissionError("External requests are not allowed")
                
    def _log_request_details(self, **kwargs):
        """Log request details using structured logging"""
        if not self.config.get('logging', {}).get('enabled', True):
            return
            
        log_data = {}
        
        if self.config.get('logging', {}).get('log_headers', False):
            log_data['headers'] = kwargs.get('headers', {})
            
        if self.config.get('logging', {}).get('log_params', False):
            log_data['params'] = kwargs.get('params', {})
            
        if self.config.get('logging', {}).get('log_data', False):
            log_data['data'] = kwargs.get('data', {})
            
        if self.config.get('logging', {}).get('log_json', False):
            log_data['json'] = kwargs.get('json', {})
            
        if log_data:
            logger.debug("Request details", extra={'payload': log_data})
            
    def request(self, method: str, url: str, **kwargs):
        """Make an HTTP request"""
        with network_request_context(method, url):
            self._check_rate_limit()
            self._check_domain(url)
            self._check_http(url)
            self._check_external(url)
            self._log_request_details(**kwargs)
            return self.session.request(method, url, **kwargs)
        
    def get(self, url: str, **kwargs):
        """Make a GET request"""
        return self.request('GET', url, **kwargs)
        
    def post(self, url: str, **kwargs):
        """Make a POST request"""
        return self.request('POST', url, **kwargs)
        
    def put(self, url: str, **kwargs):
        """Make a PUT request"""
        return self.request('PUT', url, **kwargs)
        
    def delete(self, url: str, **kwargs):
        """Make a DELETE request"""
        return self.request('DELETE', url, **kwargs)
        
    def patch(self, url: str, **kwargs):
        """Make a PATCH request"""
        return self.request('PATCH', url, **kwargs)
        
    def head(self, url: str, **kwargs):
        """Make a HEAD request"""
        return self.request('HEAD', url, **kwargs)
        
    def options(self, url: str, **kwargs):
        """Make an OPTIONS request"""
        return self.request('OPTIONS', url, **kwargs) 