"""
Shared logging and error handling utilities for Lambda functions.

Provides structured logging with consistent JSON format for CloudWatch,
performance tracking, and standardized error handling.
"""

import json
import logging
import time
import functools
from typing import Any, Dict, Optional, Callable
from datetime import datetime


class StructuredLogger:
    """
    Structured JSON logger for CloudWatch integration.

    Produces log entries with consistent schema including:
    - timestamp, level, message
    - session_id, operation_type (when set)
    - duration_ms (for performance tracking)
    - error details (for error entries)
    """

    def __init__(self, service_name: str):
        self.service_name = service_name
        self.logger = logging.getLogger(service_name)
        self.logger.setLevel(logging.INFO)
        self._session_id: Optional[str] = None
        self._operation_type: Optional[str] = None

    def set_context(self, session_id: str = None, operation_type: str = None):
        """Set context fields that will be included in all subsequent log entries."""
        if session_id is not None:
            self._session_id = session_id
        if operation_type is not None:
            self._operation_type = operation_type

    def clear_context(self):
        """Clear context fields."""
        self._session_id = None
        self._operation_type = None

    def _build_entry(self, level: str, message: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Build a structured log entry."""
        entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': level,
            'service': self.service_name,
            'message': message,
        }
        if self._session_id:
            entry['session_id'] = self._session_id
        if self._operation_type:
            entry['operation_type'] = self._operation_type
        if extra:
            entry.update(extra)
        return entry

    def info(self, message: str, **kwargs):
        entry = self._build_entry('INFO', message, kwargs if kwargs else None)
        self.logger.info(json.dumps(entry, default=str))

    def warning(self, message: str, **kwargs):
        entry = self._build_entry('WARNING', message, kwargs if kwargs else None)
        self.logger.warning(json.dumps(entry, default=str))

    def error(self, message: str, error: Optional[Exception] = None, **kwargs):
        extra = kwargs if kwargs else {}
        if error:
            extra['error_type'] = type(error).__name__
            extra['error_message'] = str(error)
        entry = self._build_entry('ERROR', message, extra)
        self.logger.error(json.dumps(entry, default=str))

    def metric(self, metric_name: str, value: float, unit: str = 'Count', **kwargs):
        """Log a metric entry for CloudWatch Metrics extraction."""
        extra = {
            'metric_name': metric_name,
            'metric_value': value,
            'metric_unit': unit,
        }
        extra.update(kwargs)
        entry = self._build_entry('METRIC', f"{metric_name}: {value} {unit}", extra)
        self.logger.info(json.dumps(entry, default=str))


def track_performance(logger: StructuredLogger):
    """Decorator to track function execution time."""
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                duration_ms = round((time.time() - start) * 1000, 2)
                logger.metric(
                    f"{func.__name__}_duration",
                    duration_ms,
                    unit='Milliseconds'
                )
                return result
            except Exception as e:
                duration_ms = round((time.time() - start) * 1000, 2)
                logger.error(
                    f"{func.__name__} failed after {duration_ms}ms",
                    error=e,
                    duration_ms=duration_ms
                )
                raise
        return wrapper
    return decorator


def create_error_response(status_code: int, message: str, error_code: str = None) -> Dict[str, Any]:
    """Create a standardized error response for API Gateway."""
    body = {'error': message}
    if error_code:
        body['error_code'] = error_code
    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
        },
        'body': json.dumps(body)
    }


SIZE_LIMITS = {
    'max_file_size_mb': 50,
    'max_rows': 100000,
    'max_columns': 500,
}


def validate_size_limits(file_size_bytes: int = None, row_count: int = None,
                         column_count: int = None) -> Optional[str]:
    """
    Validate dataset size limits. Returns error message if limits exceeded, None if OK.
    Implements Requirement 17.4.
    """
    if file_size_bytes is not None:
        max_bytes = SIZE_LIMITS['max_file_size_mb'] * 1024 * 1024
        if file_size_bytes > max_bytes:
            return f"File size ({file_size_bytes / 1024 / 1024:.1f} MB) exceeds maximum allowed size ({SIZE_LIMITS['max_file_size_mb']} MB)"

    if row_count is not None:
        if row_count > SIZE_LIMITS['max_rows']:
            return f"Row count ({row_count:,}) exceeds maximum allowed ({SIZE_LIMITS['max_rows']:,} rows)"

    if column_count is not None:
        if column_count > SIZE_LIMITS['max_columns']:
            return f"Column count ({column_count}) exceeds maximum allowed ({SIZE_LIMITS['max_columns']} columns)"

    return None
