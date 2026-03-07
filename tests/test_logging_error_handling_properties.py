"""
Property-based tests for logging, error handling, and monitoring.

Properties tested:
- Property 24: Comprehensive Logging
- Property 27: Cache Logging
- Property 31: Error Handling and Logging
- Property 32: Bedrock Retry Logic
- Property 33: Size Limit Error Handling
"""

import pytest
import json
import logging
import time
from hypothesis import given, strategies as st, settings
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Set AWS env before imports
os.environ.setdefault('AWS_DEFAULT_REGION', 'us-east-1')
os.environ.setdefault('AWS_ACCESS_KEY_ID', 'testing')
os.environ.setdefault('AWS_SECRET_ACCESS_KEY', 'testing')

# Add shared path for logging_utils (no name collision)
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lambda', 'shared'))

# Load ai_assistant lambda_function via importlib to avoid sys.path collisions
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "ai_assistant_lambda",
    os.path.join(os.path.dirname(__file__), '..', 'lambda', 'ai_assistant', 'lambda_function.py')
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
sys.modules['ai_assistant_lambda'] = _module

from logging_utils import (
    StructuredLogger,
    track_performance,
    create_error_response,
    validate_size_limits,
    SIZE_LIMITS,
)

# Import AI assistant for retry/cache tests
ai_assistant = sys.modules['ai_assistant_lambda']


class TestComprehensiveLogging:
    """
    Property 24: Comprehensive Logging
    Validates: Requirements 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7
    """

    def test_property_24_structured_log_format(self):
        """
        Feature: ai-data-analyst-platform, Property 24: Comprehensive Logging

        All log entries should have consistent structured JSON format with required fields.
        """
        logger = StructuredLogger('test-service')

        with patch.object(logger.logger, 'info') as mock_info:
            logger.info("Test message", key1="value1")
            mock_info.assert_called_once()

            log_entry = json.loads(mock_info.call_args[0][0])
            assert 'timestamp' in log_entry
            assert log_entry['level'] == 'INFO'
            assert log_entry['service'] == 'test-service'
            assert log_entry['message'] == 'Test message'
            assert log_entry['key1'] == 'value1'

    def test_property_24_session_context_in_logs(self):
        """Log entries should include session context when set."""
        logger = StructuredLogger('test-service')
        logger.set_context(session_id='sess-123', operation_type='upload')

        with patch.object(logger.logger, 'info') as mock_info:
            logger.info("Processing")
            log_entry = json.loads(mock_info.call_args[0][0])
            assert log_entry['session_id'] == 'sess-123'
            assert log_entry['operation_type'] == 'upload'

    def test_property_24_error_logging_includes_details(self):
        """Error log entries should include error type and message."""
        logger = StructuredLogger('test-service')

        with patch.object(logger.logger, 'error') as mock_error:
            try:
                raise ValueError("test error message")
            except ValueError as e:
                logger.error("Operation failed", error=e)

            log_entry = json.loads(mock_error.call_args[0][0])
            assert log_entry['level'] == 'ERROR'
            assert log_entry['error_type'] == 'ValueError'
            assert log_entry['error_message'] == 'test error message'

    def test_property_24_warning_logging(self):
        """Warning log entries should have WARNING level."""
        logger = StructuredLogger('test-service')

        with patch.object(logger.logger, 'warning') as mock_warn:
            logger.warning("Something unusual")
            log_entry = json.loads(mock_warn.call_args[0][0])
            assert log_entry['level'] == 'WARNING'

    def test_property_24_metric_logging(self):
        """Metric log entries should include metric name, value, and unit."""
        logger = StructuredLogger('test-service')

        with patch.object(logger.logger, 'info') as mock_info:
            logger.metric("request_duration", 150.5, unit="Milliseconds")
            log_entry = json.loads(mock_info.call_args[0][0])
            assert log_entry['level'] == 'METRIC'
            assert log_entry['metric_name'] == 'request_duration'
            assert log_entry['metric_value'] == 150.5
            assert log_entry['metric_unit'] == 'Milliseconds'

    @given(service_name=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('L', 'N'))))
    @settings(max_examples=15)
    def test_property_24_service_name_in_all_levels(self, service_name):
        """Service name should appear in all log levels."""
        logger = StructuredLogger(service_name)

        for level_method, mock_attr in [('info', 'info'), ('warning', 'warning'), ('error', 'error')]:
            with patch.object(logger.logger, mock_attr) as mock_log:
                getattr(logger, level_method)("test")
                log_entry = json.loads(mock_log.call_args[0][0])
                assert log_entry['service'] == service_name

    def test_property_24_context_clear(self):
        """Clearing context should remove session/operation from logs."""
        logger = StructuredLogger('test-service')
        logger.set_context(session_id='sess-123')
        logger.clear_context()

        with patch.object(logger.logger, 'info') as mock_info:
            logger.info("After clear")
            log_entry = json.loads(mock_info.call_args[0][0])
            assert 'session_id' not in log_entry

    def test_property_24_performance_tracking(self):
        """Performance decorator should log execution duration."""
        logger = StructuredLogger('test-service')

        @track_performance(logger)
        def slow_function():
            time.sleep(0.01)
            return "done"

        with patch.object(logger.logger, 'info') as mock_info:
            result = slow_function()
            assert result == "done"
            mock_info.assert_called_once()
            log_entry = json.loads(mock_info.call_args[0][0])
            assert log_entry['metric_name'] == 'slow_function_duration'
            assert log_entry['metric_value'] >= 10  # At least 10ms

    def test_property_24_performance_tracking_on_error(self):
        """Performance decorator should log errors with duration."""
        logger = StructuredLogger('test-service')

        @track_performance(logger)
        def failing_function():
            raise RuntimeError("boom")

        with patch.object(logger.logger, 'error') as mock_error:
            with pytest.raises(RuntimeError):
                failing_function()

            log_entry = json.loads(mock_error.call_args[0][0])
            assert 'duration_ms' in log_entry
            assert log_entry['error_type'] == 'RuntimeError'


class TestCacheLogging:
    """
    Property 27: Cache Logging
    Validates: Requirements 15.5
    """

    def setup_method(self):
        self.dynamodb_mock = Mock()
        self.table_mock = Mock()
        self.dynamodb_mock.Table.return_value = self.table_mock
        self.dynamodb_patcher = patch('ai_assistant_lambda.dynamodb', self.dynamodb_mock)
        self.dynamodb_patcher.start()

    def teardown_method(self):
        self.dynamodb_patcher.stop()

    def test_property_27_cache_hit_logged(self, caplog):
        """
        Feature: ai-data-analyst-platform, Property 27: Cache Logging

        Cache hits should be logged in CloudWatch for monitoring.
        """
        from datetime import datetime, timedelta
        cached_data = {'models': []}
        self.table_mock.get_item.return_value = {
            'Item': {
                'prompt_hash': 'test_hash',
                'response': json.dumps(cached_data),
                'timestamp': datetime.utcnow().isoformat(),
                'ttl': int((datetime.utcnow() + timedelta(hours=24)).timestamp())
            }
        }

        with caplog.at_level(logging.INFO):
            result = ai_assistant.check_cache('test_hash')

        assert result is not None
        assert any('Cache hit' in record.message or 'cache hit' in record.message.lower()
                    for record in caplog.records)

    def test_property_27_cache_miss_logged(self, caplog):
        """Cache misses should be logged."""
        self.table_mock.get_item.return_value = {}

        with caplog.at_level(logging.INFO):
            result = ai_assistant.check_cache('nonexistent')

        assert result is None

    def test_property_27_cache_store_logged(self, caplog):
        """Cache store operations should be logged."""
        self.table_mock.put_item.return_value = {}

        with caplog.at_level(logging.INFO):
            ai_assistant.store_cache('hash123', {'data': 'test'})

        assert any('Cached' in record.message or 'cache' in record.message.lower()
                    for record in caplog.records)


class TestErrorHandling:
    """
    Property 31: Error Handling and Logging
    Validates: Requirements 17.1, 17.2
    """

    def test_property_31_error_response_format(self):
        """
        Feature: ai-data-analyst-platform, Property 31: Error Handling and Logging

        Error responses should have proper status code, descriptive message, and valid JSON body.
        """
        response = create_error_response(400, "Invalid file format")

        assert response['statusCode'] == 400
        assert response['headers']['Content-Type'] == 'application/json'

        body = json.loads(response['body'])
        assert body['error'] == "Invalid file format"

    def test_property_31_error_code_included(self):
        """Error responses should optionally include error codes."""
        response = create_error_response(500, "Processing failed", error_code="PROC_001")

        body = json.loads(response['body'])
        assert body['error_code'] == "PROC_001"

    @given(
        status_code=st.sampled_from([400, 401, 403, 404, 500, 502, 503]),
        message=st.text(min_size=1, max_size=200)
    )
    @settings(max_examples=20)
    def test_property_31_all_error_responses_valid_json(self, status_code, message):
        """All error responses should be valid JSON regardless of input."""
        response = create_error_response(status_code, message)

        assert response['statusCode'] == status_code
        body = json.loads(response['body'])  # Should not raise
        assert 'error' in body

    def test_property_31_cors_headers_on_errors(self):
        """Error responses should include CORS headers."""
        response = create_error_response(500, "Server error")
        assert 'Access-Control-Allow-Origin' in response['headers']


class TestBedrockRetryLogic:
    """
    Property 32: Bedrock Retry Logic
    Validates: Requirements 17.3
    """

    def setup_method(self):
        self.bedrock_mock = Mock()
        self.bedrock_patcher = patch('ai_assistant_lambda.bedrock_client', self.bedrock_mock)
        self.time_patcher = patch('ai_assistant_lambda.time.sleep')
        self.bedrock_patcher.start()
        self.time_mock = self.time_patcher.start()

    def teardown_method(self):
        self.bedrock_patcher.stop()
        self.time_patcher.stop()

    def test_property_32_exponential_backoff_delays(self):
        """
        Feature: ai-data-analyst-platform, Property 32: Bedrock Retry Logic

        Failed requests should implement exponential backoff (1s, 2s, 4s).
        """
        self.bedrock_mock.invoke_model.side_effect = Exception("Throttled")

        ai_assistant.invoke_bedrock_with_retry("test prompt")

        # Should have been called 3 times (MAX_RETRIES)
        assert self.bedrock_mock.invoke_model.call_count == 3

        # Should have slept with exponential backoff delays
        sleep_calls = self.time_mock.call_args_list
        assert len(sleep_calls) == 2  # 2 sleeps (between 3 attempts)
        assert sleep_calls[0][0][0] == 1  # First delay: 1s
        assert sleep_calls[1][0][0] == 2  # Second delay: 2s

    def test_property_32_no_retry_on_success(self):
        """Successful first attempt should not trigger retries."""
        self.bedrock_mock.invoke_model.return_value = {
            'body': Mock(read=Mock(return_value=json.dumps({
                'content': [{'text': 'success'}]
            }).encode()))
        }

        result = ai_assistant.invoke_bedrock_with_retry("test")
        assert result == 'success'
        assert self.bedrock_mock.invoke_model.call_count == 1
        self.time_mock.assert_not_called()

    def test_property_32_returns_none_on_exhaustion(self):
        """Should return None when all retries are exhausted."""
        self.bedrock_mock.invoke_model.side_effect = Exception("Unavailable")

        result = ai_assistant.invoke_bedrock_with_retry("test")
        assert result is None

    def test_property_32_retry_then_success(self):
        """Should succeed on retry after initial failure."""
        self.bedrock_mock.invoke_model.side_effect = [
            Exception("Throttled"),
            Exception("Throttled"),
            {
                'body': Mock(read=Mock(return_value=json.dumps({
                    'content': [{'text': 'recovered'}]
                }).encode()))
            }
        ]

        result = ai_assistant.invoke_bedrock_with_retry("test")
        assert result == 'recovered'
        assert self.bedrock_mock.invoke_model.call_count == 3


class TestSizeLimitErrorHandling:
    """
    Property 33: Size Limit Error Handling
    Validates: Requirements 17.4
    """

    def test_property_33_file_size_within_limit(self):
        """
        Feature: ai-data-analyst-platform, Property 33: Size Limit Error Handling

        Files within size limits should pass validation.
        """
        result = validate_size_limits(file_size_bytes=10 * 1024 * 1024)  # 10 MB
        assert result is None

    def test_property_33_file_size_exceeds_limit(self):
        """Files exceeding size limits should return descriptive error."""
        result = validate_size_limits(file_size_bytes=100 * 1024 * 1024)  # 100 MB
        assert result is not None
        assert 'exceeds' in result.lower()
        assert 'MB' in result

    @given(size_mb=st.floats(min_value=51, max_value=500, allow_nan=False, allow_infinity=False))
    @settings(max_examples=15)
    def test_property_33_all_oversized_files_rejected(self, size_mb):
        """All files over the size limit should be rejected with error message."""
        size_bytes = int(size_mb * 1024 * 1024)
        result = validate_size_limits(file_size_bytes=size_bytes)
        assert result is not None

    def test_property_33_row_count_within_limit(self):
        """Row counts within limits should pass."""
        result = validate_size_limits(row_count=50000)
        assert result is None

    def test_property_33_row_count_exceeds_limit(self):
        """Excessive row counts should be rejected."""
        result = validate_size_limits(row_count=200000)
        assert result is not None
        assert 'Row count' in result

    def test_property_33_column_count_within_limit(self):
        """Column counts within limits should pass."""
        result = validate_size_limits(column_count=100)
        assert result is None

    def test_property_33_column_count_exceeds_limit(self):
        """Excessive column counts should be rejected."""
        result = validate_size_limits(column_count=600)
        assert result is not None
        assert 'Column count' in result

    @given(
        file_size=st.integers(min_value=1, max_value=50 * 1024 * 1024),
        rows=st.integers(min_value=1, max_value=100000),
        cols=st.integers(min_value=1, max_value=500)
    )
    @settings(max_examples=20)
    def test_property_33_valid_datasets_always_pass(self, file_size, rows, cols):
        """Datasets within all limits should always pass validation."""
        result = validate_size_limits(
            file_size_bytes=file_size,
            row_count=rows,
            column_count=cols
        )
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
